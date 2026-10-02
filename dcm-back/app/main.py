"""
FastAPI 진입점.

실행:
    cd backend
    uvicorn app.main:app --reload

Swagger UI : http://127.0.0.1:8000/docs
ReDoc      : http://127.0.0.1:8000/redoc
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse

from app import __version__
from app.config import BASE_DIR, settings
from app.exceptions import AppError
from app.logging_config import get_logger, setup_logging
from app.routers import analyze, chat, extract
from app.schemas import ErrorResponse, HealthResponse
from app.services import case_store, embedder, faq_store, llm_engine, ocr_engine, runtime

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 기동 ---
    logger.info("=" * 70)
    logger.info("%s v%s 기동 | env=%s", settings.APP_NAME, __version__, settings.APP_ENV)
    logger.info("문서 추출   : MS Office(docx/xlsx/xlsm/pptx), 한글(hwp/hwpx), 텍스트")
    logger.info("PDF        : PyMuPDF (+ 스캔 페이지·삽입 이미지는 PP-OCRv5)")

    if ocr_engine.is_available():
        logger.info(
            "PP-OCRv5   : 사용 가능 | det=%s | rec=%s (모델은 첫 요청 때 로드됩니다)",
            settings.OCR_DET_MODEL, settings.OCR_REC_MODEL,
        )
    else:
        reason = ocr_engine.import_error() or "미설치"
        logger.warning(
            "PP-OCRv5   : 사용 불가 - 이미지 파일은 503, 스캔 PDF 는 건너뜁니다. | %s", reason
        )
        if "WinError" in reason or "DLL" in reason:
            # torch 와 paddle 이 같은 가상환경에서 MKL/OpenMP DLL 을 두고 충돌하는 경우입니다.
            logger.warning(
                "PP-OCRv5   : torch 와의 DLL 충돌로 보입니다. 윈도우에서는 모델 스택을 지우세요. "
                "-> pip uninstall -y torch transformers sentence-transformers accelerate peft keybert "
                "(자세한 내용은 README.md 의 1. 설치 참고)"
            )

    logger.info(
        "분류 파이프라인: ③ %s | ⑤ %s%s | device=%s | 4bit=%s",
        settings.EMBED_MODEL, settings.LLM_MODEL,
        f" + LoRA({settings.LLM_ADAPTER_PATH})" if settings.LLM_ADAPTER_PATH else " (베이스)",
        runtime.resolve_device(), runtime.use_4bit(),
    )
    if settings.DEBUG:
        logger.info("DEBUG=true  : 분류 응답에 단계별 중간 결과(debug)가 포함됩니다.")

    # ④ 보조 벡터DB : CSV 를 읽어 인덱스를 준비합니다. (CSV 가 그대로면 .npy 만 읽음)
    if settings.CASE_STORE_ENABLED:
        case_store.load_or_build()
    else:
        logger.info("벡터DB     : 꺼짐 (CASE_STORE_ENABLED=false)")

    # '문의' 답변용 FAQ : CSV 를 읽어 검색 인덱스를 준비합니다. (CSV 가 그대로면 .npy 만 읽음)
    faq_store.load_or_build()

    if settings.PRELOAD_MODELS:
        # 기동은 느려지지만 첫 요청이 빨라집니다. (.env 의 PRELOAD_MODELS)
        logger.info("모델 프리로드 시작 - 가중치 다운로드에 수 분 걸릴 수 있습니다.")
        try:
            embedder.get_model()
            llm_engine.warmup()          # Qwen 로드 + KV 캐시 prefill·검증
            logger.info("모델 프리로드 완료")
        except Exception as exc:
            logger.warning("모델 프리로드 실패 - 첫 요청 때 다시 시도합니다 | %s", exc)
    else:
        logger.info("모델 로드   : 지연 로드 (첫 분류 요청 때 올립니다)")

    logger.info("Swagger UI  : http://127.0.0.1:8000/docs")
    logger.info("=" * 70)
    yield
    # --- 종료 ---
    logger.info("%s 종료", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=__version__,
    description=(
        "문서 인식 및 자동분류 통합시스템의 백엔드입니다.\n\n"
        "**현재 구현 범위** — ① 입력 → ② 텍스트 추출 → ③ 임베딩 → ④ 후보 추림 → ⑤ Qwen 판정\n\n"
        "| 단계 | 내용 | 엔드포인트 |\n"
        "|---|---|---|\n"
        "| ①② | 파일에서 텍스트 추출 | `POST /api/v1/extract/text` |\n"
        "| ②→⑤ | 키워드3개+요약 → bge-m3 → 후보 top-4 → Qwen 판정 | "
        "`POST /api/v1/analyze/text`, `POST /api/v1/analyze/file` |\n\n"
        "⑥ 민원 DB 와 ⑦ 사용자 응답은 아직 구현 범위가 아닙니다. "
        "⑤ 의 판정 결과는 `result_text` 에 텍스트로 정리되어 나옵니다.\n\n"
        "`.env` 의 `DEBUG=true` 이면 분류 응답의 `debug` 에 추출 원문 · 키워드3개+요약 · "
        "④ 후보 점수 · ⑤ 원시 출력이 모두 실립니다.\n\n"
        "PDF·이미지는 PP-OCRv5, 그 외는 전용 파이썬 라이브러리로 처리합니다. "
        "설치 여부는 `GET /health` 의 `ocr_available` 로 확인하세요.\n\n"
        "1. 아래 `POST /api/v1/extract/text` 를 펼칩니다.\n"
        "2. **Try it out** → **파일 선택** → **Execute**\n"
        "3. Response body 에서 추출된 텍스트를 확인합니다."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =============================================================================
# 예외 처리 - 모든 실패 응답을 ErrorResponse 포맷으로 통일
# =============================================================================
def _error_body(error_code: str, message: str, detail: str | None = None) -> dict:
    return ErrorResponse(
        error_code=error_code,
        message=message,
        detail=detail,
        created_at=datetime.now(timezone.utc).isoformat(),
    ).model_dump()


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("%s | %s | %s", exc.error_code, exc.message, exc.detail or "")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.error_code, exc.message, exc.detail),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content=_error_body(
            "VALIDATION_ERROR",
            "요청 형식이 올바르지 않습니다.",
            str(exc.errors()),
        ),
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("처리되지 않은 예외 | path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content=_error_body(
            "INTERNAL_ERROR",
            "서버 내부 오류가 발생했습니다.",
            f"{type(exc).__name__}: {exc}" if settings.DEBUG else None,
        ),
    )


# =============================================================================
# 기본 엔드포인트
# =============================================================================
@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """루트 접속 시 Swagger 로 이동."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["system"], summary="헬스 체크")
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.APP_NAME,
        version=__version__,
        env=settings.APP_ENV,
        ocr_available=ocr_engine.is_available(),
        ocr_det_model=settings.OCR_DET_MODEL,
        ocr_rec_model=settings.OCR_REC_MODEL,
        ocr_error=ocr_engine.import_error(),
        debug=settings.DEBUG,
        device=runtime.resolve_device(),
        embed_model=settings.EMBED_MODEL,
        embed_loaded=embedder.is_loaded(),
        llm_model=settings.LLM_MODEL,
        llm_loaded=llm_engine.is_loaded(),
        llm_adapter=settings.LLM_ADAPTER_PATH or None,
    )


# 프론트를 다른 주소(예: http://localhost:3000)에서 띄울 때만 필요합니다. (.env 의 CORS_ALLOW_ORIGINS)
if settings.cors_origins:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

app.include_router(extract.router, prefix=settings.API_PREFIX, tags=["extract"])
app.include_router(analyze.router, prefix=settings.API_PREFIX, tags=["analyze"])
app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["chat"])

# 대화 이어가기 동작 확인용 최소 화면 (front_stub/demo.html). 실제 프론트를 붙이면 끄세요.
_demo_dir = BASE_DIR / "front_stub"
if settings.CHAT_DEMO_ENABLED and _demo_dir.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/chat-demo", StaticFiles(directory=str(_demo_dir), html=True), name="chat-demo")
