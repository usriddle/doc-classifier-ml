"""
분류 파이프라인 API (③ 임베딩 -> ④ 후보 추림 -> ⑤ Gemma 판정).

POST /api/v1/analyze/text        : 문장만 넣어 ②~⑤ 테스트 (파일 없음)
POST /api/v1/analyze/file        : 파일 업로드 -> ② 추출 -> ③④⑤
GET  /api/v1/analyze/categories  : 카테고리 7종 정의
GET  /api/v1/analyze/status      : 모델 설치·로드 상태

⑤ 가 만든 도구 호출은 ⑥(tool_executor)에서 검증(소유권·상태 등)을 거쳐 바로 실행되고,
그 결과가 `tool_result` 와 `result_text` 에 담겨 나갑니다.
DEBUG=true 이면 ② 원문 / 키워드3개+요약 / ④ 후보 점수까지 `debug` 에 실립니다.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, File, Form, UploadFile

from app.config import settings
from app.exceptions import EmptyFileError, FileTooLargeError
from app.logging_config import get_logger
from app.schemas import (
    AnalyzeResponse,
    AnalyzeTextRequest,
    CandidateItem,
    CaseHitItem,
    CaseLookupDebug,
    FaqHitItem,
    FaqLookupDebug,
    CategoriesResponse,
    CategoryInfo,
    ChoiceDetail,
    ErrorResponse,
    FileMeta,
    GateResult,
    KeywordItem,
    PipelineDebug,
    PipelineStatusResponse,
    ToolCallResult,
    ToolExecutionResultSchema,
)
from app.services import categories, extraction, file_types, pipeline, runtime
from app.services.pipeline import PipelineResult

logger = get_logger(__name__)

router = APIRouter()

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "빈 파일 등 잘못된 요청"},
    413: {"model": ErrorResponse, "description": "업로드 용량 초과"},
    415: {"model": ErrorResponse, "description": "지원하지 않는 파일 형식"},
    422: {"model": ErrorResponse, "description": "추출 실패 또는 텍스트 없음"},
    503: {
        "model": ErrorResponse,
        "description": "모델 미설치·로드 실패 (bge-m3 / Gemma), 또는 PP-OCRv5 미설치",
    },
}


# =============================================================================
# 응답 조립
# =============================================================================
def _to_candidate_items(scores) -> list[CandidateItem]:
    return [
        CandidateItem(rank=c.rank, code=c.code, name=c.name, score=c.score)
        for c in scores
    ]


def _to_choice(choice) -> ChoiceDetail:
    return ChoiceDetail(
        number=choice.number,
        label=choice.label,
        score=choice.score,
        scores=choice.scores,
    )


def _to_tool_result(tr) -> ToolExecutionResultSchema | None:
    if tr is None:
        return None
    return ToolExecutionResultSchema(
        executed=tr.executed, ok=tr.ok, message=tr.message, data=tr.data, error=tr.error,
        needs_selection=tr.needs_selection, search=tr.search, warnings=list(tr.warnings),
    )


def _to_case_lookup(lookup) -> CaseLookupDebug | None:
    if lookup is None:
        return None
    return CaseLookupDebug(
        used=lookup.used,
        reason=lookup.reason,
        hits=[
            CaseHitItem(rank=h.rank, text=h.text, category=h.category, score=h.score)
            for h in lookup.hits
        ],
        case_scores=lookup.case_scores,
        candidates_before=_to_candidate_items(lookup.before),
        candidates_after=_to_candidate_items(lookup.after),
    )


def _to_faq_lookup(lookup) -> FaqLookupDebug | None:
    if lookup is None:
        return None
    return FaqLookupDebug(
        mode=lookup.mode,
        fallback=lookup.fallback,
        model_declined=lookup.model_declined,
        reason=lookup.reason,
        best_score=lookup.best_score,
        min_score=lookup.min_score,
        hits=[
            FaqHitItem(rank=h.rank, id=h.id, question=h.question, answer=h.answer, score=h.score, matched=h.matched)
            for h in lookup.hits
        ],
        selected_ids=[h.id for h in lookup.selected],
        model_answer=lookup.model_answer,
    )


def _build_response(result: PipelineResult, file_meta: FileMeta | None) -> AnalyzeResponse:
    # llm 은 항상 채워져 있습니다. ⑤ 의도 판정(6지선다)까지는 반려 여부와 무관하게
    # 항상 호출되고, 게이트는 그 결과("해당없음"인지)만 보고 통과/반려를 가릅니다.
    llm = result.llm
    gate = GateResult(
        passed=result.gate.passed,
        intent_score=result.gate.intent_score,
        enabled=result.gate.enabled,
        reason=result.gate.reason,
    )

    debug = None
    if settings.DEBUG:
        kp = result.keyphrase
        cand = result.candidate
        debug = PipelineDebug(
            source_text=result.source_text,
            sentence_count=len(kp.sentences),
            keywords=[KeywordItem(keyword=k.text, score=k.score) for k in kp.keywords],
            summary=kp.summary,
            keyphrase_method=kp.method,
            embed_query_text=kp.query_text,
            embedding_dim=result.embedding_dim,
            embedding_preview=result.embedding_preview,
            candidates_top=_to_candidate_items(cand.top),
            candidates_all=_to_candidate_items(cand.all_scores),
            candidate_margin=cand.margin,
            low_confidence=cand.low_confidence,
            case_lookup=_to_case_lookup(result.case_lookup),
            faq_lookup=_to_faq_lookup(llm.faq_lookup),
            intent_choice=_to_choice(llm.intent_choice),
            category_choice=_to_choice(llm.category_choice) if llm.category_choice.scores else None,
            llm_raw_output=llm.tool_call.raw,
            timings_ms=result.timings_ms,
            warnings=result.warnings,
            debug_text=pipeline.render_debug_text(result),
        )

    # --- 게이트 반려: 카테고리 확정·도구 호출 없이 안내 문구만 ---
    # llm.intent 는 "해당없음" 이고, category_choice.scores 는 비어 있습니다.
    # (llm_engine.decide() 가 의도 판정 직후 그 단계들을 건너뛰었기 때문입니다)
    if result.rejected:
        return AnalyzeResponse(
            status="rejected",
            file=file_meta,
            result_text=pipeline.render_result_text(result),
            gate=gate,
            intent=llm.intent.name,
            intent_code=llm.intent.code,
            intent_score=llm.intent_choice.score,
            answer=pipeline.REJECT_MESSAGE,
            debug=debug,
        )

    # --- GATE_ENABLED=false 관찰 모드로 통과된 '해당없음' ---
    # 실제로는 반려 대상이지만 관찰을 위해 막지 않은 경우라, 카테고리·도구가 없습니다.
    if llm.intent.code == "out_of_scope":
        return AnalyzeResponse(
            status="accepted",
            file=file_meta,
            result_text=pipeline.render_result_text(result),
            gate=gate,
            intent=llm.intent.name,
            intent_code=llm.intent.code,
            intent_score=llm.intent_choice.score,
            answer=(
                "GATE_ENABLED=false 관찰 모드라 반려하지 않고 통과시켰습니다. "
                "카테고리 확정과 도구 호출은 생략되었습니다."
            ),
            debug=debug,
        )

    # 문의(판정 안 함)나 조회·수정·삭제의 '없음' 이면 카테고리 필드는 비워 둡니다.
    category = categories.get(llm.category_name) if llm.category_name else None
    return AnalyzeResponse(
        status="accepted",
        file=file_meta,
        result_text=pipeline.render_result_text(result),
        gate=gate,
        intent=llm.intent.name,
        intent_code=llm.intent.code,
        intent_score=llm.intent_choice.score,
        category=category.name if category else None,
        category_code=category.code if category else None,
        category_score=llm.category_choice.score if llm.category_choice.scores else None,
        department=category.department if category else None,
        tool_call=ToolCallResult(
            called=llm.tool_call.called,
            name=llm.tool_call.name,
            arguments=llm.tool_call.arguments,
            parse_error=llm.tool_call.parse_error,
        ),
        tool_result=_to_tool_result(result.tool_result),
        answer=llm.answer,
        debug=debug,
    )


async def _read_upload(upload: UploadFile) -> bytes:
    """업로드 파일을 용량 제한을 지키며 읽습니다. (extract.py 와 같은 규칙)"""
    chunks: list[bytes] = []
    total = 0
    while chunk := await upload.read(1024 * 1024):
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise FileTooLargeError(
                f"파일이 너무 큽니다. 최대 {settings.MAX_UPLOAD_MB}MB 까지 업로드할 수 있습니다.",
                detail=f"uploaded>{total} bytes",
            )
        chunks.append(chunk)

    if total == 0:
        raise EmptyFileError("빈 파일입니다. 내용이 있는 파일을 업로드해 주세요.")
    return b"".join(chunks)


# =============================================================================
# 엔드포인트
# =============================================================================
@router.post(
    "/analyze/text",
    response_model=AnalyzeResponse,
    summary="[텍스트 입력] 민원 문장 분류",
    description=(
        "파일 없이 민원 문장만 넣어 파이프라인 ②~⑥ 을 확인합니다.\n\n"
        "**처리 순서**\n"
        "1. 원문에서 키워드 3개 + 요약문을 만듭니다. (kiwipiepy + KeyBERT + TextRank)\n"
        "2. 그 질의문을 bge-m3 로 1024차원 벡터로 만듭니다. (③)\n"
        "3. 카테고리 7종 정의문과 코사인 유사도를 계산해 상위 3개를 추립니다. (④, 후보만 좁힘) "
        "벡터DB 의 비슷한 라벨링 사례로 상위 3개를 다시 정렬합니다. "
        "(`CASE_MODE`: low_confidence 면 1위 유사도가 `CANDIDATE_MIN_SCORE` 미만일 때만, "
        "always·union 이면 매번)\n"
        "4. Gemma 4 E4B 가 의도를 6지선다로 판정합니다. 문의·접수·조회·수정·삭제 중 "
        "하나면 이어서 후보 3개 중 카테고리를 확정하고 도구 호출 JSON 을 만듭니다. "
        "**어디에도 해당하지 않아 '해당없음'으로 판정되면 카테고리·도구 없이 여기서 멈추고 "
        "`status=\"rejected\"` 와 안내 문구(`answer`)만 돌려줍니다.** "
        "카테고리(④)와는 무관한 판단이라, \"제가 어제 문의한 내용 보여줘\"처럼 특정 "
        "주제와 뚜렷이 겹치지 않는 문장도 의도(조회)만 명확하면 통과합니다. (⑤)\n"
        "5. 접수·조회·수정·삭제는 도구 호출 JSON 을 실제 SQLite DB 에 바로 실행합니다. (⑥) "
        "실행 전 소유권(본인 민원인지)·상태(이미 취소된 건 아닌지) 등을 검증하고, 통과하면 "
        "즉시 반영합니다. 결과는 `tool_result` 에 담깁니다.\n\n"
        "**첫 요청은 느립니다.** bge-m3(약 2.2GB)와 Gemma 4 E4B(약 16GB) 가중치를 내려받고 "
        "메모리에 올리기 때문입니다. 두 번째 요청부터는 로드된 모델을 재사용합니다.\n\n"
        "`user_id` 를 비우면 서버 기본값(`COMPLAINT_DEFAULT_USER`) 하나로 통일됩니다. "
        "조회·수정·삭제는 같은 `user_id` 로 등록한 민원만 대상이 됩니다.\n\n"
        "`.env` 의 `DEBUG=true` 이면 응답의 `debug` 에 추출 원문 · 키워드3개+요약 · "
        "④ 후보 점수 · ⑤ 원시 출력이 모두 실립니다. (`debug.debug_text` 는 같은 내용을 "
        "한 덩어리 텍스트로 정리한 것입니다)"
    ),
    responses=_ERROR_RESPONSES,
)
async def analyze_text(payload: AnalyzeTextRequest) -> AnalyzeResponse:
    request_id = uuid.uuid4().hex[:12]
    logger.info("[%s] 분류 요청(텍스트) | chars=%d", request_id, len(payload.text))

    started = time.perf_counter()
    result = await asyncio.to_thread(pipeline.run, payload.text, request_id, payload.user_id)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    logger.info("[%s] 분류 완료 | %dms", request_id, elapsed_ms)
    return _build_response(result, file_meta=None)


@router.post(
    "/analyze/file",
    response_model=AnalyzeResponse,
    summary="[파일 입력] 문서에서 텍스트를 뽑아 분류",
    description=(
        "파일 1개를 업로드하면 ② 텍스트 추출까지 끝낸 뒤 그대로 ③④⑤ 로 넘깁니다.\n\n"
        "이미지·PDF 는 PP-OCRv5 로 글자를 읽고, 그 외 문서는 전용 라이브러리로 추출합니다. "
        "(`POST /extract/text` 와 동일한 경로를 사용합니다)\n\n"
        "그림 ① 의 '이미지+텍스트' 입력은 선택 입력칸 `text` 에 문장을 함께 적으면 됩니다. "
        "적으면 문서에서 뽑은 텍스트 앞에 붙어 함께 분류됩니다.\n\n"
        "`.env` 의 `DEBUG=true` 이면 응답의 `debug` 에 단계별 중간 결과가 모두 실립니다."
    ),
    responses=_ERROR_RESPONSES,
)
async def analyze_file(
    file: UploadFile = File(
        ..., description="분류할 파일 (pdf / 이미지 / docx·xlsx·pptx / hwp·hwpx / txt 등)"
    ),
    text: str = Form(
        "", description="(선택) 함께 넣을 민원 문장. 그림 ① 의 '이미지+텍스트' 입력."
    ),
    user_id: str = Form(
        "", description="로그인한 사용자 취급할 식별자. 비우면 서버 기본값을 씁니다."
    ),
) -> AnalyzeResponse:
    request_id = uuid.uuid4().hex[:12]
    filename = file.filename or "unnamed"
    logger.info("[%s] 분류 요청(파일) | filename=%s", request_id, filename)

    kind = file_types.detect(filename)
    data = await _read_upload(file)

    extracted = await asyncio.to_thread(extraction.dispatch, data, kind)
    source_text = extracted.full_text
    if text.strip():
        # ① 이미지+텍스트 : 사용자가 직접 쓴 문장을 앞에 붙입니다.
        source_text = f"{text.strip()}\n\n{source_text}".strip()

    logger.info(
        "[%s] ② 추출 완료 | chars=%d | pages=%d", request_id,
        len(source_text), len(extracted.segments),
    )

    result = await asyncio.to_thread(pipeline.run, source_text, request_id, user_id)
    # ② 추출 단계 경고(스캔 페이지 건너뜀, 인코딩 추정 등)도 debug.warnings 에 함께 보여줍니다.
    result.warnings[:0] = [f"[②추출] {w}" for w in extracted.warnings if w]

    return _build_response(
        result,
        file_meta=FileMeta(
            filename=filename, extension=kind.extension, category=kind.category
        ),
    )


@router.get(
    "/analyze/categories",
    response_model=CategoriesResponse,
    summary="카테고리 7종 정의",
    description="④ 후보 추림과 ⑤ 카테고리 확정에 사용하는 카테고리 정의입니다.",
)
async def list_categories() -> CategoriesResponse:
    items = [
        CategoryInfo(
            code=c.code,
            name=c.name,
            department=c.department,
            summary=c.summary,
            keywords=list(c.keywords),
        )
        for c in categories.CATEGORIES
    ]
    return CategoriesResponse(count=len(items), categories=items)


@router.get(
    "/analyze/status",
    response_model=PipelineStatusResponse,
    summary="모델 설치·로드 상태",
    description=(
        "bge-m3 / Gemma 가 설치되어 있는지, 이미 메모리에 올라와 있는지 확인합니다. "
        "`case_store` 에서 벡터DB(라벨링된 사례) 준비 상태와 사례 수를, "
        "`faq_store` 에서 '문의' 답변용 FAQ 검색 준비 상태를 볼 수 있습니다.\n"
        "`loaded=false` 면 다음 요청에서 가중치를 내려받고 로드하느라 느릴 수 있습니다."
    ),
)
async def pipeline_status() -> PipelineStatusResponse:
    state = pipeline.status()
    return PipelineStatusResponse(
        debug=settings.DEBUG,
        device=runtime.resolve_device(),
        runtime=runtime.describe(),
        keyphrase=state["keyphrase"],   # type: ignore[arg-type]
        embedder=state["embedder"],     # type: ignore[arg-type]
        case_store=state["case_store"], # type: ignore[arg-type]
        faq_store=state["faq_store"],   # type: ignore[arg-type]
        chat=state["chat"],             # type: ignore[arg-type]
        complaint_store=state["complaint_store"],  # type: ignore[arg-type]
        llm=state["llm"],               # type: ignore[arg-type]
    )
