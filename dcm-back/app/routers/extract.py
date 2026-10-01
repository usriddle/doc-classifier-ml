"""
텍스트 추출 API.

POST /api/v1/extract/text  : 파일 1개 업로드 -> 텍스트를 JSON 으로 반환
GET  /api/v1/formats       : 지원 확장자 목록

응답 본문은 청킹/임베딩 파이프라인에 바로 넘길 수 있도록
본문(text)과 최소 식별 정보만 담습니다. 처리 방식·소요 시간·OCR 횟수 같은
운영 정보는 응답 대신 서버 로그(logs/app.log)에서 확인합니다.
"""

from __future__ import annotations

import asyncio
import time
import uuid

from fastapi import APIRouter, File, UploadFile

from app.config import settings
from app.exceptions import EmptyFileError, FileTooLargeError
from app.logging_config import get_logger
from app.schemas import (
    ErrorResponse,
    ExtractedPage,
    FileMeta,
    SupportedFormatsResponse,
    TextExtractionResponse,
)
from app.services import extraction, file_types, ocr_engine
from app.services.base import ExtractedText

logger = get_logger(__name__)

router = APIRouter()

_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "빈 파일 등 잘못된 요청"},
    413: {"model": ErrorResponse, "description": "업로드 용량 초과"},
    415: {"model": ErrorResponse, "description": "지원하지 않는 파일 형식"},
    422: {"model": ErrorResponse, "description": "텍스트 추출 실패 (파일 손상·암호 등)"},
    503: {"model": ErrorResponse, "description": "PP-OCRv5 미설치 (이미지 파일 요청 시)"},
}


async def _read_upload(upload: UploadFile) -> bytes:
    """업로드 파일을 용량 제한을 지키며 메모리로 읽어 들입니다."""
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


def _dispatch(data: bytes, kind: file_types.FileKind) -> ExtractedText:
    """확장자에 맞는 추출기를 호출합니다. (스레드에서 실행됩니다)

    /analyze/file 과 같은 경로를 쓰도록 services/extraction.py 로 옮겼습니다.
    """
    return extraction.dispatch(data, kind)


@router.post(
    "/extract/text",
    response_model=TextExtractionResponse,
    summary="파일에서 텍스트 추출",
    description=(
        "파일 1개를 업로드하면 문서 안의 텍스트를 추출해 JSON 으로 반환합니다.\n\n"
        "응답은 청킹/임베딩 파이프라인에 바로 넘길 수 있도록 본문과 최소 식별 정보만 "
        "담습니다. 처리 방식·소요 시간·OCR 횟수 등은 서버 로그에서 확인하세요.\n\n"
        "**확장자별 처리 방식**\n"
        "- PDF : 페이지마다 판단합니다. 텍스트가 있으면 PyMuPDF 가 추출하고, "
        "페이지에 박힌 이미지는 PyMuPDF 가 좌표로 잘라 PP-OCRv5 에 보냅니다. "
        "텍스트가 없는 스캔 페이지는 페이지 전체를 PP-OCRv5 로 보냅니다.\n"
        "- 이미지 : `.png` `.jpg` 등 전부 PP-OCRv5 로 처리합니다.\n"
        "- MS Office : `.docx`(docx2txt) `.xlsx` `.xlsm`(openpyxl) `.pptx`(python-pptx)\n"
        "- 한글 : `.hwp`(olefile) `.hwpx`(zipfile+xml)\n"
        "- 텍스트 : `.txt` `.csv` 등(charset-normalizer) `.rtf`(striprtf)\n\n"
        "PP-OCRv5 가 설치되어 있지 않으면 이미지 파일은 503, "
        "PDF 는 텍스트 페이지만 처리하고 로그에 경고를 남깁니다.\n\n"
        "아래 *Try it out* 버튼을 눌러 파일을 선택하면 바로 테스트할 수 있습니다."
    ),
    responses=_ERROR_RESPONSES,
)
async def extract_text(
    file: UploadFile = File(..., description="추출할 파일 (pdf / 이미지 / docx·xlsx·pptx / hwp·hwpx / txt 등)"),
) -> TextExtractionResponse:
    request_id = uuid.uuid4().hex[:12]  # 로그 추적용. 응답에는 포함하지 않습니다.
    filename = file.filename or "unnamed"

    logger.info("[%s] 추출 요청 시작 | filename=%s", request_id, filename)

    # 1) 확장자 판별 (지원하지 않는 포맷은 여기서 415 로 차단 + 로그 기록)
    try:
        kind = file_types.detect(filename)
    except Exception:
        logger.warning("[%s] 지원하지 않는 포맷 | filename=%s", request_id, filename)
        raise

    # 2) 파일 읽기 및 추출
    data = await _read_upload(file)

    started = time.perf_counter()
    result = await asyncio.to_thread(_dispatch, data, kind)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    # 3) 응답 조립 (본문 + 최소 식별 정보만)
    pages = [
        ExtractedPage(page=seg.index, label=seg.label, text=seg.text)
        for seg in result.segments
    ]

    response = TextExtractionResponse(
        file=FileMeta(
            filename=filename,
            extension=kind.extension,
            category=kind.category,
        ),
        pages=pages,
    )

    # 처리 방식·소요 시간·OCR 횟수·경고는 응답 대신 로그로만 남깁니다.
    char_count = sum(len(p.text) for p in pages)
    warnings = [w for w in result.warnings if w]
    logger.info(
        "[%s] 추출 완료 | handler=%s | parser=%s | chars=%d | pages=%d "
        "| ocr(page=%d, image=%d) | %dms%s",
        request_id, kind.handler, kind.parser, char_count, len(pages),
        result.stats.get("ocr_pages", 0), result.stats.get("ocr_images", 0),
        elapsed_ms, f" | warnings={warnings}" if warnings else "",
    )
    return response


@router.get(
    "/formats",
    response_model=SupportedFormatsResponse,
    summary="지원 파일 형식 목록",
)
async def supported_formats() -> SupportedFormatsResponse:
    return SupportedFormatsResponse(
        max_upload_mb=settings.MAX_UPLOAD_MB,
        ocr_available=ocr_engine.is_available(),
        formats=file_types.format_catalog(),  # type: ignore[arg-type]
    )