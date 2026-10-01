"""이미지 파일 추출기. 파일 전체를 PP-OCRv5 로 보냅니다."""

from __future__ import annotations

from app.exceptions import OcrUnavailableError
from app.logging_config import get_logger
from app.services import ocr_engine
from app.services.base import ExtractedText, Segment

logger = get_logger(__name__)


def extract(data: bytes) -> ExtractedText:
    if not ocr_engine.is_available():
        raise OcrUnavailableError(
            "이미지 처리에 필요한 PP-OCRv5(PaddleOCR)가 설치되어 있지 않습니다.",
            detail="README.md 의 1-3. PP-OCRv5 설치를 참고하세요.",
        )

    text = ocr_engine.run(data)
    warnings = [] if text else ["인식된 글자가 없습니다."]
    return ExtractedText(
        segments=[Segment(1, "image (ocr)", text)],
        warnings=warnings,
        stats={"ocr_pages": 1},
    )
