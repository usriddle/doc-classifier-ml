"""
① ② 입력 -> 텍스트 추출 공통 진입점.

/extract/text 와 /analyze/file 이 같은 경로를 쓰도록 분리했습니다.
확장자에 맞는 추출기를 고르는 일만 합니다.
"""

from __future__ import annotations

from app.services import doc_extractor, file_types, image_extractor, pdf_extractor
from app.services.base import ExtractedText
from app.services.file_types import FileKind


def dispatch(data: bytes, kind: FileKind) -> ExtractedText:
    """확장자에 맞는 추출기를 호출합니다. (블로킹 - 스레드에서 실행하세요)"""
    if kind.handler == file_types.PDF:
        return pdf_extractor.extract(data)
    if kind.handler == file_types.IMAGE:
        return image_extractor.extract(data)
    return doc_extractor.extract(data, kind.extension)
