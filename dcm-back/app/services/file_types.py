"""
확장자 기반 파일 판별 및 처리 담당 모듈 결정.

handler
  - DOC   : doc_extractor   (MS Office / 한글 / 텍스트)
  - PDF   : pdf_extractor   (PyMuPDF + 필요 시 PP-OCRv5)
  - IMAGE : image_extractor (PP-OCRv5 전담)
"""

from __future__ import annotations

from dataclasses import dataclass

from app.exceptions import UnsupportedFormatError

DOC = "doc"
PDF = "pdf"
IMAGE = "image"


@dataclass(frozen=True)
class FileKind:
    extension: str      # ".pdf"
    category: str       # pdf | image | office | hangul | text
    mime_type: str
    handler: str        # DOC | PDF | IMAGE
    parser: str         # 실제로 사용하는 라이브러리 이름
    requires_ocr: bool  # True 면 PP-OCRv5 없이는 전혀 처리할 수 없음


# --- PDF ---------------------------------------------------------------
# 텍스트 레이어가 있으면 PyMuPDF 만으로 처리되므로 requires_ocr 는 False 입니다.
# (스캔 PDF 나 이미지가 박힌 페이지에서만 PP-OCRv5 가 필요합니다)
_PDF = {
    ".pdf": FileKind(".pdf", "pdf", "application/pdf", PDF, "PyMuPDF + PP-OCRv5", False),
}

# --- 이미지 : 전부 PP-OCRv5 로 처리 -------------------------------------
_IMAGE_EXTS = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
    ".heic": "image/heic",
    ".heif": "image/heif",
}
_IMAGE = {
    ext: FileKind(ext, "image", mime, IMAGE, "PP-OCRv5", True)
    for ext, mime in _IMAGE_EXTS.items()
}

# --- MS Office ---------------------------------------------------------
_OFFICE = {
    ".docx": FileKind(
        ".docx", "office",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        DOC, "docx2txt", False,
    ),
    ".xlsx": FileKind(
        ".xlsx", "office",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        DOC, "openpyxl", False,
    ),
    ".xlsm": FileKind(
        ".xlsm", "office", "application/vnd.ms-excel.sheet.macroEnabled.12",
        DOC, "openpyxl", False,
    ),
    ".pptx": FileKind(
        ".pptx", "office",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        DOC, "python-pptx", False,
    ),
}

# --- 한글 ---------------------------------------------------------------
_HANGUL = {
    ".hwp": FileKind(".hwp", "hangul", "application/x-hwp", DOC, "olefile", False),
    ".hwpx": FileKind(".hwpx", "hangul", "application/hwp+zip", DOC, "zipfile+xml", False),
}

# --- 텍스트류 -----------------------------------------------------------
_TEXT_EXTS = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".tsv": "text/tab-separated-values",
    ".json": "application/json",
    ".xml": "application/xml",
    ".html": "text/html",
    ".htm": "text/html",
    ".log": "text/plain",
}
_TEXT = {
    ext: FileKind(ext, "text", mime, DOC, "charset-normalizer", False)
    for ext, mime in _TEXT_EXTS.items()
}
_TEXT[".rtf"] = FileKind(".rtf", "text", "application/rtf", DOC, "striprtf", False)

SUPPORTED: dict[str, FileKind] = {**_PDF, **_IMAGE, **_OFFICE, **_HANGUL, **_TEXT}

# 구버전 바이너리 포맷 - 별도 안내 메시지 제공
_LEGACY_HINT = {
    ".doc": ".docx",
    ".xls": ".xlsx",
    ".ppt": ".pptx",
}


def detect(filename: str) -> FileKind:
    """파일명에서 확장자를 뽑아 처리 방식을 결정합니다."""
    name = (filename or "").strip()
    if "." not in name:
        raise UnsupportedFormatError(
            "확장자가 없는 파일은 처리할 수 없습니다.",
            detail=f"filename={name!r}",
        )

    ext = "." + name.rsplit(".", 1)[-1].lower()

    kind = SUPPORTED.get(ext)
    if kind is not None:
        return kind

    if ext in _LEGACY_HINT:
        raise UnsupportedFormatError(
            f"{ext} 는 구버전 바이너리 포맷이라 지원하지 않습니다. "
            f"{_LEGACY_HINT[ext]} 로 변환 후 다시 업로드해 주세요.",
            detail=f"extension={ext}",
        )

    raise UnsupportedFormatError(
        f"지원하지 않는 파일 형식입니다: {ext}",
        detail="지원 목록은 GET /api/v1/formats 에서 확인할 수 있습니다.",
    )


_DESCRIPTIONS = {
    "pdf": (
        "페이지마다 판단합니다. 텍스트가 있으면 PyMuPDF 가 추출하고 "
        "페이지에 박힌 이미지는 좌표로 잘라 PP-OCRv5 에 보냅니다. "
        "텍스트가 없는 스캔 페이지는 페이지 전체를 PP-OCRv5 로 보냅니다."
    ),
    "image": "파일 전체를 PP-OCRv5 로 보내 글자를 인식합니다.",
    "office": "docx2txt / openpyxl / python-pptx 로 문단·시트·슬라이드 텍스트를 추출합니다.",
    "hangul": "hwp 는 본문 레코드를, hwpx 는 XML 을 직접 파싱합니다.",
    "text": "charset-normalizer 로 인코딩(UTF-8/CP949 등)을 감지해 디코딩합니다.",
}


def format_catalog() -> list[dict[str, object]]:
    """/formats 엔드포인트용 지원 포맷 목록."""
    groups: dict[str, list[FileKind]] = {}
    for kind in SUPPORTED.values():
        groups.setdefault(kind.category, []).append(kind)

    catalog = []
    for category, kinds in sorted(groups.items()):
        first = kinds[0]
        parsers = sorted({k.parser for k in kinds})
        catalog.append(
            {
                "category": category,
                "extensions": sorted(k.extension for k in kinds),
                "handler": first.handler,
                "parser": " / ".join(parsers),
                "requires_ocr": first.requires_ocr,
                "description": _DESCRIPTIONS.get(category, ""),
            }
        )
    return catalog
