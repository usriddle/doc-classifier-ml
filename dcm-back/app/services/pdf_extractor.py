"""
PDF 추출기 (PyMuPDF + PP-OCRv5).

페이지마다 아래 둘 중 하나로 처리합니다.

1. 텍스트 레이어가 없는 페이지 (스캔본 등)
   -> 페이지 전체를 이미지로 렌더링해 PP-OCRv5 로 보냅니다.

2. 텍스트 레이어가 있는 페이지
   -> 텍스트는 PyMuPDF 가 그대로 추출하고,
      페이지에 박힌 이미지는 PyMuPDF 가 좌표(bbox)를 계산해 그 영역만 잘라낸 뒤
      PP-OCRv5 로 보냅니다. 즉 이미지 안의 글자만 따로 인식합니다.
"""

from __future__ import annotations

from app.config import settings
from app.exceptions import ExtractionError
from app.logging_config import get_logger
from app.services import ocr_engine
from app.services.base import ExtractedText, Segment

logger = get_logger(__name__)


def _open(data: bytes):
    try:
        import pymupdf  # PyMuPDF 1.24+
    except ImportError:
        try:
            import fitz as pymupdf  # 구버전 import 이름
        except ImportError as exc:
            raise ExtractionError(
                "PyMuPDF 가 설치되어 있지 않습니다.",
                detail="python -m pip install -r requirements.txt",
            ) from exc

    try:
        return pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise ExtractionError(
            "PDF 를 열지 못했습니다.",
            detail=f"{type(exc).__name__}: {exc} / 암호가 걸렸거나 손상된 파일일 수 있습니다.",
        ) from exc


def _image_rects(page) -> list:
    """페이지에 배치된 이미지들의 실제 좌표(bbox) 목록."""
    rects = []
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        return rects

    for info in infos:
        bbox = info.get("bbox")
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        width, height = abs(x1 - x0), abs(y1 - y0)
        # 아이콘·구분선처럼 작은 이미지는 건너뜁니다.
        if width < settings.OCR_MIN_IMAGE_PX or height < settings.OCR_MIN_IMAGE_PX:
            continue
        rects.append((x0, y0, x1, y1))
    return rects


def extract(data: bytes) -> ExtractedText:
    document = _open(data)
    segments: list[Segment] = []
    warnings: list[str] = []
    ocr_pages = 0      # 페이지 전체를 OCR 한 횟수
    ocr_images = 0     # 좌표로 잘라 OCR 한 이미지 영역 수
    ocr_ready = ocr_engine.is_available()

    if not ocr_ready:
        warnings.append(
            "PP-OCRv5 가 설치되어 있지 않아 이미지·스캔 페이지는 건너뜁니다. "
            "(README.md 의 1-3. PP-OCRv5 설치 참고)"
        )

    try:
        for number, page in enumerate(document, start=1):
            try:
                text = (page.get_text("text") or "").strip()
            except Exception as exc:
                warnings.append(f"{number}쪽 텍스트 추출 실패: {type(exc).__name__}")
                text = ""

            # --- 1) 텍스트 레이어가 없는 페이지 : 전체 OCR ---
            if len(text) < settings.PDF_MIN_TEXT_CHARS:
                if not ocr_ready:
                    segments.append(Segment(number, f"page:{number} (건너뜀)", ""))
                    continue

                logger.info("PDF %d쪽 : 텍스트 없음 -> 페이지 전체 OCR", number)
                rendered = _render(page, number, warnings)
                ocr_text = ocr_engine.run(rendered) if rendered else ""
                if rendered:
                    ocr_pages += 1
                segments.append(Segment(number, f"page:{number} (ocr)", ocr_text))
                continue

            # --- 2) 텍스트 레이어가 있는 페이지 : 텍스트 + 이미지 영역 OCR ---
            parts = [text]

            if ocr_ready and settings.PDF_OCR_EMBEDDED_IMAGES:
                for order, rect in enumerate(_image_rects(page), start=1):
                    rendered = _render(page, number, warnings, clip=rect)
                    if not rendered:
                        continue
                    ocr_text = ocr_engine.run(rendered)
                    ocr_images += 1
                    if not ocr_text:
                        continue
                    x0, y0, x1, y1 = rect
                    parts.append(
                        f"[이미지 {order} OCR | 좌표 "
                        f"({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})]\n{ocr_text}"
                    )

            segments.append(Segment(number, f"page:{number}", "\n\n".join(parts)))
    finally:
        document.close()

    return ExtractedText(
        segments=segments,
        warnings=warnings,
        stats={"ocr_pages": ocr_pages, "ocr_images": ocr_images},
    )


def _render(page, number: int, warnings: list[str], clip=None) -> bytes | None:
    """페이지(또는 지정 영역)를 PNG 바이트로 렌더링합니다."""
    try:
        pixmap = page.get_pixmap(dpi=settings.OCR_DPI, clip=clip)
        return pixmap.tobytes("png")
    except Exception as exc:
        warnings.append(f"{number}쪽 이미지 렌더링 실패: {type(exc).__name__}: {exc}")
        return None
