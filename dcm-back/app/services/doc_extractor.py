"""
문서 텍스트 추출기 (MS Office / 한글 / 텍스트).

각 추출기는 (라벨, 텍스트) 형태의 Segment 목록을 돌려주며,
라벨은 페이지 / 시트 / 슬라이드 등 문서 종류에 따라 달라집니다.
"""

from __future__ import annotations

import io
import re
import struct
import zipfile
from xml.etree import ElementTree

from app.exceptions import ExtractionError
from app.logging_config import get_logger
from app.services.base import ExtractedText, Segment

logger = get_logger(__name__)


# =============================================================================
# 진입점
# =============================================================================
def extract(data: bytes, extension: str) -> ExtractedText:
    """확장자에 맞는 추출기를 호출합니다."""
    handlers = {
        ".docx": _extract_docx,
        ".xlsx": _extract_xlsx,
        ".xlsm": _extract_xlsx,
        ".pptx": _extract_pptx,
        ".hwp": _extract_hwp,
        ".hwpx": _extract_hwpx,
        ".rtf": _extract_rtf,
    }
    handler = handlers.get(extension, _extract_plain_text)

    try:
        result = handler(data)
    except ExtractionError:
        raise
    except ImportError as exc:  # 라이브러리 미설치
        raise ExtractionError(
            f"{extension} 처리에 필요한 라이브러리가 설치되어 있지 않습니다.",
            detail=f"{exc} / python -m pip install -r requirements.txt 를 실행하세요.",
        ) from exc
    except Exception as exc:  # 손상된 파일 등
        raise ExtractionError(
            f"{extension} 파일에서 텍스트를 추출하지 못했습니다.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    if result.char_count == 0:
        result.warnings.append("추출 결과가 비어 있습니다. 내용이 없는 파일일 수 있습니다.")
    return result


# =============================================================================
# MS Office
# =============================================================================
def _extract_docx(data: bytes) -> ExtractedText:
    import docx2txt

    text = docx2txt.process(io.BytesIO(data)) or ""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # .docx 는 페이지 경계가 렌더링 시점에 정해지므로 문서 전체를 1개 세그먼트로 처리
    return ExtractedText(
        segments=[Segment(1, "document", text)],
        warnings=["docx2txt 는 표를 행/열 구조 없이 텍스트로 펼쳐 추출합니다."] if text else [],
    )


def _extract_xlsx(data: bytes) -> ExtractedText:
    from openpyxl import load_workbook

    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    segments: list[Segment] = []

    try:
        for index, sheet in enumerate(workbook.worksheets, start=1):
            rows: list[str] = []
            for row in sheet.iter_rows(values_only=True):
                values = ["" if v is None else str(v).strip() for v in row]
                if any(values):
                    rows.append("\t".join(values).rstrip())
            segments.append(Segment(index, f"sheet:{sheet.title}", "\n".join(rows)))
    finally:
        workbook.close()

    return ExtractedText(segments=segments)


def _extract_pptx(data: bytes) -> ExtractedText:
    from pptx import Presentation

    presentation = Presentation(io.BytesIO(data))
    segments: list[Segment] = []

    for index, slide in enumerate(presentation.slides, start=1):
        lines: list[str] = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in paragraph.runs).strip()
                    if text:
                        lines.append(text)
            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    if any(cells):
                        lines.append(" | ".join(cells))

        if slide.has_notes_slide:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()
            if notes:
                lines.append(f"[발표자 노트] {notes}")

        segments.append(Segment(index, f"slide:{index}", "\n".join(lines)))

    return ExtractedText(segments=segments)


# =============================================================================
# 한글 (HWP / HWPX)
# =============================================================================
# HWP 5.0 본문 레코드 태그
_HWPTAG_BEGIN = 0x10
_HWPTAG_PARA_TEXT = _HWPTAG_BEGIN + 51  # 67

# 본문 텍스트 안에 섞여 있는 제어문자 분류 (단위: UTF-16 코드유닛)
_CTRL_CHAR = {0, 10, 13, 24, 25, 26, 27, 28, 29, 30, 31}          # 1개 소비
_CTRL_INLINE = {4, 5, 6, 7, 8, 9, 19, 20}                          # 3개 소비
_CTRL_EXTENDED = {1, 2, 3, 11, 12, 14, 15, 16, 17, 18, 21, 22, 23}  # 8개 소비


def _decode_hwp_paragraph(chunk: bytes) -> str:
    """HWPTAG_PARA_TEXT 레코드(UTF-16LE + 제어문자)를 일반 텍스트로 변환."""
    out: list[str] = []
    total = len(chunk) // 2
    i = 0
    while i < total:
        code = struct.unpack_from("<H", chunk, i * 2)[0]
        if code in _CTRL_CHAR:
            if code in (10, 13):
                out.append("\n")
            i += 1
        elif code in _CTRL_INLINE:
            i += 3
        elif code in _CTRL_EXTENDED:
            i += 8
        else:
            out.append(chr(code))
            i += 1
    return "".join(out)


def _extract_hwp(data: bytes) -> ExtractedText:
    import olefile
    import zlib

    stream = io.BytesIO(data)
    if not olefile.isOleFile(stream):
        raise ExtractionError(
            "유효한 HWP 5.0 파일이 아닙니다.",
            detail="한/글 2007 이전 포맷이거나 파일이 손상되었을 수 있습니다.",
        )

    ole = olefile.OleFileIO(stream)
    try:
        header = ole.openstream("FileHeader").read()
        is_compressed = bool(header[36] & 0x01)
        is_encrypted = bool(header[36] & 0x02)
        if is_encrypted:
            raise ExtractionError(
                "암호가 걸린 HWP 파일은 처리할 수 없습니다.",
                detail="한/글에서 암호를 해제한 뒤 다시 업로드해 주세요.",
            )

        section_names = sorted(
            (entry[1] for entry in ole.listdir() if entry[0] == "BodyText"),
            key=lambda name: int(re.sub(r"\D", "", name) or 0),
        )
        if not section_names:
            raise ExtractionError("HWP 본문(BodyText) 스트림을 찾지 못했습니다.")

        segments: list[Segment] = []
        for index, name in enumerate(section_names, start=1):
            raw = ole.openstream(f"BodyText/{name}").read()
            body = zlib.decompress(raw, -15) if is_compressed else raw

            paragraphs: list[str] = []
            cursor, size = 0, len(body)
            while cursor + 4 <= size:
                record_header = struct.unpack_from("<I", body, cursor)[0]
                tag_id = record_header & 0x3FF
                data_len = (record_header >> 20) & 0xFFF
                cursor += 4
                if data_len == 0xFFF:  # 확장 길이
                    if cursor + 4 > size:
                        break
                    data_len = struct.unpack_from("<I", body, cursor)[0]
                    cursor += 4
                if cursor + data_len > size:
                    break
                if tag_id == _HWPTAG_PARA_TEXT:
                    text = _decode_hwp_paragraph(body[cursor : cursor + data_len]).strip()
                    if text:
                        paragraphs.append(text)
                cursor += data_len

            segments.append(Segment(index, f"section:{index}", "\n".join(paragraphs)))
    finally:
        ole.close()

    return ExtractedText(segments=segments)


def _extract_hwpx(data: bytes) -> ExtractedText:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        section_files = sorted(
            name
            for name in archive.namelist()
            if re.match(r"Contents/section\d+\.xml$", name, re.IGNORECASE)
        )
        if not section_files:
            raise ExtractionError(
                "HWPX 본문(Contents/sectionN.xml)을 찾지 못했습니다.",
                detail="HWPX 규격이 아닌 파일일 수 있습니다.",
            )

        segments: list[Segment] = []
        for index, name in enumerate(section_files, start=1):
            root = ElementTree.fromstring(archive.read(name))
            paragraphs: list[str] = []
            buffer: list[str] = []

            for element in root.iter():
                tag = element.tag.split("}")[-1]
                if tag == "p":
                    if buffer:
                        paragraphs.append("".join(buffer).strip())
                        buffer = []
                elif tag == "t" and element.text:
                    buffer.append(element.text)

            if buffer:
                paragraphs.append("".join(buffer).strip())

            segments.append(
                Segment(index, f"section:{index}", "\n".join(p for p in paragraphs if p))
            )

    return ExtractedText(segments=segments)


# =============================================================================
# 텍스트류
# =============================================================================
# 한글 완성형/자모 영역 - 인코딩 판별에 사용합니다.
_HANGUL_RANGES = ((0xAC00, 0xD7A3), (0x3131, 0x318E), (0x1100, 0x11FF))

# 한국어 문서에서 실제로 쓰이는 레거시 인코딩
_KOREAN_ENCODINGS = ("cp949", "euc-kr")


def _has_hangul(text: str) -> bool:
    return any(
        any(low <= ord(ch) <= high for low, high in _HANGUL_RANGES)
        for ch in text
    )


def _decode_bytes(data: bytes) -> tuple[str, list[str]]:
    """
    인코딩을 자동 감지해 문자열로 디코딩합니다.

    순서가 중요합니다.
      1. UTF-8 (BOM 포함) - 가장 흔하므로 먼저 시도
      2. CP949 / EUC-KR 로 디코딩해 한글이 나오면 확정
         (charset-normalizer 는 짧은 한글 바이트열을 big5 등 다른 CJK 인코딩으로
          잘못 추정하는 경우가 있어, 한국어 문서를 먼저 걸러냅니다)
      3. charset-normalizer 자동 감지 - 일본어/중국어/서구권 문서 대응
      4. 그래도 실패하면 손실을 감수하고 디코딩
    """
    warnings: list[str] = []

    # 1) UTF-8
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(encoding), warnings
        except UnicodeDecodeError:
            continue

    # 2) 한국어 레거시 인코딩 (한글이 실제로 나오는지 확인)
    for encoding in _KOREAN_ENCODINGS:
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _has_hangul(text):
            warnings.append(f"{encoding} 로 디코딩했습니다. (한글 확인됨)")
            return text, warnings

    # 3) charset-normalizer 자동 감지
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best is not None:
            warnings.append(f"인코딩을 {best.encoding}(으)로 추정해 디코딩했습니다.")
            return str(best), warnings
    except ImportError:
        warnings.append("charset-normalizer 미설치로 인코딩 자동 감지를 건너뛰었습니다.")

    # 4) 마지막 수단
    for encoding in (*_KOREAN_ENCODINGS, "latin-1"):
        try:
            text = data.decode(encoding)
            warnings.append(f"{encoding} 로 디코딩했습니다.")
            return text, warnings
        except UnicodeDecodeError:
            continue

    warnings.append("인코딩을 확정하지 못해 일부 문자가 손실되었을 수 있습니다.")
    return data.decode("utf-8", errors="replace"), warnings


def _extract_plain_text(data: bytes) -> ExtractedText:
    text, warnings = _decode_bytes(data)
    return ExtractedText(segments=[Segment(1, "document", text)], warnings=warnings)


def _extract_rtf(data: bytes) -> ExtractedText:
    from striprtf.striprtf import rtf_to_text

    raw, warnings = _decode_bytes(data)
    text = rtf_to_text(raw, errors="ignore").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    return ExtractedText(segments=[Segment(1, "document", text)], warnings=warnings)
