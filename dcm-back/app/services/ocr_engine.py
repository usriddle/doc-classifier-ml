"""
PP-OCRv5 (PaddleOCR) 래퍼.

- 모델은 최초 1회만 로드해 재사용합니다. (첫 호출이 느린 이유)
- PaddleOCR 3.x 의 predict() 를 우선 사용하고, 2.x 의 ocr() 형태도 처리합니다.
- paddleocr / paddlepaddle 이 없으면 is_available() 이 False 를 반환하므로,
  호출하는 쪽에서 OCR 없이 진행할지 오류를 낼지 결정할 수 있습니다.
"""

from __future__ import annotations

import io
from functools import lru_cache

from app.config import settings
from app.exceptions import ExtractionError, OcrUnavailableError
from app.logging_config import get_logger

logger = get_logger(__name__)


_import_error: str | None = None
_available: bool | None = None   # 최초 1회만 확인하고 캐시합니다


def import_error() -> str | None:
    """is_available() 이 False 인 이유. (설치 문제 진단용)"""
    is_available()
    return _import_error


def is_available() -> bool:
    """
    paddleocr 와 paddlepaddle 을 실제로 임포트할 수 있는지 확인합니다.

    ImportError 만 잡으면 안 됩니다. paddleocr 는 paddlex -> modelscope -> torch
    순으로 딸린 패키지를 끌어오는데, 그 중 하나라도 DLL 로딩에 실패하면
    ImportError 가 아니라 OSError 가 납니다.
    (윈도우에서 자주 보는 "[WinError 127] shm.dll" 이 대표적입니다)

    그런 경우까지 여기서 잡아 False 로 내려야 서버가 기동은 되고
    OCR 만 비활성화됩니다. 기동 자체가 죽으면 안 됩니다.

    결과는 프로세스당 1회만 계산해 캐시합니다. (/health 를 부를 때마다 무거운 import 를
    다시 시도하거나 같은 경고를 반복해서 찍지 않도록) 설치 후에는 서버를 재시작하세요.
    """
    global _import_error, _available
    if _available is not None:
        return _available
    try:
        import paddle  # noqa: F401
        import paddleocr  # noqa: F401
    except Exception as exc:
        _import_error = f"{type(exc).__name__}: {exc}"
        _available = False
        logger.warning(
            "PP-OCRv5 를 사용할 수 없습니다 - OCR 없이 계속 진행합니다 | %s", _import_error
        )
        return False
    _import_error = None
    _available = True
    return True


@lru_cache
def _get_engine():
    """PaddleOCR 인스턴스 싱글턴."""
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        # ImportError 뿐 아니라 DLL 로딩 실패(OSError)도 여기서 처리합니다.
        raise OcrUnavailableError(
            "PP-OCRv5(PaddleOCR)를 불러오지 못했습니다.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    kwargs = {
        "text_detection_model_name": settings.OCR_DET_MODEL,
        "text_recognition_model_name": settings.OCR_REC_MODEL,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": settings.OCR_USE_TEXTLINE_ORIENTATION,
    }
    if settings.OCR_DEVICE:
        kwargs["device"] = settings.OCR_DEVICE

    logger.info(
        "PP-OCRv5 엔진 로드 | det=%s | rec=%s | device=%s "
        "(최초 실행 시 모델 가중치를 내려받느라 시간이 걸립니다)",
        settings.OCR_DET_MODEL, settings.OCR_REC_MODEL, settings.OCR_DEVICE or "auto",
    )

    try:
        return PaddleOCR(**kwargs)
    except TypeError as exc:
        # PaddleOCR 2.x 는 인자 이름이 다릅니다.
        logger.warning("PaddleOCR 3.x 인자를 적용하지 못해 구버전 방식으로 초기화합니다. | %s", exc)
        return PaddleOCR(use_angle_cls=True, lang="korean")
    except Exception as exc:
        raise OcrUnavailableError(
            "PP-OCRv5 엔진을 초기화하지 못했습니다.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc


def _to_array(image_bytes: bytes):
    """이미지 바이트를 OpenCV(BGR) 배열로 변환합니다."""
    import numpy as np

    try:
        import cv2

        array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if array is not None:
            return array
    except ImportError:
        pass

    # cv2 가 못 읽는 포맷(gif, heic 등)은 Pillow 로 우회
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.array(image.convert("RGB"))[:, :, ::-1].copy()  # RGB -> BGR
    except Exception as exc:
        raise ExtractionError(
            "이미지를 디코딩하지 못했습니다.",
            detail=(
                f"{type(exc).__name__}: {exc} / "
                "HEIC·HEIF 는 pillow-heif 설치가 필요합니다."
            ),
        ) from exc


def _parse(result, min_score: float) -> list[str]:
    """PaddleOCR 3.x / 2.x 응답을 읽는 순서대로 정렬한 텍스트 줄 목록으로 변환합니다."""
    # (중심 y, 왼쪽 x, 높이, 텍스트)
    boxes: list[tuple[float, float, float, str]] = []

    for item in result or []:
        payload = None

        # 3.x : OCRResult(dict 유사 객체). {"res": {...}} 형태로 감싸져 있기도 합니다.
        try:
            payload = item["res"] if "res" in item else item
        except Exception:
            payload = getattr(item, "json", None)
            if isinstance(payload, dict):
                payload = payload.get("res", payload)

        if isinstance(payload, dict) and "rec_texts" in payload:
            texts = payload.get("rec_texts") or []
            scores = payload.get("rec_scores") or [1.0] * len(texts)
            polys = payload.get("dt_polys") or [None] * len(texts)
            for text, score, poly in zip(texts, scores, polys):
                if not text or float(score) < min_score:
                    continue
                y, x, height = _poly_metrics(poly)
                boxes.append((y, x, height, str(text)))
            continue

        # 2.x : [[box, (text, score)], ...]
        if isinstance(item, (list, tuple)):
            for entry in item:
                try:
                    box, (text, score) = entry[0], entry[1]
                except Exception:
                    continue
                if not text or float(score) < min_score:
                    continue
                y, x, height = _poly_metrics(box)
                boxes.append((y, x, height, str(text)))

    return _group_rows(boxes)


def _group_rows(boxes: list[tuple[float, float, float, str]]) -> list[str]:
    """
    검출 박스를 실제 읽는 순서로 정렬합니다.

    y 좌표를 고정 구간으로 나누면 같은 줄인데도 구간 경계(예: y=34 와 y=35)를
    사이에 두고 순서가 뒤집힙니다. 그래서 글자 높이에 비례한 허용치로
    같은 줄을 묶은 뒤, 줄 안에서 x 순서대로 이어 붙입니다.
    """
    if not boxes:
        return []

    boxes.sort(key=lambda b: (b[0], b[1]))

    rows: list[tuple[float, float, list[tuple[float, str]]]] = []
    for y, x, height, text in boxes:
        if rows:
            ref_y, ref_height, items = rows[-1]
            tolerance = max(ref_height, height, 1.0) * 0.6
            if abs(y - ref_y) <= tolerance:
                items.append((x, text))
                continue
        rows.append((y, height, [(x, text)]))

    lines = []
    for _, _, items in rows:
        items.sort(key=lambda item: item[0])
        line = " ".join(text for _, text in items).strip()
        if line:
            lines.append(line)
    return lines


def _poly_metrics(poly) -> tuple[float, float, float]:
    """검출 박스의 (중심 y, 왼쪽 x, 높이). 좌표가 없으면 (0, 0, 0)."""
    try:
        points = list(poly)
        ys = [float(p[1]) for p in points]
        xs = [float(p[0]) for p in points]
        return sum(ys) / len(ys), min(xs), max(ys) - min(ys)
    except Exception:
        return 0.0, 0.0, 0.0


def run(image_bytes: bytes) -> str:
    """이미지 바이트에서 텍스트를 인식해 줄바꿈으로 이어 붙여 반환합니다."""
    engine = _get_engine()
    array = _to_array(image_bytes)

    try:
        if hasattr(engine, "predict"):
            result = engine.predict(array)
        else:  # PaddleOCR 2.x
            result = engine.ocr(array, cls=True)
    except Exception as exc:
        raise ExtractionError(
            "OCR 처리 중 오류가 발생했습니다.",
            detail=f"{type(exc).__name__}: {exc}",
        ) from exc

    return "\n".join(_parse(result, settings.OCR_MIN_SCORE)).strip()
