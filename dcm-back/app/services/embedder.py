"""
③ 임베딩 - bge-m3.

문장을 1024차원 dense 벡터로 바꿉니다. 학습/파인튜닝하지 않고 그대로 씁니다.
sentence-transformers 로 로드하며, 모델 가중치는 첫 요청 때
HuggingFace 에서 자동으로 내려받습니다. (약 2.2GB)

- 로드는 lazy singleton 입니다. (기동 시가 아니라 첫 요청 때 1회)
- 반환 벡터는 L2 정규화되어 있으므로 내적 = 코사인 유사도 입니다.
- ④ 후보 추림과 ②→③ 요약(TextRank)이 이 모듈 하나를 공유합니다.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from app.config import settings
from app.exceptions import ModelUnavailableError
from app.logging_config import get_logger
from app.services import runtime

logger = get_logger(__name__)

_model = None
_lock = threading.Lock()
_load_error: str | None = None


def is_installed() -> bool:
    """sentence-transformers 설치 여부. (가중치 다운로드 여부와는 별개)"""
    try:
        import sentence_transformers  # noqa: F401
    except Exception:
        return False
    return True


def is_loaded() -> bool:
    return _model is not None


def load_error() -> str | None:
    return _load_error


def get_model():
    """bge-m3 모델 인스턴스를 돌려줍니다. (없으면 로드)"""
    global _model, _load_error

    if _model is not None:
        return _model

    with _lock:
        if _model is not None:
            return _model

        if not is_installed():
            raise ModelUnavailableError(
                "임베딩 모델을 사용할 수 없습니다. sentence-transformers 가 설치되어 있지 않습니다.",
                detail="python -m pip install -r requirements-model.txt (README.md 의 1-4 참고)",
            )

        from sentence_transformers import SentenceTransformer

        device = runtime.resolve_device()
        logger.info(
            "bge-m3 로드 시작 | model=%s | device=%s (첫 실행이면 가중치를 내려받습니다)",
            settings.EMBED_MODEL, device,
        )
        started = time.perf_counter()
        try:
            _model = SentenceTransformer(
                settings.EMBED_MODEL,
                device=device,
                cache_folder=runtime.model_cache_dir(),
            )
        except Exception as exc:  # 다운로드 실패 / 경로 오류 등
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("bge-m3 로드 실패 | model=%s", settings.EMBED_MODEL)
            raise ModelUnavailableError(
                "임베딩 모델(bge-m3) 로드에 실패했습니다.",
                detail=_load_error if settings.DEBUG else None,
            ) from exc

        _model.max_seq_length = min(
            settings.EMBED_MAX_SEQ_LENGTH, getattr(_model, "max_seq_length", 8192) or 8192
        )
        elapsed = time.perf_counter() - started
        logger.info(
            "bge-m3 로드 완료 | dim=%d | max_seq=%d | %.1fs",
            _model.get_sentence_embedding_dimension(), _model.max_seq_length, elapsed,
        )
        _load_error = None
        return _model


def unload() -> None:
    """
    bge-m3 를 메모리(GPU)에서 내립니다. 다음 호출 때 다시 로드됩니다.

    학습 노트북에서 ④ 후보를 다 만든 뒤 Gemma 학습에 GPU 메모리를 몰아주려고 씁니다.
    KeyBERT 가 같은 인스턴스를 들고 있으므로 함께 비웁니다.
    """
    global _model
    with _lock:
        _model = None
    from app.services import keyphrase

    keyphrase.reset_keybert()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("bge-m3 언로드 완료")


def dimension() -> int:
    """임베딩 차원 수. (bge-m3 = 1024)"""
    return int(get_model().get_sentence_embedding_dimension())


def encode(texts: list[str]) -> np.ndarray:
    """문장 목록을 (N, 1024) 벡터 배열로 바꿉니다. L2 정규화된 값입니다."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    model = get_model()
    vectors = model.encode(
        texts,
        batch_size=settings.EMBED_BATCH_SIZE,
        normalize_embeddings=settings.EMBED_NORMALIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def encode_one(text: str) -> np.ndarray:
    """문장 1개를 (1024,) 벡터로 바꿉니다."""
    return encode([text])[0]


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    코사인 유사도.

    encode() 결과는 이미 정규화되어 있으므로 사실상 내적이지만,
    정규화를 끈 경우에도 맞게 동작하도록 분모를 직접 나눕니다.
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    b_norm[b_norm == 0] = 1.0
    return (a / a_norm) @ (b / b_norm).T


def status() -> dict[str, object]:
    """/health, /analyze/status 용 상태 요약."""
    return {
        "installed": is_installed(),
        "loaded": is_loaded(),
        "model": settings.EMBED_MODEL,
        "device": runtime.resolve_device(),
        "load_error": _load_error,
    }
