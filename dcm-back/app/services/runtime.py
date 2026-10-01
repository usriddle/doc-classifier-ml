"""
모델 실행 환경(디바이스) 판단 유틸.

bge-m3(③)와 Qwen(⑤)이 공통으로 사용합니다.
.env 의 DEVICE 값이 "auto" 면 CUDA 가 보이면 GPU, 아니면 CPU 로 자동 선택합니다.
"""

from __future__ import annotations

from functools import lru_cache

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache
def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


@lru_cache
def resolve_device() -> str:
    """실제로 사용할 디바이스 문자열을 돌려줍니다. ("cuda" / "cpu" / "cuda:0" 등)"""
    want = (settings.DEVICE or "auto").strip().lower()

    if want not in ("", "auto"):
        # 사용자가 명시한 값은 그대로 존중합니다. (cpu / cuda / cuda:0 / mps)
        return want

    if not torch_available():
        return "cpu"

    import torch

    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache
def is_cuda() -> bool:
    return resolve_device().startswith("cuda")


@lru_cache
def use_4bit() -> bool:
    """
    QLoRA 방식(4bit NF4) 로드 여부.

    bitsandbytes 의 4bit 양자화는 CUDA 전용이라 CPU 에서는 무조건 꺼집니다.
    (CPU 일 때는 float32 / bfloat16 으로 그냥 올립니다 - 느리지만 동작은 합니다)
    """
    if not settings.LLM_LOAD_4BIT:
        return False
    if not is_cuda():
        logger.info("CUDA 가 없어 4bit 양자화를 건너뜁니다. (CPU 모드로 로드)")
        return False
    try:
        import bitsandbytes  # noqa: F401
    except Exception:
        logger.warning(
            "bitsandbytes 가 설치되어 있지 않아 4bit 양자화를 건너뜁니다. "
            "README.md 의 1-4. 모델 스택 설치를 참고하세요."
        )
        return False
    return True


@lru_cache
def supports_bf16() -> bool:
    """
    GPU 가 bfloat16 을 지원하는지.

    bfloat16 을 하드웨어로 지원하는 건 Ampere(sm_80, RTX30/A100) 이상입니다.
    Colab 무료 티어의 **T4 는 Turing(sm_75) 이라 네이티브 지원이 없으므로** float16 을 써야 합니다.
    """
    if not is_cuda() or not torch_available():
        return False
    import torch

    # torch.cuda.is_bf16_supported() 는 쓰지 않습니다.
    # PyTorch 최신판은 기본값이 including_emulation=True 라서, 하드웨어 지원이 없는
    # T4 에서도 "에뮬레이션은 된다"는 이유로 True 를 돌려줍니다.
    # 그 값을 믿으면 T4 에서 느린 에뮬레이션 bf16 로 돌게 됩니다.
    # 그래서 compute capability 로 직접 판단합니다. (8.0 이상 = Ampere 이후 = 네이티브 bf16)
    try:
        major, _minor = torch.cuda.get_device_capability(0)
        return major >= 8
    except Exception:
        return False


def torch_dtype():
    """CPU/GPU 에 맞는 기본 dtype 을 돌려줍니다. (T4 -> float16)"""
    if not torch_available():
        return None
    import torch

    if is_cuda():
        return torch.bfloat16 if supports_bf16() else torch.float16
    return torch.float32


def compute_dtype():
    """
    4bit 양자화(QLoRA)의 연산 dtype.

    .env 의 LLM_4BIT_COMPUTE_DTYPE 이 auto 면 GPU 지원 여부를 보고 고릅니다.
    T4 에서 bfloat16 을 강제하면 매우 느려지거나 오류가 나므로 auto 를 권장합니다.
    """
    if not torch_available():
        return None
    import torch

    want = (settings.LLM_4BIT_COMPUTE_DTYPE or "auto").strip().lower()
    if want == "bfloat16":
        if not supports_bf16():
            logger.warning(
                "이 GPU 는 bfloat16 을 지원하지 않습니다(T4 등). float16 으로 대체합니다."
            )
            return torch.float16
        return torch.bfloat16
    if want == "float16":
        return torch.float16
    return torch.bfloat16 if supports_bf16() else torch.float16


def model_cache_dir() -> str | None:
    """HuggingFace 모델 캐시 경로. 비어 있으면 HF 기본 경로를 씁니다."""
    value = (settings.MODEL_CACHE_DIR or "").strip()
    return value or None


def describe() -> dict[str, object]:
    """로그/헬스체크용 실행 환경 요약."""
    info: dict[str, object] = {
        "device": resolve_device(),
        "torch": torch_available(),
        "load_4bit": use_4bit(),
    }
    if torch_available():
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["supports_bf16"] = supports_bf16()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_memory_gb"] = round(props.total_memory / 1024**3, 1)
            info["compute_capability"] = f"{props.major}.{props.minor}"
            if use_4bit():
                info["compute_dtype"] = str(compute_dtype()).replace("torch.", "")
    return info
