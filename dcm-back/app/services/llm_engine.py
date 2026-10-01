"""
⑤ Qwen3.5-4B (+ QLoRA 어댑터) 판정 엔진.

파이프라인에서 유일하게 학습·파인튜닝 대상인 모델입니다.
한 요청에서 다음 세 가지를 순서대로 처리합니다.

  1) 의도 판정      - "의도: " 뒤에 올 번호 토큰(1~6)의 확률을 forward 1회로 계산
                      (1~5 = 문의/접수/조회/수정/삭제, 6 = 해당없음 -> 게이트 반려)
  2) 카테고리 확정  - ④ 가 추린 후보 3개 중 번호 토큰(1~3) 확률을 forward 1회로 계산
  3) 도구 호출 JSON - generate 로 JSON 을 생성 (의도가 '문의'면 안내 문장을 생성)

1)2) 는 자유 생성이 아니라 "번호 토큰 1회 계산"입니다.
후보 번호에 해당하는 로짓만 뽑아 softmax 하므로
  - 형식이 깨질 수 없고(항상 유효한 번호)
  - 토큰 1개만 계산해 빠르며
  - 확신 점수(0~1)를 그대로 얻습니다.

로드 방식
  - CUDA + bitsandbytes 가 있으면 4bit NF4 (QLoRA 와 같은 양자화 설정)
    * 연산 dtype 은 GPU 에 맞춰 고릅니다. T4(Turing)는 bfloat16 이 없으므로 float16.
    * 4B 모델도 4bit 면 약 2.5GB 라 Colab T4(16GB)에 bge-m3 와 함께 올라갑니다.
  - 그 외에는 CPU float32 로 올립니다. 4B 는 CPU 에서 매우 느리니 테스트용으로만 쓰세요.
  - .env 의 LLM_ADAPTER_PATH 에 경로를 넣으면 PEFT 로 LoRA 어댑터를 얹습니다.

KV 캐시(CAG)
  고정 프리픽스(역할 지시문·도구 목록·카테고리 정의·의도 정의)를 한 번만 prefill 해 두고
  요청마다 그 캐시 복사본에 요청 구간만 이어 계산합니다. (.env 의 LLM_KV_CACHE)
  처음 만들 때 캐시 사용/미사용 결과가 같은지 검증하고, 문제가 있으면 스스로 꺼집니다.

사고(thinking) 모드
  Qwen3.5 는 기본이 사고 모드라 답 앞에 <think>...</think> 를 먼저 씁니다.
  의도/카테고리 판정은 "다음 토큰 1개"만 보므로 사고 모드가 켜져 있으면
  그 1개가 <think> 가 되어 판정이 깨집니다. 그래서 채팅 템플릿에
  enable_thinking=False 를 넘겨 끕니다. (.env 의 LLM_ENABLE_THINKING)
"""

from __future__ import annotations

import copy
import json
import re
import threading
import time
from dataclasses import dataclass, field

from app.config import BASE_DIR, settings
from app.exceptions import ModelUnavailableError
from app.logging_config import get_logger
from app.services import prompts, runtime
from app.services.prompts import Intent

logger = get_logger(__name__)

_bundle = None            # (tokenizer, model)
_adapter_info: dict | None = None   # 어댑터 run_info 확인 결과 (status 노출용)
_lock = threading.Lock()
_load_error: str | None = None


# =============================================================================
# 결과 자료구조
# =============================================================================
@dataclass
class Choice:
    """번호 토큰 1회 계산 결과."""

    number: int                  # 선택된 번호 (1-based)
    label: str                   # 선택된 항목 이름 ("접수", "도로" ...)
    score: float                 # softmax 확률 (그림의 0.93 / 0.86)
    scores: dict[str, float] = field(default_factory=dict)  # 후보 전체 점수


@dataclass
class ToolCall:
    called: bool = False
    name: str | None = None
    arguments: dict | None = None
    raw: str = ""                # 모델이 실제로 뱉은 원문 (DEBUG 확인용)
    parse_error: str | None = None


@dataclass
class LlmResult:
    intent: Intent
    intent_choice: Choice
    category_name: str
    category_choice: Choice
    tool_call: ToolCall
    answer: str = ""             # '문의' 의도일 때의 즉답
    warnings: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)


# =============================================================================
# 모델 로드
# =============================================================================
def is_installed() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def is_loaded() -> bool:
    return _bundle is not None


def load_error() -> str | None:
    return _load_error


def load_base_model():
    """
    베이스 모델(어댑터 없음)을 추론 때와 똑같은 설정으로 올려 (tokenizer, model) 을 돌려줍니다.

    서버 싱글턴(_bundle)에는 등록하지 않습니다. 학습 노트북이 이 함수로 베이스를 올린 뒤
    LoRA 를 붙여 학습하므로, 학습과 추론의 양자화·dtype 설정이 항상 같아집니다.
    """
    if not is_installed():
        raise ModelUnavailableError(
            "판정 모델을 사용할 수 없습니다. torch / transformers 가 설치되어 있지 않습니다.",
            detail="python -m pip install -r requirements-model.txt (README.md 의 1-4 참고)",
        )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = runtime.resolve_device()
    four_bit = runtime.use_4bit()
    logger.info(
        "Qwen 로드 시작 | model=%s | device=%s | 4bit=%s | compute_dtype=%s | thinking=%s "
        "(첫 실행이면 가중치를 내려받습니다)",
        settings.LLM_MODEL, device, four_bit,
        str(runtime.compute_dtype() if four_bit else runtime.torch_dtype()).replace("torch.", ""),
        settings.LLM_ENABLE_THINKING,
    )

    trust = bool(getattr(settings, "LLM_TRUST_REMOTE_CODE", False))
    tokenizer = AutoTokenizer.from_pretrained(
        settings.LLM_MODEL,
        cache_dir=runtime.model_cache_dir(),
        trust_remote_code=trust,
    )

    kwargs: dict = {"cache_dir": runtime.model_cache_dir()}
    if trust:
        kwargs["trust_remote_code"] = True
    if four_bit:
        from transformers import BitsAndBytesConfig

        # QLoRA 와 동일한 양자화 설정 (NF4 + double quant)
        # 연산 dtype 은 GPU 에 맞춰 고릅니다. (T4 -> float16, Ampere 이상 -> bfloat16)
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=runtime.compute_dtype(),
        )
        kwargs["device_map"] = {"": 0}
    else:
        kwargs["torch_dtype"] = runtime.torch_dtype()

    model = _load_weights(AutoModelForCausalLM, kwargs)
    if not four_bit:
        model = model.to(device)
    return tokenizer, model


def get_model():
    """(tokenizer, model) 튜플을 돌려줍니다. 첫 호출 때 로드합니다."""
    global _bundle, _load_error, _adapter_info

    if _bundle is not None:
        return _bundle

    with _lock:
        if _bundle is not None:
            return _bundle

        started = time.perf_counter()
        try:
            tokenizer, model = load_base_model()

            # --- QLoRA 어댑터 (있을 때만) ---
            adapter = resolve_adapter_path(settings.LLM_ADAPTER_PATH)
            if adapter:
                from peft import PeftModel

                logger.info("QLoRA 어댑터 적용 | path=%s", adapter)
                model = PeftModel.from_pretrained(model, adapter)
                _adapter_info = check_adapter(adapter)

            model.eval()

        except ModelUnavailableError:
            raise
        except Exception as exc:
            _load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Qwen 로드 실패 | model=%s", settings.LLM_MODEL)
            raise ModelUnavailableError(
                "판정 모델(Qwen) 로드에 실패했습니다.",
                detail=_load_error if settings.DEBUG else None,
            ) from exc

        logger.info("Qwen 로드 완료 | %.1fs", time.perf_counter() - started)
        _bundle = (tokenizer, model)
        _load_error = None
        return _bundle


def resolve_adapter_path(value: str | None) -> str:
    """
    LLM_ADAPTER_PATH 를 실제로 읽을 경로로 바꿉니다.

    상대 경로(예: adapters/v1_0929)는 서버를 어느 폴더에서 띄우든 backend 폴더 기준으로 찾습니다.
    그 위치에 폴더가 없으면 HF Hub repo id 로 보고 그대로 둡니다.
    """
    from pathlib import Path

    value = (value or "").strip()
    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    candidate = BASE_DIR / path
    return str(candidate) if candidate.exists() else value


def check_adapter(adapter_path: str) -> dict[str, object]:
    """
    어댑터 폴더의 run_info.json(학습 노트북이 남김)을 읽어 지금 설정과 맞는지 확인합니다.

    - base_model        : 어댑터는 학습한 베이스 모델에서만 제대로 동작합니다.
    - prompt_fingerprint: 학습 뒤 prompts.py 를 고쳤으면 입력 분포가 달라져 효과가 줄어듭니다.
    - enable_thinking   : 채팅 템플릿이 달라지므로 학습 때와 같아야 합니다.

    맞지 않아도 로드는 계속하고 경고만 남깁니다. (HF repo id 처럼 로컬 폴더가 아니면 건너뜀)
    """
    from pathlib import Path

    info: dict[str, object] = {"path": adapter_path, "run_info": None, "mismatches": []}
    path = Path(adapter_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    run_info_file = path / "run_info.json"
    if not run_info_file.is_file():
        info["note"] = "run_info.json 이 없어 학습 조건을 확인하지 않았습니다."
        return info
    try:
        run_info = json.loads(run_info_file.read_text(encoding="utf-8"))
    except Exception as exc:  # 깨진 파일이어도 서버는 계속 뜨게
        info["note"] = f"run_info.json 을 읽지 못했습니다: {exc}"
        return info

    info["run_info"] = {
        k: run_info.get(k) for k in ("run_name", "created_at", "base_model", "prompt_fingerprint")
    }
    checks = {
        "base_model": (run_info.get("base_model"), settings.LLM_MODEL),
        "prompt_fingerprint": (run_info.get("prompt_fingerprint"), prompts.prompt_fingerprint()),
        "enable_thinking": (run_info.get("enable_thinking"), settings.LLM_ENABLE_THINKING),
    }
    for key, (trained, current) in checks.items():
        if trained is not None and trained != current:
            info["mismatches"].append({"item": key, "trained": trained, "current": current})
            logger.warning(
                "어댑터 학습 조건과 현재 설정이 다릅니다 | %s | 학습=%s | 현재=%s", key, trained, current
            )
    if not info["mismatches"]:
        logger.info("어댑터 학습 조건 일치 | run=%s", run_info.get("run_name"))
    return info


def use_model(tokenizer, model) -> None:
    """
    이미 메모리에 올린 (tokenizer, model) 을 판정 엔진에 끼워 넣습니다. (학습 노트북 평가용)

    학습 중인 PeftModel 을 그대로 넣으면 decide()/judge_*() 가 운영과 완전히 같은 코드로
    그 모델을 평가합니다. 가중치가 바뀌었으므로 KV 캐시는 비웁니다.
    """
    global _bundle, _load_error
    with _lock:
        _bundle = (tokenizer, model)
        _load_error = None
    reset_kv_cache()


def reset_kv_cache() -> None:
    """
    KV 캐시를 비웁니다. 다음 판정 때 다시 prefill(+검증)합니다.

    가중치가 바뀌면(학습 스텝 진행, 어댑터 켜기/끄기) 반드시 불러야 합니다.
    예전 가중치로 만든 캐시를 쓰면 판정이 조용히 틀어집니다.
    """
    global _kv_state, _kv_disabled
    with _kv_lock:
        _kv_state = None
        _kv_disabled = None
        _kv_stats["hits"] = 0
        _kv_stats["fallbacks"] = 0


def _load_weights(auto_cls, kwargs: dict):
    """
    가중치 로드. 두 가지 호환 문제를 여기서 흡수합니다.

    1) transformers 최신판은 torch_dtype 대신 dtype 을 받습니다. (TypeError 시 재시도)
    2) Qwen3.5 체크포인트는 비전 인코더가 붙은 멀티모달 구조입니다.
       설치된 transformers 가 이 config 를 AutoModelForCausalLM 에 연결해 두지 않았으면
       "Unrecognized configuration class" ValueError 가 나므로, 그때는
       AutoModelForImageTextToText 로 다시 올립니다. 텍스트만 넣어도 로짓·generate 가
       똑같이 동작하므로 ⑤ 의 판정 로직은 바뀌지 않습니다.
    """
    def _from_pretrained(cls):
        try:
            return cls.from_pretrained(settings.LLM_MODEL, **kwargs)
        except TypeError:
            if "torch_dtype" not in kwargs:
                raise
            kwargs["dtype"] = kwargs.pop("torch_dtype")
            return cls.from_pretrained(settings.LLM_MODEL, **kwargs)

    try:
        return _from_pretrained(auto_cls)
    except ValueError as exc:
        if "Unrecognized configuration class" not in str(exc):
            raise
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:
            raise exc from None
        logger.info(
            "AutoModelForCausalLM 이 %s 를 모릅니다 - AutoModelForImageTextToText 로 다시 로드합니다.",
            settings.LLM_MODEL,
        )
        return _from_pretrained(AutoModelForImageTextToText)


# =============================================================================
# 프롬프트 조립 + KV 캐시(CAG) 재사용
# =============================================================================
# 채팅 템플릿으로 조립한 입력은 항상 이 모양입니다.
#
#   [프리픽스 구간]  <system> 고정 프리픽스 </system> <user 머리말>     <- 요청마다 똑같음
#   [요청 구간]      단계별 사용자 프롬프트 + <assistant 머리말> + "의도: " 등
#
# 프리픽스 구간을 모델에 한 번만 통과시켜 past_key_values 를 만들어 두고(prefill),
# 요청마다 그 복사본에 요청 구간만 이어 붙여 계산합니다.
# 프리픽스 구간 토큰(약 1천~2천 개)을 매번 다시 계산하지 않으므로 ⑤ 가 빨라집니다.
#
# 캐시는 forward 때마다 그 자리에서 늘어나므로 요청마다 deepcopy 한 복사본을 씁니다.
# (Qwen3.5 는 일부 층이 선형 어텐션이라 늘어난 캐시를 잘라내(crop) 되돌릴 수 없습니다)
#
# 안전장치
#   - 처음 만들 때 같은 입력을 캐시 사용/미사용으로 한 번씩 계산해 결과가 같은지 검증합니다.
#     (LLM_KV_CACHE_VERIFY) 다르면 KV 캐시를 끄고 기존 방식으로 계속합니다.
#   - 사용 중 오류가 나도 그 자리에서 끄고 기존 방식(전체 입력 계산)으로 다시 계산합니다.
#   - 두 방식은 토큰 id 가 완전히 같은 입력을 씁니다. (프리픽스 id + 요청 id 를 이어 붙임)

_SENTINEL = "⁣<<USER_PROMPT>>⁣"

_kv_lock = threading.Lock()
_kv_state: dict | None = None       # {"prefix_text", "suffix_text", "prefix_ids", "cache", ...}
_kv_disabled: str | None = None     # 끈 이유 (None 이면 사용 가능)
_kv_stats = {"hits": 0, "fallbacks": 0}


def render_template(tokenizer, user_content: str) -> str:
    """채팅 템플릿 렌더링. (추론과 학습이 같은 함수를 씁니다)"""
    messages = [
        {"role": "system", "content": prompts.build_fixed_prefix()},
        {"role": "user", "content": user_content},
    ]
    kwargs: dict = {"tokenize": False, "add_generation_prompt": True}
    try:
        # Qwen3.5 는 기본이 사고 모드입니다. 끄면 템플릿이 빈 <think></think> 를 채워 넣어
        # 생성 첫 토큰이 곧바로 우리가 원하는 번호가 됩니다.
        return tokenizer.apply_chat_template(
            messages, enable_thinking=settings.LLM_ENABLE_THINKING, **kwargs
        )
    except TypeError:
        # enable_thinking 을 모르는 템플릿(Qwen2.5 등)은 그냥 넘어갑니다.
        return tokenizer.apply_chat_template(messages, **kwargs)


def template_parts(tokenizer) -> tuple[str, str]:
    """
    (프리픽스 구간 문자열, 사용자 프롬프트 뒤에 붙는 문자열).

    모델 입력은 항상  프리픽스 + 단계별 사용자 프롬프트 + 뒷부분 + assistant_prefix  입니다.
    학습 데이터도 이 함수로 조립하므로 학습·추론 입력이 한 글자도 다르지 않습니다.
    """
    rendered = render_template(tokenizer, _SENTINEL)
    idx = rendered.find(_SENTINEL)
    if idx < 0:
        raise RuntimeError("채팅 템플릿에서 사용자 프롬프트 위치를 찾지 못했습니다.")
    return rendered[:idx], rendered[idx + len(_SENTINEL):]


def _template_parts() -> tuple[str, str]:
    tokenizer, _ = get_model()
    return template_parts(tokenizer)


def _build_chat(user_prompt: str, assistant_prefix: str = "") -> str:
    """
    고정 프리픽스(system) + 단계별 사용자 프롬프트를 채팅 템플릿으로 조립한 문자열.

    assistant_prefix 를 주면 어시스턴트 발화가 그 문자열까지 이미 쓰인 상태가 되어,
    바로 다음 토큰이 우리가 원하는 번호가 됩니다. (그림의 '"의도:" -> "1"')
    """
    prefix_text, suffix_text = _template_parts()
    return prefix_text + user_prompt + suffix_text + assistant_prefix


def _encode(user_prompt: str, assistant_prefix: str = ""):
    """(prefix_ids, rest_ids) - 둘 다 (1, n) 텐서, 모델 디바이스 위."""
    import torch

    tokenizer, model = get_model()
    state = _kv_state
    if state is not None:
        prefix_ids, suffix_text = state["prefix_ids"], state["suffix_text"]
    else:
        prefix_text, suffix_text = _template_parts()
        prefix_ids = torch.tensor(
            [tokenizer.encode(prefix_text, add_special_tokens=False)], device=model.device
        )
    rest = tokenizer.encode(user_prompt + suffix_text + assistant_prefix, add_special_tokens=False)
    return prefix_ids, torch.tensor([rest], device=model.device)


def _disable_kv(reason: str) -> None:
    global _kv_state, _kv_disabled
    with _kv_lock:
        if _kv_disabled is None:
            logger.warning("KV 캐시를 끄고 전체 입력 계산으로 전환합니다 | %s", reason)
        _kv_disabled = reason
        _kv_state = None


def _prefix_cache():
    """
    고정 프리픽스의 KV 캐시 상태. 없으면 만듭니다. 쓸 수 없으면 None.

    최초 1회 prefill 합니다. (PRELOAD_MODELS=true 면 기동 때, 아니면 첫 요청 때)
    """
    global _kv_state, _kv_disabled
    if not settings.LLM_KV_CACHE or _kv_disabled is not None:
        return None
    if _kv_state is not None:
        return _kv_state

    with _kv_lock:
        if _kv_state is not None or _kv_disabled is not None:
            return _kv_state
        import torch

        tokenizer, model = get_model()
        try:
            started = time.perf_counter()
            prefix_text, suffix_text = _template_parts()
            prefix_ids = torch.tensor(
                [tokenizer.encode(prefix_text, add_special_tokens=False)], device=model.device
            )
            with torch.inference_mode():
                out = model(
                    input_ids=prefix_ids,
                    attention_mask=torch.ones_like(prefix_ids),
                    use_cache=True,
                )
            cache = out.past_key_values
            if cache is None:
                raise RuntimeError("모델이 past_key_values 를 돌려주지 않았습니다.")
            state = {
                "prefix_text": prefix_text,
                "suffix_text": suffix_text,
                "prefix_ids": prefix_ids,
                "prefix_tokens": int(prefix_ids.shape[1]),
                "cache": cache,
                "prefill_ms": int((time.perf_counter() - started) * 1000),
                "verified": None,
            }
            logger.info(
                "KV 캐시 prefill 완료 | 프리픽스 %d토큰 | %dms",
                state["prefix_tokens"], state["prefill_ms"],
            )
        except Exception as exc:
            _kv_state = None
            _kv_disabled = f"prefill 실패 - {type(exc).__name__}: {exc}"
            logger.warning("KV 캐시 prefill 실패 - 전체 입력 계산으로 동작합니다 | %s", exc)
            return None

        _kv_state = state

    if settings.LLM_KV_CACHE_VERIFY:
        _verify_prefix_cache()
    return _kv_state


def _fresh_cache():
    """요청 1건이 쓸 캐시 복사본. (원본은 계속 재사용해야 하므로 절대 직접 넘기지 않음)"""
    state = _prefix_cache()
    if state is None:
        return None
    return copy.deepcopy(state["cache"])


def _forward_last_logits(prefix_ids, rest_ids, use_kv: bool):
    """마지막 위치의 로짓 (vocab,). use_kv 면 프리픽스 캐시 복사본 위에 요청 구간만 계산."""
    import torch

    _, model = get_model()
    total = prefix_ids.shape[1] + rest_ids.shape[1]
    with torch.inference_mode():
        if use_kv:
            cache = _fresh_cache()
            if cache is None:
                raise RuntimeError("KV 캐시가 준비되지 않았습니다.")
            out = model(
                input_ids=rest_ids,
                attention_mask=torch.ones((1, total), dtype=torch.long, device=rest_ids.device),
                past_key_values=cache,
                use_cache=True,
            )
        else:
            ids = torch.cat([prefix_ids, rest_ids], dim=1)
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    return out.logits[0, -1, :].float()


def _verify_prefix_cache() -> None:
    """같은 입력을 캐시 사용/미사용으로 계산해 번호 판정이 같은지 확인합니다."""
    import torch

    state = _kv_state
    if state is None:
        return
    try:
        probe = prompts.build_intent_prompt("우리 동네 가로등이 꺼졌어요. 밤에 너무 어두워요.")
        prefix_ids, rest_ids = _encode(probe, INTENT_ASSISTANT_PREFIX)
        ids = _number_token_ids(len(prompts.INTENTS))
        p_kv = torch.softmax(_forward_last_logits(prefix_ids, rest_ids, True)[ids], -1)
        p_full = torch.softmax(_forward_last_logits(prefix_ids, rest_ids, False)[ids], -1)
        diff = float((p_kv - p_full).abs().max())
        same = int(p_kv.argmax()) == int(p_full.argmax())
    except Exception as exc:
        _disable_kv(f"검증 실패 - {type(exc).__name__}: {exc}")
        return

    state["verified"] = {"same_choice": same, "max_prob_diff": round(diff, 5)}
    if not same or diff > 0.05:
        _disable_kv(f"검증 불일치 - 같은 번호={same}, 확률 최대 차이={diff:.4f}")
    else:
        logger.info("KV 캐시 검증 통과 | 같은 번호=%s | 확률 최대 차이=%.5f", same, diff)


def warmup() -> None:
    """모델 로드 + KV 캐시 prefill/검증. (PRELOAD_MODELS=true 일 때 기동 시 호출)"""
    get_model()
    _prefix_cache()


def kv_status() -> dict[str, object]:
    state = _kv_state
    return {
        "enabled": settings.LLM_KV_CACHE,
        "ready": state is not None,
        "prefix_tokens": state["prefix_tokens"] if state else None,
        "prefill_ms": state["prefill_ms"] if state else None,
        "verified": state["verified"] if state else None,
        "disabled_reason": _kv_disabled,
        "hits": _kv_stats["hits"],
        "fallbacks": _kv_stats["fallbacks"],
    }


def number_token_ids(tokenizer, count: int) -> list[int]:
    """'1' ~ 'count' 의 첫 토큰 id 목록. (학습 정답 토큰도 이 함수로 만듭니다)"""
    ids: list[int] = []
    for n in range(1, count + 1):
        encoded = tokenizer.encode(str(n), add_special_tokens=False)
        if not encoded:
            raise ModelUnavailableError(
                "토크나이저에서 번호 토큰을 찾지 못했습니다.", detail=f"number={n}"
            )
        ids.append(encoded[0])
    if len(set(ids)) != len(ids):
        # 숫자를 한 글자씩 토큰으로 나누지 않는 토크나이저 - 번호 판정 방식이 성립하지 않습니다.
        raise ModelUnavailableError(
            "번호 토큰이 서로 겹칩니다. 이 모델의 토크나이저로는 번호 판정을 할 수 없습니다.",
            detail=f"ids={ids}",
        )
    return ids


_TURN_END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "[|endofturn|]", "<|endofturn|>")


def turn_end_token_id(tokenizer) -> int:
    """
    어시스턴트 발화를 닫는 토큰. 학습 정답 끝에 붙이고, 생성도 여기서 멈춥니다.
    Qwen <|im_end|> · Llama 3 <|eot_id|> · Gemma <end_of_turn> · Phi <|end|> · EXAONE [|endofturn|]
    목록에 없는 모델은 eos 토큰을 씁니다.
    """
    unk = getattr(tokenizer, "unk_token_id", None)
    for name in _TURN_END_TOKENS:
        tid = tokenizer.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0 and tid != unk:
            return tid
    if tokenizer.eos_token_id is None:
        raise RuntimeError("발화 끝 토큰(<|im_end|> 등 / eos)을 찾지 못했습니다.")
    return tokenizer.eos_token_id


def _number_token_ids(count: int) -> list[int]:
    tokenizer, _ = get_model()
    return number_token_ids(tokenizer, count)


def _choose_number(prompt: str, assistant_prefix: str, labels: list[str]) -> Choice:
    """
    번호 토큰 1회 계산.

    후보 번호에 해당하는 로짓만 골라 softmax 하므로 항상 유효한 번호가 나옵니다.
    KV 캐시가 준비돼 있으면 프리픽스는 건너뛰고 요청 구간만 계산합니다.
    """
    import torch

    if settings.LLM_ENABLE_THINKING:
        # 사고 모드에서는 다음 토큰이 <think> 가 되어 번호 확률이 의미를 잃습니다.
        logger.warning(
            "LLM_ENABLE_THINKING=true 상태에서 번호 토큰 판정을 수행합니다. "
            ".env 에서 false 로 두는 것을 권장합니다."
        )

    _prefix_cache()   # 준비 안 됐으면 여기서 prefill (끈 상태면 아무것도 안 함)
    prefix_ids, rest_ids = _encode(prompt, assistant_prefix)
    candidate_ids = _number_token_ids(len(labels))

    last_logits = None
    if _kv_state is not None:
        try:
            last_logits = _forward_last_logits(prefix_ids, rest_ids, use_kv=True)
            _kv_stats["hits"] += 1
        except Exception as exc:
            _kv_stats["fallbacks"] += 1
            _disable_kv(f"판정 중 오류 - {type(exc).__name__}: {exc}")
    if last_logits is None:
        last_logits = _forward_last_logits(prefix_ids, rest_ids, use_kv=False)

    probs = torch.softmax(last_logits[candidate_ids], dim=-1).tolist()
    best = int(max(range(len(probs)), key=lambda i: probs[i]))
    return Choice(
        number=best + 1,
        label=labels[best],
        score=round(float(probs[best]), 4),
        scores={labels[i]: round(float(p), 4) for i, p in enumerate(probs)},
    )


def _generate(prompt: str, max_new_tokens: int) -> str:
    """일반 생성. (도구 호출 JSON / 문의 즉답) KV 캐시가 있으면 프리픽스 계산을 건너뜁니다."""
    import torch

    tokenizer, model = get_model()

    _prefix_cache()
    prefix_ids, rest_ids = _encode(prompt)
    ids = torch.cat([prefix_ids, rest_ids], dim=1)

    greedy = settings.LLM_TEMPERATURE <= 0
    gen_kwargs: dict = {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    }
    stops = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    stops = [stops] if isinstance(stops, int) else list(stops or [])
    end_id = turn_end_token_id(tokenizer)
    if end_id not in stops:
        gen_kwargs["eos_token_id"] = stops + [end_id]   # 학습 때 정답 끝 토큰에서도 멈추게
    if not greedy:
        # greedy 일 때 temperature/top_p 를 넘기면 transformers 가 경고를 냅니다.
        gen_kwargs["temperature"] = settings.LLM_TEMPERATURE
        gen_kwargs["top_p"] = settings.LLM_TOP_P

    generated = None
    if _kv_state is not None:
        try:
            # 전체 input_ids 와 함께 프리픽스 캐시 복사본을 넘기면
            # generate 가 캐시에 없는 뒷부분(요청 구간)만 계산합니다.
            with torch.inference_mode():
                generated = model.generate(past_key_values=_fresh_cache(), **gen_kwargs)
            _kv_stats["hits"] += 1
        except Exception as exc:
            _kv_stats["fallbacks"] += 1
            _disable_kv(f"생성 중 오류 - {type(exc).__name__}: {exc}")
            generated = None
    if generated is None:
        with torch.inference_mode():
            generated = model.generate(**gen_kwargs)

    new_tokens = generated[0][ids.shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return strip_thinking(decoded).strip()


# <think> ... </think> 사고 블록. LLM_ENABLE_THINKING=true 일 때만 나옵니다.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"^.*?</think>", re.DOTALL)

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def strip_thinking(text: str) -> str:
    """
    모델 출력에서 사고 블록을 걷어냅니다.

    사고 모드를 꺼도(LLM_ENABLE_THINKING=false) 모델이 습관적으로 <think> 를
    쓰는 경우가 있어 방어적으로 항상 제거합니다.
    닫는 태그만 남고 여는 태그가 잘린 경우도 처리합니다.
    """
    if not text:
        return ""
    cleaned = _THINK_BLOCK.sub("", text)
    if "</think>" in cleaned:
        cleaned = _OPEN_THINK.sub("", cleaned, count=1)
    return cleaned.strip()


def _parse_tool_json(raw: str, expected_tool: str | None) -> ToolCall:
    """모델 출력에서 JSON 객체를 뽑아 파싱합니다."""
    call = ToolCall(raw=raw)

    cleaned = raw.strip()
    # ```json ... ``` 코드블록을 벗깁니다.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    match = _JSON_BLOCK.search(cleaned)
    if not match:
        call.parse_error = "출력에서 JSON 객체를 찾지 못했습니다."
        return call

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        call.parse_error = f"JSON 파싱 실패: {exc}"
        return call

    if not isinstance(data, dict):
        call.parse_error = "JSON 최상위가 객체가 아닙니다."
        return call

    name = data.get("name") or expected_tool
    arguments = data.get("arguments")
    if not isinstance(arguments, dict):
        # {"name":..., "category":..., "content":...} 형태로 뱉는 경우도 받아 줍니다.
        arguments = {k: v for k, v in data.items() if k != "name"}

    call.called = True
    call.name = str(name) if name else None
    call.arguments = arguments
    return call


# =============================================================================
# 공개 API - 단계별 (decide() 와 학습 노트북 평가가 같은 함수를 씁니다)
# =============================================================================
def prepare_text(user_text: str) -> str:
    """⑤ 에 넣기 전 원문 정리 (길면 앞부분만). 학습 데이터도 같은 규칙으로 자릅니다."""
    return (user_text or "").strip()[: settings.LLM_MAX_INPUT_CHARS]


def judge_intent(text: str) -> tuple[Intent, Choice]:
    """1) 의도 판정 - 번호 토큰 1회 계산, 6지선다 (6 = 해당없음)."""
    choice = _choose_number(
        prompts.build_intent_prompt(text),
        assistant_prefix=INTENT_ASSISTANT_PREFIX,
        labels=[i.name for i in prompts.INTENTS],   # 6개 (해당없음 포함)
    )
    return prompts.intent_by_number(choice.number), choice


def judge_category(text: str, candidate_names: list[str], allow_none: bool = False) -> Choice:
    """
    2) 카테고리 확정 - ④ 가 추린 후보 중 번호 토큰 1회 계산.

    allow_none (조회·수정·삭제) 이면 마지막 번호가 '없음' 입니다. (Choice.label == prompts.CATEGORY_NONE)
    """
    return _choose_number(
        prompts.build_category_prompt(text, candidate_names, allow_none),
        assistant_prefix=CATEGORY_ASSISTANT_PREFIX,
        labels=prompts.category_labels(candidate_names, allow_none),
    )


def generate_tool_call(
    text: str, intent: Intent, category_name: str, warnings: list[str] | None = None
) -> ToolCall:
    """
    3) 도구 호출 JSON 생성 + 파싱 + 교정.

    교정 규칙 (모델 출력보다 코드가 우선)
      - 도구 이름은 의도에서 정해진 도구로 맞춥니다.
      - category 인자가 있는 도구는 방금 확정한 카테고리로 덮어씁니다. (없음이면 "")
    모델이 원래 뭐라고 썼는지는 tool_call.raw 에 그대로 남습니다.

    category_name 이 "" 또는 '없음' 이면 프롬프트에는 '없음' 으로 적습니다.
    """
    warnings = warnings if warnings is not None else []
    shown = category_name or prompts.CATEGORY_NONE
    confirmed = "" if shown == prompts.CATEGORY_NONE else shown
    raw = _generate(
        prompts.build_tool_prompt(text, intent, shown),
        settings.LLM_MAX_NEW_TOKENS_TOOL,
    )
    tool_call = _parse_tool_json(raw, intent.tool)
    if tool_call.parse_error:
        warnings.append(f"도구 호출 JSON 파싱 실패 - {tool_call.parse_error}")
    elif tool_call.name != intent.tool:
        warnings.append(
            f"모델이 예상과 다른 도구를 지목했습니다. "
            f"(예상={intent.tool}, 출력={tool_call.name}) 예상 도구로 교정했습니다."
        )
        tool_call.name = intent.tool
    if (
        not tool_call.parse_error
        and "category" in prompts.tool_param_names(intent.tool or "")
        and isinstance(tool_call.arguments, dict)
        and tool_call.arguments.get("category") != confirmed
    ):
        # category 는 방금 번호 토큰으로 확정한 값과 같아야 합니다.
        # 생성 단계에서 모델이 다른 이름(예: 후보 밖 카테고리)을 적으면 확정값으로 덮어씁니다.
        if "category" in tool_call.arguments:
            warnings.append(
                f"도구 인자 category 를 확정 카테고리로 교정했습니다. "
                f"(출력={tool_call.arguments.get('category')!r}, 확정={confirmed!r})"
            )
        tool_call.arguments["category"] = confirmed
    return tool_call


def generate_answer(text: str) -> str:
    """'문의' 즉답 생성 (도구 호출 없음)."""
    return _generate(prompts.build_answer_prompt(text), settings.LLM_MAX_NEW_TOKENS_ANSWER)


# 번호 판정 때 어시스턴트 발화 앞머리. 학습 데이터도 이 문자열을 그대로 씁니다.
INTENT_ASSISTANT_PREFIX = "의도: "
CATEGORY_ASSISTANT_PREFIX = "카테고리: "


def decide(user_text: str, candidate_names: list[str]) -> LlmResult:
    """
    ⑤ 전체 판정.

    user_text        : ② 에서 추출한 원문 (길면 앞부분만 사용)
    candidate_names  : ④ 가 추린 카테고리 후보 이름 3개
    """
    warnings: list[str] = []
    timings: dict[str, int] = {}

    text = prepare_text(user_text)
    if not candidate_names:
        candidate_names = ["기타"]
        warnings.append("후보가 비어 있어 '기타'로 대체했습니다.")

    # --- 1) 의도 판정 (번호 토큰 1회 계산, 6지선다) ---
    started = time.perf_counter()
    intent, intent_choice = judge_intent(text)
    timings["intent_ms"] = int((time.perf_counter() - started) * 1000)

    # --- 해당없음(게이트) : 카테고리 확정·도구 호출을 생략하고 바로 반환 ---
    # 5가지 의도 중 어디에도 해당하지 않는다고 Qwen 이 스스로 판단한 경우입니다.
    # forward 1회(의도 판정)만 쓰고 이후 단계를 건너뛰므로 반려가 오히려 더 빠릅니다.
    if intent.code == "out_of_scope":
        return LlmResult(
            intent=intent,
            intent_choice=intent_choice,
            category_name="",
            category_choice=Choice(number=0, label="", score=0.0, scores={}),
            tool_call=ToolCall(called=False, raw=""),
            answer="",
            warnings=warnings,
            timings_ms=timings,
        )

    # --- 2) 카테고리 확정 ---
    #   접수          : 후보 3개 중 선택
    #   조회·수정·삭제 : 후보 3개 + 없음 중 선택 (주제를 특정할 수 없으면 없음 -> category_name "")
    #   문의          : 카테고리를 쓰지 않으므로 판정하지 않음
    category_name = ""
    category_choice = Choice(number=0, label="", score=0.0, scores={})
    if prompts.category_needed(intent):
        started = time.perf_counter()
        category_choice = judge_category(text, candidate_names, prompts.category_allows_none(intent))
        timings["category_ms"] = int((time.perf_counter() - started) * 1000)
        if category_choice.label != prompts.CATEGORY_NONE:
            category_name = category_choice.label

    # --- 3) 도구 호출 JSON / 문의 즉답 ---
    started = time.perf_counter()
    answer = ""
    if intent.tool is None:
        # 그림의 '문의' 갈래 - 도구 호출 없음, DB 미사용
        answer = generate_answer(text)
        tool_call = ToolCall(called=False, raw=answer)
    else:
        tool_call = generate_tool_call(text, intent, category_name, warnings)
    timings["tool_ms"] = int((time.perf_counter() - started) * 1000)

    return LlmResult(
        intent=intent,
        intent_choice=intent_choice,
        category_name=category_name,
        category_choice=category_choice,
        tool_call=tool_call,
        answer=answer,
        warnings=warnings,
        timings_ms=timings,
    )


def status() -> dict[str, object]:
    return {
        "installed": is_installed(),
        "loaded": is_loaded(),
        "model": settings.LLM_MODEL,
        "adapter": settings.LLM_ADAPTER_PATH or None,
        "adapter_check": _adapter_info,
        "prompt_fingerprint": prompts.prompt_fingerprint(),
        "device": runtime.resolve_device(),
        "load_4bit": runtime.use_4bit(),
        "load_error": _load_error,
        "kv_cache": kv_status(),
    }
