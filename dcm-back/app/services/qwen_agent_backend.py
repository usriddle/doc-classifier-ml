"""
Qwen-Agent(함수 호출 에이전트 프레임워크)로 "어느 민원인가요?" 에 대한 사용자의 답을 해석합니다. (chat_agent.py 가 부름)
프레임워크 이름만 Qwen-Agent 이고, 실제로 말하는 모델은 llm_engine 이 올린 판정 모델(Gemma 4 E4B)입니다.

이 모듈만 qwen-agent 패키지를 import 합니다. 설치되지 않은 환경(윈도우 추출 서버 등)에서는
chat_agent.py 가 이 모듈을 불러오지 못한 것을 감지하고 규칙 기반 해석으로 대신합니다.

구성
  SharedModelLLM  : Qwen-Agent 의 LLM 인터페이스(BaseFnCallModel). 모델을 새로 올리지 않고
                    llm_engine 이 이미 올려 둔 판정 모델(4bit + QLoRA 어댑터)을 그대로 씁니다.
                    (Qwen-Agent 기본 'transformers' 타입은 모델을 따로 한 벌 더 올려 GPU 메모리가 두 배가 됨)
  도구 3개        : select_complaint / cancel_selection / start_new_request
  SelectionAgent  : Qwen-Agent 의 Agent. LLM 을 한 번 부르고, 도구 호출이 있으면 그 도구를 실행하고 끝냅니다.
                    (도구 결과를 보고 LLM 을 또 부르지 않음 - 사용자에게 보낼 문장은 서버 코드가 만듦)

함수 호출 형식은 Qwen-Agent 의 'nous' 프롬프트(<tool_call>{"name": ..., "arguments": ...}</tool_call>)입니다.
모델이 자기 고유 형식으로 답해도 SharedModelLLM 이 nous 형식으로 바꿔 Qwen-Agent 에 넘깁니다.
  Gemma 4 : <|tool_call>call:이름{키:<|"|>값<|"|>}<tool_call|>   (llm_engine.gemma_tool_calls_to_json)
  Qwen3.5 : <function=이름><parameter=키>값</parameter></function> (xml_tool_calls_to_nous)
"""

from __future__ import annotations

import json
import re
import threading
from contextlib import nullcontext
from typing import Iterator

from qwen_agent.agent import Agent
from qwen_agent.llm.base import register_llm
from qwen_agent.llm.function_calling import BaseFnCallModel
from qwen_agent.llm.schema import ASSISTANT, FUNCTION, Message
from qwen_agent.tools.base import BaseTool

from app.config import settings
from app.logging_config import get_logger
from app.services import llm_engine

logger = get_logger(__name__)

_gen_lock = threading.Lock()   # 어댑터를 잠깐 끄는 동안(CHAT_AGENT_DISABLE_ADAPTER) 다른 에이전트 호출과 겹치지 않게


# =============================================================================
# LLM - 이미 올라와 있는 판정 모델 공유
# =============================================================================
_XML_CALL = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)>(.*?)</parameter>", re.S)


def xml_tool_calls_to_nous(text: str) -> str:
    """
    Gemma 고유의 도구 호출 형식을 Qwen-Agent nous 형식으로 바꿉니다. (예전 모델 호환용) (이미 nous 형식이면 그대로)

      <tool_call>\\n<function=select_complaint>\\n<parameter=position>\\n2\\n</parameter>\\n</function>\\n</tool_call>
      -> <tool_call>\\n{"name": "select_complaint", "arguments": {"position": 2}}\\n</tool_call>
    """
    if "<function=" not in text:
        return text

    def convert(match: re.Match) -> str:
        args: dict = {}
        for key, raw in _XML_PARAM.findall(match.group(2)):
            value = raw.strip()
            try:
                args[key] = json.loads(value)
            except ValueError:
                args[key] = value
        return json.dumps({"name": match.group(1), "arguments": args}, ensure_ascii=False)

    converted = _XML_CALL.sub(convert, text)
    if "<tool_call>" not in converted:
        # <tool_call> 감싸개 없이 <function=...> 만 낸 경우
        converted = re.sub(r"(\{\"name\".*?\}\})", r"<tool_call>\n\1\n</tool_call>", converted, flags=re.S)
    return converted


def _message_text(msg: Message) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return "".join(item.text or "" for item in content if getattr(item, "text", None))


_ROLE_MAP = {"function": "user", "tool": "user"}


def _alternate(messages: list[dict]) -> list[dict]:
    """
    채팅 템플릿에 맞게 메시지를 정리합니다.
    Gemma 계열 템플릿은 system 다음에 user / model 이 번갈아 나와야 하므로,
    같은 역할이 연달아 오면 한 발화로 합치고 도구 결과(function)는 user 쪽으로 붙입니다.
    """
    out: list[dict] = []
    for m in messages:
        role = _ROLE_MAP.get(m["role"], m["role"])
        if out and out[-1]["role"] == role and role != "system":
            out[-1]["content"] = (out[-1]["content"] + "\n\n" + m["content"]).strip()
        else:
            out.append({"role": role, "content": m["content"]})
    return out


@register_llm("minwon_shared")
class SharedModelLLM(BaseFnCallModel):
    """Qwen-Agent LLM 인터페이스 - llm_engine 의 (tokenizer, model) 을 그대로 씁니다."""

    def __init__(self, cfg: dict | None = None):
        cfg = dict(cfg or {})
        cfg.setdefault("model", settings.LLM_MODEL)
        cfg.setdefault("model_type", "minwon_shared")
        generate_cfg = dict(cfg.get("generate_cfg") or {})
        generate_cfg.setdefault("fncall_prompt_type", "nous")
        # 대화가 짧아 잘라낼 일이 없습니다. 0 이면 Qwen-Agent 의 대략적인 토큰 자르기를 건너뜁니다.
        generate_cfg.setdefault("max_input_tokens", 0)
        cfg["generate_cfg"] = generate_cfg
        super().__init__(cfg)
        self.last_prompt = ""      # 디버그용 - 실제로 모델에 들어간 프롬프트
        self.last_output = ""      # 디버그용 - 모델 원문 출력 (형식 변환 전)

    def render(self, messages: list[Message]) -> str:
        """Qwen-Agent 가 도구 설명까지 붙여 만든 메시지를 판정 모델의 채팅 템플릿 문자열로."""
        tokenizer, _ = llm_engine.get_model()
        plain = _alternate([{"role": m.role, "content": _message_text(m)} for m in messages])
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        try:
            return tokenizer.apply_chat_template(plain, enable_thinking=settings.LLM_ENABLE_THINKING, **kwargs)
        except TypeError:
            return tokenizer.apply_chat_template(plain, **kwargs)

    def generate_text(self, prompt: str) -> str:
        """프롬프트 -> 모델 출력 문자열. (테스트에서 이 함수만 바꿔 끼우면 GPU 없이 흐름을 확인할 수 있음)"""
        import torch

        tokenizer, model = llm_engine.get_model()
        # llm_engine._encode 와 같은 방식 (채팅 템플릿이 특수 토큰을 이미 넣었으므로 add_special_tokens=False)
        input_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], device=model.device)
        gen_kwargs = dict(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=settings.CHAT_AGENT_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        end_id = llm_engine.turn_end_token_id(tokenizer)
        stops = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        stops = [stops] if isinstance(stops, int) else list(stops or [])
        if end_id not in stops:
            gen_kwargs["eos_token_id"] = stops + [end_id]

        disable = settings.CHAT_AGENT_DISABLE_ADAPTER and hasattr(model, "disable_adapter")
        with _gen_lock, (model.disable_adapter() if disable else nullcontext()), torch.inference_mode():
            out = model.generate(**gen_kwargs)
        new_tokens = out[0, input_ids.shape[1]:]
        # 특수 토큰을 남긴 채 디코드 -> 사고 블록 제거 -> Gemma 도구 호출을 nous 형식으로 -> 남은 특수 토큰 제거
        raw = tokenizer.decode(new_tokens, skip_special_tokens=False)
        text = llm_engine.gemma_tool_calls_to_json(llm_engine.strip_thinking(raw), wrap="nous")
        return llm_engine.remove_special_tokens(tokenizer, text)

    def _chat_no_stream(self, messages: list[Message], generate_cfg: dict) -> list[Message]:
        prompt = self.render(messages)
        raw = self.generate_text(prompt)
        self.last_prompt, self.last_output = prompt, raw
        text = xml_tool_calls_to_nous(llm_engine.strip_thinking(raw)).strip()
        return [Message(ASSISTANT, text)]

    def _chat_stream(self, messages: list[Message], delta_stream: bool, generate_cfg: dict) -> Iterator[list[Message]]:
        # 스트리밍이 필요 없어 한 번에 만든 결과를 한 번 내보냅니다.
        yield self._chat_no_stream(messages, generate_cfg)


_llm: SharedModelLLM | None = None
_llm_lock = threading.Lock()


def get_llm() -> SharedModelLLM:
    global _llm
    with _llm_lock:
        if _llm is None:
            _llm = SharedModelLLM()
        return _llm


# =============================================================================
# 도구 - 호출되면 ctx 에 결정을 기록만 합니다. (실제 수정·취소 실행은 chat.py 가 함)
# =============================================================================
class _DecisionTool(BaseTool):
    def __init__(self, ctx: dict, choice_count: int):
        super().__init__()
        self.ctx = ctx
        self.choice_count = choice_count

    def _record(self, kind: str, args: dict) -> None:
        if "decision" not in self.ctx:      # 첫 번째 도구 호출만 인정
            self.ctx["decision"] = {"kind": kind, "args": args}


class SelectComplaintTool(_DecisionTool):
    name = "select_complaint"
    description = (
        "사용자가 후보 목록 중 하나를 골랐을 때 호출한다. "
        "'두 번째 거', '마지막 거', '가로등 거', '어제 넣은 거'처럼 목록에서의 순서나 내용으로 가리키면 position 에 "
        "목록 순번을 넣는다. '37번 민원'처럼 민원 번호를 직접 말하면 complaint_id 에 그 번호를 넣는다."
    )
    parameters = {
        "type": "object",
        "properties": {
            "position": {
                "type": "integer",
                "description": "후보 목록에서의 순번 (1부터). 예: '두 번째 거' -> 2, '마지막 거' -> 목록의 개수",
            },
            "complaint_id": {
                "type": "integer",
                "description": "사용자가 민원 번호를 직접 말했을 때 그 번호 (예: '37번 민원' -> 37)",
            },
        },
        "required": [],
    }

    def call(self, params, **kwargs) -> str:
        args = self._verify_json_format_args(params)
        self._record("select", args)
        return json.dumps({"ok": True, "recorded": args}, ensure_ascii=False)


class CancelSelectionTool(_DecisionTool):
    name = "cancel_selection"
    description = "사용자가 고르지 않겠다고 할 때 호출한다. 예: '아니에요', '됐어요', '그냥 둘게요', '안 할래요'."
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params, **kwargs) -> str:
        self._record("cancel", {})
        return json.dumps({"ok": True}, ensure_ascii=False)


class StartNewRequestTool(_DecisionTool):
    name = "start_new_request"
    description = (
        "사용자의 말이 후보 고르기와 상관없는 새 요청일 때 호출한다. "
        "예: 다른 민원을 새로 접수하거나, 다른 것을 조회·문의하는 말."
    )
    parameters = {"type": "object", "properties": {}, "required": []}

    def call(self, params, **kwargs) -> str:
        self._record("new_request", {})
        return json.dumps({"ok": True}, ensure_ascii=False)


# =============================================================================
# 에이전트 - LLM 1회 + 도구 1회
# =============================================================================
class SelectionAgent(Agent):
    """Qwen-Agent Agent. 함수 호출로 '어떤 결정을 할지' 만 고르고 바로 끝냅니다."""

    def _run(self, messages: list[Message], lang: str = "en", **kwargs) -> Iterator[list[Message]]:
        output = self._call_llm(
            messages=messages,
            functions=[tool.function for tool in self.function_map.values()],
            stream=False,
            extra_generate_cfg={"lang": lang},
        )
        response: list[Message] = list(output)
        for out in output:
            use_tool, tool_name, tool_args, _ = self._detect_tool(out)
            if use_tool:
                result = self._call_tool(tool_name, tool_args)
                function_id = (out.extra or {}).get("function_id", "1")
                response.append(Message(role=FUNCTION, name=tool_name, content=result, extra={"function_id": function_id}))
                break
        yield response


def run(system_message: str, conversation: list[dict], choice_count: int) -> dict:
    """
    에이전트를 한 번 돌립니다.

    conversation : [{"role": "user"|"assistant", "content": "..."}] - 마지막이 사용자의 이번 말
    반환 : {"decision": {"kind": ..., "args": ...} 또는 None, "text": 도구 없이 한 말, "raw": 모델 원문, "prompt": 프롬프트}
    """
    ctx: dict = {}
    tools = [
        SelectComplaintTool(ctx, choice_count),
        CancelSelectionTool(ctx, choice_count),
        StartNewRequestTool(ctx, choice_count),
    ]
    llm = get_llm()
    agent = SelectionAgent(function_list=tools, llm=llm, system_message=system_message)
    responses = agent.run_nonstream(messages=conversation, lang="en")
    text = ""
    for msg in responses:
        role = msg["role"] if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        has_call = (msg.get("function_call") if isinstance(msg, dict) else msg.function_call)
        if role == ASSISTANT and not has_call and isinstance(content, str):
            text += content
    return {
        "decision": ctx.get("decision"),
        "text": text.strip(),
        "raw": llm.last_output,
        "prompt": llm.last_prompt,
    }


# 예전 이름 호환
SharedQwenLLM = SharedModelLLM
