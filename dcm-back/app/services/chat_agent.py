"""
"어느 민원인가요?" 에 대한 사용자의 답 해석 - 무엇을 골랐는지 / 그만두는지 / 다른 요청인지.

  1순위 Qwen-Agent (qwen_agent_backend.py)
        후보 목록과 최근 대화를 보여 주고, 함수 호출로 결정하게 합니다.
          select_complaint(position | complaint_id) / cancel_selection() / start_new_request()
        "두 번째 거", "마지막 거"뿐 아니라 "가로등 거", "어제 넣은 거"처럼 내용으로 가리켜도 고를 수 있습니다.
  2순위 규칙 (이 파일의 resolve_by_rules)
        qwen-agent 가 없거나(CHAT_AGENT_ENABLED=false 포함) 에이전트가 실패했을 때.
        "N번째", "첫째", "마지막", "N번 민원", 숫자만, 거절 표현만 알아듣습니다.

에이전트가 무엇을 고르든 결과는 여기서 다시 검증합니다. (후보 목록에 있는 번호만 인정)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import settings
from app.logging_config import get_logger
from app.services.session_store import ChatMessage, Choice, PendingSelection

logger = get_logger(__name__)

SELECTED = "selected"           # 후보 하나를 고름
CANCELLED = "cancelled"         # 고르지 않고 그만둠
NEW_REQUEST = "new_request"     # 고르기와 상관없는 새 요청 -> 일반 파이프라인으로
UNRESOLVED = "unresolved"       # 알아듣지 못함 -> 다시 물음


@dataclass
class Resolution:
    kind: str
    method: str                         # agent | rules | button
    choice: Choice | None = None
    note: str = ""                      # 왜 그렇게 판단했는지 (디버그·로그용)
    agent: dict = field(default_factory=dict)   # 에이전트 원시 결과 (decision, raw, text) - 에이전트를 썼을 때만


# =============================================================================
# 에이전트 사용 가능 여부
# =============================================================================
_backend = None
_backend_error: str | None = None
_backend_checked = False


def _load_backend():
    """qwen_agent_backend 를 처음 쓸 때 한 번만 불러옵니다. 실패하면 사유를 기억하고 규칙으로 갑니다."""
    global _backend, _backend_error, _backend_checked
    if not _backend_checked:
        _backend_checked = True
        try:
            from app.services import qwen_agent_backend

            _backend = qwen_agent_backend
        except Exception as exc:   # qwen-agent 미설치 등
            _backend_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Qwen-Agent 를 불러오지 못해 규칙으로만 해석합니다 | %s", _backend_error)
    return _backend


def status() -> dict[str, object]:
    if settings.CHAT_AGENT_ENABLED:
        _load_backend()
    return {
        "agent_enabled": settings.CHAT_AGENT_ENABLED,
        "agent_available": _backend is not None,
        "agent_error": _backend_error,
        "disable_adapter_for_agent": settings.CHAT_AGENT_DISABLE_ADAPTER,
    }


# =============================================================================
# 에이전트 프롬프트
# =============================================================================
def choices_text(pending: PendingSelection) -> str:
    shown = pending.shown()
    lines = [f"{c.no}. {c.label()}" for c in shown]
    if pending.total > len(shown):
        lines.append(f"(그 밖에 {pending.total - len(shown)}건 더 있음 - 목록에 없는 민원은 민원 번호로만 고를 수 있음)")
    return "\n".join(lines)


def system_message(pending: PendingSelection) -> str:
    return (
        "당신은 지방자치단체 민원 창구의 대화 도우미다.\n"
        f"사용자가 민원 {pending.action} 요청을 했는데 대상 민원이 여러 건이라 아래 목록에서 고르도록 물었다.\n\n"
        f"[처음 요청]\n{pending.source_text or '(기록 없음)'}\n\n"
        f"[후보 목록 - {pending.action} 대상]\n{choices_text(pending)}\n\n"
        "사용자의 마지막 말을 읽고 반드시 도구 하나를 호출하라.\n"
        "- 목록 중 하나를 가리키면 select_complaint. 순서('두 번째', '마지막')나 내용('가로등 거', '어제 넣은 거')으로 "
        "가리키면 position 에 목록 순번을, '37번 민원'처럼 민원 번호를 말하면 complaint_id 에 그 번호를 넣는다.\n"
        "- 고르지 않겠다고 하면 cancel_selection.\n"
        "- 고르기와 상관없는 새 요청이면 start_new_request.\n"
        "- 어느 것인지 정말 알 수 없을 때만 도구 없이 짧게 다시 물어라. 목록에 없는 내용을 지어내지 마라."
    )


def _conversation(history: list[ChatMessage], user_text: str) -> list[dict]:
    """최근 대화 + 이번 말. Qwen-Agent 메시지 형식."""
    msgs = [{"role": m.role, "content": m.text} for m in history if m.role in ("user", "assistant") and m.text]
    msgs = msgs[-max(0, settings.CHAT_HISTORY_MESSAGES):] if settings.CHAT_HISTORY_MESSAGES else []
    # Gemma 채팅 템플릿은 첫 발화가 user 여야 자연스럽습니다. (system 다음에 assistant 로 시작하지 않게)
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _int(value) -> int | None:
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _resolve_by_agent(pending: PendingSelection, user_text: str, history: list[ChatMessage]) -> Resolution:
    backend = _load_backend()
    result = backend.run(system_message(pending), _conversation(history, user_text), len(pending.shown()))
    agent_info = {"decision": result.get("decision"), "raw": result.get("raw", ""), "text": result.get("text", "")}
    decision = result.get("decision")
    if not decision:
        return Resolution(UNRESOLVED, "agent", note="에이전트가 도구를 부르지 않음 (다시 물음)", agent=agent_info)

    kind, args = decision.get("kind"), decision.get("args") or {}
    if kind == "cancel":
        return Resolution(CANCELLED, "agent", note="에이전트: cancel_selection", agent=agent_info)
    if kind == "new_request":
        return Resolution(NEW_REQUEST, "agent", note="에이전트: start_new_request", agent=agent_info)

    # select - 후보 목록 안의 번호인지 다시 확인
    cid, pos = _int(args.get("complaint_id")), _int(args.get("position"))
    choice = pending.find(complaint_id=cid) if cid else None
    if choice is None and pos:
        choice = pending.find(no=pos)
    if choice is None and cid and not pos and cid <= len(pending.shown()):
        # "2번" 을 민원 번호로 잘못 넣은 경우 - 후보 중에 그 민원 번호가 없고 순번 범위 안이면 순번으로 봄
        choice = pending.find(no=cid)
    if choice is None:
        return Resolution(UNRESOLVED, "agent", note=f"에이전트가 고른 값 {args} 이 후보 목록에 없음", agent=agent_info)
    return Resolution(SELECTED, "agent", choice=choice, note=f"에이전트: select_complaint {args}", agent=agent_info)


# =============================================================================
# 규칙 (에이전트를 못 쓸 때)
# =============================================================================
_KOREAN_ORDINAL = {"첫": 1, "한": 1, "두": 2, "둘": 2, "세": 3, "셋": 3, "네": 4, "넷": 4, "다섯": 5,
                   "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10}
_ORDINAL_WORD = re.compile(r"(첫|한|두|둘|세|셋|네|넷|다섯|여섯|일곱|여덟|아홉|열)\s*(?:번\s*째|째)")
_ORDINAL_NUM = re.compile(r"(\d+)\s*번\s*째")
_ID = re.compile(r"(\d+)\s*번(?!\s*째)")
_ONLY_NUMBER = re.compile(r"^\s*(\d+)\s*(?:이요|요|번)?\s*[.!?]?\s*$")
_LAST = re.compile(r"마지막|맨\s*(?:끝|아래|뒤)")
_REFUSE = re.compile(r"아니[요에오]?|됐[어습]|괜찮[아습]|그만|안\s*할|하지\s*마|그냥\s*(?:둘|두|놔)|없던\s*일|필요\s*없")


def resolve_by_rules(pending: PendingSelection, user_text: str) -> Resolution:
    text = user_text.strip()
    shown = pending.shown()
    n = len(shown)

    m = _ORDINAL_WORD.search(text)
    if m:
        choice = pending.find(no=_KOREAN_ORDINAL[m.group(1)])
        if choice:
            return Resolution(SELECTED, "rules", choice=choice, note=f"규칙: '{m.group(0)}'")
    m = _ORDINAL_NUM.search(text)
    if m:
        choice = pending.find(no=int(m.group(1)))
        if choice:
            return Resolution(SELECTED, "rules", choice=choice, note=f"규칙: '{m.group(0)}'")
    if _LAST.search(text) and n:
        return Resolution(SELECTED, "rules", choice=shown[-1], note="규칙: '마지막'")
    m = _ID.search(text) or _ONLY_NUMBER.search(text)
    if m:
        value = int(m.group(1))
        choice = pending.find(complaint_id=value)       # 민원 번호가 우선
        if choice is None and value <= n:
            choice = pending.find(no=value)              # 아니면 목록 순번
        if choice:
            return Resolution(SELECTED, "rules", choice=choice, note=f"규칙: '{m.group(0).strip()}'")
    if _REFUSE.search(text):
        return Resolution(CANCELLED, "rules", note="규칙: 거절 표현")
    return Resolution(UNRESOLVED, "rules", note="규칙으로 알아듣지 못함")


# =============================================================================
# 진입점
# =============================================================================
def resolve(pending: PendingSelection, user_text: str, history: list[ChatMessage]) -> Resolution:
    if settings.CHAT_AGENT_ENABLED and _load_backend() is not None:
        try:
            return _resolve_by_agent(pending, user_text, history)
        except Exception as exc:
            logger.exception("Qwen-Agent 해석 실패 - 규칙으로 대신합니다")
            res = resolve_by_rules(pending, user_text)
            res.note = f"에이전트 실패({type(exc).__name__}: {exc}) -> {res.note}"
            return res
    res = resolve_by_rules(pending, user_text)
    if settings.CHAT_AGENT_ENABLED and _backend_error:
        res.note = f"Qwen-Agent 사용 불가({_backend_error}) -> {res.note}"
    return res
