"""
대화 이어가기 - POST /api/v1/chat/message 한 번(=사용자 발화 한 번)을 처리합니다.

/analyze/* 는 요청 하나로 끝나지만, 여기서는 세션(session_store)에 상태를 남겨 다음 발화와 이어 줍니다.
지금 이어 주는 상황은 하나입니다.

  사용자 : 가로등 민원 취소해 주세요
  시스템 : 취소할 민원이 2건 있습니다. 어느 민원인가요?  1. 37번 ...  2. 41번 ...     <- 대기 상태 저장
  사용자 : 두 번째 거요                                                               <- Qwen-Agent 가 해석
  시스템 : 취소 완료 - 41번 민원 ...                                                   <- 원래 요청을 41번으로 실행

한 턴의 흐름
  1. 세션 읽기 (없으면 새로 만듦). 대기 상태가 CHAT_PENDING_TTL_MINUTES 를 넘었으면 버림
  2. 대기 상태가 있으면
       - 화면 버튼으로 고른 경우(choice_id) -> 바로 실행
       - 말로 답한 경우 -> chat_agent.resolve() (Qwen-Agent, 실패 시 규칙)
           selected    : 원래 도구 인자에 complaint_id 를 채워 tool_executor 로 실행, 대기 해제
           cancelled   : 대기 해제, "진행하지 않았습니다"
           new_request : 대기 해제 후 3 으로 (이번 말을 새 요청으로 처리)
           unresolved  : 다시 물음 (CHAT_MAX_SELECTION_ATTEMPTS 번 넘으면 대기 해제)
  3. 대기 상태가 없으면 기존 파이프라인(pipeline.run) 그대로.
     결과가 needs_selection(수정·취소 대상 여러 건)이면 후보에 번호를 붙여 묻고 대기 상태를 저장
  4. 사용자·시스템 발화를 기록

실제 DB 변경은 언제나 tool_executor 가 소유권·상태를 다시 검증한 뒤에만 합니다.
에이전트는 "후보 중 무엇을 골랐는지" 만 정하고, 그 값도 후보 목록 안의 번호인지 여기서 확인합니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import settings
from app.logging_config import get_logger
from app.services import chat_agent, pipeline, session_store, tool_executor
from app.services.chat_agent import CANCELLED, NEW_REQUEST, SELECTED, Resolution
from app.services.llm_engine import ToolCall
from app.services.pipeline import PipelineResult
from app.services.session_store import ASSISTANT, USER, Choice, PendingSelection
from app.services.tool_executor import ToolExecutionResult

logger = get_logger(__name__)

IDLE = "idle"
AWAITING_SELECTION = "awaiting_selection"

ACTION_NAME = {"update_complaint": "수정", "cancel_complaint": "취소"}


@dataclass
class ChatTurn:
    """한 턴의 결과. (라우터가 API 응답으로 바꿈)"""

    session_id: str
    session_created: bool
    state: str                                   # idle | awaiting_selection
    reply: str                                   # 사용자에게 보여 줄 문장
    kind: str                                    # analyzed | selection_asked | selected | cancelled | reasked | gave_up | no_pending
    choices: list[Choice] = field(default_factory=list)    # state=awaiting_selection 일 때 화면에 보여 줄 후보
    pending_action: str | None = None            # 대기 중인 동작 (수정 | 취소)
    resolution: Resolution | None = None         # 대기 중 답을 어떻게 해석했는지
    pipeline_result: PipelineResult | None = None   # 이번 턴에 파이프라인을 돌렸으면 그 결과
    tool_result: ToolExecutionResult | None = None  # 이번 턴에 DB 를 실행했으면 그 결과
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)


# =============================================================================
# 문장 만들기
# =============================================================================
def _question(pending: PendingSelection) -> str:
    shown = pending.shown()
    lines = [f"{pending.action}할 민원이 {pending.total}건 있습니다. 어느 민원인가요?"]
    lines += [f"{c.no}. {c.label()}" for c in shown]
    if pending.total > len(shown):
        lines.append(f"(그 밖에 {pending.total - len(shown)}건은 민원 번호로 말씀해 주세요)")
    lines.append("'두 번째 거', '37번 민원'처럼 말씀해 주세요. 그만두시려면 '아니요'라고 해 주세요.")
    return "\n".join(lines)


def _reask(pending: PendingSelection) -> str:
    left = settings.CHAT_MAX_SELECTION_ATTEMPTS - pending.attempts
    return (
        "어느 민원인지 알아듣지 못했습니다. 목록의 순서나 민원 번호로 다시 말씀해 주세요."
        + (f" (남은 기회 {left}번)" if left > 0 else "")
        + "\n" + "\n".join(f"{c.no}. {c.label()}" for c in pending.shown())
    )


def _result_reply(tr: ToolExecutionResult | None) -> str:
    if tr is None:
        return "처리하지 못했습니다."
    if tr.ok:
        return tr.message
    return f"처리하지 못했습니다 - {tr.error}"


def _pipeline_reply(result: PipelineResult) -> str:
    if result.rejected:
        return pipeline.REJECT_MESSAGE
    llm = result.llm
    if llm.intent.code == "out_of_scope":
        return pipeline.REJECT_MESSAGE
    if llm.intent.tool is None:            # 문의
        return llm.answer or "답변을 만들지 못했습니다."
    return _result_reply(result.tool_result)


# =============================================================================
# 대기 상태 만들기 / 실행
# =============================================================================
def _pending_from(result: PipelineResult, text: str) -> PendingSelection | None:
    """파이프라인 결과가 '여러 건이라 고르세요' 면 대기 상태를 만듭니다."""
    tr = result.tool_result
    call = result.llm.tool_call if result.llm else None
    if tr is None or not tr.needs_selection or not isinstance(tr.data, list) or not tr.data:
        return None
    if call is None or call.name not in ACTION_NAME:
        return None
    rows = tr.data[: session_store.MAX_STORED_CHOICES]
    choices = [
        Choice(no=i + 1, complaint_id=int(r["id"]), category=r.get("category", ""), content=r.get("content", ""),
               location=r.get("location", ""), status=r.get("status", ""), created_at=r.get("created_at", ""))
        for i, r in enumerate(rows)
    ]
    pending = PendingSelection(
        tool=call.name, action=ACTION_NAME[call.name], arguments=dict(call.arguments or {}),
        choices=choices, total=len(tr.data), question="", source_text=text,
        created_at=session_store._now(),
    )
    pending.question = _question(pending)
    return pending


def _execute(pending: PendingSelection, choice: Choice, user_id: str) -> ToolExecutionResult:
    """처음 요청의 도구를, 고른 민원 번호로 다시 실행합니다. (번호가 있으면 다른 찾기 조건보다 우선)"""
    arguments = {**pending.arguments, "complaint_id": choice.complaint_id}
    call = ToolCall(called=True, name=pending.tool, arguments=arguments, raw="(대화 이어가기 - 후보 선택)")
    return tool_executor.execute(call, user_id=user_id)


# =============================================================================
# 한 턴
# =============================================================================
def handle(
    text: str,
    user_id: str = "",
    session_id: str | None = None,
    choice_id: int | None = None,
    request_id: str = "-",
) -> ChatTurn:
    started = time.perf_counter()
    user_id = (user_id or settings.COMPLAINT_DEFAULT_USER).strip() or settings.COMPLAINT_DEFAULT_USER
    text = (text or "").strip()
    session, created = session_store.get_or_create(session_id, user_id)
    sid = session.session_id
    notes: list[str] = []

    pending = session.pending
    if pending is not None and pending.expired():
        notes.append(f"선택 대기가 {settings.CHAT_PENDING_TTL_MINUTES}분을 넘어 만료됨")
        session_store.set_pending(sid, None)
        pending = None

    history = session_store.messages(sid, limit=settings.CHAT_HISTORY_MESSAGES) if pending else []
    user_line = text or (f"[화면에서 선택] 민원 번호 {choice_id}" if choice_id else "")
    if user_line:
        session_store.add_message(sid, USER, user_line, {"choice_id": choice_id})

    turn = _turn(sid, created, user_id, text, choice_id, pending, history, notes, request_id)
    turn.timings_ms["total_ms"] = int((time.perf_counter() - started) * 1000)

    session_store.add_message(sid, ASSISTANT, turn.reply, {
        "kind": turn.kind, "state": turn.state,
        "choices": [c.complaint_id for c in turn.choices],
        "resolution": (turn.resolution.method + ":" + turn.resolution.kind) if turn.resolution else None,
    })
    logger.info(
        "[%s] 대화 | session=%s | user=%s | %s -> %s | %s",
        request_id, sid[:8], user_id, turn.kind, turn.state,
        turn.resolution.note if turn.resolution else "",
    )
    return turn


def _turn(sid, created, user_id, text, choice_id, pending, history, notes, request_id) -> ChatTurn:
    def done(**kw) -> ChatTurn:
        return ChatTurn(session_id=sid, session_created=created, notes=notes, **kw)

    # --- 화면 버튼으로 고름 ---
    if choice_id is not None:
        if pending is None:
            if not text:
                return done(state=IDLE, kind="no_pending",
                            reply="지금 고를 수 있는 목록이 없습니다. 원하시는 내용을 말씀해 주세요.")
            notes.append("고를 목록이 없어 choice_id 를 무시하고 문장을 처리함")
        else:
            choice = pending.find(complaint_id=int(choice_id))
            if choice is None:
                return done(state=AWAITING_SELECTION, kind="reasked", choices=pending.shown(),
                            pending_action=pending.action,
                            reply="목록에 없는 민원입니다. 다시 골라 주세요.\n" + "\n".join(
                                f"{c.no}. {c.label()}" for c in pending.shown()))
            res = Resolution(SELECTED, "button", choice=choice, note=f"화면에서 {choice.complaint_id}번 선택")
            return _apply_selection(sid, created, user_id, pending, res, notes)

    if not text:
        return done(state=AWAITING_SELECTION if pending else IDLE, kind="no_pending" if not pending else "reasked",
                    choices=pending.shown() if pending else [], pending_action=pending.action if pending else None,
                    reply=pending.question if pending else "말씀을 입력해 주세요.")

    # --- 대기 중인 선택에 대한 답 ---
    if pending is not None:
        started = time.perf_counter()
        res = chat_agent.resolve(pending, text, history)
        agent_ms = int((time.perf_counter() - started) * 1000)

        if res.kind == SELECTED and res.choice is not None:
            turn = _apply_selection(sid, created, user_id, pending, res, notes)
            turn.timings_ms["resolve_ms"] = agent_ms
            return turn
        if res.kind == CANCELLED:
            session_store.set_pending(sid, None)
            return done(state=IDLE, kind="cancelled", resolution=res, timings_ms={"resolve_ms": agent_ms},
                        reply=f"알겠습니다. {pending.action}하지 않았습니다.")
        if res.kind == NEW_REQUEST:
            session_store.set_pending(sid, None)
            notes.append(f"선택 대기를 끝내고 새 요청으로 처리함 ({res.note})")
            turn = _analyze(sid, created, user_id, text, notes, request_id)
            turn.resolution = res
            turn.timings_ms["resolve_ms"] = agent_ms
            return turn

        # 알아듣지 못함
        pending.attempts += 1
        if pending.attempts >= settings.CHAT_MAX_SELECTION_ATTEMPTS:
            session_store.set_pending(sid, None)
            return done(state=IDLE, kind="gave_up", resolution=res, timings_ms={"resolve_ms": agent_ms},
                        reply=f"어느 민원인지 정하지 못해 {pending.action} 요청을 마쳤습니다. 처음부터 다시 말씀해 주세요.")
        session_store.set_pending(sid, pending)
        return done(state=AWAITING_SELECTION, kind="reasked", resolution=res, choices=pending.shown(),
                    pending_action=pending.action, timings_ms={"resolve_ms": agent_ms}, reply=_reask(pending))

    # --- 새 요청 ---
    return _analyze(sid, created, user_id, text, notes, request_id)


def _apply_selection(sid, created, user_id, pending: PendingSelection, res: Resolution, notes) -> ChatTurn:
    tr = _execute(pending, res.choice, user_id)
    session_store.set_pending(sid, None)
    return ChatTurn(
        session_id=sid, session_created=created, state=IDLE, kind="selected",
        reply=_result_reply(tr), resolution=res, tool_result=tr, notes=notes,
    )


def _analyze(sid, created, user_id, text, notes, request_id) -> ChatTurn:
    started = time.perf_counter()
    result = pipeline.run(text, request_id=request_id, user_id=user_id)
    timings = {"pipeline_ms": int((time.perf_counter() - started) * 1000)}

    pending = _pending_from(result, text)
    if pending is not None:
        session_store.set_pending(sid, pending)
        return ChatTurn(
            session_id=sid, session_created=created, state=AWAITING_SELECTION, kind="selection_asked",
            reply=pending.question, choices=pending.shown(), pending_action=pending.action,
            pipeline_result=result, tool_result=result.tool_result, notes=notes, timings_ms=timings,
        )
    return ChatTurn(
        session_id=sid, session_created=created, state=IDLE, kind="analyzed",
        reply=_pipeline_reply(result), pipeline_result=result, tool_result=result.tool_result,
        notes=notes, timings_ms=timings,
    )


# =============================================================================
# 조회 / 초기화 / 상태
# =============================================================================
def get_session(session_id: str, user_id: str = "") -> dict:
    user_id = (user_id or settings.COMPLAINT_DEFAULT_USER).strip() or settings.COMPLAINT_DEFAULT_USER
    session = session_store.get(session_id, user_id)
    pending = session.pending
    if pending is not None and pending.expired():
        session_store.set_pending(session_id, None)
        pending = None
    return {
        "session_id": session.session_id,
        "user_id": session.user_id,
        "state": AWAITING_SELECTION if pending else IDLE,
        "pending_action": pending.action if pending else None,
        "choices": pending.shown() if pending else [],
        "messages": session_store.messages(session_id),
        "created_at": session.created_at,
        "updated_at": session.updated_at,
    }


def reset_session(session_id: str, user_id: str = "") -> None:
    user_id = (user_id or settings.COMPLAINT_DEFAULT_USER).strip() or settings.COMPLAINT_DEFAULT_USER
    session_store.delete(session_id, user_id)


def status() -> dict[str, object]:
    return {**session_store.status(), **chat_agent.status()}
