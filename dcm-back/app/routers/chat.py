"""
대화 이어가기 API. (프론트 연결용 - 사용 흐름은 front_stub/README.md)

  POST   /chat/message                 한 턴 보내기 (처음에는 session_id 없이)
  GET    /chat/sessions/{session_id}   세션 상태 + 대화 기록 (화면 새로고침 후 복원용)
  DELETE /chat/sessions/{session_id}   세션 지우기 (대화 새로 시작)
"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, Query

from app.config import settings
from app.exceptions import AppError
from app.logging_config import get_logger
from app.routers.analyze import _build_response, _to_tool_result
from app.schemas import (
    ChatChoiceItem,
    ChatHistoryItem,
    ChatMessageRequest,
    ChatResolutionInfo,
    ChatResponse,
    ChatSessionResponse,
)
from app.services import chat

logger = get_logger(__name__)
router = APIRouter()


class ChatBadRequestError(AppError):
    status_code = 422
    error_code = "VALIDATION_ERROR"


def _choice_items(choices) -> list[ChatChoiceItem]:
    return [
        ChatChoiceItem(no=c.no, complaint_id=c.complaint_id, label=c.label(), category=c.category,
                       content=c.content, location=c.location, status=c.status, created_at=c.created_at)
        for c in choices
    ]


def _resolution(res) -> ChatResolutionInfo | None:
    if res is None:
        return None
    return ChatResolutionInfo(
        method=res.method,
        kind=res.kind,
        selected_complaint_id=res.choice.complaint_id if res.choice else None,
        note=res.note,
        agent_decision=res.agent.get("decision") if settings.DEBUG and res.agent else None,
        agent_raw_output=res.agent.get("raw", "") if settings.DEBUG and res.agent else "",
    )


@router.post(
    "/chat/message",
    response_model=ChatResponse,
    summary="대화 한 턴 (여러 건이면 되묻고, 다음 말로 이어서 처리)",
    description=(
        "처음에는 `session_id` 없이 보내고, 응답의 `session_id` 를 다음 요청부터 그대로 보내세요.\n\n"
        "수정·취소 대상이 여러 건이면 `state=awaiting_selection` 과 `choices` 를 돌려줍니다. "
        "사용자가 \"두 번째 거요\", \"가로등 거\", \"37번 민원\" 처럼 말하면 `text` 로, "
        "화면 버튼을 누르면 `selected_complaint_id` 로 보내세요. 원래 요청이 그 민원에 실행됩니다.\n\n"
        "말로 한 답은 Qwen-Agent(함수 호출)가 해석합니다. (`CHAT_AGENT_ENABLED`, 설치되지 않았으면 규칙으로 해석)"
    ),
)
async def chat_message(payload: ChatMessageRequest) -> ChatResponse:
    if not payload.text.strip() and payload.selected_complaint_id is None:
        raise ChatBadRequestError("text 나 selected_complaint_id 중 하나는 있어야 합니다.")
    request_id = uuid.uuid4().hex[:12]
    turn = await asyncio.to_thread(
        chat.handle,
        payload.text,
        payload.user_id,
        payload.session_id or None,
        payload.selected_complaint_id,
        request_id,
    )
    return ChatResponse(
        session_id=turn.session_id,
        session_created=turn.session_created,
        state=turn.state,
        kind=turn.kind,
        reply=turn.reply,
        pending_action=turn.pending_action,
        choices=_choice_items(turn.choices),
        tool_result=_to_tool_result(turn.tool_result),
        analysis=_build_response(turn.pipeline_result, None) if turn.pipeline_result is not None else None,
        resolution=_resolution(turn.resolution),
        notes=turn.notes,
        timings_ms=turn.timings_ms,
    )


@router.get(
    "/chat/sessions/{session_id}",
    response_model=ChatSessionResponse,
    summary="대화 세션 상태와 기록",
    description="화면을 새로 열었을 때 대화 기록과 '고르는 중' 상태를 복원하는 용도입니다.",
)
async def chat_session(session_id: str, user_id: str = Query(default="")) -> ChatSessionResponse:
    data = await asyncio.to_thread(chat.get_session, session_id, user_id)
    return ChatSessionResponse(
        session_id=data["session_id"],
        user_id=data["user_id"],
        state=data["state"],
        pending_action=data["pending_action"],
        choices=_choice_items(data["choices"]),
        messages=[ChatHistoryItem(role=m.role, text=m.text, created_at=m.created_at) for m in data["messages"]],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
    )


@router.delete(
    "/chat/sessions/{session_id}",
    summary="대화 세션 지우기",
    description="'새 대화' 버튼용. 대기 중인 선택과 대화 기록이 함께 지워집니다.",
)
async def chat_reset(session_id: str, user_id: str = Query(default="")) -> dict:
    await asyncio.to_thread(chat.reset_session, session_id, user_id)
    return {"success": True, "session_id": session_id}


@router.get("/chat/status", summary="대화 기능 상태 (Qwen-Agent 사용 가능 여부, 세션 수)")
async def chat_status() -> dict:
    return await asyncio.to_thread(chat.status)
