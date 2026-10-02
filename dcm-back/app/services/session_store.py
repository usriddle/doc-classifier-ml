"""
대화 세션 저장소 - "어느 민원인가요?" 라고 되물은 상태를 다음 요청까지 기억합니다.

지금까지의 /analyze/* 는 요청 하나로 끝나서, 수정·취소 대상이 여러 건이면 후보만 돌려주고
잊어버렸습니다. 사용자가 "두 번째 거요" 라고 답해도 무엇의 두 번째인지 알 수 없었습니다.
이 모듈은 세션마다 다음을 저장합니다.

  chat_sessions   : 세션 1개 = 1행. 사용자(user_id)와 '대기 중인 선택'(pending, JSON)
  chat_messages   : 사용자·시스템 발화 기록 (에이전트에게 최근 대화를 보여 주는 용도)

대기 중인 선택(PendingSelection)
  되물을 때의 수정·취소 도구, 원래 인자(바꿀 위치·취소 사유 등), 번호를 붙인 후보 목록.
  사용자가 고르면 원래 인자에 complaint_id 만 채워 tool_executor 로 다시 실행합니다.

저장 방식은 민원 DB 와 같은 이유로 SQLite 파일(CHAT_DB_PATH)입니다. (complaint_store.py 참고)
세션은 user_id 에 묶여 있어, 다른 사용자의 session_id 로는 접근할 수 없습니다.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import BASE_DIR, settings
from app.exceptions import SessionForbiddenError, SessionNotFoundError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    pending     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS chat_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    meta        TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, id);
"""

MAX_STORED_CHOICES = 50   # 대기 상태에 저장할 최대 후보 수 (민원 번호 검증용)

USER = "user"
ASSISTANT = "assistant"


# =============================================================================
# 자료구조
# =============================================================================
@dataclass
class Choice:
    """되물을 때 보여 준 후보 1건. no 는 화면에 보인 순번(1부터)입니다."""

    no: int
    complaint_id: int
    category: str = ""
    content: str = ""
    location: str = ""
    status: str = ""
    created_at: str = ""

    def label(self) -> str:
        where = f" / {self.location}" if self.location else ""
        day = f" / {self.created_at[:10]}" if self.created_at else ""
        return f"{self.complaint_id}번 민원 ({self.category} / {self.content[:30]}{where}{day})"


@dataclass
class PendingSelection:
    """수정·취소 대상을 하나로 정하지 못해 사용자에게 고르게 한 상태."""

    tool: str                       # update_complaint | cancel_complaint
    action: str                     # 수정 | 취소 (사람이 읽는 이름)
    arguments: dict                 # 처음 요청의 도구 인자 (바꿀 위치·내용, 취소 사유 등)
    choices: list[Choice]           # 찾은 후보 전부 (최대 MAX_STORED_CHOICES). 보여 주는 건 앞 CHAT_MAX_CHOICES 개
    total: int                      # 찾은 후보 전체 수
    question: str                   # 사용자에게 한 질문
    source_text: str = ""           # 처음 요청 문장
    created_at: str = ""
    attempts: int = 0               # 고르지 못해 다시 물은 횟수

    def expired(self, now: datetime | None = None) -> bool:
        try:
            made = datetime.fromisoformat(self.created_at)
        except ValueError:
            return True
        now = now or datetime.now(timezone.utc)
        return now - made > timedelta(minutes=settings.CHAT_PENDING_TTL_MINUTES)

    def shown(self) -> list[Choice]:
        """사용자에게 보여 준(에이전트에게 주는) 후보. 민원 번호로는 shown 밖의 후보도 고를 수 있습니다."""
        return self.choices[: max(1, settings.CHAT_MAX_CHOICES)]

    def find(self, complaint_id: int | None = None, no: int | None = None) -> Choice | None:
        for c in self.choices:
            if complaint_id is not None and c.complaint_id == complaint_id:
                return c
            if no is not None and c.no == no:
                return c
        return None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "PendingSelection | None":
        if not raw:
            return None
        try:
            data = json.loads(raw)
            data["choices"] = [Choice(**c) for c in data.get("choices", [])]
            return cls(**data)
        except Exception:
            return None   # 형식이 바뀐 옛 데이터 - 대기 상태가 없는 것으로 봄


@dataclass
class ChatMessage:
    role: str
    text: str
    created_at: str
    meta: dict = field(default_factory=dict)


@dataclass
class Session:
    session_id: str
    user_id: str
    created_at: str
    updated_at: str
    pending: PendingSelection | None = None


# =============================================================================
# 연결
# =============================================================================
def db_path() -> Path:
    path = Path(settings.CHAT_DB_PATH)
    return path if path.is_absolute() else BASE_DIR / path


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=row["session_id"], user_id=row["user_id"],
        created_at=row["created_at"], updated_at=row["updated_at"],
        pending=PendingSelection.from_json(row["pending"]),
    )


# =============================================================================
# 세션
# =============================================================================
def get(session_id: str, user_id: str) -> Session:
    """세션을 읽습니다. 없으면 SessionNotFoundError, 다른 사용자 것이면 SessionForbiddenError."""
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        raise SessionNotFoundError(
            "대화 세션을 찾을 수 없습니다.", detail="session_id 를 비우고 보내면 새 세션을 만듭니다."
        )
    session = _row_to_session(row)
    if session.user_id != user_id:
        raise SessionForbiddenError("다른 사용자의 대화 세션입니다.")
    return session


def get_or_create(session_id: str | None, user_id: str) -> tuple[Session, bool]:
    """(세션, 새로 만들었는지). session_id 가 비어 있으면 새 세션을 만듭니다."""
    if session_id:
        return get(session_id, user_id), False
    now = _now()
    session = Session(session_id=uuid.uuid4().hex, user_id=user_id, created_at=now, updated_at=now)
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO chat_sessions (session_id, user_id, pending, created_at, updated_at) VALUES (?, ?, '', ?, ?)",
            (session.session_id, user_id, now, now),
        )
    return session, True


def set_pending(session_id: str, pending: PendingSelection | None) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "UPDATE chat_sessions SET pending = ?, updated_at = ? WHERE session_id = ?",
            (pending.to_json() if pending else "", _now(), session_id),
        )


def delete(session_id: str, user_id: str) -> None:
    get(session_id, user_id)   # 존재·소유 확인
    with closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))


# =============================================================================
# 대화 기록
# =============================================================================
def add_message(session_id: str, role: str, text: str, meta: dict | None = None) -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, text, meta, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, text, json.dumps(meta or {}, ensure_ascii=False, default=str), _now()),
        )
        conn.execute("UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?", (_now(), session_id))


def messages(session_id: str, limit: int | None = None) -> list[ChatMessage]:
    """오래된 순서로. limit 이 있으면 최근 limit 개만."""
    with closing(_connect()) as conn:
        if limit:
            rows = conn.execute(
                "SELECT * FROM (SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                (session_id, int(limit)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
    out = []
    for r in rows:
        try:
            meta = json.loads(r["meta"]) if r["meta"] else {}
        except ValueError:
            meta = {}
        out.append(ChatMessage(role=r["role"], text=r["text"], created_at=r["created_at"], meta=meta))
    return out


def status() -> dict[str, object]:
    try:
        with closing(_connect()) as conn:
            sessions = conn.execute("SELECT COUNT(*) FROM chat_sessions").fetchone()[0]
            waiting = conn.execute("SELECT COUNT(*) FROM chat_sessions WHERE pending != ''").fetchone()[0]
        return {"db_path": str(db_path()), "sessions": sessions, "awaiting_selection": waiting, "error": None}
    except Exception as exc:
        return {"db_path": str(db_path()), "sessions": 0, "awaiting_selection": 0, "error": f"{type(exc).__name__}: {exc}"}
