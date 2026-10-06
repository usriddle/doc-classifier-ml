"""
⑥ 민원 DB (관계형).

⑤ 가 만든 도구 호출(register_complaint / get_complaints / update_complaint /
cancel_complaint)을 실제로 실행하는 저장소입니다. Gemma 는 이 파일을 전혀 모르고,
"무엇을 어떤 값으로" 만 정할 뿐입니다 - 실행은 tool_executor.py 가 이 모듈의
함수를 호출해서 합니다.

저장 방식은 SQLite 파일 하나(`data/complaints.db`, 표준 라이브러리 `sqlite3`)입니다.
PostgreSQL(pgvector 처럼)이 아니라 SQLite 를 고른 이유는:
  - 이 DB 를 실제로 써야 하는 곳이 Colab(모델) 과 PC(추출) 양쪽 다이고,
    Colab 은 PC 의 PostgreSQL 에 네트워크로 닿을 수 없습니다. (벡터DB 와 같은 제약)
  - 민원 등록·조회·수정·취소는 트래픽이 낮고 스키마가 단순해 SQLite 로 충분합니다.
  - 파일 하나라 설치가 필요 없고, PC ↔ Colab 간 이동도 파일 복사로 끝납니다.
운영 규모가 커지면 이 파일의 함수 시그니처를 유지한 채 PostgreSQL 구현으로
바꿔 끼울 수 있습니다. (case_store.py 의 numpy/pgvector 분리와 같은 자리)

테이블
  complaints         : 민원 1건 = 1행. id 는 자동 증가하며 그림의 "id=1043" 이 이 값입니다.
  complaint_history  : 상태가 바뀔 때마다 한 행씩 쌓입니다. (접수/수정/취소)

로그인 사용자 가정 : 이 시스템은 아직 실제 인증이 없어, 호출자가 넘기는
user_id 문자열을 "로그인한 사용자"로 취급합니다. 조회·수정·취소는 항상
"그 user_id 가 등록한 민원인지" 를 같이 확인해 다른 사용자의 민원에는
접근하지 못하게 합니다.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from app.config import BASE_DIR, settings
from app.services import categories

_SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     TEXT NOT NULL,
    category    TEXT NOT NULL,
    content     TEXT NOT NULL,
    location    TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT '접수',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_complaints_user ON complaints(user_id);

CREATE TABLE IF NOT EXISTS complaint_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id  INTEGER NOT NULL REFERENCES complaints(id),
    status        TEXT NOT NULL,
    note          TEXT NOT NULL DEFAULT '',
    changed_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_complaint ON complaint_history(complaint_id);
"""


@dataclass
class Complaint:
    id: int
    user_id: str
    category: str
    content: str
    location: str
    status: str
    created_at: str
    updated_at: str

    @property
    def department(self) -> str:
        return categories.get(self.category).department


@dataclass
class HistoryEntry:
    status: str
    note: str
    changed_at: str


# =============================================================================
# 연결 / 스키마
# =============================================================================
def db_path() -> Path:
    path = Path(settings.COMPLAINT_DB_PATH)
    return path if path.is_absolute() else BASE_DIR / path


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout : 여러 요청이 거의 동시에 쓰기를 시도해도 잠깐 기다렸다 진행합니다.
    conn = sqlite3.connect(str(path), timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _row_to_complaint(row: sqlite3.Row) -> Complaint:
    return Complaint(
        id=row["id"], user_id=row["user_id"], category=row["category"],
        content=row["content"], location=row["location"], status=row["status"],
        created_at=row["created_at"], updated_at=row["updated_at"],
    )


# =============================================================================
# 접수 (INSERT)
# =============================================================================
def register(user_id: str, category: str, content: str, location: str = "") -> Complaint:
    """
    새 민원을 등록합니다. 중복을 허용하므로 같은 내용이어도 그대로 새 행을 만듭니다.
    category 가 7종에 없으면 categories.get() 이 '기타'로 대체합니다. (예외를 던지지 않음. 예전 9종 이름은 지금 이름으로 바꿔 받음)
    """
    cat_name = categories.get(category).name
    text = (content or "").strip() or "(내용 없음)"
    loc = (location or "").strip()
    now = _now()

    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO complaints(user_id, category, content, location, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, '접수', ?, ?)",
            (user_id, cat_name, text, loc, now, now),
        )
        cid = int(cur.lastrowid)
        conn.execute(
            "INSERT INTO complaint_history(complaint_id, status, note, changed_at) VALUES (?, '접수', '', ?)",
            (cid, now),
        )
        conn.commit()

    complaint = get_one(user_id, cid)
    assert complaint is not None  # 방금 넣은 행이므로 항상 찾아야 정상입니다.
    return complaint


# =============================================================================
# 조회 (SELECT, 읽기 전용)
# =============================================================================
def get_one(user_id: str, complaint_id: int) -> Complaint | None:
    """본인(user_id) 민원이 아니면 None 을 돌려줍니다. (다른 사람 민원 조회 차단)"""
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM complaints WHERE id = ? AND user_id = ?", (complaint_id, user_id),
        ).fetchone()
    return _row_to_complaint(row) if row else None


def list_all(user_id: str, limit: int = 20) -> list[Complaint]:
    """최근 등록순. 그림의 '목록' 조회입니다."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM complaints WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
    return [_row_to_complaint(r) for r in rows]


def list_for_search(user_id: str, exclude_cancelled: bool = False, limit: int = 1000) -> list[Complaint]:
    """번호 없이 찾을 때의 검색 대상. 최근 접수순(created_at). (조건 거르기는 tool_executor 가 합니다)"""
    sql = "SELECT * FROM complaints WHERE user_id = ?"
    if exclude_cancelled:
        sql += " AND status != '취소'"
    sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
    with _connect() as conn:
        rows = conn.execute(sql, (user_id, limit)).fetchall()
    return [_row_to_complaint(r) for r in rows]


def history(user_id: str, complaint_id: int) -> list[HistoryEntry] | None:
    """본인 민원이 아니면 None. 있으면 접수부터 최신까지 시간순."""
    if get_one(user_id, complaint_id) is None:
        return None
    with _connect() as conn:
        rows = conn.execute(
            "SELECT status, note, changed_at FROM complaint_history "
            "WHERE complaint_id = ? ORDER BY id ASC",
            (complaint_id,),
        ).fetchall()
    return [HistoryEntry(status=r["status"], note=r["note"], changed_at=r["changed_at"]) for r in rows]


# =============================================================================
# 수정 / 취소 (UPDATE)
# =============================================================================
def update(user_id: str, complaint_id: int, content: str = "", location: str = "") -> Complaint | None:
    """
    호출 전에 tool_executor 가 소유권·상태(취소 아님)·변경할 값 존재 여부를
    이미 검증했다고 가정합니다. content/location 중 빈 문자열은 "바꾸지 않음"입니다.
    """
    complaint = get_one(user_id, complaint_id)
    if complaint is None:
        return None

    new_content = content.strip() or complaint.content
    new_location = location.strip() if location.strip() else complaint.location
    now = _now()

    with _connect() as conn:
        conn.execute(
            "UPDATE complaints SET content = ?, location = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (new_content, new_location, now, complaint_id, user_id),
        )
        conn.execute(
            "INSERT INTO complaint_history(complaint_id, status, note, changed_at) VALUES (?, '수정', ?, ?)",
            (complaint_id, f"내용/위치 변경", now),
        )
        conn.commit()

    return get_one(user_id, complaint_id)


def cancel(user_id: str, complaint_id: int, reason: str = "") -> Complaint | None:
    """상태를 '취소'로 바꿉니다. (soft delete - 행을 지우지 않습니다)"""
    complaint = get_one(user_id, complaint_id)
    if complaint is None:
        return None

    now = _now()
    with _connect() as conn:
        conn.execute(
            "UPDATE complaints SET status = '취소', updated_at = ? WHERE id = ? AND user_id = ?",
            (now, complaint_id, user_id),
        )
        conn.execute(
            "INSERT INTO complaint_history(complaint_id, status, note, changed_at) VALUES (?, '취소', ?, ?)",
            (complaint_id, (reason or "").strip(), now),
        )
        conn.commit()

    return get_one(user_id, complaint_id)


# =============================================================================
# 상태 (status 엔드포인트 / debug 용)
# =============================================================================
def status() -> dict[str, object]:
    try:
        with _connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]
            by_status = dict(
                conn.execute("SELECT status, COUNT(*) FROM complaints GROUP BY status").fetchall()
            )
        return {"ready": True, "path": str(db_path()), "complaints": total, "by_status": by_status, "error": None}
    except Exception as exc:  # 디스크 권한 등 - 서버는 계속 뜨고 이유만 보여줍니다.
        return {"ready": False, "path": str(db_path()), "complaints": 0, "by_status": {}, "error": str(exc)}
