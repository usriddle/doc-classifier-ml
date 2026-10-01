"""
벡터DB 저장소 - PostgreSQL + pgvector.

case_store 가 CASE_STORE_BACKEND=pgvector 일 때 사용합니다. (기본값 numpy 면 쓰지 않음)

테이블 (이름은 .env 의 CASE_PG_TABLE, 기본 complaint_cases)
  id          BIGSERIAL PRIMARY KEY
  text        TEXT          사례 원문
  category    TEXT          카테고리 라벨 (categories.py 의 name)
  query_text  TEXT          임베딩한 "키워드 + 요약" 질의문
  embedding   vector(1024)  bge-m3 벡터 (L2 정규화)
  + HNSW 인덱스 (vector_cosine_ops) : 코사인 거리 <=> 로 가까운 사례를 찾습니다.

메타 테이블 case_store_meta (key, value)
  "<테이블>:fingerprint" 에 CSV·임베딩 설정 지문을 저장해, 바뀌었을 때만 다시 적재합니다.

bge-m3 가 없는 곳(윈도우)에서는 사례를 임베딩할 수 없으므로,
Colab 이 만든 data/cases.npy + data/cases.meta.json 을 가져와 그대로 적재합니다.
"""

from __future__ import annotations

import re

import numpy as np

from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)

META_TABLE = "case_store_meta"
_IDENT = re.compile(r"^[a-z_][a-z0-9_]{0,62}$")


class PgStoreError(RuntimeError):
    """사용자에게 그대로 보여줄 수 있는 설정·연결 오류."""


def installed() -> bool:
    try:
        import psycopg  # noqa: F401
    except Exception:
        return False
    return True


def table_name() -> str:
    name = (settings.CASE_PG_TABLE or "").strip().lower()
    if not _IDENT.match(name):
        raise PgStoreError(
            f"CASE_PG_TABLE={settings.CASE_PG_TABLE!r} 는 테이블 이름으로 쓸 수 없습니다. "
            "(영문 소문자·숫자·밑줄, 63자 이하)"
        )
    return name


def _connect():
    if not installed():
        raise PgStoreError("psycopg 가 설치되어 있지 않습니다. python -m pip install -r requirements.txt")
    if not (settings.CASE_PG_DSN or "").strip():
        raise PgStoreError(
            "CASE_PG_DSN 이 비어 있습니다. .env 에 "
            "CASE_PG_DSN=postgresql://postgres:비밀번호@localhost:5432/minwon 형태로 적으세요."
        )
    import psycopg

    try:
        return psycopg.connect(settings.CASE_PG_DSN, connect_timeout=5)
    except Exception as exc:
        raise PgStoreError(
            f"PostgreSQL 에 접속하지 못했습니다 - {type(exc).__name__}: {exc}"
        ) from exc


def _vec(v: np.ndarray) -> str:
    """pgvector 텍스트 표현 '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(x):.7g}" for x in np.asarray(v).ravel()) + "]"


def _parse_vec(text: str) -> np.ndarray:
    return np.array([float(x) for x in text.strip("[]").split(",") if x], dtype=np.float32)


def _ensure_extension(cur) -> None:
    cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    if cur.fetchone():
        return
    cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
    if not cur.fetchone():
        raise PgStoreError(
            "이 PostgreSQL 에 pgvector 확장이 설치되어 있지 않습니다. "
            "README 의 1-6. PostgreSQL + pgvector 준비를 따라 설치하세요."
        )
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:
        raise PgStoreError(
            "vector 확장을 켜지 못했습니다(권한 부족일 수 있음). psql 에서 postgres 계정으로 "
            f"'CREATE EXTENSION vector;' 를 실행하세요. - {exc}"
        ) from exc


def read_state() -> tuple[str | None, int]:
    """(저장된 지문, 사례 수). 테이블이 없으면 (None, 0)."""
    from psycopg import sql

    table = table_name()
    with _connect() as conn, conn.cursor() as cur:
        _ensure_extension(cur)
        cur.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            .format(sql.Identifier(META_TABLE))
        )
        conn.commit()
        cur.execute("SELECT to_regclass(%s)", (table,))
        if cur.fetchone()[0] is None:
            return None, 0
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table)))
        count = int(cur.fetchone()[0])
        cur.execute(
            sql.SQL("SELECT value FROM {} WHERE key = %s").format(sql.Identifier(META_TABLE)),
            (f"{table}:fingerprint",),
        )
        row = cur.fetchone()
        return (row[0] if row else None), count


def replace_all(vectors: np.ndarray, cases: list, fingerprint: str) -> int:
    """테이블을 새로 만들고 사례 전체를 적재합니다. (한 트랜잭션 - 실패하면 이전 상태 유지)"""
    from psycopg import sql

    table = table_name()
    dim = int(vectors.shape[1])
    ident = sql.Identifier(table)
    with _connect() as conn:
        with conn.transaction(), conn.cursor() as cur:
            _ensure_extension(cur)
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(ident))
            cur.execute(
                sql.SQL(
                    "CREATE TABLE {} ("
                    " id BIGSERIAL PRIMARY KEY,"
                    " text TEXT NOT NULL,"
                    " category TEXT NOT NULL,"
                    " query_text TEXT NOT NULL,"
                    " embedding vector({}) NOT NULL,"
                    " created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                ).format(ident, sql.Literal(dim))
            )
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {} (text, category, query_text, embedding) "
                    "VALUES (%s, %s, %s, %s::vector)"
                ).format(ident),
                [(c.text, c.category, c.query_text, _vec(v)) for c, v in zip(cases, vectors)],
            )
            cur.execute(
                sql.SQL("CREATE INDEX {} ON {} USING hnsw (embedding vector_cosine_ops)").format(
                    sql.Identifier(f"{table}_embedding_hnsw"), ident
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX {} ON {} (category)").format(
                    sql.Identifier(f"{table}_category_idx"), ident
                )
            )
            cur.execute(
                sql.SQL(
                    "INSERT INTO {} (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value"
                ).format(sql.Identifier(META_TABLE)),
                (f"{table}:fingerprint", fingerprint),
            )
    logger.info("pgvector 적재 완료 | table=%s | %d건 | dim=%d", table, len(cases), dim)
    return len(cases)


def search(query_vector: np.ndarray, top_k: int) -> list[tuple[str, str, float]]:
    """(text, category, 코사인 유사도) 를 가까운 순서로 top_k 개."""
    from psycopg import sql

    q = _vec(query_vector)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL(
                "SELECT text, category, 1 - (embedding <=> %s::vector) AS score "
                "FROM {} ORDER BY embedding <=> %s::vector LIMIT %s"
            ).format(sql.Identifier(table_name())),
            (q, q, int(top_k)),
        )
        return [(t, c, float(s)) for t, c, s in cur.fetchall()]


def sample(n: int = 3) -> list[tuple[str, str, np.ndarray]]:
    """확인용 - 저장된 사례 n개와 그 벡터. (scripts/sync_cases.py 에서 사용)"""
    from psycopg import sql

    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT text, category, embedding::text FROM {} ORDER BY random() LIMIT %s")
            .format(sql.Identifier(table_name())),
            (int(n),),
        )
        return [(t, c, _parse_vec(v)) for t, c, v in cur.fetchall()]
