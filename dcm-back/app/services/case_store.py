"""
④ 보조 - 벡터DB (라벨링된 사례).

비슷한 과거 사례의 카테고리 라벨로 ④ 의 top-3 후보를 다시 정렬합니다.
언제·어떻게 쓸지는 .env 의 CASE_MODE 로 고릅니다.
  low_confidence : ④ 1위 유사도 < CANDIDATE_MIN_SCORE 일 때만 섞기 (기존 방식)
  always         : 매번 섞기
  union          : 매번 섞고, 사례 점수 1위 카테고리가 top-3 밖이면 3위 자리에 넣기
사용자와 무관한 분류 보조 전용입니다. (중복 확인·개인 검색에는 쓰지 않습니다)

저장 방식 : .env 의 CASE_STORE_BACKEND 로 고릅니다. 검색 결과는 둘이 같습니다.
  numpy    (기본) 메모리에서 numpy 로 계산. 추가 설치 없음 - Colab 테스트용
  pgvector PostgreSQL + pgvector 테이블에서 SQL 로 검색 (pg_store.py) - PC / 운영 서버용

  data/cases.csv        원본. 컬럼 text, category (사람이 편집하는 파일)
  data/cases.npy        사례 질의문 벡터 (N, 1024) - 자동 생성
  data/cases.meta.json  사례 목록 + 지문(fingerprint) - 자동 생성
  두 파일은 numpy 모드의 저장소이자, bge-m3 가 없는 PC 에 벡터를 옮기는 전달 형식입니다.

임베딩 대상 : ④ 와 똑같이 keyphrase.build_query() 의 "키워드 + 요약" 질의문.
  검색할 때와 같은 방식으로 만들어야 유사도가 의미를 가집니다.

인덱스 생성 : 서버 기동 시 load_or_build() 가 CSV 를 읽습니다.
  - CSV 내용·임베딩 모델·질의문 설정이 그대로면 .npy 만 읽습니다. (bge-m3 로드 불필요)
  - 하나라도 바뀌었으면 사례를 다시 임베딩해 .npy 를 새로 씁니다. (bge-m3 로드 필요)
  - pgvector 모드는 DB 에 저장된 지문이 같으면 아무것도 하지 않고, 다르면
    .npy(없으면 새로 임베딩)의 내용으로 테이블을 통째로 다시 적재합니다.

재정렬 방식 (카테고리 c 마다)
  사례 점수(c)  = CASE_MIN_SCORE 이상인 유사 사례(최대 k개) 중 라벨이 c 인 것들의 유사도 합
                  / k (= CASE_TOP_K, 기준을 못 넘은 자리는 0표)   (0~1, 코사인과 같은 눈금)
                  통과한 사례 수로 나누지 않고 k 로 나누므로, 사례가 1~2건뿐이면 영향이 그만큼 작아집니다.
                  (겨우 기준을 넘은 사례 한 건이 만장일치처럼 순위를 뒤집는 것을 막음)
  최종 점수(c)  = (1 - w) * ④ 점수(c) + w * 사례 점수(c)       w = CASE_BLEND_WEIGHT
  -> 최종 점수로 다시 정렬해 top-3 를 ⑤ 에 넘깁니다. ⑤ 프롬프트는 바뀌지 않습니다.
  union 은 여기에 더해 사례 점수 1위 카테고리를 후보에 반드시 넣습니다.
  (④ 가 '확신한 채로 틀려도' 사례가 가리키는 카테고리가 후보에 남도록)

계산식은 case_scores_from() / combine() 두 순수 함수에 있고, 학습 노트북의 측정 셀
(training/calibrate.py)도 같은 함수를 써서 운영과 같은 결과를 냅니다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import BASE_DIR, settings
from app.logging_config import get_logger
from app.services import categories, embedder, keyphrase, pg_store
from app.services.candidates import CandidateResult, CategoryScore

logger = get_logger(__name__)

# 질의문 만드는 방식이 코드 수준에서 바뀌면 올려 주세요. 기존 .npy 를 무효화합니다.
_QUERY_VERSION = 1
# 재정렬 계산식이 바뀌면 올려 주세요. 학습 노트북의 ④ 후보 캐시(candidates.json)를 무효화합니다.
# 2: 사례 점수를 '통과한 사례 수' 대신 CASE_TOP_K 로 나눔
# 3: CASE_MODE(low_confidence/always/union) 추가
RERANK_VERSION = 3

MODES = ("low_confidence", "always", "union")


def mode(value: str | None = None) -> str:
    """CASE_MODE 정규화. 모르는 값이면 기존 방식(low_confidence)."""
    v = (value if value is not None else settings.CASE_MODE or "").strip().lower()
    return v if v in MODES else "low_confidence"


def should_consult(low_confidence: bool, case_mode: str | None = None) -> bool:
    """이번 요청에서 벡터DB 를 열지."""
    return mode(case_mode) != "low_confidence" or bool(low_confidence)

_lock = threading.Lock()
_vectors: np.ndarray | None = None
_cases: list["Case"] = []
_attempted = False
_error: str | None = None
_source = ""          # "cache" | "built" | ""
_skipped = 0          # CSV 에서 건너뛴 행 수
_count = 0            # 사용 가능한 사례 수 (pgvector 는 DB 에만 있고 메모리에 올리지 않음)
_pg_ready = False


def backend() -> str:
    value = (settings.CASE_STORE_BACKEND or "numpy").strip().lower()
    return "pgvector" if value in ("pgvector", "postgres", "postgresql", "pg") else "numpy"


# =============================================================================
# 자료구조
# =============================================================================
@dataclass
class Case:
    text: str
    category: str       # 카테고리 이름 (categories.py 의 name)
    query_text: str     # 임베딩한 "키워드 + 요약" 질의문


@dataclass
class CaseHit:
    rank: int
    text: str
    category: str
    score: float        # 질의 벡터와 사례 벡터의 코사인 유사도


@dataclass
class CaseLookup:
    """벡터DB 조회 결과 (DEBUG 확인용)."""

    used: bool = False                 # 실제로 후보 순서에 반영했는지
    reason: str = ""                   # 왜 열었는지 / 왜 반영하지 않았는지
    hits: list[CaseHit] = field(default_factory=list)            # 임계값 통과한 사례
    case_scores: dict[str, float] = field(default_factory=dict)  # 카테고리별 사례 점수
    before: list[CategoryScore] = field(default_factory=list)    # 재정렬 전 top-k (④ 그대로)
    before_all: list[CategoryScore] = field(default_factory=list)  # 재정렬 전 7종 전체
    after: list[CategoryScore] = field(default_factory=list)     # 재정렬 후 top-k (⑤ 에 전달)


# =============================================================================
# 경로 / 지문
# =============================================================================
def csv_path() -> Path:
    path = Path(settings.CASE_CSV_PATH)
    return path if path.is_absolute() else BASE_DIR / path


def _vector_path() -> Path:
    return csv_path().with_suffix(".npy")


def _meta_path() -> Path:
    return csv_path().with_suffix(".meta.json")


def _fingerprint(csv_bytes: bytes) -> str:
    """CSV 내용 + 임베딩 모델 + 질의문 설정. 하나라도 바뀌면 다시 임베딩합니다."""
    parts = [
        hashlib.sha256(csv_bytes).hexdigest(),
        settings.EMBED_MODEL,
        str(settings.EMBED_NORMALIZE),
        str(settings.EMBED_MAX_SEQ_LENGTH),
        str(settings.KEYWORD_TOP_K),
        str(settings.KEYWORD_USE_MMR),
        str(settings.KEYWORD_MMR_DIVERSITY),
        str(settings.SUMMARY_MAX_SENTENCES),
        str(settings.SUMMARY_MAX_CHARS),
        str(_QUERY_VERSION),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# =============================================================================
# CSV 읽기
# =============================================================================
def _read_csv(raw: bytes) -> tuple[list[tuple[str, str]], int]:
    """(text, category 이름) 목록과 건너뛴 행 수. 카테고리는 이름 또는 코드로 적을 수 있습니다."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = {(f or "").strip().lower() for f in (reader.fieldnames or [])}
    if not {"text", "category"} <= fields:
        raise ValueError(f"CSV 에 text, category 컬럼이 필요합니다. (현재: {reader.fieldnames})")

    rows: list[tuple[str, str]] = []
    skipped = 0
    for line_no, row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        text, label = row.get("text", ""), row.get("category", "")
        if not text:
            skipped += 1
            continue
        category = categories.BY_NAME.get(label) or categories.BY_CODE.get(label.lower())
        if category is None:
            skipped += 1
            logger.warning(
                "사례 CSV %d행 건너뜀 - 알 수 없는 카테고리 %r (가능: %s)",
                line_no, label, "/".join(categories.NAMES),
            )
            continue
        rows.append((text, category.name))
    return rows, skipped


# =============================================================================
# 로드 / 생성
# =============================================================================
def load_or_build() -> None:
    """서버 기동 시 호출. 실패해도 예외를 올리지 않고 status() 의 error 에 남깁니다."""
    global _vectors, _cases, _attempted, _error, _source, _skipped, _count, _pg_ready

    with _lock:
        _attempted = True
        if not settings.CASE_STORE_ENABLED:
            _error = None
            return

        path = csv_path()
        try:
            if not path.exists():
                raise FileNotFoundError(f"사례 CSV 가 없습니다: {path}")

            raw = path.read_bytes()
            rows, skipped = _read_csv(raw)
            if not rows:
                raise ValueError("사례 CSV 에 사용할 수 있는 행이 없습니다.")
            fingerprint = _fingerprint(raw)
            _skipped = skipped

            if backend() == "pgvector":
                _load_pgvector(rows, fingerprint)
            else:
                cached = _load_cache(fingerprint, len(rows))
                if cached is not None:
                    _vectors, _cases = cached
                    _source = "cache"
                else:
                    _vectors, _cases = _build(rows)
                    _save_cache(fingerprint)
                    _source = "built"
                _count = len(_cases)

            _error = None
            logger.info(
                "벡터DB 준비 완료 | backend=%s | 사례 %d건 (건너뜀 %d) | %s | %s",
                backend(), _count, skipped, _SOURCE_TEXT.get(_source, _source), path.name,
            )
        except Exception as exc:
            _vectors, _cases, _source, _count, _pg_ready = None, [], "", 0, False
            _error = f"{type(exc).__name__}: {getattr(exc, 'message', None) or exc}"
            logger.warning("벡터DB 를 사용할 수 없습니다 - 사례 보조 없이 진행합니다 | %s", _error)


_SOURCE_TEXT = {
    "cache": ".npy 캐시 사용",
    "built": "새로 임베딩",
    "pgvector": "DB 그대로 사용",
    "cache->pgvector": ".npy 를 DB 에 적재",
    "built->pgvector": "새로 임베딩해 DB 에 적재",
}


def _load_pgvector(rows: list[tuple[str, str]], fingerprint: str) -> None:
    """DB 지문이 같으면 그대로 쓰고, 다르면 .npy(없으면 새로 임베딩)로 다시 적재합니다."""
    global _vectors, _cases, _source, _count, _pg_ready

    db_fp, db_count = pg_store.read_state()
    if db_fp == fingerprint and db_count > 0:
        _source, _count = "pgvector", db_count
    else:
        if db_fp is not None:
            logger.info("DB 의 사례가 CSV 와 달라 다시 적재합니다. (table=%s)", pg_store.table_name())
        cached = _load_cache(fingerprint, len(rows))
        if cached is not None:
            vectors, cases = cached
            _source = "cache->pgvector"
        else:
            try:
                vectors, cases = _build(rows)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc} 이 PC 에서는 Colab 이 만든 data/cases.npy 와 data/cases.meta.json 을 "
                    "data 폴더에 복사하면 그대로 DB 에 적재합니다. (CSV 도 Colab 과 같은 파일이어야 합니다)"
                ) from None
            _vectors, _cases = vectors, cases
            _save_cache(fingerprint)
            _source = "built->pgvector"
        _count = pg_store.replace_all(vectors, cases, fingerprint)

    # 검색은 DB 에서 하므로 메모리에 벡터를 들고 있지 않습니다.
    _vectors, _cases, _pg_ready = None, [], True


def _load_cache(fingerprint: str, expected: int) -> tuple[np.ndarray, list[Case]] | None:
    vec_path, meta_path = _vector_path(), _meta_path()
    if not (vec_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            logger.info("사례 CSV 또는 임베딩 설정이 바뀌어 벡터DB 를 다시 만듭니다.")
            return None
        vectors = np.load(vec_path)
        cases = [Case(**c) for c in meta.get("cases", [])]
        if vectors.ndim != 2 or len(cases) != vectors.shape[0] or len(cases) != expected:
            return None
        return vectors.astype(np.float32, copy=False), cases
    except Exception as exc:
        logger.warning("벡터DB 캐시를 읽지 못해 다시 만듭니다 | %s", exc)
        return None


def _build(rows: list[tuple[str, str]]) -> tuple[np.ndarray, list[Case]]:
    if not embedder.is_installed():
        # 윈도우(①② 추출 전용) 환경이면 정상입니다. 벡터DB 는 모델 스택이 있는 쪽에서만 씁니다.
        raise RuntimeError(
            "sentence-transformers 미설치 - 벡터DB 는 모델 스택이 있는 환경(Colab/GPU 서버)에서 만들어집니다."
        )
    logger.info("벡터DB 생성 시작 | 사례 %d건 -> 키워드+요약 질의문 -> bge-m3", len(rows))
    started = time.perf_counter()
    cases = [
        Case(text=text, category=name, query_text=keyphrase.build_query(text).query_text)
        for text, name in rows
    ]
    vectors = embedder.encode([c.query_text for c in cases])
    logger.info("벡터DB 생성 완료 | shape=%s | %.1fs", tuple(vectors.shape), time.perf_counter() - started)
    return vectors, cases


def _save_cache(fingerprint: str) -> None:
    try:
        np.save(_vector_path(), _vectors)
        _meta_path().write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "embed_model": settings.EMBED_MODEL,
                    "count": len(_cases),
                    "cases": [c.__dict__ for c in _cases],
                },
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # 읽기 전용 폴더 등 - 메모리에는 올라와 있으므로 계속 진행
        logger.warning("벡터DB 캐시 파일을 쓰지 못했습니다 (다음 기동 때 다시 임베딩) | %s", exc)


def is_ready() -> bool:
    if backend() == "pgvector":
        return _pg_ready and _count > 0
    return _vectors is not None and bool(_cases)


# =============================================================================
# 조회 / 재정렬
# =============================================================================
def search_raw(query_vector: np.ndarray, top_k: int | None = None) -> list[tuple[str, str, float]]:
    """질의 벡터와 가장 비슷한 사례 top-k 를 (text, category, score) 로. 임계값을 적용하지 않습니다."""
    if not is_ready():
        return []
    k = min(top_k or settings.CASE_TOP_K, _count)
    if backend() == "pgvector":
        return [(t, c, float(sc)) for t, c, sc in pg_store.search(query_vector, k)]
    sims = embedder.cosine(query_vector, _vectors)[0]
    order = np.argsort(-sims)[:k]
    return [(_cases[int(i)].text, _cases[int(i)].category, float(sims[i])) for i in order]


def search(query_vector: np.ndarray, top_k: int | None = None) -> list[CaseHit]:
    """질의 벡터와 가장 비슷한 사례 top-k (CASE_MIN_SCORE 미만은 제외)."""
    hits = []
    for text, category, score in search_raw(query_vector, top_k):
        if score < settings.CASE_MIN_SCORE:
            break
        hits.append(CaseHit(len(hits) + 1, text, category, round(score, 4)))
    return hits


# =============================================================================
# 계산식 (순수 함수 - 운영과 측정 셀이 함께 씀)
# =============================================================================
def case_scores_from(
    hits: list[tuple[str, float]], top_k: int | None = None, min_score: float | None = None
) -> dict[str, float]:
    """
    (카테고리, 유사도) 목록 -> 카테고리별 사례 점수 (0~1).

    min_score 미만은 버리고, 합을 '통과한 사례 수' 가 아니라 k 로 나눕니다.
    """
    k = top_k or settings.CASE_TOP_K
    floor = settings.CASE_MIN_SCORE if min_score is None else min_score
    kept = [(c, s) for c, s in list(hits)[:k] if s >= floor]
    total: dict[str, float] = {}
    for category, score in kept:
        total[category] = total.get(category, 0.0) + score
    denom = max(k, len(kept), 1)
    return {name: round(v / denom, 4) for name, v in sorted(total.items(), key=lambda x: -x[1])}


def combine(
    base: list[tuple[str, float]],
    case_scores: dict[str, float],
    case_mode: str | None = None,
    weight: float | None = None,
    top_k: int | None = None,
) -> tuple[list[tuple[str, float]], str]:
    """
    ④ 점수(전체, 내림차순) + 사례 점수 -> 새 순서(전체) 와 설명 한 줄.

      섞기  : (1-w)*④ + w*사례 로 다시 정렬
      union : 섞은 뒤, 사례 점수 1위 카테고리가 top-k 밖이면 k 번째 자리에 끼워 넣음
    """
    w = min(max(settings.CASE_BLEND_WEIGHT if weight is None else weight, 0.0), 1.0)
    k = top_k or settings.CANDIDATE_TOP_K
    order = sorted(
        ((name, (1 - w) * score + w * case_scores.get(name, 0.0)) for name, score in base),
        key=lambda x: -x[1],
    )
    note = f"(1-{w:.2f})×④ + {w:.2f}×사례"
    if mode(case_mode) == "union" and case_scores:
        first = max(case_scores.items(), key=lambda x: x[1])[0]
        names = [n for n, _ in order]
        if first in names and names.index(first) >= k:
            item = order.pop(names.index(first))
            order.insert(k - 1, item)
            note += f" + 사례 1위 '{first}' 를 {k}위에 포함"
    return order, note

def rerank(cand: CandidateResult) -> tuple[CandidateResult, CaseLookup]:
    """
    ④ 결과가 애매할 때 사례 라벨로 후보를 재정렬합니다.

    호출하는 쪽(pipeline)이 should_consult() 일 때만 부릅니다.
    반영할 사례가 없으면 ④ 결과를 그대로 돌려줍니다.
    """
    lookup = CaseLookup(
        before=list(cand.top), before_all=list(cand.all_scores), after=list(cand.top)
    )

    if not _attempted:
        load_or_build()   # 기동 시 호출되지 않은 경우(노트북에서 pipeline.run 직접 호출 등)
    if not is_ready():
        lookup.reason = f"벡터DB 사용 불가 - {_error or '사례 없음'}"
        return cand, lookup
    if cand.query_vector is None:
        lookup.reason = "질의 벡터가 없어 조회하지 않음"
        return cand, lookup

    best = cand.best
    case_mode = mode()
    if case_mode != "low_confidence":
        lookup.reason = f"CASE_MODE={case_mode} 이라 매번 사례 조회"
    else:
        lookup.reason = (
            f"④ 1위 유사도 {best.score:.4f} < {settings.CANDIDATE_MIN_SCORE:.2f} 이라 사례 조회"
            if best else "④ 후보가 비어 있어 사례 조회"
        )
    try:
        lookup.hits = search(cand.query_vector)
    except Exception as exc:   # DB 접속 끊김 등 - 판정은 ④ 결과로 계속합니다
        logger.warning("벡터DB 검색 실패 - ④ 결과를 그대로 씁니다 | %s", exc)
        lookup.reason += f" - 검색 실패({type(exc).__name__}: {exc}), 반영하지 않음"
        return cand, lookup
    if not lookup.hits:
        lookup.reason += f" - 유사도 {settings.CASE_MIN_SCORE:.2f} 이상인 사례가 없어 반영하지 않음"
        return cand, lookup

    # 카테고리별 사례 점수 (0~1, 코사인과 같은 눈금). 통과한 사례 수가 아니라 k 로 나눕니다.
    used = len(lookup.hits)
    lookup.case_scores = case_scores_from([(h.category, h.score) for h in lookup.hits])

    k = len(cand.top) or settings.CANDIDATE_TOP_K
    order, note = combine([(c.name, c.score) for c in cand.all_scores], lookup.case_scores, top_k=k)
    all_scores = [
        CategoryScore(rank=i + 1, code=categories.BY_NAME[name].code, name=name, score=round(score, 4))
        for i, (name, score) in enumerate(order)
    ]
    top = all_scores[:k]

    lookup.used = True
    lookup.after = list(top)
    lookup.reason += f" - 사례 {used}건 반영 | {note}"

    return (
        CandidateResult(
            top=top,
            all_scores=all_scores,
            query_vector=cand.query_vector,
            low_confidence=cand.low_confidence,   # ④ 원래 판단을 그대로 표시
        ),
        lookup,
    )


def status() -> dict[str, object]:
    info = {
        "enabled": settings.CASE_STORE_ENABLED,
        "backend": backend(),
        "ready": is_ready(),
        "csv_path": str(csv_path()),
        "cases": _count,
        "skipped_rows": _skipped,
        "source": _source or None,
        "top_k": settings.CASE_TOP_K,
        "min_score": settings.CASE_MIN_SCORE,
        "blend_weight": settings.CASE_BLEND_WEIGHT,
        "error": _error,
    }
    if backend() == "pgvector":
        info["pg_table"] = settings.CASE_PG_TABLE
        info["pg_driver_installed"] = pg_store.installed()
    return info
