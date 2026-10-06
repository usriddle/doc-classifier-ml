"""
'문의' 답변용 FAQ 검색 (RAG).

'문의' 로 판정된 질문과 비슷한 FAQ 몇 개만 골라 답변 프롬프트에 넣습니다.
FAQ 가 수백 개로 늘어도 프롬프트에는 FAQ_TOP_K 개만 들어가므로 길이·속도가 그대로이고,
비슷한 FAQ 가 하나도 없으면 Gemma 를 부르지 않고 고정 안내 문구로 답합니다. (지어낸 답 방지)

  data/faq.csv        원본 (사람이 편집하는 파일)
                      컬럼 id, question, variants, answer, category, source, updated_at
                      필수는 question, answer. variants 는 같은 질문의 다른 표현을 | 로 구분.
  data/faq.npy        질문·다른 표현 벡터 (M, 1024) - 자동 생성
  data/faq.meta.json  FAQ 목록 + 벡터별 소속 FAQ + 지문(fingerprint) - 자동 생성

인덱스 : 서버 기동 시 load_or_build() 가 CSV 를 읽습니다. (case_store.py 와 같은 방식)
  - CSV 내용·임베딩 설정이 그대로면 .npy 만 읽습니다.
  - 바뀌었으면 다시 임베딩합니다. (bge-m3 는 ③ 과 같은 인스턴스를 씁니다)

점수 : FAQ 하나에 question + variants 의 벡터가 여러 개 있고, 그중 질문과 가장 가까운 값을
  그 FAQ 의 점수로 씁니다. (CANDIDATE_SCORING=multi 와 같은 원리 - 표현이 달라도 잘 걸림)
  ④ 와 달리 키워드·요약으로 압축하지 않고 질문 원문을 그대로 임베딩합니다.
  (문의는 보통 한 문장이라 압축하면 오히려 뜻이 흐려짐)

동작 (lookup)
  FAQ_ENABLED=true  : 상위 FAQ_TOP_K 개 중 FAQ_MIN_SCORE 이상만 프롬프트에 넣음
                      하나도 없으면 fallback=True -> 고정 문구 (FAQ_FALLBACK_MESSAGE)
  FAQ_ENABLED=false : 검색하지 않고 CSV 의 FAQ 전부를 넣음 (예전 방식. FAQ 가 적을 때 비교용)
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
from app.services import categories, embedder

logger = get_logger(__name__)

# 인덱스 만드는 방식(임베딩 대상 문장 구성 등)이 코드 수준에서 바뀌면 올려 주세요. 기존 .npy 를 무효화합니다.
_INDEX_VERSION = 1
VARIANT_SEP = "|"

_lock = threading.Lock()
_faqs: list["Faq"] = []
_vectors: np.ndarray | None = None
_owners: np.ndarray | None = None      # 벡터 i 가 속한 FAQ 의 인덱스
_attempted = False
_error: str | None = None
_source = ""                            # "cache" | "built" | ""
_skipped = 0


# =============================================================================
# 자료구조
# =============================================================================
@dataclass
class Faq:
    id: str
    question: str
    answer: str
    variants: list[str] = field(default_factory=list)
    category: str = ""          # 비어 있으면 공통
    source: str = ""
    updated_at: str = ""

    def texts(self) -> list[str]:
        """임베딩할 문장들 (질문 + 다른 표현)."""
        return [self.question] + [v for v in self.variants if v != self.question]


@dataclass
class FaqHit:
    rank: int
    id: str
    question: str
    answer: str
    score: float                # 질문과 가장 가까운 표현의 코사인 유사도
    matched: str = ""           # 가장 가까웠던 표현 (question 또는 variants 중 하나)


@dataclass
class FaqLookup:
    """FAQ 검색 결과. (DEBUG 확인용 + 답변 근거 기록)"""

    mode: str = "search"        # search | all | unavailable
    fallback: bool = False      # True 면 Gemma 를 부르지 않고 고정 문구로 답함
    reason: str = ""
    hits: list[FaqHit] = field(default_factory=list)       # 상위 FAQ_TOP_K (기준 미달 포함 - 디버그용)
    selected: list[FaqHit] = field(default_factory=list)   # 실제로 프롬프트에 넣은 FAQ
    best_score: float = 0.0
    min_score: float = 0.0
    model_declined: bool = False  # Gemma 가 "확인이 어렵다"고 답해 고정 문구로 바꿨는지
    model_answer: str = ""        # Gemma 가 실제로 생성한 답 (Gemma 를 부르지 않았으면 빈 문자열)


# =============================================================================
# 경로 / 지문
# =============================================================================
def csv_path() -> Path:
    path = Path(settings.FAQ_CSV_PATH)
    return path if path.is_absolute() else BASE_DIR / path


def _vector_path() -> Path:
    return csv_path().with_suffix(".npy")


def _meta_path() -> Path:
    return csv_path().with_suffix(".meta.json")


def _fingerprint(csv_bytes: bytes) -> str:
    parts = [
        hashlib.sha256(csv_bytes).hexdigest(),
        settings.EMBED_MODEL,
        str(settings.EMBED_NORMALIZE),
        str(settings.EMBED_MAX_SEQ_LENGTH),
        str(_INDEX_VERSION),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# =============================================================================
# CSV 읽기
# =============================================================================
def read_csv(raw: bytes) -> tuple[list[Faq], int]:
    """FAQ 목록과 건너뛴 행 수. (question 또는 answer 가 비면 건너뜀)"""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = {(f or "").strip().lower() for f in (reader.fieldnames or [])}
    if not {"question", "answer"} <= fields:
        raise ValueError(f"FAQ CSV 에 question, answer 컬럼이 필요합니다. (현재: {reader.fieldnames})")

    faqs: list[Faq] = []
    seen: set[str] = set()
    skipped = 0
    for line_no, row in enumerate(reader, start=2):
        row = {(k or "").strip().lower(): (v or "").strip() for k, v in row.items()}
        question, answer = row.get("question", ""), row.get("answer", "")
        if not question or not answer:
            skipped += 1
            logger.warning("FAQ CSV %d행 건너뜀 - question 또는 answer 가 비어 있음", line_no)
            continue

        faq_id = row.get("id") or f"faq_line{line_no}"
        if faq_id in seen:
            logger.warning("FAQ CSV %d행 - id %r 가 중복되어 %r 로 바꿉니다", line_no, faq_id, f"{faq_id}_{line_no}")
            faq_id = f"{faq_id}_{line_no}"
        seen.add(faq_id)

        variants: list[str] = []
        for v in (row.get("variants") or "").split(VARIANT_SEP):
            v = v.strip()
            if v and v != question and v not in variants:
                variants.append(v)

        label = row.get("category", "")
        category = ""
        if label:
            found = categories.BY_NAME.get(label) or categories.BY_CODE.get(label.lower())
            if found is None:
                logger.warning(
                    "FAQ CSV %d행 - 알 수 없는 카테고리 %r 는 비워 둡니다 (가능: %s)",
                    line_no, label, "/".join(categories.NAMES),
                )
            else:
                category = found.name

        faqs.append(Faq(
            id=faq_id, question=question, answer=answer, variants=variants, category=category,
            source=row.get("source", ""), updated_at=row.get("updated_at", ""),
        ))
    return faqs, skipped


# =============================================================================
# 로드 / 생성
# =============================================================================
def load_or_build() -> None:
    """서버 기동 시 호출. 실패해도 예외를 올리지 않고 status() 의 error 에 남깁니다."""
    global _faqs, _vectors, _owners, _attempted, _error, _source, _skipped

    with _lock:
        _attempted = True
        path = csv_path()
        try:
            if not path.exists():
                raise FileNotFoundError(f"FAQ CSV 가 없습니다: {path}")
            raw = path.read_bytes()
            faqs, skipped = read_csv(raw)
            if not faqs:
                raise ValueError("FAQ CSV 에 사용할 수 있는 행이 없습니다.")
            _faqs, _skipped = faqs, skipped
        except Exception as exc:
            _faqs, _vectors, _owners, _source = [], None, None, ""
            _error = f"{type(exc).__name__}: {exc}"
            logger.warning("FAQ 를 읽지 못했습니다 - '문의' 는 고정 안내 문구로 답합니다 | %s", _error)
            return

        if not settings.FAQ_ENABLED:
            # 검색하지 않으므로 임베딩도 만들지 않습니다. (전부 넣기 모드)
            _vectors, _owners, _source, _error = None, None, "", None
            logger.info("FAQ 준비 완료 | %d건 (건너뜀 %d) | FAQ_ENABLED=false - 검색 없이 전부 사용", len(faqs), skipped)
            return

        try:
            fingerprint = _fingerprint(raw)
            cached = _load_cache(fingerprint, faqs)
            if cached is not None:
                _vectors, _owners = cached
                _source = "cache"
            else:
                _vectors, _owners = _build(faqs)
                _save_cache(fingerprint)
                _source = "built"
            _error = None
            logger.info(
                "FAQ 준비 완료 | %d건 (건너뜀 %d) · 표현 %d개 | %s | %s",
                len(faqs), skipped, int(_vectors.shape[0]),
                ".npy 캐시 사용" if _source == "cache" else "새로 임베딩", path.name,
            )
        except Exception as exc:
            _vectors, _owners, _source = None, None, ""
            _error = f"{type(exc).__name__}: {exc}"
            logger.warning("FAQ 임베딩을 만들지 못했습니다 - 검색 없이 FAQ 전부를 씁니다 | %s", _error)


def _build(faqs: list[Faq]) -> tuple[np.ndarray, np.ndarray]:
    if not embedder.is_installed():
        raise RuntimeError("sentence-transformers 미설치 - FAQ 검색은 모델 스택이 있는 환경에서만 됩니다.")
    texts: list[str] = []
    owners: list[int] = []
    for i, faq in enumerate(faqs):
        for t in faq.texts():
            texts.append(t)
            owners.append(i)
    logger.info("FAQ 임베딩 시작 | FAQ %d건 · 표현 %d개 -> bge-m3", len(faqs), len(texts))
    started = time.perf_counter()
    vectors = embedder.encode(texts)
    logger.info("FAQ 임베딩 완료 | shape=%s | %.1fs", tuple(vectors.shape), time.perf_counter() - started)
    return vectors.astype(np.float32, copy=False), np.asarray(owners, dtype=np.int32)


def _load_cache(fingerprint: str, faqs: list[Faq]) -> tuple[np.ndarray, np.ndarray] | None:
    vec_path, meta_path = _vector_path(), _meta_path()
    if not (vec_path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("fingerprint") != fingerprint:
            logger.info("FAQ CSV 또는 임베딩 설정이 바뀌어 FAQ 인덱스를 다시 만듭니다.")
            return None
        vectors = np.load(vec_path)
        owners = np.asarray(meta.get("owners", []), dtype=np.int32)
        expected = sum(len(f.texts()) for f in faqs)
        if vectors.ndim != 2 or vectors.shape[0] != expected or owners.shape[0] != expected:
            return None
        return vectors.astype(np.float32, copy=False), owners
    except Exception as exc:
        logger.warning("FAQ 캐시를 읽지 못해 다시 만듭니다 | %s", exc)
        return None


def _save_cache(fingerprint: str) -> None:
    try:
        np.save(_vector_path(), _vectors)
        _meta_path().write_text(
            json.dumps(
                {
                    "fingerprint": fingerprint,
                    "embed_model": settings.EMBED_MODEL,
                    "count": len(_faqs),
                    "owners": [int(o) for o in (_owners if _owners is not None else [])],
                    "faqs": [f.__dict__ for f in _faqs],
                },
                ensure_ascii=False, indent=1,
            ),
            encoding="utf-8",
        )
    except Exception as exc:  # 읽기 전용 폴더 등 - 메모리에는 올라와 있으므로 계속 진행
        logger.warning("FAQ 캐시 파일을 쓰지 못했습니다 (다음 기동 때 다시 임베딩) | %s", exc)


def is_ready() -> bool:
    """검색할 수 있는 상태인지 (FAQ 와 벡터가 모두 있음)."""
    return bool(_faqs) and _vectors is not None and _owners is not None


def entries() -> list[Faq]:
    if not _attempted:
        load_or_build()
    return list(_faqs)


# =============================================================================
# 검색
# =============================================================================
def rank_faqs(sims: np.ndarray, owners: np.ndarray, faq_count: int) -> list[tuple[int, float, int]]:
    """
    표현별 유사도 -> FAQ 별 최고 점수로 정렬. (순수 함수 - 측정 코드도 같이 씀)

    반환: [(FAQ 인덱스, 점수, 가장 가까웠던 표현의 벡터 인덱스), ...] 점수 내림차순
    """
    best = np.full(faq_count, -np.inf, dtype=np.float32)
    best_vec = np.full(faq_count, -1, dtype=np.int64)
    for vec_idx, (owner, score) in enumerate(zip(owners.tolist(), sims.tolist())):
        if score > best[owner]:
            best[owner] = score
            best_vec[owner] = vec_idx
    order = np.argsort(-best)
    return [(int(i), float(best[i]), int(best_vec[i])) for i in order if best_vec[i] >= 0]


def _matched_text(vec_idx: int) -> str:
    """벡터 인덱스 -> 그 벡터의 원래 문장."""
    if _owners is None:
        return ""
    owner = int(_owners[vec_idx])
    first = int(np.argmax(_owners == owner))     # 그 FAQ 의 첫 벡터 위치
    texts = _faqs[owner].texts()
    offset = vec_idx - first
    return texts[offset] if 0 <= offset < len(texts) else ""


def _all_as_hits(faqs: list[Faq]) -> list[FaqHit]:
    return [FaqHit(rank=i + 1, id=f.id, question=f.question, answer=f.answer, score=0.0) for i, f in enumerate(faqs)]


def lookup(question: str) -> FaqLookup:
    """질문과 비슷한 FAQ 를 고릅니다. 실패해도 예외를 올리지 않습니다. (fallback 으로 처리)"""
    if not _attempted:
        load_or_build()

    min_score = float(settings.FAQ_MIN_SCORE)
    result = FaqLookup(min_score=min_score)

    if not _faqs:
        result.mode, result.fallback = "unavailable", True
        result.reason = f"FAQ 를 사용할 수 없어 고정 문구로 답함 - {_error or 'FAQ 없음'}"
        return result

    if not settings.FAQ_ENABLED or not is_ready():
        # 검색 없이 전부 넣기. FAQ_ENABLED=false 이거나, 임베딩을 못 만든 경우
        result.mode = "all"
        result.selected = _all_as_hits(_faqs)
        result.reason = (
            "FAQ_ENABLED=false - 검색 없이 FAQ 전부를 넣음" if not settings.FAQ_ENABLED
            else f"FAQ 검색 불가({_error}) - FAQ 전부를 넣음"
        )
        return result

    try:
        text = (question or "").strip()[: settings.FAQ_QUERY_MAX_CHARS]
        query = embedder.encode_one(text)
        sims = embedder.cosine(query, _vectors)[0]
    except Exception as exc:
        logger.warning("FAQ 검색 실패 - FAQ 전부를 넣습니다 | %s", exc)
        result.mode = "all"
        result.selected = _all_as_hits(_faqs)
        result.reason = f"FAQ 검색 실패({type(exc).__name__}: {exc}) - FAQ 전부를 넣음"
        return result

    ranked = rank_faqs(sims, _owners, len(_faqs))[: max(1, settings.FAQ_TOP_K)]
    for i, (faq_idx, score, vec_idx) in enumerate(ranked):
        faq = _faqs[faq_idx]
        result.hits.append(FaqHit(
            rank=i + 1, id=faq.id, question=faq.question, answer=faq.answer,
            score=round(score, 4), matched=_matched_text(vec_idx),
        ))
    result.selected = [h for h in result.hits if h.score >= min_score]
    result.best_score = result.hits[0].score if result.hits else 0.0

    if result.selected:
        result.reason = (
            f"1위 {result.best_score:.4f} ≥ 기준 {min_score:.2f} - "
            f"FAQ {len(result.selected)}개를 프롬프트에 넣음"
        )
    else:
        result.fallback = True
        result.reason = (
            f"1위 {result.best_score:.4f} < 기준 {min_score:.2f} - 비슷한 FAQ 가 없어 "
            "Gemma 를 부르지 않고 고정 문구로 답함"
        )
    return result


def status() -> dict[str, object]:
    return {
        "enabled": settings.FAQ_ENABLED,
        "ready": is_ready(),
        "csv_path": str(csv_path()),
        "faqs": len(_faqs),
        "vectors": int(_vectors.shape[0]) if _vectors is not None else 0,
        "skipped_rows": _skipped,
        "source": _source or None,
        "top_k": settings.FAQ_TOP_K,
        "min_score": settings.FAQ_MIN_SCORE,
        "error": _error,
    }
