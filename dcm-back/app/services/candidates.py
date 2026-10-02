"""
④ 후보 추림 - 코사인 유사도 top-4. (개수는 .env 의 CANDIDATE_TOP_K)

③ 에서 만든 질의 벡터(키워드3개+요약문)와 7종 카테고리 정의문 벡터의
코사인 유사도를 계산해 상위 4개를 고릅니다. 학습하지 않습니다.

여기서 고른 4개만 ⑤ Qwen 에게 넘어가고, Qwen 이 그 중 하나를 확정합니다.
(7종 전체를 넘기지 않으므로 프롬프트가 짧아지고 판정이 안정적입니다)

카테고리 벡터는 최초 1회 계산해 메모리에 캐시합니다.

점수 계산 방식 (.env 의 CANDIDATE_SCORING)
  single : 카테고리마다 정의문(설명 + 키워드 전체) 벡터 1개와 비교합니다.
           주제가 많은 카테고리(행정안전 = 서류·재난·방범·과태료·선거 …)는 벡터가 여러 주제의
           평균이 되어 어느 주제와도 가깝지 않게 되는 약점이 있습니다.
  multi  : 정의문 벡터 + 키워드 하나하나의 벡터 중 가장 가까운 것의 점수를 씁니다.
           "방범 CCTV" 같은 개별 주제와 직접 비교하므로 평균 효과가 줄어듭니다.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.services import categories, embedder
from app.services.categories import Category

logger = get_logger(__name__)

_category_vectors: np.ndarray | None = None
_term_vectors: np.ndarray | None = None   # multi 용 키워드 벡터 (T, d)
_term_owner: np.ndarray | None = None     # 각 키워드가 속한 카테고리 인덱스 (T,)
_lock = threading.Lock()


@dataclass
class CategoryScore:
    rank: int
    code: str
    name: str
    score: float

    @property
    def category(self) -> Category:
        return categories.get(self.code)


@dataclass
class CandidateResult:
    """④ 결과."""

    top: list[CategoryScore] = field(default_factory=list)      # 상위 CANDIDATE_TOP_K 개 (Qwen 입력)
    all_scores: list[CategoryScore] = field(default_factory=list)  # 7종 전체 (DEBUG 확인용)
    query_vector: np.ndarray | None = None
    low_confidence: bool = False   # 1위 점수가 임계값 미만. 참고용 표시일 뿐 반려 기준 아님

    @property
    def best(self) -> CategoryScore | None:
        return self.top[0] if self.top else None

    @property
    def margin(self) -> float:
        """1위와 2위의 점수 차이. 작을수록 애매한 요청입니다."""
        if len(self.top) < 2:
            return 1.0
        return round(self.top[0].score - self.top[1].score, 4)


def _get_category_vectors() -> np.ndarray:
    """7종 카테고리 정의문 임베딩 (캐시)."""
    global _category_vectors

    if _category_vectors is not None:
        return _category_vectors

    with _lock:
        if _category_vectors is not None:
            return _category_vectors
        texts = [c.embedding_text for c in categories.CATEGORIES]
        logger.info("카테고리 정의문 %d종 임베딩 시작", len(texts))
        _category_vectors = embedder.encode(texts)
        logger.info(
            "카테고리 정의문 임베딩 완료 | shape=%s", tuple(_category_vectors.shape)
        )
        return _category_vectors


def _get_term_vectors() -> tuple[np.ndarray, np.ndarray]:
    """multi 용 - 카테고리 키워드 하나하나의 임베딩 (캐시)."""
    global _term_vectors, _term_owner

    if _term_vectors is not None and _term_owner is not None:
        return _term_vectors, _term_owner

    with _lock:
        if _term_vectors is not None and _term_owner is not None:
            return _term_vectors, _term_owner
        texts, owner = [], []
        for idx, c in enumerate(categories.CATEGORIES):
            for kw in c.keywords:
                texts.append(kw)
                owner.append(idx)
        logger.info("카테고리 키워드 %d개 임베딩 시작 (CANDIDATE_SCORING=multi)", len(texts))
        _term_vectors = embedder.encode(texts)
        _term_owner = np.asarray(owner, dtype=np.int64)
        return _term_vectors, _term_owner


def scoring_mode(value: str | None = None) -> str:
    v = (value if value is not None else settings.CANDIDATE_SCORING or "single").strip().lower()
    return "multi" if v == "multi" else "single"


def score_matrix(query_vectors: np.ndarray, scoring: str | None = None) -> np.ndarray:
    """
    질의 벡터 (N, d) -> 카테고리 점수 (N, 7). categories.CATEGORIES 순서.

    운영(select)과 학습 노트북의 측정 셀(training/calibrate.py)이 같은 함수를 씁니다.
    """
    q = np.atleast_2d(query_vectors)
    sims = embedder.cosine(q, _get_category_vectors())
    if scoring_mode(scoring) == "single":
        return sims
    term_vectors, owner = _get_term_vectors()
    term_sims = embedder.cosine(q, term_vectors)          # (N, T)
    best = sims.copy()
    for idx in range(len(categories.CATEGORIES)):
        cols = owner == idx
        if cols.any():
            best[:, idx] = np.maximum(best[:, idx], term_sims[:, cols].max(axis=1))
    return best


def reset_cache() -> None:
    """카테고리 정의를 바꾼 뒤 벡터를 다시 계산하고 싶을 때 사용합니다."""
    global _category_vectors, _term_vectors, _term_owner
    with _lock:
        _category_vectors = None
        _term_vectors = None
        _term_owner = None


def select(query_text: str) -> CandidateResult:
    """
    질의문 -> 카테고리 후보 top-k (CANDIDATE_TOP_K).

    query_text 는 keyphrase.build_query() 가 만든 "키워드: ...\\n요약: ..." 문자열입니다.
    """
    query_vector = embedder.encode_one(query_text)
    sims = score_matrix(query_vector)[0]

    order = np.argsort(-sims)
    all_scores = [
        CategoryScore(
            rank=rank + 1,
            code=categories.CATEGORIES[idx].code,
            name=categories.CATEGORIES[idx].name,
            score=round(float(sims[idx]), 4),
        )
        for rank, idx in enumerate(order)
    ]

    top_k = min(settings.CANDIDATE_TOP_K, len(all_scores))
    top = all_scores[:top_k]

    low_confidence = bool(top) and top[0].score < settings.CANDIDATE_MIN_SCORE
    if low_confidence:
        # 참고용 로그일 뿐입니다. 반려 여부는 ⑤ 의 의도 판정(해당없음)으로 정해지며
        # 여기(④ 카테고리 유사도)와는 무관합니다. "제가 어제 문의한 내용 보여줘"처럼
        # 카테고리가 애매해도 의도가 명확하면 정상 통과합니다.
        logger.info(
            "카테고리 유사도 낮음(참고용) | top1=%s(%.4f) < %.2f",
            top[0].name, top[0].score, settings.CANDIDATE_MIN_SCORE,
        )

    return CandidateResult(
        top=top,
        all_scores=all_scores,
        query_vector=query_vector,
        low_confidence=low_confidence,
    )
