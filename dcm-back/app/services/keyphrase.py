"""
②→③ 전처리 : 추출 원문 -> "키워드 3개 + 요약문".

추출된 원문을 그대로 임베딩하면 문서가 길수록 주제가 희석되므로,
임베딩 모델에 넘기기 전에 핵심만 남긴 짧은 질의문으로 압축합니다.

  1) kiwipiepy  : 한국어 형태소 분석 -> 문장 분리 + 명사구 후보 추출
  2) KeyBERT    : ③ 의 bge-m3 를 그대로 재사용해 후보 중 키워드 3개 선정
  3) TextRank   : 문장 임베딩 그래프에서 중심 문장을 골라 추출 요약

세 단계 모두 추가 모델 다운로드가 없습니다.
(KeyBERT 와 TextRank 가 embedder.py 의 bge-m3 인스턴스를 공유합니다)

KeyBERT / kiwipiepy 가 없거나 실패하면 빈도 기반 폴백으로 내려가며,
서버가 멈추지는 않습니다. (warnings 에 사유가 기록됩니다)
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field

import numpy as np

from app.config import settings
from app.logging_config import get_logger
from app.services import embedder

logger = get_logger(__name__)

# --- kiwipiepy 명사 계열 태그 -------------------------------------------
# NNG 일반명사 / NNP 고유명사 / SL 외국어 / SH 한자 / SN 숫자
_NOUN_TAGS = {"NNG", "NNP", "SL", "SH"}
_NOUN_TAGS_WITH_NUM = _NOUN_TAGS | {"SN"}

# 키워드로 뽑혀도 의미가 없는 말들
_STOPWORDS = {
    "관련", "사항", "내용", "경우", "정도", "생각", "때문", "대하", "위하", "통하",
    "민원", "신청", "처리", "문의", "요청", "대상", "결과", "확인", "발생", "필요",
    "가능", "제출", "안내", "부분", "이상", "이하", "등등", "기타", "오늘", "어제",
    "지금", "우리", "그것", "이것", "저것", "본인", "여기", "거기", "저기",
}

_kiwi = None
_kiwi_lock = threading.Lock()
_kiwi_error: str | None = None

_keybert = None
_keybert_lock = threading.Lock()


# =============================================================================
# 결과 자료구조
# =============================================================================
@dataclass
class Keyword:
    text: str
    score: float


@dataclass
class KeyphraseResult:
    """②→③ 전처리 결과. `query_text` 가 임베딩 모델에 넘어갑니다."""

    keywords: list[Keyword] = field(default_factory=list)
    summary: str = ""
    query_text: str = ""
    sentences: list[str] = field(default_factory=list)
    method: str = ""                      # 실제로 사용한 경로 (keybert / frequency 등)
    warnings: list[str] = field(default_factory=list)

    @property
    def keyword_texts(self) -> list[str]:
        return [k.text for k in self.keywords]


# =============================================================================
# kiwipiepy
# =============================================================================
def kiwi_installed() -> bool:
    try:
        import kiwipiepy  # noqa: F401
    except Exception:
        return False
    return True


def _get_kiwi():
    """Kiwi 형태소 분석기 싱글턴. 실패하면 None 을 돌려줍니다."""
    global _kiwi, _kiwi_error

    if _kiwi is not None or _kiwi_error is not None:
        return _kiwi

    with _kiwi_lock:
        if _kiwi is not None or _kiwi_error is not None:
            return _kiwi
        try:
            from kiwipiepy import Kiwi

            _kiwi = Kiwi()
            logger.info("kiwipiepy 형태소 분석기 로드 완료")
        except Exception as exc:
            _kiwi_error = f"{type(exc).__name__}: {exc}"
            logger.warning("kiwipiepy 로드 실패 - 정규식 폴백을 사용합니다 | %s", _kiwi_error)
        return _kiwi


# =============================================================================
# 텍스트 정리 / 문장 분리
# =============================================================================
def normalize(text: str) -> str:
    """추출 원문에서 과도한 공백·제어문자를 정리합니다."""
    if not text:
        return ""
    text = text.replace("​", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """문장 분리. kiwipiepy 가 있으면 그걸 쓰고, 없으면 정규식으로 자릅니다."""
    kiwi = _get_kiwi()
    if kiwi is not None:
        try:
            sents = [s.text.strip() for s in kiwi.split_into_sents(text)]
            sents = [s for s in sents if s]
            if sents:
                return sents
        except Exception as exc:
            logger.warning("kiwi 문장 분리 실패 - 정규식 폴백 | %s", exc)

    parts = re.split(r"(?<=[.!?。？！])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


# =============================================================================
# 명사구 후보 추출
# =============================================================================
def extract_noun_phrases(text: str) -> list[str]:
    """
    KeyBERT 에 넘길 키워드 후보를 만듭니다.

    연속된 명사는 하나의 복합명사로 묶습니다. (예: 도로 + 포장 + 파손 -> '도로 포장 파손')
    단일 명사도 함께 후보에 넣어 짧은 민원 문장에서도 후보가 비지 않게 합니다.
    """
    kiwi = _get_kiwi()
    if kiwi is None:
        return _regex_noun_phrases(text)

    try:
        tokens = kiwi.tokenize(text)
    except Exception as exc:
        logger.warning("kiwi 형태소 분석 실패 - 정규식 폴백 | %s", exc)
        return _regex_noun_phrases(text)

    phrases: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if not buffer:
            return
        if len(buffer) > 1:
            phrases.append(" ".join(buffer))
        buffer.clear()

    for token in tokens:
        form, tag = token.form, token.tag
        if tag in _NOUN_TAGS_WITH_NUM:
            if tag == "SN" and not buffer:
                # 숫자 단독은 키워드로 쓰지 않습니다.
                continue
            if len(form) >= 2 and form not in _STOPWORDS:
                phrases.append(form)
            buffer.append(form)
        else:
            flush()
    flush()

    return _dedupe(phrases)


# 정규식 폴백에서 걸러낼 용언 활용형 어미. ("부탁드립니다", "위험합니다" 등)
_VERB_TAIL = re.compile(
    r"(습니다|입니다|합니다|됩니다|드립니다|했습니다|하세요|해주세요|했어요|"
    r"이에요|예요|어요|네요|해요|이다|하다|되다|있다|없다)$"
)


def _regex_noun_phrases(text: str) -> list[str]:
    """
    kiwipiepy 가 없을 때 쓰는 최소 폴백.

    형태소 분석이 없으므로 용언 활용형을 어미 패턴으로만 걸러냅니다.
    품질이 떨어지니 실제 운영에서는 kiwipiepy 를 설치하세요.
    """
    words = re.findall(r"[가-힣]{2,}|[A-Za-z][A-Za-z0-9]{2,}", text)
    return _dedupe(
        w for w in words if w not in _STOPWORDS and not _VERB_TAIL.search(w)
    )


def _dedupe(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


# =============================================================================
# 키워드 선정 (KeyBERT + bge-m3)
# =============================================================================
def keybert_installed() -> bool:
    try:
        import keybert  # noqa: F401
    except Exception:
        return False
    return True


def _get_keybert():
    """
    KeyBERT 싱글턴.

    embedder.get_model() 로 이미 로드된 bge-m3 인스턴스를 그대로 주입하므로
    KeyBERT 가 별도 모델(기본값 all-MiniLM)을 내려받지 않습니다.
    """
    global _keybert
    if _keybert is not None:
        return _keybert

    with _keybert_lock:
        if _keybert is not None:
            return _keybert
        from keybert import KeyBERT

        _keybert = KeyBERT(model=embedder.get_model())
        logger.info("KeyBERT 초기화 완료 (백본: %s)", settings.EMBED_MODEL)
        return _keybert


def reset_keybert() -> None:
    """KeyBERT 인스턴스를 비웁니다. (embedder.unload() 가 부름)"""
    global _keybert
    with _keybert_lock:
        _keybert = None


def _keywords_by_keybert(text: str, candidates: list[str], top_k: int) -> list[Keyword]:
    kw_model = _get_keybert()
    pairs = kw_model.extract_keywords(
        text,
        candidates=candidates,
        top_n=top_k,
        use_mmr=settings.KEYWORD_USE_MMR,
        diversity=settings.KEYWORD_MMR_DIVERSITY,
    )
    return [Keyword(text=str(word), score=round(float(score), 4)) for word, score in pairs]


def _keywords_by_frequency(text: str, candidates: list[str], top_k: int) -> list[Keyword]:
    """KeyBERT 를 쓸 수 없을 때의 폴백. 등장 빈도 x 길이 가중치로 고릅니다."""
    scored: list[tuple[str, float]] = []
    for cand in candidates:
        count = text.count(cand)
        if count == 0:
            continue
        # 긴 복합명사에 가중치를 주어 '도로'보다 '도로 포장 파손'이 먼저 오게 합니다.
        scored.append((cand, count * (1.0 + 0.3 * cand.count(" ")) * min(len(cand), 8) / 8))

    scored.sort(key=lambda x: x[1], reverse=True)
    if not scored:
        return []
    top = max(scored[0][1], 1e-6)
    return [Keyword(text=w, score=round(s / top, 4)) for w, s in scored[:top_k]]


# =============================================================================
# 요약 (문장 임베딩 TextRank)
# =============================================================================
def _textrank(sentences: list[str], top_k: int) -> list[int]:
    """
    문장 임베딩 유사도 그래프에 PageRank 를 돌려 중심 문장 인덱스를 고릅니다.

    반환값은 원문 순서대로 정렬된 인덱스 목록입니다.
    """
    vectors = embedder.encode(sentences)
    sim = embedder.cosine(vectors, vectors)

    np.fill_diagonal(sim, 0.0)
    sim = np.clip(sim, 0.0, None)

    row_sum = sim.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    transition = sim / row_sum

    n = len(sentences)
    damping = settings.SUMMARY_DAMPING
    scores = np.full(n, 1.0 / n, dtype=np.float32)
    for _ in range(settings.SUMMARY_ITERATIONS):
        updated = (1.0 - damping) / n + damping * (transition.T @ scores)
        if np.allclose(updated, scores, atol=1e-6):
            scores = updated
            break
        scores = updated

    picked = np.argsort(-scores)[:top_k]
    return sorted(int(i) for i in picked)


def _summarize(sentences: list[str], warnings: list[str]) -> str:
    """추출 요약. 문장이 적으면 원문을 그대로 사용합니다."""
    max_sents = settings.SUMMARY_MAX_SENTENCES

    if not sentences:
        return ""
    if len(sentences) <= max_sents:
        # 짧은 민원(1~3문장)은 이미 요약문이나 마찬가지입니다.
        return " ".join(sentences)[: settings.SUMMARY_MAX_CHARS]

    try:
        picked = _textrank(sentences, max_sents)
    except Exception as exc:
        warnings.append(f"TextRank 요약 실패 - 앞 문장으로 대체: {type(exc).__name__}")
        logger.warning("TextRank 요약 실패 | %s", exc)
        picked = list(range(max_sents))

    summary = " ".join(sentences[i] for i in picked)
    return summary[: settings.SUMMARY_MAX_CHARS]


# =============================================================================
# 공개 API
# =============================================================================
def build_query(raw_text: str) -> KeyphraseResult:
    """
    추출 원문 -> 키워드 3개 + 요약문 -> 임베딩용 질의문(query_text).

    이 함수의 `query_text` 가 ③ bge-m3 의 입력이 됩니다.
    """
    warnings: list[str] = []
    text = normalize(raw_text)

    if not text:
        return KeyphraseResult(method="empty", warnings=["원문이 비어 있습니다."])

    limit = settings.PIPELINE_MAX_INPUT_CHARS
    if len(text) > limit:
        warnings.append(f"원문이 길어 앞 {limit}자만 사용했습니다. (전체 {len(text)}자)")
        text = text[:limit]

    sentences = split_sentences(text)
    candidates = extract_noun_phrases(text)
    top_k = settings.KEYWORD_TOP_K

    # --- 키워드 3개 ---
    keywords: list[Keyword] = []
    method = "keybert"
    if not candidates:
        method = "none"
        warnings.append("명사구 후보를 찾지 못했습니다.")
    elif keybert_installed():
        try:
            keywords = _keywords_by_keybert(text, candidates, top_k)
        except Exception as exc:
            method = "frequency"
            warnings.append(f"KeyBERT 실패 - 빈도 기반으로 대체: {type(exc).__name__}")
            logger.warning("KeyBERT 키워드 추출 실패 | %s", exc)
            keywords = _keywords_by_frequency(text, candidates, top_k)
    else:
        method = "frequency"
        warnings.append("keybert 미설치 - 빈도 기반 키워드를 사용했습니다.")
        keywords = _keywords_by_frequency(text, candidates, top_k)

    if not keywords and candidates:
        keywords = _keywords_by_frequency(text, candidates, top_k)

    # --- 요약문 ---
    summary = _summarize(sentences, warnings)

    # --- 임베딩에 넘길 질의문 ---
    keyword_line = ", ".join(k.text for k in keywords) if keywords else "(없음)"
    query_text = f"키워드: {keyword_line}\n요약: {summary}".strip()

    if not kiwi_installed():
        warnings.append("kiwipiepy 미설치 - 정규식 기반 후보 추출을 사용했습니다.")

    return KeyphraseResult(
        keywords=keywords,
        summary=summary,
        query_text=query_text,
        sentences=sentences,
        method=method,
        warnings=warnings,
    )


def status() -> dict[str, object]:
    return {
        "kiwipiepy": kiwi_installed(),
        "keybert": keybert_installed(),
        "kiwi_error": _kiwi_error,
    }
