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

상투 문구 거르기 (QUERY_DROP_BOILERPLATE, 기본 켬)
  "벌써 여러 번 말씀드렸어요", "신속한 처리 부탁드립니다", "주민들 불편이 너무 커요" 처럼
  어느 카테고리에나 붙는 문장은 ④ 에는 잡음입니다. 짧은 민원일수록 키워드·요약이 이런 말에 끌려가
  핵심어(벌집, 소화기 …)가 흐려집니다. 그래서 질의문을 만들 때만 걸러냅니다.
    1) 정해 둔 문구 패턴 (_BOILERPLATE)
    2) 명사가 없거나 일반 명사(_GENERIC_NOUNS)뿐인 짧은 문장
    안전장치: 카테고리 키워드가 들어간 문장은 남김 / 전부 걸리면 원문 그대로
  ⑤ Qwen 입력·원문 저장은 바꾸지 않습니다. (이 모듈의 query_text 에만 적용)

요약 기준 (SUMMARY_FULL_TEXT_CHARS, SUMMARY_KEEP_FIRST)
  상투 문구를 뺀 본문이 짧으면(기본 200자 이하) 요약하지 않고 전부 씁니다.
  길어서 TextRank 로 고를 때도 첫 문장(대개 핵심)은 항상 넣습니다.
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
    dropped: list[str] = field(default_factory=list)   # 질의문에서 뺀 상투 문장
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
# 상투 문구 거르기 (④ 질의문 전용)
# =============================================================================
# 질의문 만드는 방식이 바뀌면 올립니다. (벡터DB·후보 캐시 지문에 들어감)
QUERY_VERSION = 2

# 문장 전체가 이런 말이면 카테고리 정보가 없습니다. (공백을 지운 문장에 대해 검사)
_BOILERPLATE = tuple(re.compile(p) for p in (
    # 인사·형식
    r"^(안녕하세요|안녕하십니까|수고(가)?많으십니다|수고하십니다|문의드립니다|민원(을)?넣습니다|민원드립니다|급해서요|죄송한데|저기요)$",
    r"어디에물어봐야할지몰라서",
    r"^민원관련해서요",
    # 요청
    r"^(빠른|신속한|조속한)?(처리|조치|해결|확인|답변|검토)(를|좀)?(부탁|요청|바랍|해주|해줘|해요|요망|좀)",
    r"^(빨리|얼른|제발|꼭|좀)*(처리|조치|해결|확인|조사|검토)(좀)?(해|하여|해서)?(주세요|주십시오|줘요|줘|주시면|달라)",
    r"^빨리좀요?$",
    r"^(빨리|얼른|제발|꼭)?(좀)?(해|고쳐|치워|와)(주세요|주십시오|줘요|줘)$",
    r"^(부탁|부탁드립니다|부탁드려요|부탁합니다)",
    # 반복·경과
    r"(벌써|이미)?(여러|몇|수차례|몇번이나)번?(이나)?(말씀|신고|민원|얘기|연락)(을|를)?(드렸|했|넣었|드려도|해도)",
    r"^(며칠|몇주|한달|몇달|일주일|보름)(째|이나|동안)?(그대로|이상태|방치)",
    r"(확인|처리|조치)(이|가)?너무늦어",
    r"^(아직도|여전히)(그대로|해결이안|처리가안)",
    # 감정·상황
    r"^(진짜|정말|너무)*(너무하네요|화나요|미치겠|짜증나|답답해|무서워요|불안해요|걱정돼요|걱정이에요|힘들어요)",
    r"(주민|사람|아이)들?(이|의)?(불편|피해|걱정)(이|가)?(너무|많이|커|심해)",
    r"^지나갈때마다(신경|불안|무서)",
    r"^신경(이)?쓰여요",
    r"^사진(도|을)?찍어(두|놨|놓)",
    # 문서 잔해
    r"^[-=_~·.\s]{3,}$",
    r"신청인\[?이름\]?",
    r"본문서는스캔본",
))

# 이것만으로는 주제를 알 수 없는 명사. 키워드 후보에서도 뺍니다.
_GENERIC_NOUNS = {
    "말씀", "처리", "조치", "확인", "해결", "부탁", "요청", "답변", "연락", "검토", "조사",
    "불편", "피해", "걱정", "주민", "사람", "아이", "사진", "번", "며칠", "일주일", "보름", "달",
    "민원", "신고", "상태", "정도", "문제", "상황", "부분", "생각", "마음", "시간", "하루", "매일",
    "진짜", "정말", "지금", "오늘", "요즘", "계속", "신속", "빨리", "여기", "거기", "동네", "우리",
    "때", "뭐", "좀", "이상", "그대로", "방치", "수차례", "여러", "몇", "제발", "답답", "짜증",
}

# 한국어 조사 (정규식 폴백에서 명사 끝에 붙은 것 떼기)
_JOSA = re.compile(r"(들|이|가|은|는|을|를|에|에서|의|도|만|과|와|로|으로|께서|까지|부터|이나|나)+$")

# "~예요/~이에요/~입니다" (장소 이름만 말하는 문장)
_COPULA = re.compile(r"(이에요|예요|이요|입니다|이랍니다|랍니다)$")

_protected: set[str] | None = None


def _protected_terms() -> set[str]:
    """카테고리 키워드와 그 낱말들. 이게 들어간 문장은 상투 문구로 보지 않습니다."""
    global _protected
    if _protected is None:
        from app.services import categories   # 순환 import 피하기

        terms: set[str] = set()
        for cat in categories.CATEGORIES:
            for kw in cat.keywords:
                terms.add(kw.replace(" ", ""))
                terms.update(w for w in kw.split() if len(w) >= 2 and w not in _GENERIC_NOUNS)
        _protected = terms
    return _protected


def _content_nouns(sentence: str) -> list[str]:
    """문장 속 명사 중 일반 명사가 아닌 것."""
    kiwi = _get_kiwi()
    nouns: list[str] = []
    if kiwi is not None:
        try:
            nouns = [t.form for t in kiwi.tokenize(sentence) if t.tag in _NOUN_TAGS]
        except Exception:
            nouns = []
    else:
        for w in re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", sentence):
            stem = _COPULA.sub("", w)          # "새봄삼거리예요" -> "새봄삼거리"
            if stem != w and len(stem) >= 2:
                nouns.append(stem)
                continue
            if _VERB_TAIL.search(w):
                continue
            nouns.append(_JOSA.sub("", w) or w)
    return [n for n in nouns if n not in _GENERIC_NOUNS and n not in _STOPWORDS]


def is_boilerplate(sentence: str) -> bool:
    """카테고리를 가르는 정보가 없는 상투 문장인지."""
    s = sentence.strip()
    if not s:
        return True
    squashed = re.sub(r"[\s!?.~,ㅠㅜ]+", "", s)
    if any(term in squashed for term in _protected_terms()):
        return False
    if any(p.search(squashed) for p in _BOILERPLATE):
        return True
    # 짧은 문장인데 주제가 될 명사가 없음 ("며칠째 그대로예요", "진짜 미치겠어요")
    return len(squashed) <= 20 and not _content_nouns(s)


def drop_boilerplate(sentences: list[str]) -> tuple[list[str], list[str]]:
    """(남긴 문장, 뺀 문장). 전부 걸리면 아무것도 빼지 않습니다."""
    if not getattr(settings, "QUERY_DROP_BOILERPLATE", True):
        return sentences, []
    kept = [s for s in sentences if not is_boilerplate(s)]
    if not kept:
        return sentences, []
    return kept, [s for s in sentences if s not in kept]


def _generic_phrase(phrase: str) -> bool:
    return all(w in _GENERIC_NOUNS for w in phrase.split())


def query_signature() -> str:
    """질의문을 바꾸는 설정 전체. 벡터DB·후보 캐시가 이 값이 바뀌면 다시 계산합니다."""
    import hashlib

    parts = [
        f"v{QUERY_VERSION}",
        str(settings.KEYWORD_TOP_K), str(settings.KEYWORD_USE_MMR), str(settings.KEYWORD_MMR_DIVERSITY),
        str(settings.SUMMARY_MAX_SENTENCES), str(settings.SUMMARY_MAX_CHARS),
        str(getattr(settings, "QUERY_DROP_BOILERPLATE", True)),
        str(getattr(settings, "SUMMARY_FULL_TEXT_CHARS", 200)),
        str(getattr(settings, "SUMMARY_KEEP_FIRST", True)),
        str(settings.PIPELINE_MAX_INPUT_CHARS),
        "|".join(p.pattern for p in _BOILERPLATE), ",".join(sorted(_GENERIC_NOUNS)),
    ]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()[:16]


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
    """
    추출 요약.
      - 문장이 SUMMARY_MAX_SENTENCES 이하이거나 전체가 SUMMARY_FULL_TEXT_CHARS 이하면 전부 씁니다.
      - 그보다 길면 TextRank 로 고르되, SUMMARY_KEEP_FIRST 면 첫 문장은 항상 넣습니다.
        (상투 문장끼리 서로 비슷해 '중심 문장'으로 뽑히고 정작 핵심 첫 문장이 빠지는 일을 막음)
    """
    max_sents = settings.SUMMARY_MAX_SENTENCES

    if not sentences:
        return ""
    joined = " ".join(sentences)
    if len(sentences) <= max_sents or len(joined) <= getattr(settings, "SUMMARY_FULL_TEXT_CHARS", 200):
        return joined[: settings.SUMMARY_MAX_CHARS]

    try:
        picked = _textrank(sentences, max_sents)
    except Exception as exc:
        warnings.append(f"TextRank 요약 실패 - 앞 문장으로 대체: {type(exc).__name__}")
        logger.warning("TextRank 요약 실패 | %s", exc)
        picked = list(range(max_sents))

    if getattr(settings, "SUMMARY_KEEP_FIRST", True) and 0 not in picked:
        picked = sorted([0] + picked[:-1]) if len(picked) >= max_sents else sorted([0] + picked)

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
    # 상투 문구를 뺀 본문으로 키워드·요약을 만듭니다. (원문은 그대로 - ⑤ 는 원문을 봄)
    core, dropped = drop_boilerplate(sentences)
    core_text = " ".join(core) if dropped else text
    candidates = [c for c in extract_noun_phrases(core_text) if not _generic_phrase(c)]
    top_k = settings.KEYWORD_TOP_K

    # --- 키워드 3개 ---
    keywords: list[Keyword] = []
    method = "keybert"
    if not candidates:
        method = "none"
        warnings.append("명사구 후보를 찾지 못했습니다.")
    elif keybert_installed():
        try:
            keywords = _keywords_by_keybert(core_text, candidates, top_k)
        except Exception as exc:
            method = "frequency"
            warnings.append(f"KeyBERT 실패 - 빈도 기반으로 대체: {type(exc).__name__}")
            logger.warning("KeyBERT 키워드 추출 실패 | %s", exc)
            keywords = _keywords_by_frequency(core_text, candidates, top_k)
    else:
        method = "frequency"
        warnings.append("keybert 미설치 - 빈도 기반 키워드를 사용했습니다.")
        keywords = _keywords_by_frequency(core_text, candidates, top_k)

    if not keywords and candidates:
        keywords = _keywords_by_frequency(core_text, candidates, top_k)

    # --- 요약문 ---
    summary = _summarize(core, warnings)

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
        dropped=dropped,
        method=method,
        warnings=warnings,
    )


def status() -> dict[str, object]:
    return {
        "kiwipiepy": kiwi_installed(),
        "keybert": keybert_installed(),
        "kiwi_error": _kiwi_error,
    }
