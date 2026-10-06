"""
정답 레코드 -> 학습 샘플(토큰 id + loss 마스크).

가장 중요한 원칙: 학습 입력은 추론 입력과 토큰 단위로 같아야 합니다.
그래서 입력을 직접 조립하지 않고 운영 코드의 함수를 그대로 씁니다.

  고정 프리픽스·채팅 템플릿 : llm_engine.template_parts(tokenizer)
  단계별 사용자 프롬프트    : prompts.build_intent_prompt / build_category_prompt / build_tool_prompt
  번호 정답 토큰            : llm_engine.number_token_ids(tokenizer, n)
  어시스턴트 앞머리         : llm_engine.INTENT_ASSISTANT_PREFIX / CATEGORY_ASSISTANT_PREFIX
  원문 정리                 : keyphrase.normalize -> llm_engine.prepare_text
  ④ 후보 3개               : pipeline.run_candidates (벡터DB 재정렬 포함)

토큰화도 추론과 같게 합니다. 추론은 [프리픽스] 와 [요청 구간] 을 따로 encode 해
이어 붙이므로, 학습도 똑같이 따로 encode 한 뒤 정답 토큰을 뒤에 붙입니다.

  input_ids = 프리픽스 ids + 요청 구간 ids + 정답 ids + 끝 토큰
  labels    = -100 ...      -100 ...        정답 ids + 끝 토큰     <- 여기만 loss
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.config import BASE_DIR, settings
from app.services import case_store, categories, keyphrase, llm_engine, prompts
from training.records import GoldRecord

IGNORE = -100
TASKS = ("tool", "intent", "category")


def llm_text(record: GoldRecord) -> str:
    """⑤ 가 실제로 보는 원문. (pipeline.run -> llm_engine.decide 와 같은 정리)"""
    return llm_engine.prepare_text(keyphrase.normalize(record.text))


# =============================================================================
# ④ 후보 - 운영과 같은 코드로 계산 + 파일 캐시
# =============================================================================
def _candidate_signature() -> str:
    """후보 결과를 바꾸는 설정들의 지문. 바뀌면 캐시를 버리고 다시 계산합니다."""
    csv_path = BASE_DIR / settings.CASE_CSV_PATH if hasattr(settings, "CASE_CSV_PATH") else None
    csv_hash = ""
    if csv_path and Path(csv_path).is_file():
        csv_hash = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:16]
    parts = {
        "embed": settings.EMBED_MODEL,
        "min_score": settings.CANDIDATE_MIN_SCORE,
        "case_enabled": settings.CASE_STORE_ENABLED,
        "case_blend": getattr(settings, "CASE_BLEND_WEIGHT", None),
        "case_min": getattr(settings, "CASE_MIN_SCORE", None),
        "case_top_k": getattr(settings, "CASE_TOP_K", None),
        "cases_csv": csv_hash,
        "rerank_version": case_store.RERANK_VERSION,
        "case_mode": case_store.mode(),
        "scoring": getattr(settings, "CANDIDATE_SCORING", "single"),
        "cand_top_k": settings.CANDIDATE_TOP_K,
        "query": keyphrase.query_signature(),   # ④ 질의문 만드는 방식 (상투 문구 제거 등)
        # 요약(프롬프트)뿐 아니라 키워드까지 포함한 임베딩 정의문 전체 - 키워드만 고쳐도 캐시를 버립니다.
        "categories": hashlib.sha256(
            "\n".join([categories.prompt_block()] + [c.embedding_text for c in categories.CATEGORIES]).encode()
        ).hexdigest()[:16],
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def compute_candidates(
    records: list[GoldRecord], cache_path: str | Path | None = None, progress: bool = True
) -> dict[str, dict]:
    """
    레코드별 ④ 후보(벡터DB 재정렬 포함)를 계산합니다. 결과: {record_id: {...}}

    bge-m3 가 필요합니다. 같은 문장·같은 설정이면 캐시 파일에서 바로 읽습니다.
    """
    from app.services import pipeline

    signature = _candidate_signature()
    cache: dict = {"signature": signature, "items": {}}
    if cache_path and Path(cache_path).is_file():
        loaded = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        if loaded.get("signature") == signature:
            cache = loaded

    out: dict[str, dict] = {}
    todo = []
    for r in records:
        key = hashlib.sha256(keyphrase.normalize(r.text).encode("utf-8")).hexdigest()[:20]
        if key in cache["items"]:
            out[r.id] = cache["items"][key]
        else:
            todo.append((r, key))

    iterator = todo
    if progress and todo:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(todo, desc="④ 후보 계산")
        except ImportError:
            pass
    for r, key in iterator:
        stage = pipeline.run_candidates(keyphrase.normalize(r.text), request_id=f"train-{r.id}")
        item = {
            "names": stage.candidate_names,
            "scores": [round(float(c.score), 4) for c in stage.candidate.top],
            "low_confidence": bool(stage.candidate.low_confidence),
            "case_used": bool(stage.case_lookup.used) if stage.case_lookup else False,
        }
        cache["items"][key] = item
        out[r.id] = item

    if cache_path and todo:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return out


def category_for_tool_prompt(record: GoldRecord, cands: dict[str, dict]) -> str:
    """
    도구 JSON 프롬프트에 넣을 '확정된 카테고리'. 정답 카테고리를 그대로 씁니다. (teacher forcing)

    조회·수정·삭제의 '없음' 도 그대로 '없음' 입니다. (운영에서도 없음이면 프롬프트에 '없음' 으로 적힘)
    정답이 비어 있는 옛 데이터만 ④ 1위로 채웁니다.
    """
    if record.category:
        return record.category
    if record.category_allows_none:
        return prompts.CATEGORY_NONE
    names = (cands.get(record.id) or {}).get("names") or []
    return names[0] if names else categories.FALLBACK.name


# =============================================================================
# 샘플
# =============================================================================
@dataclass
class Sample:
    task: str                   # tool / intent / category
    record_id: str
    input_ids: list[int]
    labels: list[int]
    target_text: str            # 사람이 읽는 정답 (확인용)
    note: str = ""              # 예: "후보 순서 섞음", "정답을 후보에 끼워 넣음"

    @property
    def length(self) -> int:
        return len(self.input_ids)

    @property
    def target_length(self) -> int:
        return sum(1 for x in self.labels if x != IGNORE)

    def to_dict(self) -> dict:
        return {"input_ids": self.input_ids, "labels": self.labels}


@dataclass
class BuildStats:
    counts: Counter = field(default_factory=Counter)
    available: Counter = field(default_factory=Counter)
    too_long: list[str] = field(default_factory=list)
    category_miss: list[str] = field(default_factory=list)      # 정답이 ④ 후보 밖
    category_injected: list[str] = field(default_factory=list)  # 학습용으로 정답을 끼워 넣음
    gold_position: Counter = field(default_factory=Counter)     # 카테고리 정답이 몇 번째 후보인지
    lengths: list[int] = field(default_factory=list)


class SampleBuilder:
    """토크나이저 하나로 세 종류 샘플을 만듭니다."""

    def __init__(self, tokenizer, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        prefix_text, self.suffix_text = llm_engine.template_parts(tokenizer)
        self.prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
        self.intent_ids = llm_engine.number_token_ids(tokenizer, len(prompts.INTENTS))
        self.end_ids = [self._end_token_id()]

    def _end_token_id(self) -> int:
        """어시스턴트 발화를 닫는 토큰. 서버 생성이 멈추는 토큰과 같은 함수로 정합니다."""
        return llm_engine.turn_end_token_id(self.tokenizer)

    @property
    def prefix_tokens(self) -> int:
        return len(self.prefix_ids)

    def _make(
        self, task: str, record: GoldRecord, user_prompt: str, assistant_prefix: str,
        target_ids: list[int], target_text: str, note: str = "",
    ) -> Sample:
        rest = self.tokenizer.encode(
            user_prompt + self.suffix_text + assistant_prefix, add_special_tokens=False
        )
        answer = list(target_ids) + self.end_ids
        context = self.prefix_ids + rest
        return Sample(
            task=task,
            record_id=record.id,
            input_ids=context + answer,
            labels=[IGNORE] * len(context) + answer,
            target_text=target_text,
            note=note,
        )

    # --- 과제별 ---------------------------------------------------------
    def intent_sample(self, record: GoldRecord) -> Sample:
        number = prompts.INTENTS.index(record.intent_obj) + 1
        return self._make(
            "intent", record,
            prompts.build_intent_prompt(llm_text(record)),
            llm_engine.INTENT_ASSISTANT_PREFIX,
            [self.intent_ids[number - 1]],
            f"{number} ({record.intent})",
        )

    def category_sample(self, record: GoldRecord, names: list[str], note: str = "") -> Sample:
        """names = ④ 후보 (CANDIDATE_TOP_K 개). 조회·수정·삭제면 마지막에 '없음' 이 붙습니다. (운영과 같은 선택지)"""
        allow_none = record.category_allows_none
        labels = prompts.category_labels(names, allow_none)
        number = labels.index(record.category) + 1
        ids = llm_engine.number_token_ids(self.tokenizer, len(labels))
        return self._make(
            "category", record,
            prompts.build_category_prompt(llm_text(record), names, allow_none),
            llm_engine.CATEGORY_ASSISTANT_PREFIX,
            [ids[number - 1]],
            f"{number} ({record.category}) / 후보={labels}",
            note,
        )

    def tool_sample(self, record: GoldRecord, category_name: str) -> Sample:
        target = prompts.format_tool_call(record.tool, record.tool_arguments())
        return self._make(
            "tool", record,
            prompts.build_tool_prompt(llm_text(record), record.intent_obj, category_name),
            "",
            self.tokenizer.encode(target, add_special_tokens=False),
            target,
        )

    # --- 확인용 ---------------------------------------------------------
    def describe(self, sample: Sample, tail_chars: int = 500) -> str:
        """샘플을 사람이 읽게 풀어 씁니다. [학습 대상] 표시 부분만 loss 가 걸립니다."""
        first = next(i for i, x in enumerate(sample.labels) if x != IGNORE)
        context = self.tokenizer.decode(sample.input_ids[:first])
        target = self.tokenizer.decode(sample.input_ids[first:])
        shown = context if len(context) <= tail_chars else "…(앞부분 생략)…" + context[-tail_chars:]
        return (
            f"과제={sample.task} | 레코드={sample.record_id} | 총 {sample.length}토큰 "
            f"(학습 대상 {sample.target_length}토큰){' | ' + sample.note if sample.note else ''}\n"
            f"{shown}【학습 대상 ▶ {target} ◀】"
        )


# =============================================================================
# 분할 하나 -> 샘플 목록 (과제 비율 조절 포함)
# =============================================================================
def build_samples(
    records: list[GoldRecord],
    builder: SampleBuilder,
    cands: dict[str, dict],
    *,
    train: bool,
    mix: dict[str, float] | None = None,
    shuffle_prob: float = 0.5,
    seed: int = 42,
    none_boost: int = 1,
) -> tuple[list[Sample], BuildStats]:
    """
    mix : 과제 비율. 예) {"tool": 0.7, "intent": 0.15, "category": 0.15}
          도구 JSON 샘플 수를 기준으로 나머지 과제 수를 정합니다. (가진 데이터보다 많이는 못 만듦)
    train=True 일 때만
      - 카테고리 정답이 ④ 후보 밖이면 정답을 후보에 끼워 넣어 샘플을 만듭니다.
        (검증/평가에서는 끼워 넣지 않고 '④ 놓침'으로 따로 셉니다)
      - shuffle_prob 확률로 후보 순서를 섞은 샘플을 하나 더 만듭니다.
        ④ 가 대부분 정답을 1번에 올려 주므로, 안 섞으면 '무조건 1번' 지름길을 배웁니다.
      - none_boost 배만큼 카테고리 정답이 '없음' 인 샘플을 늘립니다. (조회·수정·삭제의 없음 판정 연습)

    의도 샘플은 의도별로 고르게 뽑습니다. (sqrt(레코드 수) 비율 - 적은 의도를 조금 더 많이)
    각 의도 안에서는 '경계' 태그 문장을 먼저 넣습니다.
    예전에는 '경계·문의·해당없음 전부' 를 먼저 넣었는데, 그 수가 할당량보다 많으면 조회·수정·삭제
    의도 샘플이 하나도 안 들어가 의도 정확도가 크게 떨어졌습니다.
    """
    mix = mix or {"tool": 0.7, "intent": 0.15, "category": 0.15}
    rng = random.Random(seed)
    stats = BuildStats()
    out: list[Sample] = []

    def add(sample: Sample) -> None:
        if sample.length > builder.max_length:
            stats.too_long.append(f"{sample.task}:{sample.record_id}({sample.length})")
            return
        out.append(sample)
        stats.counts[sample.task] += 1
        stats.lengths.append(sample.length)

    # 1) 도구 JSON - 기준 과제, 전부 사용
    tool_records = [r for r in records if r.tool]
    stats.available["tool"] = len(tool_records)
    for r in tool_records:
        add(builder.tool_sample(r, category_for_tool_prompt(r, cands)))

    base = max(stats.counts["tool"], 1)

    def quota(task: str, available: int) -> int:
        if mix.get("tool", 0) <= 0 or not tool_records:
            return available
        return min(available, round(base * mix.get(task, 0) / mix["tool"]))

    # 2) 의도 - 의도별로 고르게 (sqrt 비율). 각 의도 안에서는 '경계' 문장 먼저
    by_intent: dict[str, list[GoldRecord]] = {}
    for r in records:
        by_intent.setdefault(r.intent, []).append(r)
    stats.available["intent"] = len(records)
    total_q = quota("intent", len(records))
    # 문의·해당없음은 도구 샘플이 없어 의도 샘플로만 배우므로 1.5배 더 줍니다.
    weights = {k: len(v) ** 0.5 * (1.5 if v[0].tool is None else 1.0) for k, v in by_intent.items()}
    wsum = sum(weights.values()) or 1.0
    for intent_name in sorted(by_intent):
        rs = by_intent[intent_name][:]
        rng.shuffle(rs)
        rs.sort(key=lambda r: 0 if "경계" in r.tags else 1)
        k = min(len(rs), max(1, round(total_q * weights[intent_name] / wsum)))
        for r in rs[:k]:
            add(builder.intent_sample(r))

    # 3) 카테고리 (접수 + 조회·수정·삭제. 조회·수정·삭제는 '없음' 선택지가 붙음)
    cat_samples: list[Sample] = []
    for r in records:
        names = list((cands.get(r.id) or {}).get("names") or [])
        if not r.category_label or not names:
            continue
        if r.category == prompts.CATEGORY_NONE:
            # 정답이 '없음' - 항상 마지막 번호. 순서 대신 후보 조합을 바꾼 샘플로 none_boost 만큼 늘림
            stats.gold_position["없음"] += 1
            cat_samples.append(builder.category_sample(r, names))
            for _ in range(max(0, none_boost - 1) if train else 0):
                shuffled = names[:]
                rng.shuffle(shuffled)
                cat_samples.append(builder.category_sample(r, shuffled, "없음 늘림"))
        elif r.category in names:
            stats.gold_position[names.index(r.category) + 1] += 1
            cat_samples.append(builder.category_sample(r, names))
            if train and rng.random() < shuffle_prob and len(names) > 1:
                shuffled = names[:]
                while shuffled.index(r.category) == names.index(r.category):
                    rng.shuffle(shuffled)
                cat_samples.append(builder.category_sample(r, shuffled, "후보 순서 섞음"))
        else:
            stats.category_miss.append(r.id)
            if train:
                injected = names[:-1]
                injected.insert(rng.randrange(len(injected) + 1), r.category)
                stats.category_injected.append(r.id)
                cat_samples.append(builder.category_sample(r, injected, "정답을 후보에 끼워 넣음"))
    rng.shuffle(cat_samples)
    stats.available["category"] = len(cat_samples)
    for s in cat_samples[: quota("category", len(cat_samples))]:
        add(s)

    rng.shuffle(out)
    return out, stats


def limit_samples(
    samples: list[Sample], limit: int | None, records: list[GoldRecord], seed: int = 42
) -> list[Sample]:
    """
    샘플 수를 limit 개로 줄입니다. (학습 시간을 줄이기 위함)

    (과제, 의도) 묶음별 비율을 그대로 유지하며 줄이므로, 도구 JSON / 의도 / 카테고리 비율과
    접수·조회·수정·삭제·문의·해당없음 구성이 원래와 같게 남습니다. 묶음마다 최소 1개는 남깁니다.
    같은 seed 면 항상 같은 샘플이 뽑힙니다.
    """
    if not limit or len(samples) <= limit:
        return samples
    intent_of = {r.id: r.intent for r in records}
    groups: dict[tuple[str, str], list[Sample]] = {}
    for s in samples:
        groups.setdefault((s.task, intent_of.get(s.record_id, "")), []).append(s)
    keys = sorted(groups)
    quota = {k: min(len(groups[k]), max(1, round(limit * len(groups[k]) / len(samples)))) for k in keys}
    while sum(quota.values()) > limit:            # 반올림으로 넘친 만큼 가장 큰 묶음에서 뺌
        biggest = max(keys, key=lambda k: quota[k])
        if quota[biggest] <= 1:
            break
        quota[biggest] -= 1
    rng = random.Random(seed)
    picked: list[Sample] = []
    for k in keys:
        picked += rng.sample(groups[k], quota[k])
    rng.shuffle(picked)
    return picked


def length_report(samples: list[Sample]) -> dict[str, int]:
    if not samples:
        return {}
    lengths = sorted(s.length for s in samples)
    return {
        "개수": len(lengths),
        "최소": lengths[0],
        "중간": lengths[len(lengths) // 2],
        "최대": lengths[-1],
        "학습대상_최대": max(s.target_length for s in samples),
    }
