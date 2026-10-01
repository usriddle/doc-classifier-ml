"""
④ 후보 방식 측정 - 벡터DB 를 언제·어떻게 쓸지, 카테고리 점수를 어떻게 낼지 records 로 비교합니다.

학습 노트북 6-1 셀에서 씁니다. 모델 학습과 무관하고, bge-m3 만 있으면 됩니다.

  1) collect()  : 문장마다 질의문 벡터를 한 번만 만들고(캐시), 아래 재료를 모아 둡니다.
                  - 카테고리 7종 점수 (single / multi 두 방식 모두)
                  - 비슷한 사례 상위 N개의 (카테고리, 유사도)  ← 문턱과 무관하게 전부 조회
  2) sweep()    : 방식 × 문턱 × 가중치 조합마다 top-3 를 다시 계산합니다. (산수라 몇 초)
  3) choose()   : 조정용(train+val)에서 고르고, 평가용(test)은 확인만 합니다.

계산식은 운영과 같은 함수(candidates.score_matrix, case_store.case_scores_from / combine)를
그대로 부르므로, 여기서 고른 값이 서버에서도 같은 결과를 냅니다.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from app.config import settings
from app.services import candidates, case_store, categories, embedder, keyphrase, prompts
from training.records import GoldRecord

NAMES = [c.name for c in categories.CATEGORIES]
MODE_RANK = {"low_confidence": 0, "always": 1, "union": 2}


# =============================================================================
# 1) 재료 모으기
# =============================================================================
@dataclass
class ScoreTable:
    ids: list[str]
    texts: list[str]
    gold: list[str]
    splits: list[str]
    scores: dict[str, np.ndarray]           # "single" / "multi" -> (N, 7), NAMES 순서
    hits: list[list[tuple[str, float]]]     # 문장별 비슷한 사례 (카테고리, 유사도), 유사도 내림차순

    def __len__(self) -> int:
        return len(self.ids)


def _query_signature(texts: list[str]) -> str:
    parts = [
        settings.EMBED_MODEL, str(settings.EMBED_NORMALIZE), str(settings.EMBED_MAX_SEQ_LENGTH),
        str(settings.KEYWORD_TOP_K), str(settings.KEYWORD_USE_MMR), str(settings.KEYWORD_MMR_DIVERSITY),
        str(settings.SUMMARY_MAX_SENTENCES), str(settings.SUMMARY_MAX_CHARS),
        str(settings.PIPELINE_MAX_INPUT_CHARS),
        hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _query_vectors(texts: list[str], cache_path: Path | None, progress: bool) -> np.ndarray:
    """문장 -> ②③ 질의문(키워드+요약) -> bge-m3 벡터. 같은 문장·설정이면 캐시에서 읽습니다."""
    sig = _query_signature(texts)
    if cache_path and cache_path.is_file():
        try:
            data = np.load(cache_path, allow_pickle=False)
            if str(data["signature"]) == sig and data["vectors"].shape[0] == len(texts):
                return data["vectors"]
        except Exception:
            pass

    iterator = texts
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(texts, desc="②③ 질의문 만들기")
        except ImportError:
            pass
    queries = [keyphrase.build_query(keyphrase.normalize(t)).query_text for t in iterator]
    vectors = embedder.encode(queries)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, signature=np.array(sig), vectors=vectors)
    return vectors


def collect(
    records: list[GoldRecord],
    split_of: dict[str, str],
    cache_path: str | Path | None = None,
    max_hits: int = 10,
    progress: bool = True,
) -> ScoreTable:
    """정답 카테고리(7종)가 있는 레코드만 모읍니다. ('없음' 제외) split_of: {record_id: train/val/test}"""
    recs = [r for r in records if r.category and r.category != prompts.CATEGORY_NONE]
    if not case_store._attempted:
        case_store.load_or_build()

    texts = [r.text for r in recs]
    q = _query_vectors(texts, Path(cache_path) if cache_path else None, progress)

    scores = {
        "single": candidates.score_matrix(q, "single"),
        "multi": candidates.score_matrix(q, "multi"),
    }
    k = max(max_hits, settings.CASE_TOP_K)
    hits = []
    if case_store.is_ready():
        for i in range(len(recs)):
            hits.append([(c, float(s)) for _, c, s in case_store.search_raw(q[i], k)])
    else:
        hits = [[] for _ in recs]

    return ScoreTable(
        ids=[r.id for r in recs],
        texts=texts,
        gold=[r.category for r in recs],
        splits=[split_of.get(r.id, "train") for r in recs],
        scores=scores,
        hits=hits,
    )


# =============================================================================
# 2) 한 방식으로 top-3 다시 계산
# =============================================================================
@dataclass(frozen=True)
class Strategy:
    scoring: str = "single"          # single / multi
    mode: str = "low_confidence"     # low_confidence / always / union
    threshold: float = 0.45          # low_confidence 일 때만 의미
    weight: float = 0.5
    case_top_k: int = 5
    case_min: float = 0.5

    @classmethod
    def current(cls) -> "Strategy":
        """지금 .env 설정."""
        return cls(
            scoring=candidates.scoring_mode(),
            mode=case_store.mode(),
            threshold=settings.CANDIDATE_MIN_SCORE,
            weight=settings.CASE_BLEND_WEIGHT,
            case_top_k=settings.CASE_TOP_K,
            case_min=settings.CASE_MIN_SCORE,
        )

    def label(self) -> str:
        head = f"{self.scoring} · {self.mode}"
        if self.mode == "low_confidence":
            head += f"(<{self.threshold:.2f})"
        return f"{head} · w={self.weight:.2f}"

    def env(self) -> dict[str, str]:
        out = {
            "CANDIDATE_SCORING": self.scoring,
            "CASE_MODE": self.mode,
            "CASE_BLEND_WEIGHT": f"{self.weight:g}",
            "CASE_TOP_K": str(self.case_top_k),
            "CASE_MIN_SCORE": f"{self.case_min:g}",
        }
        if self.mode == "low_confidence":
            out["CANDIDATE_MIN_SCORE"] = f"{self.threshold:g}"
        return out


def top_names(table: ScoreTable, strat: Strategy, i: int, top_k: int | None = None) -> tuple[list[str], bool]:
    """(top-k 이름, 벡터DB 를 열었는지) - 운영 pipeline.run_candidates 와 같은 순서로 계산합니다."""
    k = top_k or settings.CANDIDATE_TOP_K
    raw = table.scores[strat.scoring][i]
    order = np.argsort(-raw)
    base = [(NAMES[j], round(float(raw[j]), 4)) for j in order]
    low = base[0][1] < strat.threshold
    if not case_store.should_consult(low, strat.mode):
        return [n for n, _ in base[:k]], False
    cs = case_store.case_scores_from(table.hits[i], strat.case_top_k, strat.case_min)
    if not cs:          # 기준을 넘은 사례가 없으면 운영도 ④ 그대로
        return [n for n, _ in base[:k]], True
    new, _ = case_store.combine(base, cs, strat.mode, strat.weight, k)
    return [n for n, _ in new[:k]], True


@dataclass
class Result:
    strategy: Strategy
    n: int
    miss: int
    top1: int
    opened: int
    fixed: int = 0
    broken: int = 0
    miss_by_cat: dict[str, int] = field(default_factory=dict)
    total_by_cat: dict[str, int] = field(default_factory=dict)
    miss_ids: list[str] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return 1 - self.miss / self.n if self.n else 0.0

    @property
    def worst(self) -> tuple[str, float]:
        rates = {c: self.miss_by_cat.get(c, 0) / t for c, t in self.total_by_cat.items() if t}
        if not rates:
            return "-", 0.0
        name = max(rates, key=rates.get)
        return name, rates[name]


def evaluate(
    table: ScoreTable, strat: Strategy, index: list[int], baseline_miss: set[str] | None = None
) -> Result:
    res = Result(strategy=strat, n=len(index), miss=0, top1=0, opened=0)
    tot, mis = Counter(), Counter()
    for i in index:
        names, opened = top_names(table, strat, i)
        gold = table.gold[i]
        tot[gold] += 1
        res.opened += opened
        if names and names[0] == gold:
            res.top1 += 1
        if gold not in names:
            res.miss += 1
            mis[gold] += 1
            res.miss_ids.append(table.ids[i])
    res.miss_by_cat, res.total_by_cat = dict(mis), dict(tot)
    if baseline_miss is not None:
        now = set(res.miss_ids)
        res.fixed = len(baseline_miss - now)
        res.broken = len(now - baseline_miss)
    return res


def grid(
    thresholds=(0.45, 0.5, 0.55, 0.6, 0.65, 0.7),
    weights=(0.2, 0.3, 0.4, 0.5, 0.6, 0.8),
    union_weights=(0.0, 0.2, 0.3, 0.4, 0.5),
    scorings=("single", "multi"),
    case_top_k: int | None = None,
    case_min: float | None = None,
) -> list[Strategy]:
    kt = case_top_k or settings.CASE_TOP_K
    cm = settings.CASE_MIN_SCORE if case_min is None else case_min
    out = []
    for sc in scorings:
        for t in thresholds:
            for w in weights:
                out.append(Strategy(sc, "low_confidence", t, w, kt, cm))
        for w in weights:
            out.append(Strategy(sc, "always", settings.CANDIDATE_MIN_SCORE, w, kt, cm))
        for w in union_weights:
            out.append(Strategy(sc, "union", settings.CANDIDATE_MIN_SCORE, w, kt, cm))
    return out


# =============================================================================
# 3) 비교 · 고르기
# =============================================================================
@dataclass
class Sweep:
    tune_index: list[int]
    test_index: list[int]
    baseline: Result
    baseline_test: Result
    results: list[Result]


def sweep(table: ScoreTable, strategies: list[Strategy] | None = None, baseline: Strategy | None = None) -> Sweep:
    tune = [i for i, s in enumerate(table.splits) if s in ("train", "val")]
    test = [i for i, s in enumerate(table.splits) if s == "test"]
    base = baseline or Strategy.current()
    b_tune = evaluate(table, base, tune)
    b_test = evaluate(table, base, test)
    base_miss = set(b_tune.miss_ids)
    results = [evaluate(table, s, tune, base_miss) for s in (strategies or grid())]
    return Sweep(tune, test, b_tune, b_test, results)


def choose(sw: Sweep, slack: float = 0.002) -> Result:
    """
    놓침이 가장 적은 방식 근처(조정용 건수의 slack 이내, 최소 2건)에서
    망가뜨린 건수 → 단순한 방식(single, low_confidence < always < union) → 작은 가중치 순으로 고릅니다.
    """
    best = min(r.miss for r in sw.results)
    margin = max(2, int(round(slack * max(r.n for r in sw.results))))
    near = [r for r in sw.results if r.miss <= best + margin]
    return min(
        near,
        key=lambda r: (r.broken, r.strategy.scoring == "multi", MODE_RANK[r.strategy.mode], r.strategy.weight, r.miss),
    )


def rows(results: list[Result]) -> list[dict]:
    out = []
    for r in results:
        worst, rate = r.worst
        out.append({
            "방식": r.strategy.label(),
            "놓침": r.miss,
            "후보3 적중률": f"{r.recall:.1%}",
            "1위 적중률": f"{r.top1 / r.n:.1%}" if r.n else "-",
            "살림": r.fixed,
            "망가뜨림": r.broken,
            "최악 카테고리": f"{worst} {rate:.0%}",
            "벡터DB 열림": f"{r.opened / r.n:.0%}" if r.n else "-",
        })
    return out


def per_category(results: dict[str, Result]) -> list[dict]:
    """{'열 이름': Result} -> 카테고리별 '놓침/전체 (비율)' 표."""
    out = []
    for name in NAMES:
        row = {"카테고리": name}
        for col, r in results.items():
            t = r.total_by_cat.get(name, 0)
            m = r.miss_by_cat.get(name, 0)
            row[col] = f"{m}/{t} ({m / t:.0%})" if t else "-"
        out.append(row)
    return out


def consistency(table: ScoreTable, cands: dict[str, dict]) -> tuple[int, int, list[str]]:
    """
    지금 설정으로 여기서 계산한 top-3 가 6단계(운영 코드) 결과와 같은지.
    (배치 임베딩과 1건씩 임베딩의 미세한 소수점 차이로 동점 근처 몇 건은 다를 수 있습니다)
    """
    strat = Strategy.current()
    diff = []
    for i, rid in enumerate(table.ids):
        if rid not in cands:
            continue
        names, _ = top_names(table, strat, i)
        if names != cands[rid]["names"]:
            diff.append(rid)
    total = sum(1 for rid in table.ids if rid in cands)
    return total - len(diff), total, diff


def miss_examples(table: ScoreTable, strat: Strategy, index: list[int], limit: int = 30) -> list[dict]:
    out = []
    for i in index:
        names, opened = top_names(table, strat, i)
        if table.gold[i] not in names:
            cs = case_store.case_scores_from(table.hits[i], strat.case_top_k, strat.case_min)
            out.append({
                "id": table.ids[i],
                "문장": table.texts[i],
                "정답": table.gold[i],
                "후보 3개": " / ".join(names),
                "사례 1위": (max(cs.items(), key=lambda x: x[1])[0] if cs else "(없음)"),
                "최근접 사례 유사도": round(table.hits[i][0][1], 3) if table.hits[i] else None,
            })
            if len(out) >= limit:
                break
    return out


def update_env(path: str | Path, values: dict[str, str]) -> list[str]:
    """
    .env 의 KEY=값 줄만 바꿉니다. (주석·다른 줄은 그대로, 없는 키는 끝에 추가)
    바뀐 내용을 '키: 이전 -> 새 값' 목록으로 돌려줍니다.
    """
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    changes, seen = [], set()
    for idx, line in enumerate(lines):
        m = re.match(r"^\s*([A-Z0-9_]+)\s*=\s*([^#]*?)(\s*#.*)?$", line)
        if not m or m.group(1) not in values:
            continue
        key, old, comment = m.group(1), m.group(2).strip(), m.group(3) or ""
        seen.add(key)
        new = values[key]
        if old != new:
            head = f"{key}={new}"
            lines[idx] = head + (" " * max(1, 34 - len(head)) + comment.strip() if comment else "")
            changes.append(f"{key}: {old} -> {new}")
    added = [k for k in values if k not in seen]
    if added:
        lines += ["", "# 학습 노트북 6-1 측정 셀이 추가한 값"] + [f"{k}={values[k]}" for k in added]
        changes += [f"{k}: (없음) -> {values[k]}" for k in added]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changes


def save(sw: Sweep, chosen: Result, path: str | Path) -> None:
    """측정 결과를 JSON 으로 남깁니다. (나중에 어떤 근거로 값을 골랐는지 확인용)"""
    data = {
        "baseline": {"strategy": asdict(sw.baseline.strategy), **{k: v for k, v in rows([sw.baseline])[0].items()}},
        "chosen": {"strategy": asdict(chosen.strategy), **rows([chosen])[0]},
        "top": [{"strategy": asdict(r.strategy), **row} for r, row in zip(
            sorted(sw.results, key=lambda r: r.miss)[:30], rows(sorted(sw.results, key=lambda r: r.miss)[:30]))],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
