"""
평가 - 운영 코드(llm_engine.judge_* / generate_tool_call)로 채점합니다.

평가용으로 판정 로직을 따로 만들지 않습니다. llm_engine.use_model() 로 평가할 모델을
끼워 넣고 운영 함수 그대로 부르므로, 여기서 나온 점수가 곧 서버에서의 동작입니다.
(KV 캐시도 운영과 똑같이 쓰며, 가중치가 바뀔 때마다 비우고 다시 검증합니다)

과제마다 앞 단계 정답을 넣고 그 단계만 채점합니다. (앞 단계 오류가 섞이지 않게)
  의도     : 원문 -> 6지선다
  카테고리 : 원문 + ④ 후보 3개 (조회·수정·삭제는 + 없음) -> 정답 번호
             (정답이 후보 밖이면 '④ 놓침'으로 따로 셈. '없음' 정답은 항상 선택지에 있음)
  도구 JSON: 원문 + 정답 의도 + 정답 카테고리 -> JSON
"""

from __future__ import annotations

import csv
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from app.services import llm_engine, prompts
from training.records import GoldRecord, _as_int, _squash
from training.samples import category_for_tool_prompt, llm_text

# -----------------------------------------------------------------------------
# 지표 정의 - role: 목표(좋아져야 함) / 방어(떨어지면 안 됨) / 보조(참고)
#             better: high(높을수록 좋음) / low(낮을수록 좋음)
# -----------------------------------------------------------------------------
METRICS: dict[str, dict[str, str]] = {
    "의도 정확도":            {"role": "방어", "better": "high"},
    "해당없음 재현율":         {"role": "보조", "better": "high"},   # 해당없음을 해당없음으로
    "오반려율":               {"role": "방어", "better": "low"},    # 정상 민원을 해당없음으로
    "카테고리 정확도":         {"role": "방어", "better": "high"},
    "카테고리(정답 2위 이하) 정확도": {"role": "보조", "better": "high"},
    "카테고리 없음 재현율":     {"role": "목표", "better": "high"},   # 주제 없는 조회·수정·삭제를 없음으로
    "카테고리 없음 오판율":     {"role": "방어", "better": "low"},    # 주제가 있는데 없음으로
    "JSON 파싱 성공률":        {"role": "방어", "better": "high"},
    "도구 이름 일치율":        {"role": "방어", "better": "high"},
    "location 정확 일치":      {"role": "목표", "better": "high"},
    "location 지어냄 비율":    {"role": "목표", "better": "low"},    # 원문에 없는 위치를 채움
    "위치없음 빈칸 유지율":     {"role": "목표", "better": "high"},   # 정답이 "" 일 때 "" 로 둠
    "content 유사도":          {"role": "목표", "better": "high"},   # 글자 단위 ROUGE-L F1
    "complaint_id 정확 일치":  {"role": "목표", "better": "high"},
    "field 정확 일치":         {"role": "목표", "better": "high"},
    "keyword 정확 일치":       {"role": "목표", "better": "high"},   # 찾을 민원 대상 (원문 그대로)
    "period 정확 일치":        {"role": "목표", "better": "high"},   # 접수 시점 (원문 그대로)
    "찾기 조건 지어냄 비율":    {"role": "목표", "better": "low"},    # keyword·period 를 원문에 없는 말로
    "수정 빈칸 규칙 준수":      {"role": "목표", "better": "high"},   # 안 바꾸는 항목을 "" 로 둠
    "reason 유사도":           {"role": "보조", "better": "high"},
}


@dataclass
class EvalReport:
    label: str
    metrics: dict[str, dict] = field(default_factory=dict)   # 이름 -> {"value", "n"}
    by_tag: dict[str, dict[str, dict]] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    kv_cache: dict | None = None

    def value(self, name: str) -> float | None:
        m = self.metrics.get(name)
        return None if m is None else m["value"]

    def to_json(self) -> dict:
        return {
            "label": self.label, "metrics": self.metrics, "by_tag": self.by_tag,
            "seconds": round(self.seconds, 1), "kv_cache": self.kv_cache,
        }

    def save(self, directory: str | Path, name: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(
            json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self.rows:
            keys: list[str] = []
            for row in self.rows:
                keys += [k for k in row if k not in keys]
            with (d / f"{name}_rows.csv").open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.rows)

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(label=data["label"], metrics=data["metrics"], by_tag=data.get("by_tag", {}),
                   seconds=data.get("seconds", 0.0), kv_cache=data.get("kv_cache"))


# =============================================================================
# 문자열 비교 도구
# =============================================================================
def char_rouge_l(pred: str, gold: str) -> float:
    """공백을 뺀 글자 단위 ROUGE-L F1. 둘 다 비었으면 1."""
    a, b = _squash(pred), _squash(gold)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for ch in a:
        cur = [0]
        for j, ch2 in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if ch == ch2 else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(a), lcs / len(b)
    return 0.0 if lcs == 0 else 2 * p * r / (p + r)


def is_hallucinated_location(pred: str, source_text: str) -> bool:
    """예측 위치가 원문에 그대로 없으면 '지어냄'. (공백 무시)"""
    p = _squash(pred)
    return bool(p) and p not in _squash(source_text)


# =============================================================================
# 어댑터 켜기/끄기 (같은 세션에서 학습 전후 비교)
# =============================================================================
@contextmanager
def adapter(enabled: bool):
    """
    PeftModel 이면 enabled=False 동안 어댑터를 끄고 베이스 모델로 동작합니다.
    가중치가 바뀌는 것과 같으므로 앞뒤로 KV 캐시를 비웁니다.
    """
    _, model = llm_engine.get_model()
    llm_engine.reset_kv_cache()
    try:
        if enabled or not hasattr(model, "disable_adapter"):
            yield
        else:
            with model.disable_adapter():
                yield
    finally:
        llm_engine.reset_kv_cache()


@contextmanager
def _eval_mode():
    _, model = llm_engine.get_model()
    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        yield
    finally:
        if was_training:
            model.train()


# =============================================================================
# 평가
# =============================================================================
def _rate(hits: list[bool]) -> dict:
    return {"value": round(sum(hits) / len(hits), 4) if hits else None, "n": len(hits)}


def _mean(values: list[float]) -> dict:
    return {"value": round(sum(values) / len(values), 4) if values else None, "n": len(values)}


def evaluate(
    records: list[GoldRecord],
    cands: dict[str, dict],
    label: str,
    tasks: tuple[str, ...] = ("intent", "category", "tool"),
    progress: bool = True,
) -> EvalReport:
    """지금 llm_engine 에 올라가 있는 모델로 records 를 채점합니다."""
    # 가중치가 마지막 캐시 생성 뒤 바뀌었을 수 있으므로(학습 재실행 등) 항상 새로 만듭니다.
    llm_engine.reset_kv_cache()
    started = time.perf_counter()
    rows: list[dict] = []
    iterator = records
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(records, desc=f"평가({label})")
        except ImportError:
            pass

    with _eval_mode():
        for r in iterator:
            text = llm_text(r)
            row: dict = {"id": r.id, "tags": ",".join(r.auto_tags()), "text": r.text,
                         "gold_intent": r.intent}

            if "intent" in tasks:
                intent, choice = llm_engine.judge_intent(text)
                row.update(pred_intent=intent.name, intent_score=choice.score,
                           intent_ok=intent.name == r.intent)

            names = list((cands.get(r.id) or {}).get("names") or [])
            if "category" in tasks and r.category_label and names:
                allow_none = r.category_allows_none
                labels = prompts.category_labels(names, allow_none)
                row["candidates"] = "/".join(labels)
                row["gold_category"] = r.category
                if r.category in labels:
                    choice = llm_engine.judge_category(text, names, allow_none)
                    row.update(pred_category=choice.label, category_score=choice.score,
                               category_ok=choice.label == r.category,
                               gold_position=labels.index(r.category) + 1)
                    if allow_none:
                        none_gold = r.category == prompts.CATEGORY_NONE
                        row["none_gold"] = none_gold
                        row["none_pred"] = choice.label == prompts.CATEGORY_NONE
                else:
                    row["category_miss"] = True   # ④ 가 정답을 후보에 못 올림 (⑤ 책임 아님)

            if "tool" in tasks and r.tool:
                cat = category_for_tool_prompt(r, cands)
                call = llm_engine.generate_tool_call(text, r.intent_obj, cat)
                own = llm_engine._parse_tool_json(call.raw, None)   # 교정 전 모델 자신의 출력
                gold = r.tool_arguments()
                pred = call.arguments if isinstance(call.arguments, dict) else {}
                row.update(tool=r.tool, raw=call.raw, gold_json=prompts.format_tool_call(r.tool, gold),
                           parse_ok=call.parse_error is None, name_ok=own.name == r.tool)
                if call.parse_error is None:
                    _score_arguments(row, r, gold, pred)
            rows.append(row)

    report = EvalReport(label=label, rows=rows, seconds=time.perf_counter() - started,
                        kv_cache=llm_engine.kv_status())
    report.metrics = _aggregate(rows)
    tags = sorted({t for row in rows for t in row["tags"].split(",") if t})
    report.by_tag = {t: _aggregate([row for row in rows if t in row["tags"].split(",")]) for t in tags}
    return report


def _score_arguments(row: dict, r: GoldRecord, gold: dict, pred: dict) -> None:
    def s(key: str) -> str:
        return str(pred.get(key, "") if pred.get(key) is not None else "").strip()

    if "location" in gold:
        g, p = str(gold["location"]).strip(), s("location")
        row.update(gold_location=g, pred_location=p, location_ok=_squash(g) == _squash(p),
                   location_hallucinated=is_hallucinated_location(p, r.text))
        if not g:
            row["location_blank_kept"] = not p
    if "content" in gold:
        g, p = str(gold["content"]).strip(), s("content")
        row.update(gold_content=g, pred_content=p)
        if r.tool == "update_complaint" and not g:
            row["update_blank_ok"] = not p
        elif g:
            row["content_score"] = round(char_rouge_l(p, g), 4)
    if r.tool == "update_complaint" and "location" in gold and not str(gold["location"]).strip():
        row["update_blank_ok"] = row.get("update_blank_ok", True) and not s("location")
    if "complaint_id" in gold:
        row.update(gold_id=gold["complaint_id"], pred_id=pred.get("complaint_id"),
                   id_ok=_as_int(gold["complaint_id"]) == _as_int(pred.get("complaint_id")))
    if "field" in gold:
        row.update(gold_field=gold["field"], pred_field=s("field"), field_ok=gold["field"] == s("field"))
    for key in ("keyword", "period"):
        if key in gold:
            g, p = str(gold[key]).strip(), s(key)
            row.update({f"gold_{key}": g, f"pred_{key}": p, f"{key}_ok": _squash(g) == _squash(p)})
            made_up = bool(p) and _squash(p) not in _squash(r.text)
            row["find_hallucinated"] = row.get("find_hallucinated", False) or made_up
    if "reason" in gold:
        g, p = str(gold["reason"]).strip(), s("reason")
        row.update(gold_reason=g, pred_reason=p, reason_score=round(char_rouge_l(p, g), 4))


def _aggregate(rows: list[dict]) -> dict[str, dict]:
    def col(key: str) -> list:
        return [row[key] for row in rows if key in row]

    intent_rows = [row for row in rows if "pred_intent" in row]
    oos = [row["pred_intent"] == "해당없음" for row in intent_rows if row["gold_intent"] == "해당없음"]
    false_reject = [row["pred_intent"] == "해당없음" for row in intent_rows if row["gold_intent"] != "해당없음"]
    cat_hard = [row["category_ok"] for row in rows
                if row.get("gold_position", 1) > 1 and "category_ok" in row and not row.get("none_gold")]
    none_recall = [row["none_pred"] for row in rows if row.get("none_gold") is True]
    none_false = [row["none_pred"] for row in rows if row.get("none_gold") is False]
    return {
        "의도 정확도": _rate(col("intent_ok")),
        "해당없음 재현율": _rate(oos),
        "오반려율": _rate(false_reject),
        "카테고리 정확도": _rate(col("category_ok")),
        "카테고리(정답 2위 이하) 정확도": _rate(cat_hard),
        "카테고리 없음 재현율": _rate(none_recall),
        "카테고리 없음 오판율": _rate(none_false),
        "④ 놓침(참고)": {"value": len(col("category_miss")), "n": len(col("category_miss"))},
        "JSON 파싱 성공률": _rate(col("parse_ok")),
        "도구 이름 일치율": _rate(col("name_ok")),
        "location 정확 일치": _rate(col("location_ok")),
        "location 지어냄 비율": _rate(col("location_hallucinated")),
        "위치없음 빈칸 유지율": _rate(col("location_blank_kept")),
        "content 유사도": _mean(col("content_score")),
        "complaint_id 정확 일치": _rate(col("id_ok")),
        "field 정확 일치": _rate(col("field_ok")),
        "keyword 정확 일치": _rate(col("keyword_ok")),
        "period 정확 일치": _rate(col("period_ok")),
        "찾기 조건 지어냄 비율": _rate(col("find_hallucinated")),
        "수정 빈칸 규칙 준수": _rate(col("update_blank_ok")),
        "reason 유사도": _mean(col("reason_score")),
    }


# =============================================================================
# 학습 전후 비교 + 판정
# =============================================================================
def compare(before: EvalReport, after: EvalReport, min_delta: float = 0.02) -> list[dict]:
    """
    지표별 전후 비교표.

    차이가 min_delta(기본 2%p)보다 작거나, 평가 건수 기준 한 건(1/n) 이하이면 '비슷' 으로 봅니다.
    평가셋이 작으면 한 건 차이로 점수가 크게 흔들리기 때문입니다.
    """
    table = []
    for name, spec in METRICS.items():
        b, a = before.metrics.get(name, {}), after.metrics.get(name, {})
        bv, av, n = b.get("value"), a.get("value"), a.get("n") or b.get("n") or 0
        row = {"지표": name, "역할": spec["role"], "학습 전": bv, "학습 후": av, "평가 건수": n,
               "변화": None, "판단": "측정 안 됨"}
        if bv is not None and av is not None and n:
            delta = av - bv
            threshold = max(min_delta, 1.0 / n)
            good = delta if spec["better"] == "high" else -delta
            row["변화"] = round(delta, 4)
            row["판단"] = "좋아짐" if good > threshold else "나빠짐" if good < -threshold else "비슷"
        table.append(row)
    return table


def verdict(table: list[dict]) -> tuple[str, list[str]]:
    """
    판정 기준
      - 목표 지표 중 하나 이상이 '좋아짐'
      - 방어 지표 중 '나빠짐' 이 없음
      - 'location 지어냄 비율' 이 '나빠짐' 이 아님 (실무 위험이 가장 큰 지표)
    """
    reasons = []
    improved = [r["지표"] for r in table if r["역할"] == "목표" and r["판단"] == "좋아짐"]
    worsened_guard = [r["지표"] for r in table if r["역할"] == "방어" and r["판단"] == "나빠짐"]
    worsened_target = [r["지표"] for r in table if r["역할"] == "목표" and r["판단"] == "나빠짐"]
    halluc = next((r for r in table if r["지표"] == "location 지어냄 비율"), None)

    if improved:
        reasons.append(f"좋아진 목표 지표: {improved}")
    else:
        reasons.append("좋아진 목표 지표가 없습니다.")
    if worsened_guard:
        reasons.append(f"나빠진 방어 지표: {worsened_guard} - 기존 능력이 흔들렸습니다.")
    if worsened_target:
        reasons.append(f"나빠진 목표 지표: {worsened_target}")
    if halluc and halluc["판단"] == "나빠짐":
        reasons.append("위치를 지어내는 비율이 늘었습니다. 운영에 쓰면 위험합니다.")

    ok = bool(improved) and not worsened_guard and not (halluc and halluc["판단"] == "나빠짐")
    return ("성공 - 운영에 써 볼 만합니다" if ok else "보류 - 데이터나 설정을 고쳐 다시 학습하세요"), reasons


def failures(report: EvalReport, task: str, limit: int = 10) -> list[dict]:
    """틀린 사례 모음 (task: intent / category / location / content / parse / id / keyword / period)."""
    key = {"intent": "intent_ok", "category": "category_ok", "location": "location_ok",
           "parse": "parse_ok", "id": "id_ok", "keyword": "keyword_ok", "period": "period_ok"}.get(task)
    if task == "content":
        rows = sorted((r for r in report.rows if "content_score" in r), key=lambda r: r["content_score"])
        return rows[:limit]
    if task == "hallucination":
        return [r for r in report.rows if r.get("location_hallucinated") or r.get("find_hallucinated")][:limit]
    return [r for r in report.rows if key in r and not r[key]][:limit]


# =============================================================================
# 학습 중 점검용 - 몇 건만 빠르게
# =============================================================================
def spot_check(records: list[GoldRecord], cands: dict[str, dict]) -> list[dict]:
    """고정된 몇 건의 의도·도구 출력을 뽑습니다. 학습이 진행되며 출력이 어떻게 바뀌는지 봅니다."""
    out = []
    llm_engine.reset_kv_cache()   # 가중치가 바뀌었으니 캐시를 새로 만듭니다
    with _eval_mode():
        for r in records:
            text = llm_text(r)
            intent, choice = llm_engine.judge_intent(text)
            item = {"id": r.id, "정답 의도": r.intent, "예측 의도": f"{intent.name}({choice.score:.2f})"}
            if r.tool:
                call = llm_engine.generate_tool_call(text, r.intent_obj, category_for_tool_prompt(r, cands))
                item["정답 JSON"] = prompts.format_tool_call(r.tool, r.tool_arguments())
                item["모델 출력"] = call.raw
            out.append(item)
    llm_engine.reset_kv_cache()
    return out
