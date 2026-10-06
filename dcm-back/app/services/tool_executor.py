"""
⑤ 가 만든 도구 호출 JSON 을 실제로 실행합니다. (그림의 ⑤→⑥ 화살표)

Gemma 는 "어떤 도구를, 어떤 값으로 부를지"(tool_call.name / arguments) 만 정합니다.
그 JSON 을 DB 에 실제로 쓰거나 읽는 코드는 전부 여기 있는, Gemma 와 무관한 결정적
(deterministic) 파이썬 함수입니다. Gemma 가 만든 값은 여기서 다시 한번
스키마·소유권·상태를 검증한 뒤에만 실행됩니다 - 모델이 잘못된 id 나 없는
카테고리를 적어도 그대로 실행되지 않습니다. (그림의 '규칙검사')

대상 민원 찾기 (조회·수정·삭제)
  시민은 민원 번호를 기억하지 못하는 경우가 대부분이라, 번호 대신 아래 조건으로도 찾습니다.
    complaint_id : 번호를 말했으면 그 민원 하나 (다른 조건보다 우선)
    category     : ⑤ 가 확정한 카테고리 ("" = 없음, 조건 아님)
    keyword      : 민원 대상 표현. 내용·위치에 글자 그대로 있으면 일치,
                   없으면 bge-m3 의미 유사도 >= SEARCH_SEMANTIC_MIN_SCORE 인 민원
    period       : 시점 표현 ("어제", "지난주") -> period.py 가 날짜 범위로 바꿈
  못 찾으면 조건을 조금 풀어 한 번 더 찾습니다. (카테고리 조건 빼기 -> 기간 앞뒤 하루 넓히기)
  ⑤ 의 카테고리 판정이 틀렸거나, 시민이 날짜를 하루 정도 착각한 경우를 구제합니다.

결과
  조회 : 찾은 민원이 1건이면 상태(또는 이력), 여러 건이면 목록.
  수정·삭제 :
    - 조건이 하나도 없으면 실행하지 않습니다. (어떤 민원인지 알 수 없음)
      진행 중 민원 목록을 needs_selection=True 로 돌려주어 화면에서 고르게 합니다.
    - 찾은 민원이 여러 건이어도 실행하지 않고 전부 돌려줍니다. (needs_selection=True)
    - 정확히 1건일 때만 바로 실행합니다. (별도의 확인 절차 없음)
  이미 취소된 민원은 수정·삭제 대상에서 뺍니다.

문의(intent.tool is None)는 도구가 없으므로 이 모듈을 거치지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import settings
from app.logging_config import get_logger
from app.services import categories, complaint_store, period as period_mod
from app.services.complaint_store import Complaint
from app.services.llm_engine import ToolCall

logger = get_logger(__name__)


@dataclass
class ToolExecutionResult:
    executed: bool = False          # DB 에 실제로 반영됐는지 (검증 실패면 False)
    ok: bool = True                 # 검증을 통과했는지. False 면 error 를 보세요
    message: str = ""               # 사람이 읽는 결과 문장 (그림의 ⑦ 사용자 응답)
    data: object = None             # 구조화된 결과 (dict 1건 또는 list, API 응답용)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    # 수정·삭제 대상이 여러 건이거나 특정되지 않아 실행하지 않았을 때 True.
    # data 에 후보 민원 목록이 들어 있으니, 화면에서 사용자가 하나를 고르게 하면 됩니다.
    needs_selection: bool = False
    # 어떤 조건으로 찾았는지 (번호·카테고리·키워드·기간, 조건을 풀었는지)
    search: dict | None = None


def _to_dict(c: Complaint) -> dict:
    return {
        "id": c.id, "category": c.category, "content": c.content, "location": c.location,
        "status": c.status, "department": c.department,
        "created_at": c.created_at, "updated_at": c.updated_at,
    }


def _parse_id(value: object) -> int | None:
    """모델이 문자열("42")·실수(42.0)·None 등으로 낼 수 있어 관대하게 파싱합니다."""
    if value is None:
        return None
    try:
        n = int(float(value))
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def execute(tool_call: ToolCall, user_id: str) -> ToolExecutionResult:
    """도구 이름으로 분기해 실제 DB 작업을 실행합니다."""
    if tool_call.parse_error or not tool_call.called:
        return ToolExecutionResult(
            ok=False,
            error=f"도구 호출 JSON 이 없어 실행하지 못했습니다. ({tool_call.parse_error or '빈 응답'})",
        )

    args = tool_call.arguments if isinstance(tool_call.arguments, dict) else {}
    name = tool_call.name

    if name == "register_complaint":
        return _register(args, user_id)
    if name == "get_complaints":
        return _search(args, user_id)
    if name == "update_complaint":
        return _update(args, user_id)
    if name == "cancel_complaint":
        return _cancel(args, user_id)
    return ToolExecutionResult(ok=False, error=f"알 수 없는 도구입니다: {name!r}")


# =============================================================================
# 접수 - 중복 허용, 바로 INSERT
# =============================================================================
def _register(args: dict, user_id: str) -> ToolExecutionResult:
    content = str(args.get("content") or "").strip()
    if not content:
        return ToolExecutionResult(ok=False, error="접수할 민원 내용이 비어 있습니다.")

    category = str(args.get("category") or "")
    location = str(args.get("location") or "")
    complaint = complaint_store.register(user_id, category, content, location)

    where = f" ({complaint.location})" if complaint.location else ""
    return ToolExecutionResult(
        executed=True,
        message=f"접수 완료 (id={complaint.id}) - {complaint.category}{where} / 담당 {complaint.department}",
        data=_to_dict(complaint),
    )


# =============================================================================
# 대상 민원 찾기 (조회·수정·삭제 공통)
# =============================================================================
@dataclass
class FindResult:
    matches: list[Complaint] = field(default_factory=list)
    has_condition: bool = False          # 번호·카테고리·키워드·기간 중 하나라도 있었는지
    error: str | None = None             # 번호로 찾았는데 없을 때
    info: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)   # 조건을 풀었거나 해석하지 못한 기록

    def describe(self) -> str:
        parts = []
        if self.info.get("complaint_id"):
            parts.append(f"{self.info['complaint_id']}번")
        if self.info.get("category"):
            parts.append(self.info["category"])
        if self.info.get("keyword"):
            parts.append(f"'{self.info['keyword']}'")
        if self.info.get("period_text"):
            parts.append(self.info["period_text"])
        return " · ".join(parts) or "조건 없음"


def _norm(text: str) -> str:
    return "".join(str(text or "").split()).lower()


def _keyword_filter(pool: list[Complaint], keyword: str, notes: list[str]) -> list[Complaint]:
    """글자 포함 우선, 없으면 의미 유사도로 보조."""
    key = _norm(keyword)
    exact = [c for c in pool if key and (key in _norm(c.content) or key in _norm(c.location))]
    if exact or not pool:
        return exact
    try:
        from app.services import embedder

        if not embedder.is_installed():
            return []
        texts = [f"{c.content} {c.location}".strip() for c in pool]
        vectors = embedder.encode([keyword] + texts)
        sims = embedder.cosine(vectors[0], vectors[1:])[0]
    except Exception as exc:   # 모델이 없는 환경(PC 추출 전용) 등 - 글자 포함만으로 판단
        logger.info("민원 찾기 - 의미 유사도 생략 (%s)", exc)
        return []
    floor = settings.SEARCH_SEMANTIC_MIN_SCORE
    ranked = sorted(
        ((c, float(s)) for c, s in zip(pool, sims) if s >= floor), key=lambda x: -x[1]
    )
    if ranked:
        notes.append(
            f"'{keyword}' 이(가) 글자 그대로 있는 민원이 없어 의미가 비슷한 민원으로 찾음 "
            f"(유사도 {ranked[0][1]:.2f}~{ranked[-1][1]:.2f}, 기준 {floor:.2f})"
        )
    return [c for c, _ in ranked]


def _find(args: dict, user_id: str, exclude_cancelled: bool) -> FindResult:
    cid = _parse_id(args.get("complaint_id"))
    category = str(args.get("category") or "").strip()
    if category and category not in categories.NAMES:
        # 7종 이름·코드(예전 9종 이름 포함)가 아니면(없음 등) 카테고리 조건으로 쓰지 않습니다. ('기타'로 바꾸면 엉뚱하게 걸러짐)
        legacy = categories.LEGACY.get(category) or categories.LEGACY.get(category.lower())
        known = categories.BY_NAME.get(legacy) if legacy else categories.BY_CODE.get(category.lower())
        category = known.name if known else ""
    keyword = str(args.get("keyword") or "").strip()
    period_text = str(args.get("period") or "").strip()

    res = FindResult(info={"complaint_id": cid or 0, "category": category, "keyword": keyword,
                           "period_text": period_text})

    # --- 번호가 있으면 그 민원만 ---
    if cid is not None:
        res.has_condition = True
        complaint = complaint_store.get_one(user_id, cid)
        if complaint is None:
            res.error = f"{cid}번 민원을 찾을 수 없거나 본인 민원이 아닙니다."
        else:
            res.matches = [complaint]
        return res

    period = period_mod.parse(period_text) if period_text else None
    if period_text and period is None:
        res.notes.append(f"시점 표현 '{period_text}' 을(를) 해석하지 못해 기간 조건 없이 찾음")
    if period is not None:
        res.info["period"] = period.describe()

    res.has_condition = bool(category or keyword or (period and not period.vague))
    pool = complaint_store.list_for_search(user_id, exclude_cancelled=exclude_cancelled)
    if not res.has_condition:
        res.matches = pool
        return res

    def run(use_category: bool, per: period_mod.Period | None) -> list[Complaint]:
        rows = pool
        if per is not None and not per.vague and not per.latest:
            rows = [c for c in rows if (m := period_mod.to_local(c.created_at)) and per.contains(m)]
        if use_category and category:
            rows = [c for c in rows if c.category == category]
        if keyword:
            rows = _keyword_filter(rows, keyword, res.notes)
        if per is not None and per.latest and rows:
            rows = rows[:1]          # pool 이 최근 등록순이므로 첫 번째가 가장 최근
        return rows

    matches = run(True, period)
    if not matches and category and (keyword or period):
        matches = run(False, period)
        if matches:
            res.notes.append(f"카테고리 '{category}' 조건을 빼고 찾음 (카테고리 판정이 달랐을 수 있음)")
    if not matches and period is not None and period.start is not None:
        wider = period.widened(1)
        matches = run(bool(category), wider) or (run(False, wider) if category else [])
        if matches:
            res.notes.append(f"기간을 앞뒤 하루 넓혀 찾음 ({wider.describe()})")
    res.matches = matches
    return res


def _search_meta(found: FindResult) -> dict:
    return {**found.info, "condition": found.describe(), "notes": found.notes, "matched": len(found.matches)}


def _selection(found: FindResult, action: str, reason: str) -> ToolExecutionResult:
    """수정·삭제 대상을 하나로 정하지 못함 -> 실행하지 않고 후보를 돌려줌."""
    rows = found.matches
    summary = ", ".join(f"{c.id}번({c.category} / {c.content[:20]})" for c in rows[:10])
    more = f" 외 {len(rows) - 10}건" if len(rows) > 10 else ""
    return ToolExecutionResult(
        executed=False,
        ok=True,
        needs_selection=True,
        message=f"{reason} {action}할 민원을 골라 주세요. - {summary}{more}" if rows
        else f"{reason} 진행 중인 민원이 없습니다.",
        data=[_to_dict(c) for c in rows],
        search=_search_meta(found),
        warnings=list(found.notes),
    )


# =============================================================================
# 조회 - 읽기 전용
# =============================================================================
def _search(args: dict, user_id: str) -> ToolExecutionResult:
    field_name = str(args.get("field") or "list").strip().lower()
    found = _find(args, user_id, exclude_cancelled=False)
    meta = _search_meta(found)
    if found.error:
        return ToolExecutionResult(ok=False, error=found.error, search=meta)

    rows = found.matches
    if not rows:
        msg = "접수하신 민원이 없습니다." if not found.has_condition else f"조건({found.describe()})에 맞는 민원이 없습니다."
        return ToolExecutionResult(executed=True, message=msg, data=[], search=meta, warnings=list(found.notes))

    if len(rows) > 1 or field_name == "list":
        shown = rows[:20]
        summary = ", ".join(f"{c.id}번({c.category}/{c.status})" for c in shown)
        head = f"조건({found.describe()})에 맞는 민원" if found.has_condition else "민원"
        return ToolExecutionResult(
            executed=True, message=f"{head} {len(rows)}건 - {summary}",
            data=[_to_dict(c) for c in shown], search=meta, warnings=list(found.notes),
        )

    complaint = rows[0]
    cid = complaint.id
    if field_name == "history":
        hist = complaint_store.history(user_id, cid) or []
        lines = "; ".join(f"{h.changed_at} {h.status}" + (f"({h.note})" if h.note else "") for h in hist)
        return ToolExecutionResult(
            executed=True, message=f"{cid}번 민원 이력 {len(hist)}건 - {lines}",
            data=[{"status": h.status, "note": h.note, "changed_at": h.changed_at} for h in hist],
            search=meta, warnings=list(found.notes),
        )

    # field=status 또는 그 외 -> 현재 상태 1건
    return ToolExecutionResult(
        executed=True,
        message=(
            f"{cid}번 민원 상태: {complaint.status} "
            f"({complaint.category} / {complaint.content[:30]} / {complaint.location or '위치 없음'} / 담당 {complaint.department})"
        ),
        data=_to_dict(complaint), search=meta, warnings=list(found.notes),
    )


def _one_target(args: dict, user_id: str, action: str) -> tuple[Complaint | None, ToolExecutionResult | None, FindResult]:
    """수정·삭제 대상 1건을 정합니다. 못 정하면 (None, 돌려줄 결과)."""
    found = _find(args, user_id, exclude_cancelled=True)
    if found.error:
        return None, ToolExecutionResult(ok=False, error=found.error, search=_search_meta(found)), found
    if not found.has_condition:
        return None, _selection(found, action, "어떤 민원인지 알 수 없어 실행하지 않았습니다."), found
    if not found.matches:
        return None, ToolExecutionResult(
            ok=False, error=f"조건({found.describe()})에 맞는 진행 중 민원이 없습니다.",
            search=_search_meta(found), warnings=list(found.notes),
        ), found
    if len(found.matches) > 1:
        return None, _selection(
            found, action, f"조건({found.describe()})에 맞는 민원이 {len(found.matches)}건이라 실행하지 않았습니다."
        ), found
    complaint = found.matches[0]
    if complaint.status == "취소":
        return None, ToolExecutionResult(
            ok=False, error=f"{complaint.id}번 민원은 이미 취소된 상태입니다.", search=_search_meta(found)
        ), found
    return complaint, None, found


# =============================================================================
# 수정 - 대상 1건이 정해지면 바로 UPDATE
# =============================================================================
def _update(args: dict, user_id: str) -> ToolExecutionResult:
    new_content = str(args.get("content") or "").strip()
    new_location = str(args.get("location") or "").strip()
    if not new_content and not new_location:
        return ToolExecutionResult(ok=False, error="바꿀 내용이나 위치 중 최소 하나는 있어야 합니다.")

    complaint, early, found = _one_target(args, user_id, "수정")
    if early is not None:
        return early
    cid = complaint.id

    changes = []
    if new_content and new_content != complaint.content:
        changes.append(f"내용 '{complaint.content}' → '{new_content}'")
    if new_location and new_location != complaint.location:
        changes.append(f"위치 '{complaint.location or '(없음)'}' → '{new_location}'")
    if not changes:
        return ToolExecutionResult(ok=False, error="기존 값과 같아 바뀌는 내용이 없습니다.", search=_search_meta(found))

    updated = complaint_store.update(user_id, cid, new_content, new_location)
    summary = f"{cid}번 민원: " + " / ".join(changes)
    return ToolExecutionResult(
        executed=True, message=f"수정 완료 - {summary}", data=_to_dict(updated),
        search=_search_meta(found), warnings=list(found.notes),
    )


# =============================================================================
# 취소(삭제) - 대상 1건이 정해지면 바로 soft delete
# =============================================================================
def _cancel(args: dict, user_id: str) -> ToolExecutionResult:
    complaint, early, found = _one_target(args, user_id, "취소")
    if early is not None:
        return early
    cid = complaint.id

    reason = str(args.get("reason") or "").strip()
    updated = complaint_store.cancel(user_id, cid, reason)
    summary = f"{cid}번 민원({complaint.category} / {complaint.content[:30]})을 취소" + (
        f" - 사유: {reason}" if reason else ""
    )
    return ToolExecutionResult(
        executed=True, message=f"취소 완료 - {summary}", data=_to_dict(updated),
        search=_search_meta(found), warnings=list(found.notes),
    )
