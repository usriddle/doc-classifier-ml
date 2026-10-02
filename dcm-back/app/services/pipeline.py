"""
②③④⑤ 오케스트레이션.

  ② 추출 원문
     -> ②→③ 키워드 3개 + 요약문        (keyphrase)
     -> ③  bge-m3 임베딩 1024차원       (embedder)
     -> ④  코사인 유사도 top-4          (candidates, 카테고리 후보만 좁힘 - 게이트 아님)
        └ 벡터DB 사례로 top-4 재정렬 (case_store, CASE_MODE: 애매할 때만/매번/매번+후보 보장)
     -> ⑤  Qwen 의도/카테고리/도구 JSON (llm_engine)
        └ 의도가 '문의' 면 FAQ 검색(faq_store) 후 비슷한 FAQ 만 넣어 즉답 (없으면 고정 문구)
        -> ⑥  민원 DB 실제 실행 (tool_executor, 의도가 접수/조회/수정/삭제일 때만)

게이트(반려 여부)는 ⑤ 의 의도 판정 결과로 정합니다. 의도는 6지선다이며
1~5(문의/접수/조회/수정/삭제) 중 하나면 통과, 6("해당없음")이면 반려입니다.
카테고리 주제와는 무관합니다 - "제가 어제 문의한 내용 보여줘"처럼 도로·환경 같은
카테고리와 상관없는 조회 요청도 의도만 명확하면 통과합니다.

⑤ 가 만든 도구 호출 JSON 은 ⑥ 에서 tool_executor 가 실제로 실행합니다.
접수·조회·수정·삭제 모두 (소유권·상태 등) 규칙검사를 통과하면 바로 실행됩니다.
(자세한 설계는 tool_executor.py 상단 설명 참고)

⑦ 사용자 응답은 render_result_text() 가 사람이 읽을 수 있는 텍스트로 정리해서 돌려줍니다.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from app.config import settings
from app.exceptions import NoTextError
from app.logging_config import get_logger
from app.services import candidates, case_store, categories, complaint_store, embedder, faq_store, keyphrase, llm_engine, tool_executor
from app.services.candidates import CandidateResult
from app.services.case_store import CaseLookup
from app.services.keyphrase import KeyphraseResult
from app.services.llm_engine import LlmResult
from app.services.tool_executor import ToolExecutionResult

logger = get_logger(__name__)


def _pad(text: str, width: int) -> str:
    """한글(전각) 폭을 고려해 오른쪽을 공백으로 채웁니다. (텍스트 표 정렬용)"""
    display = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(0, width - display)


# 반려 시 사용자에게 돌려줄 안내 문구 (⑦ 응답을 대신합니다)
REJECT_MESSAGE = (
    "죄송합니다. 문의·접수·조회·수정·삭제 중 무엇을 원하시는지 확인하지 못해 "
    "처리하지 못했습니다. 다시 한번 말씀해 주시겠어요? "
    "(예: '가로등이 꺼졌어요' - 접수 / '제가 어제 넣은 민원 보여주세요' - 조회)"
)


@dataclass
class GateDecision:
    """
    게이트 판정.

    ⑤ 의 의도 판정(6지선다) 결과를 기준으로 합니다. 1~5(문의/접수/조회/수정/삭제)
    중 하나로 판정되면 통과, 6("해당없음")으로 판정되면 반려입니다.
    카테고리 유사도(④)는 더 이상 반려 기준이 아니며, 후보를 좁히는 데만 쓰입니다.
    """

    passed: bool                 # True 면 결과를 그대로 사용
    intent_score: float          # 선택된 의도(통과든 해당없음이든)의 확신도
    enabled: bool                # GATE_ENABLED
    reason: str = ""             # 반려(또는 관찰 모드 통과) 사유 (로그·debug 용)


@dataclass
class PipelineResult:
    source_text: str                       # ② 추출 원문 (정리 후)
    keyphrase: KeyphraseResult
    candidate: CandidateResult
    gate: GateDecision
    llm: LlmResult | None = None           # ⑤ 는 항상 호출됩니다. 반려 여부는 gate.passed 로 판단하세요.
    case_lookup: CaseLookup | None = None  # ④ 가 애매해 벡터DB 를 열었을 때만 채워집니다
    tool_result: ToolExecutionResult | None = None  # ⑥ 실행 결과 (문의/반려면 None)
    embedding_dim: int = 0
    embedding_preview: list[float] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return not self.gate.passed


def _check_gate(llm: LlmResult) -> GateDecision:
    """
    ⑤ 의도 판정 기준 게이트.

    Qwen 이 6지선다(문의/접수/조회/수정/삭제/해당없음) 중 "해당없음"을 고르면
    반려합니다. 카테고리(④)와 무관하므로 "제가 어제 문의한 내용 보여줘"처럼
    도로·환경 같은 7종 어디와도 뚜렷이 겹치지 않는 문장도, 의도(조회)만 명확하면
    통과합니다.

    llm_engine.decide() 는 "해당없음"일 때 카테고리 확정·도구 호출을 생략하고
    바로 반환하므로, 여기서는 그 결과를 보고 통과/반려만 가릅니다.
    """
    enabled = settings.GATE_ENABLED
    intent_score = llm.intent_choice.score

    if llm.intent.code != "out_of_scope":
        return GateDecision(True, intent_score, enabled)

    reason = (
        f"의도 판정 결과 '해당없음' (확신 {intent_score:.4f}) - "
        "문의·접수·조회·수정·삭제 중 어디에도 해당하지 않는다고 판단"
    )
    if not enabled:
        # 관찰 모드: 반려 대상이지만 통과시키고 표시만 합니다.
        return GateDecision(True, intent_score, enabled, reason + " / GATE_ENABLED=false 라 통과")
    return GateDecision(False, intent_score, enabled, reason)


@dataclass
class CandidateStage:
    """②→④ 까지의 결과. (⑤ 에 넘길 후보 이름 목록 포함)"""

    text: str
    keyphrase: KeyphraseResult
    candidate: CandidateResult
    case_lookup: CaseLookup | None
    timings_ms: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def candidate_names(self) -> list[str]:
        return [c.name for c in self.candidate.top]


def run_candidates(text: str, request_id: str = "-") -> CandidateStage:
    """
    정리된 원문 하나를 ②→③→④(+벡터DB 재정렬)까지만 통과시킵니다.

    run() 이 이 함수를 그대로 쓰고, 학습 노트북도 카테고리 학습 샘플의 후보 4개를
    이 함수로 만듭니다. 그래서 학습 때 보는 후보 분포가 운영과 같습니다.
    """
    timings: dict[str, int] = {}
    warnings: list[str] = []

    # --- ②→③ 키워드 3개 + 요약문 -------------------------------------
    started = time.perf_counter()
    kp = keyphrase.build_query(text)
    timings["keyphrase_ms"] = int((time.perf_counter() - started) * 1000)
    warnings.extend(kp.warnings)
    logger.info(
        "[%s] ②→③ 키워드=%s | 요약=%d자 | method=%s | %dms",
        request_id, kp.keyword_texts, len(kp.summary), kp.method, timings["keyphrase_ms"],
    )

    # --- ③ 임베딩 + ④ 후보 추림 ---------------------------------------
    started = time.perf_counter()
    cand = candidates.select(kp.query_text)
    timings["candidate_ms"] = int((time.perf_counter() - started) * 1000)
    logger.info(
        "[%s] ④ 후보 top%d=%s | %dms",
        request_id, len(cand.top),
        [(c.name, c.score) for c in cand.top], timings["candidate_ms"],
    )

    # --- ④ 보조 : 벡터DB (CASE_MODE: 애매할 때만 / 매번 / 매번+후보 보장) ----
    lookup: CaseLookup | None = None
    if settings.CASE_STORE_ENABLED and case_store.should_consult(cand.low_confidence):
        started = time.perf_counter()
        cand, lookup = case_store.rerank(cand)
        timings["case_ms"] = int((time.perf_counter() - started) * 1000)
        if not lookup.used:
            warnings.append(f"벡터DB: {lookup.reason}")
        logger.info(
            "[%s] ④ 벡터DB | %s | 사례=%s | 재정렬 후 top%d=%s | %dms",
            request_id, "반영" if lookup.used else "미반영",
            [(h.category, h.score) for h in lookup.hits], len(cand.top),
            [(c.name, c.score) for c in cand.top], timings["case_ms"],
        )

    return CandidateStage(
        text=text, keyphrase=kp, candidate=cand, case_lookup=lookup,
        timings_ms=timings, warnings=warnings,
    )


def run(
    source_text: str,
    request_id: str = "-",
    user_id: str = "",
) -> PipelineResult:
    """
    추출 원문 하나를 ②→⑥ 까지 통과시킵니다.

    user_id : "로그인한 사용자" 취급할 식별자. 비우면 settings.COMPLAINT_DEFAULT_USER 를 씁니다.
              같은 문장이라도 user_id 가 다르면 서로의 민원을 조회·수정·취소할 수 없습니다.
    """
    text = keyphrase.normalize(source_text)
    if not text:
        raise NoTextError(
            "텍스트가 비어 있어 분류할 수 없습니다.",
            detail="OCR 결과가 없거나 내용이 없는 파일일 수 있습니다.",
        )
    user_id = (user_id or settings.COMPLAINT_DEFAULT_USER).strip() or settings.COMPLAINT_DEFAULT_USER

    stage = run_candidates(text, request_id)
    kp, cand, lookup = stage.keyphrase, stage.candidate, stage.case_lookup
    timings: dict[str, int] = dict(stage.timings_ms)
    warnings: list[str] = list(stage.warnings)

    embedding_dim = 0
    preview: list[float] = []
    if cand.query_vector is not None:
        embedding_dim = int(cand.query_vector.shape[0])
        preview = [round(float(v), 4) for v in cand.query_vector[: settings.DEBUG_VECTOR_PREVIEW]]

    # --- ⑤ Qwen 판정 (의도 6지선다 -> 해당없음이면 카테고리·도구는 내부에서 생략) ---
    started = time.perf_counter()
    llm = llm_engine.decide(text, [c.name for c in cand.top])
    timings["llm_ms"] = int((time.perf_counter() - started) * 1000)
    timings.update(llm.timings_ms)
    warnings.extend(llm.warnings)
    logger.info(
        "[%s] ⑤ 의도=%s(%.2f) | 카테고리=%s(%.2f) | 도구=%s | %dms",
        request_id, llm.intent.name, llm.intent_choice.score,
        llm.category_name or "-", llm.category_choice.score,
        llm.tool_call.name or "-", timings["llm_ms"],
    )

    # --- 게이트 : ⑤ 의 의도 판정이 "해당없음"이면 반려 --------------------
    # ⑤ 는 이미 위에서 호출을 마쳤으므로, 여기서는 그 결과(llm)를 보고 통과/반려만
    # 가릅니다. llm 은 반려된 경우에도 그대로 담아 반환해 debug 에서 확인할 수 있게 합니다.
    gate = _check_gate(llm)
    if gate.reason:
        warnings.append(gate.reason)
    if not gate.passed:
        logger.info("[%s] 게이트 반려 | %s", request_id, gate.reason)

    # --- ⑥ 민원 DB : 도구 호출 실제 실행 (통과 + 도구가 있는 의도일 때만) ---
    tool_result: ToolExecutionResult | None = None
    if gate.passed and llm.intent.tool is not None:
        started = time.perf_counter()
        tool_result = tool_executor.execute(llm.tool_call, user_id=user_id)
        timings["tool_exec_ms"] = int((time.perf_counter() - started) * 1000)
        if not tool_result.ok:
            warnings.append(f"⑥ 실행 실패: {tool_result.error}")
        warnings.extend(f"⑥ {w}" for w in tool_result.warnings)
        logger.info(
            "[%s] ⑥ 민원DB | user=%s | 도구=%s | 실행=%s | %s | %dms",
            request_id, user_id, llm.tool_call.name, tool_result.executed,
            tool_result.error or tool_result.message, timings["tool_exec_ms"],
        )

    return PipelineResult(
        source_text=text,
        keyphrase=kp,
        candidate=cand,
        gate=gate,
        llm=llm,
        case_lookup=lookup,
        tool_result=tool_result,
        embedding_dim=embedding_dim,
        embedding_preview=preview,
        timings_ms=timings,
        warnings=warnings,
    )


# =============================================================================
# 텍스트 리포트 - Swagger 에서 눈으로 확인하는 용도
# =============================================================================
def render_result_text(result: PipelineResult) -> str:
    """⑤ 판정 결과를 사람이 읽는 텍스트 한 덩어리로 만듭니다."""
    if result.rejected:
        return "\n".join([
            "===== 게이트 (⑤ 의도 판정) : 반려 =====",
            "처리      : 하지 않음 (의도 '해당없음' - 카테고리 확정·도구 호출 생략)",
            f"사유      : {result.gate.reason}",
            f"안내 문구 : {REJECT_MESSAGE}",
        ])

    llm = result.llm

    if llm.intent.code == "out_of_scope":
        # GATE_ENABLED=false 인 관찰 모드에서만 여기까지 옵니다.
        # (켜져 있으면 이 경우 result.rejected 가 True 라 위에서 이미 반환됩니다)
        return "\n".join([
            "===== ⑤ 판정 결과 =====",
            f"의도      : 해당없음 (확신 {llm.intent_choice.score:.2f}) - GATE_ENABLED=false 라 통과됨",
            "카테고리  : 판정 안 함 (해당없음이라 생략)",
            "도구 호출 : 없음",
        ])

    if llm.category_name:
        category = categories.get(llm.category_name)
        category_line = f"{category.name} (확신 {llm.category_choice.score:.2f}) / 담당 {category.department}"
    elif llm.category_choice.scores:
        category_line = f"없음 (확신 {llm.category_choice.score:.2f}) - 문장에서 민원 주제를 특정할 수 없음"
    else:
        category_line = "판정 안 함 (문의는 카테고리를 쓰지 않음)"

    lines = [
        "===== ⑤ 판정 결과 =====",
        f"의도      : {llm.intent.name} (확신 {llm.intent_choice.score:.2f})",
        f"카테고리  : {category_line}",
    ]

    if llm.tool_call.called:
        lines.append(f"도구 호출 : {llm.tool_call.name}")
        lines.append("인자      :")
        for key, value in (llm.tool_call.arguments or {}).items():
            lines.append(f"  - {key} = {value}")

        # --- ⑥ 민원 DB 실행 결과 (그림의 ⑦ 사용자 응답에 해당) ---
        tr = result.tool_result
        lines.append("")
        if tr is None:
            lines.append("⑥ 실행    : (실행 안 함)")
        elif not tr.ok:
            lines.append(f"⑥ 실행    : 실패")
            lines.append(f"오류      : {tr.error}")
        elif tr.needs_selection:
            lines.append("⑥ 실행    : 보류 - 대상 민원을 하나로 정하지 못해 실행하지 않음 (화면에서 선택)")
            lines.append(f"응답      : {tr.message}")
        else:
            lines.append("⑥ 실행    : 완료")
            lines.append(f"응답      : {tr.message}")
        if tr is not None and tr.search:
            lines.append(f"찾은 조건 : {tr.search.get('condition')} → {tr.search.get('matched')}건")
    elif llm.intent.tool is None:
        lines.append("도구 호출 : 없음 (문의 - FAQ 검색 후 즉답, DB 미사용)")
        lines.append(f"즉답      : {llm.answer or '(생성 실패)'}")
        fl = llm.faq_lookup
        if fl is not None:
            if fl.fallback or fl.model_declined:
                lines.append("근거 FAQ : 없음 - 고정 안내 문구로 답함")
            else:
                lines.append("근거 FAQ : " + ", ".join(f"{h.id}({h.score:.2f})" for h in fl.selected))
    else:
        lines.append(f"도구 호출 : 실패 ({llm.tool_call.parse_error})")

    if llm.warnings:
        lines.append("")
        lines.append("경고      : " + " / ".join(llm.warnings))

    return "\n".join(lines)


def _render_faq_debug(llm: LlmResult) -> list[str]:
    """'문의' 답변의 FAQ 검색(RAG) 구간."""
    fl = llm.faq_lookup
    lines = ["===== ⑤ 문의 답변 : FAQ 검색 (카테고리 확정·도구 호출 없음) ====="]
    if fl is None:
        return lines + ["(FAQ 검색 결과 없음)", ""]
    lines.append(f"방식 : {fl.mode} | 기준 FAQ_MIN_SCORE={fl.min_score:.2f} | 1위 {fl.best_score:.4f}")
    lines.append(f"상태 : {fl.reason}")
    if fl.hits:
        lines.append(f"-- 상위 {len(fl.hits)}개 --")
        chosen = {h.id for h in fl.selected}
        for h in fl.hits:
            mark = " <-- 프롬프트에 넣음" if h.id in chosen else ""
            lines.append(f"  {h.rank}. [{h.id}] {h.score:.4f}  {h.question}{mark}")
            if h.matched and h.matched != h.question:
                lines.append(f"       (가장 가까운 표현: {h.matched})")
    elif fl.selected:
        lines.append(f"-- 프롬프트에 넣은 FAQ {len(fl.selected)}개 (검색 없이 전부) --")
        for h in fl.selected:
            lines.append(f"  [{h.id}] {h.question}")
    lines += [
        "",
        "-- Qwen 원시 출력 --",
        fl.model_answer or "(호출 안 함 - 고정 문구)",
        "",
        "-- 최종 답변 --",
        llm.answer or "(없음)",
        "",
    ]
    return lines


def render_debug_text(result: PipelineResult) -> str:
    """DEBUG=true 일 때만 나가는 단계별 전체 덤프."""
    kp = result.keyphrase
    cand = result.candidate
    llm = result.llm

    keyword_line = (
        ", ".join(f"{k.text}({k.score:.3f})" for k in kp.keywords) if kp.keywords else "(없음)"
    )

    lines = [
        "===== ② 추출 원문 =====",
        result.source_text,
        "",
        f"(원문 {len(result.source_text)}자 / 문장 {len(kp.sentences)}개)",
        "",
        "===== ②→③ 키워드 3개 + 요약문 =====",
        f"키워드({kp.method}) : {keyword_line}",
        f"요약문             : {kp.summary or '(없음)'}",
        f"뺀 상투 문장       : {' / '.join(getattr(kp, 'dropped', []) or []) or '(없음)'}",
        "",
        "-- 임베딩 모델에 넘긴 질의문 --",
        kp.query_text,
        "",
        "===== ③ 임베딩 (bge-m3) =====",
        f"차원 : {result.embedding_dim}",
        f"앞 {len(result.embedding_preview)}개 : {result.embedding_preview}",
        "",
        "===== ④ 카테고리 후보 추림 (코사인 유사도, 게이트 아님) =====",
    ]
    lookup = result.case_lookup
    # 벡터DB 로 재정렬했다면 ④ 원래 점수를 보여 주고, ⑤ 에 넘긴 후보는 아래 구간에서 보여 줍니다.
    top4 = lookup.before if lookup else cand.top
    all4 = lookup.before_all if lookup and lookup.before_all else cand.all_scores
    lines.append(f"-- 상위 {len(top4)}개" + (" (재정렬 전)" if lookup and lookup.used else " (⑤ 카테고리 확정에 전달)") + " --")
    for c in top4:
        lines.append(f"  {c.rank}. {_pad(c.name, 10)} {c.score:.4f}")
    lines += ["", "-- 7종 전체 --"]
    for c in all4:
        lines.append(f"  {c.rank}. {_pad(c.name, 10)} {c.score:.4f}")
    lines += [
        "",
        f"1위 유사도 {top4[0].score:.4f} / 기준 {settings.CANDIDATE_MIN_SCORE:.2f} -> "
        + ("애매함(low_confidence)" if cand.low_confidence else "충분")
        + (" - 벡터DB 조회" if lookup is not None else " - 벡터DB 조회 안 함")
        if top4 else "(후보 없음)",
        "(참고 : 이 값은 반려 기준이 아닙니다. 반려는 아래 ⑤ 의도 판정의 "
        "'해당없음' 여부로만 정합니다)",
        "",
    ]

    if lookup is not None:
        lines.append("===== ④ 보조 : 벡터DB (라벨링된 사례) =====")
        lines.append(f"상태 : {lookup.reason}")
        if lookup.hits:
            lines.append("-- 유사 사례 --")
            for h in lookup.hits:
                lines.append(f"  {h.rank}. [{_pad(h.category, 10)}] {h.score:.4f}  {h.text}")
            lines.append("-- 카테고리별 사례 점수 --")
            for name, score in lookup.case_scores.items():
                lines.append(f"  {_pad(name, 10)} {score:.4f}")
        if lookup.used:
            lines.append(
                f"-- 재정렬 후 상위 {len(lookup.after)}개 (⑤ 카테고리 확정에 전달) | CASE_MODE={case_store.mode()} --"
            )
            for c in lookup.after:
                lines.append(f"  {c.rank}. {_pad(c.name, 10)} {c.score:.4f}")
        lines.append("")

    lines += ["===== ⑤ 의도 판정 (번호 토큰 1회 계산, 6지선다) ====="]
    for name, score in llm.intent_choice.scores.items():
        mark = " <-- 선택" if name == llm.intent_choice.label else ""
        lines.append(f"  {_pad(name, 8)} {score:.4f}{mark}")
    lines += [
        "",
        "===== 게이트 (⑤ 의도 판정 기준) =====",
        f"선택된 의도 : {llm.intent.name} (확신 {result.gate.intent_score:.4f})",
        f"게이트      : {'켜짐' if result.gate.enabled else '꺼짐 (관찰 모드)'}",
        f"판정        : {'통과' if result.gate.passed else '반려'}"
        + (f"  - {result.gate.reason}" if result.gate.reason else ""),
        "",
    ]

    if not result.rejected and llm.intent.code != "out_of_scope" and llm.intent.tool is None:
        # '문의' - 카테고리 확정·도구 호출 없이 FAQ 검색 후 답변
        lines += _render_faq_debug(llm)
        lines.append("===== 소요 시간 =====")
        for key, value in result.timings_ms.items():
            lines.append(f"  {key:<14} {value}ms")
        if result.warnings:
            lines += ["", "===== 경고 ====="]
            lines += [f"  - {w}" for w in result.warnings]
        return "\n".join(lines)

    if result.rejected or not llm.category_choice.scores:
        # 반려된 경우, 또는 GATE_ENABLED=false 관찰 모드에서 '해당없음'이 통과된 경우
        # (둘 다 llm_engine.decide() 가 카테고리 확정·도구 호출을 생략한 상태입니다)
        note = (
            "(의도가 '해당없음'으로 판정되어 생략됨 - forward 1회만 사용)"
            if result.rejected else
            "(GATE_ENABLED=false 관찰 모드 - '해당없음'이 통과되어도 카테고리·도구는 생략됩니다)"
        )
        lines += ["===== ⑤ 카테고리 확정 / 도구 호출 =====", note, "", "===== 소요 시간 ====="]
        for key, value in result.timings_ms.items():
            lines.append(f"  {key:<14} {value}ms")
        if result.warnings:
            lines += ["", "===== 경고 ====="]
            lines += [f"  - {w}" for w in result.warnings]
        return "\n".join(lines)

    lines += [f"===== ⑤ 카테고리 확정 (후보 {len(result.candidate.top)}개 중) ====="]
    for name, score in llm.category_choice.scores.items():
        mark = " <-- 선택" if name == llm.category_choice.label else ""
        lines.append(f"  {_pad(name, 10)} {score:.4f}{mark}")
    lines += [
        "",
        "-- 모델 원시 출력 --",
        llm.tool_call.raw or "(없음)",
        "",
    ]

    tr = result.tool_result
    if llm.tool_call.called and tr is not None:
        lines.append("===== ⑥ 민원 DB 실행 결과 =====")
        lines.append(f"실행됨      : {tr.executed}")
        lines.append(f"검증 통과   : {tr.ok}")
        if tr.needs_selection:
            lines.append("선택 필요   : True (대상이 여러 건이거나 특정되지 않아 실행하지 않음)")
        if tr.search:
            lines.append(f"찾은 조건   : {tr.search}")
        if tr.error:
            lines.append(f"오류        : {tr.error}")
        lines.append(f"응답 문구   : {tr.message}")
        if tr.data is not None:
            lines.append(f"결과 데이터 : {tr.data}")
        lines.append("")

    lines.append("===== 소요 시간 =====")
    for key, value in result.timings_ms.items():
        lines.append(f"  {key:<14} {value}ms")

    if result.warnings:
        lines += ["", "===== 경고 ====="]
        lines += [f"  - {w}" for w in result.warnings]

    return "\n".join(lines)


def _chat_status() -> dict[str, object]:
    try:
        from app.services import chat   # chat 이 pipeline 을 import 하므로 여기서 불러옵니다

        return chat.status()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def status() -> dict[str, object]:
    """모델 설치·로드 상태 요약."""
    return {
        "keyphrase": keyphrase.status(),
        "embedder": embedder.status(),
        "case_store": case_store.status(),
        "faq_store": faq_store.status(),
        "chat": _chat_status(),
        "complaint_store": complaint_store.status(),
        "llm": llm_engine.status(),
    }
