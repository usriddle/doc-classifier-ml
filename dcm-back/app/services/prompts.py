"""
⑤ Qwen 에게 주는 프롬프트 모음.

KV 캐시(CAG) 대상인 고정 프리픽스
  - 역할 지시문
  - 도구 목록 (이름·용도·인자 이름. 인자 설명이 담긴 전체 스키마는 도구 호출 단계 프롬프트에만)
  - 카테고리 정의 (7종)
  - 의도 정의 (5종 + 해당없음)
를 build_fixed_prefix() 하나로 모아두었습니다.

안내 지식(FAQ)은 프리픽스에 넣지 않고 '문의' 즉답 프롬프트(build_answer_prompt)에만 넣습니다.
의도·카테고리·도구 판정에는 쓰이지 않으므로, 프리픽스에 두면 모든 요청과 모든 학습 샘플이 길어지기만 합니다.
FAQ 원본은 data/faq.csv 이고, faq_store 가 질문과 비슷한 것만 골라 build_answer_prompt 에 넘깁니다. (RAG)

프리픽스에는 모델의 판단에 필요한 내용만 둡니다. INSERT/UPDATE, 소유권 검사처럼 서버 코드가
처리하는 동작 설명은 넣지 않습니다. 프리픽스는 요청마다 토큰이 한 글자도 달라지면 안 되므로 날짜·사용자 정보 같은
동적인 값은 넣지 마세요.

llm_engine 이 이 프리픽스를 한 번만 prefill 해 past_key_values 로 재사용합니다.
(.env 의 LLM_KV_CACHE) 이 함수의 결과가 바뀌면 캐시도 다시 만들어야 하므로
프리픽스를 고친 뒤에는 서버를 재시작하세요.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from app.config import settings
from app.services import categories


# =============================================================================
# 의도 5종 - 그림 ⑤ 아래의 다섯 갈래
# =============================================================================
@dataclass(frozen=True)
class Intent:
    code: str
    name: str
    tool: str | None      # 호출할 도구 이름 (문의는 없음)
    description: str


INTENTS: tuple[Intent, ...] = (
    Intent(
        code="inquiry", name="문의", tool=None,
        description="처리 기간·담당 부서·신청 방법 등을 묻는 단순 질문. DB 를 거치지 않고 안내 지식으로 바로 답한다.",
    ),
    Intent(
        code="register", name="접수", tool="register_complaint",
        description="새로운 민원을 등록해 달라는 요청. 고장·불편·피해 신고가 여기에 해당한다.",
    ),
    Intent(
        code="search", name="조회", tool="get_complaints",
        description="이미 넣은 민원의 상태·이력·목록을 확인하려는 요청. 읽기 전용이다.",
    ),
    Intent(
        code="update", name="수정", tool="update_complaint",
        description="이미 넣은 민원의 내용이나 위치를 바꿔 달라는 요청.",
    ),
    Intent(
        code="delete", name="삭제", tool="cancel_complaint",
        description="이미 넣은 민원을 취소·철회해 달라는 요청.",
    ),
    Intent(
        code="out_of_scope", name="해당없음", tool=None,
        description=(
            "문의·접수·조회·수정·삭제 어디에도 해당하지 않는 요청. "
            "지자체 민원과 무관한 잡담·일반 상식 질문·광고·욕설, 의미를 알 수 없는 문자열, "
            "또는 무엇을 원하는지 특정할 수 없는 모호한 말이 여기에 해당한다. "
            "1~5 번 중 하나로 보기 애매하면 억지로 끼워 맞추지 말고 이 번호를 고른다."
        ),
    ),
)

# 실제로 실행되는 의도 5종. INTENTS 전체(6종)에서 게이트용 '해당없음'을 뺀 목록입니다.
VALID_INTENTS: tuple[Intent, ...] = tuple(i for i in INTENTS if i.code != "out_of_scope")
OUT_OF_SCOPE: Intent = next(i for i in INTENTS if i.code == "out_of_scope")

INTENT_BY_CODE: dict[str, Intent] = {i.code: i for i in INTENTS}

# 조회·수정·삭제는 기존 민원을 '찾는' 요청이라 주제가 드러나지 않을 수 있습니다.
# 이 의도들은 카테고리 후보에 '없음'을 더해, 특정할 수 없으면 없음으로 판정합니다.
CATEGORY_NONE = "없음"
FIND_TOOLS = ("get_complaints", "update_complaint", "cancel_complaint")


def category_allows_none(intent: "Intent") -> bool:
    """카테고리 판정에 '없음' 선택지를 주는 의도인지. (조회·수정·삭제)"""
    return intent.tool in FIND_TOOLS


def category_needed(intent: "Intent") -> bool:
    """카테고리 판정이 필요한 의도인지. (문의·해당없음은 카테고리를 쓰지 않음)"""
    return intent.tool is not None
INTENT_BY_NAME: dict[str, Intent] = {i.name: i for i in INTENTS}


def intent_by_number(number: int) -> Intent:
    """1-based 번호로 의도를 찾습니다. (범위를 벗어나면 '해당없음' - 안전한 쪽으로 반려)"""
    if 1 <= number <= len(INTENTS):
        return INTENTS[number - 1]
    return OUT_OF_SCOPE


# =============================================================================
# 도구 정의 JSON - 그림의 '도구 정의 JSON'
# =============================================================================
FIND_PARAMS: dict[str, str] = {
    "complaint_id": "민원 번호. 없으면 0",
    "category": "확정 카테고리. 없음이면 빈 문자열",
    "keyword": "찾을 민원의 대상 명사(원문 그대로, 번호·상태 말 빼고). 없으면 빈 문자열",
    "period": "접수 시점(원문 그대로). 없으면 빈 문자열",
}

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "register_complaint",
        "description": "새 민원을 접수한다.",
        "parameters": {
            # 7종 목록은 [카테고리 정의] 에 있고, 값은 코드가 확정 카테고리로 덮어씁니다.
            "category": "확정된 카테고리",
            "content": "민원 내용 요약",
            "location": "민원이 발생한 위치. 없으면 빈 문자열",
        },
    },
    # 조회·수정·삭제의 앞 네 인자(FIND_PARAMS)는 대상 민원을 '찾는' 조건입니다.
    # 번호를 모르는 시민이 "어제 넣은 가로등 민원"처럼 말해도 찾을 수 있게 합니다.
    # category 는 코드가 확정 카테고리(없음이면 "")로 덮어씁니다.
    {
        "name": "get_complaints",
        "description": "본인 민원의 상태·이력·목록을 조회한다.",
        "parameters": {**FIND_PARAMS, "field": "status/history/list 중 하나"},
    },
    {
        "name": "update_complaint",
        "description": "기존 민원의 내용이나 위치를 수정한다.",
        "parameters": {
            **FIND_PARAMS,
            "content": "새 내용. 바꾸지 않으면 빈 문자열",
            "location": "새 위치. 바꾸지 않으면 빈 문자열",
        },
    },
    {
        "name": "cancel_complaint",
        "description": "기존 민원을 취소한다.",
        "parameters": {**FIND_PARAMS, "reason": "취소 사유. 없으면 빈 문자열"},
    },
]

TOOL_BY_NAME: dict[str, dict] = {t["name"]: t for t in TOOL_DEFINITIONS}


# =============================================================================
# 안내 지식 (FAQ) - '문의' 의도의 답변 근거
# 이제 FAQ 는 data/faq.csv 에서 관리하고, faq_store 가 질문과 비슷한 것만 골라 넣습니다.
#
# 아래 _LEGACY_GUIDE_KNOWLEDGE 는 답변에 쓰이지 않습니다. 프롬프트 지문(prompt_fingerprint)을
# FAQ 를 CSV 로 옮기기 전과 같게 유지하려고 남겨 둔 고정 문자열입니다.
#   - '문의' 답변은 학습 대상이 아니므로, FAQ 를 바꿔도 어댑터를 다시 학습할 필요가 없습니다.
#   - 지문이 그대로라 기존 어댑터의 run_info.json 과 학습 노트북의 기준선 캐시가 계속 유효합니다.
# 이 문자열은 고치지 마세요. (고치면 지문이 바뀌어 기존 어댑터에 경고가 납니다)
# =============================================================================
_LEGACY_GUIDE_KNOWLEDGE = """[안내 지식 - 자주 묻는 질문]
Q. 민원 처리 기간은 얼마나 걸리나요?
A. 일반 민원은 접수 후 7일 이내 처리하며, 현장 확인이 필요한 경우 최대 14일이 걸립니다.

Q. 접수한 민원을 취소할 수 있나요?
A. 처리 시작 전까지는 취소할 수 있습니다.

Q. 민원은 어떻게 접수하나요?
A. 민원 내용을 문장으로 적어 주시면 됩니다. 위치(도로명, 건물명 등)를 함께 적으면 더 빠르게 처리됩니다.

Q. 접수한 민원의 처리 상태는 어떻게 확인하나요?
A. 민원 번호를 알려 주시면 상태를 확인할 수 있습니다. 번호를 모르시면 내가 접수한 민원 목록을 조회해 보세요.

Q. 취소한 민원을 다시 살릴 수 있나요?
A. 취소한 민원은 되돌릴 수 없습니다. 같은 내용으로 새로 접수해 주세요.

Q. 어떤 종류의 민원을 접수할 수 있나요?
A. 행정안전, 국토교통(도로·교통), 주택건축, 환경·위생, 보건복지, 소방 분야 민원을 접수할 수 있습니다. 어디에 해당하는지 모르시면 내용만 적어 주세요. 자동으로 분류합니다."""

ROLE_INSTRUCTION = """당신은 지방자치단체 민원 접수 시스템의 판정 모델이다.
사용자의 민원 문장을 읽고 (1) 의도 1개, (2) 카테고리 1개, (3) 실행할 도구 호출 JSON 을 정한다.
한 요청에는 의도가 반드시 하나만 있다. 추측한 정보를 지어내지 말고, 주어진 문장에 있는 내용만 사용한다."""


def intent_block() -> str:
    lines = ["[의도 정의 - 5종 + 해당없음]"]
    for i, intent in enumerate(INTENTS, start=1):
        lines.append(f"{i}. {intent.name} : {intent.description}")
    lines.append("조회·수정·삭제는 민원 번호 대신 주제·시점으로 가리켜도 된다.")
    lines.append("전입신고·여권·대관 같은 행정 업무의 처리 지연이나 절차 불편을 알리는 것은 접수, 이 창구에 이미 넣은 민원의 진행을 묻는 것은 조회다.")
    return "\n".join(lines)


def tool_block() -> str:
    """
    도구 목록 - 이름·용도·인자 이름만 한 줄씩.

    인자별 설명이 담긴 전체 스키마(TOOL_DEFINITIONS)는 도구 호출 JSON 을 만들 때
    build_tool_prompt 의 [사용할 도구] 에 그 도구 하나만 넣습니다. 프리픽스에까지 전부 넣으면
    같은 내용이 두 번 들어가 모든 요청·모든 학습 샘플이 길어집니다. (의도·카테고리 판정에는 쓰이지 않음)
    """
    lines = ["[도구 목록]"]
    for tool in TOOL_DEFINITIONS:
        lines.append(f"- {tool['name']} : {tool['description']} (인자: {', '.join(tool['parameters'])})")
    lines.append("조회·수정·삭제의 complaint_id·category·keyword·period 는 대상 민원을 찾는 조건이다.")
    return "\n".join(lines)


def build_fixed_prefix() -> str:
    """
    매 요청마다 동일한 고정 프리픽스. (그림의 KV 캐시 대상 구간)

    CAG 를 붙일 때 이 문자열만 한 번 prefill 하면 됩니다.
    """
    return "\n\n".join(
        [
            ROLE_INSTRUCTION,
            tool_block(),
            categories.prompt_block(),
            intent_block(),
        ]
    )


# =============================================================================
# 단계별 사용자 프롬프트
# =============================================================================
def build_intent_prompt(user_text: str) -> str:
    """
    의도 판정 - 모델은 번호 1~6 중 하나만 내면 됩니다.

    1~5 는 실행 가능한 의도, 6(해당없음)은 게이트 역할입니다.
    민원 주제(카테고리)와 무관하게, 문의·접수·조회·수정·삭제 중 무엇을 원하는지만
    특정되면 1~5 중 하나를 고르면 됩니다. 예를 들어 "제가 어제 문의한 내용 보여줘"는
    도로·환경 같은 카테고리와 무관하게 조회(3번)입니다.
    """
    return (
        f"[민원 문장]\n{user_text}\n\n"
        "위 문장이 문의·접수·조회·수정·삭제 중 무엇을 원하는지 판단하시오.\n"
        "그중 하나가 명확하면 그 번호를, 다섯 중 어디에도 해당하지 않으면 6(해당없음)을 고르시오.\n"
        "설명 없이 번호 하나만 출력하시오."
    )


def category_labels(candidate_names: list[str], allow_none: bool = False) -> list[str]:
    """카테고리 판정의 선택지. allow_none 이면 마지막에 '없음' 을 붙입니다."""
    labels = [categories.get(n).name for n in candidate_names]
    return labels + [CATEGORY_NONE] if allow_none else labels


def build_category_prompt(user_text: str, candidate_names: list[str], allow_none: bool = False) -> str:
    """
    카테고리 확정 - ④ 가 추린 후보(CANDIDATE_TOP_K 개, 기본 4) 중에서만 고릅니다.

    후보는 번호와 이름만 적습니다. 각 카테고리의 설명은 이미 고정 프리픽스의
    [카테고리 정의] 에 있으므로 여기서 반복하지 않습니다. (캐시 밖 토큰을 줄임)

    allow_none (조회·수정·삭제) 이면 마지막 번호로 '없음' 을 더합니다.
    "어제 넣은 거 취소해 줘"처럼 민원 주제가 문장에 드러나지 않으면 없음이 정답입니다.
    """
    labels = category_labels(candidate_names, allow_none)
    lines = [f"[민원 문장]\n{user_text}", "", "[카테고리 후보]"]
    for i, name in enumerate(labels, start=1):
        lines.append(f"{i}. {name}")
    lines.append("")
    if allow_none:
        lines += [
            "위 문장이 가리키는 민원의 카테고리를 후보 중에서 하나만 고르시오.",
            f"문장만으로 민원 주제를 특정할 수 없으면 {len(labels)}({CATEGORY_NONE})을 고르시오.",
        ]
    else:
        lines.append("위 민원 문장에 가장 알맞은 카테고리를 후보 중에서 하나만 고르시오.")
    lines.append("설명 없이 번호 하나만 출력하시오.")
    return "\n".join(lines)


def build_tool_prompt(user_text: str, intent: Intent, category_name: str) -> str:
    """도구 호출 JSON 생성. (스키마는 프리픽스의 도구 정의와 같은 한 줄 형식)"""
    tool = TOOL_BY_NAME.get(intent.tool or "", None)
    schema = json.dumps(tool, ensure_ascii=False) if tool else "{}"
    return (
        f"[민원 문장]\n{user_text}\n\n"
        f"판정된 의도: {intent.name}\n"
        f"확정된 카테고리: {category_name}\n\n"
        f"[사용할 도구]\n{schema}\n\n"
        '위 도구를 호출하는 JSON 을 출력하시오. 형식은 '
        '{"name": "도구이름", "arguments": {...}} 이다.\n'
        "설명이나 코드블록 없이 JSON 객체 하나만 출력하시오."
    )


# Qwen 이 이 문구로 답하면 llm_engine 이 FAQ_FALLBACK_MESSAGE 로 바꿉니다.
ANSWER_DECLINE_PHRASE = "확인이 어렵다"


def build_answer_prompt(user_text: str, faqs: list[tuple[str, str]]) -> str:
    """
    '문의' 의도 - 도구를 부르지 않고 안내 지식만으로 답합니다.

    faqs : faq_store.lookup() 이 고른 (질문, 답변) 목록. 질문과 비슷한 것만 들어옵니다.
           (FAQ_ENABLED=false 면 CSV 의 FAQ 전부)
    """
    blocks = [f"Q. {q}\nA. {a}" for q, a in faqs]
    knowledge = "[안내 지식 - 관련 FAQ]\n" + "\n\n".join(blocks)
    return (
        f"{knowledge}\n\n"
        f"[민원 문장]\n{user_text}\n\n"
        "위 질문에 [안내 지식]만 사용해 2문장 이내로 답하시오. "
        f"안내 지식으로 답할 수 없는 질문이면 '{ANSWER_DECLINE_PHRASE}'고만 답하시오."
    )


def _legacy_answer_prompt_template() -> str:
    """FAQ 를 CSV 로 옮기기 전의 '문의' 프롬프트 틀. 프롬프트 지문 계산에만 씁니다. (위 _LEGACY_GUIDE_KNOWLEDGE 설명 참고)"""
    return (
        f"{_LEGACY_GUIDE_KNOWLEDGE}\n\n"
        "[민원 문장]\n{TEXT}\n\n"
        "위 질문에 [안내 지식]만 사용해 2문장 이내로 답하시오. "
        "안내 지식에 없는 내용은 '확인이 어렵다'고 답하시오."
    )


# =============================================================================
# 학습용 - 도구 호출 JSON 정답 직렬화 / 프롬프트 지문
# =============================================================================
def tool_param_names(tool_name: str) -> list[str]:
    """도구 인자 이름 목록 (TOOL_DEFINITIONS 에 적힌 순서)."""
    tool = TOOL_BY_NAME.get(tool_name)
    return list(tool["parameters"].keys()) if tool else []


def format_tool_call(tool_name: str, arguments: dict) -> str:
    """
    도구 호출 JSON 을 학습 정답용 표준 형태로 직렬화합니다.

    - 한 줄짜리 {"name": ..., "arguments": {...}}
    - 인자 키 순서는 TOOL_DEFINITIONS 순서로 고정 (없는 키는 채우지 않음)
    - 한글은 그대로(ensure_ascii=False), 코드블록·설명 없음

    학습 데이터의 정답이 전부 이 형태여야 모델이 한 가지 형식만 배웁니다.
    추론 쪽 파서(llm_engine._parse_tool_json)는 이 형태를 그대로 읽습니다.
    """
    order = tool_param_names(tool_name)
    ordered = {k: arguments[k] for k in order if k in arguments}
    # 스키마에 없는 키는 뒤에 붙이지 않고 버립니다. (학습 정답에 잡음이 섞이지 않게)
    return json.dumps({"name": tool_name, "arguments": ordered}, ensure_ascii=False)


def prompt_fingerprint() -> str:
    """
    ⑤ 프롬프트 전체의 지문(sha256 앞 16자리).

    고정 프리픽스와 단계별 프롬프트 틀이 한 글자라도 바뀌면 값이 달라집니다.
    어댑터를 학습할 때 이 값을 run_info.json 에 남겨 두고, 서버가 어댑터를 올릴 때
    지금 값과 비교해 "학습 때와 프롬프트가 다르다"는 경고를 냅니다.

    '문의' 답변 프롬프트는 학습 대상이 아니므로 고정된 예전 틀을 넣습니다.
    FAQ(data/faq.csv)나 build_answer_prompt 를 고쳐도 지문은 바뀌지 않습니다.
    """
    # 카테고리 후보 수(CANDIDATE_TOP_K)가 바뀌면 선택지 번호가 달라지므로 지문도 바뀌어야 합니다.
    k = max(1, min(settings.CANDIDATE_TOP_K, len(categories.NAMES)))
    parts = [
        build_fixed_prefix(),
        build_intent_prompt("{TEXT}"),
        build_category_prompt("{TEXT}", list(categories.NAMES[:k])),
        build_category_prompt("{TEXT}", list(categories.NAMES[:k]), allow_none=True),
    ]
    for intent in VALID_INTENTS:
        if intent.tool:
            parts.append(build_tool_prompt("{TEXT}", intent, "{CATEGORY}"))
    parts.append(_legacy_answer_prompt_template())
    digest = hashlib.sha256("\n\u241e\n".join(parts).encode("utf-8")).hexdigest()
    return digest[:16]
