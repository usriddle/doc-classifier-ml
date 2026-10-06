"""
정답 레코드 - 학습·평가 데이터의 원본.

한 줄(JSONL)에 민원 문장 하나와 그 정답을 적습니다.

  {"id": "r0001", "group": "g0001", "text": "아파트 앞 도로에 포트홀이 생겼어요",
   "intent": "접수", "category": "교통·국토",
   "arguments": {"content": "아파트 앞 도로 포트홀 발생", "location": "아파트 앞 도로"},
   "tags": ["구어체"]}

필드
  id         레코드 고유 id (필수, 중복 불가)
  group      같은 원문을 말만 바꿔 늘린 변형들은 같은 group 으로 묶습니다. (필수 아님, 없으면 id)
             같은 group 은 반드시 같은 쪽(학습/검증/평가)에 들어가 평가 점수가 부풀지 않습니다.
  text       민원 문장 원문 (필수)
  intent     문의 / 접수 / 조회 / 수정 / 삭제 / 해당없음 (필수)
  category   접수      : 7종 중 하나 (필수)
             조회·수정·삭제 : 7종 중 하나 또는 "없음" (필수) - 문장이 가리키는 민원의 주제
             문의·해당없음 : 적지 않음 (카테고리를 판정하지 않음)
  arguments  도구 인자 정답. 접수·조회·수정·삭제는 필수. 문의·해당없음은 적지 않음
             (category 인자는 위 category 로 자동 채움 - "없음" 이면 "")
               접수 : content, location
               조회 : complaint_id, keyword, period, field(status/history/list)
               수정 : complaint_id, keyword, period, content, location   (안 바꾸는 항목은 "")
               삭제 : complaint_id, keyword, period, reason              (사유 없으면 "")
  tags       자유 태그. "경계"(헷갈리기 쉬운 문장)는 의도 학습 샘플을 고를 때 우선합니다.
  split      train / val / test 로 강제 지정할 때만 적습니다. 없으면 group 해시로 자동 배정.

라벨링 규칙 (정답이 흔들리면 모델도 흔들립니다)
  - location 은 원문에 있는 표현을 그대로 옮깁니다. 원문에 위치가 없거나 "우리 동네", "여기",
    "근처"처럼 장소를 좁혀 주지 않는 말만 있으면 반드시 "".
    장소가 두 번 나오면 민원이 발생한 곳, 수정의 "A에서 B로"는 새 위치 B 가 정답입니다.
    민원 대상인 시설물·물건(가로수, 담장, 전봇대, CCTV, 소화전, 벤치, 나무 …)은 location 에 넣지 않습니다.
    ("들꽃길 주택 담장이 기울어서" -> "들꽃길 주택") 사람이 찾아가는 장소(가게, 계단, 공사장, 정류장)는 넣습니다.
  - 의도: 전입신고·여권·대관 같은 행정 업무의 처리 지연·절차 불편을 알리는 말은 접수,
    이 창구에 이미 넣은 민원의 진행을 묻는 말("제가 넣은 OO 민원 처리됐나요")은 조회입니다.
  - 카테고리 경계 (요약은 app/services/categories.py 의 설명 괄호 안에도 적혀 있습니다)
      공사장 소음 -> 주택·건축, 공사장 근로자 안전 -> 노동·기업, 공사장 옆 보행 안전 -> 교통·국토
      도로 먼지·거리 청소, 도로 위 쓰레기·동물 사체 -> 환경·위생
      신호등·차선·표지판·주정차·버스, 포장·포트홀·가로등·보도블록 -> 교통·국토
      (단지 안 주차장·도로·시설은 주택·건축, 불·연기·인명 구조·소방차 진입은 문화·행정·안전)
      근로자 임금·해고·산재, 사업주 자금·판로·지역화폐·전통시장 지원 -> 노동·기업
      경비원 임금·폭언 -> 노동·기업, 경비실 휴게 공간 시설 -> 주택·건축
      노인 일자리·생계 지원 -> 보건·복지, 그 밖의 일자리·취업 지원 -> 노동·기업
      화재·구조·재난·민원서류·도서관·체육시설·축제 -> 문화·행정·안전
      시청 대표 홈페이지 장애, 공무원·직원 칭찬, 소비자 제품 분쟁 -> 기타
      (분야 전용 시스템 장애는 그 분야. 예: 지역화폐 앱 -> 노동·기업, 도서관 예약 -> 문화·행정·안전)
  - content 는 "무엇이 어떤 상태인지"를 명사형으로 짧게 씁니다. (예: "가로등 꺼짐")
    위치·기간·요청 문구는 넣지 않습니다.
  - complaint_id 는 원문에 번호가 있을 때만 그 숫자, 없으면 0 입니다.
  - 조회·수정·삭제의 찾기 조건 (시민은 번호를 모르는 경우가 대부분)
      keyword : 찾을 민원의 **대상 명사만** 원문 그대로. (예: "가로등", "층간소음", "주차", "맨홀 뚜껑")
                번호·동·호·층·노선(3402번, 101동, 2층), 장소 수식(사당역 3번 출구, 공원), 상태·요청 말
                (고장, 파손, 불법, 지연, 방역)은 뺍니다. "101동 엘리베이터 고장" -> "엘리베이터".
                한 덩어리 명사(맨홀 뚜껑, 방범 CCTV, 음식물 쓰레기, 버스 정류장)는 그대로 둡니다.
                "민원", "신고 건", "그거", "안전 문제"처럼 대상을 알 수 없는 말뿐이면 "".
                수정의 새 내용·새 위치는 keyword 가 아니라 content·location 입니다.
      period  : 접수 시점을 원문 표현 그대로. (예: "어제", "지난주", "방금") 없으면 "".
      category: 대상이 7종 중 하나로 분명하면 그 이름, 주제가 드러나지 않거나 애매하면 "없음".
                (예: "가로등 민원" -> 교통·국토, "버스 민원" -> 교통·국토, "소음 민원" -> 없음(층간·공사·생활 소음이 갈림),
                 "어제 넣은 거" -> 없음)
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from app.services import categories, prompts

SPLITS = ("train", "val", "test")
NONE = prompts.CATEGORY_NONE
FIND_KEYS = ("keyword", "period")   # 원문에 그대로 있어야 하는 찾기 조건
TOOL_INTENTS = {i.name: i for i in prompts.VALID_INTENTS if i.tool}


@dataclass
class GoldRecord:
    id: str
    text: str
    intent: str
    group: str = ""
    category: str = ""
    arguments: dict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    split: str = ""

    @property
    def intent_obj(self) -> prompts.Intent:
        return prompts.INTENT_BY_NAME[self.intent]

    @property
    def tool(self) -> str | None:
        return self.intent_obj.tool

    @property
    def category_allows_none(self) -> bool:
        return prompts.category_allows_none(self.intent_obj)

    @property
    def category_label(self) -> str:
        """⑤ 카테고리 판정의 정답 라벨. (7종 이름 또는 '없음', 판정하지 않는 의도면 "")"""
        return self.category if prompts.category_needed(self.intent_obj) else ""

    def tool_arguments(self) -> dict:
        """정답 도구 인자 (category 를 채워 넣은 완성본. '없음' 이면 "")."""
        args = dict(self.arguments or {})
        if self.tool and "category" in prompts.tool_param_names(self.tool):
            args["category"] = "" if self.category == NONE else self.category
        return args

    def auto_tags(self) -> list[str]:
        """평가표를 쪼개 볼 때 쓰는 자동 태그."""
        tags = [f"의도:{self.intent}"]
        args = self.arguments or {}
        if self.tool in ("register_complaint", "update_complaint"):
            tags.append("위치있음" if str(args.get("location", "")).strip() else "위치없음")
        if self.tool in prompts.FIND_TOOLS:
            tags.append("id있음" if _as_int(args.get("complaint_id")) > 0 else "id없음")
            tags.append("카테고리없음" if self.category == NONE else "카테고리있음")
            tags.append("키워드있음" if str(args.get("keyword", "")).strip() else "키워드없음")
            tags.append("시점있음" if str(args.get("period", "")).strip() else "시점없음")
        return tags + [t for t in self.tags if t not in tags]


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


# =============================================================================
# 읽기 + 검증
# =============================================================================
def load_records(path: str | Path) -> list[GoldRecord]:
    """JSONL 을 읽습니다. 형식 오류는 ValueError 로 줄 번호와 함께 알려 줍니다."""
    records: list[GoldRecord] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno} JSON 형식 오류 - {exc}") from exc
        records.append(
            GoldRecord(
                id=str(data.get("id", f"line{lineno}")),
                text=str(data.get("text", "")),
                intent=str(data.get("intent", "")),
                group=str(data.get("group") or data.get("id", f"line{lineno}")),
                category=str(data.get("category") or ""),
                arguments=data.get("arguments") or {},
                tags=list(data.get("tags") or []),
                split=str(data.get("split") or ""),
            )
        )
    return records


# location 끝에 붙으면 안 되는 시설물·물건 (lint 경고용)
OBJECT_WORDS = {
    "가로수", "나무", "담장", "전봇대", "전신주", "CCTV", "소화전", "소화기", "벤치", "신호등", "가로등",
    "보안등", "맨홀", "하수구", "배수구", "비상벨", "엘리베이터", "에스컬레이터", "스프링클러", "현수막",
    "쓰레기통", "표지판", "과속방지턱", "볼라드", "방화문", "펜스", "난간", "보도블록", "간판", "변압기",
}


def lint(records: list[GoldRecord]) -> list[str]:
    """오류는 아니지만 라벨링 규칙과 어긋나 보이는 것. (학습은 그대로 진행됩니다)"""
    out = []
    for r in records:
        loc = str((r.arguments or {}).get("location", "") or "").split()
        if len(loc) >= 2 and loc[-1] in OBJECT_WORDS:
            out.append(f"{r.id}: location 끝에 물건 이름 '{loc[-1]}' - 장소까지만 적었는지 확인 ({' '.join(loc)})")
    return out


def validate(records: list[GoldRecord]) -> list[str]:
    """정답 레코드의 문제를 사람이 읽을 문장 목록으로 돌려줍니다. 빈 목록이면 통과."""
    errors: list[str] = []
    seen: Counter = Counter(r.id for r in records)
    for rid, n in seen.items():
        if n > 1:
            errors.append(f"[{rid}] id 가 {n}번 중복됩니다.")

    for r in records:
        tag = f"[{r.id}]"
        if not r.text.strip():
            errors.append(f"{tag} text 가 비어 있습니다.")
        if r.intent not in prompts.INTENT_BY_NAME:
            errors.append(f"{tag} intent '{r.intent}' 는 {list(prompts.INTENT_BY_NAME)} 중 하나여야 합니다.")
            continue
        if r.split and r.split not in SPLITS:
            errors.append(f"{tag} split '{r.split}' 는 {SPLITS} 중 하나여야 합니다.")
        allowed = list(categories.NAMES) + ([NONE] if r.category_allows_none else [])
        if r.category and r.category not in allowed:
            errors.append(f"{tag} category '{r.category}' 는 {allowed} 중 하나여야 합니다.")

        tool = r.tool
        if tool is None:
            if r.arguments:
                errors.append(f"{tag} '{r.intent}' 는 도구를 부르지 않으므로 arguments 를 비워 두세요.")
            if r.category:
                errors.append(f"{tag} '{r.intent}' 는 카테고리를 판정하지 않으므로 category 를 비워 두세요.")
            continue

        if tool == "register_complaint" and not r.category:
            errors.append(f"{tag} 접수는 category 가 필요합니다. (도구 인자 category 의 정답)")
        if tool in prompts.FIND_TOOLS and not r.category:
            errors.append(f"{tag} {r.intent} 는 category 가 필요합니다. 7종 중 하나 또는 \"{NONE}\".")

        args = r.arguments or {}
        expected = [k for k in prompts.tool_param_names(tool) if k != "category"]
        missing = [k for k in expected if k not in args]
        extra = [k for k in args if k not in expected]
        if missing:
            errors.append(f"{tag} {tool} 인자 누락: {missing} (값이 없으면 \"\" 또는 0 으로 적으세요)")
        if extra:
            errors.append(f"{tag} {tool} 에 없는 인자: {extra}")

        if "complaint_id" in args and _as_int(args["complaint_id"]) < 0:
            errors.append(f"{tag} complaint_id 는 0 이상의 정수여야 합니다.")
        if tool == "get_complaints" and args.get("field") not in ("status", "history", "list"):
            errors.append(f"{tag} field 는 status/history/list 중 하나여야 합니다.")
        for key in ("content", "location", "reason"):
            if key in args and not isinstance(args[key], str):
                errors.append(f"{tag} {key} 는 문자열이어야 합니다.")

        # location 을 지어낸 정답은 모델에게 '지어내기'를 가르칩니다.
        loc = str(args.get("location", "")).strip()
        if loc and _squash(loc) not in _squash(r.text):
            errors.append(
                f"{tag} location '{loc}' 이 원문에 그대로 없습니다. 원문 표현을 옮기거나 \"\" 로 두세요."
            )
        for key in FIND_KEYS:
            val = str(args.get(key, "")).strip()
            if key in args and not isinstance(args[key], str):
                errors.append(f"{tag} {key} 는 문자열이어야 합니다.")
            elif val and _squash(val) not in _squash(r.text):
                errors.append(f"{tag} {key} '{val}' 이 원문에 그대로 없습니다. 원문 표현을 옮기거나 \"\" 로 두세요.")
        if tool in ("update_complaint",) and not (str(args.get("content", "")).strip() or loc):
            errors.append(f"{tag} 수정인데 content 와 location 이 둘 다 비어 있습니다.")
    return errors


def _squash(text: str) -> str:
    """공백을 없앤 비교용 문자열."""
    return "".join(str(text).split())


# =============================================================================
# 학습 / 검증 / 평가 분할
# =============================================================================
def assign_split(record: GoldRecord, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)) -> str:
    """
    group 해시로 분할을 정합니다.

    데이터를 나중에 더 넣어도 기존 레코드의 배정이 바뀌지 않으므로 평가셋이 고정됩니다.
    (무작위 셔플로 나누면 데이터를 추가할 때마다 평가셋이 바뀌어 학습 전후 비교가 무의미해집니다)
    """
    if record.split in SPLITS:
        return record.split
    bucket = int(hashlib.sha1(record.group.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


def split_records(
    records: list[GoldRecord], ratios: tuple[float, float, float] = (0.7, 0.15, 0.15)
) -> dict[str, list[GoldRecord]]:
    out: dict[str, list[GoldRecord]] = {s: [] for s in SPLITS}
    for r in records:
        out[assign_split(r, ratios)].append(r)
    return out


def split_warnings(parts: dict[str, list[GoldRecord]]) -> list[str]:
    """분할이 치우쳤는지 확인합니다. (평가셋에 어떤 의도가 아예 없으면 그 의도는 채점이 안 됨)"""
    warns: list[str] = []
    group_sides: dict[str, set[str]] = {}
    for side, recs in parts.items():
        for r in recs:
            group_sides.setdefault(r.group, set()).add(side)
    leaked = [g for g, sides in group_sides.items() if len(sides) > 1]
    if leaked:
        warns.append(f"같은 group 이 여러 분할에 걸쳐 있습니다(split 강제 지정 때문): {leaked[:5]}")

    for side in ("val", "test"):
        present = {r.intent for r in parts[side]}
        absent = [i.name for i in prompts.INTENTS if i.name not in present]
        if absent:
            warns.append(f"{side} 에 없는 의도: {absent} - 이 의도는 {side} 에서 채점되지 않습니다.")
    if len(parts["test"]) < 30:
        warns.append(
            f"평가셋이 {len(parts['test'])}건뿐입니다. 한 건이 점수를 수 % 씩 움직이므로 "
            "작은 차이는 개선으로 보면 안 됩니다."
        )
    return warns


def summarize(records: list[GoldRecord]) -> dict[str, dict[str, int]]:
    """의도·도구·태그별 건수."""
    by_intent = Counter(r.intent for r in records)
    by_tag: Counter = Counter()
    for r in records:
        by_tag.update(r.auto_tags())
    by_category = Counter(r.category or "(없음)" for r in records)
    return {"의도": dict(by_intent), "카테고리": dict(by_category), "태그": dict(by_tag)}


def cases_overlap(records: list[GoldRecord], cases_csv: str | Path) -> list[str]:
    """
    평가 레코드 문장이 벡터DB 사례(cases.csv)에 그대로 들어 있으면 id 목록을 돌려줍니다.

    그런 문장은 벡터DB 가 정답 카테고리를 '외워서' 올려 주므로 카테고리 점수가 부풀려집니다.
    """
    path = Path(cases_csv)
    if not path.is_file():
        return []
    import csv

    with path.open(encoding="utf-8-sig", newline="") as f:
        case_texts = {_squash(row.get("text", "")) for row in csv.DictReader(f)}
    return [r.id for r in records if _squash(r.text) in case_texts]
