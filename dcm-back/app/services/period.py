"""
시점 표현 -> 날짜 범위. (조회·수정·삭제의 period 인자)

⑤ Qwen 은 "어제", "지난주" 같은 **원문 표현을 그대로 옮기기만** 하고, 날짜 계산은 여기서 합니다.
(모델이 날짜를 계산하면 틀리기 쉽고, 오늘 날짜를 프롬프트에 넣으면 고정 프리픽스·KV 캐시가 깨집니다)

parse("어제", now) -> Period(start=어제 00:00, end=오늘 00:00)
기준 시각은 .env 의 TIMEZONE (기본 Asia/Seoul) 입니다. DB 의 created_at(UTC)과 비교할 때 맞춰 바꿉니다.

알아듣는 표현
  오늘 / 어제 / 그제·그저께 / 엊그제
  N시간 전 · 한두 시간 전 (오늘, 자정 근처면 어제까지)
  N일 전 · 사흘 전 · 나흘 전 · 며칠 전(최근 7일) · 일주일 전 · 보름 전 · N주 전 · 한 달 전
  이번 주·금주 / 지난주·저번 주 / 지지난주 / 주말·지난 주말
  요일: 월요일·화욜 (가장 가까운 지난 그 요일, 오늘 포함) / 지난주 월요일 / 이번 주 수요일
  이번 달·이달 / 지난달·저번 달 / 올해 / 작년 / 작년 12월 / 올해 3월
  M월 D일 / M월 / D일 (이번 달)
  초·중순·말: 9월 초(1~10일) / 지난달 중순(11~20일) / 이번 달 말(21일~) / 올해 초(1~3월) / 작년 말(10~12월)
  최근·요즘·근래 (최근 30일)
  방금·아까·마지막·가장 최근·제일 최근·최근에 넣은 (가장 최근 1건)
  지난번·저번에·전에·예전에 (범위를 좁히지 않음 - '막연한 과거')
알아듣지 못한 표현은 None 을 돌려주고, 호출하는 쪽은 기간 조건 없이 찾습니다. (경고를 남김)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from app.config import settings

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def local_tz():
    name = getattr(settings, "TIMEZONE", "") or "Asia/Seoul"
    if ZoneInfo is not None:
        try:
            return ZoneInfo(name)
        except Exception:
            pass
    return timezone(timedelta(hours=9))


def now_local() -> datetime:
    return datetime.now(local_tz())


@dataclass
class Period:
    text: str                       # 원문 표현
    start: datetime | None = None   # 포함 (지역 시각)
    end: datetime | None = None     # 미포함
    latest: bool = False            # 가장 최근 1건만
    vague: bool = False             # '지난번'처럼 범위를 좁히지 않는 표현

    def describe(self) -> str:
        if self.latest:
            return f"'{self.text}' → 가장 최근 1건"
        if self.vague or not self.start:
            return f"'{self.text}' → 기간 제한 없음"
        last = (self.end - timedelta(seconds=1)) if self.end else None
        if last and last.date() != self.start.date():
            return f"'{self.text}' → {self.start:%Y-%m-%d} ~ {last:%Y-%m-%d}"
        return f"'{self.text}' → {self.start:%Y-%m-%d}"

    def contains(self, moment: datetime) -> bool:
        if self.vague or self.latest or not self.start:
            return True
        return self.start <= moment < (self.end or moment + timedelta(seconds=1))

    def widened(self, days: int = 1) -> "Period":
        """앞뒤로 며칠 넓힌 범위. (자정 근처 접수, 기억 착오 대비)"""
        if self.vague or self.latest or not self.start:
            return self
        return Period(self.text, self.start - timedelta(days=days), (self.end or self.start) + timedelta(days=days))


_NUM_WORDS = {
    "한": 1, "하루": 1, "두": 2, "이틀": 2, "세": 3, "사흘": 3, "네": 4, "나흘": 4, "닷새": 5,
    "다섯": 5, "엿새": 6, "여섯": 6, "이레": 7, "일곱": 7, "여드레": 8, "여덟": 8, "아흐레": 9,
    "아홉": 9, "열흘": 10, "열": 10,
}


def _day(d: date, tz) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=tz)


def _range(d1: date, d2_exclusive: date, text: str, tz) -> Period:
    return Period(text, _day(d1, tz), _day(d2_exclusive, tz))


def _month_start(d: date) -> date:
    return d.replace(day=1)


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


_WEEKDAYS = "월화수목금토일"


def _part_of_month(start: date, end_exclusive: date, part: str | None) -> tuple[date, date]:
    """한 달 범위를 초(1~10일)·중순(11~20일)·말(21일~)로 좁힙니다."""
    if part == "초":
        return start, min(end_exclusive, start.replace(day=11))
    if part == "중순":
        return start.replace(day=11), min(end_exclusive, start.replace(day=21))
    if part == "말":
        return start.replace(day=21), end_exclusive
    return start, end_exclusive


def _part_of_year(year: int, part: str | None) -> tuple[date, date]:
    """한 해를 초(1~3월)·중순(4~9월)·말(10~12월)로 좁힙니다."""
    if part == "초":
        return date(year, 1, 1), date(year, 4, 1)
    if part == "중순":
        return date(year, 4, 1), date(year, 10, 1)
    if part == "말":
        return date(year, 10, 1), date(year + 1, 1, 1)
    return date(year, 1, 1), date(year + 1, 1, 1)


def _part(t: str) -> str | None:
    t = t.replace("주말", "")          # '주말'의 '말'은 월말이 아님
    m = re.search(r"(초|중순|중반|말|하순|상순|초반|말경|말쯤)", t)
    if not m:
        return None
    w = m.group(1)
    return {"상순": "초", "초반": "초", "중반": "중순", "하순": "말", "말경": "말", "말쯤": "말"}.get(w, w)


def _clip(d1: date, d2: date, today: date) -> tuple[date, date]:
    """미래 쪽은 오늘까지로 자릅니다."""
    return d1, min(d2, today + timedelta(days=1))


def _number(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return _NUM_WORDS.get(token)


def parse(text: str, now: datetime | None = None) -> Period | None:
    """시점 표현 하나를 해석합니다. 빈 문자열이면 None, 모르는 표현도 None."""
    raw = (text or "").strip()
    if not raw:
        return None
    now = now or now_local()
    tz = now.tzinfo or local_tz()
    today = now.date()
    t = re.sub(r"\s+", "", raw)

    # --- 가장 최근 1건 ---
    if re.search(r"방금|아까|좀전|조금전|금방|마지막|가장최근|제일최근|최근에(넣|접수|신고|등록|올)", t):
        return Period(raw, latest=True)

    # --- 막연한 과거 (범위를 좁히지 않음) ---
    if re.fullmatch(r"(지난번|저번|저번에|전에|예전에|이전에|옛날에|전번|지난번에)(에)?", t):
        return Period(raw, vague=True)

    # 연도 지정 (작년 12월, 올해 3월)
    year = None
    if "재작년" in t:
        year = today.year - 2
    elif re.search(r"작년|지난해|전년", t):
        year = today.year - 1
    elif re.search(r"올해|금년", t):
        year = today.year
    part = _part(t)

    # --- 날짜 (M월 D일 / M월 / D일) ---
    m = re.search(r"(\d{1,2})월(\d{1,2})일", t)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        try:
            d = date(year or today.year, month, day)
            if year is None and d > today:
                d = date(today.year - 1, month, day)
            return _range(d, d + timedelta(days=1), raw, tz)
        except ValueError:
            return None
    m = re.search(r"(\d{1,2})월", t)
    if m and "개월" not in t:
        month = int(m.group(1))
        if 1 <= month <= 12:
            if year is None:
                year = today.year if month <= today.month else today.year - 1
            start = date(year, month, 1)
            d1, d2 = _part_of_month(start, _add_months(start, 1), part)
            return _range(*_clip(d1, d2, today), raw, tz)
        return None
    m = re.fullmatch(r"(\d{1,2})일(에|날|쯤|경)?", t)
    if m:
        try:
            d = today.replace(day=int(m.group(1)))
        except ValueError:
            return None
        if d > today:
            d = _add_months(today, -1).replace(day=d.day)
        return _range(d, d + timedelta(days=1), raw, tz)

    # --- N시간 / N분 전 (오늘. 자정을 넘기면 어제부터) ---
    m = re.search(r"(\d+|한두|두세|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)(시간|분)(전|쯤전|정도전)", t)
    if m:
        n = {"한두": 2, "두세": 3}.get(m.group(1)) or _number(m.group(1)) or 1
        back = now - (timedelta(hours=n + 1) if m.group(2) == "시간" else timedelta(minutes=n + 30))
        return _range(min(back.date(), today), today + timedelta(days=1), raw, tz)

    # --- 보름 ---
    if re.search(r"보름", t):
        d = today - timedelta(days=15)
        return _range(d - timedelta(days=3), d + timedelta(days=4), raw, tz)

    # --- 요일 (지난주 월요일 / 이번 주 수요일 / 월요일) ---
    m = re.search(r"([월화수목금토일])(요일|욜)", t)
    if m:
        wd = _WEEKDAYS.index(m.group(1))
        monday = today - timedelta(days=today.weekday())
        if re.search(r"지지난주", t):
            d = monday - timedelta(days=14) + timedelta(days=wd)
        elif re.search(r"지난주|저번주|전주", t):
            d = monday - timedelta(days=7) + timedelta(days=wd)
        elif re.search(r"이번주|금주", t):
            d = monday + timedelta(days=wd)
            if d > today:
                return None
        else:
            d = today - timedelta(days=(today.weekday() - wd) % 7)   # 가장 가까운 지난 그 요일 (오늘 포함)
        return _range(d, d + timedelta(days=1), raw, tz)

    # --- N일 / N주 / N달 전 ---
    m = re.search(r"(\d+|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)(일|주|주일|달|개월)전", t)
    if m:
        n = _number(m.group(1)) or 0
        unit = m.group(2)
        if unit == "일":
            d = today - timedelta(days=n)
            return _range(d, d + timedelta(days=1), raw, tz)
        if unit in ("주", "주일"):
            d = today - timedelta(days=7 * n)
            return _range(d - timedelta(days=3), d + timedelta(days=4), raw, tz)
        d = _add_months(today, -n)
        return _range(d, _add_months(d, 1), raw, tz)
    m = re.search(r"(하루|이틀|사흘|나흘|닷새|엿새|이레|여드레|아흐레|열흘)전", t)
    if m:
        d = today - timedelta(days=_NUM_WORDS[m.group(1)])
        return _range(d, d + timedelta(days=1), raw, tz)
    if re.search(r"며칠전|몇일전|며칠", t):
        return _range(today - timedelta(days=7), today + timedelta(days=1), raw, tz)
    if re.search(r"일주일전", t):
        d = today - timedelta(days=7)
        return _range(d - timedelta(days=3), d + timedelta(days=4), raw, tz)

    # --- 날 ---
    if "엊그제" in t or "엊그저께" in t:
        return _range(today - timedelta(days=3), today, raw, tz)
    if "그저께" in t or "그제" in t:
        d = today - timedelta(days=2)
        return _range(d, d + timedelta(days=1), raw, tz)
    if "어제" in t:
        d = today - timedelta(days=1)
        return _range(d, d + timedelta(days=1), raw, tz)
    if "오늘" in t:
        return _range(today, today + timedelta(days=1), raw, tz)

    # --- 주 ---
    monday = today - timedelta(days=today.weekday())
    if "지지난주" in t:
        return _range(monday - timedelta(days=14), monday - timedelta(days=7), raw, tz)
    if "주말" in t:
        sat = monday - timedelta(days=2)          # 지난 토요일
        if "지난" not in t and "저번" not in t and today.weekday() >= 5:
            sat = monday + timedelta(days=5)      # 오늘이 주말이면 이번 주말
        return _range(sat, sat + timedelta(days=2), raw, tz)
    if re.search(r"지난주|저번주|전주", t):
        return _range(monday - timedelta(days=7), monday, raw, tz)
    if re.search(r"이번주|금주", t):
        return _range(monday, today + timedelta(days=1), raw, tz)

    # --- 달 / 해 ---
    if re.search(r"지난달|저번달|전달", t):
        start = _add_months(today, -1)
        d1, d2 = _part_of_month(start, _month_start(today), part)
        return _range(d1, d2, raw, tz)
    if re.search(r"이번달|이달|금월", t):
        d1, d2 = _part_of_month(_month_start(today), _add_months(today, 1), part)
        if d1 > today:
            return None
        return _range(*_clip(d1, d2, today), raw, tz)
    if "올초" in t:
        part, year = "초", today.year
    if year is not None:
        d1, d2 = _part_of_year(year, part)
        if d1 > today:
            return None
        return _range(*_clip(d1, d2, today), raw, tz)

    # --- 최근 ---
    if re.search(r"최근|요즘|요새|근래|얼마전", t):
        return _range(today - timedelta(days=30), today + timedelta(days=1), raw, tz)

    if re.search(r"지난번|저번|예전|이전|옛날", t):
        return Period(raw, vague=True)
    return None


def to_local(iso_text: str) -> datetime | None:
    """DB 의 created_at(ISO, UTC) -> 지역 시각."""
    try:
        moment = datetime.fromisoformat(iso_text)
    except (TypeError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(local_tz())
