"""
카테고리 개편 매핑 (예전 이름 -> 지금 7종).

  지금 7종   : 노동·기업 / 교통·국토 / 주택·건축 / 환경·위생 / 문화·행정·안전 / 보건·복지 / 기타

  2026-10 두 번째 개편 (9종 -> 7종) - 이름만 합치므로 문장 내용을 볼 필요가 없습니다.
    예전 9종 : 노동 / 기업 / 교통 / 주택·건축 / 환경·위생 / 건설·국토 / 문화·행정·안전 / 보건·복지 / 기타
    노동, 기업          -> 노동·기업
    교통, 건설·국토     -> 교통·국토
    나머지 5종          -> 그대로

  2026-10 첫 번째 개편 (처음 7종 -> 9종) 을 아직 거치지 않은 데이터도 바로 7종으로 옮깁니다.
    처음 7종 : 행정안전 / 국토교통 / 주택건축 / 환경·위생 / 보건복지 / 소방 / 기타
    행정안전, 소방      -> 문화·행정·안전
    주택건축            -> 주택·건축
    보건복지            -> 보건·복지
    환경·위생           -> 환경·위생
    국토교통            -> 교통·국토 (예전에는 교통 / 건설·국토 로 나눴지만 이제 하나라 나눌 필요 없음)
    기타                -> 기타, 단 도서관·체육관·문화시설·공연·축제는 문화·행정·안전,
                           지역화폐·상품권·소상공인·전통시장은 노동·기업, 국유지·토지 보상은 교통·국토
                           (공무원·직원 칭찬은 시설과 상관없이 기타)
                           '기타' 는 예전·지금 이름이 같아서, 처음 7종 데이터일 때만 이 규칙으로 다시 봅니다.

학습 데이터(records.jsonl)·사례(cases.csv) 재라벨과 민원 DB(complaints.db) 이전(scripts/migrate_categories.py)이
같은 함수를 씁니다.
"""

from __future__ import annotations

import re

# 예전 9종에서 이름이 바뀐 것
NINE_TO_SEVEN = {
    "노동": "노동·기업",
    "기업": "노동·기업",
    "교통": "교통·국토",
    "건설·국토": "교통·국토",
}

# 처음 7종 (첫 번째 개편 이전)
LEGACY7_NAMES = ("행정안전", "국토교통", "주택건축", "환경·위생", "보건복지", "소방", "기타")
LEGACY7_DIRECT = {
    "행정안전": "문화·행정·안전",
    "소방": "문화·행정·안전",
    "주택건축": "주택·건축",
    "보건복지": "보건·복지",
    "환경·위생": "환경·위생",
    "국토교통": "교통·국토",
}
# 처음 7종에만 있던 이름 (이게 하나라도 있으면 처음 7종 데이터로 봄 - 기타를 다시 나눔)
LEGACY7_ONLY = tuple(n for n in LEGACY7_NAMES if n not in ("환경·위생", "기타"))

# 지금 이름이 아닌, 옮겨야 하는 예전 이름 전부
OLD_NAMES = tuple(NINE_TO_SEVEN) + LEGACY7_ONLY

# --- 처음 7종의 기타 다시 나누기 ------------------------------------------------------
_CULTURE = re.compile(r"도서관|열람실|평생학습|수영장|체육관|체육센터|문화회관|문화센터|문화예술|공연|전시|박물관|미술관|축제|대관")
_BUSINESS_FROM_ETC = re.compile(r"지역화폐|상품권|가맹점|소상공인|전통시장|창업")
_LAND_FROM_ETC = re.compile(r"국유지|공유지|토지 ?보상|토지 ?수용")


def split_legacy_etc(*texts: str) -> str:
    """
    처음 7종의 '기타' 레코드를 문장 내용으로 다시 나눕니다.
    texts 는 짧은 요약(content·keyword)을 먼저, 원문을 마지막에 넘기세요.
    """
    joined = " ".join(t for t in texts if t)
    if "칭찬" in joined:          # 공무원·직원 칭찬은 시설 종류와 상관없이 기타
        return "기타"
    if _CULTURE.search(joined):
        return "문화·행정·안전"
    # 노동·기업·교통·국토 로 옮기는 것은 민원 대상(짧은 요약)으로만 판단합니다.
    # 원문에는 장소 이름(예: '서부전통시장')이 섞여 있어 대상과 상관없이 걸리기 때문입니다.
    labels = [t for t in texts[:-1] if t] if len(texts) > 1 else []
    target = " ".join(labels) if labels else joined
    if _BUSINESS_FROM_ETC.search(target):
        return "노동·기업"
    if _LAND_FROM_ETC.search(target):
        return "교통·국토"
    return "기타"


def map_category(old: str, *texts: str, legacy_etc: bool = False) -> str:
    """
    예전 카테고리 이름 (+ 문장들) -> 지금 7종 이름. '없음'·빈 값·이미 지금 이름이면 그대로.

    legacy_etc=True 면 '기타' 를 처음 7종의 기타로 보고 내용으로 다시 나눕니다.
    (예전 9종·지금 7종의 기타는 그대로 기타)
    """
    old = (old or "").strip()
    if old in ("", "없음"):
        return old
    if old in NINE_TO_SEVEN:
        return NINE_TO_SEVEN[old]
    if old in LEGACY7_DIRECT:
        return LEGACY7_DIRECT[old]
    if old == "기타" and legacy_etc:
        return split_legacy_etc(*texts)
    return old   # 이미 지금 이름이면 그대로
