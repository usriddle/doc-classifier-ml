"""
벡터DB 적재 + 동작 확인 스크립트.

서버를 띄우지 않고 .env 설정 그대로 벡터DB 를 준비한 뒤, 저장된 사례로 검색이 되는지 확인합니다.

    cd backend
    python scripts/sync_cases.py

- CASE_STORE_BACKEND=pgvector 이면
    DB 의 지문이 CSV 와 다를 때 data/cases.npy + data/cases.meta.json 을 테이블에 적재합니다.
    (bge-m3 가 없는 PC 는 Colab 이 만든 두 파일을 data 폴더에 복사해 두어야 합니다)
- CASE_STORE_BACKEND=numpy 이면
    .npy 캐시를 확인하거나(없으면 bge-m3 로 새로 만듭니다) 메모리에 올립니다.

확인 방법 : 저장된 사례 3개를 골라 "그 사례 자신의 벡터"로 검색합니다.
  1위가 자기 자신(유사도 1.0000)이고 뒤에 같은 카테고리 사례가 이어지면 정상입니다.
  (질의문을 새로 임베딩하지 않으므로 bge-m3 가 없어도 확인할 수 있습니다)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))
os.chdir(BACKEND_DIR)

from app.config import settings  # noqa: E402
from app.services import case_store, pg_store  # noqa: E402


def main() -> int:
    print(f"backend  : {case_store.backend()}")
    print(f"csv      : {case_store.csv_path()}")
    if case_store.backend() == "pgvector":
        print(f"table    : {settings.CASE_PG_TABLE}")

    case_store.load_or_build()
    st = case_store.status()
    print(f"ready    : {st['ready']}  (사례 {st['cases']}건, 건너뜀 {st['skipped_rows']}, source={st['source']})")
    if not st["ready"]:
        print(f"\n[실패] {st['error']}")
        return 1

    if case_store.backend() == "pgvector":
        samples = pg_store.sample(3)
    else:
        import random

        picked = random.sample(range(len(case_store._cases)), k=min(3, len(case_store._cases)))
        samples = [
            (case_store._cases[i].text, case_store._cases[i].category, case_store._vectors[i])
            for i in picked
        ]

    print("\n자기 벡터로 검색 (1위가 자기 자신이면 정상)")
    ok = True
    for text, category, vector in samples:
        hits = case_store.search(vector, top_k=3)
        print(f"\n  질의 [{category}] {text}")
        for h in hits:
            print(f"    {h.rank}. [{h.category}] {h.score:.4f}  {h.text}")
        if not hits or hits[0].text != text:
            ok = False
    print("\n결과     :", "정상" if ok else "확인 필요 - 1위가 자기 자신이 아닙니다")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
