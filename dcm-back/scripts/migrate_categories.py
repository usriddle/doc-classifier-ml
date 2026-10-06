"""
민원 DB(complaints.db) 카테고리 이전 - 예전 이름 -> 지금 7종.

    cd backend
    python scripts/migrate_categories.py            # 바뀔 내용만 보여 줌 (DB 는 그대로)
    python scripts/migrate_categories.py --apply    # 백업 파일을 만든 뒤 실제로 바꿈

- 변환 규칙은 학습 데이터 재라벨과 같은 training/category_migration.map_category 를 씁니다.
    예전 9종 : 노동, 기업 -> 노동·기업 / 교통, 건설·국토 -> 교통·국토 (이름만 합침)
    처음 7종 : 행정안전·소방 -> 문화·행정·안전, 국토교통 -> 교통·국토, 주택건축 -> 주택·건축, 보건복지 -> 보건·복지
- 이미 7종 이름이면 건드리지 않으므로 여러 번 실행해도 안전합니다.
- '기타' 는 이름이 같아서 처음 7종 DB(행정안전·국토교통 같은 이름이 남아 있는 DB)일 때만 내용으로 다시 봅니다.
  (도서관·체육시설 -> 문화·행정·안전, 지역화폐·전통시장 -> 노동·기업, 국유지·토지 보상 -> 교통·국토)
  자동 판단 대신 직접 정하려면 --split-etc(항상 다시 봄) / --keep-etc(절대 안 봄)를 붙이세요.
- --apply 는 같은 폴더에 complaints.db.bak-YYYYmmdd-HHMMSS 백업을 먼저 만듭니다.
- 바뀐 민원마다 처리 이력(complaint_history)에 '카테고리 변경 A -> B' 한 줄을 남깁니다. (상태는 그대로)
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.config import settings  # noqa: E402
from app.services import categories  # noqa: E402
from training.category_migration import LEGACY7_ONLY, OLD_NAMES, map_category  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="complaints.db 카테고리 예전 이름 -> 7종 이전")
    parser.add_argument("--db", default=settings.COMPLAINT_DB_PATH, help="민원 DB 경로 (기본: .env 의 COMPLAINT_DB_PATH)")
    parser.add_argument("--apply", action="store_true", help="실제로 바꿉니다. 없으면 미리보기만")
    etc = parser.add_mutually_exclusive_group()
    etc.add_argument("--split-etc", action="store_true",
                     help="'기타' 민원을 처음 7종의 기타로 보고 내용으로 다시 나눕니다.")
    etc.add_argument("--keep-etc", action="store_true",
                     help="'기타' 민원은 다시 나누지 않습니다.")
    args = parser.parse_args()

    path = Path(args.db)
    if not path.is_absolute():
        path = (BACKEND_DIR / path).resolve()
    if not path.exists():
        print(f"민원 DB 가 없습니다: {path}  (아직 접수된 민원이 없으면 이전할 것도 없습니다)")
        return 0

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, category, content, location, status FROM complaints").fetchall()

    # 처음 7종 이름이 남아 있으면 첫 번째 개편도 안 거친 DB - 기타도 다시 나눔
    legacy7 = any((row["category"] or "") in LEGACY7_ONLY for row in rows)
    split_etc = args.split_etc or (legacy7 and not args.keep_etc)

    changes: list[tuple[int, str, str, str]] = []
    unknown: Counter[str] = Counter()
    for row in rows:
        old = row["category"] or ""
        if old == "기타":
            if split_etc:
                new = map_category(old, row["content"] or "", legacy_etc=True)
                if new != old:
                    changes.append((row["id"], old, new, row["content"] or ""))
            continue
        if old in categories.NAMES:
            continue
        if old not in OLD_NAMES:
            unknown[old] += 1
            continue
        new = map_category(old, row["content"] or "")
        changes.append((row["id"], old, new, row["content"] or ""))

    print(f"DB: {path}")
    print(f"전체 {len(rows)}건 | 바꿀 것 {len(changes)}건 | 기타 다시 나누기: {'예' if split_etc else '아니오'}"
          + (" (처음 7종 이름이 남아 있음)" if legacy7 else ""))
    for (old, new), n in sorted(Counter((o, n) for _, o, n, _ in changes).items()):
        print(f"  {old:6} -> {new:10} {n:5}건")
    if unknown:
        print("  ※ 예전·지금 어느 이름에도 없는 값(그대로 둠):", dict(unknown))
    for cid, old, new, content in changes[:20]:
        print(f"    #{cid:<5} {old} -> {new} | {content[:40]}")
    if len(changes) > 20:
        print(f"    … 외 {len(changes) - 20}건")

    if not args.apply:
        print("\n미리보기입니다. 실제로 바꾸려면 --apply 를 붙여 다시 실행하세요.")
        conn.close()
        return 0
    if not changes:
        print("바꿀 것이 없습니다.")
        conn.close()
        return 0

    backup = path.with_name(f"{path.name}.bak-{datetime.now():%Y%m%d-%H%M%S}")
    conn.close()
    shutil.copy2(path, backup)
    print(f"\n백업: {backup}")

    conn = sqlite3.connect(str(path))
    now = datetime.now().isoformat(timespec="seconds")
    with conn:
        for cid, old, new, _ in changes:
            conn.execute("UPDATE complaints SET category = ? WHERE id = ?", (new, cid))
            status = conn.execute("SELECT status FROM complaints WHERE id = ?", (cid,)).fetchone()[0]
            conn.execute(
                "INSERT INTO complaint_history(complaint_id, status, note, changed_at) VALUES (?, ?, ?, ?)",
                (cid, status, f"카테고리 변경 {old} -> {new} (분류 개편)", now),
            )
    conn.close()
    print(f"{len(changes)}건을 바꿨습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
