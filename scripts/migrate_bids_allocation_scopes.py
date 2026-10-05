#!/usr/bin/env python3
"""
Move BIDS allocations from per-lesion-type numbering to per-dataset numbering.

The migration itself lives in utils/bids_allocation_migration.py and runs by
itself the first time an old-format table is opened (KI-060). This script is
the manual front-end: see what would move, or apply on purpose.

Numbers are never changed; the old table is kept as
bids_patient_allocation_legacy and a copy of the DB is taken before writing.

Usage:
    python scripts/migrate_bids_allocation_scopes.py                    # dry run
    python scripts/migrate_bids_allocation_scopes.py --apply            # for real
    python scripts/migrate_bids_allocation_scopes.py --db path/to.db --apply
"""

import argparse
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.bids_allocation_migration import (  # noqa: E402
    MigrationConflict,
    find_conflicts,
    migrate as _migrate,
)

DEFAULT_DB = Path(__file__).resolve().parents[1] / "backend" / "data" / "brain_lesion.db"

__all__ = ["find_conflicts", "migrate"]


def migrate(db_path: Union[str, Path], dry_run: bool = True) -> dict:
    """CLI semantics: print the outcome; exit 1 on in-dataset conflicts."""
    try:
        result = _migrate(db_path, dry_run=dry_run)
    except MigrationConflict as conflict:
        print("Конфликты внутри датасета — миграция остановлена:")
        for line in conflict.conflicts:
            print("  -", line)
        raise SystemExit(1)
    if result is None:
        print("Таблица уже в новом формате (или её нет) — делать нечего.")
        return {}
    print("Сухой прогон:" if dry_run else "Готово:", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--apply", action="store_true",
                        help="Without it the script only reports what it would do")
    args = parser.parse_args()
    migrate(args.db, dry_run=not args.apply)


if __name__ == "__main__":
    main()
