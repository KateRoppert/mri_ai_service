#!/usr/bin/env python3
"""
Move BIDS allocations from per-lesion-type numbering to per-dataset numbering.

Numbers are never changed: a number already issued is an entity name in Kappa.
Rows are only re-filed — into the dataset they were uploaded to, or into the
local scope when they were never uploaded. The old table is kept under
bids_patient_allocation_legacy so the change can be undone.

See docs/superpowers/specs/2026-09-21-bids-numbering-per-dataset-design.md.

Usage:
    python scripts/migrate_bids_allocation_scopes.py                    # dry run
    python scripts/migrate_bids_allocation_scopes.py --apply            # for real
    python scripts/migrate_bids_allocation_scopes.py --db path/to.db --apply
"""

import argparse
import sqlite3
from pathlib import Path
from typing import Union

DEFAULT_DB = Path(__file__).resolve().parents[1] / "backend" / "data" / "brain_lesion.db"


def _subject(bids_id: str) -> str:
    """'sub-001_ses-002' -> 'sub-001'."""
    return (bids_id or "").split("_", 1)[0]


def find_conflicts(db_path: Union[str, Path]) -> list:
    """Problems that make an automatic move unsafe, as readable lines.

    Only collisions INSIDE one dataset matter. The same number in two
    different datasets is exactly what this migration separates, not a
    conflict — that is the real sub-003 case (datasets 133 and 249).
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT kappa_dataset_id, bids_id, original_patient_id "
            "FROM patient_registry WHERE kappa_dataset_id IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()

    by_number: dict = {}
    by_person: dict = {}
    for dataset_id, bids_id, original in rows:
        by_number.setdefault((dataset_id, _subject(bids_id)), set()).add(original)
        by_person.setdefault((dataset_id, original), set()).add(_subject(bids_id))

    conflicts = []
    for (dataset_id, subject), people in sorted(by_number.items()):
        if len(people) > 1:
            conflicts.append(
                f"dataset {dataset_id}: {subject} is held by {len(people)} patients "
                f"({', '.join(sorted(people))})")
    for (dataset_id, original), subjects in sorted(by_person.items()):
        if len(subjects) > 1:
            conflicts.append(
                f"dataset {dataset_id}: patient {original} holds {len(subjects)} numbers "
                f"({', '.join(sorted(subjects))})")
    return conflicts


def migrate(db_path: Union[str, Path], dry_run: bool = True) -> dict:
    """Re-file every allocation into its scope. Aborts (SystemExit) on any
    in-dataset conflict rather than guessing how to split it."""
    conflicts = find_conflicts(db_path)
    if conflicts:
        print("Конфликты внутри датасета — миграция остановлена:")
        for line in conflicts:
            print("  -", line)
        raise SystemExit(1)

    conn = sqlite3.connect(str(db_path))
    try:
        legacy = conn.execute(
            "SELECT lesion_type, original_patient_id, bids_id, created_at "
            "FROM bids_patient_allocation"
        ).fetchall()
        registry = {
            (lesion, original): dataset_id
            for dataset_id, original, lesion in conn.execute(
                "SELECT kappa_dataset_id, original_patient_id, lesion_type "
                "FROM patient_registry WHERE kappa_dataset_id IS NOT NULL")
        }
        floors: dict = {}
        for dataset_id, bids_id in conn.execute(
            "SELECT kappa_dataset_id, bids_id FROM patient_registry "
            "WHERE kappa_dataset_id IS NOT NULL"
        ):
            try:
                number = int(_subject(bids_id).split("-", 1)[1])
            except (IndexError, ValueError):
                continue
            floors[dataset_id] = max(floors.get(dataset_id, 0), number)

        planned = []
        for lesion, original, bids_id, created_at in legacy:
            dataset_id = registry.get((lesion, original))
            scope = f"ds:{dataset_id}" if dataset_id else f"local:{lesion}"
            planned.append((scope, original, bids_id, created_at))

        result = {
            "moved_to_datasets": sum(1 for s, *_ in planned if s.startswith("ds:")),
            "moved_to_local": sum(1 for s, *_ in planned if s.startswith("local:")),
            "floors": len(floors),
        }
        if dry_run:
            print("Сухой прогон:", result)
            return result

        conn.execute("BEGIN IMMEDIATE")
        conn.execute("ALTER TABLE bids_patient_allocation "
                     "RENAME TO bids_patient_allocation_legacy")
        conn.execute("""
            CREATE TABLE bids_patient_allocation (
                scope                TEXT NOT NULL,
                original_patient_id  TEXT NOT NULL,
                bids_id              TEXT NOT NULL,
                created_at           TEXT NOT NULL,
                PRIMARY KEY (scope, original_patient_id),
                UNIQUE (scope, bids_id)
            )""")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bids_scope_floor (
                scope TEXT PRIMARY KEY,
                floor INTEGER NOT NULL
            )""")
        conn.executemany(
            "INSERT INTO bids_patient_allocation "
            "(scope, original_patient_id, bids_id, created_at) VALUES (?,?,?,?)",
            planned)
        conn.executemany(
            "INSERT OR REPLACE INTO bids_scope_floor (scope, floor) VALUES (?,?)",
            [(f"ds:{dataset_id}", floor) for dataset_id, floor in floors.items()])
        conn.commit()
        print("Готово:", result)
        return result
    finally:
        conn.close()


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
