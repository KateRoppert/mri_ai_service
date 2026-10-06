"""
Move BIDS allocations from per-lesion-type numbering to per-dataset numbering.

Numbers are never changed: a number already issued is an entity name in Kappa.
Rows are only re-filed — into the dataset they were uploaded to, or into the
local scope when they were never uploaded. The old table is kept under
bids_patient_allocation_legacy, and a copy of the whole DB is taken first, so
the change can be undone.

Runs by itself the first time anything opens an old-format table through
utils.bids_allocator._connect (KI-060); scripts/migrate_bids_allocation_scopes.py
is the manual front-end (dry run by default).

See docs/superpowers/specs/2026-09-21-bids-numbering-per-dataset-design.md and
docs/superpowers/specs/2026-10-05-ki-060-auto-bids-migration-design.md.
"""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

TABLE = "bids_patient_allocation"
LEGACY_TABLE = "bids_patient_allocation_legacy"
FLOOR_TABLE = "bids_scope_floor"
SCRIPT = "scripts/migrate_bids_allocation_scopes.py"


class MigrationConflict(RuntimeError):
    """In-dataset conflicts that make an automatic move unsafe."""

    def __init__(self, conflicts: list):
        self.conflicts = conflicts
        lines = "\n".join(f"  - {c}" for c in conflicts)
        super().__init__(
            "BIDS-нумерацию нельзя перевести на датасеты автоматически — "
            "конфликты внутри датасета:\n"
            f"{lines}\n"
            "База не изменена. Разберите конфликты в patient_registry и "
            f"проверьте сухим прогоном: python {SCRIPT}"
        )


def _subject(bids_id: str) -> str:
    """'sub-001_ses-002' -> 'sub-001'."""
    return (bids_id or "").split("_", 1)[0]


def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def is_legacy(conn: sqlite3.Connection) -> bool:
    """True when the allocation table exists in the per-lesion-type format."""
    cols = _columns(conn, TABLE)
    return "lesion_type" in cols and "scope" not in cols


def _registry_rows(conn: sqlite3.Connection) -> list:
    """(kappa_dataset_id, bids_id, original_patient_id, lesion_type) of
    uploaded sessions; none when the DB never had a registry (CLI-only)."""
    if not _columns(conn, "patient_registry"):
        return []
    return conn.execute(
        "SELECT kappa_dataset_id, bids_id, original_patient_id, lesion_type "
        "FROM patient_registry WHERE kappa_dataset_id IS NOT NULL"
    ).fetchall()


def _conflicts(rows: list) -> list:
    """Only collisions INSIDE one dataset matter. The same number in two
    different datasets is exactly what this migration separates, not a
    conflict — that is the real sub-003 case (datasets 133 and 249)."""
    by_number: dict = {}
    by_person: dict = {}
    for dataset_id, bids_id, original, _lesion in rows:
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


def find_conflicts(db_path: Union[str, Path]) -> list:
    """Problems that make an automatic move unsafe, as readable lines."""
    conn = sqlite3.connect(str(db_path))
    try:
        return _conflicts(_registry_rows(conn))
    finally:
        conn.close()


def _plan(conn: sqlite3.Connection):
    rows = _registry_rows(conn)
    conflicts = _conflicts(rows)
    if conflicts:
        raise MigrationConflict(conflicts)

    registry = {(lesion, original): dataset_id for dataset_id, _b, original, lesion in rows}
    floors: dict = {}
    for dataset_id, bids_id, _original, _lesion in rows:
        try:
            number = int(_subject(bids_id).split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        floors[dataset_id] = max(floors.get(dataset_id, 0), number)

    planned = []
    for lesion, original, bids_id, created_at in conn.execute(
        f"SELECT lesion_type, original_patient_id, bids_id, created_at FROM {TABLE}"
    ):
        dataset_id = registry.get((lesion, original))
        scope = f"ds:{dataset_id}" if dataset_id else f"local:{lesion}"
        planned.append((scope, original, bids_id, created_at))

    result = {
        "moved_to_datasets": sum(1 for s, *_ in planned if s.startswith("ds:")),
        "moved_to_local": sum(1 for s, *_ in planned if s.startswith("local:")),
        "floors": len(floors),
    }
    return planned, floors, result


def _backup(db_path: Path) -> Path:
    """Copy the whole DB with SQLite's online backup — consistent even while
    the backend holds the file open."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = db_path.with_name(f"{db_path.name}.bak-{stamp}-before-scope-migration")
    n = 1
    while target.exists():  # two migrations in one second: never overwrite
        target = db_path.with_name(f"{db_path.name}.bak-{stamp}-{n}-before-scope-migration")
        n += 1
    src = sqlite3.connect(str(db_path))
    dst = sqlite3.connect(str(target))
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    return target


def migrate(db_path: Union[str, Path], dry_run: bool = True) -> Optional[dict]:
    """Re-file every allocation into its scope.

    Returns the counts, or None when the table is not in the old format
    (already migrated — possibly by a concurrent process — or absent).
    Raises MigrationConflict, changing nothing, on any in-dataset conflict.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None)
    try:
        if not is_legacy(conn):
            return None
        planned, floors, result = _plan(conn)
        if dry_run:
            return result

        conn.execute("BEGIN IMMEDIATE")
        backup = None
        try:
            # Whoever got the write lock second finds the work done.
            if not is_legacy(conn):
                conn.execute("ROLLBACK")
                return None
            # Re-plan under the lock: rows may have changed since the first look.
            planned, floors, result = _plan(conn)
            # Backup only by the process that really migrates, and only now:
            # taken before the lock, two racing processes picked the same
            # file name in the same second and the loser deleted the winner's
            # copy. Reading through another connection is allowed while this
            # one holds the write lock but has written nothing yet.
            backup = _backup(db_path)
            conn.execute(f"ALTER TABLE {TABLE} RENAME TO {LEGACY_TABLE}")
            conn.execute(f"""
                CREATE TABLE {TABLE} (
                    scope                TEXT NOT NULL,
                    original_patient_id  TEXT NOT NULL,
                    bids_id              TEXT NOT NULL,
                    created_at           TEXT NOT NULL,
                    PRIMARY KEY (scope, original_patient_id),
                    UNIQUE (scope, bids_id)
                )""")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {FLOOR_TABLE} (
                    scope TEXT PRIMARY KEY,
                    floor INTEGER NOT NULL
                )""")
            conn.executemany(
                f"INSERT INTO {TABLE} (scope, original_patient_id, bids_id, created_at) "
                "VALUES (?,?,?,?)", planned)
            conn.executemany(
                f"INSERT OR REPLACE INTO {FLOOR_TABLE} (scope, floor) VALUES (?,?)",
                [(f"ds:{dataset_id}", floor) for dataset_id, floor in floors.items()])
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            if backup is not None:
                backup.unlink(missing_ok=True)  # nothing changed — no undo needed
            raise

        logger.warning(
            "BIDS-нумерация переведена на датасеты (KI-060): %d записей в датасеты, "
            "%d в локальную нумерацию, %d нижних границ; старая таблица — %s, "
            "копия базы — %s",
            result["moved_to_datasets"], result["moved_to_local"], result["floors"],
            LEGACY_TABLE, backup.name,
        )
        return result
    finally:
        conn.close()
