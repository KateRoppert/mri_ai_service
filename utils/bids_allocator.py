"""
Persistent, concurrency-safe allocation of BIDS subject IDs.

Problem this solves
--------------------
Stage 01 used to number patients sub-001, sub-002, ... from scratch on every
run, because the mapping lived in a per-run file (output_dir/dataset_mapping.json).
Two runs writing to the same Kappa dataset therefore both produced "sub-001" for
DIFFERENT patients, colliding in Kappa and corrupting the longitudinal view (a
report opened for one patient showed another patient's timeline).

That was fixed by scoping allocation to lesion_type (one SQLite table, one
numbering space per lesion_type) on the assumption that lesion_type maps 1:1 to
a Kappa dataset. The per-user Kappa dataset change broke that assumption: every
Kappa user now gets their own dataset per lesion_type, and Kate also creates
datasets by hand. A brand-new dataset needs to start at sub-001, not continue a
global counter — see docs/superpowers/specs/2026-09-21-bids-numbering-per-dataset-design.md.

Design
------
Allocation is keyed by an opaque `scope` string rather than lesion_type:
  * `dataset_scope(dataset_id)`  -> "ds:337"          one Kappa dataset
  * `local_scope(lesion_type)`   -> "local:glioblastoma"   CLI runs, no Kappa
  * `pending_scope(run_id)`      -> "pending:<run_id>"     Kappa was unreachable
                                                            at run start; bound to
                                                            a real dataset scope
                                                            once one is created

A scope's next number is the higher of two things: what this table has already
allocated in that scope, and an externally supplied "floor" (set_floor) — the
highest sub-NNN the Kappa dataset itself already contains. The floor is what
makes a hand-made dataset, or a dataset another machine has been writing to,
number correctly: the dataset is the authority, this table is a cache of it.

Guarantees:
  * Stable: the same patient always gets the same bids_id (PRIMARY KEY reuse).
  * Monotonic: a new patient gets the next free number within its scope.
  * Collision-proof: UNIQUE(scope, bids_id) makes it physically impossible for
    two patients to share a bids_id within a scope, even under a logic bug.
  * Concurrency-safe: allocation runs inside a BEGIN IMMEDIATE transaction, so
    two pipeline runs started at once serialize instead of racing on the counter.
    (Two DIFFERENT machines racing into the same Kappa dataset at the same
    moment are not covered — see the spec's Limitations.)

Dependency-free (stdlib sqlite3 only) so it can be imported both from the
standalone Stage 01 subprocess and from the FastAPI backend without pulling in
either side's package.
"""

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Union

# utils/ sits at <repo>/utils, so parents[1] is the repo root. In the container
# the repo root is /app, giving /app/backend/data/brain_lesion.db — the same
# file the backend uses (see backend/config.py database_url).
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "backend" / "data" / "brain_lesion.db"

_TABLE = "bids_patient_allocation"
_FLOOR_TABLE = "bids_scope_floor"


def dataset_scope(dataset_id: int) -> str:
    """Numbering space of a Kappa dataset."""
    return f"ds:{dataset_id}"


def local_scope(lesion_type: str) -> str:
    """Numbering space for runs with no Kappa behind them (CLI)."""
    return f"local:{lesion_type}"


def pending_scope(run_id: str) -> str:
    """Temporary numbering space for a run that had no dataset yet because
    Kappa was unreachable at start.

    Bound to a real dataset with rebind_scope() once upload resolves one.

    This scope starts at sub-001, so binding it into a dataset that already
    holds numbers DOES collide — that assumption was wrong, and the collision
    used to surface as a UNIQUE violation mid-upload. rebind_scope now leaves
    colliding numbers behind for the uploader to report as a name clash, and
    backend/numbering.py avoids the scope entirely whenever the dataset can be
    determined from configs/kappa_datasets.yaml without Kappa.
    """
    return f"pending:{run_id}"


def _resolve_db_path(db_path: Optional[Union[str, Path]]) -> Path:
    """Resolve the DB path: explicit arg > BRAIN_LESION_DB env > default."""
    if db_path is not None:
        return Path(db_path)
    env = os.environ.get("BRAIN_LESION_DB")
    if env:
        return Path(env)
    return DEFAULT_DB_PATH


def _connect(db_path: Optional[Union[str, Path]]) -> sqlite3.Connection:
    """Open a connection with a busy timeout and ensure the tables exist."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout lets a concurrent run wait for the write lock instead of erroring.
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            scope                TEXT NOT NULL,
            original_patient_id  TEXT NOT NULL,
            bids_id              TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            PRIMARY KEY (scope, original_patient_id),
            UNIQUE (scope, bids_id)
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_FLOOR_TABLE} (
            scope TEXT PRIMARY KEY,
            floor INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def _next_bids_id(conn: sqlite3.Connection, scope: str) -> str:
    """Next sub-NNN for a scope: above both what we allocated and what the
    dataset already contains (the floor)."""
    rows = conn.execute(
        f"SELECT bids_id FROM {_TABLE} WHERE scope = ?", (scope,)
    ).fetchall()
    max_n = 0
    for (bids_id,) in rows:
        # bids_id is "sub-NNN"; ignore anything that does not parse.
        try:
            max_n = max(max_n, int(bids_id.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    floor_row = conn.execute(
        f"SELECT floor FROM {_FLOOR_TABLE} WHERE scope = ?", (scope,)
    ).fetchone()
    if floor_row:
        max_n = max(max_n, int(floor_row[0]))
    return f"sub-{max_n + 1:03d}"


def get_or_allocate(
    scope: str,
    original_patient_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> str:
    """Return the stable bids_id for a patient within a scope, allocating one
    if needed.

    Atomic: the read-or-insert runs inside a BEGIN IMMEDIATE transaction so
    concurrent callers on this machine cannot allocate the same number.
    """
    conn = _connect(db_path)
    try:
        # BEGIN IMMEDIATE takes the write lock up front, serialising allocation
        # across processes (two pipeline runs starting at the same time).
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT bids_id FROM {_TABLE} "
            f"WHERE scope = ? AND original_patient_id = ?",
            (scope, original_patient_id),
        ).fetchone()
        if row is not None:
            conn.commit()
            return row[0]

        bids_id = _next_bids_id(conn, scope)
        conn.execute(
            f"INSERT INTO {_TABLE} "
            f"(scope, original_patient_id, bids_id, created_at) "
            f"VALUES (?, ?, ?, ?)",
            (scope, original_patient_id, bids_id,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return bids_id
    finally:
        conn.close()


def get_bids_id(
    scope: str,
    original_patient_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Read-only lookup: bids_id for a patient within a scope, or None."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            f"SELECT bids_id FROM {_TABLE} "
            f"WHERE scope = ? AND original_patient_id = ?",
            (scope, original_patient_id),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_original_id(
    scope: str,
    bids_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Reverse lookup: the real patient id behind a bids_id within a scope."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            f"SELECT original_patient_id FROM {_TABLE} "
            f"WHERE scope = ? AND bids_id = ?",
            (scope, bids_id),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_allocations(
    scope: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Return the full {original_patient_id: bids_id} map for a scope."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT original_patient_id, bids_id FROM {_TABLE} "
            f"WHERE scope = ?",
            (scope,),
        ).fetchall()
        return {orig: bids for orig, bids in rows}
    finally:
        conn.close()


def set_floor(scope: str, floor: int, db_path: Optional[Union[str, Path]] = None) -> None:
    """Raise the scope's floor to `floor` (never lowers it).

    The floor is what the Kappa dataset already contains, read externally
    (backend/kappa_dataset_resolver.py) and pushed in here. Persisting it means
    a later allocation in this scope — even from a process that never saw the
    dataset itself, e.g. Stage 01's subprocess — numbers above what the dataset
    held, instead of restarting at sub-001.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT floor FROM {_FLOOR_TABLE} WHERE scope = ?", (scope,)
        ).fetchone()
        current = int(row[0]) if row else 0
        if floor > current:
            conn.execute(
                f"INSERT INTO {_FLOOR_TABLE} (scope, floor) VALUES (?, ?) "
                f"ON CONFLICT(scope) DO UPDATE SET floor = excluded.floor",
                (scope, int(floor)),
            )
        conn.commit()
    finally:
        conn.close()


def rebind_scope(
    old_scope: str, new_scope: str, db_path: Optional[Union[str, Path]] = None
) -> int:
    """Move every allocation from one scope to another; returns the count moved.

    Used when a pending scope's dataset finally gets created (backend/kappa_uploader.py).

    Numbers already taken in the target scope are LEFT BEHIND rather than
    moved. A blanket UPDATE raised UNIQUE(scope, bids_id) and took the whole
    upload down with a 500 — and a crash is the worst possible answer here,
    because the collision itself is a real situation an operator must be told
    about. The rows that stay put keep their pending scope, and the uploader's
    name-clash check reports them per session, which is what the operator can
    actually act on.
    """
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        taken = {
            row[0] for row in conn.execute(
                f"SELECT bids_id FROM {_TABLE} WHERE scope = ?", (new_scope,)
            )
        }
        moved = 0
        for bids_id, in conn.execute(
            f"SELECT bids_id FROM {_TABLE} WHERE scope = ?", (old_scope,)
        ).fetchall():
            if bids_id in taken:
                continue
            conn.execute(
                f"UPDATE {_TABLE} SET scope = ? WHERE scope = ? AND bids_id = ?",
                (new_scope, old_scope, bids_id),
            )
            moved += 1
        # Пол старого скоупа убираем, только если из него всё уехало: иначе
        # оставшиеся записи потеряли бы нижнюю границу нумерации.
        remaining = conn.execute(
            f"SELECT COUNT(*) FROM {_TABLE} WHERE scope = ?", (old_scope,)
        ).fetchone()[0]
        if not remaining:
            conn.execute(f"DELETE FROM {_FLOOR_TABLE} WHERE scope = ?", (old_scope,))
        conn.commit()
        return moved
    finally:
        conn.close()
