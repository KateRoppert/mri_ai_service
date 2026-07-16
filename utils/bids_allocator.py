"""
Persistent, concurrency-safe allocation of BIDS subject IDs.

Problem this solves
--------------------
Stage 01 used to number patients sub-001, sub-002, ... from scratch on every
run, because the mapping lived in a per-run file (output_dir/dataset_mapping.json).
Two runs writing to the same Kappa dataset therefore both produced "sub-001" for
DIFFERENT patients, colliding in Kappa and corrupting the longitudinal view (a
report opened for one patient showed another patient's timeline).

Design
------
A single SQLite table is the source of truth for the mapping
(original_patient_id -> bids_id), scoped by lesion_type. lesion_type maps 1:1 to
a Kappa dataset in the current configuration (multiple_sclerosis -> 158,
glioblastoma -> 133), so this is exactly the "one numbering space per dataset"
boundary, and it matches the BIDS convention of numbering subjects within a
dataset.

Guarantees:
  * Stable: the same patient always gets the same bids_id (PRIMARY KEY reuse).
  * Monotonic: a new patient gets the next free number within its lesion_type.
  * Collision-proof: UNIQUE(lesion_type, bids_id) makes it physically impossible
    for two patients to share a bids_id, even under a logic bug.
  * Concurrency-safe: allocation runs inside a BEGIN IMMEDIATE transaction, so
    two pipeline runs started at once serialize instead of racing on the counter.

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


def _resolve_db_path(db_path: Optional[Union[str, Path]]) -> Path:
    """Resolve the DB path: explicit arg > BRAIN_LESION_DB env > default."""
    if db_path is not None:
        return Path(db_path)
    env = os.environ.get("BRAIN_LESION_DB")
    if env:
        return Path(env)
    return DEFAULT_DB_PATH


def _connect(db_path: Optional[Union[str, Path]]) -> sqlite3.Connection:
    """Open a connection with a busy timeout and ensure the table exists."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # timeout lets a concurrent run wait for the write lock instead of erroring.
    conn = sqlite3.connect(str(path), timeout=30.0)
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_TABLE} (
            lesion_type          TEXT NOT NULL,
            original_patient_id  TEXT NOT NULL,
            bids_id              TEXT NOT NULL,
            created_at           TEXT NOT NULL,
            PRIMARY KEY (lesion_type, original_patient_id),
            UNIQUE (lesion_type, bids_id)
        )
        """
    )
    conn.commit()
    return conn


def _next_bids_id(conn: sqlite3.Connection, lesion_type: str) -> str:
    """Compute the next sub-NNN for a lesion_type from the max existing number."""
    rows = conn.execute(
        f"SELECT bids_id FROM {_TABLE} WHERE lesion_type = ?", (lesion_type,)
    ).fetchall()
    max_n = 0
    for (bids_id,) in rows:
        # bids_id is "sub-NNN"; ignore anything that does not parse.
        try:
            max_n = max(max_n, int(bids_id.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return f"sub-{max_n + 1:03d}"


def get_or_allocate(
    lesion_type: str,
    original_patient_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> str:
    """Return the stable bids_id for a patient, allocating one if needed.

    Atomic: the read-or-insert runs inside a BEGIN IMMEDIATE transaction so
    concurrent callers cannot allocate the same number.
    """
    conn = _connect(db_path)
    try:
        # BEGIN IMMEDIATE takes the write lock up front, serialising allocation
        # across processes (two pipeline runs starting at the same time).
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            f"SELECT bids_id FROM {_TABLE} "
            f"WHERE lesion_type = ? AND original_patient_id = ?",
            (lesion_type, original_patient_id),
        ).fetchone()
        if row is not None:
            conn.commit()
            return row[0]

        bids_id = _next_bids_id(conn, lesion_type)
        conn.execute(
            f"INSERT INTO {_TABLE} "
            f"(lesion_type, original_patient_id, bids_id, created_at) "
            f"VALUES (?, ?, ?, ?)",
            (lesion_type, original_patient_id, bids_id,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return bids_id
    finally:
        conn.close()


def get_bids_id(
    lesion_type: str,
    original_patient_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Read-only lookup: bids_id for a patient, or None if not yet allocated."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            f"SELECT bids_id FROM {_TABLE} "
            f"WHERE lesion_type = ? AND original_patient_id = ?",
            (lesion_type, original_patient_id),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_original_id(
    lesion_type: str,
    bids_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[str]:
    """Reverse lookup: the real patient id behind a bids_id (for the clinical UI)."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            f"SELECT original_patient_id FROM {_TABLE} "
            f"WHERE lesion_type = ? AND bids_id = ?",
            (lesion_type, bids_id),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def get_allocations(
    lesion_type: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Dict[str, str]:
    """Return the full {original_patient_id: bids_id} map for a lesion_type."""
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            f"SELECT original_patient_id, bids_id FROM {_TABLE} "
            f"WHERE lesion_type = ?",
            (lesion_type,),
        ).fetchall()
        return {orig: bids for orig, bids in rows}
    finally:
        conn.close()
