"""
Tests for the persistent BIDS subject-ID allocator (utils/bids_allocator.py).

These lock in the production guarantees the per-run JSON mapping could not give:
stable IDs across runs, monotonic numbering for new patients, per-lesion_type
scoping, and collision-free allocation under concurrent processes.
"""

import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from utils import bids_allocator


def _load_reorganize():
    """Import scripts/01_reorganize_folders.py (digit-prefixed filename)."""
    spec = importlib.util.spec_from_file_location(
        "reorganize_folders_alloc", SCRIPTS_DIR / "01_reorganize_folders.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["reorganize_folders_alloc"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def db_path(tmp_path):
    """A throwaway SQLite DB per test."""
    return tmp_path / "alloc.db"


def test_same_patient_same_id_across_calls(db_path):
    """A patient keeps its bids_id on repeat runs (the whole point of the fix)."""
    first = bids_allocator.get_or_allocate("multiple_sclerosis", "P000915", db_path)
    second = bids_allocator.get_or_allocate("multiple_sclerosis", "P000915", db_path)
    assert first == second == "sub-001"


def test_new_patient_gets_next_number(db_path):
    """The scenario that broke: a second run must continue numbering, not restart."""
    # Run 1 (earlier, separate) processed P000915.
    assert bids_allocator.get_or_allocate("multiple_sclerosis", "P000915", db_path) == "sub-001"
    # Run 2 processed three other patients — they must NOT reuse sub-001.
    assert bids_allocator.get_or_allocate("multiple_sclerosis", "P000010", db_path) == "sub-002"
    assert bids_allocator.get_or_allocate("multiple_sclerosis", "P000063", db_path) == "sub-003"
    assert bids_allocator.get_or_allocate("multiple_sclerosis", "P000067", db_path) == "sub-004"
    # And P000915 is still sub-001.
    assert bids_allocator.get_or_allocate("multiple_sclerosis", "P000915", db_path) == "sub-001"


def test_scoped_per_lesion_type(db_path):
    """Numbering is independent per lesion_type (= per Kappa dataset)."""
    assert bids_allocator.get_or_allocate("multiple_sclerosis", "P000010", db_path) == "sub-001"
    # Glioblastoma is a different dataset — its own numbering restarts at 001.
    assert bids_allocator.get_or_allocate("glioblastoma", "UPENN-GBM-00001", db_path) == "sub-001"
    assert bids_allocator.get_or_allocate("glioblastoma", "UPENN-GBM-00002", db_path) == "sub-002"
    # The MS space is unaffected.
    assert bids_allocator.get_or_allocate("multiple_sclerosis", "P000063", db_path) == "sub-002"


def test_reverse_and_readonly_lookups(db_path):
    """Read helpers used by the clinical UI."""
    bids_allocator.get_or_allocate("multiple_sclerosis", "P000915", db_path)
    assert bids_allocator.get_bids_id("multiple_sclerosis", "P000915", db_path) == "sub-001"
    assert bids_allocator.get_bids_id("multiple_sclerosis", "P999999", db_path) is None
    assert bids_allocator.get_original_id("multiple_sclerosis", "sub-001", db_path) == "P000915"
    assert bids_allocator.get_original_id("multiple_sclerosis", "sub-099", db_path) is None
    assert bids_allocator.get_allocations("multiple_sclerosis", db_path) == {"P000915": "sub-001"}


def test_readonly_lookup_does_not_allocate(db_path):
    """get_bids_id must never create a mapping as a side effect."""
    assert bids_allocator.get_bids_id("multiple_sclerosis", "P000010", db_path) is None
    assert bids_allocator.get_allocations("multiple_sclerosis", db_path) == {}


def _alloc_worker(args):
    """Top-level so it is picklable by the process pool."""
    db_path, lesion_type, original_id = args
    return bids_allocator.get_or_allocate(lesion_type, original_id, db_path)


def test_concurrent_allocation_no_collisions(db_path):
    """20 distinct patients allocated across 8 processes must get 20 distinct IDs.

    This is the production race: several pipeline runs writing to the same
    dataset at once. BEGIN IMMEDIATE + UNIQUE must yield a clean 1..20 numbering
    with no duplicates and no lost patients.
    """
    patients = [f"P{i:06d}" for i in range(20)]
    args = [(str(db_path), "multiple_sclerosis", p) for p in patients]

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=8) as pool:
        results = pool.map(_alloc_worker, args)

    # Every patient got an id, all ids are distinct, and they form exactly 1..20.
    assert len(set(results)) == 20
    assert sorted(results) == [f"sub-{i:03d}" for i in range(1, 21)]

    # The persisted map agrees and is internally consistent.
    allocations = bids_allocator.get_allocations("multiple_sclerosis", db_path)
    assert len(allocations) == 20
    assert len(set(allocations.values())) == 20


def test_idmapper_db_mode_reproduces_bug_scenario(db_path):
    """IDMapper backed by the allocator must not repeat the two-run collision.

    Exact scenario from the incident: an earlier run processed P000915 alone,
    then a later run processed three other SibBMS patients. Before the fix both
    runs restarted at sub-001. With the DB-backed mapper the second run must
    continue numbering, and P000915 keeps sub-001.
    """
    reorg = _load_reorganize()

    # Run 1: P000915 alone.
    mapper1 = reorg.IDMapper(lesion_type="multiple_sclerosis", db_path=db_path)
    assert mapper1.get_patient_id("P000915") == "sub-001"

    # Run 2: a fresh mapper (new process would have empty in-memory cache).
    mapper2 = reorg.IDMapper(lesion_type="multiple_sclerosis", db_path=db_path)
    assert mapper2.get_patient_id("P000010") == "sub-002"
    assert mapper2.get_patient_id("P000063") == "sub-003"
    assert mapper2.get_patient_id("P000067") == "sub-004"
    # Re-processing P000915 in run 2 still returns its original id.
    assert mapper2.get_patient_id("P000915") == "sub-001"


def test_idmapper_legacy_mode_without_lesion_type(db_path):
    """Without a lesion_type the mapper keeps the old in-memory counter."""
    reorg = _load_reorganize()
    mapper = reorg.IDMapper()  # no lesion_type → legacy path, no DB
    assert mapper.get_patient_id("A") == "sub-001"
    assert mapper.get_patient_id("B") == "sub-002"
    assert mapper.get_patient_id("A") == "sub-001"  # stable within the run
