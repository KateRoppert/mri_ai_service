import pytest
from utils.bids_allocator import (
    dataset_scope, local_scope, pending_scope,
    get_or_allocate, get_bids_id, get_original_id, get_allocations,
    set_floor, rebind_scope,
)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "alloc.db"


def test_each_scope_starts_at_one(db):
    """The whole point: a fresh dataset numbers from sub-001, even for a
    patient who already has a number somewhere else."""
    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-001"
    assert get_or_allocate(dataset_scope(349), "P001", db) == "sub-001"


def test_number_is_stable_within_a_scope(db):
    first = get_or_allocate(dataset_scope(337), "P001", db)
    get_or_allocate(dataset_scope(337), "P002", db)

    assert get_or_allocate(dataset_scope(337), "P001", db) == first


def test_numbers_increment_within_a_scope(db):
    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-001"
    assert get_or_allocate(dataset_scope(337), "P002", db) == "sub-002"


def test_floor_from_an_existing_dataset(db):
    """A dataset already holding sub-001..sub-007 must continue at sub-008,
    even though this machine's table knows nothing about those patients."""
    set_floor(dataset_scope(337), 7, db)

    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-008"


def test_floor_never_lowers(db):
    set_floor(dataset_scope(337), 7, db)
    set_floor(dataset_scope(337), 3, db)

    assert get_or_allocate(dataset_scope(337), "P001", db) == "sub-008"


def test_floor_below_allocated_numbers_is_harmless(db):
    get_or_allocate(dataset_scope(337), "P001", db)
    get_or_allocate(dataset_scope(337), "P002", db)
    set_floor(dataset_scope(337), 1, db)

    assert get_or_allocate(dataset_scope(337), "P003", db) == "sub-003"


def test_scopes_do_not_leak(db):
    get_or_allocate(dataset_scope(337), "P001", db)

    assert get_bids_id(dataset_scope(349), "P001", db) is None
    assert get_allocations(dataset_scope(349), db) == {}


def test_reverse_lookup_is_scoped(db):
    get_or_allocate(dataset_scope(337), "P001", db)
    get_or_allocate(dataset_scope(349), "P002", db)

    assert get_original_id(dataset_scope(337), "sub-001", db) == "P001"
    assert get_original_id(dataset_scope(349), "sub-001", db) == "P002"


def test_rebind_moves_a_pending_scope(db):
    """A run numbered while Kappa was unreachable keeps its numbers when the
    dataset finally gets created."""
    get_or_allocate(pending_scope("run-1"), "P001", db)
    get_or_allocate(pending_scope("run-1"), "P002", db)

    moved = rebind_scope(pending_scope("run-1"), dataset_scope(350), db)

    assert moved == 2
    assert get_bids_id(dataset_scope(350), "P001", db) == "sub-001"
    assert get_allocations(pending_scope("run-1"), db) == {}


def test_local_scope_is_separate_from_datasets(db):
    get_or_allocate(local_scope("glioblastoma"), "P001", db)

    assert get_bids_id(dataset_scope(337), "P001", db) is None


def test_concurrent_allocation_does_not_reuse_a_number(db):
    """Two Stage 01 processes starting at once must not take the same number.
    BEGIN IMMEDIATE serialises them."""
    import threading

    results = []
    def worker(pid):
        results.append(get_or_allocate(dataset_scope(337), pid, db))

    threads = [threading.Thread(target=worker, args=(f"P{i:03d}",)) for i in range(8)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert len(set(results)) == 8


def test_rebind_leaves_colliding_numbers_behind(tmp_path):
    """A pending scope starts at sub-001, so binding it into a dataset that
    already holds sub-001 collides. A blanket UPDATE raised UNIQUE and took
    the whole upload down with a 500 — the worst answer, because the
    collision is precisely what the operator needs told."""
    db = tmp_path / "alloc.db"

    # Датасет уже занял sub-001 другим пациентом.
    assert get_or_allocate("ds:351", "KA01", db_path=db) == "sub-001"
    # Офлайн-прогон независимо выдал sub-001 своему.
    assert get_or_allocate("pending:run-x", "KA18", db_path=db) == "sub-001"

    moved = rebind_scope("pending:run-x", "ds:351", db_path=db)

    assert moved == 0
    # Конфликтующая запись осталась на месте — её разберёт проверка name_clash.
    assert get_bids_id("pending:run-x", "KA18", db_path=db) == "sub-001"
    # И чужой номер не перезаписан.
    assert get_bids_id("ds:351", "KA01", db_path=db) == "sub-001"


def test_rebind_still_moves_what_does_not_collide(tmp_path):
    db = tmp_path / "alloc.db"
    assert get_or_allocate("ds:351", "KA01", db_path=db) == "sub-001"
    assert get_or_allocate("pending:run-y", "KA18", db_path=db) == "sub-001"
    assert get_or_allocate("pending:run-y", "KA19", db_path=db) == "sub-002"

    moved = rebind_scope("pending:run-y", "ds:351", db_path=db)

    assert moved == 1
    assert get_bids_id("ds:351", "KA19", db_path=db) == "sub-002"
    assert get_bids_id("pending:run-y", "KA18", db_path=db) == "sub-001"
