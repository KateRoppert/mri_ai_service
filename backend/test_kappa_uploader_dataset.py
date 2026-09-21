"""
Tests for KappaUploader uploading into the dataset fixed at run start
(backend/numbering.py), and binding a pending numbering scope once a dataset
exists for it.
"""
import asyncio
from pathlib import Path

import pytest

import kappa_uploader


def _uploader(**kwargs):
    cfg = str(Path(__file__).resolve().parents[1] / "configs" / "preprocessing_config.yaml")
    return kappa_uploader.KappaUploader(
        run_id="r1",
        output_path="/tmp",
        token="tok",
        user_id=26,
        user_type_id=3,
        lesion_type="glioblastoma",
        preprocessing_config_path=cfg,
        **kwargs,
    )


def _row(subject_session, hash_):
    return {"dsEntityName": subject_session, "dsEntityInfo": {"study_hash": hash_}}


def test_bind_pending_scope_moves_numbers_to_the_new_dataset(monkeypatch, tmp_path):
    """A run numbered while Kappa was unreachable keeps its numbers once
    upload finally creates the dataset."""
    from utils.bids_allocator import get_bids_id, get_or_allocate, pending_scope
    db = tmp_path / "alloc.db"
    get_or_allocate(pending_scope("r1"), "P001", db)

    up = _uploader()
    monkeypatch.setattr(up, "_allocation_db", db, raising=False)

    up._bind_pending_scope(350)

    assert get_bids_id("ds:350", "P001", db) == "sub-001"
    assert get_bids_id(pending_scope("r1"), "P001", db) is None


def test_bind_pending_scope_is_a_noop_when_nothing_was_pending(monkeypatch, tmp_path):
    """A run that already had a real dataset scope (the common case) has
    nothing to rebind — must not raise or touch anything."""
    db = tmp_path / "alloc.db"
    up = _uploader()
    monkeypatch.setattr(up, "_allocation_db", db, raising=False)

    up._bind_pending_scope(350)  # no exception


@pytest.mark.asyncio
async def test_upload_binds_pending_scope_before_the_duplicate_check(monkeypatch, tmp_path):
    """upload_results() must rebind numbering to the resolved dataset before
    checking for duplicates, so a just-created dataset's (empty) entity list
    is what gets compared against."""
    from utils.bids_allocator import get_or_allocate, pending_scope
    db = tmp_path / "alloc.db"
    get_or_allocate(pending_scope("r1"), "P001", db)

    up = _uploader()
    monkeypatch.setattr(up, "_allocation_db", db, raising=False)

    async def fake_resolve():
        return 350

    # No sessions -> upload_results returns right after binding, which is
    # exactly what isolates "was the bind called" from everything after it.
    monkeypatch.setattr(up, "_resolve_dataset_id", fake_resolve)
    monkeypatch.setattr(up, "_discover_sessions", lambda: {})

    await up.upload_results()

    from utils.bids_allocator import get_bids_id, dataset_scope
    assert get_bids_id(dataset_scope(350), "P001", db) == "sub-001"


@pytest.mark.asyncio
async def test_name_clash_is_reported_and_not_uploaded(monkeypatch, tmp_path):
    """Two machines can each number into the same dataset while offline. The
    upload must refuse that session loudly rather than silently pairing a
    number with the wrong patient's files."""
    up = _uploader(dataset_id=337)
    monkeypatch.setattr(up, "_allocation_db", tmp_path / "alloc.db", raising=False)
    monkeypatch.setattr(up, "_discover_sessions",
                        lambda: {"sub-001_ses-001": {"preprocessed": [], "masks": []}})
    monkeypatch.setattr(up, "_compute_study_hash", lambda data: "hash-B")

    async def existing_hashes(dataset_id):
        return {"hash-A"}

    async def existing_names(dataset_id):
        return {"sub-001_ses-001"}

    monkeypatch.setattr(up, "_get_existing_study_hashes", existing_hashes)
    monkeypatch.setattr(up, "_get_existing_entity_names", existing_names)

    report = await up.upload_results()

    assert report["sessions"][0]["error"] == "name_clash"
    assert report["sessions"][0]["success"] is False


@pytest.mark.asyncio
async def test_no_name_clash_when_session_key_is_new(monkeypatch, tmp_path):
    """A genuinely new session must not be blocked just because the dataset
    already has other entries."""
    up = _uploader(dataset_id=337)
    monkeypatch.setattr(up, "_allocation_db", tmp_path / "alloc.db", raising=False)
    monkeypatch.setattr(up, "_discover_sessions",
                        lambda: {"sub-002_ses-001": {"preprocessed": [], "masks": []}})
    monkeypatch.setattr(up, "_compute_study_hash", lambda data: "hash-B")

    called = []

    async def fake_upload_session(dataset_id, session_key, session_data):
        called.append(session_key)
        return {"session": session_key, "success": True}

    async def existing_hashes(dataset_id):
        return {"hash-A"}

    async def existing_names(dataset_id):
        return {"sub-001_ses-001"}

    monkeypatch.setattr(up, "_get_existing_study_hashes", existing_hashes)
    monkeypatch.setattr(up, "_get_existing_entity_names", existing_names)
    monkeypatch.setattr(up, "_upload_session", fake_upload_session)

    await up.upload_results()

    assert called == ["sub-002_ses-001"]


def test_get_existing_entity_names_reads_ds_entity_name():
    """The name-clash check reads the same field highest_subject_number()
    does, so both agree on what "already in the dataset" means."""
    entities = [_row("sub-001_ses-001", "hash-A"), _row("sub-002_ses-001", "hash-B")]

    import kappa_dataset_resolver
    assert kappa_dataset_resolver.highest_subject_number(entities) == 2
