"""
Tests for backend/numbering.py: deciding the BIDS numbering scope for a run.

The scope has to be fixed before Stage 01 issues any subject id, so this
resolves (or creates) the Kappa dataset up front rather than at upload —
see docs/superpowers/specs/2026-09-21-bids-numbering-per-dataset-design.md.
"""
import pytest

import numbering


@pytest.mark.asyncio
async def test_no_kappa_session_means_local_scope():
    """A CLI-style run with no Kappa behind it keeps today's numbering."""
    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id=None)

    assert scope == "local:glioblastoma"
    assert dataset_id is None
    assert warning is None


@pytest.mark.asyncio
async def test_unknown_session_means_local_scope(monkeypatch):
    """An expired or unknown session_id must not crash the start."""
    monkeypatch.setattr(numbering, "get_session", lambda sid: None)

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="stale")

    assert scope == "local:glioblastoma"
    assert dataset_id is None
    assert warning is None


@pytest.mark.asyncio
async def test_mapped_dataset_becomes_the_scope(monkeypatch):
    monkeypatch.setattr(numbering, "get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26, "user_type_id": 3})
    monkeypatch.setattr(numbering, "compute_preprocessing_id", lambda path: "1099b9cd")
    async def fake_resolve(**kwargs):
        return 337
    async def fake_floor(**kwargs):
        return 7
    recorded = {}
    monkeypatch.setattr(numbering, "resolve_or_create", fake_resolve)
    monkeypatch.setattr(numbering, "dataset_floor", fake_floor)
    monkeypatch.setattr(numbering, "set_floor",
                        lambda scope, floor: recorded.update(scope=scope, floor=floor))

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="s1")

    assert (scope, dataset_id, warning) == ("ds:337", 337, None)
    assert recorded == {"scope": "ds:337", "floor": 7}


@pytest.mark.asyncio
async def test_unreachable_kappa_with_no_dataset_falls_back_to_pending(monkeypatch):
    """The run must still start. Numbers are issued in a temporary scope and
    bound to a dataset at upload time."""
    monkeypatch.setattr(numbering, "get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 52, "user_type_id": 4})
    monkeypatch.setattr(numbering, "compute_preprocessing_id", lambda path: "1099b9cd")
    async def boom(**kwargs):
        raise OSError("connection refused")
    monkeypatch.setattr(numbering, "resolve_or_create", boom)

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="s1")

    assert scope == "pending:r1"
    assert dataset_id is None
    assert "Kappa" in warning


@pytest.mark.asyncio
async def test_no_dataset_and_kappa_reachable_falls_back_to_pending(monkeypatch):
    """resolve_or_create(create=False semantics aside) returning None (e.g. the
    mapping truly has nothing and creation itself failed) is the same corner."""
    monkeypatch.setattr(numbering, "get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 52, "user_type_id": 4})
    monkeypatch.setattr(numbering, "compute_preprocessing_id", lambda path: "1099b9cd")
    async def fake_resolve(**kwargs):
        return None
    monkeypatch.setattr(numbering, "resolve_or_create", fake_resolve)

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="s1")

    assert scope == "pending:r1"
    assert dataset_id is None
    assert warning is not None


@pytest.mark.asyncio
async def test_unknown_floor_does_not_reset_numbering(monkeypatch):
    """Kappa unreachable while the dataset IS known: keep the dataset scope and
    leave the floor alone. Writing 0 would restart a populated dataset at 1."""
    monkeypatch.setattr(numbering, "get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26, "user_type_id": 3})
    monkeypatch.setattr(numbering, "compute_preprocessing_id", lambda path: "1099b9cd")
    async def fake_resolve(**kwargs):
        return 337
    async def no_floor(**kwargs):
        return None
    calls = []
    monkeypatch.setattr(numbering, "resolve_or_create", fake_resolve)
    monkeypatch.setattr(numbering, "dataset_floor", no_floor)
    monkeypatch.setattr(numbering, "set_floor", lambda *a: calls.append(a))

    scope, dataset_id, warning = await numbering.scope_for_run(
        run_id="r1", lesion_type="glioblastoma", kappa_session_id="s1")

    assert (scope, dataset_id) == ("ds:337", 337)
    assert calls == []


def test_scope_from_dataset_id_uses_the_dataset():
    """Resume/requeue path: no Kappa call, just derive the scope from the
    parent run's already-fixed dataset."""
    assert numbering.scope_from_dataset_id(337, "glioblastoma") == "ds:337"


def test_scope_from_dataset_id_falls_back_to_local_without_one():
    """A parent run that never got a dataset (e.g. started before this
    existed, or itself never resolved one) keeps local numbering."""
    assert numbering.scope_from_dataset_id(None, "glioblastoma") == "local:glioblastoma"
