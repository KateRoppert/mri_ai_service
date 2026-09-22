"""
Tests for dataset-scoped patient lookups (patient_registry.py).

Two people can now legitimately be called sub-001 — one per dataset. These
tests pin down that find_by_bids_id/find_by_bids_subject never merge them
when given the right dataset_ids, and never split one person's own sessions
across their datasets either.

Uses the shared, session-scoped test DB set up by backend/conftest.py — same
convention as test_patient_registry.py: test_-prefixed identifiers plus
explicit cleanup, since this DB is not reset between test files.
"""
from database import SessionLocal
from patient_registry import (
    find_by_bids_id,
    find_by_bids_subject,
    find_by_patient_id,
    register_patient,
)
from registry_models import PatientRegistry


def cleanup_test_data():
    db = SessionLocal()
    try:
        db.query(PatientRegistry).filter(
            PatientRegistry.study_hash.like("test_longsc_%")
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _register(study_hash, bids_id, original_id, dataset_id, lesion_type="multiple_sclerosis"):
    return register_patient(
        study_hash=study_hash,
        bids_id=bids_id,
        original_patient_id=original_id,
        lesion_type=lesion_type,
        kappa_dataset_id=dataset_id,
    )


def test_same_number_in_two_datasets_stays_two_people():
    cleanup_test_data()
    _register("test_longsc_1", "sub-001_ses-001", "test_longsc_P100", 158)
    _register("test_longsc_2", "sub-001_ses-001", "test_longsc_P200", 338)

    found = find_by_bids_subject("sub-001", dataset_ids={158})

    assert {r["original_patient_id"] for r in found} == {"test_longsc_P100"}
    cleanup_test_data()


def test_one_person_across_two_datasets_is_joined():
    """The scenario current -> 158 to 338 exists for: the same real person's
    older sessions in one dataset and newer ones in another must both surface
    once the caller widens to the account's datasets."""
    cleanup_test_data()
    _register("test_longsc_3", "sub-003_ses-001", "test_longsc_P100", 158)
    _register("test_longsc_4", "sub-001_ses-002", "test_longsc_P100", 338)

    found = find_by_bids_subject("sub-003", dataset_ids={158, 338})
    assert found  # the initial lookup succeeds within the narrower dataset

    original = found[0]["original_patient_id"]
    all_sessions = find_by_patient_id(original)
    sessions_in_account = {
        r["bids_id"] for r in all_sessions if r["kappa_dataset_id"] in {158, 338}
    }

    assert sessions_in_account == {"sub-003_ses-001", "sub-001_ses-002"}
    cleanup_test_data()


def test_no_filter_keeps_the_old_unscoped_behaviour():
    """Callers with no dataset context (dataset_ids=None) keep seeing
    everything, same as before this change — needed for code paths that only
    have a bare BIDS id."""
    cleanup_test_data()
    _register("test_longsc_5", "sub-001_ses-001", "test_longsc_P100", 158)
    _register("test_longsc_6", "sub-001_ses-001", "test_longsc_P200", 338)

    found = find_by_bids_subject("sub-001")

    assert len({r["original_patient_id"] for r in found
               if r["original_patient_id"] in {"test_longsc_P100", "test_longsc_P200"}}) == 2
    cleanup_test_data()


def test_record_with_no_dataset_is_always_included():
    """An un-uploaded run's record exists only on this machine — it has no
    dataset to disambiguate against, so it must never be filtered out."""
    cleanup_test_data()
    _register("test_longsc_7", "sub-009_ses-001", "test_longsc_P900", None)

    found = find_by_bids_subject("sub-009", dataset_ids={158, 338})

    assert {r["original_patient_id"] for r in found} == {"test_longsc_P900"}
    cleanup_test_data()


def test_find_by_bids_id_is_scoped_the_same_way():
    cleanup_test_data()
    _register("test_longsc_8", "sub-001_ses-001", "test_longsc_P100", 158)
    _register("test_longsc_9", "sub-001_ses-001", "test_longsc_P200", 338)

    found = find_by_bids_id("sub-001_ses-001", dataset_ids={338})

    assert {r["original_patient_id"] for r in found} == {"test_longsc_P200"}
    cleanup_test_data()


def test_resolve_longitudinal_records_widens_within_the_account(monkeypatch, tmp_path):
    """End-to-end through app._resolve_longitudinal_records: a run_id resolves
    to a dataset, the dataset resolves to its owner, and the lookup widens to
    every dataset of that owner — but no one else's."""
    import app
    import kappa_dataset_mapping as kdm
    from types import SimpleNamespace

    monkeypatch.setattr(kdm, "MAPPING_FILE", tmp_path / "kappa_datasets.yaml")
    kdm.set_dataset_id(26, "multiple_sclerosis", "1e2b93ad", 158)
    kdm.set_dataset_id(26, "multiple_sclerosis", "current", 338)

    cleanup_test_data()
    _register("test_longsc_10", "sub-003_ses-001", "test_longsc_P100", 158)
    _register("test_longsc_11", "sub-001_ses-002", "test_longsc_P100", 338)
    # A different account's dataset must never be pulled in even if reachable.
    _register("test_longsc_12", "sub-001_ses-001", "test_longsc_P999", 349)

    fake_run = SimpleNamespace(kappa_dataset_id=158)
    session = SimpleNamespace()
    monkeypatch.setattr(app, "get_pipeline_run", lambda db, run_id: fake_run)

    records = app._resolve_longitudinal_records(
        "sub-003", "multiple_sclerosis", "run-1", session)

    assert {r["bids_id"] for r in records} == {"sub-003_ses-001", "sub-001_ses-002"}
    assert all(r["original_patient_id"] == "test_longsc_P100" for r in records)
    cleanup_test_data()


def test_resolve_longitudinal_records_without_run_id_is_unscoped(monkeypatch, tmp_path):
    """No run_id -> no dataset context -> old behaviour, needed for links that
    predate this change."""
    import app

    cleanup_test_data()
    _register("test_longsc_13", "sub-001_ses-001", "test_longsc_P100", 158)
    _register("test_longsc_14", "sub-001_ses-001", "test_longsc_P200", 338)

    records = app._resolve_longitudinal_records(
        "sub-001", "multiple_sclerosis", None, None)

    found_people = {r["original_patient_id"] for r in records
                    if r["original_patient_id"] in {"test_longsc_P100", "test_longsc_P200"}}
    assert found_people == {"test_longsc_P100", "test_longsc_P200"}
    cleanup_test_data()
