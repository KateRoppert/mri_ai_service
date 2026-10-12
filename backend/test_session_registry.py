"""A run's sessions reach the registry when the run ends, not when Kappa does.

KI-059: `GET /api/longitudinal/{patient}` builds its points from
`patient_registry`, and those rows were written ONLY by
KappaUploader._register_patient. A run with no Kappa session, or one whose
upload failed, never got rows — so the doctor saw no dynamics at all, even
though every stage-08 report was sitting on disk.

The Kappa ids are an attribute of a session, not its reason to exist.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import session_registry
from patient_registry import find_by_bids_id, find_by_patient_id


def _run(output_path, dataset_id=None, run_id="run-reg"):
    return SimpleNamespace(
        run_id=run_id, output_path=str(output_path),
        lesion_type="multiple_sclerosis", kappa_dataset_id=dataset_id,
    )


def _session_on_disk(root, sub, ses, patient_id, study_uid):
    import json
    d = root / "preprocessed" / sub / ses / "anat"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{sub}_{ses}_t1.nii.gz").write_bytes(b"x")
    m = root / "metadata" / sub / ses / "anat" / "t1"
    m.mkdir(parents=True, exist_ok=True)
    (m / "0001.json").write_text(json.dumps({"identification": {
        "PatientID": patient_id, "StudyInstanceUID": study_uid,
        "StudyDate": "20260101", "PatientName": "Test",
    }}), encoding="utf-8")


def test_a_run_without_kappa_still_registers_its_sessions(tmp_path):
    """The whole point: no Kappa session, no upload, rows exist anyway."""
    _session_on_disk(tmp_path, "sub-401", "ses-001", "KI059A", "1.2.401")

    n = session_registry.register_run_sessions(_run(tmp_path))

    assert n == 1
    assert find_by_patient_id("KI059A"), "сессия не попала в реестр"


def test_every_session_of_a_run_is_registered(tmp_path):
    _session_on_disk(tmp_path, "sub-402", "ses-001", "KI059B", "1.2.402a")
    _session_on_disk(tmp_path, "sub-402", "ses-002", "KI059B", "1.2.402b")

    assert session_registry.register_run_sessions(_run(tmp_path)) == 2
    assert len(find_by_patient_id("KI059B")) == 2


def test_the_runs_dataset_is_recorded_so_scoped_lookups_find_it(tmp_path):
    """_resolve_longitudinal_records scopes a bids_id lookup by dataset, so a
    row left with NULL would be invisible to it. The dataset is fixed at run
    start by numbering.py — we know it before any upload."""
    _session_on_disk(tmp_path, "sub-403", "ses-001", "KI059C", "1.2.403")

    session_registry.register_run_sessions(_run(tmp_path, dataset_id=351))

    found = find_by_bids_id("sub-403_ses-001", {351})
    assert found, "строка не нашлась в скоупе своего датасета"
    assert found[0]["kappa_entity_id"] in (None, ""), "entity_id ещё не известен"


def test_the_upload_fills_the_entity_id_into_the_same_row(tmp_path):
    """register_patient upserts by study_hash, so the uploader must update
    the row we made, not add a second one — that would double the session in
    the longitudinal view."""
    from patient_registry import register_patient
    import session_identity

    _session_on_disk(tmp_path, "sub-404", "ses-001", "KI059D", "1.2.404")
    session_registry.register_run_sessions(_run(tmp_path, dataset_id=351))

    before = len(find_by_patient_id("KI059D"))

    files = session_identity.discover_preprocessed_sessions(tmp_path)["sub-404_ses-001"]
    register_patient(
        bids_id="sub-404_ses-001",
        study_hash=session_identity.study_hash(tmp_path, files),
        original_patient_id="KI059D",
        kappa_entity_id="entity-404",
        kappa_dataset_id=351,
    )

    rows = find_by_patient_id("KI059D")
    assert len(rows) == before, "выгрузка создала вторую строку вместо обновления"
    assert rows[0]["kappa_entity_id"] == "entity-404"


def test_a_session_without_metadata_is_skipped_not_guessed(tmp_path):
    """No study_hash means no stable identity, and register_patient keys on
    it. A row under an empty hash would collide with every other such row."""
    d = tmp_path / "preprocessed" / "sub-405" / "ses-001" / "anat"
    d.mkdir(parents=True, exist_ok=True)
    (d / "sub-405_ses-001_t1.nii.gz").write_bytes(b"x")

    assert session_registry.register_run_sessions(_run(tmp_path)) == 0


def test_a_run_with_no_output_registers_nothing(tmp_path):
    assert session_registry.register_run_sessions(_run(tmp_path)) == 0


def test_a_failure_on_one_session_does_not_lose_the_others(tmp_path):
    """Registration is bookkeeping around a run that already succeeded; one
    bad row must not cost the rest."""
    _session_on_disk(tmp_path, "sub-406", "ses-001", "KI059E", "1.2.406a")
    _session_on_disk(tmp_path, "sub-406", "ses-002", "KI059E", "1.2.406b")

    calls = {"n": 0}
    real = session_registry.register_patient

    def _flaky(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("db busy")
        return real(**kw)

    with patch.object(session_registry, "register_patient", _flaky):
        n = session_registry.register_run_sessions(_run(tmp_path))

    assert n == 1, "вторая сессия не зарегистрирована после сбоя первой"


# --- The wiring, not just the function -------------------------------------

@pytest.mark.asyncio
async def test_the_monitor_registers_on_completion(tmp_path):
    """Third time this project has shipped a function whose caller never
    invoked it, so the loop gets its own test."""
    from unittest.mock import AsyncMock
    import pipeline_monitor as pm
    from database import SessionLocal, create_pipeline_run

    db = SessionLocal()
    try:
        run = create_pipeline_run(db, input_path="/in", output_path=str(tmp_path))
        run.status = "completed"
        db.commit()

        with patch.object(pm.pipeline_monitor, "_send_update", new=AsyncMock()), \
             patch.object(pm.pipeline_monitor, "_start_queued_requeue",
                          new=AsyncMock()), \
             patch.object(pm.pipeline_monitor, "_register_sessions_locally") as reg:
            await pm.pipeline_monitor._monitor_loop(
                run.run_id, str(tmp_path), None, "multiple_sclerosis")

        assert reg.call_count == 1
        assert reg.call_args.args[0].run_id == run.run_id
    finally:
        db.close()


@pytest.mark.asyncio
async def test_a_registry_failure_does_not_stop_the_kappa_upload(tmp_path):
    """The upload is the more important of the two and runs after it."""
    from unittest.mock import AsyncMock
    import pipeline_monitor as pm
    from database import SessionLocal, create_pipeline_run

    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path=str(tmp_path),
            kappa_upload_status="pending",
        )
        run.status = "completed"
        db.commit()

        with patch.object(pm.pipeline_monitor, "_send_update", new=AsyncMock()), \
             patch.object(pm.pipeline_monitor, "_start_queued_requeue",
                          new=AsyncMock()), \
             patch("session_registry.register_run_sessions",
                   side_effect=RuntimeError("db busy")), \
             patch.object(pm.pipeline_monitor, "_record_delivery") as delivery:
            await pm.pipeline_monitor._monitor_loop(
                run.run_id, str(tmp_path), "sid", "multiple_sclerosis")

        # It got as far as deciding about delivery, which is past the registry.
        assert delivery.called
    finally:
        db.close()


def test_the_longitudinal_resolver_finds_a_run_that_never_saw_kappa(tmp_path):
    """The acceptance criterion of KI-059. The frontend passes a BIDS subject
    ("sub-407"), which resolves through find_by_bids_subject — scoped by
    dataset when the run has one. A CLI run has none, so the lookup is
    unscoped and the locally-written row is found."""
    import app
    from database import SessionLocal, create_pipeline_run

    _session_on_disk(tmp_path, "sub-407", "ses-001", "KI059F", "1.2.407a")
    _session_on_disk(tmp_path, "sub-407", "ses-002", "KI059F", "1.2.407b")

    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db, input_path="/in", output_path=str(tmp_path),
            lesion_type="multiple_sclerosis",
        )
        # No Kappa session: kappa_dataset_id stays NULL, nothing is uploaded.
        session_registry.register_run_sessions(run)

        records = app._resolve_longitudinal_records(
            "sub-407", "multiple_sclerosis", run.run_id, db)

        assert len(records) == 2, (
            "динамика не видит сессии прогона без Kappa — ровно то, "
            "что описывает KI-059"
        )
    finally:
        db.close()
