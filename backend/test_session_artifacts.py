"""Deleting one session's stage outputs so it gets rebuilt on the next run."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from session_artifacts import (
    STAGE_DIRS, delete_session_artifacts, purge_sessions_marked_for_reprocess,
)


def _make_run(root: Path, sessions=(("sub-001", "ses-001"), ("sub-002", "ses-001"))):
    """A run directory with per-stage output for each session, plus the
    bids_organized input that must survive."""
    for patient, session in sessions:
        for stage in STAGE_DIRS:
            d = root / stage / patient / session / "anat"
            d.mkdir(parents=True, exist_ok=True)
            (d / "file.nii.gz").write_bytes(b"x")
        bids = root / "bids_organized" / patient / session / "anat" / "t1"
        bids.mkdir(parents=True, exist_ok=True)
        (bids / "0001.dcm").write_bytes(b"x")
    return root


def test_removes_the_session_from_every_stage(tmp_path):
    _make_run(tmp_path)
    removed = delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    assert len(removed) == len(STAGE_DIRS)
    for stage in STAGE_DIRS:
        assert not (tmp_path / stage / "sub-001" / "ses-001").exists()


def test_keeps_the_pipeline_input(tmp_path):
    """bids_organized holds the corrected assignment — deleting it would
    throw away the very thing the doctor just fixed."""
    _make_run(tmp_path)
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    assert (tmp_path / "bids_organized" / "sub-001" / "ses-001").exists()


def test_leaves_other_patients_alone(tmp_path):
    _make_run(tmp_path)
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    for stage in STAGE_DIRS:
        assert (tmp_path / stage / "sub-002" / "ses-001").exists()


def test_removes_the_patient_directory_when_it_empties(tmp_path):
    _make_run(tmp_path, sessions=(("sub-001", "ses-001"),))
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    for stage in STAGE_DIRS:
        assert not (tmp_path / stage / "sub-001").exists()


def test_keeps_the_patient_directory_when_another_session_remains(tmp_path):
    _make_run(tmp_path, sessions=(("sub-001", "ses-001"), ("sub-001", "ses-002")))
    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    for stage in STAGE_DIRS:
        assert (tmp_path / stage / "sub-001" / "ses-002").exists()


@pytest.mark.parametrize("patient,session", [
    ("../..", "ses-001"),
    ("sub-001", "../../etc"),
    ("sub-001/../..", "ses-001"),
    ("", "ses-001"),
])
def test_refuses_ids_that_could_escape_the_run_directory(tmp_path, patient, session):
    """This is a delete driven by values read out of a JSON file. Escaping
    the run directory must be impossible by construction, not by luck."""
    _make_run(tmp_path)
    with pytest.raises(ValueError):
        delete_session_artifacts(str(tmp_path), patient, session)
    assert (tmp_path / "nifti" / "sub-001" / "ses-001").exists()


def test_purge_clears_the_flags_it_acted_on(tmp_path):
    _make_run(tmp_path)
    bids = tmp_path / "bids_organized"
    mapping = {
        "patients": {
            "sub-001": {"original_id": "P1", "sessions": {
                "ses-001": {"status": "complete", "needs_reprocess": True},
            }},
            "sub-002": {"original_id": "P2", "sessions": {
                "ses-001": {"status": "complete"},
            }},
        }
    }
    (bids / "dataset_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")

    purged = purge_sessions_marked_for_reprocess(str(tmp_path))

    assert list(purged) == ["sub-001/ses-001"]
    assert not (tmp_path / "nifti" / "sub-001" / "ses-001").exists()
    assert (tmp_path / "nifti" / "sub-002" / "ses-001").exists()

    after = json.loads((bids / "dataset_mapping.json").read_text(encoding="utf-8"))
    assert after["patients"]["sub-001"]["sessions"]["ses-001"]["needs_reprocess"] is False


def test_purge_is_a_no_op_without_a_mapping(tmp_path):
    assert purge_sessions_marked_for_reprocess(str(tmp_path)) == {}


def test_metadata_is_never_deleted(tmp_path):
    """metadata/ is written by stage 01 as it copies DICOM, not by a later
    stage — and stage 02, which could rebuild it, is disabled in this
    pipeline. Since bids_organized/ is deliberately kept, stage 01 skips the
    patient on a requeue and never rewrites it, so deleting metadata loses it
    for good.

    The loss is not cosmetic: _compute_study_hash reads PatientID and
    StudyInstanceUID from there. Without it the hash is None, every session
    already in the dataset comes back as name_clash, and nothing uploads.
    Seen for real on run fbc37b13: "0 из 2, номер занят другими данными".
    """
    assert "metadata" not in STAGE_DIRS

    _make_run(tmp_path)
    meta = tmp_path / "metadata" / "sub-001" / "ses-001" / "anat" / "t1"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "scan.json").write_text('{"identification": {}}', encoding="utf-8")

    delete_session_artifacts(str(tmp_path), "sub-001", "ses-001")

    assert (meta / "scan.json").exists(), "метаданные удалены — восстановить их нечем"
