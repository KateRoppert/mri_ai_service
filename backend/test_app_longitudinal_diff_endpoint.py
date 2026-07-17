import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app, pipeline_manager

client = TestClient(app)


def _fake_record(bids_id, scan_date, run_id):
    return {"bids_id": bids_id, "scan_date": scan_date, "pipeline_run_id": run_id, "lesion_type": "multiple_sclerosis"}


def _fake_run(output_path="/fake/output"):
    return SimpleNamespace(output_path=output_path)


def test_404_when_no_sessions_found():
    with patch("app.find_by_patient_id", return_value=[]), \
         patch("app.find_by_bids_id", return_value=[]), \
         patch("app.find_by_bids_subject", return_value=[]):
        response = client.get("/api/longitudinal/P999/diff")
    assert response.status_code == 404


def test_404_when_only_one_session():
    records = [_fake_record("sub-001_ses-001", "2022-01-01", "run-1")]
    with patch("app.find_by_patient_id", return_value=records), \
         patch("app.get_pipeline_run", return_value=_fake_run()):
        response = client.get("/api/longitudinal/P000915/diff")
    assert response.status_code == 404


def test_200_with_two_sessions_computes_one_pair(tmp_path):
    records = [
        _fake_record("sub-001_ses-001", "2022-01-18", "run-1"),
        _fake_record("sub-001_ses-002", "2023-03-25", "run-2"),
    ]
    fake_diff_result = {
        "new_count": 1, "growing_count": 0, "stable_count": 0, "resolved_count": 0,
        "lesions": [{"label": 1, "status": "new", "volume_cm3": 0.5, "previous_volume_cm3": 0.0}],
    }
    with patch("app.find_by_patient_id", return_value=records), \
         patch("app.get_pipeline_run", return_value=_fake_run()), \
         patch.object(pipeline_manager, "get_segmask_label_path", return_value=Path("/fake/labels.nii.gz")), \
         patch("app.cached_compare_labeled_masks", return_value=fake_diff_result):
        response = client.get("/api/longitudinal/sub-001/diff?lesion_type=multiple_sclerosis")

    assert response.status_code == 200
    body = response.json()
    assert len(body["pairs"]) == 1
    assert body["pairs"][0]["from_session_id"] == "ses-001"
    assert body["pairs"][0]["to_session_id"] == "ses-002"
    assert body["pairs"][0]["new_count"] == 1


def test_pair_skipped_when_label_file_missing():
    records = [
        _fake_record("sub-001_ses-001", "2022-01-18", "run-1"),
        _fake_record("sub-001_ses-002", "2023-03-25", "run-2"),
    ]
    with patch("app.find_by_patient_id", return_value=records), \
         patch("app.get_pipeline_run", return_value=_fake_run()), \
         patch.object(pipeline_manager, "get_segmask_label_path", return_value=None):
        response = client.get("/api/longitudinal/sub-001/diff?lesion_type=multiple_sclerosis")

    assert response.status_code == 200
    assert response.json()["pairs"] == []
