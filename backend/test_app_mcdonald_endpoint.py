import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))

from fastapi.testclient import TestClient
from app import app, pipeline_manager


client = TestClient(app)


def _fake_run(current_stage=8, output_path="/fake/output"):
    return SimpleNamespace(current_stage=current_stage, output_path=output_path)


def test_404_when_run_not_found():
    with patch("app.get_pipeline_run", return_value=None):
        response = client.get("/api/mcdonald-reports/nonexistent-run")
    assert response.status_code == 404


def test_400_when_stage_not_completed():
    with patch("app.get_pipeline_run", return_value=_fake_run(current_stage=3)):
        response = client.get("/api/mcdonald-reports/some-run")
    assert response.status_code == 400


def test_404_when_no_reports_found():
    with patch("app.get_pipeline_run", return_value=_fake_run()):
        with patch.object(pipeline_manager, "get_mcdonald_reports", return_value=None):
            response = client.get("/api/mcdonald-reports/some-run")
    assert response.status_code == 404


def test_200_with_reports():
    sample_report = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001",
        "session_id": "ses-001",
        "total_lesion_count": 1,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular"},
    }
    with patch("app.get_pipeline_run", return_value=_fake_run()):
        with patch.object(pipeline_manager, "get_mcdonald_reports", return_value=[sample_report]):
            response = client.get("/api/mcdonald-reports/some-run")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    assert body["reports"][0]["zones"]["spinal_cord"]["supported"] is False
