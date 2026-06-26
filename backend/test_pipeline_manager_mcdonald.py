import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def _write_report(seg_dir: Path, subject: str, session: str, data: dict):
    out_dir = seg_dir / subject / session / "anat" / "multiple_sclerosis"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{subject}_{session}_t1_mcdonald_report.json"
    report_path.write_text(json.dumps(data), encoding="utf-8")
    return report_path


def test_returns_none_when_segmentation_dir_missing(tmp_path):
    manager = PipelineManager()
    result = manager.get_mcdonald_reports(str(tmp_path / "does_not_exist"))
    assert result is None


def test_returns_none_when_no_mcdonald_reports_found(tmp_path):
    seg_dir = tmp_path / "segmentation"
    seg_dir.mkdir()
    manager = PipelineManager()
    result = manager.get_mcdonald_reports(str(tmp_path))
    assert result is None


def test_loads_all_mcdonald_reports(tmp_path):
    seg_dir = tmp_path / "segmentation"
    seg_dir.mkdir()
    report_data = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001",
        "session_id": "ses-001",
        "total_lesion_count": 2,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 1, "total_volume_cm3": 0.2},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular", "2": "juxtacortical"},
    }
    _write_report(seg_dir, "sub-001", "ses-001", report_data)

    manager = PipelineManager()
    result = manager.get_mcdonald_reports(str(tmp_path))

    assert result is not None
    assert len(result) == 1
    assert result[0]["total_lesion_count"] == 2
    assert result[0]["zones"]["spinal_cord"] == {"supported": False}


def test_mcdonald_response_model_validates_real_shape():
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    from models import McDonaldReportResponse

    sample = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001",
        "session_id": "ses-001",
        "total_lesion_count": 2,
        "zones": {
            "periventricular": {"lesion_count": 1, "total_volume_cm3": 0.5},
            "juxtacortical": {"lesion_count": 1, "total_volume_cm3": 0.2},
            "infratentorial": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "deep_white_matter": {"lesion_count": 0, "total_volume_cm3": 0.0},
            "spinal_cord": {"supported": False},
        },
        "lesion_zones_by_label": {"1": "periventricular", "2": "juxtacortical"},
    }

    parsed = McDonaldReportResponse(**sample)
    assert parsed.zones["periventricular"].lesion_count == 1
    assert parsed.zones["spinal_cord"].supported is False
