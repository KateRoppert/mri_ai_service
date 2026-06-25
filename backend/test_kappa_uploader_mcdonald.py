import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from kappa_uploader import KappaUploader

PROJECT_ROOT = Path(__file__).parent.parent
REAL_PREPROCESSING_CONFIG = PROJECT_ROOT / "configs" / "preprocessing_config.yaml"


def _make_uploader(output_path: Path) -> KappaUploader:
    return KappaUploader(
        run_id="test-run",
        output_path=str(output_path),
        token="dummy-token",
        user_id=1,
        user_type_id=1,
        lesion_type="multiple_sclerosis",
        preprocessing_config_path=str(REAL_PREPROCESSING_CONFIG),
    )


def _build_session_dirs(output_path: Path, subject: str, session: str):
    preproc_dir = output_path / "preprocessed" / subject / session / "anat"
    preproc_dir.mkdir(parents=True, exist_ok=True)
    (preproc_dir / f"{subject}_{session}_t1.nii.gz").write_bytes(b"")

    seg_dir = output_path / "segmentation" / subject / session / "anat" / "multiple_sclerosis"
    seg_dir.mkdir(parents=True, exist_ok=True)
    (seg_dir / f"{subject}_{session}_t1_segmask.nii.gz").write_bytes(b"")
    return seg_dir


def test_discover_sessions_finds_mcdonald_report(tmp_path):
    seg_dir = _build_session_dirs(tmp_path, "sub-001", "ses-001")
    mcdonald_data = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001", "session_id": "ses-001",
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
    report_path = seg_dir / "sub-001_ses-001_t1_mcdonald_report.json"
    report_path.write_text(json.dumps(mcdonald_data), encoding="utf-8")

    uploader = _make_uploader(tmp_path)
    sessions = uploader._discover_sessions()

    assert "sub-001_ses-001" in sessions
    assert sessions["sub-001_ses-001"]["mcdonald_report"] == report_path


def test_build_entity_info_includes_mcdonald_report(tmp_path):
    seg_dir = _build_session_dirs(tmp_path, "sub-001", "ses-001")
    mcdonald_data = {
        "mask_file": "sub-001_ses-001_t1_segmask.nii.gz",
        "patient_id": "sub-001", "session_id": "ses-001",
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
    (seg_dir / "sub-001_ses-001_t1_mcdonald_report.json").write_text(
        json.dumps(mcdonald_data), encoding="utf-8"
    )

    uploader = _make_uploader(tmp_path)
    sessions = uploader._discover_sessions()
    info = uploader._build_entity_info("sub-001_ses-001", sessions["sub-001_ses-001"])

    assert info["mcdonald_report"]["total_lesion_count"] == 1
    assert info["mcdonald_report"]["zones"]["spinal_cord"] == {"supported": False}
