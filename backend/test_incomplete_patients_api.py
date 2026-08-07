import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def _write_mapping(output_dir: Path, patients: dict):
    bids_dir = output_dir / "bids_organized"
    bids_dir.mkdir(parents=True, exist_ok=True)
    mapping = {"patients": patients, "output_dir": str(bids_dir), "created_at": "x", "updated_at": "x"}
    (bids_dir / "dataset_mapping.json").write_text(json.dumps(mapping), encoding="utf-8")


class TestGetIncompletePatients:
    def test_returns_only_incomplete_non_discarded_sessions(self, tmp_path):
        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "complete",
                        "series": {"t1": {}, "t1c": {}, "t2": {}, "t2fl": {}},
                        "unrecognized_series": [],
                    },
                    "ses-002": {
                        "original_date": "20230115",
                        "status": "incomplete",
                        "series": {"t1c": {}},
                        "unrecognized_series": [
                            {"original_path": "/raw/weird", "series_description": "xyz", "slice_count": 20}
                        ],
                    },
                },
            },
            "sub-002": {
                "original_id": "P2",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "discarded",
                        "series": {"t1c": {}},
                        "unrecognized_series": [],
                    },
                },
            },
        })
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))

        assert len(result) == 1
        assert result[0]["patient_id"] == "sub-001"
        assert result[0]["session_id"] == "ses-002"
        assert result[0]["status"] == "incomplete"
        assert result[0]["available"] == ["t1c"]
        assert len(result[0]["unrecognized_series"]) == 1

    def test_missing_mapping_file_returns_empty_list(self, tmp_path):
        pm = PipelineManager()
        result = pm.get_incomplete_patients(str(tmp_path))
        assert result == []


import shutil


class TestRelabelSeries:
    def _make_dicom_series(self, series_dir: Path, n_files=2):
        series_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_files):
            (series_dir / f"IM-{i:04d}.dcm").write_bytes(b"fake dicom bytes")
        return series_dir

    def test_relabel_completes_session_and_moves_to_main_tree(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-001" / "ses-001"
        raw_series = self._make_dicom_series(tmp_path / "raw" / "weird_series")

        _write_mapping(tmp_path, {
            "sub-001": {
                "original_id": "P1",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}, "t2": {}, "t2fl": {}},
                        "unrecognized_series": [
                            {
                                "original_path": str(raw_series),
                                "series_description": "xyz | xyz",
                                "slice_count": 2,
                            }
                        ],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        pm = PipelineManager()
        result = pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-001",
            session_id="ses-001",
            original_path=str(raw_series),
            modality="t1c",
        )

        assert result["status"] == "complete"
        # session moved out of _incomplete/
        assert not incomplete_dir.exists()
        assert (bids_dir / "sub-001" / "ses-001" / "anat" / "t1c").exists()
        copied_files = list((bids_dir / "sub-001" / "ses-001" / "anat" / "t1c").glob("*.dcm"))
        assert len(copied_files) == 2

        # dataset_mapping.json updated on disk
        mapping = json.loads((bids_dir / "dataset_mapping.json").read_text())
        session = mapping["patients"]["sub-001"]["sessions"]["ses-001"]
        assert session["status"] == "complete"
        assert "t1c" in session["series"]
        assert session["unrecognized_series"] == []

    def test_relabel_leaves_session_incomplete_if_still_missing_modalities(self, tmp_path):
        bids_dir = tmp_path / "bids_organized"
        incomplete_dir = bids_dir / "_incomplete" / "sub-002" / "ses-001"
        raw_series = self._make_dicom_series(tmp_path / "raw" / "weird_series_2")

        _write_mapping(tmp_path, {
            "sub-002": {
                "original_id": "P2",
                "sessions": {
                    "ses-001": {
                        "original_date": "20230101",
                        "status": "incomplete",
                        "series": {"t1": {}},
                        "unrecognized_series": [
                            {"original_path": str(raw_series), "series_description": "xyz", "slice_count": 2}
                        ],
                    },
                },
            },
        })
        incomplete_dir.mkdir(parents=True, exist_ok=True)

        pm = PipelineManager()
        result = pm.relabel_series(
            output_path=str(tmp_path),
            patient_id="sub-002",
            session_id="ses-001",
            original_path=str(raw_series),
            modality="t2",
        )

        assert result["status"] == "incomplete"
        # still under _incomplete/ — not yet complete
        assert (bids_dir / "_incomplete" / "sub-002" / "ses-001" / "anat" / "t2").exists()
        assert not (bids_dir / "sub-002").exists()
