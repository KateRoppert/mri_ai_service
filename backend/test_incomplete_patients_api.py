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
