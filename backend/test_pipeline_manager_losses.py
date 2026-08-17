"""
KI-054: stages 01/03/04/05/06 already write their own
{stage}_incomplete_data.json into their output directories — nobody
reads them. This aggregates them into one flat "who got lost, where, why"
list. Confirmed real shapes on BO data:
  - stage 01: {"patient_id": "sub-133", "incomplete_sessions": [{"session_id": "ses-001", "missing": [...], "available": [...]}]}
  - stage 05: {"patient_id": "214", "incomplete_sessions": [{"session_id": "001", "reason": "modality_mismatch", "missing_in_output": [...]}]}
  - stage 06: {"patient_id": "214", "incomplete_sessions": [{"session_id": "001", "reason": "patient_missing_in_output"}]}
Stages 07/08 don't write this report at all yet (out of scope here).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def _write_loss_report(output_dir: Path, relative_path: str, content: dict):
    report_file = output_dir / relative_path
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(content), encoding="utf-8")


class TestGetPipelineLosses:
    def test_aggregates_losses_from_multiple_stages(self, tmp_path):
        _write_loss_report(tmp_path, "bids_organized/incomplete_data/01_reorganize_folders_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "sub-133", "incomplete_sessions": [
                    {"session_id": "ses-001", "missing": ["t1", "t1c"]}
                ]}
            ]
        })
        _write_loss_report(tmp_path, "segmentation/incomplete_data/segmentation_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "214", "incomplete_sessions": [
                    {"session_id": "001", "reason": "patient_missing_in_output"}
                ]}
            ]
        })

        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))

        assert len(losses) == 2
        by_stage = {loss["stage"]: loss for loss in losses}
        assert by_stage["01_reorganize"]["patient_id"] == "sub-133"
        assert by_stage["01_reorganize"]["session_id"] == "ses-001"
        assert "t1" in by_stage["01_reorganize"]["reason"]
        assert by_stage["06_segmentation"]["patient_id"] == "214"
        assert by_stage["06_segmentation"]["reason"] == "patient_missing_in_output"

    def test_missing_report_files_are_skipped_not_errors(self, tmp_path):
        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))
        assert losses == []

    def test_reason_synthesized_from_reason_and_missing_in_output(self, tmp_path):
        _write_loss_report(tmp_path, "preprocessed/incomplete_data/preprocessing_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "214", "incomplete_sessions": [
                    {"session_id": "001", "reason": "modality_mismatch", "missing_in_output": ["t1c"]}
                ]}
            ]
        })
        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))

        assert len(losses) == 1
        assert "modality_mismatch" in losses[0]["reason"]
        assert "t1c" in losses[0]["reason"]

    def test_reason_synthesized_from_missing_only_when_no_reason_field(self, tmp_path):
        _write_loss_report(tmp_path, "bids_organized/incomplete_data/01_reorganize_folders_incomplete_data.json", {
            "incomplete_data": [
                {"patient_id": "sub-050", "incomplete_sessions": [
                    {"session_id": "ses-001", "missing": ["t2fl"], "available": ["t1", "t1c", "t2"]}
                ]}
            ]
        })
        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))

        assert len(losses) == 1
        assert "t2fl" in losses[0]["reason"]

    def test_malformed_json_report_is_skipped_not_fatal(self, tmp_path):
        report_file = tmp_path / "bids_organized" / "incomplete_data" / "01_reorganize_folders_incomplete_data.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text("{not valid json", encoding="utf-8")

        pm = PipelineManager()
        losses = pm.get_pipeline_losses(str(tmp_path))
        assert losses == []
