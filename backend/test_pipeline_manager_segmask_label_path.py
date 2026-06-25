import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline_manager import PipelineManager


def test_returns_none_when_directory_missing(tmp_path):
    manager = PipelineManager()
    result = manager.get_segmask_label_path(str(tmp_path), "sub-001", "ses-001")
    assert result is None


def test_returns_none_when_no_labels_file(tmp_path):
    seg_dir = tmp_path / "segmentation" / "sub-001" / "ses-001" / "anat" / "multiple_sclerosis"
    seg_dir.mkdir(parents=True)
    (seg_dir / "sub-001_ses-001_t1_segmask.nii.gz").write_bytes(b"")  # no _labels file
    manager = PipelineManager()
    result = manager.get_segmask_label_path(str(tmp_path), "sub-001", "ses-001")
    assert result is None


def test_finds_labels_file(tmp_path):
    seg_dir = tmp_path / "segmentation" / "sub-001" / "ses-001" / "anat" / "multiple_sclerosis"
    seg_dir.mkdir(parents=True)
    labels_path = seg_dir / "sub-001_ses-001_t1_segmask_labels.nii.gz"
    labels_path.write_bytes(b"")
    manager = PipelineManager()
    result = manager.get_segmask_label_path(str(tmp_path), "sub-001", "ses-001")
    assert result == labels_path
