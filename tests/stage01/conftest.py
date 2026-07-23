"""
Shared fixtures for tests/stage01/ — writes minimal, readable-but-fake
DICOM files to disk so ModalityDetector.detect_modality() (which calls
pydicom.dcmread(..., stop_before_pixels=True, force=True)) can read them
without needing real clinical pixel data.
"""
from pathlib import Path
from typing import Optional

import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ImplicitVRLittleEndian, generate_uid


def write_fake_dicom_series(
    series_dir: Path,
    protocol_name: str = "",
    series_description: str = "",
    slice_thickness: Optional[float] = None,
    inversion_time: Optional[float] = None,
    contrast_bolus_agent: str = "",
    image_type: Optional[list] = None,
    n_files: int = 1,
) -> Path:
    """Write n_files minimal DICOM files into series_dir. Returns series_dir."""
    series_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPClassUID = generate_uid()
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.is_little_endian = True
        ds.is_implicit_VR = True
        ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
        ds.ProtocolName = protocol_name
        ds.SeriesDescription = series_description
        if slice_thickness is not None:
            ds.SliceThickness = slice_thickness
        if inversion_time is not None:
            ds.InversionTime = inversion_time
        if contrast_bolus_agent:
            ds.ContrastBolusAgent = contrast_bolus_agent
        if image_type is not None:
            ds.ImageType = image_type
        ds.save_as(str(series_dir / f"IM-{i:04d}.dcm"), write_like_original=False)
    return series_dir


@pytest.fixture
def make_dicom_series():
    return write_fake_dicom_series
