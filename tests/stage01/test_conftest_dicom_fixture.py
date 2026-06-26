import pydicom


def test_writes_readable_dicom_with_expected_tags(make_dicom_series, tmp_path):
    series_dir = make_dicom_series(
        tmp_path / "series1",
        protocol_name="T1-TFE (3D brain)",
        series_description="T1-TFE (3D brain)",
        slice_thickness=1.1,
        inversion_time=1660.0,
        contrast_bolus_agent="Gadovist",
        n_files=3,
    )
    files = sorted(series_dir.glob("*.dcm"))
    assert len(files) == 3

    dcm = pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    assert dcm.ProtocolName == "T1-TFE (3D brain)"
    assert float(dcm.SliceThickness) == 1.1
    assert float(dcm.InversionTime) == 1660.0
    assert str(dcm.ContrastBolusAgent) == "Gadovist"


def test_defaults_omit_optional_tags(make_dicom_series, tmp_path):
    series_dir = make_dicom_series(
        tmp_path / "series2",
        protocol_name="T2-TSE (axi brain)",
        series_description="T2-TSE (axi brain)",
    )
    dcm = pydicom.dcmread(str(next(series_dir.glob("*.dcm"))), stop_before_pixels=True, force=True)
    assert dcm.get((0x0018, 0x0050)) is None
    assert dcm.get((0x0018, 0x0082)) is None
