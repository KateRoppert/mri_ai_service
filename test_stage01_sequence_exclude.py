"""Diffusion, DTI and perfusion are never anatomical-modality candidates.

UPENN-GBM exposed this. Stage 01 put `ep2d_diff_MDDW_IPAT` and
`ep2d_DTI_30dir` into the t2 slot for two of three patients and pushed the
real `T2 SAG SPACE` out as a dedup loser.

Two defects combined:

1. Neither name matches any t2 keyword ('t2', 'tse', 'fse', 't2w'), so
   detection falls through to `_detect_by_technical_params`, which calls
   anything with TR > 2000 and TE > 70 a t2. Diffusion EPI always is:
   measured on the real data, TR 4200-5800 and TE 83-91.
2. Every usable UPENN series is CaPTk-processed and tagged
   DERIVED/SECONDARY/OTHER, so the derived penalty (x0.1) put the genuine
   T2 at score 0.111 against the raw diffusion's 0.500.

This file covers (1), the root cause: a sequence that is not an anatomical
acquisition must be refused before any detection path, including the TR/TE
fallback. Fixing it also makes (2) harmless for the t2 slot, because the
genuine T2 is then the only candidate left.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
# Both: the stage imports its siblings (performance_monitor) by bare name.
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "reorg01", SCRIPTS_DIR / "01_reorganize_folders.py")
reorg = importlib.util.module_from_spec(_spec)
sys.modules["reorg01"] = reorg
_spec.loader.exec_module(reorg)

from utils.config_loader import load_series_scoring_config


@pytest.fixture
def detector():
    return reorg.ModalityDetector(
        logging.getLogger("test_sequence_exclude"),
        scoring_config=load_series_scoring_config(),
    )


def _write_series(directory: Path, description: str, tr=None, te=None, ti=None):
    """One minimal MR DICOM file, enough for detect_modality to read."""
    directory.mkdir(parents=True, exist_ok=True)
    ds = Dataset()
    ds.SeriesDescription = description
    ds.ProtocolName = description
    ds.Modality = "MR"
    ds.SOPClassUID = MRImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.PatientID = "TEST"
    if tr is not None:
        ds.RepetitionTime = tr
    if te is not None:
        ds.EchoTime = te
    if ti is not None:
        ds.InversionTime = ti
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    path = directory / "0001.dcm"
    pydicom.dcmwrite(path, ds, enforce_file_format=True)
    return directory


# --- The root cause: the TR/TE fallback must not be reachable for these ----

@pytest.mark.parametrize("description,tr,te", [
    ("ep2d_diff_MDDW_IPAT", 5800, 83),   # UPENN-GBM-00001, measured
    ("ep2d_diff_MDDW_IPAT", 4200, 83),   # UPENN-GBM-00002, measured
    ("ep2d_DTI_30dir", 5400, 91),        # UPENN-GBM-00003, measured
    ("DWI-EPI 2b (axi)", 4000, 90),      # clinical_dicom naming
    ("sADC DWI-EPI 2b (axi)", 4000, 90),
])
def test_diffusion_is_not_a_t2_candidate(detector, tmp_path, description, tr, te):
    """These satisfy TR > 2000 and TE > 70, which is the whole of the t2
    rule in _detect_by_technical_params. Without a hard refusal they become
    t2 candidates and can win the slot."""
    series = _write_series(tmp_path / "series", description, tr=tr, te=te)

    modality, _desc, _meta = detector.detect_modality(series)

    assert modality is None, (
        f"{description!r} распознано как {modality} — диффузия попала "
        f"в кандидаты на анатомическую модальность"
    )


@pytest.mark.parametrize("description,tr,te", [
    ("ep2d_perf 12 CC BOLUS", 2000, 45),
    ("ep2d_perf 3cc pre 12bolus", 2000, 45),
    ("perf_presto CLEAR", 2000, 45),
])
def test_perfusion_is_not_a_candidate(detector, tmp_path, description, tr, te):
    """Perfusion escaped by luck, not by design: TR is exactly 2000 and the
    rule is strictly greater. A scanner one millisecond over would have put
    a bolus series into the t2 slot."""
    series = _write_series(tmp_path / "series", description, tr=tr, te=te)

    modality, _desc, _meta = detector.detect_modality(series)

    assert modality is None


# --- And the genuine series must survive ----------------------------------

def test_the_real_upenn_t2_is_kept(detector, tmp_path):
    """The one that was being thrown away. Measured: TR 3200, TE 458."""
    series = _write_series(
        tmp_path / "series", "T2 SAG SPACE: Processed_CaPTk", tr=3200, te=458)

    modality, _desc, _meta = detector.detect_modality(series)

    assert modality == "t2"


@pytest.mark.parametrize("description", [
    # Real descriptions harvested from all five DICOM datasets in data/.
    # Substring matching has bitten this project before ('mpr' matched
    # inside primary 'mprage'), so the guard is against the keywords
    # catching anything anatomical.
    "t2_Flair_axial: Processed_CaPTk",
    "T2 SAG SPACE: Processed_CaPTk",
    "t1 axial: Processed_CaPTk",
    "t1 axial stealth-post : Processed_CaPTk",
    "t1_mprage_sag_p2_iso",
    "t2_tse_tra_p2",
    "T2W_TSE CLEAR",
    "T1W_3D_TFE",
    "FLAIR_longTR CLEAR",
    "t2_flair_sag_p2_iso",
])
def test_anatomical_series_are_untouched(detector, description):
    """No new keyword may match a real anatomical series."""
    combined = description.lower()
    assert detector._is_excluded_sequence(combined) is None, (
        f"{description!r} исключено как неанатомическая последовательность"
    )


def test_anatomy_exclusion_still_works(detector):
    """The existing hard filter is not disturbed by the new one."""
    assert detector._is_excluded_anatomy("c-spine t2 sag") is not None
    assert detector._is_excluded_anatomy("t2_tse_tra_p2") is None
