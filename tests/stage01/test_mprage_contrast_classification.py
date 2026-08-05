"""
Bug: contrast-enhanced MPRAGE (and t1_mpr_*) series were classified as plain
t1 instead of t1c, because MODALITY_PATTERNS['t1c']['exclude'] contained the
text token 'mpr' — intended to reject derived MPR reconstructions, but 'mpr'
is also a substring of 'mprage' and of primary 't1_mpr_*' acquisitions.

Real-world impact (data/dropbox_33): KA01, KA02, KA05, KA07, KA10, KA13,
KA101, KA102, KA105, KA106, KA107, KA12 (12 patients, contrast-enhanced
MPRAGE) and KA06, KA08 (2 patients, t1_mpr_* with contrast) all lost t1c —
both their T1 series collapsed to plain "t1", the deduplicator kept one by
tie-break, and the contrast series was silently discarded. Since t1c is a
required modality for glioblastoma segmentation, all 14 patients were
skipped at the segmentation stage.

Fix: the correct discriminator between a primary acquisition and a derived
reconstruction is the DICOM ImageType tag (0008,0008), not the series name.
'mpr' is removed from the text exclude lists; an ImageType-based check
replaces it.

Correction (data/clinical_dicom/BO): the original version of this check also
treated 'SECONDARY' appearing anywhere in ImageType as a reconstruction
marker, per the textbook DICOM convention (value[1] = PRIMARY or SECONDARY).
That was never backed by a real counter-example — the one confirmed
reconstruction at the time (117-152's MPR series) already had DERIVED as
value[0]. Real data proved the theory wrong: BO's scanner/site tags
ORIGINAL/SECONDARY on essentially every series it produces, including plain
primary T1/T2/FLAIR acquisitions — the SECONDARY-anywhere check was
silently excluding real primary diagnostic series (dark-fluid FLAIR among
them) across the whole dataset. DERIVED as value[0] remains the only
validated signal; SECONDARY alone is no longer treated as exclusion.
"""
import sys
import importlib.util
import logging
from pathlib import Path

from pydicom.dataset import Dataset

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_mprage")
ModalityDetector = reorganize_mod.ModalityDetector


def _dataset_with_image_type(image_type):
    ds = Dataset()
    if image_type is not None:
        ds.ImageType = image_type
    return ds


class TestIsDerivedReconstruction:
    def test_original_primary_is_not_derived(self):
        dcm = _dataset_with_image_type(['ORIGINAL', 'PRIMARY', 'M', 'NORM'])
        assert ModalityDetector._is_derived_reconstruction(dcm) is False

    def test_derived_first_value_is_derived(self):
        dcm = _dataset_with_image_type(['DERIVED', 'PRIMARY', 'M', 'NONE'])
        assert ModalityDetector._is_derived_reconstruction(dcm) is True

    def test_secondary_alone_without_derived_is_not_excluded(self):
        """Regression for data/clinical_dicom/BO: real primary acquisitions
        there are tagged ORIGINAL/SECONDARY — SECONDARY alone must not be
        treated as a reconstruction marker (see module docstring)."""
        dcm = _dataset_with_image_type(['ORIGINAL', 'SECONDARY'])
        assert ModalityDetector._is_derived_reconstruction(dcm) is False

    def test_derived_with_secondary_is_still_derived(self):
        """DERIVED as value[0] remains sufficient on its own, whether or not
        SECONDARY also appears."""
        dcm = _dataset_with_image_type(['DERIVED', 'SECONDARY', 'MPR'])
        assert ModalityDetector._is_derived_reconstruction(dcm) is True

    def test_missing_image_type_fails_open_not_derived(self):
        dcm = _dataset_with_image_type(None)
        assert ModalityDetector._is_derived_reconstruction(dcm) is False


class TestMprageContrastClassification:
    def test_contrast_mprage_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Regression for KA01/KA07: t1_mprage + ContrastBolusAgent='anonymized'."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_mprage_fs_sag_1mm_iso",
            series_description="t1_mprage_fs_sag_1mm_iso",
            contrast_bolus_agent="anonymized",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"
        assert tech_meta["has_contrast"] is True

    def test_km_marker_mprage_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Regression for KA12: t1_mprage_sag_1mm_KM (KM marker, no explicit agent)."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_mprage_sag_1mm_KM",
            series_description="t1_mprage_sag_1mm_KM",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"

    def test_primary_t1_mpr_with_km_classified_as_t1c(self, make_dicom_series, tmp_path):
        """Regression for KA06: t1_mpr_sag_KM, a primary (non-MPRAGE) acquisition."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_mpr_sag_KM",
            series_description="t1_mpr_sag_KM",
            contrast_bolus_agent="anonymized",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM', 'SH', 'FIL'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"

    def test_derived_mpr_reconstruction_is_classified_but_flagged(self, make_dicom_series, tmp_path):
        """A DERIVED MPR reconstruction of a contrast T1 is still classified
        normally (t1c) — it's real, usable data — but flagged so scoring can
        deprioritize it against a primary candidate (see score_series)."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_mprage_sag_1mm_iso_MPR_MPR cor",
            series_description="t1_mprage_sag_1mm_iso_MPR_MPR cor",
            contrast_bolus_agent="anonymized",
            image_type=['DERIVED', 'PRIMARY', 'M', 'NONE'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1c"
        assert tech_meta["is_derived_reconstruction"] is True

    def test_real_bo_dark_fluid_flair_with_secondary_imagetype_classified_as_t2fl(self, make_dicom_series, tmp_path):
        """Regression for data/clinical_dicom/BO: real primary FLAIR series
        there carry ORIGINAL/SECONDARY — must not be excluded."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_space_dark-fluid_sag_fs_iso",
            series_description="t2_space_dark-fluid_sag_fs_iso",
            image_type=['ORIGINAL', 'SECONDARY', 'M', 'ND', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t2fl"

    def test_real_bo_mpr_reconstruction_classified_and_flagged(self, make_dicom_series, tmp_path):
        """A genuine BO MPR reconstruction (DERIVED, also carries SECONDARY)
        is classified normally (t2) and flagged as derived — DERIVED alone
        remains sufficient to detect it."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_space_sag_p4_0.9 fast_MPR_Tra",
            series_description="t2_space_sag_p4_0.9 fast_MPR_Tra",
            image_type=['DERIVED', 'SECONDARY', 'MPR', 'ND', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t2"
        assert tech_meta["is_derived_reconstruction"] is True

    def test_plain_mprage_without_contrast_still_classified_as_t1(self, make_dicom_series, tmp_path):
        """Unchanged behavior: no contrast marker anywhere -> plain t1."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_mprage_sag_p2",
            series_description="t1_mprage_sag_p2",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality == "t1"
        assert tech_meta["has_contrast"] is False

    def test_t2_tse_classification_unaffected(self, make_dicom_series, tmp_path):
        """Unchanged behavior: removing 'mpr' from t2's exclude list must not
        cause any regression (t2 series never contained 'mpr' in this dataset,
        but the pattern's exclude list changed, so verify explicitly)."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_tse_tra_4mm",
            series_description="t2_tse_tra_4mm",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t2"

    def test_flair_classification_unaffected(self, make_dicom_series, tmp_path):
        """Unchanged behavior: removing 'mpr' from t2fl's exclude list must not
        cause any regression."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t2_space_flair_fs",
            series_description="t2_space_flair_fs",
            image_type=['ORIGINAL', 'PRIMARY', 'M', 'NORM'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t2fl"
