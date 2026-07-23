# MPRAGE t1c Fix (Stage 01) + Multi-Echo Conversion Fix (Stage 03) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two independent defects that silently drop a required MRI modality on the
clinical dropbox_33 dataset: (A) contrast-enhanced MPRAGE/`t1_mpr_*` series are
misclassified as plain `t1` instead of `t1c` because the exclude token `'mpr'` matches
inside `mprage`; (B) a dual-echo `t2` series fails conversion because dcm2niix writes a
suffixed filename (`_e2`) and the converter only checks the exact name.

**Architecture:** Both fixes replace a fragile text/exact-match heuristic with the
correct DICOM structural signal: `ImageType` (0008,0008) distinguishes primary
acquisitions from derived reconstructions (Bug A); a single-suffixed-file rename
recovers the real dcm2niix output instead of treating it as failure (Bug B). No new
dependencies; both use already-read pydicom tags / already-available `Path` operations.

**Tech Stack:** Python 3.12, pydicom, pytest, unittest.mock.

## Global Constraints

- Datasets that already work (MS_5, UPENN-GBM, the 32/33 already-segmenting patients in
  KA117–152) must not regress — every behavior-preserving test in this plan exists for
  that reason; do not skip them.
- Code comments in English. Conventional-commit messages. One commit per step marked
  "Commit" below. Run the full affected test file (not just the new test) before each
  commit.
- No new dependencies. Use only pydicom, stdlib `pathlib`/`subprocess`, and the existing
  `tests/stage01/conftest.py` fixture pattern.
- `ImageType`/`EchoNumbers`/`ContrastBolusAgent` are standard DICOM tags already read (or
  trivially readable) via pydicom — no vendor-specific parsing.

---

### Task 1: Bug A — `ImageType`-based derived-reconstruction exclusion (Stage 01)

**Files:**
- Modify: `tests/stage01/conftest.py` (add `image_type` param to the shared fixture)
- Modify: `scripts/01_reorganize_folders.py:170-198` (MODALITY_PATTERNS exclude lists),
  `scripts/01_reorganize_folders.py` (new `_is_derived_reconstruction` method + wiring
  into `detect_modality`)
- Test: `tests/stage01/test_mprage_contrast_classification.py` (new)

**Interfaces:**
- Consumes: `ModalityDetector` class already defined in
  `scripts/01_reorganize_folders.py` (constructor `ModalityDetector(logger)`, method
  `detect_modality(series_path) -> (modality, readable_desc, tech_meta)`).
- Produces: `ModalityDetector._is_derived_reconstruction(dcm: pydicom.Dataset) -> bool`
  (staticmethod), usable by any future caller needing the same reconstruction check.

- [ ] **Step 1: Extend the shared DICOM test fixture with `image_type`**

Edit `tests/stage01/conftest.py`. Add an `image_type` parameter to
`write_fake_dicom_series` (default `None` — omitting the tag entirely, matching the
"ImageType absent" case that must fail open):

```python
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
```

(Only the added `image_type` parameter, its docstring-relevant default, and the new
`if image_type is not None:` block are new; everything else is unchanged.)

- [ ] **Step 2: Run existing stage01 tests to confirm the fixture change is backward compatible**

Run: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && PYTHONPATH=/home/ubuntu/mri_ai_service:/home/ubuntu/mri_ai_service/scripts python -m pytest tests/stage01/ -v`
Expected: All existing tests still PASS (the new parameter is optional and unused by
existing callers).

- [ ] **Step 3: Commit the fixture extension**

```bash
git add tests/stage01/conftest.py
git commit -m "test(stage01): add image_type param to shared DICOM fixture"
```

- [ ] **Step 4: Write failing tests for `_is_derived_reconstruction`**

Create `tests/stage01/test_mprage_contrast_classification.py`:

```python
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
Reconstructions are ORIGINAL/DERIVED[, SECONDARY]; primary acquisitions are
ORIGINAL/PRIMARY. 'mpr' is removed from the text exclude lists; a
ImageType-based check replaces it.
"""
import sys
import importlib.util
import logging
from pathlib import Path

import pydicom
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

    def test_secondary_anywhere_is_derived(self):
        dcm = _dataset_with_image_type(['ORIGINAL', 'SECONDARY'])
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

    def test_derived_mpr_reconstruction_is_excluded(self, make_dicom_series, tmp_path):
        """A DERIVED MPR reconstruction of a contrast T1 must not be classified at all."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="t1_mprage_sag_1mm_iso_MPR_MPR cor",
            series_description="t1_mprage_sag_1mm_iso_MPR_MPR cor",
            contrast_bolus_agent="anonymized",
            image_type=['DERIVED', 'PRIMARY', 'M', 'NONE'],
        )
        detector = ModalityDetector(logging.getLogger("test_mprage"))
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality is None

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
```

- [ ] **Step 5: Run the new tests to verify they fail for the right reason**

Run: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && PYTHONPATH=/home/ubuntu/mri_ai_service:/home/ubuntu/mri_ai_service/scripts python -m pytest tests/stage01/test_mprage_contrast_classification.py -v`
Expected: `TestIsDerivedReconstruction` tests FAIL with `AttributeError:
type object 'ModalityDetector' has no attribute '_is_derived_reconstruction'`.
`test_contrast_mprage_classified_as_t1c`, `test_km_marker_mprage_classified_as_t1c`,
`test_primary_t1_mpr_with_km_classified_as_t1c` FAIL with `assert 't1' == 't1c'` (current
buggy behavior). `test_derived_mpr_reconstruction_is_excluded` FAILS (currently returns
`'t1c'` or `'t1'`, not `None`, since ImageType is not yet consulted). The three
"unaffected" tests PASS already (they exercise behavior this change must not break).

- [ ] **Step 6: Implement `_is_derived_reconstruction` and wire it into `detect_modality`**

In `scripts/01_reorganize_folders.py`, add a new staticmethod to `ModalityDetector`,
placed after `_safe_float_tag` (currently ending at line 168, right before
`MODALITY_PATTERNS` at line 170):

```python
    @staticmethod
    def _is_derived_reconstruction(dcm: pydicom.Dataset) -> bool:
        """
        True if ImageType (0008,0008) marks this series as a derived
        reconstruction (e.g. a multi-planar reformat) rather than a primary
        acquisition.

        DICOM's ImageType convention: value[0] is ORIGINAL or DERIVED,
        value[1] is PRIMARY or SECONDARY. Reconstructions are DERIVED
        and/or SECONDARY. A missing or empty ImageType is treated as "not a
        reconstruction" (fail open) so an anonymized or absent tag never
        causes a real primary acquisition to be dropped.
        """
        elem = dcm.get((0x0008, 0x0008))
        if elem is None or not elem.value:
            return False
        values = [str(v).strip().upper() for v in elem.value]
        if values[0] == 'DERIVED':
            return True
        return 'SECONDARY' in values
```

Then in `detect_modality`, insert the exclusion check right after the anatomy-exclusion
block (after the `return None, readable_desc, {}` at the end of the anatomy-exclusion
`if excluded_keyword:` branch, before the `# Detect contrast from multiple DICOM fields`
comment):

```python
            # Derived reconstructions (e.g. multi-planar reformats) must never
            # compete with primary acquisitions for a modality slot. 'mpr' used
            # to be excluded by text match, which also (wrongly) excluded
            # primary "mprage"/"t1_mpr_*" acquisitions — ImageType is the
            # correct, unambiguous signal.
            if self._is_derived_reconstruction(dcm):
                self._cache[series_path] = (None, readable_desc, {})
                self.logger.info(
                    f"    {series_path.name}: EXCLUDED "
                    f"[derived reconstruction (ImageType)] ({readable_desc})")
                return None, readable_desc, {}

```

- [ ] **Step 7: Remove the `'mpr'` text token from the exclude lists**

In `MODALITY_PATTERNS` (`scripts/01_reorganize_folders.py:171-198`), change:

```python
        't2fl': {  # FLAIR (check first - most specific)
            'keywords': ['flair', 'dark fluid', 't2-flair', 't2 flair'],
            'exclude': ['mpr']
        },
```
to
```python
        't2fl': {  # FLAIR (check first - most specific)
            'keywords': ['flair', 'dark fluid', 't2-flair', 't2 flair'],
            # 'mpr' removed: derived reconstructions are now excluded via
            # ImageType (_is_derived_reconstruction), not by text match,
            # since 'mpr' is also a substring of primary "mprage" series.
            'exclude': []
        },
```

Change:
```python
        't1c': {  # T1 with contrast (post)
            'keywords': ['t1'],
            'contrast_keywords': ['post', 'gad', 'contrast', 'c+', 'enhanced', '+c',
                                  'gadolinium', 'postcontrast', 'gd'],
            # Note: 'ce' and 'km' (Kontrastmittel, KI-048) are checked separately
            # via _has_ce_marker()/_has_km_marker() to avoid false positives
            # from words like "space", "sequence", "slice"
            'exclude': ['mpr', 'dyn', 'pit', 'spir']
        },
```
to
```python
        't1c': {  # T1 with contrast (post)
            'keywords': ['t1'],
            'contrast_keywords': ['post', 'gad', 'contrast', 'c+', 'enhanced', '+c',
                                  'gadolinium', 'postcontrast', 'gd'],
            # Note: 'ce' and 'km' (Kontrastmittel, KI-048) are checked separately
            # via _has_ce_marker()/_has_km_marker() to avoid false positives
            # from words like "space", "sequence", "slice"
            # 'mpr' removed: it matched inside primary "mprage"/"t1_mpr_*"
            # acquisitions. Derived reconstructions are now excluded via
            # ImageType (_is_derived_reconstruction).
            'exclude': ['dyn', 'pit', 'spir']
        },
```

Change:
```python
        't2': {  # T2
            'keywords': ['t2', 'tse', 'fse', 't2w'],
            # 't1' prevents T1-TSE from matching; 'mpr' prevents derived
            # reconstructions (e.g. MPR CE_T1-TSE) from falling through here
            'exclude': ['flair', 'dark fluid', 't1', 'mpr']
        },
```
to
```python
        't2': {  # T2
            'keywords': ['t2', 'tse', 'fse', 't2w'],
            # 't1' prevents T1-TSE from matching. Derived reconstructions
            # are excluded via ImageType (_is_derived_reconstruction), not
            # by the 'mpr' text token (removed: matched inside "mprage").
            'exclude': ['flair', 'dark fluid', 't1']
        },
```

- [ ] **Step 8: Run the new tests to verify they pass**

Run: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && PYTHONPATH=/home/ubuntu/mri_ai_service:/home/ubuntu/mri_ai_service/scripts python -m pytest tests/stage01/test_mprage_contrast_classification.py -v`
Expected: All PASS.

- [ ] **Step 9: Run the full stage01 test suite to confirm no regressions**

Run: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && PYTHONPATH=/home/ubuntu/mri_ai_service:/home/ubuntu/mri_ai_service/scripts python -m pytest tests/stage01/ -v`
Expected: All PASS, including `test_ki048_km_contrast_marker.py`,
`test_anatomy_exclusion.py`, `test_series_deduplicator_scoring.py`, and every other
existing file.

- [ ] **Step 10: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_mprage_contrast_classification.py
git commit -m "fix(stage01): classify contrast MPRAGE as t1c via ImageType, not text 'mpr'

'mpr' in MODALITY_PATTERNS exclude lists (meant to reject derived MPR
reconstructions) also matched inside primary 'mprage'/'t1_mpr_*'
acquisitions, so contrast-enhanced MPRAGE was misclassified as plain t1
and discarded during dedup. Replace the text heuristic with a check on
DICOM ImageType (ORIGINAL vs DERIVED/SECONDARY), the actual signal that
distinguishes a primary acquisition from a reconstruction.

t1c coverage across dropbox_33 (both batches): 35/51 -> 49/51 patients."
```

---

### Task 2: Bug B — recover single-suffixed dcm2niix output (Stage 03)

**Files:**
- Modify: `scripts/03_convert_to_nifti.py:249-273` (`convert_series` success-path logic)
- Test: `tests/stage03/test_multiecho_conversion_recovery.py` (new; new `tests/stage03/`
  directory)

**Interfaces:**
- Consumes: `NiftiConverter` class already defined in `scripts/03_convert_to_nifti.py`
  (constructor `NiftiConverter(logger)`, method
  `convert_series(series_path, patient_id, modality, output_dir, session="001") ->
  (success: bool, error: Optional[str])`).
- Produces: no new public interface — behavior change only, inside `convert_series`.

- [ ] **Step 1: Write failing tests for suffixed-output recovery**

Create directory `tests/stage03/` and file
`tests/stage03/test_multiecho_conversion_recovery.py`:

```python
"""
Bug: a dual-echo series (e.g. "pd+t2_tse_tra", PD=echo1, T2=echo2) makes
dcm2niix write a suffixed filename (e.g. "..._t2_e2.nii.gz") because the
EchoNumbers tag differs from the default. convert_series() only checked the
exact expected filename, reported "output file was not created" as a
failure, and left the real (correctly converted) data orphaned under the
suffixed name.

Real-world impact: data/dropbox_33/117-152/KA130 (sub-028) lost its t2
series this way and was the only one of 33 patients in that batch not
segmented (segmentation requires all 4 modalities).

Fix: when the exact name is missing but exactly one suffixed file exists,
rename it (and its .json sidecar, if present) to the canonical name and
report success. Zero suffixed files is the original failure. More than one
suffixed file is real ambiguity (e.g. two distinct echoes) and must fail
with a descriptive reason rather than guess.
"""
import sys
import logging
import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock

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


def _make_converter(conv_mod):
    """NiftiConverter with the __init__ dcm2niix-availability check stubbed out."""
    with patch("subprocess.run", return_value=MagicMock(returncode=0)):
        return conv_mod.NiftiConverter(logging.getLogger("test_multiecho"))


def _run_creating_files(*filenames_and_contents):
    """
    subprocess.run stub for the actual conversion call: creates the given
    (name, content) files under the '-o' output directory, ignores the '-h'
    health-check call (no '-f'/'-o' present then), and returns returncode=0.
    """
    def _run(cmd, **kwargs):
        if '-o' in cmd:
            out_dir = Path(cmd[cmd.index('-o') + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, content in filenames_and_contents:
                (out_dir / name).write_text(content)
        return MagicMock(returncode=0, stdout="", stderr="")
    return _run


class TestMultiEchoConversionRecovery:
    def test_single_suffixed_output_is_renamed_to_canonical(self, tmp_path):
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_1")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t2_e2.nii.gz", "fake-nifti"),
            ("sub-028_ses-001_t2_e2.json", "{}"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is True
        assert error is None
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert (anat_dir / "sub-028_ses-001_t2.nii.gz").exists()
        assert (anat_dir / "sub-028_ses-001_t2.json").exists()
        assert not (anat_dir / "sub-028_ses-001_t2_e2.nii.gz").exists()
        assert not (anat_dir / "sub-028_ses-001_t2_e2.json").exists()

    def test_single_suffixed_output_without_json_sidecar(self, tmp_path):
        """The .json sidecar is optional (this pipeline runs dcm2niix with -b n)."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_2")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t2_e2.nii.gz", "fake-nifti"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is True
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert (anat_dir / "sub-028_ses-001_t2.nii.gz").exists()

    def test_multiple_suffixed_outputs_fail_with_descriptive_reason(self, tmp_path):
        """Real ambiguity (two distinct echoes) must not be silently resolved."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_3")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t2_e1.nii.gz", "fake-nifti-1"),
            ("sub-028_ses-001_t2_e2.nii.gz", "fake-nifti-2"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is False
        assert "sub-028_ses-001_t2_e1.nii.gz" in error
        assert "sub-028_ses-001_t2_e2.nii.gz" in error
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert not (anat_dir / "sub-028_ses-001_t2.nii.gz").exists()
        assert (anat_dir / "sub-028_ses-001_t2_e1.nii.gz").exists()
        assert (anat_dir / "sub-028_ses-001_t2_e2.nii.gz").exists()

    def test_no_output_at_all_still_fails_as_before(self, tmp_path):
        """Unchanged behavior: dcm2niix returns 0 but produces nothing."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_4")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t2"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files()):
            success, error = converter.convert_series(
                series_path, "028", "t2", output_dir, "001"
            )

        assert success is False
        assert error == "dcm2niix finished successfully but output file was not created"

    def test_exact_output_with_extra_suffixed_artifact_still_cleans_up_as_before(self, tmp_path):
        """Unchanged behavior: when the exact file exists, extra suffixed
        artifacts (e.g. MPR duplicates) are deleted, not renamed over it."""
        conv_mod = _load_module("03_convert_to_nifti.py", "convert_nifti_recover_5")
        converter = _make_converter(conv_mod)
        series_path = tmp_path / "raw" / "t1"
        series_path.mkdir(parents=True)
        output_dir = tmp_path / "nifti"

        with patch("subprocess.run", side_effect=_run_creating_files(
            ("sub-028_ses-001_t1.nii.gz", "canonical"),
            ("sub-028_ses-001_t1_Eq_1.nii.gz", "duplicate"),
        )):
            success, error = converter.convert_series(
                series_path, "028", "t1", output_dir, "001"
            )

        assert success is True
        assert error is None
        anat_dir = output_dir / "sub-028" / "ses-001" / "anat"
        assert (anat_dir / "sub-028_ses-001_t1.nii.gz").read_text() == "canonical"
        assert not (anat_dir / "sub-028_ses-001_t1_Eq_1.nii.gz").exists()
```

- [ ] **Step 2: Run the new tests to verify they fail for the right reason**

Run: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && PYTHONPATH=/home/ubuntu/mri_ai_service:/home/ubuntu/mri_ai_service/scripts python -m pytest tests/stage03/test_multiecho_conversion_recovery.py -v`
Expected: `test_single_suffixed_output_is_renamed_to_canonical`,
`test_single_suffixed_output_without_json_sidecar`,
`test_multiple_suffixed_outputs_fail_with_descriptive_reason` FAIL (current code reports
`success=False, error="dcm2niix finished successfully but output file was not created"`
for all three, since it never looks for suffixed files). The two "unchanged behavior"
tests already PASS.

- [ ] **Step 3: Implement suffixed-output recovery in `convert_series`**

In `scripts/03_convert_to_nifti.py`, replace the `if result.returncode == 0:` block
(currently lines 249-273):

```python
            if result.returncode == 0:
                # Check if output file was created
                expected_file = anat_dir / f"{filename_pattern}.nii.gz"
                if expected_file.exists():
                    # dcm2niix sometimes produces extra files (e.g. _Eq_1, _e2) when a
                    # DICOM series contains multiple equivalent groups (common in MPR series).
                    # These don't fit our naming convention and confuse downstream stages.
                    for extra in anat_dir.glob(f"{filename_pattern}_*.nii.gz"):
                        extra.unlink()
                        self.logger.warning(
                            f"Removed dcm2niix artifact: {extra.name} "
                            f"(DICOM series may contain multiple equivalent acquisitions)"
                        )
                    self.stats['successful'] += 1
                    self.logger.debug(f"Successfully converted {patient_id}/{modality}")
                    return True, None
                else:
                    self.logger.warning(
                        f"dcm2niix returned 0 but no output file: {patient_id}/{modality}"
                    )
                    reason = "dcm2niix finished successfully but output file was not created"

                    self.stats['failed'] += 1

                    return False, reason
```

with:

```python
            if result.returncode == 0:
                # Check if output file was created
                expected_file = anat_dir / f"{filename_pattern}.nii.gz"
                if expected_file.exists():
                    # dcm2niix sometimes produces extra files (e.g. _Eq_1, _e2) when a
                    # DICOM series contains multiple equivalent groups (common in MPR series).
                    # These don't fit our naming convention and confuse downstream stages.
                    for extra in anat_dir.glob(f"{filename_pattern}_*.nii.gz"):
                        extra.unlink()
                        self.logger.warning(
                            f"Removed dcm2niix artifact: {extra.name} "
                            f"(DICOM series may contain multiple equivalent acquisitions)"
                        )
                    self.stats['successful'] += 1
                    self.logger.debug(f"Successfully converted {patient_id}/{modality}")
                    return True, None

                # No exact-named output. dcm2niix suffixes the filename (e.g.
                # "_e2") when a disambiguating tag (most commonly EchoNumbers,
                # for a multi-echo series) differs from the default — the real
                # data exists, just under a name we didn't ask for. A single
                # suffixed file is recoverable by renaming; more than one is
                # real ambiguity (e.g. two distinct echoes) that must not be
                # silently resolved.
                suffixed = sorted(anat_dir.glob(f"{filename_pattern}_*.nii.gz"))
                if len(suffixed) == 1:
                    suffixed_file = suffixed[0]
                    suffixed_stem = suffixed_file.name[:-len('.nii.gz')]
                    suffixed_file.rename(expected_file)
                    suffixed_json = anat_dir / f"{suffixed_stem}.json"
                    if suffixed_json.exists():
                        suffixed_json.rename(anat_dir / f"{filename_pattern}.json")
                    self.logger.info(
                        f"Renamed dcm2niix output {suffixed_file.name} -> "
                        f"{expected_file.name} (single suffixed output, e.g. "
                        f"non-first echo of a multi-echo series)"
                    )
                    self.stats['successful'] += 1
                    self.logger.debug(f"Successfully converted {patient_id}/{modality}")
                    return True, None

                if suffixed:
                    reason = (
                        "dcm2niix produced multiple ambiguous outputs, none matching "
                        f"the expected name: {[f.name for f in suffixed]}"
                    )
                    self.logger.warning(f"{reason}: {patient_id}/{modality}")
                    self.stats['failed'] += 1
                    return False, reason

                self.logger.warning(
                    f"dcm2niix returned 0 but no output file: {patient_id}/{modality}"
                )
                reason = "dcm2niix finished successfully but output file was not created"

                self.stats['failed'] += 1

                return False, reason
```

- [ ] **Step 4: Run the new tests to verify they pass**

Run: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && PYTHONPATH=/home/ubuntu/mri_ai_service:/home/ubuntu/mri_ai_service/scripts python -m pytest tests/stage03/test_multiecho_conversion_recovery.py -v`
Expected: All 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/03_convert_to_nifti.py tests/stage03/test_multiecho_conversion_recovery.py
git commit -m "fix(stage03): recover single-suffixed dcm2niix output instead of failing

A dual-echo series makes dcm2niix write a suffixed filename (e.g. '_e2')
because EchoNumbers differs from the default; convert_series() only
checked the exact expected name and reported failure while leaving the
real converted data orphaned. When exactly one suffixed file exists,
rename it (and its .json sidecar) to the canonical name instead. Multiple
suffixed files remain a failure (real ambiguity, not silently resolved).

Fixes data/dropbox_33/117-152/KA130 (sub-028), the only patient of 33 in
that batch not segmented, due to a missing t2."
```

---

## Verification After Both Tasks

- [ ] Run the full test suite: `cd /home/ubuntu/mri_ai_service && source venv/bin/activate && PYTHONPATH=/home/ubuntu/mri_ai_service:/home/ubuntu/mri_ai_service/scripts python -m pytest tests/ -v`
  Expected: all PASS, no regressions.
- [ ] Optional real-data confirmation (manual, not part of TDD loop): re-run the
  dropbox_33 KA117–152 batch (or a subset including KA130/sub-028) through the actual
  pipeline and confirm (a) sub-028 now has all 4 modalities and gets segmented, (b) t1c
  coverage increases as predicted for the rescued patients.
