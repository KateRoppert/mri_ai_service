# Stage 01 Protocol-Specific Series Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `SeriesDeduplicator._select_best_series`'s `max(slice_count)` heuristic with a config-driven scoring system (text keywords, resolution, failure markers, FLAIR TI) and a hard anatomy exclusion filter, closing KI-027 and KI-001.

**Architecture:** `ModalityDetector.detect_modality()` gains a hard anatomy-exclusion check and starts returning technical metadata (slice thickness, inversion time, contrast flag) alongside modality — read once, no second DICOM pass. A new module-level `score_series()` function combines that metadata with YAML-configured text weights to pick the best candidate among same-modality series in `SeriesDeduplicator._select_best_series`. All tunable weights live in `configs/series_scoring.yaml`, loaded via a new `load_series_scoring_config()` in `utils/config_loader.py`.

**Tech Stack:** Python 3.12, pydicom, PyYAML, pytest.

## Global Constraints

- Design source of truth: `docs/superpowers/specs/2026-06-26-stage01-protocol-scoring-design.md` — read it before starting if anything below is ambiguous.
- Anatomy exclusion (spine/cervical/pituitary/orbit/neck) is a **hard filter, never a fallback** — even if it leaves a modality with zero candidates for a session. Do not add a fallback path.
- Do **not** modify `ModalityDetector._match_modality`, `_has_ce_marker`, `_detect_contrast`, or `_detect_by_technical_params` — they already work and are out of scope.
- Do **not** modify `FileOrganizer.copy_series` / `_extract_and_save_metadata` (anonymization/metadata flow) — selection logic changes only affect *which* series reaches that code, not the code itself.
- All new tests go under `tests/stage01/` — never add new test files at the repo root or inside `scripts/`.
- Config-driven: scoring weights live in `configs/series_scoring.yaml`, not hardcoded in Python. Missing/corrupt config file must raise `ConfigValidationError` (no silent fallback at the config-loading level).
- Before every commit: `python -m py_compile scripts/01_reorganize_folders.py utils/config_loader.py` must succeed, and the task's own tests must pass.
- Commit style: conventional commits, one commit per task (per CLAUDE.md project convention).
- `scripts/01_reorganize_folders.py` has no module-level package name compatible with `import` (filename starts with a digit) — load it in tests via `importlib.util.spec_from_file_location`, exactly like the existing `tests root /test_stage01_stage03_fixes.py` does. Use a **unique** `module_name` string per test file when calling this loader, to avoid `sys.modules` collisions between test files in the same pytest run.

---

### Task 1: `configs/series_scoring.yaml` + `load_series_scoring_config()`

**Files:**
- Create: `configs/series_scoring.yaml`
- Modify: `utils/config_loader.py` (append function at end of file)
- Test: `tests/stage01/test_series_scoring_config.py`

**Interfaces:**
- Produces: `load_series_scoring_config() -> Dict[str, Any]` in `utils.config_loader`, raising `utils.config_loader.ConfigValidationError` on missing/invalid file. Returned dict has top-level keys: `anatomy_exclude`, `failure_markers`, `text_weights`, `resolution_scoring`, `flair_ti_bonus`.

- [ ] **Step 1: Write the failing test**

Create `tests/stage01/test_series_scoring_config.py`:

```python
import sys
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from utils.config_loader import load_series_scoring_config, ConfigValidationError


def test_loads_real_config_with_expected_top_level_keys():
    config = load_series_scoring_config()
    assert {'anatomy_exclude', 'failure_markers', 'text_weights',
            'resolution_scoring', 'flair_ti_bonus'} <= config.keys()


def test_text_weights_has_all_four_modalities():
    config = load_series_scoring_config()
    assert {'t1', 't1c', 't2', 't2fl'} <= config['text_weights'].keys()


def test_anatomy_exclude_keywords_present():
    config = load_series_scoring_config()
    keywords = config['anatomy_exclude']['keywords']
    assert 'spine' in keywords
    assert 'pituitary' in keywords


def test_missing_file_raises_config_validation_error(monkeypatch, tmp_path):
    import utils.config_loader as config_loader_mod
    fake_module_file = tmp_path / "subdir" / "config_loader.py"
    fake_module_file.parent.mkdir(parents=True)
    monkeypatch.setattr(config_loader_mod, "__file__", str(fake_module_file))
    with pytest.raises(ConfigValidationError):
        config_loader_mod.load_series_scoring_config()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/stage01/test_series_scoring_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_series_scoring_config'` (function doesn't exist yet) and `FileNotFoundError` for `configs/series_scoring.yaml` (file doesn't exist yet).

- [ ] **Step 3: Create `configs/series_scoring.yaml`**

```yaml
# Scoring weights for selecting the best series among multiple candidates
# of the same modality in scripts/01_reorganize_folders.py
# (SeriesDeduplicator._select_best_series). See KI-027, KI-001 in
# KNOWN_ISSUES.md and docs/superpowers/specs/2026-06-26-stage01-protocol-scoring-design.md.

anatomy_exclude:
  # Hard filter — if any keyword is a substring of "protocol_name |
  # series_description" (lowercased), the series is excluded entirely,
  # never becomes a candidate for any brain modality. No fallback.
  keywords: [spine, cervical, c-spine, pituitary, orbit, neck]

failure_markers:
  keywords: [failed, repeat, redo, retry, motion, artifact, incomplete]
  penalty: 0.5

text_weights:
  t1:
    tfe: 3.0
    tse: 2.0
    se: 1.0
    "3d": 1.5
    mpr: 0.1
    clear: 1.2
    brain: 1.1
  t1c:
    tfe: 3.0
    tse: 2.0
    se: 1.0
    "3d": 1.5
    mpr: 0.1
    brain: 1.1
    gad: 1.3
    "+c": 1.3
    contrast: 1.2
  t2:
    tse: 2.0
    "3d": 1.3
    mpr: 0.1
    sense: 1.3
    brain: 1.1
  t2fl:
    "3d": 2.0
    mpr: 0.1
    sense: 1.3
    long: 1.2
    brain: 1.1

resolution_scoring:
  reference_slice_thickness_mm: 1.0
  min_factor: 0.5
  max_factor: 2.0

flair_ti_bonus:
  threshold_ms: 1500
  bonus: 1.2
```

- [ ] **Step 4: Add `load_series_scoring_config()` to `utils/config_loader.py`**

Append to the end of `utils/config_loader.py` (after `load_lesion_type_config`):

```python


def load_series_scoring_config() -> Dict[str, Any]:
    """
    Load series scoring configuration from configs/series_scoring.yaml.

    Used by Stage 01 (scripts/01_reorganize_folders.py) to select the best
    series among multiple candidates for the same modality, and to exclude
    non-brain anatomy series. See KI-027, KI-001 in KNOWN_ISSUES.md.

    Returns:
        Dict with keys: anatomy_exclude, failure_markers, text_weights,
        resolution_scoring, flair_ti_bonus.

    Raises:
        ConfigValidationError: If the file is missing or YAML is invalid.
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'series_scoring.yaml'

    if not config_path.exists():
        raise ConfigValidationError(f"Config file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigValidationError(f"Failed to parse YAML: {e}")

    return config
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_series_scoring_config.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add configs/series_scoring.yaml utils/config_loader.py tests/stage01/test_series_scoring_config.py
git commit -m "feat(stage01): add series_scoring.yaml config and loader (KI-027)"
```

---

### Task 2: Shared fake-DICOM test fixture (`tests/stage01/conftest.py`)

**Files:**
- Create: `tests/stage01/conftest.py`
- Test: `tests/stage01/test_conftest_dicom_fixture.py`

**Interfaces:**
- Produces: pytest fixture `make_dicom_series(series_dir: Path, protocol_name: str = "", series_description: str = "", slice_thickness: Optional[float] = None, inversion_time: Optional[float] = None, contrast_bolus_agent: str = "", n_files: int = 1) -> Path`. Available automatically to every test file under `tests/stage01/` (no import needed — it's a pytest fixture, discovered via conftest.py).

- [ ] **Step 1: Write the failing test**

Create `tests/stage01/test_conftest_dicom_fixture.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/stage01/test_conftest_dicom_fixture.py -v`
Expected: FAIL with `fixture 'make_dicom_series' not found`

- [ ] **Step 3: Create `tests/stage01/conftest.py`**

```python
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
        ds.save_as(str(series_dir / f"IM-{i:04d}.dcm"), write_like_original=False)
    return series_dir


@pytest.fixture
def make_dicom_series():
    return write_fake_dicom_series
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_conftest_dicom_fixture.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add tests/stage01/conftest.py tests/stage01/test_conftest_dicom_fixture.py
git commit -m "test(stage01): add shared fake-DICOM fixture for Stage 01 tests"
```

---

### Task 3: `SeriesInfo` technical fields + `detect_modality` returns tech metadata

**Files:**
- Modify: `scripts/01_reorganize_folders.py:53-61` (`SeriesInfo` dataclass)
- Modify: `scripts/01_reorganize_folders.py:139-224` (`ModalityDetector.__init__`, `detect_modality`)
- Modify: `scripts/01_reorganize_folders.py:1049-1063` (`_process_one_patient_core`, the sole caller of `detect_modality`)
- Test: `tests/stage01/test_modality_detector_technical_meta.py`

**Interfaces:**
- Consumes: `make_dicom_series` fixture from Task 2.
- Produces: `ModalityDetector.detect_modality(series_path: Path) -> Tuple[Optional[str], str, Dict]` where the dict has keys `slice_thickness_mm: Optional[float]`, `ti_ms: Optional[float]`, `has_contrast: bool`. `SeriesInfo` dataclass gains fields `slice_thickness_mm: Optional[float] = None`, `ti_ms: Optional[float] = None`, `has_contrast: bool = False`. New staticmethod `ModalityDetector._safe_float_tag(dcm, tag: Tuple[int, int]) -> Optional[float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/stage01/test_modality_detector_technical_meta.py`:

```python
import logging
import sys
import importlib.util
from pathlib import Path

import pytest

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_techmeta")
ModalityDetector = reorganize_mod.ModalityDetector


@pytest.fixture
def detector():
    return ModalityDetector(logging.getLogger("test_techmeta"))


class TestDetectModalityTechnicalMeta:
    def test_returns_three_tuple(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1-TFE (3D brain)",
            series_description="T1-TFE (3D brain)",
        )
        result = detector.detect_modality(series_dir)
        assert len(result) == 3

    def test_slice_thickness_extracted(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1-TFE (3D brain)",
            series_description="T1-TFE (3D brain)",
            slice_thickness=1.1,
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['slice_thickness_mm'] == 1.1

    def test_ti_extracted_for_flair(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="FLAIR (3D brain)",
            series_description="FLAIR (3D brain)",
            inversion_time=1660.0,
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['ti_ms'] == 1660.0

    def test_has_contrast_true_when_bolus_agent_present(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="CE_T1-TFE (3D brain)",
            series_description="CE_T1-TFE (3D brain)",
            contrast_bolus_agent="Gadovist",
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['has_contrast'] is True

    def test_missing_tags_default_to_none(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T2-TSE (axi brain)",
            series_description="T2-TSE (axi brain)",
        )
        _, _, tech_meta = detector.detect_modality(series_dir)
        assert tech_meta['slice_thickness_mm'] is None
        assert tech_meta['ti_ms'] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/stage01/test_modality_detector_technical_meta.py -v`
Expected: FAIL — `detect_modality` still returns a 2-tuple, `len(result) == 3` fails and unpacking `_, _, tech_meta = ...` raises `ValueError: not enough values to unpack`.

- [ ] **Step 3: Update `SeriesInfo` dataclass**

In `scripts/01_reorganize_folders.py`, find:

```python
@dataclass
class SeriesInfo:
    """Information about a DICOM series."""
    original_path: Path
    patient_id: str
    date: str  # YYYYMMDD format
    modality: Optional[str] = None
    slice_count: int = 0
    series_description: str = ""
```

Replace with:

```python
@dataclass
class SeriesInfo:
    """Information about a DICOM series."""
    original_path: Path
    patient_id: str
    date: str  # YYYYMMDD format
    modality: Optional[str] = None
    slice_count: int = 0
    series_description: str = ""
    slice_thickness_mm: Optional[float] = None
    ti_ms: Optional[float] = None
    has_contrast: bool = False
```

- [ ] **Step 4: Update `ModalityDetector.__init__` cache type annotation**

Find:

```python
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._cache: Dict[Path, Tuple[Optional[str], str]] = {}  # series_path -> (modality, description)
```

Replace with:

```python
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._cache: Dict[Path, Tuple[Optional[str], str, Dict]] = {}  # series_path -> (modality, description, tech_meta)
```

- [ ] **Step 5: Add `_safe_float_tag` staticmethod**

Find the `_has_ce_marker` staticmethod:

```python
    @staticmethod
    def _has_ce_marker(text: str) -> bool:
        """Check if 'ce' appears as a standalone contrast-enhanced marker."""
        return bool(ModalityDetector._CE_PATTERN.search(text))
```

Add immediately after it:

```python

    @staticmethod
    def _safe_float_tag(dcm: pydicom.Dataset, tag: Tuple[int, int]) -> Optional[float]:
        """Read a DICOM tag as float, returning None if missing or unparsable."""
        val = dcm.get(tag)
        if val is None:
            return None
        try:
            return float(val.value if hasattr(val, 'value') else val)
        except (ValueError, TypeError):
            return None
```

- [ ] **Step 6: Update the three early-return points and the docstring in `detect_modality`**

In the docstring, find:

```python
        Returns:
            (modality, combined_description) where:
                - modality: 't1', 't1c', 't2', 't2fl', or None
                - combined_description: "ProtocolName | SeriesDescription"
        """
```

Replace with:

```python
        Returns:
            (modality, combined_description, tech_meta) where:
                - modality: 't1', 't1c', 't2', 't2fl', or None
                - combined_description: "ProtocolName | SeriesDescription"
                - tech_meta: dict with slice_thickness_mm, ti_ms, has_contrast
                  (empty dict {} when modality detection short-circuited
                  before reading these, e.g. no DICOM files found)
        """
```

Find (no DICOM files found):

```python
        dicom_files = sorted([f for f in series_path.rglob("*.dcm") if f.is_file()])
        if not dicom_files:
            self.logger.warning(f"No DICOM files found in {series_path}")
            self._cache[series_path] = (None, "")
            return None, ""
```

Replace with:

```python
        dicom_files = sorted([f for f in series_path.rglob("*.dcm") if f.is_file()])
        if not dicom_files:
            self.logger.warning(f"No DICOM files found in {series_path}")
            self._cache[series_path] = (None, "", {})
            return None, "", {}
```

Find (no ProtocolName/SeriesDescription):

```python
            if not combined_text:
                self.logger.warning(f"No ProtocolName or SeriesDescription in {dicom_files[0]}")
                self._cache[series_path] = (None, "")
                return None, ""
```

Replace with:

```python
            if not combined_text:
                self.logger.warning(f"No ProtocolName or SeriesDescription in {dicom_files[0]}")
                self._cache[series_path] = (None, "", {})
                return None, "", {}
```

Find (exception handler at the end of the method):

```python
        except Exception as e:
            self.logger.warning(f"Failed to read {dicom_files[0]}: {e}")
            self._cache[series_path] = (None, "")
            return None, ""
```

Replace with:

```python
        except Exception as e:
            self.logger.warning(f"Failed to read {dicom_files[0]}: {e}")
            self._cache[series_path] = (None, "", {})
            return None, "", {}
```

- [ ] **Step 7: Build tech_meta and update the success-path return**

Find:

```python
            # Level 1: Pattern matching on combined text
            modality = self._match_modality(combined_text, has_contrast)
            detection_method = "pattern"
            
            # Level 2: Technical parameter fallback (TR/TE/TI)
            if modality is None:
                modality = self._detect_by_technical_params(dcm, has_contrast)
                if modality:
                    detection_method = "technical_params"
            
            # Cache result
            self._cache[series_path] = (modality, readable_desc)
            
            # Log at INFO level so detection decisions are always visible
            series_name = series_path.name
            if modality:
                self.logger.info(
                    f"    {series_name}: {modality} "
                    f"[{detection_method}, contrast={has_contrast}] "
                    f"({readable_desc})")
            else:
                self.logger.info(
                    f"    {series_name}: NO MATCH "
                    f"[contrast={has_contrast}] "
                    f"({readable_desc})")
            
            return modality, readable_desc
```

Replace with:

```python
            # Level 1: Pattern matching on combined text
            modality = self._match_modality(combined_text, has_contrast)
            detection_method = "pattern"
            
            # Level 2: Technical parameter fallback (TR/TE/TI)
            if modality is None:
                modality = self._detect_by_technical_params(dcm, has_contrast)
                if modality:
                    detection_method = "technical_params"

            # Technical metadata for downstream series scoring (KI-027) —
            # read once here so SeriesDeduplicator never re-reads the DICOM.
            tech_meta = {
                'slice_thickness_mm': self._safe_float_tag(dcm, (0x0018, 0x0050)),
                'ti_ms': self._safe_float_tag(dcm, (0x0018, 0x0082)),
                'has_contrast': has_contrast,
            }

            # Cache result
            self._cache[series_path] = (modality, readable_desc, tech_meta)
            
            # Log at INFO level so detection decisions are always visible
            series_name = series_path.name
            if modality:
                self.logger.info(
                    f"    {series_name}: {modality} "
                    f"[{detection_method}, contrast={has_contrast}] "
                    f"({readable_desc})")
            else:
                self.logger.info(
                    f"    {series_name}: NO MATCH "
                    f"[contrast={has_contrast}] "
                    f"({readable_desc})")
            
            return modality, readable_desc, tech_meta
```

- [ ] **Step 8: Update the sole caller in `_process_one_patient_core`**

Find:

```python
        # Detect modality
        modality, series_description = modality_detector.detect_modality(series_dir)
        
        # Filter: only copy modalities required for this lesion type
        if modality in MODALITY_BIDS_SUFFIX and modality in required_modalities:
            series_info = SeriesInfo(
                original_path=series_dir,
                patient_id=original_patient_id,
                date=date,
                modality=modality,
                series_description=series_description,
            )
```

Replace with:

```python
        # Detect modality
        modality, series_description, tech_meta = modality_detector.detect_modality(series_dir)
        
        # Filter: only copy modalities required for this lesion type
        if modality in MODALITY_BIDS_SUFFIX and modality in required_modalities:
            series_info = SeriesInfo(
                original_path=series_dir,
                patient_id=original_patient_id,
                date=date,
                modality=modality,
                series_description=series_description,
                slice_thickness_mm=tech_meta.get('slice_thickness_mm'),
                ti_ms=tech_meta.get('ti_ms'),
                has_contrast=tech_meta.get('has_contrast', False),
            )
```

- [ ] **Step 9: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_modality_detector_technical_meta.py -v`
Expected: 5 passed

Also run the pre-existing Stage 01 tests to confirm no regression:
Run: `python -m pytest test_stage01_stage03_fixes.py -v`
Expected: all passed (this file only calls `_match_modality` directly, not `detect_modality`, so it is unaffected by the signature change — confirm this explicitly)

- [ ] **Step 10: Verify compilation**

Run: `python -m py_compile scripts/01_reorganize_folders.py`
Expected: no output, exit code 0

- [ ] **Step 11: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_modality_detector_technical_meta.py
git commit -m "feat(stage01): detect_modality returns technical metadata (KI-027)"
```

---

### Task 4: Hard anatomy exclusion in `ModalityDetector`

**Files:**
- Modify: `scripts/01_reorganize_folders.py` (`ModalityDetector.__init__`, new method, `detect_modality`)
- Test: `tests/stage01/test_anatomy_exclusion.py`

**Interfaces:**
- Consumes: `make_dicom_series` fixture (Task 2); `ModalityDetector.detect_modality` 3-tuple contract (Task 3).
- Produces: `ModalityDetector.__init__(self, logger, scoring_config: Optional[Dict] = None)`; `ModalityDetector._is_excluded_anatomy(combined_text: str) -> Optional[str]` (returns matched keyword or `None`).

- [ ] **Step 1: Write the failing test**

Create `tests/stage01/test_anatomy_exclusion.py`:

```python
import logging
import sys
import importlib.util
from pathlib import Path

import pytest

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_anatomy")
ModalityDetector = reorganize_mod.ModalityDetector

ANATOMY_CONFIG = {'anatomy_exclude': {'keywords': ['spine', 'cervical', 'c-spine', 'pituitary', 'orbit', 'neck']}}


@pytest.fixture
def detector():
    return ModalityDetector(logging.getLogger("test_anatomy"), scoring_config=ANATOMY_CONFIG)


@pytest.fixture
def detector_no_config():
    return ModalityDetector(logging.getLogger("test_anatomy_noconfig"))


class TestIsExcludedAnatomy:
    def test_c_spine_text_excluded(self, detector):
        assert detector._is_excluded_anatomy("ce_t1-tse (sag c-spine)") == 'c-spine'

    def test_cervical_text_excluded(self, detector):
        assert detector._is_excluded_anatomy("t2-tse cervical spine sagittal") is not None

    def test_pituitary_text_excluded(self, detector):
        assert detector._is_excluded_anatomy("t1 pituitary dynamic") == 'pituitary'

    def test_normal_brain_text_not_excluded(self, detector):
        assert detector._is_excluded_anatomy("t1-tfe (3d brain)") is None

    def test_no_scoring_config_excludes_nothing(self, detector_no_config):
        # Backward-compatible default: without an explicit scoring_config,
        # no anatomy keywords are configured, so nothing is excluded.
        assert detector_no_config._is_excluded_anatomy("ce_t1-tse (sag c-spine)") is None


class TestDetectModalityExcludesAnatomy:
    def test_c_spine_series_returns_none_modality(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="CE_T1-TSE (sag C-spine)",
            series_description="CE_T1-TSE (sag C-spine)",
        )
        modality, _, tech_meta = detector.detect_modality(series_dir)
        assert modality is None
        assert tech_meta == {}

    def test_normal_brain_series_not_excluded(self, detector, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="T1-TFE (3D brain)",
            series_description="T1-TFE (3D brain)",
        )
        modality, _, _ = detector.detect_modality(series_dir)
        assert modality == "t1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/stage01/test_anatomy_exclusion.py -v`
Expected: FAIL — `ModalityDetector(...)` raises `TypeError: __init__() got an unexpected keyword argument 'scoring_config'`, and `_is_excluded_anatomy` doesn't exist.

- [ ] **Step 3: Update `ModalityDetector.__init__`**

Find (already updated by Task 3):

```python
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._cache: Dict[Path, Tuple[Optional[str], str, Dict]] = {}  # series_path -> (modality, description, tech_meta)
```

Replace with:

```python
    def __init__(self, logger: logging.Logger, scoring_config: Optional[Dict] = None):
        self.logger = logger
        self._cache: Dict[Path, Tuple[Optional[str], str, Dict]] = {}  # series_path -> (modality, description, tech_meta)
        self._anatomy_exclude_keywords: List[str] = (
            (scoring_config or {}).get('anatomy_exclude', {}).get('keywords', [])
        )
```

- [ ] **Step 4: Add `_is_excluded_anatomy` method**

Find the start of `_match_modality`:

```python
    def _match_modality(self, combined_text: str, has_contrast: bool = False) -> Optional[str]:
```

Insert immediately before it:

```python
    def _is_excluded_anatomy(self, combined_text: str) -> Optional[str]:
        """
        Check whether combined_text indicates non-brain anatomy this
        pipeline must never treat as a brain modality (e.g. cervical
        spine, pituitary). Returns the matched keyword, or None.
        """
        for keyword in self._anatomy_exclude_keywords:
            if keyword in combined_text:
                return keyword
        return None

```

- [ ] **Step 5: Wire the exclusion check into `detect_modality`**

Find:

```python
            # Store readable description for logging
            readable_desc = f"{dcm.get('ProtocolName', 'N/A')} | {dcm.get('SeriesDescription', 'N/A')}"
            
            # Detect contrast from multiple DICOM fields
            has_contrast = self._detect_contrast(dcm, combined_text)
```

Replace with:

```python
            # Store readable description for logging
            readable_desc = f"{dcm.get('ProtocolName', 'N/A')} | {dcm.get('SeriesDescription', 'N/A')}"

            # Hard anatomy exclusion (KI-027) — non-brain series (e.g. spine)
            # must never be treated as a brain modality candidate, even if
            # no other candidate exists for that modality in the session.
            excluded_keyword = self._is_excluded_anatomy(combined_text)
            if excluded_keyword:
                self._cache[series_path] = (None, readable_desc, {})
                self.logger.info(
                    f"    {series_path.name}: EXCLUDED "
                    f"[non-brain anatomy: '{excluded_keyword}'] ({readable_desc})")
                return None, readable_desc, {}

            # Detect contrast from multiple DICOM fields
            has_contrast = self._detect_contrast(dcm, combined_text)
```

- [ ] **Step 6: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_anatomy_exclusion.py -v`
Expected: 7 passed

Run regression check: `python -m pytest test_stage01_stage03_fixes.py tests/stage01/ -v`
Expected: all passed

- [ ] **Step 7: Verify compilation**

Run: `python -m py_compile scripts/01_reorganize_folders.py`
Expected: no output, exit code 0

- [ ] **Step 8: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_anatomy_exclusion.py
git commit -m "feat(stage01): hard-exclude non-brain anatomy series (KI-027)"
```

---

### Task 5: KI-001 — fix misleading log for post-contrast FLAIR

**Files:**
- Modify: `scripts/01_reorganize_folders.py` (`detect_modality` logging block)
- Test: `tests/stage01/test_ki001_flair_contrast_log.py`

**Interfaces:**
- Consumes: `make_dicom_series` fixture (Task 2); `detect_modality` 3-tuple contract (Task 3/4).
- Produces: no new public interface — behavior-only change to log wording.

- [ ] **Step 1: Write the failing test**

Create `tests/stage01/test_ki001_flair_contrast_log.py`:

```python
import io
import logging
import sys
import importlib.util
from pathlib import Path

import pytest

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_ki001")
ModalityDetector = reorganize_mod.ModalityDetector


def _detect_with_captured_log(series_dir):
    logger = logging.getLogger("test_ki001")
    logger.setLevel(logging.INFO)
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        detector = ModalityDetector(logger)
        result = detector.detect_modality(series_dir)
    finally:
        logger.removeHandler(handler)
    return result, log_capture.getvalue()


class TestKi001FlairContrastLog:
    def test_flair_with_contrast_log_is_informational_not_alarming(self, make_dicom_series, tmp_path):
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="FLAIR (3D brain) post-contrast",
            series_description="FLAIR (3D brain) post-contrast",
            contrast_bolus_agent="Gadovist",
        )
        (modality, _, _), log_output = _detect_with_captured_log(series_dir)
        assert modality == "t2fl"
        assert "contrast=True" not in log_output
        assert "informational" in log_output

    def test_t1c_with_contrast_log_unchanged(self, make_dicom_series, tmp_path):
        """Non-FLAIR contrast logging keeps its original, unmodified wording."""
        series_dir = make_dicom_series(
            tmp_path / "series1",
            protocol_name="CE_T1-TFE (3D brain)",
            series_description="CE_T1-TFE (3D brain)",
            contrast_bolus_agent="Gadovist",
        )
        (modality, _, _), log_output = _detect_with_captured_log(series_dir)
        assert modality == "t1c"
        assert "contrast=True" in log_output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/stage01/test_ki001_flair_contrast_log.py -v`
Expected: FAIL on `test_flair_with_contrast_log_is_informational_not_alarming` — log still contains `"contrast=True"` and not `"informational"`.

- [ ] **Step 3: Update the logging block in `detect_modality`**

Find:

```python
            # Log at INFO level so detection decisions are always visible
            series_name = series_path.name
            if modality:
                self.logger.info(
                    f"    {series_name}: {modality} "
                    f"[{detection_method}, contrast={has_contrast}] "
                    f"({readable_desc})")
            else:
                self.logger.info(
                    f"    {series_name}: NO MATCH "
                    f"[contrast={has_contrast}] "
                    f"({readable_desc})")
```

Replace with:

```python
            # Log at INFO level so detection decisions are always visible
            series_name = series_path.name
            if modality == 't2fl' and has_contrast:
                # KI-001: post-contrast FLAIR is a normal clinical case,
                # not a detection ambiguity — phrase the log accordingly.
                self.logger.info(
                    f"    {series_name}: {modality} "
                    f"[{detection_method}, contrast marker present "
                    f"(informational — common for post-contrast FLAIR, "
                    f"does not affect classification)] "
                    f"({readable_desc})")
            elif modality:
                self.logger.info(
                    f"    {series_name}: {modality} "
                    f"[{detection_method}, contrast={has_contrast}] "
                    f"({readable_desc})")
            else:
                self.logger.info(
                    f"    {series_name}: NO MATCH "
                    f"[contrast={has_contrast}] "
                    f"({readable_desc})")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_ki001_flair_contrast_log.py -v`
Expected: 2 passed

Run regression check: `python -m pytest test_stage01_stage03_fixes.py tests/stage01/ -v`
Expected: all passed

- [ ] **Step 5: Verify compilation**

Run: `python -m py_compile scripts/01_reorganize_folders.py`
Expected: no output, exit code 0

- [ ] **Step 6: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_ki001_flair_contrast_log.py
git commit -m "fix(stage01): stop logging post-contrast FLAIR as ambiguous (KI-001)"
```

---

### Task 6: `score_series()` scoring function

**Files:**
- Modify: `scripts/01_reorganize_folders.py` — add module-level function immediately before the `SeriesDeduplicator` class
- Test: `tests/stage01/test_series_scoring.py`

**Interfaces:**
- Consumes: `SeriesInfo` dataclass (Task 3 fields: `slice_thickness_mm`, `ti_ms`).
- Produces: `score_series(series_info: SeriesInfo, modality: str, scoring_config: Dict, logger: logging.Logger) -> float`. Returns `1.0` when no configured signal matches (callers must treat this as "no opinion, use a tie-break").

- [ ] **Step 1: Write the failing test**

Create `tests/stage01/test_series_scoring.py`:

```python
import logging
import sys
import importlib.util
from pathlib import Path

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_scoring")
SeriesInfo = reorganize_mod.SeriesInfo
score_series = reorganize_mod.score_series

SCORING_CONFIG = {
    'failure_markers': {'keywords': ['failed', 'repeat', 'motion'], 'penalty': 0.5},
    'text_weights': {
        't1c': {'tfe': 3.0, 'tse': 2.0, '3d': 1.5, 'mpr': 0.1},
        't2fl': {'3d': 2.0, 'mpr': 0.1},
    },
    'resolution_scoring': {
        'reference_slice_thickness_mm': 1.0, 'min_factor': 0.5, 'max_factor': 2.0,
    },
    'flair_ti_bonus': {'threshold_ms': 1500, 'bonus': 1.2},
}

LOGGER = logging.getLogger("test_scoring")


def _series(desc, modality="t1c", slice_thickness=None, ti=None):
    return SeriesInfo(
        original_path=Path(f"/fake/{desc}"),
        patient_id="P1", date="20230101",
        modality=modality, series_description=desc,
        slice_thickness_mm=slice_thickness, ti_ms=ti,
    )


class TestScoreSeries:
    def test_3d_scores_higher_than_mpr_reformat(self):
        threed = _series("CE_T1-TFE (3D brain)")
        mpr = _series("MPR CE_T1-TFE 2.5mm (cor brain)")
        assert score_series(threed, "t1c", SCORING_CONFIG, LOGGER) > score_series(mpr, "t1c", SCORING_CONFIG, LOGGER)

    def test_tfe_scores_higher_than_tse(self):
        tfe = _series("CE_T1-TFE (3D brain)")
        tse = _series("CE_T1-TSE (3D brain)")
        assert score_series(tfe, "t1c", SCORING_CONFIG, LOGGER) > score_series(tse, "t1c", SCORING_CONFIG, LOGGER)

    def test_failure_marker_reduces_score(self):
        clean = _series("CE_T1-TFE (3D brain)")
        repeated = _series("CE_T1-TFE (3D brain) repeat motion")
        assert score_series(repeated, "t1c", SCORING_CONFIG, LOGGER) < score_series(clean, "t1c", SCORING_CONFIG, LOGGER)

    def test_thinner_slice_scores_higher(self):
        thin = _series("CE_T1-TFE (3D brain)", slice_thickness=1.0)
        thick = _series("CE_T1-TFE (3D brain)", slice_thickness=2.5)
        assert score_series(thin, "t1c", SCORING_CONFIG, LOGGER) > score_series(thick, "t1c", SCORING_CONFIG, LOGGER)

    def test_flair_long_ti_gets_bonus(self):
        long_ti = _series("FLAIR (3D brain)", modality="t2fl", ti=1800.0)
        short_ti = _series("FLAIR (3D brain)", modality="t2fl", ti=1000.0)
        assert score_series(long_ti, "t2fl", SCORING_CONFIG, LOGGER) > score_series(short_ti, "t2fl", SCORING_CONFIG, LOGGER)

    def test_no_signal_returns_neutral_score(self):
        plain = _series("some unrelated text")
        assert score_series(plain, "t1c", SCORING_CONFIG, LOGGER) == 1.0

    def test_missing_resolution_tag_does_not_crash(self):
        no_thickness = _series("CE_T1-TFE (3D brain)", slice_thickness=None)
        assert score_series(no_thickness, "t1c", SCORING_CONFIG, LOGGER) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/stage01/test_series_scoring.py -v`
Expected: FAIL with `AttributeError: module 'reorganize_folders_scoring' has no attribute 'score_series'`

- [ ] **Step 3: Add `score_series()` to `scripts/01_reorganize_folders.py`**

Find the start of the `SeriesDeduplicator` class:

```python
class SeriesDeduplicator:
    """Select best series when multiple exist for same modality."""
```

Insert immediately before it:

```python
def score_series(
    series_info: 'SeriesInfo',
    modality: str,
    scoring_config: Dict,
    logger: logging.Logger,
) -> float:
    """
    Compute a selection score for one candidate series within a modality
    group (KI-027). Combines text keyword weights, a failure-marker
    penalty, a slice-thickness resolution factor, and (FLAIR only) a
    long-TI bonus. Higher is better.

    Returns 1.0 when no signal in scoring_config applies to this series —
    callers must fall back to another tie-break (e.g. slice_count) in
    that case, since 1.0 carries no preference information.
    """
    combined_text = series_info.series_description.lower()
    score = 1.0

    text_weights = scoring_config.get('text_weights', {}).get(modality, {})
    for keyword, weight in text_weights.items():
        if keyword in combined_text:
            score *= weight
            logger.debug(
                f"      [{series_info.original_path.name}] "
                f"weight {weight} for '{keyword}'")

    failure_cfg = scoring_config.get('failure_markers', {})
    failure_keywords = failure_cfg.get('keywords', [])
    if any(kw in combined_text for kw in failure_keywords):
        penalty = failure_cfg.get('penalty', 1.0)
        score *= penalty
        logger.debug(
            f"      [{series_info.original_path.name}] "
            f"failure marker penalty {penalty}")

    res_cfg = scoring_config.get('resolution_scoring', {})
    reference_mm = res_cfg.get('reference_slice_thickness_mm')
    if reference_mm and series_info.slice_thickness_mm:
        min_factor = res_cfg.get('min_factor', 0.5)
        max_factor = res_cfg.get('max_factor', 2.0)
        factor = reference_mm / series_info.slice_thickness_mm
        factor = max(min_factor, min(max_factor, factor))
        score *= factor
        logger.debug(
            f"      [{series_info.original_path.name}] "
            f"resolution factor {factor:.2f}")

    if modality == 't2fl':
        ti_cfg = scoring_config.get('flair_ti_bonus', {})
        threshold = ti_cfg.get('threshold_ms')
        if threshold and series_info.ti_ms and series_info.ti_ms > threshold:
            bonus = ti_cfg.get('bonus', 1.0)
            score *= bonus
            logger.debug(
                f"      [{series_info.original_path.name}] "
                f"long-TI bonus {bonus}")

    return score


class SeriesDeduplicator:
    """Select best series when multiple exist for same modality."""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_series_scoring.py -v`
Expected: 7 passed

- [ ] **Step 5: Verify compilation**

Run: `python -m py_compile scripts/01_reorganize_folders.py`
Expected: no output, exit code 0

- [ ] **Step 6: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_series_scoring.py
git commit -m "feat(stage01): add score_series() protocol-aware scoring function (KI-027)"
```

---

### Task 7: Wire `score_series` into `SeriesDeduplicator` and both pipeline entry points

**Files:**
- Modify: `scripts/01_reorganize_folders.py` (import line, `SeriesDeduplicator.__init__`/`deduplicate_session`/`_select_best_series`, `process_single_patient`, `run_sequential`)
- Test: `tests/stage01/test_series_deduplicator_scoring.py`

**Interfaces:**
- Consumes: `score_series` (Task 6), `load_series_scoring_config` (Task 1).
- Produces: `SeriesDeduplicator.__init__(self, logger, scoring_config: Optional[Dict] = None)`; `SeriesDeduplicator._select_best_series(self, series_list: List[SeriesInfo], modality: str) -> SeriesInfo`.

- [ ] **Step 1: Write the failing test**

Create `tests/stage01/test_series_deduplicator_scoring.py`:

```python
import logging
import sys
import importlib.util
from pathlib import Path

import pytest

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


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_dedup")
SeriesInfo = reorganize_mod.SeriesInfo
SeriesDeduplicator = reorganize_mod.SeriesDeduplicator

sys.path.insert(0, str(PROJ_ROOT))
from utils.config_loader import load_series_scoring_config


def _series(tmp_path, name, modality, n_files=1):
    series_dir = tmp_path / name
    series_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (series_dir / f"IM-{i:04d}.dcm").write_bytes(b"fake")
    return SeriesInfo(
        original_path=series_dir, patient_id="P1", date="20230101",
        modality=modality, series_description=name,
    )


@pytest.fixture
def deduplicator():
    return SeriesDeduplicator(
        logging.getLogger("test_dedup"),
        scoring_config=load_series_scoring_config(),
    )


class TestSelectBestSeries:
    def test_picks_3d_over_mpr_reformat(self, deduplicator, tmp_path):
        threed = _series(tmp_path, "CE_T1-TFE (3D brain)", "t1c")
        mpr = _series(tmp_path, "MPR CE_T1-TFE 2.5mm (cor brain)", "t1c")
        best = deduplicator._select_best_series([threed, mpr], "t1c")
        assert best is threed

    def test_single_candidate_returned_unchanged(self, deduplicator, tmp_path):
        only = _series(tmp_path, "CE_T1-TFE (3D brain)", "t1c")
        best = deduplicator._select_best_series([only], "t1c")
        assert best is only

    def test_falls_back_to_slice_count_when_scores_tie(self, deduplicator, tmp_path):
        a = _series(tmp_path, "unrelated text a", "t1c", n_files=1)
        b = _series(tmp_path, "unrelated text b", "t1c", n_files=2)
        best = deduplicator._select_best_series([a, b], "t1c")
        assert best is b


class TestSeriesDeduplicatorConfigDefault:
    def test_no_scoring_config_does_not_crash(self, tmp_path):
        """A deduplicator built without scoring_config (e.g. ad-hoc script
        usage) must still work, falling back to slice_count for everything."""
        dedup = SeriesDeduplicator(logging.getLogger("test_dedup_noconfig"))
        a = _series(tmp_path, "series a", "t1c", n_files=1)
        b = _series(tmp_path, "series b", "t1c", n_files=3)
        best = dedup._select_best_series([a, b], "t1c")
        assert best is b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/stage01/test_series_deduplicator_scoring.py -v`
Expected: FAIL — `SeriesDeduplicator(...)` raises `TypeError` on `scoring_config` kwarg, and `_select_best_series` still takes 1 positional arg, not 2.

- [ ] **Step 3: Update the import line**

Find:

```python
from utils.config_loader import load_lesion_type_config
```

Replace with:

```python
from utils.config_loader import load_lesion_type_config, load_series_scoring_config
```

- [ ] **Step 4: Update `SeriesDeduplicator.__init__`, `deduplicate_session`, `_select_best_series`**

Find:

```python
class SeriesDeduplicator:
    """Select best series when multiple exist for same modality."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.duplicates_removed = 0
```

Replace with:

```python
class SeriesDeduplicator:
    """Select best series when multiple exist for same modality."""

    def __init__(self, logger: logging.Logger, scoring_config: Optional[Dict] = None):
        self.logger = logger
        self.duplicates_removed = 0
        self.scoring_config = scoring_config or {}
```

Find:

```python
        for modality, series_list in session.series.items():
            if isinstance(series_list, list):
                if len(series_list) > 1:
                    # Multiple series for same modality
                    best_series = self._select_best_series(series_list)
                    deduplicated.series[modality] = best_series
```

Replace with:

```python
        for modality, series_list in session.series.items():
            if isinstance(series_list, list):
                if len(series_list) > 1:
                    # Multiple series for same modality
                    best_series = self._select_best_series(series_list, modality)
                    deduplicated.series[modality] = best_series
```

Find:

```python
    def _select_best_series(self, series_list: List[SeriesInfo]) -> SeriesInfo:
        """Select series with most slices."""
        # Count slices for each series (use rglob to include nested files)
        for series in series_list:
            series.slice_count = len(list(series.original_path.rglob("*.dcm")))

        # Return series with maximum slice count
        return max(series_list, key=lambda s: s.slice_count)
```

Replace with:

```python
    def _select_best_series(self, series_list: List[SeriesInfo], modality: str) -> SeriesInfo:
        """
        Select the best series using protocol-aware scoring (KI-027),
        falling back to slice count when scoring does not differentiate
        candidates (e.g. no scoring_config, or no signal matched).
        """
        # Count slices for each series (use rglob to include nested files) —
        # kept as the tie-break fallback signal.
        for series in series_list:
            series.slice_count = len(list(series.original_path.rglob("*.dcm")))

        scored = [
            (series, score_series(series, modality, self.scoring_config, self.logger))
            for series in series_list
        ]
        max_score = max(score for _, score in scored)
        top_candidates = [series for series, score in scored if score == max_score]

        if len(top_candidates) == 1:
            return top_candidates[0]

        # Tie (including the common case where no scoring signal applies
        # and every candidate scored 1.0) — fall back to slice count.
        return max(top_candidates, key=lambda s: s.slice_count)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_series_deduplicator_scoring.py -v`
Expected: 4 passed

- [ ] **Step 6: Wire config loading into `process_single_patient`**

Find:

```python
        # Initialize fresh components for this process
        modality_detector = ModalityDetector(logger)
        scanner = DatasetScanner(logger)
        grouper = SessionGrouper(logger)
        deduplicator = SeriesDeduplicator(logger)
```

Replace with:

```python
        # Initialize fresh components for this process
        scoring_config = load_series_scoring_config()
        modality_detector = ModalityDetector(logger, scoring_config=scoring_config)
        scanner = DatasetScanner(logger)
        grouper = SessionGrouper(logger)
        deduplicator = SeriesDeduplicator(logger, scoring_config=scoring_config)
```

- [ ] **Step 7: Wire config loading into `run_sequential`**

Find:

```python
    # Initialize components
    modality_detector = ModalityDetector(logger)
    # Use provided id_mapper or create new one
    if id_mapper is None:
        id_mapper = IDMapper()
    scanner = DatasetScanner(logger)
    grouper = SessionGrouper(logger)
    deduplicator = SeriesDeduplicator(logger)
```

Replace with:

```python
    # Initialize components
    scoring_config = load_series_scoring_config()
    modality_detector = ModalityDetector(logger, scoring_config=scoring_config)
    # Use provided id_mapper or create new one
    if id_mapper is None:
        id_mapper = IDMapper()
    scanner = DatasetScanner(logger)
    grouper = SessionGrouper(logger)
    deduplicator = SeriesDeduplicator(logger, scoring_config=scoring_config)
```

- [ ] **Step 8: Run full regression check**

Run: `python -m pytest test_stage01_stage03_fixes.py test_config_loader.py test_orchestrator.py tests/stage01/ -v`
Expected: all passed

- [ ] **Step 9: Verify compilation**

Run: `python -m py_compile scripts/01_reorganize_folders.py utils/config_loader.py`
Expected: no output, exit code 0

- [ ] **Step 10: Commit**

```bash
git add scripts/01_reorganize_folders.py tests/stage01/test_series_deduplicator_scoring.py
git commit -m "feat(stage01): wire score_series into SeriesDeduplicator (KI-027)"
```

---

### Task 8: Integration test against real `data/MS_5/P000915`

**Files:**
- Test: `tests/stage01/test_ms5_integration.py`

**Interfaces:**
- Consumes: `DatasetScanner`, `ModalityDetector`, `SessionGrouper`, `SeriesDeduplicator`, `SeriesInfo`, `MODALITY_BIDS_SUFFIX` from `scripts/01_reorganize_folders.py` (all from prior tasks); `load_series_scoring_config` from `utils.config_loader`.
- Produces: no new code — this task only adds a regression test confirming the whole Stage 01 selection chain behaves correctly against real clinical data with verified real conflicts (see design spec).

- [ ] **Step 1: Write the test**

This task has no "make it fail first" step in the usual TDD sense — Tasks 1–7 already implement everything this test exercises. Write it as a verification/regression test directly.

Create `tests/stage01/test_ms5_integration.py`:

```python
import logging
import sys
import importlib.util
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
DATASET_DIR = PROJ_ROOT / "data" / "MS_5" / "P000915"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))


def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


reorganize_mod = _load_module("01_reorganize_folders.py", "reorganize_folders_integration")

from utils.config_loader import load_series_scoring_config

pytestmark = pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason="data/MS_5/P000915 fixture dataset not present in this checkout",
)


def _build_sessions():
    logger = logging.getLogger("test_ms5_integration")
    scoring_config = load_series_scoring_config()
    scanner = reorganize_mod.DatasetScanner(logger)
    detector = reorganize_mod.ModalityDetector(logger, scoring_config=scoring_config)
    grouper = reorganize_mod.SessionGrouper(logger)
    dedup = reorganize_mod.SeriesDeduplicator(logger, scoring_config=scoring_config)

    series_list = []
    for series_dir, date_folder_name in scanner.scan_patient_series(DATASET_DIR):
        date = scanner.parse_date_from_series_name(date_folder_name) if date_folder_name else None
        if not date:
            date = scanner.parse_date_from_series_name(series_dir.name)
        if not date:
            continue
        modality, desc, tech_meta = detector.detect_modality(series_dir)
        if modality not in reorganize_mod.MODALITY_BIDS_SUFFIX:
            continue
        series_list.append(reorganize_mod.SeriesInfo(
            original_path=series_dir, patient_id=DATASET_DIR.name, date=date,
            modality=modality, series_description=desc,
            slice_thickness_mm=tech_meta.get('slice_thickness_mm'),
            ti_ms=tech_meta.get('ti_ms'),
            has_contrast=tech_meta.get('has_contrast', False),
        ))

    sessions = grouper.group_by_date(series_list)
    return [dedup.deduplicate_session(s) for s in sessions]


class TestMS5RealDataIntegration:
    def test_two_sessions_found(self):
        sessions = _build_sessions()
        assert len(sessions) == 2

    def test_t1c_selects_3d_not_mpr(self):
        # series_dir.name is just the bare numeric folder ID (e.g. "1101");
        # the actual protocol text lives in series_description
        # ("ProtocolName | SeriesDescription"), read from DICOM tags.
        sessions = _build_sessions()
        for session in sessions:
            t1c = session.series.get('t1c')
            assert t1c is not None, f"session {session.date} lost t1c entirely"
            assert 'mpr' not in t1c.series_description.lower()

    def test_t1_selects_3d_not_mpr(self):
        sessions = _build_sessions()
        for session in sessions:
            t1 = session.series.get('t1')
            assert t1 is not None, f"session {session.date} lost t1 entirely"
            assert 'mpr' not in t1.series_description.lower()

    def test_flair_selects_3d_not_mpr(self):
        sessions = _build_sessions()
        for session in sessions:
            flair = session.series.get('t2fl')
            assert flair is not None, f"session {session.date} lost t2fl entirely"
            assert 'mpr' not in flair.series_description.lower()

    def test_no_spine_series_in_any_modality_group(self):
        sessions = _build_sessions()
        for session in sessions:
            for modality, series in session.series.items():
                desc_lower = series.series_description.lower()
                assert 'spine' not in desc_lower
                assert 'cervical' not in desc_lower

    def test_t2_selects_brain_series_not_cspine(self):
        sessions = _build_sessions()
        for session in sessions:
            t2 = session.series.get('t2')
            assert t2 is not None, f"session {session.date} lost t2 entirely"
            assert 'spine' not in t2.series_description.lower()
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/stage01/test_ms5_integration.py -v`
Expected: 6 passed

If `test_t1c_selects_3d_not_mpr` or `test_flair_selects_3d_not_mpr` fail, print the scoring debug log for that session (set `logging.getLogger("test_ms5_integration").setLevel(logging.DEBUG)` and add a `StreamHandler` before calling `_build_sessions()`) to see which keyword weights are winning — this is the dataset where weights were verified by hand in the design spec, so a failure here means a transcription error in `configs/series_scoring.yaml`, not a flaw in the approach.

- [ ] **Step 3: Run the entire Stage 01 test suite together**

Run: `python -m pytest test_stage01_stage03_fixes.py test_config_loader.py test_orchestrator.py tests/stage01/ -v`
Expected: all passed, no `sys.modules` collisions between test files (each uses a distinct `module_name` per Global Constraints)

- [ ] **Step 4: Commit**

```bash
git add tests/stage01/test_ms5_integration.py
git commit -m "test(stage01): add MS_5 real-data integration test for series scoring (KI-027)"
```

---

## Post-plan note

This plan closes KI-027 and KI-001. Update `KNOWN_ISSUES.md` to mark both as ✅ closed (with the merge/branch commit hash once merged) as a follow-up — not part of this plan's tasks, since that edit should reference the actual final commit hash after review, not a hash known in advance.
