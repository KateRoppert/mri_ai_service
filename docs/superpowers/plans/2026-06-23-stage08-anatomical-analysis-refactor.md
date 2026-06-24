# Stage 08 Anatomical Analysis Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the misnamed `08_lobar_localization.py` (which does lobar localization for GBM *and* lesion counting for MS) into a renamed dispatcher (`08_anatomical_analysis.py`) plus focused, independently-testable modules, behind a shared `AnatomicalAnalyzerBase` interface — with zero behavior change.

**Architecture:** Extract `compute_lesion_stats()` into its own module (`scripts/lesion_stats.py`). Introduce `AnatomicalAnalyzerBase` ABC (`scripts/anatomical_analyzer_base.py`) and make `LobarAnalyzer` implement it. Rename the stage script and its config/orchestrator references end-to-end. No new lesion-type branching is added yet — `LobarAnalyzer` remains the only analyzer; the dispatcher hook for a second (MS) analyzer is added in the follow-up plan (McDonald zones).

**Tech Stack:** Python 3.12, pytest, nibabel, numpy, scipy.ndimage, PyYAML.

## Global Constraints

- Zero behavior change: output JSON files (`*_lobar_report.json`, `*_lesion_stats_report.json`), CLI args, and pipeline config semantics stay identical — only names and file layout change.
- Every renamed reference must be updated in the same task that introduces the rename (no broken intermediate state left for a later task to fix).
- Conventional commits (`type(scope): message`), one commit per task minimum.
- Run `python -m py_compile` (or import) on every touched `.py` file before committing — project convention is "code must at least compile" before commit.

---

### Task 1: Extract `compute_lesion_stats` into `scripts/lesion_stats.py`

**Files:**
- Create: `scripts/lesion_stats.py`
- Create: `test_lesion_stats.py` (repo root — matches the existing flat test layout used by `test_stage08_fixes.py`, `test_config_loader.py`, etc. No `tests/` package, no `conftest.py`: the project imports stage scripts via `sys.path.insert` + bare module names, not dotted `scripts.foo` imports — stay consistent with that, see Note below.)
- Modify: `scripts/08_lobar_localization.py:33-89` (remove the function; import from new module instead — done in Task 3 alongside the rename, to avoid a broken intermediate import in this task). **For this task, leave `08_lobar_localization.py` untouched** — just create the new module with an identical copy of the function so both versions exist temporarily. Task 3 removes the duplicate.

**Interfaces:**
- Produces: `lesion_stats.compute_lesion_stats(mask_path: Path) -> tuple[dict, np.ndarray, np.ndarray]` — same signature and return shape as the current inline function (`stats_dict`, `labeled_array` int16, `affine`). Also exports `lesion_stats.MIN_LESION_VOLUME_MM3: float = 5.0`.

**Note on import style:** `scripts/08_anatomical_analysis.py` is executed as a subprocess (`python scripts/08_anatomical_analysis.py ...`, see `pipeline_config.yaml`), so every module inside `scripts/` uses bare absolute imports (`from lobar_analysis import LobarAnalyzer`, `from performance_monitor import ...`) that resolve because the script's own directory is `sys.path[0]` at runtime. Tests replicate this by inserting `scripts/` into `sys.path` and importing the same bare names — **do not** add `scripts/__init__.py` or import via `scripts.lesion_stats`: that creates a second, distinct module object for the same file (one via the bare name, one via the dotted package name), and `issubclass`/`isinstance` checks across the two break silently.

- [ ] **Step 1: Write the failing test**

Create `test_lesion_stats.py` at the repo root:

```python
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from lesion_stats import compute_lesion_stats, MIN_LESION_VOLUME_MM3


def _write_mask(tmp_path: Path, data: np.ndarray, voxel_mm=(1.0, 1.0, 1.0)) -> Path:
    affine = np.diag(list(voxel_mm) + [1.0])
    img = nib.Nifti1Image(data.astype(np.uint8), affine)
    path = tmp_path / "sub-001_ses-001_t1_segmask.nii.gz"
    nib.save(img, str(path))
    return path


def test_counts_two_separate_lesions_above_threshold(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[2:5, 2:5, 2:5] = 1   # 27 voxels = 27 mm3 (1mm voxels) -> kept
    data[10:13, 10:13, 10:13] = 1  # another 27-voxel blob, far away -> kept
    mask_path = _write_mask(tmp_path, data)

    stats, labeled, affine = compute_lesion_stats(mask_path)

    assert stats["lesion_count"] == 2
    assert len(stats["lesion_volumes_cm3"]) == 2
    assert labeled.max() == 2


def test_sub_threshold_blob_excluded_from_count_but_kept_in_total(tmp_path):
    data = np.zeros((20, 20, 20), dtype=np.uint8)
    data[0, 0, 0] = 1  # 1 voxel = 1 mm3, below MIN_LESION_VOLUME_MM3 (5.0)
    data[10:13, 10:13, 10:13] = 1  # 27 mm3, kept
    mask_path = _write_mask(tmp_path, data)

    stats, labeled, affine = compute_lesion_stats(mask_path)

    assert stats["lesion_count"] == 1  # only the 27mm3 blob is "counted"
    total_voxels = int((data > 0).sum())
    assert stats["total_volume_cm3"] == pytest.approx(total_voxels / 1000.0, abs=1e-6)
    assert labeled.max() == 2  # both blobs still get a label in the saved array
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_lesion_stats.py -v`
Expected: `ModuleNotFoundError: No module named 'lesion_stats'` — `scripts/lesion_stats.py` does not exist yet.

- [ ] **Step 3: Write the implementation — `scripts/lesion_stats.py`**

```python
"""
Per-lesion connected-component statistics for binary lesion masks (MS).

Counts individual lesions (connected components), computes per-lesion and
total burden volumes. Used by Stage 08 for multiple_sclerosis masks, where
each connected component corresponds to one clinically distinct lesion
(unlike GBM, which is analyzed as lobar regions of a single tumor mass).
"""

from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label

# Components smaller than this are treated as noise for counting purposes —
# they are excluded from lesion_count, the per-lesion list, and hover lookup.
# They still contribute to total_volume_cm3 (full lesion burden). 5 mm³ matches
# what is visually counted as a discrete lesion (sub-visible 3-4 voxel specks
# are dropped); see KI-037 for the upstream determinism this mitigates.
MIN_LESION_VOLUME_MM3 = 5.0


def compute_lesion_stats(mask_path: Path) -> Tuple[dict, np.ndarray, np.ndarray]:
    """
    Count connected components (individual lesions) in a binary mask.
    Used for MS where each component = one lesion.

    Components below MIN_LESION_VOLUME_MM3 are treated as noise: excluded from
    lesion_count / lesion_volumes_cm3 / lesion_volumes_by_label, but their
    volume is still included in total_volume_cm3 (full burden).

    Returns (stats_dict, labeled_array, affine):
      stats_dict: lesion_count, total_volume_cm3, mean_lesion_volume_cm3,
                  lesion_volumes_cm3 (sorted desc, kept lesions, for display/table),
                  lesion_volumes_by_label ({str(label): volume_cm3}, kept, for hover).
      labeled_array: int array, EVERY component its own integer label 1..N
                     (not filtered — the saved mask keeps all blobs; only the
                     stats/hover map drop sub-threshold ones).
      affine: source affine (to save the labeled mask).
    """
    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj)
    voxel_vol_mm3 = float(np.prod(np.abs(np.diag(img.affine[:3, :3]))))
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
    min_cm3 = MIN_LESION_VOLUME_MM3 / 1000.0

    binary = (data > 0).astype(np.uint8)
    labeled, n_components = ndimage_label(binary)

    all_volumes = {}
    for i in range(1, n_components + 1):
        voxel_count = int(np.sum(labeled == i))
        all_volumes[str(i)] = round(voxel_count * voxel_vol_cm3, 4)

    total = round(sum(all_volumes.values()), 4)  # full burden, unfiltered

    kept_by_label = {lbl: v for lbl, v in all_volumes.items() if v >= min_cm3}
    kept_volumes = sorted(kept_by_label.values(), reverse=True)
    count = len(kept_volumes)
    mean = round(sum(kept_volumes) / count, 4) if count > 0 else 0.0

    stats = {
        "lesion_count": count,
        "total_volume_cm3": total,
        "mean_lesion_volume_cm3": mean,
        "lesion_volumes_cm3": kept_volumes,
        "lesion_volumes_by_label": kept_by_label,
    }
    return stats, labeled.astype(np.int16), img.affine
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_lesion_stats.py -v`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add scripts/lesion_stats.py test_lesion_stats.py
git commit -m "refactor(stage08): extract compute_lesion_stats into scripts/lesion_stats.py"
```

---

### Task 2: Introduce `AnatomicalAnalyzerBase` and make `LobarAnalyzer` implement it

**Files:**
- Create: `scripts/anatomical_analyzer_base.py`
- Test: `test_anatomical_analyzer_base.py` (repo root, same convention as Task 1)
- Modify: `scripts/lobar_analysis.py:18` (class declaration + import)

**Interfaces:**
- Produces: `AnatomicalAnalyzerBase` (ABC) with abstract methods `analyze_mask(self, mask_path: Path) -> Optional[Dict]` and `save_report(self, report: Dict, output_path: Path) -> bool`. `LobarAnalyzer` becomes a concrete subclass — no change to its existing public methods.

- [ ] **Step 1: Write the failing test**

Create `test_anatomical_analyzer_base.py` at the repo root:

```python
import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from anatomical_analyzer_base import AnatomicalAnalyzerBase


def test_cannot_instantiate_base_directly():
    with pytest.raises(TypeError):
        AnatomicalAnalyzerBase()


def test_subclass_missing_methods_cannot_be_instantiated():
    class Incomplete(AnatomicalAnalyzerBase):
        def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
            return None
        # save_report intentionally not implemented

    with pytest.raises(TypeError):
        Incomplete()


def test_complete_subclass_can_be_instantiated():
    class Complete(AnatomicalAnalyzerBase):
        def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
            return {"ok": True}

        def save_report(self, report: Dict, output_path: Path) -> bool:
            return True

    instance = Complete()
    assert instance.analyze_mask(Path("x")) == {"ok": True}


def test_lobar_analyzer_is_an_anatomical_analyzer():
    from lobar_analysis import LobarAnalyzer
    assert issubclass(LobarAnalyzer, AnatomicalAnalyzerBase)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_anatomical_analyzer_base.py -v`
Expected: `ModuleNotFoundError: No module named 'anatomical_analyzer_base'`

- [ ] **Step 3: Write the implementation — `scripts/anatomical_analyzer_base.py`**

```python
"""
Shared interface for per-lesion-type anatomical analyzers run at Stage 08.

Each lesion type gets its own analyzer (LobarAnalyzer for glioblastoma,
MSZoneAnalyzer for multiple_sclerosis) behind this common contract, so the
Stage 08 dispatcher and future lesion types (e.g. brain metastases, Этап 8)
don't need to special-case each analyzer's internals.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional


class AnatomicalAnalyzerBase(ABC):
    """Analyzes anatomical localization of lesions for one lesion type."""

    @abstractmethod
    def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
        """Analyze a single segmentation mask. Returns a report dict, or None on failure."""
        raise NotImplementedError

    @abstractmethod
    def save_report(self, report: Dict, output_path: Path) -> bool:
        """Save a report dict as JSON. Returns True on success."""
        raise NotImplementedError
```

- [ ] **Step 4: Make `LobarAnalyzer` inherit from it**

In `scripts/lobar_analysis.py`, add the import after the existing imports (after line 13, `from typing import Dict, Optional`):

```python
from anatomical_analyzer_base import AnatomicalAnalyzerBase
```

Change line 18 from:

```python
class LobarAnalyzer:
```

to:

```python
class LobarAnalyzer(AnatomicalAnalyzerBase):
```

`LobarAnalyzer` already implements both `analyze_mask` and `save_report` with matching signatures — no other change needed in this file.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_anatomical_analyzer_base.py -v`
Expected: `4 passed`

- [ ] **Step 6: Compile-check the modified file**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/lobar_analysis.py scripts/anatomical_analyzer_base.py`
Expected: no output, exit code 0

- [ ] **Step 7: Commit**

```bash
git add scripts/anatomical_analyzer_base.py test_anatomical_analyzer_base.py scripts/lobar_analysis.py
git commit -m "refactor(stage08): introduce AnatomicalAnalyzerBase, LobarAnalyzer implements it"
```

---

### Task 3: Rename `08_lobar_localization.py` → `08_anatomical_analysis.py`, remove duplicated `compute_lesion_stats`

**Files:**
- Rename: `scripts/08_lobar_localization.py` → `scripts/08_anatomical_analysis.py`
- Modify: `scripts/08_anatomical_analysis.py` (post-rename: imports, docstring, usage string, remove inline `compute_lesion_stats`)
- Modify: `test_stage08_fixes.py:45,113` (load the renamed file)

**Interfaces:**
- Consumes: `lesion_stats.compute_lesion_stats` (Task 1), `AnatomicalAnalyzerBase`-backed `LobarAnalyzer` (Task 2).
- Produces: same CLI contract as before (`08_anatomical_analysis.py <input_dir> <output_dir> --config ... --preprocessing-config ... --lesion-type ...`) — argument names and behavior unchanged.

- [ ] **Step 1: Rename the file with git (preserves history)**

```bash
cd /home/ubuntu/mri_ai_service
git mv scripts/08_lobar_localization.py scripts/08_anatomical_analysis.py
```

- [ ] **Step 2: Update the module docstring and usage string**

In `scripts/08_anatomical_analysis.py`, change lines 1-10 from:

```python
#!/usr/bin/env python3
"""
Lobar Localization: determine anatomical location of lesions by brain lobe.

Overlays segmentation masks with a lobar atlas to compute per-lobe
volumes for each lesion class (NCR, ED, NET, ET).

Usage:
    python 08_lobar_localization.py <input_dir> <output_dir> [options]
"""
```

to:

```python
#!/usr/bin/env python3
"""
Anatomical Analysis: per-lesion-type anatomical localization of lesions.

Dispatches to a lesion-type-specific AnatomicalAnalyzerBase implementation:
- glioblastoma: LobarAnalyzer — overlays the mask with a cortical lobe atlas
  to compute per-lobe volumes for each BraTS class (NCR, ED, NET, ET).
- multiple_sclerosis: per-lesion connected-component stats via lesion_stats
  (McDonald zone classification is added in a follow-up plan).

Usage:
    python 08_anatomical_analysis.py <input_dir> <output_dir> [options]
"""
```

- [ ] **Step 3: Replace the inline `compute_lesion_stats` with an import**

In `scripts/08_anatomical_analysis.py`, remove the full function body that currently sits between the docstring/imports block and `def setup_logging` (originally lines 33-89 in the pre-rename file — the `MIN_LESION_VOLUME_MM3` constant and the entire `def compute_lesion_stats(...)` function). Replace the local import block (originally lines 24-28):

```python
import yaml
import nibabel as nib
import numpy as np
from scipy.ndimage import label as ndimage_label
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from lobar_analysis import LobarAnalyzer
```

with:

```python
import yaml
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from lobar_analysis import LobarAnalyzer
from lesion_stats import compute_lesion_stats
```

(`nibabel`, `numpy`, `scipy.ndimage.label` were only used by the now-removed `compute_lesion_stats` — `process_one_mask` calls `nib.save` further down, so keep checking: it does — see Step 4.)

- [ ] **Step 4: Keep `nibabel` import for `process_one_mask`'s `nib.save` call**

`process_one_mask` (further down in the same file) calls `nib.save(nib.Nifti1Image(labeled, affine), str(labels_path))` — `nibabel` is still needed directly in this file. Restore it in the import block from Step 3, final result:

```python
import yaml
import nibabel as nib
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from lobar_analysis import LobarAnalyzer
from lesion_stats import compute_lesion_stats
```

- [ ] **Step 5: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/08_anatomical_analysis.py`
Expected: no output, exit code 0

- [ ] **Step 6: Update `test_stage08_fixes.py` to load the renamed file**

In `test_stage08_fixes.py`, change line 2 (module docstring) from:

```python
Tests for auto-tune parallelism in stage 08 (08_lobar_localization.py).
```

to:

```python
Tests for auto-tune parallelism in stage 08 (08_anatomical_analysis.py).
```

Change line 45 from:

```python
loc_mod = _load_module("08_lobar_localization.py", "loc08")
```

to:

```python
loc_mod = _load_module("08_anatomical_analysis.py", "loc08")
```

Change line 113 from:

```python
        "08_lobar_localization.py",
```

to:

```python
        "08_anatomical_analysis.py",
```

Also add `"lesion_stats"` to the list of mocked modules near the top of the file (line 31-34), since `08_anatomical_analysis.py` now imports it and the test harness mocks heavy deps the same way it already mocks `lobar_analysis`:

Change:

```python
# lobar_analysis requires nibabel/ants — mock the whole module
if "lobar_analysis" not in sys.modules:
    sys.modules["lobar_analysis"] = MagicMock()
if "ants" not in sys.modules:
    sys.modules["ants"] = MagicMock()
```

to:

```python
# lobar_analysis requires nibabel/ants — mock the whole module
if "lobar_analysis" not in sys.modules:
    sys.modules["lobar_analysis"] = MagicMock()
if "ants" not in sys.modules:
    sys.modules["ants"] = MagicMock()
if "lesion_stats" not in sys.modules:
    sys.modules["lesion_stats"] = MagicMock()
```

- [ ] **Step 7: Run the existing stage 08 test suite**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_stage08_fixes.py -v`
Expected: `9 passed` (same 9 tests as before the rename — `TestAutoTuneParallelismStage08` x5, `TestFindMasksStage08` x4)

- [ ] **Step 8: Run the full new test suite together**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_stage08_fixes.py test_lesion_stats.py test_anatomical_analyzer_base.py -v`
Expected: all pass, no import errors

- [ ] **Step 9: Commit**

```bash
git add scripts/08_anatomical_analysis.py test_stage08_fixes.py
git commit -m "refactor(stage08): rename 08_lobar_localization.py to 08_anatomical_analysis.py"
```

---

### Task 4: Rename the stage key in pipeline config, config loader, and orchestrator

**Files:**
- Modify: `pipeline_config.yaml:136,138`
- Modify: `utils/config_loader.py:112,214,243,267`
- Modify: `orchestrator.py:69`
- Modify: `test_config_loader.py:91,93,114,281` (test fixtures — these `stage_08_lobar_localization` references were added in a prior baseline-fix commit, before this plan was written; they must be renamed in this same task or Step 7's repo-wide grep will fail)

**Interfaces:**
- Produces: stage key `stage_08_anatomical_analysis` (replaces `stage_08_lobar_localization`) used consistently across config, loader, orchestrator, and the config-loader test fixtures.

- [ ] **Step 1: Update `pipeline_config.yaml`**

Change (line 136-138):

```yaml
  stage_08_lobar_localization:
    enabled: true
    script: scripts/08_lobar_localization.py
```

to:

```yaml
  stage_08_anatomical_analysis:
    enabled: true
    script: scripts/08_anatomical_analysis.py
```

- [ ] **Step 2: Update `utils/config_loader.py` — required stages list (line 112)**

Change:

```python
    required_stages = ['stage_01_reorganize', 'stage_02_metadata', 
                       'stage_03_convert', 'stage_04_quality',
                       'stage_05_preprocessing', 'stage_06_segmentation', 
                       'stage_07_inverse_transform', 'stage_08_lobar_localization']
```

to:

```python
    required_stages = ['stage_01_reorganize', 'stage_02_metadata', 
                       'stage_03_convert', 'stage_04_quality',
                       'stage_05_preprocessing', 'stage_06_segmentation', 
                       'stage_07_inverse_transform', 'stage_08_anatomical_analysis']
```

- [ ] **Step 3: Update `utils/config_loader.py` — input mapping (line 214)**

Change:

```python
        'stage_08_lobar_localization': root_output / output_struct['stage_06'],
```

to:

```python
        'stage_08_anatomical_analysis': root_output / output_struct['stage_06'],
```

- [ ] **Step 4: Update `utils/config_loader.py` — output mapping (line 243)**

Change:

```python
        'stage_08_lobar_localization': root_output / output_struct['stage_08'],
```

to:

```python
        'stage_08_anatomical_analysis': root_output / output_struct['stage_08'],
```

- [ ] **Step 5: Update `utils/config_loader.py` — enabled stages order list (line 267)**

Change:

```python
        'stage_07_inverse_transform',
        'stage_08_lobar_localization'
    ]
```

to:

```python
        'stage_07_inverse_transform',
        'stage_08_anatomical_analysis'
    ]
```

- [ ] **Step 6: Update `orchestrator.py` (line 69)**

Change:

```python
    if stage_name in (
        'stage_01_reorganize',
        'stage_04_quality',
        'stage_05_preprocessing',
        'stage_06_segmentation',
        'stage_07_inverse_transform',
        'stage_08_lobar_localization',
    ):
```

to:

```python
    if stage_name in (
        'stage_01_reorganize',
        'stage_04_quality',
        'stage_05_preprocessing',
        'stage_06_segmentation',
        'stage_07_inverse_transform',
        'stage_08_anatomical_analysis',
    ):
```

- [ ] **Step 7: Update `test_config_loader.py` fixtures**

In `test_config_loader.py`, `create_minimal_valid_config()` has a stage entry (line 91-95):

```python
            'stage_08_lobar_localization': {
                'enabled': True,
                'script': 'scripts/08_lobar_localization.py',
                'args': {}
            }
```

Change to:

```python
            'stage_08_anatomical_analysis': {
                'enabled': True,
                'script': 'scripts/08_anatomical_analysis.py',
                'args': {}
            }
```

In `create_script_files()`'s `script_names` list (line 114), change:

```python
        '08_lobar_localization.py'
```

to:

```python
        '08_anatomical_analysis.py'
```

In `test_get_enabled_stages()`'s local `stages` dict (line 281), change:

```python
            'stage_08_lobar_localization': {'enabled': False}
```

to:

```python
            'stage_08_anatomical_analysis': {'enabled': False}
```

- [ ] **Step 8: Confirm no stale references remain anywhere in the repo**

Run: `cd /home/ubuntu/mri_ai_service && grep -rn "stage_08_lobar_localization\|08_lobar_localization" --include="*.py" --include="*.yaml" --include="*.yml" . 2>/dev/null | grep -v venv | grep -v node_modules`
Expected: no output (empty)

- [ ] **Step 9: Run the existing config/orchestrator test suites**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_config_loader.py test_real_config.py test_orchestrator.py -v`
Expected: all pass (these tests validate `pipeline_config.yaml` structure and stage ordering — they must keep passing unchanged since only the *name* of the stage changed, not its position or shape)

- [ ] **Step 10: Commit**

```bash
git add pipeline_config.yaml utils/config_loader.py orchestrator.py test_config_loader.py
git commit -m "refactor(pipeline): rename stage_08_lobar_localization to stage_08_anatomical_analysis"
```

---

### Task 5: Full regression pass

**Files:** none (verification only)

- [ ] **Step 1: Run the entire repo-root test suite**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_stage08_fixes.py test_config_loader.py test_real_config.py test_orchestrator.py test_resampling.py test_lesion_stats.py test_anatomical_analyzer_base.py -v`
Expected: all pass

- [ ] **Step 2: Compile-check every touched file in one pass**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/08_anatomical_analysis.py scripts/lobar_analysis.py scripts/lesion_stats.py scripts/anatomical_analyzer_base.py utils/config_loader.py orchestrator.py`
Expected: no output, exit code 0

- [ ] **Step 3: Smoke-test the renamed CLI entry point's argument parsing**

Run: `cd /home/ubuntu/mri_ai_service && python scripts/08_anatomical_analysis.py --help`
Expected: argparse help text prints, listing `input_dir`, `output_dir`, `--config`, `--preprocessing-config`, `--lesion-type {glioblastoma,multiple_sclerosis}`, with no import errors.

- [ ] **Step 4: Push**

```bash
git push
```

---

## Self-Review Notes (for the plan author, already applied above)

- **Spec coverage:** This plan covers only the "Architecture: разделение Stage 08" section of the design spec. McDonald zones (A) and lesion diff (B) are deliberately out of scope — they are separate follow-up plans per the spec's own module boundaries (`ms_localization.py`, `backend/lesion_diff.py`), and depend on this rename landing first (both reference `08_anatomical_analysis.py` / `AnatomicalAnalyzerBase` by name).
- **Type consistency:** `compute_lesion_stats` signature (`Path -> Tuple[dict, np.ndarray, np.ndarray]`) is identical in Task 1's extraction and Task 3's import — verified against the original inline function's `return stats, labeled.astype(np.int16), img.affine`.
- **No placeholders:** every step shows literal before/after code or an exact command with expected output.
