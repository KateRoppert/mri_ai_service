# MS McDonald Zone Classification — Pipeline Plan (2a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For multiple_sclerosis masks, classify each lesion (connected component) into a McDonald zone — periventricular, juxtacortical, infratentorial, or deep_white_matter — and emit `*_mcdonald_report.json` from Stage 08. Spinal cord is explicitly unsupported and surfaced as such, not silently dropped.

**Architecture:** One-time data-prep (`fetch_ms_zone_atlases.sh` + `register_ms_zone_atlases.py`) produces binary ventricle/infratentorial atlas masks per template (SRI24, MNI152_FSL, MNI152_ICBM), committed to `data/templates/ms_zones/`. A new `MSZoneAnalyzer(AnatomicalAnalyzerBase)` overlays these (plus the existing cortical atlas, reused for "juxtacortical") against each lesion's connected component. `08_anatomical_analysis.py`'s dispatcher picks `MSZoneAnalyzer` vs `LobarAnalyzer` by `--lesion-type`. This plan is pipeline/backend-only — the API endpoint and frontend rendering are plan 2b, which depends on this plan's `*_mcdonald_report.json` schema.

**Tech Stack:** Python 3.12, ANTsPy, nibabel, numpy, scipy.ndimage, PyYAML, pytest.

**Depends on:** `docs/superpowers/plans/2026-06-23-stage08-anatomical-analysis-refactor.md` (must be executed first — this plan references `scripts/08_anatomical_analysis.py`, `scripts/lesion_stats.py`, and `AnatomicalAnalyzerBase` by the names that plan creates).

## Global Constraints

- Spinal cord zone is unsupported in this iteration — must appear in any per-zone enumeration as an explicit "unsupported" entry, never silently omitted (per design spec).
- Zero behavior change to glioblastoma processing — `LobarAnalyzer` and `*_lobar_report.json` stay exactly as they are.
- "Touching" a zone = non-zero overlap after binary dilation by a configurable number of voxels (default 1) — accounts for registration jitter, per approved design spec.
- Conventional commits, one commit per task minimum.
- Every touched `.py` file must pass `python -m py_compile` before commit.

---

### Task 1: Acquire raw atlases from the Docker image and verify label indices

This task is operational (Docker + manual inspection), not unit-testable. It produces the raw inputs Task 3's registration script consumes.

**Files:**
- Create: `scripts/fetch_ms_zone_atlases.sh`

- [ ] **Step 1: Write the fetch script**

```bash
#!/usr/bin/env bash
# One-time: extract FSL standard-space atlases needed for MS McDonald zone
# classification from the project's Docker image (which bundles FSL).
# Run once; output is committed to data/templates/ms_zones_raw/.
set -euo pipefail

OUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/templates/ms_zones_raw"
mkdir -p "$OUT_DIR"

IMAGE="kateroppert/mri-ai-service:latest"
FSL_ATLASES="/usr/local/fsl/data/atlases"

echo "Extracting atlases from $IMAGE into $OUT_DIR ..."

docker run --rm -v "$OUT_DIR:/out" "$IMAGE" bash -c "
  cp '$FSL_ATLASES/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz' /out/ &&
  cp '$FSL_ATLASES/HarvardOxford-Subcortical.xml' /out/ &&
  cp '$FSL_ATLASES/Cerebellum/Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz' /out/ &&
  cp '$FSL_ATLASES/Cerebellum_MNIflirt.xml' /out/
"

echo "Done. Files in $OUT_DIR:"
ls -la "$OUT_DIR"
```

- [ ] **Step 2: Run it**

Run: `bash scripts/fetch_ms_zone_atlases.sh`
Expected: `data/templates/ms_zones_raw/` contains `HarvardOxford-sub-maxprob-thr25-1mm.nii.gz`, `HarvardOxford-Subcortical.xml`, `Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz`, `Cerebellum_MNIflirt.xml`.

**If the FSL atlas paths inside the image differ from the ones hardcoded above** (image layout may differ from a bare FSL install — check with `docker run --rm kateroppert/mri-ai-service:latest find /usr/local/fsl -iname "HarvardOxford-sub*"` first if the copy fails), adjust `FSL_ATLASES` and the file names accordingly before re-running.

- [ ] **Step 3: Verify the actual label indices (do not trust hardcoded assumptions)**

FSL atlas label indices for `HarvardOxford-sub` have varied slightly across FSL releases (some versions include "Cerebral White Matter"/"Cerebral Cortex" as labels 1-2 and 12-13, shifting everything else). Verify the indices actually present in this image's copy before writing them into code:

Run:
```bash
grep -i "ventricle\|brain-stem\|brainstem" data/templates/ms_zones_raw/HarvardOxford-Subcortical.xml
```
Expected output: lines like `<label index="2" ...>Left Lateral Ventricle</label>`, showing the exact `index` values for "Left Lateral Ventricle", "Right Lateral Ventricle", and "Brain-Stem". **Note the `index` value down** — it is the 0-indexed label number; the corresponding voxel value in the maxprob NIfTI is `index + 1` (FSL atlas XML convention: `index="0"` is the first non-background label, which is voxel value `1`).

Run: `grep -i "label" data/templates/ms_zones_raw/Cerebellum_MNIflirt.xml | head -5`
Expected: confirms the Cerebellum-MNIflirt atlas's labels are cerebellar lobules (any non-zero voxel = cerebellum) — there is no "exclude this" case, every label in this atlas is part of the infratentorial zone.

Write the three resolved voxel values (lateral ventricle ×2, brain-stem ×1) into `scripts/register_ms_zone_atlases.py` in Task 3, Step 1 — replacing the `# VERIFIED:` placeholders with the real numbers from this step's output.

- [ ] **Step 4: Commit**

```bash
git add scripts/fetch_ms_zone_atlases.sh
git commit -m "chore(ms-zones): add script to fetch raw FSL atlases for McDonald zones"
```

(`data/templates/ms_zones_raw/*` is a local intermediate, not committed — it's large and reproducible by re-running the script. Add it to `.gitignore` in this commit too: append `data/templates/ms_zones_raw/` to `.gitignore`.)

---

### Task 2: Extract shared atlas-resampling utility (DRY refactor, behavior-preserving)

`LobarAnalyzer._resample_atlas_to_mask` (nearest-neighbor resample of a standard-space atlas onto a subject's mask grid) will be needed by `MSZoneAnalyzer` too. Extract it once instead of duplicating.

**Files:**
- Create: `scripts/atlas_resample.py`
- Test: `test_atlas_resample.py` (repo root)
- Modify: `scripts/lobar_analysis.py:64-102` (`_resample_atlas_to_mask` becomes a thin wrapper)

**Interfaces:**
- Produces: `atlas_resample.resample_to_grid(atlas_path: Path, target_affine: np.ndarray, target_shape: tuple) -> Optional[np.ndarray]` — nearest-neighbor resampled int array, or `None` on failure (logs the error, does not raise).

- [ ] **Step 1: Write the failing test**

Create `test_atlas_resample.py`:

```python
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from atlas_resample import resample_to_grid


def test_resamples_identical_grid_unchanged(tmp_path):
    data = np.zeros((10, 10, 10), dtype=np.int16)
    data[3:6, 3:6, 3:6] = 7
    affine = np.eye(4)
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), str(atlas_path))

    result = resample_to_grid(atlas_path, target_affine=affine, target_shape=(10, 10, 10))

    assert result is not None
    assert result.shape == (10, 10, 10)
    assert result[4, 4, 4] == 7
    assert result[0, 0, 0] == 0


def test_resamples_finer_grid_to_coarser_target(tmp_path):
    # Atlas at 0.5mm voxels, target grid at 1mm voxels covering the same
    # physical space — a coarser target should still pick up the labeled region.
    data = np.zeros((20, 20, 20), dtype=np.int16)
    data[6:12, 6:12, 6:12] = 3
    atlas_affine = np.diag([0.5, 0.5, 0.5, 1.0])
    atlas_path = tmp_path / "atlas_fine.nii.gz"
    nib.save(nib.Nifti1Image(data, atlas_affine), str(atlas_path))

    target_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    result = resample_to_grid(atlas_path, target_affine=target_affine, target_shape=(10, 10, 10))

    assert result is not None
    assert result.shape == (10, 10, 10)
    # Physical center of the labeled region (~ voxel index 4-5 in 1mm space) is labeled.
    assert result[4, 4, 4] == 3


def test_returns_none_on_missing_atlas(tmp_path):
    missing = tmp_path / "does_not_exist.nii.gz"
    result = resample_to_grid(missing, target_affine=np.eye(4), target_shape=(5, 5, 5))
    assert result is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_atlas_resample.py -v`
Expected: `ModuleNotFoundError: No module named 'atlas_resample'`

- [ ] **Step 3: Write the implementation — `scripts/atlas_resample.py`**

```python
"""
Shared nearest-neighbor resampling of a standard-space label atlas onto a
target voxel grid. Used by LobarAnalyzer (cortical lobes) and MSZoneAnalyzer
(McDonald zones) — both overlay a fixed standard-space atlas with a subject
mask that may be on a slightly different grid (resolution/shape mismatch
after independent registration runs).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform

logger = logging.getLogger(__name__)


def resample_to_grid(
    atlas_path: Path,
    target_affine: np.ndarray,
    target_shape: Tuple[int, int, int],
) -> Optional[np.ndarray]:
    """
    Nearest-neighbor resample a standard-space atlas NIfTI onto a target grid.

    Args:
        atlas_path: path to the standard-space label atlas NIfTI.
        target_affine: affine of the grid to resample onto (e.g. a subject mask's).
        target_shape: shape of the grid to resample onto.

    Returns:
        Resampled int array of `target_shape`, or None on failure.
    """
    try:
        atlas_nii = nib.load(str(atlas_path))
        atlas_data = np.asarray(atlas_nii.dataobj).astype(np.float64)
        atlas_affine = atlas_nii.affine

        # Combined: atlas_voxel = inv(atlas_affine) @ target_affine @ target_voxel
        transform = np.linalg.inv(atlas_affine) @ target_affine
        matrix = transform[:3, :3]
        offset = transform[:3, 3]

        resampled = affine_transform(
            atlas_data,
            matrix,
            offset=offset,
            output_shape=target_shape,
            order=0,  # nearest-neighbor
            mode='constant',
            cval=0,
        )
        return resampled.astype(int)

    except Exception as e:
        logger.error(f"Failed to resample atlas {atlas_path}: {e}")
        return None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_atlas_resample.py -v`
Expected: `3 passed`

- [ ] **Step 5: Make `LobarAnalyzer` use the shared utility**

In `scripts/lobar_analysis.py`, add the import after line 13 (`from typing import Dict, Optional`):

```python
from atlas_resample import resample_to_grid
```

Replace the body of `_resample_atlas_to_mask` (lines 64-102) — keep the method as a thin wrapper so the existing call site (`self._resample_atlas_to_mask(mask_nii)` inside `analyze_mask`) needs no change:

```python
    def _resample_atlas_to_mask(self, mask_nii) -> Optional[np.ndarray]:
        """Resample atlas to match mask grid using nearest-neighbor."""
        resampled = resample_to_grid(
            self.atlas_path,
            target_affine=mask_nii.affine,
            target_shape=mask_nii.shape[:3],
        )
        if resampled is not None:
            logger.info(f"  Resampled atlas: {resampled.shape}, "
                        f"non-zero: {(resampled > 0).sum()}")
        return resampled
```

You can now delete the now-unused `from scipy.ndimage import affine_transform` local import inside the old method body (it was a local `from scipy.ndimage import affine_transform` import statement inside the function, not a module-level one — removing the function body removes it automatically since it was scoped there).

- [ ] **Step 6: Run the full existing lobar/stage08 test suite to confirm zero behavior change**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_stage08_fixes.py test_lesion_stats.py test_anatomical_analyzer_base.py test_atlas_resample.py -v`
Expected: all pass

- [ ] **Step 7: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/atlas_resample.py scripts/lobar_analysis.py`
Expected: no output, exit code 0

- [ ] **Step 8: Commit**

```bash
git add scripts/atlas_resample.py test_atlas_resample.py scripts/lobar_analysis.py
git commit -m "refactor(stage08): extract resample_to_grid shared by LobarAnalyzer and MSZoneAnalyzer"
```

---

### Task 3: One-time registration script — produce binary ventricle/infratentorial atlases per template

**Files:**
- Create: `scripts/register_ms_zone_atlases.py`
- Test: `test_register_ms_zone_atlases.py` (unit tests the binarization logic only — the ANTs registration calls are integration-level and require real ANTs + real atlas files, exercised manually in Step 5, not in CI)

**Interfaces:**
- Produces (on disk, committed): `data/templates/ms_zones/ventricles_{SRI24,MNI152_FSL,MNI152_ICBM}.nii.gz` and `data/templates/ms_zones/infratentorial_{SRI24,MNI152_FSL,MNI152_ICBM}.nii.gz` — each a binary (0/1) int16 NIfTI in the matching template's grid.
- Produces (importable): `register_ms_zone_atlases.binarize_labels(data: np.ndarray, label_values: set[int]) -> np.ndarray` — the pure logic under test.

- [ ] **Step 0: Fix a latent bug in `scripts/preprocessing_steps/registration.py::apply_transform` before relying on it**

This script (Step 3 below) is the first caller anywhere in the codebase to invoke `apply_transform(..., interpolation="NearestNeighbor")`. The function's only other call site (`registration.py:525`, inside `register_modalities`) never passes `interpolation` at all, relying on the default `"Linear"`. Reading the function body (`scripts/preprocessing_steps/registration.py:252-259`):

```python
        interp_map = {
            "Linear": 1,
            "NearestNeighbor": 0,
            "BSpline": 3,
            "Gaussian": 4
        }
        interp_type = interp_map.get(interpolation, 1)
```

`interp_type` is computed but never used — the actual call passes the raw string instead:

```python
        warped = ants.apply_transforms(
            fixed=fixed,
            moving=moving,
            transformlist=[str(transform_path)],
            interpolator=interpolation.lower(),
            whichtoinvert=[False]
        )
```

`"NearestNeighbor".lower()` produces `"nearestneighbor"`, which ANTsPy's `apply_transforms` does not recognize (it expects exact camelCase `"nearestNeighbor"` — confirmed by the *other* function in this same file, `inverse_transform_mask` at line 581, which hardcodes `interpolator="nearestNeighbor"` directly for exactly this reason). Left as-is, Step 5 below would either error or — worse — silently fall back to a default and linearly interpolate a binary label mask into fractional values.

Fix it in `scripts/preprocessing_steps/registration.py`, changing line 266 from:

```python
            interpolator=interpolation.lower(),
```

to:

```python
            interpolator=interpolation,
```

This is safe for the existing call site: `register_modalities` never passes `interpolation`, so it still gets the default `"Linear"` string unchanged (`"Linear"` is also what `ants.apply_transforms` expects verbatim — case-sensitive, capital L). Confirm no other call sites exist:

Run: `cd /home/ubuntu/mri_ai_service && grep -rn "apply_transform(" --include="*.py" . 2>/dev/null | grep -v venv`
Expected: only the function definition (`registration.py:225`) and the one call site (`registration.py:525`) — both already accounted for above.

Compile-check: `python -m py_compile scripts/preprocessing_steps/registration.py` → no output.

Commit this fix on its own before continuing:

```bash
git add scripts/preprocessing_steps/registration.py
git commit -m "fix(registration): apply_transform must not lowercase the ANTs interpolator string"
```

- [ ] **Step 1: Write the failing test for the binarization helper**

Create `test_register_ms_zone_atlases.py`:

```python
import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from register_ms_zone_atlases import binarize_labels


def test_binarize_keeps_only_listed_labels():
    data = np.array([0, 1, 2, 3, 4, 5])
    result = binarize_labels(data, {2, 4})
    assert list(result) == [0, 0, 1, 0, 1, 0]


def test_binarize_returns_int_array():
    data = np.array([[0, 6], [6, 0]])
    result = binarize_labels(data, {6})
    assert result.dtype.kind in ("i", "u")
    assert result.tolist() == [[0, 1], [1, 0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_register_ms_zone_atlases.py -v`
Expected: `ModuleNotFoundError: No module named 'register_ms_zone_atlases'`

- [ ] **Step 3: Write the implementation — `scripts/register_ms_zone_atlases.py`**

Replace the two `# VERIFIED:` integers below with the real voxel values found in Task 1 Step 3's `grep` output before running this script for real.

```python
#!/usr/bin/env python3
"""
One-time data-prep: register MS McDonald-zone atlases (lateral ventricles,
brainstem+cerebellum) from native FSL MNI152 space into each template listed
in lobar_atlas_config.yaml (SRI24, MNI152_FSL, MNI152_ICBM).

Prerequisite: run scripts/fetch_ms_zone_atlases.sh first to populate
data/templates/ms_zones_raw/.

Output (committed to git, same convention as the existing cortical atlas):
  data/templates/ms_zones/ventricles_<TEMPLATE>.nii.gz
  data/templates/ms_zones/infratentorial_<TEMPLATE>.nii.gz

Usage:
    python scripts/register_ms_zone_atlases.py
"""

import logging
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "preprocessing_steps"))
from registration import register_to_atlas, apply_transform  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "templates" / "ms_zones_raw"
OUT_DIR = PROJECT_ROOT / "data" / "templates" / "ms_zones"
WORK_DIR = PROJECT_ROOT / "data" / "templates" / "ms_zones_work"  # intermediates, not committed

# Voxel values in HarvardOxford-sub-maxprob-thr25-1mm.nii.gz.
# VERIFIED: replace with the real "index"+1 values from
# `grep -i "ventricle" data/templates/ms_zones_raw/HarvardOxford-Subcortical.xml`
# (Task 1, Step 3 of the implementation plan) before running this script.
LEFT_LATERAL_VENTRICLE = 2   # VERIFIED: placeholder, confirm against XML
RIGHT_LATERAL_VENTRICLE = 13  # VERIFIED: placeholder, confirm against XML
BRAIN_STEM = 7                # VERIFIED: placeholder, confirm against XML

# Source template the raw atlases are natively aligned to (FSL's own MNI152).
SOURCE_TEMPLATE_T1 = PROJECT_ROOT / "data" / "templates" / "MNI152_T1_1mm.nii.gz"

# Target templates: name -> (template T1 path, needs_registration)
# MNI152_FSL is the atlas's native space (SOURCE_TEMPLATE_T1 itself) — no
# registration needed, just binarize and copy.
TARGETS = {
    "MNI152_FSL": (PROJECT_ROOT / "data" / "templates" / "MNI152_T1_1mm.nii.gz", False),
    "SRI24": (PROJECT_ROOT / "data" / "templates" / "sri24_t1.nii.gz", True),
    "MNI152_ICBM": (PROJECT_ROOT / "data" / "templates" / "mni_icbm152_t1_tal_nlin_sym_09a.nii", True),
}


def binarize_labels(data: np.ndarray, label_values: set) -> np.ndarray:
    """Return a 0/1 int array: 1 where `data` matches any value in `label_values`."""
    return np.isin(data, list(label_values)).astype(np.int16)


def build_zone_masks() -> tuple:
    """Binarize the raw HarvardOxford-sub + Cerebellum atlases in native MNI152_FSL space."""
    sub_img = nib.load(str(RAW_DIR / "HarvardOxford-sub-maxprob-thr25-1mm.nii.gz"))
    sub_data = np.asarray(sub_img.dataobj)

    cerebellum_img = nib.load(str(RAW_DIR / "Cerebellum-MNIflirt-maxprob-thr25-1mm.nii.gz"))
    cerebellum_data = np.asarray(cerebellum_img.dataobj)

    ventricle_mask = binarize_labels(sub_data, {LEFT_LATERAL_VENTRICLE, RIGHT_LATERAL_VENTRICLE})
    # Infratentorial = brainstem (from HarvardOxford-sub) OR any cerebellar
    # lobule (every non-zero label in Cerebellum-MNIflirt is cerebellum).
    brainstem_mask = binarize_labels(sub_data, {BRAIN_STEM})
    cerebellum_mask = (cerebellum_data > 0).astype(np.int16)
    infratentorial_mask = np.clip(brainstem_mask + cerebellum_mask, 0, 1).astype(np.int16)

    return ventricle_mask, infratentorial_mask, sub_img.affine


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    ventricle_mask, infratentorial_mask, native_affine = build_zone_masks()
    ventricle_path = WORK_DIR / "ventricles_native.nii.gz"
    infratentorial_path = WORK_DIR / "infratentorial_native.nii.gz"
    nib.save(nib.Nifti1Image(ventricle_mask, native_affine), str(ventricle_path))
    nib.save(nib.Nifti1Image(infratentorial_mask, native_affine), str(infratentorial_path))
    logger.info(f"Native-space binary masks written to {WORK_DIR}")

    for template_name, (template_t1, needs_registration) in TARGETS.items():
        if not needs_registration:
            for src, label in [(ventricle_path, "ventricles"), (infratentorial_path, "infratentorial")]:
                dest = OUT_DIR / f"{label}_{template_name}.nii.gz"
                dest.write_bytes(src.read_bytes())
                logger.info(f"{template_name}: copied {label} (native space) -> {dest}")
            continue

        transform_path = WORK_DIR / f"mni152fsl_to_{template_name}_transform.mat"
        warped_t1_path = WORK_DIR / f"mni152fsl_to_{template_name}_t1.nii.gz"

        logger.info(f"Registering MNI152_FSL T1 -> {template_name} ...")
        result = register_to_atlas(
            moving_path=SOURCE_TEMPLATE_T1,
            atlas_path=template_t1,
            output_path=warped_t1_path,
            transform_path=transform_path,
            registration_type="Affine",  # two different population templates -> allow scale/shear
        )
        if not result["success"]:
            raise RuntimeError(f"Registration to {template_name} failed: {result.get('error')}")

        for src, label in [(ventricle_path, "ventricles"), (infratentorial_path, "infratentorial")]:
            dest = OUT_DIR / f"{label}_{template_name}.nii.gz"
            apply_result = apply_transform(
                moving_path=src,
                fixed_path=template_t1,
                transform_path=transform_path,
                output_path=dest,
                interpolation="NearestNeighbor",
            )
            if not apply_result["success"]:
                raise RuntimeError(f"Applying transform for {label}/{template_name} failed: {apply_result.get('error')}")
            logger.info(f"{template_name}: registered {label} -> {dest}")

    logger.info(f"Done. Zone atlases written to {OUT_DIR}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_register_ms_zone_atlases.py -v`
Expected: `2 passed`

- [ ] **Step 5: Run the full script manually (requires ANTsPy and Task 1's raw atlases)**

First, fill in the real `LEFT_LATERAL_VENTRICLE`, `RIGHT_LATERAL_VENTRICLE`, `BRAIN_STEM` values from Task 1 Step 3's verification before running this.

Run: `cd /home/ubuntu/mri_ai_service && python scripts/register_ms_zone_atlases.py`
Expected: log lines for each of the 3 templates ending in "Done. Zone atlases written to .../data/templates/ms_zones". Inspect the output with `python -c "import nibabel as nib; import numpy as np; img = nib.load('data/templates/ms_zones/ventricles_SRI24.nii.gz'); print(img.shape, np.unique(np.asarray(img.dataobj)))"` — expect shape matching `data/templates/sri24_t1.nii.gz` and unique values `[0 1]`, with a non-trivial number of `1` voxels (ventricles are a small but non-empty structure — sanity-check with `.sum()` being in the low thousands of voxels at 1mm³, not 0 and not most of the brain).

- [ ] **Step 6: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/register_ms_zone_atlases.py`
Expected: no output, exit code 0

- [ ] **Step 7: Commit**

```bash
git add scripts/register_ms_zone_atlases.py test_register_ms_zone_atlases.py data/templates/ms_zones/ .gitignore
git commit -m "feat(ms-zones): add one-time registration script + committed zone atlases"
```

(Add `data/templates/ms_zones_work/` to `.gitignore` alongside `ms_zones_raw/` in this commit — intermediates, not the final committed atlases in `data/templates/ms_zones/`.)

---

### Task 4: `MSZoneAnalyzer` — per-lesion zone classification

**Files:**
- Create: `configs/ms_zones_config.yaml`
- Create: `scripts/ms_localization.py`
- Test: `test_ms_localization.py` (repo root)

**Interfaces:**
- Consumes: `atlas_resample.resample_to_grid` (Task 2), `lesion_stats.compute_lesion_stats` (Stage 08 refactor plan, Task 1).
- Produces: `MSZoneAnalyzer(AnatomicalAnalyzerBase)` with `MSZoneAnalyzer.REPORT_SUFFIX = "_mcdonald_report.json"` and `analyze_mask(mask_path: Path) -> Optional[Dict]` returning:
  ```python
  {
    "mask_file": str,
    "total_lesion_count": int,
    "zones": {
      "periventricular":  {"lesion_count": int, "total_volume_cm3": float},
      "juxtacortical":     {"lesion_count": int, "total_volume_cm3": float},
      "infratentorial":    {"lesion_count": int, "total_volume_cm3": float},
      "deep_white_matter": {"lesion_count": int, "total_volume_cm3": float},
      "spinal_cord":        {"supported": False},
    },
    "lesion_zones_by_label": {"1": "periventricular", "2": "deep_white_matter", ...},
  }
  ```

- [ ] **Step 1: Write `configs/ms_zones_config.yaml`**

```yaml
# MS McDonald zone classification config (Этап 5.6).
# Cortex atlas for "juxtacortical" is reused from lobar_atlas_config.yaml
# (same per-template cortical atlas already used for GBM lobar localization)
# and is NOT duplicated here — the dispatcher passes its resolved path in.

templates:
  SRI24:
    ventricle_atlas: "data/templates/ms_zones/ventricles_SRI24.nii.gz"
    infratentorial_atlas: "data/templates/ms_zones/infratentorial_SRI24.nii.gz"
  MNI152_FSL:
    ventricle_atlas: "data/templates/ms_zones/ventricles_MNI152_FSL.nii.gz"
    infratentorial_atlas: "data/templates/ms_zones/infratentorial_MNI152_FSL.nii.gz"
  MNI152_ICBM:
    ventricle_atlas: "data/templates/ms_zones/ventricles_MNI152_ICBM.nii.gz"
    infratentorial_atlas: "data/templates/ms_zones/infratentorial_MNI152_ICBM.nii.gz"

# Voxel dilation applied to each zone atlas before testing lesion overlap —
# tolerates registration jitter. Same value used for inter-session lesion
# matching in the lesion-diff plan, kept here as its own knob since the two
# features can be tuned independently.
dilation_voxels: 1
```

- [ ] **Step 2: Write the failing test**

Create `test_ms_localization.py`:

```python
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from ms_localization import MSZoneAnalyzer


def _save(path: Path, data: np.ndarray, affine=None):
    affine = affine if affine is not None else np.eye(4)
    nib.save(nib.Nifti1Image(data.astype(np.int16), affine), str(path))


@pytest.fixture
def zone_atlases(tmp_path):
    """20x20x20 1mm grid. Ventricle = a 2x2x2 block at [8:12,8:12,8:12].
    Cortex = a full-plane shell at x=0 and x=19. Infratentorial = a 2x2x2
    block at [4:6,4:6,4:6]. All three are placed with at least a 2-voxel
    gap between them (and between infratentorial and the cortex planes),
    so that with dilation_voxels=1 (1-voxel dilation on each zone) no two
    zones overlap by accident — each test's lesion placement is what
    decides which single zone gets touched."""
    shape = (20, 20, 20)

    ventricle = np.zeros(shape, dtype=np.int16)
    ventricle[8:12, 8:12, 8:12] = 1
    ventricle_path = tmp_path / "ventricles.nii.gz"
    _save(ventricle_path, ventricle)

    cortex = np.zeros(shape, dtype=np.int16)
    cortex[0, :, :] = 1
    cortex[19, :, :] = 1
    cortex_path = tmp_path / "cortex.nii.gz"
    _save(cortex_path, cortex)

    infratentorial = np.zeros(shape, dtype=np.int16)
    infratentorial[4:6, 4:6, 4:6] = 1
    infratentorial_path = tmp_path / "infratentorial.nii.gz"
    _save(infratentorial_path, infratentorial)

    return {"ventricle": ventricle_path, "cortex": cortex_path, "infratentorial": infratentorial_path}


def test_lesion_touching_ventricle_is_periventricular(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[9:11, 9:11, 7:9] = 1  # touches the ventricle block at z=8
    mask_path = tmp_path / "sub-001_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["periventricular"]["lesion_count"] == 1
    assert report["zones"]["juxtacortical"]["lesion_count"] == 0
    assert report["zones"]["spinal_cord"] == {"supported": False}


def test_lesion_touching_cortex_is_juxtacortical(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[0:2, 5:7, 5:7] = 1  # touches cortex shell at x=0
    mask_path = tmp_path / "sub-002_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["juxtacortical"]["lesion_count"] == 1
    assert report["zones"]["periventricular"]["lesion_count"] == 0


def test_lesion_in_infratentorial_zone(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[4:6, 4:6, 4:6] = 1  # exactly the infratentorial block
    mask_path = tmp_path / "sub-003_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["infratentorial"]["lesion_count"] == 1


def test_lesion_touching_nothing_is_deep_white_matter(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[14:16, 14:16, 14:16] = 1  # isolated, far from all three zones
    mask_path = tmp_path / "sub-004_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["deep_white_matter"]["lesion_count"] == 1


def test_periventricular_takes_priority_over_juxtacortical(tmp_path):
    # Dedicated atlases (not the shared `zone_atlases` fixture) so the
    # ventricle and cortex zones are close enough for one lesion to touch
    # both directly — this is what actually exercises the priority order.
    shape = (20, 20, 20)

    ventricle = np.zeros(shape, dtype=np.int16)
    ventricle[5:7, 5:7, 5:7] = 1  # z in [5,7)
    ventricle_path = tmp_path / "ventricles.nii.gz"
    _save(ventricle_path, ventricle)

    cortex = np.zeros(shape, dtype=np.int16)
    cortex[5:7, 5:7, 8:10] = 1  # z in [8,10) — adjacent gap, not overlapping ventricle
    cortex_path = tmp_path / "cortex.nii.gz"
    _save(cortex_path, cortex)

    infratentorial = np.zeros(shape, dtype=np.int16)
    infratentorial[0:2, 0:2, 0:2] = 1  # far from everything
    infratentorial_path = tmp_path / "infratentorial.nii.gz"
    _save(infratentorial_path, infratentorial)

    # Lesion spans z=[6,9): touches ventricle at z=6 AND cortex at z=8.
    mask = np.zeros(shape, dtype=np.uint8)
    mask[5:7, 5:7, 6:9] = 1
    mask_path = tmp_path / "sub-005_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=ventricle_path,
        cortex_atlas_path=cortex_path,
        infratentorial_atlas_path=infratentorial_path,
        dilation_voxels=0,  # isolate the priority logic from dilation tolerance
    )
    report = analyzer.analyze_mask(mask_path)

    assert report["zones"]["periventricular"]["lesion_count"] == 1
    assert report["zones"]["juxtacortical"]["lesion_count"] == 0


def test_save_report_writes_json(tmp_path, zone_atlases):
    mask = np.zeros((20, 20, 20), dtype=np.uint8)
    mask[14:16, 14:16, 14:16] = 1
    mask_path = tmp_path / "sub-006_ses-001_t1_segmask.nii.gz"
    _save(mask_path, mask)

    analyzer = MSZoneAnalyzer(
        ventricle_atlas_path=zone_atlases["ventricle"],
        cortex_atlas_path=zone_atlases["cortex"],
        infratentorial_atlas_path=zone_atlases["infratentorial"],
        dilation_voxels=1,
    )
    report = analyzer.analyze_mask(mask_path)
    out_path = tmp_path / "out" / "report.json"

    assert analyzer.save_report(report, out_path) is True
    assert out_path.exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_ms_localization.py -v`
Expected: `ModuleNotFoundError: No module named 'ms_localization'`

- [ ] **Step 4: Write the implementation — `scripts/ms_localization.py`**

```python
"""
McDonald zone classification for multiple_sclerosis lesions.

Classifies each connected-component lesion into one McDonald 2017 zone —
periventricular, juxtacortical, or infratentorial — falling back to
deep_white_matter when a lesion touches none of those, with a fixed
hierarchy (periventricular > juxtacortical > infratentorial) matching
McDonald's clinical priority order. Spinal cord is not supported (no
spine-registration infrastructure in this pipeline) and is surfaced in the
report as explicitly unsupported, not silently omitted.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, label as ndimage_label

from anatomical_analyzer_base import AnatomicalAnalyzerBase
from atlas_resample import resample_to_grid

logger = logging.getLogger(__name__)

ZONE_ORDER = ("periventricular", "juxtacortical", "infratentorial", "deep_white_matter")


class MSZoneAnalyzer(AnatomicalAnalyzerBase):
    """Classifies MS lesions into McDonald zones using standard-space atlases."""

    REPORT_SUFFIX = "_mcdonald_report.json"

    def __init__(
        self,
        ventricle_atlas_path: Path,
        cortex_atlas_path: Path,
        infratentorial_atlas_path: Path,
        dilation_voxels: int = 1,
    ):
        self.ventricle_atlas_path = Path(ventricle_atlas_path)
        self.cortex_atlas_path = Path(cortex_atlas_path)
        self.infratentorial_atlas_path = Path(infratentorial_atlas_path)
        self.dilation_voxels = dilation_voxels

    def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
        try:
            mask_nii = nib.load(str(mask_path))
            mask_data = np.asarray(mask_nii.dataobj)
            voxel_vol_mm3 = float(np.prod(mask_nii.header.get_zooms()[:3]))
            voxel_vol_cm3 = voxel_vol_mm3 / 1000.0
            target_affine = mask_nii.affine
            target_shape = mask_nii.shape[:3]

            ventricle = resample_to_grid(self.ventricle_atlas_path, target_affine, target_shape)
            cortex = resample_to_grid(self.cortex_atlas_path, target_affine, target_shape)
            infratentorial = resample_to_grid(self.infratentorial_atlas_path, target_affine, target_shape)
            if ventricle is None or cortex is None or infratentorial is None:
                logger.error(f"Failed to resample one or more zone atlases for {mask_path.name}")
                return None

            structure = np.ones((3, 3, 3)) if self.dilation_voxels > 0 else None
            ventricle_zone = (ventricle > 0)
            cortex_zone = (cortex > 0)
            infratentorial_zone = (infratentorial > 0)
            for _ in range(self.dilation_voxels):
                ventricle_zone = binary_dilation(ventricle_zone, structure=structure)
                cortex_zone = binary_dilation(cortex_zone, structure=structure)
                infratentorial_zone = binary_dilation(infratentorial_zone, structure=structure)

            binary = (mask_data > 0).astype(np.uint8)
            labeled, n_components = ndimage_label(binary)

            zone_counts = {zone: 0 for zone in ZONE_ORDER}
            zone_volumes_cm3 = {zone: 0.0 for zone in ZONE_ORDER}
            lesion_zones_by_label = {}

            for i in range(1, n_components + 1):
                lesion_voxels = (labeled == i)
                voxel_count = int(lesion_voxels.sum())
                volume_cm3 = voxel_count * voxel_vol_cm3

                if (lesion_voxels & ventricle_zone).any():
                    zone = "periventricular"
                elif (lesion_voxels & cortex_zone).any():
                    zone = "juxtacortical"
                elif (lesion_voxels & infratentorial_zone).any():
                    zone = "infratentorial"
                else:
                    zone = "deep_white_matter"

                zone_counts[zone] += 1
                zone_volumes_cm3[zone] = round(zone_volumes_cm3[zone] + volume_cm3, 4)
                lesion_zones_by_label[str(i)] = zone

            zones_report = {
                zone: {"lesion_count": zone_counts[zone], "total_volume_cm3": zone_volumes_cm3[zone]}
                for zone in ZONE_ORDER
            }
            zones_report["spinal_cord"] = {"supported": False}

            return {
                "mask_file": mask_path.name,
                "total_lesion_count": n_components,
                "zones": zones_report,
                "lesion_zones_by_label": lesion_zones_by_label,
            }

        except Exception as e:
            logger.error(f"Failed to analyze {mask_path.name}: {e}")
            return None

    def save_report(self, report: Dict, output_path: Path) -> bool:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save report to {output_path}: {e}")
            return False
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_ms_localization.py -v`
Expected: `6 passed`

- [ ] **Step 6: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/ms_localization.py`
Expected: no output, exit code 0

- [ ] **Step 7: Commit**

```bash
git add configs/ms_zones_config.yaml scripts/ms_localization.py test_ms_localization.py
git commit -m "feat(ms-zones): add MSZoneAnalyzer with McDonald zone classification"
```

---

### Task 5: Wire `MSZoneAnalyzer` into the Stage 08 dispatcher

**Files:**
- Modify: `scripts/08_anatomical_analysis.py` (argparse, config loading, analyzer selection, report filename, `process_one_mask`, `main`)
- Modify: `pipeline_config.yaml:136-148` (add `ms-zones-config` arg)
- Modify: `test_stage08_fixes.py` (mock the new module + config arg in `_run_main_08`)

**Interfaces:**
- Consumes: `MSZoneAnalyzer` (Task 4), `LobarAnalyzer` (unchanged).
- Produces: `08_anatomical_analysis.py --lesion-type multiple_sclerosis` now additionally requires `--ms-zones-config <path>` and writes `*_mcdonald_report.json` instead of `*_lobar_report.json` for MS masks.

- [ ] **Step 1: Add the new CLI argument**

In `scripts/08_anatomical_analysis.py`, in `main()`'s argument parser, after the existing `--preprocessing-config` argument, add:

```python
    parser.add_argument("--ms-zones-config", type=Path, default=None,
                        help="Path to ms_zones_config.yaml (required when --lesion-type multiple_sclerosis)")
```

- [ ] **Step 2: Validate it when needed, load it, resolve atlas paths**

After the existing block that loads `lobar_config`/`preprocessing_config` and resolves `atlas_path`/`mapping_path` (right after the `resolve_atlas_path(...)` call), add:

```python
    ms_zone_atlas_paths = None
    if args.lesion_type == "multiple_sclerosis":
        if not args.ms_zones_config or not args.ms_zones_config.exists():
            logger.error("multiple_sclerosis requires --ms-zones-config pointing to a valid ms_zones_config.yaml")
            return 1
        with open(args.ms_zones_config, 'r') as f:
            ms_zones_config = yaml.safe_load(f)
        template_name = preprocessing_config.get("atlas", {}).get("name", "SRI24")
        zone_templates = ms_zones_config.get("templates", {})
        if template_name not in zone_templates:
            logger.error(f"No MS zone atlases registered for template '{template_name}'. "
                         f"Available: {list(zone_templates.keys())}")
            return 1
        ms_zone_atlas_paths = {
            "ventricle_atlas_path": project_root / zone_templates[template_name]["ventricle_atlas"],
            "cortex_atlas_path": atlas_path,  # reuse the already-resolved cortical atlas
            "infratentorial_atlas_path": project_root / zone_templates[template_name]["infratentorial_atlas"],
            "dilation_voxels": ms_zones_config.get("dilation_voxels", 1),
        }
```

- [ ] **Step 3: Add a small analyzer-construction helper**

Add this function near the top of the file, after the imports:

```python
def build_analyzer(lesion_type: str, atlas_path: Path, mapping_path: Path,
                    seg_classes: Dict, ms_zone_atlas_paths: Optional[Dict]):
    """Build the AnatomicalAnalyzerBase implementation for this lesion type."""
    if lesion_type == "multiple_sclerosis":
        from ms_localization import MSZoneAnalyzer
        return MSZoneAnalyzer(**ms_zone_atlas_paths)
    return LobarAnalyzer(atlas_path, mapping_path, seg_classes)


REPORT_SUFFIX_BY_LESION_TYPE = {
    "glioblastoma": "_lobar_report.json",
    "multiple_sclerosis": "_mcdonald_report.json",
}
```

- [ ] **Step 4: Update `process_one_mask` to use them**

Change its signature and body. Current signature:

```python
def process_one_mask(
    mask_path: Path,
    subject_id: str,
    session_id: str,
    output_dir: Path,
    atlas_path: Path,
    mapping_path: Path,
    seg_classes: Dict,
    lesion_type: str,
    analyzer: Optional["LobarAnalyzer"] = None,
) -> Dict:
```

becomes:

```python
def process_one_mask(
    mask_path: Path,
    subject_id: str,
    session_id: str,
    output_dir: Path,
    atlas_path: Path,
    mapping_path: Path,
    seg_classes: Dict,
    lesion_type: str,
    ms_zone_atlas_paths: Optional[Dict] = None,
    analyzer: Optional["AnatomicalAnalyzerBase"] = None,
) -> Dict:
```

Inside the function, change:

```python
    try:
        if analyzer is None:
            analyzer = LobarAnalyzer(atlas_path, mapping_path, seg_classes)
        report = analyzer.analyze_mask(mask_path)
```

to:

```python
    try:
        if analyzer is None:
            analyzer = build_analyzer(lesion_type, atlas_path, mapping_path, seg_classes, ms_zone_atlas_paths)
        report = analyzer.analyze_mask(mask_path)
```

And change the report filename line from:

```python
        mask_stem = mask_path.name.replace("_segmask.nii.gz", "")
        report_name = f"{mask_stem}_lobar_report.json"
```

to:

```python
        mask_stem = mask_path.name.replace("_segmask.nii.gz", "")
        report_name = f"{mask_stem}{REPORT_SUFFIX_BY_LESION_TYPE[lesion_type]}"
```

- [ ] **Step 5: Pass `ms_zone_atlas_paths` through both call sites in `main()`**

Sequential mode — change:

```python
        shared_analyzer = LobarAnalyzer(atlas_path, mapping_path, seg_classes)
        for idx, (mask_path, subj, sess) in enumerate(masks, 1):
            logger.info(f"\n[{idx}/{len(masks)}] {subj}/{sess}")
            result = process_one_mask(
                mask_path, subj, sess, args.output_dir,
                atlas_path, mapping_path, seg_classes,
                args.lesion_type,
                analyzer=shared_analyzer,
            )
```

to:

```python
        shared_analyzer = build_analyzer(args.lesion_type, atlas_path, mapping_path, seg_classes, ms_zone_atlas_paths)
        for idx, (mask_path, subj, sess) in enumerate(masks, 1):
            logger.info(f"\n[{idx}/{len(masks)}] {subj}/{sess}")
            result = process_one_mask(
                mask_path, subj, sess, args.output_dir,
                atlas_path, mapping_path, seg_classes,
                args.lesion_type,
                ms_zone_atlas_paths=ms_zone_atlas_paths,
                analyzer=shared_analyzer,
            )
```

Parallel mode — change:

```python
            future_map = {
                executor.submit(
                    process_one_mask,
                    mask_path, subj, sess, args.output_dir,
                    atlas_path, mapping_path, seg_classes,
                    args.lesion_type,
                ): (subj, sess)
                for mask_path, subj, sess in masks
            }
```

to:

```python
            future_map = {
                executor.submit(
                    process_one_mask,
                    mask_path, subj, sess, args.output_dir,
                    atlas_path, mapping_path, seg_classes,
                    args.lesion_type,
                    ms_zone_atlas_paths,
                ): (subj, sess)
                for mask_path, subj, sess in masks
            }
```

(Positional here because `ProcessPoolExecutor.submit` pickles plain args fine; `ms_zone_atlas_paths` is a plain dict of `Path`/`int`, picklable.)

- [ ] **Step 6: Update `pipeline_config.yaml`**

Change the `stage_08_anatomical_analysis` block (from the Stage 08 refactor plan) to add the new arg:

```yaml
  stage_08_anatomical_analysis:
    enabled: true
    script: scripts/08_anatomical_analysis.py
    args:
      config: configs/lobar_atlas_config.yaml
      preprocessing-config: configs/preprocessing_config.yaml
      ms-zones-config: configs/ms_zones_config.yaml
      mode: parallel
      workers: 14
      skip-existing: true
      benchmark: false
      max_subjects: null
      results_dir: /media/ssd/roppert/UPENN-GBM_preprocessed_4/benchmark_results/stage_08
```

(Passing `ms-zones-config` unconditionally is harmless for glioblastoma runs — `build_analyzer`/the validation block only consult it when `--lesion-type multiple_sclerosis`.)

- [ ] **Step 7: Update `test_stage08_fixes.py`'s mocks and config helper**

Add `"ms_localization"` to the mocked-modules block (same place as the `"lesion_stats"` mock added in the Stage 08 refactor plan):

```python
if "ms_localization" not in sys.modules:
    sys.modules["ms_localization"] = MagicMock()
```

In `_make_minimal_configs`, add a minimal `ms_zones_config.yaml` so `_run_main_08` keeps working for glioblastoma-mode tests (which now also load this file path but don't require it to exist, per Step 2's `if args.lesion_type == "multiple_sclerosis"` guard — no change needed to `_make_minimal_configs` or `_run_main_08` itself, since all existing tests use the default `glioblastoma` lesion type and never hit the new validation branch). Confirm this by re-running the suite in Step 8 rather than guessing.

- [ ] **Step 8: Run the full stage 08 test suite**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_stage08_fixes.py -v`
Expected: `9 passed` (unchanged — the new code path is only exercised for `multiple_sclerosis`, which these tests don't use)

- [ ] **Step 9: Compile-check**

Run: `cd /home/ubuntu/mri_ai_service && python -m py_compile scripts/08_anatomical_analysis.py`
Expected: no output, exit code 0

- [ ] **Step 10: Commit**

```bash
git add scripts/08_anatomical_analysis.py pipeline_config.yaml test_stage08_fixes.py
git commit -m "feat(stage08): dispatch to MSZoneAnalyzer for multiple_sclerosis masks"
```

---

### Task 6: End-to-end smoke test on real data + full regression

**Files:** none (verification only)

- [ ] **Step 1: Run the entire touched test suite**

Run: `cd /home/ubuntu/mri_ai_service && python -m pytest test_stage08_fixes.py test_lesion_stats.py test_anatomical_analyzer_base.py test_atlas_resample.py test_register_ms_zone_atlases.py test_ms_localization.py -v`
Expected: all pass

- [ ] **Step 2: Manual smoke test against `data/MS_5/P000915`**

This requires the pipeline already producing a `*_segmask.nii.gz` for at least one session of `P000915` (i.e. Stages 01-06 already run for this subject — if not yet available, run `orchestrator.py` for this input first, or point directly at an existing `segmentation/` output directory from a prior run). Then run Stage 08 standalone:

```bash
cd /home/ubuntu/mri_ai_service
python scripts/08_anatomical_analysis.py \
  <path_to_segmentation_output_dir> \
  <path_to_output_dir> \
  --config configs/lobar_atlas_config.yaml \
  --preprocessing-config configs/preprocessing_config.yaml \
  --ms-zones-config configs/ms_zones_config.yaml \
  --lesion-type multiple_sclerosis \
  --mode sequential
```

Expected: a `*_mcdonald_report.json` next to the existing `*_lesion_stats_report.json` for `sub-P000915`. Inspect it with `python -m json.tool <path_to_report>` — confirm `zones.spinal_cord == {"supported": false}` is present, and that `zones.periventricular.lesion_count + zones.juxtacortical.lesion_count + zones.infratentorial.lesion_count + zones.deep_white_matter.lesion_count` equals `total_lesion_count`.

- [ ] **Step 3: Push**

```bash
git push
```

## Self-Review Notes

- **Spec coverage:** Implements the full "(A) McDonald-классификация" section of the design spec, including the explicit spinal-cord-unsupported requirement and the atlas-sourcing-from-Docker decision. Frontend rendering is intentionally deferred to plan 2b per the spec's own module boundary (`ms_localization.py` here; `normalizeKappaEntity`/`ClinicalReportContent.jsx` in 2b).
- **Type consistency:** `MSZoneAnalyzer.__init__` keyword names (`ventricle_atlas_path`, `cortex_atlas_path`, `infratentorial_atlas_path`, `dilation_voxels`) match exactly between Task 4's class definition, Task 4's tests, and Task 5's `build_analyzer(...)` call (`**ms_zone_atlas_paths` unpacking the dict built in Task 5 Step 2 — keys there are spelled identically).
- **No placeholders:** the only intentionally-unresolved values are the 3 atlas label integers in Task 3, which are explicitly flagged as needing a real verification step (Task 1, Step 3) rather than asserted as fact — this is a real-world data dependency the plan author cannot verify without the actual FSL XML file in hand, not a deferred design decision.
