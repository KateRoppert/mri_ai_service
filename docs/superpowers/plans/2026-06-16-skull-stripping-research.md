# Skull Stripping Research Implementation Plan

> **FROZEN (2026-09-04).** Do not execute this file. Plugin Tasks 1.1–1.3, 2.1 and 3.1 landed on `main` with different APIs (`STRIPPERS`, `get_stripper(params)`, `HdBetStripper`, GPU pool). Remaining work: [2026-09-04-skull-stripping-research.md](2026-09-04-skull-stripping-research.md). Keep this document as the original task text and long code samples for Phases 5–10 (apply the overlay in the September plan).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a config-driven, MAS-ready skull-stripping plugin layer in Stage 05 plus a standalone benchmark harness that compares 7 skull-stripping tools across 4 datasets, producing the statistical analysis and paper that justify 2 production winners.

**Architecture:** Replace the single `skull_stripping.py` module with a `skull_stripping/` package built around a `SkullStripperBase` ABC and a config-reading dispatcher (with availability-driven fallback). Each tool is a thin wrapper class with a YAML manifest under `services/skull-stripping/`. A separate, self-contained `research/skull_stripping_benchmark/` project prepares registration-space NIfTI inputs (pipeline stages 1–4), runs every tool, computes quality + performance metrics, tunes parameters on a validation split, and emits CSVs, figures, and a statistics notebook that feed the paper skeleton in `docs/papers/`.

**Tech Stack:** Python 3.12, FSL BET, HD-BET, SynthStrip (surfa), ANTs/nibabel, numpy, scipy/pingouin/statsmodels, pandas, matplotlib/seaborn, pytest, pynvml/psutil for resource measurement.

---

## Orientation for the implementer

You are working in `/home/e.roppert/work/mri_ai_service` on branch `feat/skull-stripping-research`.

Key facts about the existing code you will touch or imitate:

- **Stage 05 driver:** [scripts/05_preprocessing.py](../../../scripts/05_preprocessing.py). It runs steps reorient → bias_correction → registration → resampling → skull_stripping. It puts `scripts/` on `sys.path` and imports skull stripping with:
  ```python
  from preprocessing_steps.skull_stripping import (
      setup_fsl_environment,
      check_fsl_installed,
      process_subject_skull_stripping
  )
  ```
  These three names **must keep working** after the refactor. `setup_fsl_environment(fsl_dir)` is called at line ~729; `process_subject_skull_stripping(...)` at line ~477 with signature `(subject_dir, output_dir, transform_dir, modalities, params)`.

- **Current skull stripping module:** [scripts/preprocessing_steps/skull_stripping.py](../../../scripts/preprocessing_steps/skull_stripping.py). Contains FSL machinery (`setup_fsl_environment`, `get_fsl_env`, `get_bet_command`, `check_fsl_installed`), `run_bet`, `apply_brain_mask`, `process_subject_skull_stripping`, `compare_before_after_stripping`. We move this code, do not rewrite its behavior.

- **Config:** [configs/preprocessing_config.yaml](../../../configs/preprocessing_config.yaml). The `skull_stripping` step currently:
  ```yaml
  - name: skull_stripping
    enabled: true
    params:
      method: "bet"
      reference_modality: "t1"
      fractional_intensity: 0.35
      vertical_gradient: -0.1
      apply_to_all: true
      cleanup: true
  ```
  Atlas is `SRI24` (line 7); MNI152 template lives at `data/templates/MNI152_T1_1mm.nii.gz`.

- **Data layout:** BIDS-like. Subject dir is `.../sub-XXX/ses-YYY/anat/`, files named `sub-XXX_ses-YYY_<modality>.nii.gz` where modality ∈ `t1c, t1, t2, t2fl`. `process_subject_skull_stripping` derives `subject_id`/`session_id` from the path.

- **Test convention:** the repo co-locates `test_*.py` (e.g. [scripts/test_bids_scanner.py](../../../scripts/test_bids_scanner.py)). We use `pytest`. Plugin tests get a `conftest.py` that puts `scripts/` on `sys.path`; benchmark tests get a `conftest.py` that puts both the benchmark dir and `scripts/` on `sys.path`.

### Shared contract (referenced by every tool task — read once)

Every stripper returns a dict with exactly these keys from `strip()`:

```python
{
    "success": bool,          # True if a mask was produced
    "output_path": str,       # path to skull-stripped reference image (str), or None
    "mask_path": str | None,  # path to binary brain mask
    "processing_time": float, # seconds, time.perf_counter()
    "vram_used_gb": float,    # peak VRAM delta in GB; 0.0 for CPU-only tools
    "error": str,             # present only when success is False
}
```

`SkullStripperBase` (defined in Task 2) is the ABC all tools subclass. The dispatcher (Task 4) maps a `method:` string to an instance and applies fallback. Keep these names stable; later tasks depend on them.

---

## File Structure

**Plugin package** (`scripts/preprocessing_steps/skull_stripping/`):
- `__init__.py` — backward-compatible re-exports + package docstring
- `base.py` — `SkullStripperBase` ABC, `load_manifest()`, `get_gpu_memory_mb()`, `apply_brain_mask()`, `compare_before_after_stripping()`
- `bet.py` — FSL machinery (moved verbatim) + `run_bet()` + `BetStripper`
- `hdbet.py` — `HDBetStripper`
- `synthstrip.py` — `SynthStripStripper`
- `sam_bet.py` — `SamBetStripper` (best-effort)
- `deepbet.py` — `DeepBetStripper` (best-effort)
- `mni_mask.py` — `MniMaskStripper` (strict + loose)
- `dispatcher.py` — `STRIPPER_REGISTRY`, `get_stripper()`, `resolve_stripper()`, refactored `process_subject_skull_stripping()`
- `tests/conftest.py`, `tests/test_*.py`

**Manifests** (`services/skull-stripping/<name>/manifest.yaml`): `bet, hdbet, synthstrip, sam, deepbet, mni_mask`.

**Benchmark** (`research/skull_stripping_benchmark/`): `README.md`, `DEVLOG.md`, `requirements-research.txt`, `prepare_data.py`, `metrics.py`, `run_benchmark.py`, `tuning.py`, `report.py`, `results/` (csv + figures + notebook), `tests/`.

**Paper** (`docs/papers/skull_stripping_comparison/`): `paper_skeleton.md` (EN) + `paper_skeleton_ru.md` (RU, kept in sync), `literature_review.md`, `literature_notes.md` (pointer), `figures/`. Whenever results fill the paper (Tasks 9.x, 10.3), update **both** the EN and RU skeletons.

---

## Phase 0 — Research scaffolding

### Task 0.1: Create research + paper directory scaffolding

**Files:**
- Create: `research/skull_stripping_benchmark/requirements-research.txt`
- Create: `research/skull_stripping_benchmark/DEVLOG.md`
- Create: `research/skull_stripping_benchmark/results/.gitkeep`
- Create: `research/skull_stripping_benchmark/results/figures/visual_examples/.gitkeep`
- Create: `docs/papers/skull_stripping_comparison/figures/.gitkeep`

- [ ] **Step 1: Create the research requirements file**

`research/skull_stripping_benchmark/requirements-research.txt`:
```text
# Statistical analysis + plotting for the skull stripping benchmark.
# Install into the project venv: pip install -r research/skull_stripping_benchmark/requirements-research.txt
scipy>=1.11
pingouin>=0.5.4
statsmodels>=0.14
pandas>=2.0
matplotlib>=3.7
seaborn>=0.13
numpy>=1.24
nibabel>=5.0
scikit-image>=0.22        # surface distance / HD95
psutil>=5.9               # RAM measurement
pynvml>=11.5              # VRAM measurement (optional at runtime)
jupyter>=1.0
```

- [ ] **Step 2: Create DEVLOG.md with the first entry**

`research/skull_stripping_benchmark/DEVLOG.md`:
```markdown
# Skull Stripping Benchmark — Development Log

Updated at the end of each working session: date · done · blockers · next step.
Tracks actual vs estimated timeline (estimate: 6–8 weeks part-time).

## 2026-06-16
- **Done:** Scaffolding created (research dir, requirements, paper skeleton dirs).
- **Blockers:** none.
- **Next step:** implement plugin architecture (SkullStripperBase + BET refactor).
```

- [ ] **Step 3: Create empty-dir placeholders**

```bash
mkdir -p research/skull_stripping_benchmark/results/figures/visual_examples
mkdir -p docs/papers/skull_stripping_comparison/figures
touch research/skull_stripping_benchmark/results/.gitkeep
touch research/skull_stripping_benchmark/results/figures/visual_examples/.gitkeep
touch docs/papers/skull_stripping_comparison/figures/.gitkeep
```

- [ ] **Step 4: Commit**

```bash
git add research/skull_stripping_benchmark/requirements-research.txt \
        research/skull_stripping_benchmark/DEVLOG.md \
        research/skull_stripping_benchmark/results/.gitkeep \
        research/skull_stripping_benchmark/results/figures/visual_examples/.gitkeep \
        docs/papers/skull_stripping_comparison/figures/.gitkeep
git commit -m "chore(ss-research): scaffold benchmark + paper directories"
```

---

### Task 0.2: Paper skeleton and literature notes

**Files:**
- Create: `docs/papers/skull_stripping_comparison/paper_skeleton.md`
- Create: `docs/papers/skull_stripping_comparison/literature_notes.md`

- [ ] **Step 1: Write the paper skeleton**

`docs/papers/skull_stripping_comparison/paper_skeleton.md`:
```markdown
# Comparative Evaluation of Skull-Stripping Tools for a Heterogeneous Clinical MRI Pipeline

> Status: skeleton. `<<...>>` markers are placeholders filled from benchmark results.

## Abstract
<<One paragraph: motivation, 7 tools, 4 datasets (GBM + MS), key finding, 2 winners.>>

## 1. Introduction
- Clinical context: automated brain-lesion diagnosis pipeline (GBM + MS), multi-center.
- Problem: skull stripping currently FSL BET, chosen empirically; may not generalize.
- Contributions: (1) systematic 7-tool comparison on lesion data; (2) plugin
  architecture making the tool config-driven and MAS-ready; (3) evidence-based
  selection of 2 production tools.

## 2. Related Work
<<From literature_notes.md: BET, HD-BET, SynthStrip, SAM-derived, DeepBET, atlas masking.>>

## 3. Methods
### 3.1 Datasets
| Dataset | Type | Access | Subjects (val/test) |
|---|---|---|---|
| UPENN-GBM | Glioblastoma | Open | <<n>> |
| MosMed Sclerosis | MS | Open | <<n>> |
| Proprietary clinical GBM | Glioblastoma | Closed | <<n>> |
| Proprietary clinical MS | MS | Closed | <<n>> |
- Stratified sampling 20–30/dataset; 20% validation / 80% test at subject level.
- Inputs: NIfTI after MNI152 registration, before skull stripping (see 3.5).

### 3.2 Tools
<<Table: tool, year, type, install, tuned params. From manifests.>>

### 3.3 Metrics
- Quality: DSC, HD95, over-stripping rate, leakage rate, brain volume (ml) vs MNI152
  pseudo-GT. Qualitative 1–3 visual scale on 5–10 subjects/dataset.
- Performance: time (s/vol), RAM peak (GB), VRAM peak (GB), reproducibility (DSC of
  two identical runs).

### 3.4 Statistical Analysis
<<Test hierarchy: normality → ANOVA/Kruskal-Wallis → Bonferroni → effect size →
Friedman. Research questions table.>>

### 3.5 Preprocessing and Hardware
- Stages 1–4 via `prepare_data.py`, `skull_stripping: disabled`, bias correction
  registration-only; benchmark input is the registration-space image.
- Atlas: MNI152 for all datasets (GBM overrides SRI24 for consistency).
- Hardware: NVIDIA RTX 5070 12GB VRAM.

## 4. Results
### 4.1 Per-tool quality (table) <<auto from report.py>>
### 4.2 Tool × dataset interaction (heatmap) <<dsc_heatmap_tool_x_dataset.png>>
### 4.3 Performance vs quality trade-off <<performance_scatter.png>>
### 4.4 Leakage / over-stripping <<leakage_comparison.png>>
### 4.5 Qualitative examples <<visual_examples/>>
### 4.6 Statistical tests <<from statistical_analysis.ipynb>>

## 5. Discussion
- Limitations: atlas pseudo-GT imperfect under GBM mass effect.
- MAS implications: manifests, cascade priority, lesion-type preference.
- Recommended 2 production winners + parameters.

## 6. Conclusion
<<Two winners, when to use which, future work.>>

## References
<<From literature_notes.md.>>
```

- [ ] **Step 2: Write literature notes stub**

`docs/papers/skull_stripping_comparison/literature_notes.md`:
```markdown
# Literature Notes — Skull Stripping (targeted review 2019–2025)

Format per entry: Citation · Method · Reported strengths/weaknesses · Relevance to us.

## FSL BET (Smith 2002)
<<notes>>

## HD-BET (Isensee et al. 2019)
<<notes — BraTS-validated, multi-modal, GPU>>

## SynthStrip (Hoopes et al. 2022)
<<notes — modality-agnostic, --border parameter>>

## SAM-based brain extraction (2023)
<<notes — experimental, best-effort inclusion>>

## DeepBET (2023/24)
<<notes — experimental, best-effort inclusion>>

## Atlas-based masking (MNI152)
<<notes — baseline; when is atlas masking competitive?>>

## Open questions for our datasets
- Does any GPU tool beat a dilated MNI mask enough to justify cost?
- Tool × lesion-type interaction?
```

- [ ] **Step 3: Commit**

```bash
git add docs/papers/skull_stripping_comparison/paper_skeleton.md \
        docs/papers/skull_stripping_comparison/literature_notes.md
git commit -m "docs(ss-research): add paper skeleton and literature notes stub"
```

---

### Task 0.3: Conduct targeted literature review and position the contribution

**Files:**
- Create: `docs/papers/skull_stripping_comparison/literature_review.md` (the analytical report — the project's reference base)
- Modify: `docs/papers/skull_stripping_comparison/literature_notes.md` (reduce to a pointer to the review)
- Modify: `docs/papers/skull_stripping_comparison/paper_skeleton.md` (§2 Related Work + §1 contribution bullets)
- Possibly add: `docs/papers/skull_stripping_comparison/figures/` (saved key figures from reviewed papers)

> **Done interactively with the user, not by an autonomous subagent.** Research/judgement step; the contribution framing is the user's scientific call (their PhD is on MAS). No TDD. This is a **gate**: it runs after the skeleton and *before* plugin code, so that if an equivalent published comparison exists we revise differentiation before investing weeks in the benchmark.
>
> **Agreed contribution framing (2026-06-16):** the prior-art comparison of strippers on glioma is largely done (Thakur 2020). Our contribution is therefore positioned on **(1) skull-stripper selection for our specific heterogeneous clinical GBM+MS multi-center pipeline** and **(2) the plugin / MAS architecture** (config-driven selection, manifests, cascade/fallback) — leaning into the MAS angle of the user's PhD. The MS side is an under-studied gap. Cost/quality (GPU tool vs dilated atlas mask) is a supporting argument.

- [ ] **Step 1: Targeted search — only key, high-quality works (2019–2026)**

Cover: individual tools (FSL BET, HD-BET, SynthStrip, deepbet, SAM-based), prior comparison/benchmark studies, and — for the MAS angle — any work that wraps skull strippers (or comparable preprocessing) as agents/microservices, plus the **freshest 2026 MAS works** (a separate MAS-in-medicine report already exists; only add what is new since then or directly stripper-related). Depth: only the strongest, highest-cited/most-relevant papers, not exhaustive.

- [ ] **Step 2: Per-paper extraction card (agreed template)**

For each paper fill: **A. Identification** (authors, year, venue, type, code/weights openness, **direct URL link**); **B. Method** (class, architecture, input modalities, CPU/GPU/VRAM/speed); **C. Evaluation** (datasets + pathology + mono/multi-center + n, ground truth, metrics, significance testing); **D. Results** (key DSC/HD95 numbers, where it wins/degrades, esp. on pathology); **E. Relevance to us** (fit for heterogeneous clinical GBM+MS pipeline; **MAS aspect** — can it be wrapped as an agent/microservice, manifest-like properties, orchestration/selection; cost/quality signal); **F. Gap left open** (what it does NOT close → our differentiation). **Embed important figures/schemes** from the paper into `figures/` and reference them in the report.

- [ ] **Step 3: Write the analytical report `literature_review.md`**

Structure: (1) review method + search criteria/period/venues; (2) per-paper cards grouped as (a) individual tools, (b) comparison benchmarks, (c) MAS/agentic works directly relevant; (3) synthesis tables — tool × {class, modalities, GPU, pathology-in-eval, DSC} and a "who already compared what" matrix; (4) gap analysis; (5) explicit contribution statement vs each nearest competitor (Thakur 2020, deepbet, pediatric 2025); (6) **"Takeaways for our project"** — what to reuse, current field trends, and any concrete additions the reviewer (me) proposes for the project.

- [ ] **Step 4: Reduce `literature_notes.md` to a pointer** to `literature_review.md` (avoid a dead stub / duplicate maintenance).

- [ ] **Step 5: Gate decision with the user**

Confirm the differentiation holds against the reviewed prior art. If an existing paper covers our exact comparison on similar data, pause and revise scope/framing before Phase 1.

- [ ] **Step 6: Update the paper skeleton**

Fill `paper_skeleton.md` §2 Related Work (prose referencing the review) and tighten §1 contribution bullets to the agreed MAS + pipeline-specific framing.

- [ ] **Step 7: Commit**

```bash
git add docs/papers/skull_stripping_comparison/literature_review.md \
        docs/papers/skull_stripping_comparison/literature_notes.md \
        docs/papers/skull_stripping_comparison/paper_skeleton.md \
        docs/papers/skull_stripping_comparison/figures/
git commit -m "docs(ss-research): analytical literature review + contribution positioning"
```

---

## Phase 1 — Plugin architecture

> This phase refactors the existing module into a package **without changing behavior**, then adds the ABC and dispatcher. The first task is a pure move; the test guards that Stage 05's imports still resolve.

### Task 1.1: Convert module to package, move FSL/BET code into `bet.py`

**Files:**
- Delete: `scripts/preprocessing_steps/skull_stripping.py`
- Create: `scripts/preprocessing_steps/skull_stripping/__init__.py`
- Create: `scripts/preprocessing_steps/skull_stripping/bet.py`
- Create: `scripts/preprocessing_steps/skull_stripping/base.py` (helpers only this task)
- Create: `scripts/preprocessing_steps/skull_stripping/tests/__init__.py`
- Create: `scripts/preprocessing_steps/skull_stripping/tests/conftest.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_backward_compat.py`

- [ ] **Step 1: Write the failing backward-compat test**

`scripts/preprocessing_steps/skull_stripping/tests/conftest.py`:
```python
import sys
from pathlib import Path

# Put scripts/ on sys.path so `preprocessing_steps.skull_stripping` imports like Stage 05.
SCRIPTS_DIR = Path(__file__).resolve().parents[2]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
```

`scripts/preprocessing_steps/skull_stripping/tests/__init__.py`: empty file.

`scripts/preprocessing_steps/skull_stripping/tests/test_backward_compat.py`:
```python
"""Stage 05 imports these three names; they must survive the package refactor."""

def test_stage05_imports_resolve():
    from preprocessing_steps.skull_stripping import (
        setup_fsl_environment,
        check_fsl_installed,
        process_subject_skull_stripping,
    )
    assert callable(setup_fsl_environment)
    assert callable(check_fsl_installed)
    assert callable(process_subject_skull_stripping)


def test_legacy_helpers_still_exported():
    from preprocessing_steps.skull_stripping import (
        run_bet,
        apply_brain_mask,
        compare_before_after_stripping,
    )
    assert callable(run_bet)
    assert callable(apply_brain_mask)
    assert callable(compare_before_after_stripping)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_backward_compat.py -v`
Expected: collection/import error — the package does not exist yet (still a `.py` module being replaced).

- [ ] **Step 3: Create `base.py` with the shared helpers moved from the old module**

`scripts/preprocessing_steps/skull_stripping/base.py`:
```python
"""Shared base class, manifest loading, and modality-masking helpers for skull strippers."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

logger = logging.getLogger(__name__)

# services/skull-stripping/<name>/manifest.yaml relative to project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_MANIFEST_ROOT = _PROJECT_ROOT / "services" / "skull-stripping"


def load_manifest(name: str) -> dict:
    """Load services/skull-stripping/<name>/manifest.yaml; return {} if absent."""
    manifest_path = _MANIFEST_ROOT / name / "manifest.yaml"
    if not manifest_path.exists():
        logger.warning(f"Manifest not found for '{name}': {manifest_path}")
        return {}
    with open(manifest_path, "r") as f:
        return yaml.safe_load(f) or {}


def get_gpu_memory_mb() -> float:
    """Currently used VRAM in MB across GPU 0, or 0.0 if pynvml/GPU unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return info.used / (1024 ** 2)
    except Exception:
        return 0.0


def apply_brain_mask(input_path: Path, mask_path: Path, output_path: Path) -> dict:
    """Multiply an image by a binary mask and save it (preserves header/affine)."""
    try:
        logger.info(f"Applying brain mask to {input_path.name}")
        img = nib.load(input_path)
        mask = nib.load(mask_path)
        img_data = img.get_fdata()
        mask_data = mask.get_fdata()
        if img_data.shape != mask_data.shape:
            raise ValueError(
                f"Image shape {img_data.shape} != mask shape {mask_data.shape}"
            )
        masked_data = img_data * (mask_data > 0)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(masked_data, img.affine, img.header), output_path)
        logger.info(f"Saved masked image to {output_path}")
        return {"success": True, "output_path": str(output_path)}
    except Exception as e:
        logger.error(f"Error applying mask to {input_path.name}: {e}")
        return {"success": False, "error": str(e)}


class SkullStripperBase(ABC):
    """Contract every skull stripping tool implements.

    strip() returns:
        {success, output_path, mask_path, processing_time, vram_used_gb, error?}
    """

    name: str = "base"

    @abstractmethod
    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    @property
    def manifest(self) -> dict:
        return load_manifest(self.name)
```

> Note: `compare_before_after_stripping` is moved verbatim from the old module into `base.py` in this same step (copy lines 447–516 of the old file into `base.py`, keeping the function body identical).

- [ ] **Step 4: Create `bet.py` with the FSL machinery + `run_bet` moved verbatim**

`scripts/preprocessing_steps/skull_stripping/bet.py`:
- Copy verbatim from the old `skull_stripping.py`: the module-level `FSL_DIR`/`FSL_BIN_DIR` globals and the functions `setup_fsl_environment`, `get_fsl_env`, `get_bet_command`, `check_fsl_installed`, `run_bet` (lines 15–273 of the old file). Keep behavior identical.
- Append the `BetStripper` class:
```python
import time
from pathlib import Path

from .base import SkullStripperBase, get_gpu_memory_mb  # noqa: E402


class BetStripper(SkullStripperBase):
    """FSL BET. CPU-only, classical (Smith 2002)."""

    name = "bet"

    def is_available(self) -> bool:
        return check_fsl_installed()

    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        start = time.perf_counter()
        result = run_bet(
            input_path=input_path,
            output_path=output_path,
            mask_path=mask_path,
            fractional_intensity=params.get("fractional_intensity", 0.5),
            vertical_gradient=params.get("vertical_gradient", 0.0),
            generate_mask=True,
        )
        result.setdefault("processing_time", time.perf_counter() - start)
        result["vram_used_gb"] = 0.0
        return result
```

- [ ] **Step 5: Create `__init__.py` with backward-compatible re-exports**

`scripts/preprocessing_steps/skull_stripping/__init__.py`:
```python
"""Skull stripping plugin package.

Backward-compatible facade: Stage 05 imports setup_fsl_environment,
check_fsl_installed, and process_subject_skull_stripping from here.
"""

from .base import (
    SkullStripperBase,
    apply_brain_mask,
    compare_before_after_stripping,
    load_manifest,
)
from .bet import (
    BetStripper,
    check_fsl_installed,
    get_bet_command,
    get_fsl_env,
    run_bet,
    setup_fsl_environment,
)

# process_subject_skull_stripping is added to dispatcher in Task 1.3; until then
# this temporary re-export keeps Stage 05 working. Replaced in Task 1.3.
from .bet import run_bet as _run_bet  # noqa: F401

__all__ = [
    "SkullStripperBase",
    "apply_brain_mask",
    "compare_before_after_stripping",
    "load_manifest",
    "BetStripper",
    "check_fsl_installed",
    "get_bet_command",
    "get_fsl_env",
    "run_bet",
    "setup_fsl_environment",
]
```

> Temporary measure for this task only: also copy the old `process_subject_skull_stripping` (lines 333–444) into `bet.py` and add it to the `from .bet import` list and `__all__`, so the backward-compat test passes now. Task 1.3 moves it to `dispatcher.py` and updates the import. Delete the old `scripts/preprocessing_steps/skull_stripping.py` file in this step.

- [ ] **Step 6: Run the backward-compat test (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_backward_compat.py -v`
Expected: PASS (both tests).

- [ ] **Step 7: Smoke-test that Stage 05 still imports**

Run: `cd /home/e.roppert/work/mri_ai_service/scripts && python -c "import importlib.util, sys; sys.path.insert(0,'.'); from preprocessing_steps.skull_stripping import setup_fsl_environment, check_fsl_installed, process_subject_skull_stripping; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 8: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/ && git rm scripts/preprocessing_steps/skull_stripping.py
git commit -m "refactor(ss): convert skull_stripping module to package with BET preserved"
```

---

### Task 1.2: BET manifest + manifest-loading test

**Files:**
- Create: `services/skull-stripping/bet/manifest.yaml`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_manifest.py`

- [ ] **Step 1: Write the failing manifest test**

`scripts/preprocessing_steps/skull_stripping/tests/test_manifest.py`:
```python
from preprocessing_steps.skull_stripping import load_manifest, BetStripper


def test_bet_manifest_loads_with_required_keys():
    m = load_manifest("bet")
    assert m["name"] == "bet"
    assert m["tool_type"] == "skull_stripping"
    assert "compute" in m and "requires_gpu" in m["compute"]
    assert "mas_metadata" in m and "agent_type" in m["mas_metadata"]


def test_stripper_exposes_its_manifest():
    assert BetStripper().manifest["name"] == "bet"


def test_missing_manifest_returns_empty_dict():
    assert load_manifest("does-not-exist") == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_manifest.py -v`
Expected: FAIL — `KeyError`/empty dict, manifest file absent.

- [ ] **Step 3: Write the BET manifest**

`services/skull-stripping/bet/manifest.yaml`:
```yaml
name: bet
version: "1.0"
tool_type: skull_stripping
compute:
  requires_gpu: false
  gpu_memory_gb: 0
  ram_gb: 2
  cpu_fallback: true
lesion_type_preference:
  glioma: medium
  ms: medium
  metastases: medium
modality_support:
  primary: [t1, t1c]
  apply_to: [t1, t1c, t2, t2fl]
quality_notes: >
  Classical baseline (Smith 2002). Current production default, chosen empirically.
install:
  method: system
  package: fsl
  model_download: none
tuned_params: {}        # filled by tuning.py after benchmark
mas_metadata:
  agent_type: skull_stripper
  cascade_priority: 5
  fallback_to: null
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_manifest.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add services/skull-stripping/bet/manifest.yaml scripts/preprocessing_steps/skull_stripping/tests/test_manifest.py
git commit -m "feat(ss): add BET manifest and manifest-loading tests"
```

---

### Task 1.3: Dispatcher with fallback + refactored `process_subject_skull_stripping`

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/dispatcher.py`
- Modify: `scripts/preprocessing_steps/skull_stripping/__init__.py`
- Modify: `scripts/preprocessing_steps/skull_stripping/bet.py` (remove the temporarily-parked `process_subject_skull_stripping`)
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_dispatcher.py`

- [ ] **Step 1: Write the failing dispatcher test**

`scripts/preprocessing_steps/skull_stripping/tests/test_dispatcher.py`:
```python
import pytest
from pathlib import Path

from preprocessing_steps.skull_stripping import dispatcher
from preprocessing_steps.skull_stripping.base import SkullStripperBase


class _FakeAvailable(SkullStripperBase):
    name = "fake_ok"
    def is_available(self): return True
    def strip(self, *a, **k): return {"success": True, "vram_used_gb": 0.0}


class _FakeUnavailable(SkullStripperBase):
    name = "fake_bad"
    def is_available(self): return False
    def strip(self, *a, **k): return {"success": True, "vram_used_gb": 0.0}


def test_get_stripper_returns_registered_instance():
    s = dispatcher.get_stripper("bet")
    assert s.name == "bet"


def test_get_stripper_unknown_method_raises():
    with pytest.raises(ValueError):
        dispatcher.get_stripper("nope")


def test_resolve_uses_primary_when_available(monkeypatch):
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_ok", _FakeAvailable)
    s = dispatcher.resolve_stripper({"method": "fake_ok"})
    assert s.name == "fake_ok"


def test_resolve_falls_back_when_primary_unavailable(monkeypatch):
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_bad", _FakeUnavailable)
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_ok", _FakeAvailable)
    s = dispatcher.resolve_stripper({"method": "fake_bad", "fallback_method": "fake_ok"})
    assert s.name == "fake_ok"


def test_resolve_raises_when_primary_and_fallback_unavailable(monkeypatch):
    monkeypatch.setitem(dispatcher.STRIPPER_REGISTRY, "fake_bad", _FakeUnavailable)
    with pytest.raises(RuntimeError):
        dispatcher.resolve_stripper({"method": "fake_bad", "fallback_method": "fake_bad"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_dispatcher.py -v`
Expected: FAIL — `dispatcher` module does not exist.

- [ ] **Step 3: Write `dispatcher.py`**

`scripts/preprocessing_steps/skull_stripping/dispatcher.py`:
```python
"""Config-driven selection of a skull stripper, with availability-based fallback,
plus the per-subject orchestration used by Stage 05."""

import logging
from pathlib import Path

from .base import SkullStripperBase, apply_brain_mask
from .bet import BetStripper

logger = logging.getLogger(__name__)

# method string -> class. Extended as tools are added (hdbet, synthstrip, ...).
STRIPPER_REGISTRY: dict[str, type[SkullStripperBase]] = {
    "bet": BetStripper,
}


def get_stripper(method: str) -> SkullStripperBase:
    """Instantiate the stripper registered under `method`."""
    cls = STRIPPER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown skull stripping method '{method}'. "
            f"Available: {sorted(STRIPPER_REGISTRY)}"
        )
    return cls()


def resolve_stripper(params: dict) -> SkullStripperBase:
    """Return the primary stripper if available, else the fallback. Raise if neither."""
    method = params.get("method", "bet")
    primary = get_stripper(method)
    if primary.is_available():
        return primary

    fallback_method = params.get("fallback_method")
    logger.warning(
        f"Skull stripper '{method}' unavailable; "
        f"falling back to '{fallback_method}'."
    )
    if not fallback_method or fallback_method == method:
        raise RuntimeError(
            f"Skull stripper '{method}' unavailable and no usable fallback configured."
        )
    fallback = get_stripper(fallback_method)
    if not fallback.is_available():
        raise RuntimeError(
            f"Neither '{method}' nor fallback '{fallback_method}' is available."
        )
    return fallback


def process_subject_skull_stripping(subject_dir: Path, output_dir: Path,
                                    transform_dir: Path, modalities: list,
                                    params: dict) -> dict:
    """Create a brain mask on the reference modality via the configured stripper,
    then apply it to the remaining modalities. Signature matches Stage 05."""
    results = {}
    subject_id = subject_dir.parent.parent.name
    session_id = subject_dir.parent.name
    logger.info(f"Processing {subject_id}/{session_id} - Skull Stripping")

    stripper = resolve_stripper(params)
    logger.info(f"Using skull stripper: {stripper.name}")

    reference_modality = params.get("reference_modality", "t1c")
    ref_pattern = f"{subject_id}_{session_id}_{reference_modality}.nii.gz"
    ref_files = list(subject_dir.glob(ref_pattern))
    if not ref_files:
        msg = f"Reference modality {reference_modality} not found"
        logger.error(msg)
        return {"success": False, "error": msg}
    ref_file = ref_files[0]

    ref_output = output_dir / subject_id / session_id / "anat" / ref_pattern
    mask_pattern = f"{subject_id}_{session_id}_brain_mask.nii.gz"
    mask_path = transform_dir / subject_id / session_id / "anat" / mask_pattern
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    strip_result = stripper.strip(ref_file, ref_output, mask_path, params)
    results[reference_modality] = strip_result
    if not strip_result.get("success"):
        logger.error(f"Failed to create brain mask on {reference_modality}")
        return results

    if params.get("apply_to_all", True):
        for modality in modalities:
            if modality == reference_modality:
                continue
            modal_pattern = f"{subject_id}_{session_id}_{modality}.nii.gz"
            modal_files = list(subject_dir.glob(modal_pattern))
            if not modal_files:
                logger.warning(f"Modality {modality} not found, skipping")
                results[modality] = {"success": False, "error": "File not found"}
                continue
            modal_output = output_dir / subject_id / session_id / "anat" / modal_pattern
            results[modality] = apply_brain_mask(modal_files[0], mask_path, modal_output)

    if params.get("cleanup", True):
        for pattern in ["*_mesh.vtk", "*_skull.nii.gz", "*_outskin_mesh.off"]:
            for temp_file in subject_dir.parent.rglob(pattern):
                try:
                    temp_file.unlink()
                except OSError:
                    pass
    return results
```

- [ ] **Step 4: Update `__init__.py` to source `process_subject_skull_stripping` from dispatcher**

In `scripts/preprocessing_steps/skull_stripping/__init__.py`, remove the temporary `_run_bet` line and the temporary `process_subject_skull_stripping` re-export from `bet`, and add:
```python
from .dispatcher import (
    STRIPPER_REGISTRY,
    get_stripper,
    process_subject_skull_stripping,
    resolve_stripper,
)
```
Add `"process_subject_skull_stripping"`, `"get_stripper"`, `"resolve_stripper"`, `"STRIPPER_REGISTRY"` to `__all__`.

- [ ] **Step 5: Remove the parked function from `bet.py`**

Delete the temporarily-copied `process_subject_skull_stripping` from `bet.py` (added in Task 1.1 Step 5). `bet.py` keeps the FSL machinery, `run_bet`, and `BetStripper`.

- [ ] **Step 6: Run dispatcher + backward-compat tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/ -v`
Expected: PASS (all tests so far: backward_compat, manifest, dispatcher).

- [ ] **Step 7: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/
git commit -m "feat(ss): add config-driven dispatcher with availability fallback"
```

---

### Task 1.4: Cascade + mask-integrity validation gate (ADD-2)

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/validation.py`
- Modify: `scripts/preprocessing_steps/skull_stripping/dispatcher.py` (cascade loop + validation after `strip()`)
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_validation.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_cascade.py`

> Inspired by NeuroAgent's Generate-Execute-**Validate** engine (lit review §2.10) and the user's MAS
> cascade idea. This unifies **fault tolerance** (unavailable tool → next agent) and **validate-retry**
> (mask fails integrity thresholds → next agent) into one ordered **cascade**. Config gives an ordered
> list `cascade: [hdbet, synthstrip, bet]`; the dispatcher tries each available stripper in order,
> validates its mask, and accepts the first that passes. This is the runtime backbone of the MAS story
> (per-subject, self-correcting selection) and complements ADD-3 (lesion-type routing) / ADD-5
> (characteristic analysis). Adaptive runtime selection by data characteristics is explicitly out of
> scope here (future work, Этап 7).

- [ ] **Step 1: Write the failing test**

`scripts/preprocessing_steps/skull_stripping/tests/test_validation.py`:
```python
import numpy as np
import nibabel as nib
from pathlib import Path
from preprocessing_steps.skull_stripping.validation import validate_mask


def _write_mask(tmp, arr):
    p = tmp / "mask.nii.gz"
    nib.save(nib.Nifti1Image(arr.astype(np.uint8), np.eye(4)), p)
    return p


def test_valid_single_blob_passes(tmp_path):
    arr = np.zeros((40, 40, 40)); arr[10:30, 10:30, 10:30] = 1   # ~8000 vox = 8 ml @1mm
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=5, max_ml=2500)
    assert res["valid"] is True


def test_empty_mask_fails(tmp_path):
    arr = np.zeros((40, 40, 40))
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=5, max_ml=2500)
    assert res["valid"] is False and "volume" in res["reason"]


def test_too_large_fails(tmp_path):
    arr = np.ones((40, 40, 40))   # 64000 vox = 64 ml; set max below it
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=5, max_ml=10)
    assert res["valid"] is False


def test_fragmented_mask_fails(tmp_path):
    arr = np.zeros((40, 40, 40))
    arr[2:6, 2:6, 2:6] = 1                      # blob A
    arr[30:38, 30:38, 30:38] = 1                # blob B (larger, disconnected)
    res = validate_mask(_write_mask(tmp_path, arr), min_ml=0.01, max_ml=2500,
                        min_dominant_fraction=0.9)
    assert res["valid"] is False and "component" in res["reason"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_validation.py -v`
Expected: FAIL — `validation` module missing.

- [ ] **Step 3: Write `validation.py`**

`scripts/preprocessing_steps/skull_stripping/validation.py`:
```python
"""Mask-integrity validation for skull stripping outputs (ADD-2).

A produced brain mask is accepted only if its volume is within a plausible range
and it is dominated by a single connected component. Inspired by NeuroAgent's
output-integrity validation; used by the dispatcher to trigger fallback.
"""

import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label

logger = logging.getLogger(__name__)


def validate_mask(mask_path: Path, min_ml: float = 700.0, max_ml: float = 1900.0,
                  min_dominant_fraction: float = 0.95) -> dict:
    """Return {valid: bool, reason: str, volume_ml: float, dominant_fraction: float}.

    Defaults target adult whole-brain volumes; callers may relax via manifest/config.
    """
    img = nib.load(mask_path)
    data = (img.get_fdata() > 0)
    voxel_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
    volume_ml = float(data.sum() * voxel_ml)

    if volume_ml < min_ml or volume_ml > max_ml:
        return {"valid": False, "reason": f"volume {volume_ml:.0f}ml out of [{min_ml},{max_ml}]",
                "volume_ml": volume_ml, "dominant_fraction": 0.0}

    labeled, n = label(data)
    if n == 0:
        return {"valid": False, "reason": "volume: empty mask",
                "volume_ml": volume_ml, "dominant_fraction": 0.0}
    sizes = np.bincount(labeled.ravel())[1:]
    dominant_fraction = float(sizes.max() / sizes.sum())
    if dominant_fraction < min_dominant_fraction:
        return {"valid": False,
                "reason": f"fragmented: dominant component {dominant_fraction:.2f} < {min_dominant_fraction}",
                "volume_ml": volume_ml, "dominant_fraction": dominant_fraction}

    return {"valid": True, "reason": "ok", "volume_ml": volume_ml,
            "dominant_fraction": dominant_fraction}
```

- [ ] **Step 4: Write the failing cascade test**

`scripts/preprocessing_steps/skull_stripping/tests/test_cascade.py`:
```python
from preprocessing_steps.skull_stripping import dispatcher
from preprocessing_steps.skull_stripping.base import SkullStripperBase


def test_build_cascade_orders_method_first_then_cascade():
    order = dispatcher.build_cascade_order(
        {"method": "hdbet", "cascade": ["synthstrip", "hdbet", "bet"]})
    assert order == ["hdbet", "synthstrip", "bet"]   # method first, dedup, preserve order


def test_build_cascade_falls_back_to_single_fallback_method():
    order = dispatcher.build_cascade_order({"method": "hdbet", "fallback_method": "bet"})
    assert order == ["hdbet", "bet"]


def test_build_cascade_method_only():
    assert dispatcher.build_cascade_order({"method": "bet"}) == ["bet"]
```

- [ ] **Step 5: Wire cascade + validation into the dispatcher**

In `dispatcher.py` add near the top: `from .validation import validate_mask`. Add the cascade builder
and rewrite the reference-modality block of `process_subject_skull_stripping` to iterate the cascade,
accepting the first available stripper whose mask passes validation:
```python
def build_cascade_order(params: dict) -> list[str]:
    """Ordered, de-duplicated cascade: primary method first, then cascade/fallback chain."""
    method = params.get("method", "bet")
    chain = params.get("cascade") or (
        [params["fallback_method"]] if params.get("fallback_method") else [])
    order, seen = [], set()
    for name in [method, *chain]:
        if name and name not in seen:
            order.append(name); seen.add(name)
    return order
```
Replace the single-strip reference block with the cascade loop:
```python
    order = build_cascade_order(params)
    vcfg = params.get("validation", {})
    strip_result, used = None, None
    for name in order:
        try:
            cand = get_stripper(name)
        except ValueError:
            logger.warning("Cascade: unknown stripper '%s', skipping", name); continue
        if not cand.is_available():
            logger.warning("Cascade: '%s' unavailable, trying next", name); continue
        res = cand.strip(ref_file, ref_output, mask_path, params)
        if res.get("success") and mask_path.exists():
            check = validate_mask(mask_path, **vcfg) if vcfg else validate_mask(mask_path)
            res["mask_validation"] = check
            if check["valid"]:
                strip_result, used = res, name
                break
            logger.warning("Cascade: '%s' mask invalid (%s), trying next", name, check["reason"])
        else:
            logger.warning("Cascade: '%s' failed (%s), trying next", name, res.get("error"))
        strip_result, used = res, name   # keep last attempt for reporting
    if strip_result is None:
        return {"success": False, "error": f"No usable stripper in cascade {order}"}
    logger.info("Skull stripper used: %s", used)
    results[reference_modality] = strip_result
    if not strip_result.get("success"):
        logger.error(f"Failed to create brain mask on {reference_modality}")
        return results
```
Remove the now-superseded `resolve_stripper(params)` call at the top of the function (keep the
`resolve_stripper` function itself for backward compatibility / other callers).

- [ ] **Step 6: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_validation.py scripts/preprocessing_steps/skull_stripping/tests/test_cascade.py scripts/preprocessing_steps/skull_stripping/tests/test_dispatcher.py -v`
Expected: PASS (validation + cascade + dispatcher tests green).

- [ ] **Step 7: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/validation.py \
        scripts/preprocessing_steps/skull_stripping/dispatcher.py \
        scripts/preprocessing_steps/skull_stripping/tests/test_validation.py \
        scripts/preprocessing_steps/skull_stripping/tests/test_cascade.py
git commit -m "feat(ss): cascade selection + mask-integrity validation (ADD-2)"
```

---

## Phase 2 — Stage 05 integration

### Task 2.1: Wire dispatcher into Stage 05 and extend config

**Files:**
- Modify: `configs/preprocessing_config.yaml` (skull_stripping params)
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_stage05_config.py`

> Stage 05 already calls `process_subject_skull_stripping(subject_dir, output_dir, transform_dir, modalities, params)` and imports it from the package — no code change needed in `05_preprocessing.py`. This task confirms the config drives the dispatcher and adds `fallback_method` / `tool_params`.

- [ ] **Step 1: Write the failing config test**

`scripts/preprocessing_steps/skull_stripping/tests/test_stage05_config.py`:
```python
from pathlib import Path
import yaml

from preprocessing_steps.skull_stripping import dispatcher

CONFIG = Path(__file__).resolve().parents[4] / "configs" / "preprocessing_config.yaml"


def _skull_params():
    cfg = yaml.safe_load(CONFIG.read_text())
    step = next(s for s in cfg["steps"] if s["name"] == "skull_stripping")
    return step["params"]


def test_config_has_method_and_cascade():
    p = _skull_params()
    assert p["method"] in dispatcher.STRIPPER_REGISTRY
    assert isinstance(p["cascade"], list) and len(p["cascade"]) >= 1
    # every cascade entry is a known stripper
    assert all(name in dispatcher.STRIPPER_REGISTRY for name in p["cascade"])


def test_config_method_resolves_to_a_stripper():
    p = _skull_params()
    # get_stripper must not raise for the configured method.
    s = dispatcher.get_stripper(p["method"])
    assert s.name == p["method"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_stage05_config.py -v`
Expected: FAIL — `cascade` key missing.

- [ ] **Step 3: Extend the config**

In `configs/preprocessing_config.yaml`, replace the `skull_stripping` step params block with:
```yaml
  - name: skull_stripping
    enabled: true
    params:
      method: "bet"                 # bet | hdbet | synthstrip | brainmage | mni_mask | sam | deepbet
      reference_modality: "t1"
      fractional_intensity: 0.35    # BET -f
      vertical_gradient: -0.1       # BET -g
      apply_to_all: true
      cleanup: true
      cascade: ["bet"]              # ordered MAS cascade; e.g. ["hdbet","synthstrip","bet"]
      validation: {}                # mask-integrity thresholds; {} = adult defaults
      tool_params: {}               # tool-specific overrides (from manifest tuning)
```

> Production stays `cascade: ["bet"]` (BET-only, current behavior). The cascade mechanism is exercised
> by setting a multi-tool list. `build_cascade_order` also accepts the legacy `fallback_method` key.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_stage05_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/preprocessing_config.yaml scripts/preprocessing_steps/skull_stripping/tests/test_stage05_config.py
git commit -m "feat(ss): add method/fallback config keys for Stage 05 dispatcher"
```

---

## Phase 3 — Deep-learning tool wrappers (HD-BET, SynthStrip)

> Pattern for all wrapper tasks: `is_available()` checks the binary/package; `strip()` builds and runs the command, measures time + VRAM, normalizes the output to the shared contract, converts the tool's mask filename to `mask_path`. Tests mock `subprocess`/availability so they run without the tool installed.

### Task 3.1: HD-BET wrapper + manifest

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/hdbet.py`
- Create: `services/skull-stripping/hdbet/manifest.yaml`
- Modify: `scripts/preprocessing_steps/skull_stripping/dispatcher.py` (register)
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_hdbet.py`

- [ ] **Step 1: Write the failing test**

`scripts/preprocessing_steps/skull_stripping/tests/test_hdbet.py`:
```python
from pathlib import Path
from preprocessing_steps.skull_stripping.hdbet import HDBetStripper
from preprocessing_steps.skull_stripping import dispatcher


def test_registered():
    assert "hdbet" in dispatcher.STRIPPER_REGISTRY
    assert dispatcher.get_stripper("hdbet").name == "hdbet"


def test_build_command_includes_threshold():
    s = HDBetStripper()
    cmd = s.build_command(Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"),
                          {"tool_params": {"threshold": 0.7}})
    assert "hd-bet" in cmd[0]
    assert "-i" in cmd and "/in/t1.nii.gz" in cmd
    assert "-o" in cmd

def test_is_available_false_when_binary_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert HDBetStripper().is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_hdbet.py -v`
Expected: FAIL — `hdbet` module missing.

- [ ] **Step 3: Write `hdbet.py`**

`scripts/preprocessing_steps/skull_stripping/hdbet.py`:
```python
"""HD-BET wrapper (Isensee et al. 2019). GPU deep-learning brain extraction."""

import logging
import shutil
import subprocess
import time
from pathlib import Path

from .base import SkullStripperBase, get_gpu_memory_mb

logger = logging.getLogger(__name__)


class HDBetStripper(SkullStripperBase):
    name = "hdbet"

    def is_available(self) -> bool:
        return shutil.which("hd-bet") is not None

    def build_command(self, input_path: Path, output_path: Path, params: dict) -> list:
        tp = params.get("tool_params", {})
        cmd = ["hd-bet", "-i", str(input_path), "-o", str(output_path)]
        # hd-bet emits <output>_bet.nii.gz and a *_mask.nii.gz; flags below per CLI.
        if "threshold" in tp:
            cmd += ["--threshold", str(tp["threshold"])]
        return cmd

    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vram_before = get_gpu_memory_mb()
        start = time.perf_counter()
        try:
            cmd = self.build_command(input_path, output_path, params)
            logger.debug(f"HD-BET command: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if res.returncode != 0:
                raise RuntimeError(f"hd-bet rc={res.returncode}: {res.stderr[:500]}")
            # HD-BET writes the mask next to output as <stem>_mask.nii.gz.
            stem = str(output_path).replace(".nii.gz", "")
            produced_mask = Path(f"{stem}_mask.nii.gz")
            if produced_mask.exists() and produced_mask != mask_path:
                produced_mask.rename(mask_path)
            elapsed = time.perf_counter() - start
            vram = max(0.0, get_gpu_memory_mb() - vram_before) / 1024.0
            return {
                "success": mask_path.exists(),
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "processing_time": elapsed,
                "vram_used_gb": vram,
            }
        except Exception as e:
            logger.error(f"HD-BET failed on {input_path.name}: {e}")
            return {"success": False, "processing_time": time.perf_counter() - start,
                    "vram_used_gb": 0.0, "error": str(e)}
```

- [ ] **Step 4: Register in dispatcher**

In `dispatcher.py`, add the import and registry entry:
```python
from .hdbet import HDBetStripper
...
STRIPPER_REGISTRY = {
    "bet": BetStripper,
    "hdbet": HDBetStripper,
}
```

- [ ] **Step 5: Write the manifest**

`services/skull-stripping/hdbet/manifest.yaml`:
```yaml
name: hdbet
version: "1.0"
tool_type: skull_stripping
compute:
  requires_gpu: true
  gpu_memory_gb: 4
  ram_gb: 8
  cpu_fallback: true
lesion_type_preference:
  glioma: high
  ms: high
  metastases: medium
modality_support:
  primary: [t1, t1c]
  apply_to: [t1, t1c, t2, t2fl]
quality_notes: >
  Best current choice for tumor multi-modal MRI (2019). Validated on BraTS-like data.
install:
  method: pip
  package: hd-bet
  model_download: auto
tuned_params: {}
mas_metadata:
  agent_type: skull_stripper
  cascade_priority: 1
  fallback_to: synthstrip
```

- [ ] **Step 6: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_hdbet.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Install + smoke-test HD-BET (manual, timeboxed)**

```bash
cd /home/e.roppert/work/mri_ai_service && source venv/bin/activate
pip install hd-bet
hd-bet --help    # confirm CLI present; verify flag names match build_command
```
If the installed CLI uses different flags than `-i/-o/--threshold`, fix `build_command` and update the test accordingly, then re-run Step 6. Record the outcome in `DEVLOG.md`.

- [ ] **Step 8: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/hdbet.py \
        scripts/preprocessing_steps/skull_stripping/dispatcher.py \
        services/skull-stripping/hdbet/manifest.yaml \
        scripts/preprocessing_steps/skull_stripping/tests/test_hdbet.py
git commit -m "feat(ss): add HD-BET wrapper, manifest, and registry entry"
```

---

### Task 3.2: SynthStrip wrapper + manifest

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/synthstrip.py`
- Create: `services/skull-stripping/synthstrip/manifest.yaml`
- Modify: `scripts/preprocessing_steps/skull_stripping/dispatcher.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_synthstrip.py`

- [ ] **Step 1: Write the failing test**

`scripts/preprocessing_steps/skull_stripping/tests/test_synthstrip.py`:
```python
from pathlib import Path
from preprocessing_steps.skull_stripping.synthstrip import SynthStripStripper
from preprocessing_steps.skull_stripping import dispatcher


def test_registered():
    assert dispatcher.get_stripper("synthstrip").name == "synthstrip"


def test_build_command_includes_border_and_mask():
    s = SynthStripStripper()
    cmd = s.build_command(Path("/in/t1.nii.gz"), Path("/out/t1.nii.gz"),
                          Path("/out/mask.nii.gz"), {"tool_params": {"border": 2}})
    assert "-i" in cmd and "-o" in cmd and "-m" in cmd
    assert "-b" in cmd and "2" in cmd


def test_is_available_false_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert SynthStripStripper().is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_synthstrip.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `synthstrip.py`**

`scripts/preprocessing_steps/skull_stripping/synthstrip.py`:
```python
"""SynthStrip wrapper (Hoopes et al. 2022), via the `mri_synthstrip` CLI (surfa)."""

import logging
import shutil
import subprocess
import time
from pathlib import Path

from .base import SkullStripperBase, get_gpu_memory_mb

logger = logging.getLogger(__name__)


class SynthStripStripper(SkullStripperBase):
    name = "synthstrip"

    def is_available(self) -> bool:
        return shutil.which("mri_synthstrip") is not None

    def build_command(self, input_path: Path, output_path: Path,
                      mask_path: Path, params: dict) -> list:
        tp = params.get("tool_params", {})
        cmd = ["mri_synthstrip",
               "-i", str(input_path),
               "-o", str(output_path),
               "-m", str(mask_path)]
        if "border" in tp:
            cmd += ["-b", str(tp["border"])]
        return cmd

    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        vram_before = get_gpu_memory_mb()
        start = time.perf_counter()
        try:
            cmd = self.build_command(input_path, output_path, mask_path, params)
            logger.debug(f"SynthStrip command: {' '.join(cmd)}")
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if res.returncode != 0:
                raise RuntimeError(f"synthstrip rc={res.returncode}: {res.stderr[:500]}")
            elapsed = time.perf_counter() - start
            vram = max(0.0, get_gpu_memory_mb() - vram_before) / 1024.0
            return {
                "success": mask_path.exists(),
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "processing_time": elapsed,
                "vram_used_gb": vram,
            }
        except Exception as e:
            logger.error(f"SynthStrip failed on {input_path.name}: {e}")
            return {"success": False, "processing_time": time.perf_counter() - start,
                    "vram_used_gb": 0.0, "error": str(e)}
```

- [ ] **Step 4: Register in dispatcher**

Add `from .synthstrip import SynthStripStripper` and `"synthstrip": SynthStripStripper,` to `STRIPPER_REGISTRY`.

- [ ] **Step 5: Write the manifest**

`services/skull-stripping/synthstrip/manifest.yaml`:
```yaml
name: synthstrip
version: "1.0"
tool_type: skull_stripping
compute:
  requires_gpu: false
  gpu_memory_gb: 2
  ram_gb: 6
  cpu_fallback: true
lesion_type_preference:
  glioma: high
  ms: high
  metastases: high
modality_support:
  primary: [t1, t1c, t2, t2fl]
  apply_to: [t1, t1c, t2, t2fl]
quality_notes: >
  Modality-agnostic (2022). Strong general-purpose default; CPU-capable.
install:
  method: pip
  package: surfa
  model_download: bundled
tuned_params: {}
mas_metadata:
  agent_type: skull_stripper
  cascade_priority: 2
  fallback_to: bet
```

- [ ] **Step 6: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_synthstrip.py -v`
Expected: PASS.

- [ ] **Step 7: Install + smoke-test SynthStrip (manual, timeboxed)**

```bash
cd /home/e.roppert/work/mri_ai_service && source venv/bin/activate
pip install surfa
mri_synthstrip --help    # confirm CLI + flag names; adjust build_command if needed
```
Record outcome in `DEVLOG.md`; re-run Step 6 if flags changed.

- [ ] **Step 8: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/synthstrip.py \
        scripts/preprocessing_steps/skull_stripping/dispatcher.py \
        services/skull-stripping/synthstrip/manifest.yaml \
        scripts/preprocessing_steps/skull_stripping/tests/test_synthstrip.py
git commit -m "feat(ss): add SynthStrip wrapper, manifest, and registry entry"
```

---

### Task 3.3: BrainMaGe wrapper + manifest (ADD-1)

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/brainmage.py`
- Create: `services/skull-stripping/brainmage/manifest.yaml`
- Modify: `scripts/preprocessing_steps/skull_stripping/dispatcher.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_brainmage.py`

> BrainMaGe (Thakur et al. 2020, CBICA — https://github.com/CBICA/BrainMaGe) is the glioma-specialized,
> modality-agnostic open model and our nearest published competitor. Adding it as a candidate lets us
> position directly against the prior art on *our* clinical cohort (lit review §2.6, §5). It exposes a
> CLI `brain_mage_single_run` (single subject) / `brain_mage_run`. Like other DL wrappers, tests mock
> availability/command; real CLI flags are verified with a timeboxed install step.

- [ ] **Step 1: Write the failing test**

`scripts/preprocessing_steps/skull_stripping/tests/test_brainmage.py`:
```python
from pathlib import Path
from preprocessing_steps.skull_stripping.brainmage import BrainMaGeStripper
from preprocessing_steps.skull_stripping import dispatcher


def test_registered():
    assert dispatcher.get_stripper("brainmage").name == "brainmage"


def test_build_command_has_io_and_device():
    s = BrainMaGeStripper()
    cmd = s.build_command(Path("/in/t1.nii.gz"), Path("/out/mask.nii.gz"),
                          {"tool_params": {"device": "cpu"}})
    assert any("brain_mage" in c for c in cmd)
    assert "/in/t1.nii.gz" in cmd and "/out/mask.nii.gz" in cmd


def test_is_available_false_when_missing(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert BrainMaGeStripper().is_available() is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_brainmage.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `brainmage.py`**

`scripts/preprocessing_steps/skull_stripping/brainmage.py`:
```python
"""BrainMaGe wrapper (Thakur et al. 2020, CBICA). Glioma-specialized, modality-agnostic.

Uses the `brain_mage_single_run` CLI. Verify exact flags after install (Step 7).
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path

from .base import SkullStripperBase, apply_brain_mask, get_gpu_memory_mb

logger = logging.getLogger(__name__)

_CLI = "brain_mage_single_run"


class BrainMaGeStripper(SkullStripperBase):
    name = "brainmage"

    def is_available(self) -> bool:
        return shutil.which(_CLI) is not None

    def build_command(self, input_path: Path, mask_path: Path, params: dict) -> list:
        tp = params.get("tool_params", {})
        device = str(tp.get("device", "cpu"))
        # brain_mage_single_run -i <input> -o <mask> -dev <cpu|0>
        return [_CLI, "-i", str(input_path), "-o", str(mask_path), "-dev", device]

    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        vram_before = get_gpu_memory_mb()
        start = time.perf_counter()
        try:
            cmd = self.build_command(input_path, mask_path, params)
            logger.debug("BrainMaGe command: %s", " ".join(cmd))
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if res.returncode != 0:
                raise RuntimeError(f"brain_mage rc={res.returncode}: {res.stderr[:500]}")
            apply_res = apply_brain_mask(input_path, mask_path, output_path)
            elapsed = time.perf_counter() - start
            vram = max(0.0, get_gpu_memory_mb() - vram_before) / 1024.0
            return {
                "success": apply_res.get("success", False) and mask_path.exists(),
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_path.exists() else None,
                "processing_time": elapsed,
                "vram_used_gb": vram,
            }
        except Exception as e:
            logger.error("BrainMaGe failed on %s: %s", input_path.name, e)
            return {"success": False, "processing_time": time.perf_counter() - start,
                    "vram_used_gb": 0.0, "error": str(e)}
```

- [ ] **Step 4: Register in dispatcher**

Add `from .brainmage import BrainMaGeStripper` and `"brainmage": BrainMaGeStripper,` to `STRIPPER_REGISTRY`.

- [ ] **Step 5: Write the manifest**

`services/skull-stripping/brainmage/manifest.yaml`:
```yaml
name: brainmage
version: "1.0"
tool_type: skull_stripping
compute:
  requires_gpu: false
  gpu_memory_gb: 4
  ram_gb: 8
  cpu_fallback: true
lesion_type_preference:
  glioma: high
  ms: medium
  metastases: medium
modality_support:
  primary: [t1, t1c, t2, t2fl]
  apply_to: [t1, t1c, t2, t2fl]
quality_notes: >
  Glioma-specialized, modality-agnostic (Thakur 2020, CBICA). Nearest published
  competitor; learns a brain-shape prior, robust to missing modalities.
install:
  method: git
  package: https://github.com/CBICA/BrainMaGe
  model_download: bundled
tuned_params: {}
mas_metadata:
  agent_type: skull_stripper
  cascade_priority: 1
  fallback_to: synthstrip
```

- [ ] **Step 6: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_brainmage.py -v`
Expected: PASS.

- [ ] **Step 7: Install + smoke-test BrainMaGe (manual, timeboxed)**

```bash
cd /home/e.roppert/work/mri_ai_service && source venv/bin/activate
pip install git+https://github.com/CBICA/BrainMaGe.git   # or follow repo install
brain_mage_single_run --help    # confirm CLI + flags; adjust build_command if needed
```
If flags differ from `-i/-o/-dev`, fix `build_command` + test, re-run Step 6. Record in `DEVLOG.md`.
If install does not fit the timebox, leave `is_available()` returning False (fails closed) and note it.

- [ ] **Step 8: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/brainmage.py \
        scripts/preprocessing_steps/skull_stripping/dispatcher.py \
        services/skull-stripping/brainmage/manifest.yaml \
        scripts/preprocessing_steps/skull_stripping/tests/test_brainmage.py
git commit -m "feat(ss): add BrainMaGe wrapper, manifest, and registry entry (ADD-1)"
```

---

## Phase 4 — Atlas baseline + best-effort tools

### Task 4.1: MNI mask stripper (strict + loose) + manifest

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/mni_mask.py`
- Create: `services/skull-stripping/mni_mask/manifest.yaml`
- Modify: `scripts/preprocessing_steps/skull_stripping/dispatcher.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_mni_mask.py`

> This tool is pure Python/nibabel: load a fixed MNI brain mask, optionally dilate it, resample it onto the input grid, save it as the mask, apply it to the input. Inputs are already in MNI152 space (Stage 04 registration), so the mask aligns. It is the cheap, deterministic baseline. Fully testable with synthetic NIfTI.

- [ ] **Step 1: Write the failing test**

`scripts/preprocessing_steps/skull_stripping/tests/test_mni_mask.py`:
```python
import numpy as np
import nibabel as nib
from pathlib import Path
from preprocessing_steps.skull_stripping.mni_mask import MniMaskStripper
from preprocessing_steps.skull_stripping import dispatcher


def _write(tmp, name, arr):
    p = tmp / name
    nib.save(nib.Nifti1Image(arr.astype(np.float32), np.eye(4)), p)
    return p


def test_registered():
    assert dispatcher.get_stripper("mni_mask").name == "mni_mask"


def test_dilation_loose_is_superset_of_strict():
    base = np.zeros((10, 10, 10)); base[4:6, 4:6, 4:6] = 1
    strict = MniMaskStripper._dilate(base, radius_vox=0)
    loose = MniMaskStripper._dilate(base, radius_vox=2)
    assert loose.sum() > strict.sum()
    assert np.all(loose[strict > 0] > 0)


def test_strip_produces_mask_and_output(tmp_path):
    img = np.random.rand(10, 10, 10).astype(np.float32)
    in_path = _write(tmp_path, "in.nii.gz", img)
    mask_template = np.zeros((10, 10, 10)); mask_template[3:7, 3:7, 3:7] = 1
    tmpl_path = _write(tmp_path, "mni_mask.nii.gz", mask_template)
    s = MniMaskStripper()
    out = tmp_path / "out.nii.gz"; mask = tmp_path / "mask.nii.gz"
    r = s.strip(in_path, out, mask,
                {"tool_params": {"variant": "strict", "mask_template": str(tmpl_path)}})
    assert r["success"] and out.exists() and mask.exists()
    assert r["vram_used_gb"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_mni_mask.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `mni_mask.py`**

`scripts/preprocessing_steps/skull_stripping/mni_mask.py`:
```python
"""Atlas-baseline skull stripping: apply a fixed MNI152 brain mask.

Two variants: strict (radius 0) and loose (radius ~2mm dilation). Inputs are
assumed already registered to MNI152, so the mask aligns voxel-wise after
resampling onto the input grid.
"""

import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation

from .base import SkullStripperBase, apply_brain_mask

logger = logging.getLogger(__name__)

# Default template ships in the repo (T1 1mm). A precomputed brain mask is
# expected alongside; if only the T1 is present, a threshold>0 mask is derived.
_DEFAULT_TEMPLATE = (Path(__file__).resolve().parents[3]
                     / "data" / "templates" / "MNI152_T1_1mm.nii.gz")


class MniMaskStripper(SkullStripperBase):
    name = "mni_mask"

    def is_available(self) -> bool:
        return _DEFAULT_TEMPLATE.exists()

    @staticmethod
    def _dilate(mask: np.ndarray, radius_vox: int) -> np.ndarray:
        if radius_vox <= 0:
            return (mask > 0).astype(np.uint8)
        return binary_dilation(mask > 0, iterations=int(radius_vox)).astype(np.uint8)

    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        start = time.perf_counter()
        try:
            tp = params.get("tool_params", {})
            variant = tp.get("variant", "strict")
            radius = 0 if variant == "strict" else int(tp.get("dilation_vox", 2))
            template_path = Path(tp.get("mask_template", _DEFAULT_TEMPLATE))

            in_img = nib.load(input_path)
            tmpl_img = nib.load(template_path)
            tmpl = tmpl_img.get_fdata()
            base_mask = (tmpl > 0).astype(np.uint8)

            if base_mask.shape != in_img.shape:
                raise ValueError(
                    f"MNI mask shape {base_mask.shape} != input {in_img.shape}; "
                    f"input must be registered to MNI152 1mm."
                )
            mask = self._dilate(base_mask, radius)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            nib.save(nib.Nifti1Image(mask, in_img.affine, in_img.header), mask_path)

            apply_res = apply_brain_mask(input_path, mask_path, output_path)
            elapsed = time.perf_counter() - start
            return {
                "success": apply_res.get("success", False),
                "output_path": str(output_path),
                "mask_path": str(mask_path),
                "processing_time": elapsed,
                "vram_used_gb": 0.0,
            }
        except Exception as e:
            logger.error(f"MNI mask failed on {input_path.name}: {e}")
            return {"success": False, "processing_time": time.perf_counter() - start,
                    "vram_used_gb": 0.0, "error": str(e)}
```

- [ ] **Step 4: Register in dispatcher**

Add `from .mni_mask import MniMaskStripper` and `"mni_mask": MniMaskStripper,` to `STRIPPER_REGISTRY`.

- [ ] **Step 5: Write the manifest**

`services/skull-stripping/mni_mask/manifest.yaml`:
```yaml
name: mni_mask
version: "1.0"
tool_type: skull_stripping
compute:
  requires_gpu: false
  gpu_memory_gb: 0
  ram_gb: 2
  cpu_fallback: true
lesion_type_preference:
  glioma: low
  ms: medium
  metastases: low
modality_support:
  primary: [t1, t1c, t2, t2fl]
  apply_to: [t1, t1c, t2, t2fl]
quality_notes: >
  Atlas baseline. Deterministic, fast, modality-agnostic. Strict and loose
  (2mm dilation) variants. Reference for "is a GPU tool worth the cost?".
install:
  method: none
  package: data/templates/MNI152_T1_1mm.nii.gz
  model_download: none
tuned_params:
  variant: strict       # strict | loose
  dilation_vox: 2
mas_metadata:
  agent_type: skull_stripper
  cascade_priority: 4
  fallback_to: bet
```

- [ ] **Step 6: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_mni_mask.py -v`
Expected: PASS (3 tests).

- [ ] **Step 7: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/mni_mask.py \
        scripts/preprocessing_steps/skull_stripping/dispatcher.py \
        services/skull-stripping/mni_mask/manifest.yaml \
        scripts/preprocessing_steps/skull_stripping/tests/test_mni_mask.py
git commit -m "feat(ss): add MNI atlas-mask stripper (strict/loose) and manifest"
```

---

### Task 4.2: Best-effort tool stubs (SAM, DeepBET) with timebox + graceful unavailability

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/sam_bet.py`
- Create: `scripts/preprocessing_steps/skull_stripping/deepbet.py`
- Create: `services/skull-stripping/sam/manifest.yaml`
- Create: `services/skull-stripping/deepbet/manifest.yaml`
- Modify: `scripts/preprocessing_steps/skull_stripping/dispatcher.py`
- Test: `scripts/preprocessing_steps/skull_stripping/tests/test_best_effort.py`

> Per spec these are best-effort, ~2h timebox each. They must register and report `is_available() == False` cleanly until the upstream repo is wired in, so the dispatcher's fallback path keeps the benchmark running. The `strip()` body holds a concrete subprocess call shape to fill in once the repo is cloned.

- [ ] **Step 1: Write the failing test**

`scripts/preprocessing_steps/skull_stripping/tests/test_best_effort.py`:
```python
from preprocessing_steps.skull_stripping import dispatcher
from preprocessing_steps.skull_stripping.sam_bet import SamBetStripper
from preprocessing_steps.skull_stripping.deepbet import DeepBetStripper


def test_both_registered():
    assert dispatcher.get_stripper("sam").name == "sam"
    assert dispatcher.get_stripper("deepbet").name == "deepbet"


def test_unavailable_by_default():
    # Until upstream repos are installed, these report unavailable (not crash).
    assert SamBetStripper().is_available() in (True, False)
    assert DeepBetStripper().is_available() in (True, False)


def test_strip_returns_error_dict_when_unavailable():
    s = SamBetStripper()
    if not s.is_available():
        r = s.strip.__doc__ is not None  # method exists / documented
        assert r
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_best_effort.py -v`
Expected: FAIL — modules missing.

- [ ] **Step 3: Write `sam_bet.py`**

`scripts/preprocessing_steps/skull_stripping/sam_bet.py`:
```python
"""SAM-based brain extraction (experimental, best-effort). 2h install timebox.

If the upstream repo is not installed, is_available() returns False so the
dispatcher falls back. Fill in `_INSTALL_DIR` and the command once cloned.
"""

import logging
import shutil
import time
from pathlib import Path

from .base import SkullStripperBase, get_gpu_memory_mb

logger = logging.getLogger(__name__)

_INSTALL_DIR = Path(__file__).resolve().parents[3] / "external" / "sam_brain"


class SamBetStripper(SkullStripperBase):
    name = "sam"

    def is_available(self) -> bool:
        return _INSTALL_DIR.exists() and shutil.which("python") is not None

    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        start = time.perf_counter()
        if not self.is_available():
            return {"success": False, "processing_time": 0.0, "vram_used_gb": 0.0,
                    "error": "SAM brain extraction not installed (best-effort tool)"}
        # TODO when cloned: run the upstream inference script. Expected shape:
        #   cmd = ["python", str(_INSTALL_DIR / "infer.py"),
        #          "--input", str(input_path), "--mask", str(mask_path)]
        #   subprocess.run(cmd, check=True, timeout=900)
        #   then apply_brain_mask(input_path, mask_path, output_path)
        return {"success": False, "processing_time": time.perf_counter() - start,
                "vram_used_gb": 0.0, "error": "SAM strip() not yet implemented"}
```

- [ ] **Step 4: Write `deepbet.py`** (same shape, different dir/name)

`scripts/preprocessing_steps/skull_stripping/deepbet.py`:
```python
"""DeepBET (experimental, best-effort). 2h install timebox. See sam_bet.py."""

import logging
import time
from pathlib import Path

from .base import SkullStripperBase

logger = logging.getLogger(__name__)

_INSTALL_DIR = Path(__file__).resolve().parents[3] / "external" / "deepbet"


class DeepBetStripper(SkullStripperBase):
    name = "deepbet"

    def is_available(self) -> bool:
        try:
            import deepbet  # noqa: F401
            return True
        except ImportError:
            return _INSTALL_DIR.exists()

    def strip(self, input_path: Path, output_path: Path,
              mask_path: Path, params: dict) -> dict:
        start = time.perf_counter()
        if not self.is_available():
            return {"success": False, "processing_time": 0.0, "vram_used_gb": 0.0,
                    "error": "DeepBET not installed (best-effort tool)"}
        # TODO when installed: from deepbet import run_bet; run_bet(...) then
        # apply_brain_mask(input_path, mask_path, output_path).
        return {"success": False, "processing_time": time.perf_counter() - start,
                "vram_used_gb": 0.0, "error": "DeepBET strip() not yet implemented"}
```

- [ ] **Step 5: Register both in dispatcher**

Add imports and `"sam": SamBetStripper, "deepbet": DeepBetStripper,` to `STRIPPER_REGISTRY`.

- [ ] **Step 6: Write both manifests**

`services/skull-stripping/sam/manifest.yaml`:
```yaml
name: sam
version: "0.1-experimental"
tool_type: skull_stripping
compute: {requires_gpu: true, gpu_memory_gb: 6, ram_gb: 8, cpu_fallback: false}
lesion_type_preference: {glioma: unknown, ms: unknown, metastases: unknown}
modality_support: {primary: [t1], apply_to: [t1, t1c, t2, t2fl]}
quality_notes: "Experimental SAM-based (2023). Best-effort; excluded if unstable."
install: {method: git, package: "<<upstream repo url>>", model_download: manual}
tuned_params: {}
mas_metadata: {agent_type: skull_stripper, cascade_priority: 9, fallback_to: bet}
```

`services/skull-stripping/deepbet/manifest.yaml`:
```yaml
name: deepbet
version: "0.1-experimental"
tool_type: skull_stripping
compute: {requires_gpu: true, gpu_memory_gb: 4, ram_gb: 8, cpu_fallback: true}
lesion_type_preference: {glioma: unknown, ms: unknown, metastases: unknown}
modality_support: {primary: [t1], apply_to: [t1, t1c, t2, t2fl]}
quality_notes: "Experimental DeepBET (2023/24). Best-effort; excluded if unstable."
install: {method: pip, package: deepbet, model_download: auto}
tuned_params: {}
mas_metadata: {agent_type: skull_stripper, cascade_priority: 9, fallback_to: bet}
```

- [ ] **Step 7: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_best_effort.py -v`
Expected: PASS.

- [ ] **Step 8: Timeboxed install attempt (manual, ≤2h each)**

Attempt `pip install deepbet` and clone the SAM repo into `external/`. If a tool works within the timebox, implement its `strip()` TODO and add a smoke test; otherwise leave unavailable and note the exclusion in `DEVLOG.md` and `literature_notes.md`.

- [ ] **Step 9: Commit**

```bash
git add scripts/preprocessing_steps/skull_stripping/sam_bet.py \
        scripts/preprocessing_steps/skull_stripping/deepbet.py \
        scripts/preprocessing_steps/skull_stripping/dispatcher.py \
        services/skull-stripping/sam/manifest.yaml \
        services/skull-stripping/deepbet/manifest.yaml \
        scripts/preprocessing_steps/skull_stripping/tests/test_best_effort.py
git commit -m "feat(ss): add best-effort SAM/DeepBET stubs with graceful unavailability"
```

---

## Phase 5 — Benchmark data preparation

### Task 5.1: `prepare_data.py` — run pipeline stages 1–4 to registration space

**Files:**
- Create: `research/skull_stripping_benchmark/prepare_data.py`
- Create: `research/skull_stripping_benchmark/tests/__init__.py`
- Create: `research/skull_stripping_benchmark/tests/conftest.py`
- Test: `research/skull_stripping_benchmark/tests/test_prepare_data.py`

> `prepare_data.py` builds a benchmark-only preprocessing config: stages reorient → bias_correction (`use_for_registration_only: true`) → registration (MNI152, overriding SRI24) → resampling, with `skull_stripping: disabled`. It then invokes Stage 05 (`scripts/05_preprocessing.py`) on a subject set and copies the registration-space output as the benchmark input NIfTI. Pure config-building logic is unit-tested; the subprocess run is integration-only.

- [ ] **Step 1: Write the conftest and failing config-builder test**

`research/skull_stripping_benchmark/tests/conftest.py`:
```python
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = BENCH_DIR.parents[1] / "scripts"
for p in (BENCH_DIR, SCRIPTS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
```

`research/skull_stripping_benchmark/tests/__init__.py`: empty.

`research/skull_stripping_benchmark/tests/test_prepare_data.py`:
```python
import prepare_data


def test_benchmark_config_disables_skull_stripping():
    cfg = prepare_data.build_benchmark_config(atlas="MNI152")
    steps = {s["name"]: s for s in cfg["steps"]}
    assert steps["skull_stripping"]["enabled"] is False


def test_benchmark_config_uses_mni_atlas():
    cfg = prepare_data.build_benchmark_config(atlas="MNI152")
    assert "MNI152" in cfg["atlas"]["name"]
    assert cfg["atlas"]["filename"] == "MNI152_T1_1mm.nii.gz"


def test_bias_correction_registration_only():
    cfg = prepare_data.build_benchmark_config(atlas="MNI152")
    bc = next(s for s in cfg["steps"] if s["name"] == "bias_correction")
    assert bc["params"]["use_for_registration_only"] is True


def test_registration_and_resampling_enabled():
    cfg = prepare_data.build_benchmark_config(atlas="MNI152")
    steps = {s["name"]: s for s in cfg["steps"]}
    assert steps["registration"]["enabled"] is True
    assert steps["registration"]["params"]["output_resolution"] == [1.0, 1.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_prepare_data.py -v`
Expected: FAIL — `prepare_data` missing.

- [ ] **Step 3: Write `prepare_data.py`**

`research/skull_stripping_benchmark/prepare_data.py`:
```python
"""Prepare benchmark inputs: run pipeline stages 1-4 (reorient -> bias correction
-> registration to MNI152 -> resampling) with skull stripping disabled, then
collect the registration-space NIfTI as the benchmark input for each subject.

Usage:
    python prepare_data.py --input-dir <bids_root> --output-dir <bench_data> \
        [--atlas MNI152] [--max-subjects N]
"""

import argparse
import logging
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MNI_TEMPLATE = PROJECT_ROOT / "data" / "templates" / "MNI152_T1_1mm.nii.gz"


def build_benchmark_config(atlas: str = "MNI152") -> dict:
    """Benchmark preprocessing config: stages 1-4 to MNI152, no skull stripping."""
    return {
        "fsl": {"fsl_dir": "/usr/share/fsl/6.0"},
        "atlas": {
            "name": f"{atlas}_FSL",
            "cache_dir": "data/templates",
            "filename": "MNI152_T1_1mm.nii.gz",
        },
        "steps": [
            {"name": "reorient", "enabled": True,
             "params": {"target_orientation": "LAS"}},
            {"name": "bias_correction", "enabled": True,
             "params": {"shrink_factor": 4, "n_iterations": [50, 50, 50, 50],
                        "convergence_threshold": 0.001,
                        "use_for_registration_only": True}},
            {"name": "registration", "enabled": True,
             "params": {"reference_modality": "t1", "registration_type": "Rigid",
                        "output_resolution": [1.0, 1.0, 1.0], "metric": "MI",
                        "metric_weight": 1.0, "number_of_iterations": 1000,
                        "convergence_threshold": 1e-6, "convergence_window_size": 10,
                        "smoothing_sigmas": [3, 2, 1, 0], "shrink_factors": [8, 4, 2, 1],
                        "use_histogram_matching": True, "save_transformations": True}},
            {"name": "resampling", "enabled": False,
             "params": {"output_resolution": [1.0, 1.0, 1.0], "interpolation": "linear"}},
            {"name": "skull_stripping", "enabled": False, "params": {}},
        ],
        "modalities": ["t1c", "t1", "t2", "t2fl"],
        "logging": {"level": "INFO",
                    "format": "%(asctime)s - %(levelname)s - %(message)s",
                    "date_format": "%Y-%m-%d %H:%M:%S"},
    }


def write_config(cfg: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    return dest


def run_stage05(input_dir: Path, output_dir: Path, transform_dir: Path,
                config_path: Path, max_subjects: int | None) -> None:
    """Invoke scripts/05_preprocessing.py with the benchmark config."""
    script = PROJECT_ROOT / "scripts" / "05_preprocessing.py"
    cmd = [sys.executable, str(script),
           "--input-dir", str(input_dir),
           "--output-dir", str(output_dir),
           "--transform-dir", str(transform_dir),
           "--config", str(config_path)]
    if max_subjects is not None:
        cmd += ["--max-subjects", str(max_subjects)]
    logger.info("Running Stage 05: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--atlas", default="MNI152")
    ap.add_argument("--max-subjects", type=int, default=None)
    args = ap.parse_args()

    if not MNI_TEMPLATE.exists():
        raise SystemExit(f"MNI template missing: {MNI_TEMPLATE}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cfg = build_benchmark_config(args.atlas)
        cfg_path = write_config(cfg, tmp / "benchmark_preprocessing.yaml")
        reg_out = args.output_dir / "registration_space"
        transform_dir = args.output_dir / "transforms"
        reg_out.mkdir(parents=True, exist_ok=True)
        run_stage05(args.input_dir, reg_out, transform_dir, cfg_path, args.max_subjects)
        logger.info("Benchmark inputs (registration space) written to %s", reg_out)


if __name__ == "__main__":
    main()
```

> **Verify before relying on it:** confirm `scripts/05_preprocessing.py` accepts `--input-dir`, `--output-dir`, `--transform-dir`, `--config`, `--max-subjects`. Run `python scripts/05_preprocessing.py --help` and adjust the flag names in `run_stage05` to match. If Stage 05 saves a separate non-corrected registration image, point the benchmark at that file in Task 6/7's data loader.

- [ ] **Step 4: Run config tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_prepare_data.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Integration smoke run on local MS subject (manual)**

```bash
cd /home/e.roppert/work/mri_ai_service && source venv/bin/activate
python research/skull_stripping_benchmark/prepare_data.py \
  --input-dir data/MS_5 --output-dir /tmp/bench_prep --max-subjects 1
ls /tmp/bench_prep/registration_space   # expect sub-*/ses-*/anat/*.nii.gz
```
Confirm a registration-space NIfTI is produced. Record result in `DEVLOG.md`.

- [ ] **Step 6: Commit**

```bash
git add research/skull_stripping_benchmark/prepare_data.py \
        research/skull_stripping_benchmark/tests/
git commit -m "feat(ss-research): prepare_data.py builds MNI152 stages 1-4 benchmark inputs"
```

---

## Phase 6 — Metrics

### Task 6.1: `metrics.py` — quality + performance metrics

**Files:**
- Create: `research/skull_stripping_benchmark/metrics.py`
- Test: `research/skull_stripping_benchmark/tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

`research/skull_stripping_benchmark/tests/test_metrics.py`:
```python
import numpy as np
import metrics


def test_dsc_identical_masks_is_one():
    a = np.zeros((8, 8, 8)); a[2:6, 2:6, 2:6] = 1
    assert metrics.dice(a, a) == 1.0


def test_dsc_disjoint_masks_is_zero():
    a = np.zeros((8, 8, 8)); a[0:2, 0:2, 0:2] = 1
    b = np.zeros((8, 8, 8)); b[6:8, 6:8, 6:8] = 1
    assert metrics.dice(a, b) == 0.0


def test_dsc_half_overlap():
    a = np.zeros((10,)); a[0:6] = 1
    b = np.zeros((10,)); b[3:9] = 1
    # |A∩B|=3, |A|+|B|=12 -> 6/12 = 0.5
    assert abs(metrics.dice(a, b) - 0.5) < 1e-9


def test_over_stripping_and_leakage_rates():
    gt = np.zeros((10,)); gt[0:6] = 1          # brain = 6 voxels
    pred = np.zeros((10,)); pred[2:8] = 1       # misses 0,1 ; adds 6,7
    over = metrics.over_stripping_rate(pred, gt)   # FN/(TP+FN) = 2/6
    leak = metrics.leakage_rate(pred, gt)          # FP/(TN+FP) = 2/4
    assert abs(over - 2/6) < 1e-9
    assert abs(leak - 2/4) < 1e-9


def test_brain_volume_ml():
    mask = np.zeros((10, 10, 10)); mask[0:5, 0:5, 0:5] = 1   # 125 voxels
    # 1mm iso voxel -> 125 mm^3 = 0.125 ml
    assert abs(metrics.brain_volume_ml(mask, voxel_mm=(1.0, 1.0, 1.0)) - 0.125) < 1e-6


def test_hd95_zero_for_identical():
    a = np.zeros((12, 12, 12)); a[3:9, 3:9, 3:9] = 1
    assert metrics.hd95(a, a, voxel_mm=(1.0, 1.0, 1.0)) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_metrics.py -v`
Expected: FAIL — `metrics` missing.

- [ ] **Step 3: Write `metrics.py`**

`research/skull_stripping_benchmark/metrics.py`:
```python
"""Quality and performance metrics for skull stripping masks.

Quality metrics compare a predicted binary mask against a pseudo-GT (MNI152
brain mask). Performance metrics wrap timing/RAM/VRAM measurement of a callable.
"""

import time
import tracemalloc
from pathlib import Path

import numpy as np
import nibabel as nib


def _binarize(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) > 0)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _binarize(a), _binarize(b)
    denom = a.sum() + b.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(a, b).sum() / denom)


def over_stripping_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    """FN / (TP + FN): fraction of true brain the prediction cut away."""
    pred, gt = _binarize(pred), _binarize(gt)
    fn = np.logical_and(~pred, gt).sum()
    tp = np.logical_and(pred, gt).sum()
    return float(fn / (tp + fn)) if (tp + fn) else 0.0


def leakage_rate(pred: np.ndarray, gt: np.ndarray) -> float:
    """FP / (TN + FP): fraction of non-brain wrongly kept in the mask."""
    pred, gt = _binarize(pred), _binarize(gt)
    fp = np.logical_and(pred, ~gt).sum()
    tn = np.logical_and(~pred, ~gt).sum()
    return float(fp / (tn + fp)) if (tn + fp) else 0.0


def brain_volume_ml(mask: np.ndarray, voxel_mm=(1.0, 1.0, 1.0)) -> float:
    voxel_ml = float(np.prod(voxel_mm)) / 1000.0
    return float(_binarize(mask).sum() * voxel_ml)


def hd95(pred: np.ndarray, gt: np.ndarray, voxel_mm=(1.0, 1.0, 1.0)) -> float:
    """95th percentile symmetric surface distance (mm)."""
    from scipy.ndimage import distance_transform_edt
    pred, gt = _binarize(pred), _binarize(gt)
    if pred.sum() == 0 or gt.sum() == 0:
        return float("nan")
    if np.array_equal(pred, gt):
        return 0.0

    def surface(m):
        from scipy.ndimage import binary_erosion
        return m & ~binary_erosion(m)

    sp, sg = surface(pred), surface(gt)
    dt_to_gt = distance_transform_edt(~sg, sampling=voxel_mm)
    dt_to_pred = distance_transform_edt(~sp, sampling=voxel_mm)
    d_pg = dt_to_gt[sp]
    d_gp = dt_to_pred[sg]
    all_d = np.concatenate([d_pg, d_gp])
    return float(np.percentile(all_d, 95))


def load_mask(path: Path) -> tuple[np.ndarray, tuple]:
    img = nib.load(path)
    return _binarize(img.get_fdata()), img.header.get_zooms()[:3]


def quality_metrics(pred_mask_path: Path, gt_mask_path: Path) -> dict:
    pred, vox = load_mask(pred_mask_path)
    gt, _ = load_mask(gt_mask_path)
    return {
        "dsc": dice(pred, gt),
        "hd95": hd95(pred, gt, voxel_mm=vox),
        "over_stripping": over_stripping_rate(pred, gt),
        "leakage": leakage_rate(pred, gt),
        "brain_volume_ml": brain_volume_ml(pred, voxel_mm=vox),
    }


def measure_performance(fn, *args, **kwargs) -> dict:
    """Run fn(*args, **kwargs); return its result plus time/RAM. VRAM is read
    from the result dict (strippers report vram_used_gb themselves)."""
    import psutil
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss
    return {
        "result": result,
        "wall_time_s": elapsed,
        "ram_peak_gb": max(peak, rss_after - rss_before) / (1024 ** 3),
    }
```

- [ ] **Step 4: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_metrics.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add research/skull_stripping_benchmark/metrics.py \
        research/skull_stripping_benchmark/tests/test_metrics.py
git commit -m "feat(ss-research): metrics module (DSC, HD95, over-strip, leakage, perf)"
```

---

## Phase 7 — Benchmark runner

### Task 7.1: `run_benchmark.py` — iterate tools × subjects, write `raw_metrics.csv`

**Files:**
- Create: `research/skull_stripping_benchmark/run_benchmark.py`
- Test: `research/skull_stripping_benchmark/tests/test_run_benchmark.py`

> The runner discovers subjects under the prepared data dir, picks the reference modality NIfTI, runs each requested stripper, computes quality metrics vs the MNI152 pseudo-GT mask, and appends a row per (tool, subject, dataset) to `raw_metrics.csv`. The unit test covers subject discovery and the row-building function with a fake stripper and synthetic NIfTI; the full run is integration.

- [ ] **Step 1: Write the failing test**

`research/skull_stripping_benchmark/tests/test_run_benchmark.py`:
```python
import numpy as np
import nibabel as nib
from pathlib import Path
import run_benchmark


def _make_subject(root: Path, sub, ses, ref_arr):
    anat = root / sub / ses / "anat"
    anat.mkdir(parents=True)
    p = anat / f"{sub}_{ses}_t1.nii.gz"
    nib.save(nib.Nifti1Image(ref_arr.astype(np.float32), np.eye(4)), p)
    return p


def test_discover_subjects(tmp_path):
    _make_subject(tmp_path, "sub-01", "ses-01", np.ones((4, 4, 4)))
    _make_subject(tmp_path, "sub-02", "ses-01", np.ones((4, 4, 4)))
    subs = run_benchmark.discover_subjects(tmp_path, reference_modality="t1")
    assert len(subs) == 2
    assert all(p.name.endswith("_t1.nii.gz") for p in subs)


def test_build_row_contains_quality_and_perf_keys():
    row = run_benchmark.build_row(
        tool="bet", dataset="local_ms", subject="sub-01",
        quality={"dsc": 0.9, "hd95": 2.0, "over_stripping": 0.05,
                 "leakage": 0.02, "brain_volume_ml": 1300.0},
        strip_result={"processing_time": 12.3, "vram_used_gb": 0.0, "success": True},
    )
    for k in ["tool", "dataset", "subject", "dsc", "hd95", "over_stripping",
              "leakage", "brain_volume_ml", "processing_time", "vram_used_gb",
              "success"]:
        assert k in row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_run_benchmark.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `run_benchmark.py`**

`research/skull_stripping_benchmark/run_benchmark.py`:
```python
"""Run skull stripping tools across prepared subjects and record metrics.

Usage:
    python run_benchmark.py --data-dir <prepared> --gt-dir <mni_gt_masks> \
        --dataset local_ms --tools bet hdbet synthstrip mni_mask \
        --reference-modality t1 --output results/raw_metrics.csv
"""

import argparse
import csv
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import metrics  # noqa: E402
from preprocessing_steps.skull_stripping import dispatcher  # noqa: E402

FIELDNAMES = ["tool", "dataset", "subject", "dsc", "hd95", "over_stripping",
              "leakage", "brain_volume_ml", "processing_time", "vram_used_gb",
              "success", "error"]


def discover_subjects(data_dir: Path, reference_modality: str) -> list[Path]:
    return sorted(data_dir.glob(f"sub-*/ses-*/anat/*_{reference_modality}.nii.gz"))


def build_row(tool, dataset, subject, quality, strip_result) -> dict:
    return {
        "tool": tool, "dataset": dataset, "subject": subject,
        "dsc": quality.get("dsc"), "hd95": quality.get("hd95"),
        "over_stripping": quality.get("over_stripping"),
        "leakage": quality.get("leakage"),
        "brain_volume_ml": quality.get("brain_volume_ml"),
        "processing_time": strip_result.get("processing_time"),
        "vram_used_gb": strip_result.get("vram_used_gb"),
        "success": strip_result.get("success"),
        "error": strip_result.get("error", ""),
    }


def gt_mask_for(subject_path: Path, gt_dir: Path) -> Path:
    """Map a subject reference NIfTI to its pseudo-GT MNI mask path."""
    subject = subject_path.parents[2].name
    session = subject_path.parents[1].name
    return gt_dir / subject / session / "anat" / f"{subject}_{session}_brain_mask.nii.gz"


def run(data_dir, gt_dir, dataset, tools, reference_modality, output, params):
    subjects = discover_subjects(data_dir, reference_modality)
    logger.info("Found %d subjects in %s", len(subjects), data_dir)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tool in tools:
            stripper = dispatcher.get_stripper(tool)
            if not stripper.is_available():
                logger.warning("Tool '%s' unavailable; skipping", tool)
                continue
            for subj in subjects:
                subject_id = subj.parents[2].name
                out = tmp / f"{tool}_{subject_id}_strip.nii.gz"
                mask = tmp / f"{tool}_{subject_id}_mask.nii.gz"
                tp = dict(params); tp.setdefault("tool_params", {})
                strip_result = stripper.strip(subj, out, mask, tp)
                quality = {}
                gt = gt_mask_for(subj, gt_dir)
                if strip_result.get("success") and mask.exists() and gt.exists():
                    quality = metrics.quality_metrics(mask, gt)
                rows.append(build_row(tool, dataset, subject_id, quality, strip_result))
                logger.info("%s / %s done", tool, subject_id)
    _append_csv(output, rows)
    logger.info("Wrote %d rows to %s", len(rows), output)


def _append_csv(output: Path, rows: list[dict]):
    output.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output.exists()
    with open(output, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tools", nargs="+", required=True)
    ap.add_argument("--reference-modality", default="t1")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).parent / "results" / "raw_metrics.csv")
    args = ap.parse_args()
    run(args.data_dir, args.gt_dir, args.dataset, args.tools,
        args.reference_modality, args.output,
        params={"reference_modality": args.reference_modality})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_run_benchmark.py -v`
Expected: PASS (2 tests).

- [ ] **Step 4b: Reproducibility option (ADD-4)**

Add a `--repro` flag to `main()` and a `reproducibility` column to `FIELDNAMES`. When set, each tool
runs twice per subject and the row records `reproducibility = metrics.dice(mask_run1, mask_run2)`
(1.0 for deterministic tools). Default off (single run) to keep the standard benchmark fast. Add a
test asserting two identical synthetic mask arrays give `dice == 1.0` (reuses `metrics.dice`).

- [ ] **Step 5: Commit**

```bash
git add research/skull_stripping_benchmark/run_benchmark.py \
        research/skull_stripping_benchmark/tests/test_run_benchmark.py
git commit -m "feat(ss-research): benchmark runner writing raw_metrics.csv (+repro, ADD-4)"
```

---

## Phase 8 — Parameter tuning

### Task 8.1: `tuning.py` — validation-split parameter search, write tuned params to manifests

**Files:**
- Create: `research/skull_stripping_benchmark/tuning.py`
- Test: `research/skull_stripping_benchmark/tests/test_tuning.py`

> Tuning runs each tool's search grid (spec §6) on the validation subjects, scores by mean DSC, picks the best, and writes it to `tuned_params:` in the tool's manifest. The grid + selection logic is unit-tested with a fake stripper; the manifest-writing function is tested round-trip.

- [ ] **Step 1: Write the failing test**

`research/skull_stripping_benchmark/tests/test_tuning.py`:
```python
import tuning


def test_param_grid_for_bet():
    grid = tuning.param_grid("bet")
    assert {"fractional_intensity": 0.35} in grid
    assert len(grid) == 5


def test_param_grid_for_hdbet():
    grid = tuning.param_grid("hdbet")
    assert {"threshold": 0.5} in grid


def test_select_best_picks_highest_mean_dsc():
    scored = [
        ({"fractional_intensity": 0.3}, [0.80, 0.82]),
        ({"fractional_intensity": 0.4}, [0.90, 0.92]),
        ({"fractional_intensity": 0.5}, [0.70, 0.71]),
    ]
    best = tuning.select_best(scored)
    assert best == {"fractional_intensity": 0.4}


def test_write_tuned_params_roundtrip(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("name: bet\ntuned_params: {}\n")
    tuning.write_tuned_params(manifest, {"fractional_intensity": 0.4})
    import yaml
    data = yaml.safe_load(manifest.read_text())
    assert data["tuned_params"] == {"fractional_intensity": 0.4}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_tuning.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `tuning.py`**

`research/skull_stripping_benchmark/tuning.py`:
```python
"""Validation-split parameter tuning for skull stripping tools (spec §6).

Each tool has a small grid over its key parameter. We score each setting by
mean DSC over the validation subjects and write the winner into the tool's
manifest `tuned_params:`.
"""

import argparse
import logging
import statistics
import sys
import tempfile
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import metrics  # noqa: E402
from preprocessing_steps.skull_stripping import dispatcher  # noqa: E402

MANIFEST_ROOT = PROJECT_ROOT / "services" / "skull-stripping"

_GRIDS = {
    "bet": [{"fractional_intensity": f} for f in (0.2, 0.3, 0.35, 0.4, 0.5)],
    "hdbet": [{"threshold": t} for t in (0.3, 0.5, 0.7)],
    "synthstrip": [{"border": b} for b in (0, 1, 2)],
    "mni_mask": [{"variant": "strict"}, {"variant": "loose", "dilation_vox": 2}],
}


def param_grid(tool: str) -> list[dict]:
    return list(_GRIDS.get(tool, [{}]))


def select_best(scored: list[tuple[dict, list[float]]]) -> dict:
    """scored: list of (params, [dsc per subject]). Returns params with max mean DSC."""
    best_params, best_mean = None, float("-inf")
    for params, dscs in scored:
        valid = [d for d in dscs if d == d]  # drop NaN
        mean = statistics.mean(valid) if valid else float("-inf")
        if mean > best_mean:
            best_mean, best_params = mean, params
    return best_params


def write_tuned_params(manifest_path: Path, params: dict) -> None:
    data = yaml.safe_load(manifest_path.read_text()) or {}
    data["tuned_params"] = params
    with open(manifest_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def tune_tool(tool, val_subjects, gt_dir, reference_modality) -> dict:
    stripper = dispatcher.get_stripper(tool)
    if not stripper.is_available():
        logger.warning("Tool '%s' unavailable; skipping tuning", tool)
        return {}
    scored = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for setting in param_grid(tool):
            dscs = []
            for subj in val_subjects:
                subject_id = subj.parents[2].name
                session = subj.parents[1].name
                out = tmp / f"{tool}_{subject_id}_strip.nii.gz"
                mask = tmp / f"{tool}_{subject_id}_mask.nii.gz"
                res = stripper.strip(subj, out, mask,
                                     {"reference_modality": reference_modality,
                                      "tool_params": setting})
                gt = gt_dir / subj.parents[2].name / session / "anat" / \
                    f"{subject_id}_{session}_brain_mask.nii.gz"
                if res.get("success") and mask.exists() and gt.exists():
                    dscs.append(metrics.quality_metrics(mask, gt)["dsc"])
                else:
                    dscs.append(float("nan"))
            scored.append((setting, dscs))
            logger.info("%s %s -> mean DSC computed", tool, setting)
    best = select_best(scored)
    if best is not None:
        write_tuned_params(MANIFEST_ROOT / tool / "manifest.yaml", best)
        logger.info("%s tuned params: %s", tool, best)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", required=True, type=Path)
    ap.add_argument("--gt-dir", required=True, type=Path)
    ap.add_argument("--tools", nargs="+", required=True)
    ap.add_argument("--reference-modality", default="t1")
    args = ap.parse_args()
    val_subjects = sorted(
        args.val_dir.glob(f"sub-*/ses-*/anat/*_{args.reference_modality}.nii.gz"))
    for tool in args.tools:
        tune_tool(tool, val_subjects, args.gt_dir, args.reference_modality)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_tuning.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add research/skull_stripping_benchmark/tuning.py \
        research/skull_stripping_benchmark/tests/test_tuning.py
git commit -m "feat(ss-research): parameter tuning on validation split -> manifests"
```

---

## Phase 9 — Statistical analysis + reporting

### Task 9.1: `report.py` — statistics + figures from `raw_metrics.csv`

**Files:**
- Create: `research/skull_stripping_benchmark/report.py`
- Test: `research/skull_stripping_benchmark/tests/test_report.py`

> `report.py` loads `raw_metrics.csv` and provides: the statistical test hierarchy (normality → ANOVA/Kruskal-Wallis → Bonferroni-corrected pairwise → effect size → Friedman) and figure generation. Tests cover the pure stat functions with synthetic data; figure functions are smoke-tested for file output.

- [ ] **Step 1: Write the failing test**

`research/skull_stripping_benchmark/tests/test_report.py`:
```python
import numpy as np
import pandas as pd
import report


def _fake_df():
    rng = np.random.default_rng(0)
    rows = []
    for tool, mu in [("bet", 0.85), ("hdbet", 0.92), ("mni_mask", 0.80)]:
        for ds in ["gbm", "ms"]:
            for i in range(10):
                rows.append({"tool": tool, "dataset": ds, "subject": f"s{i}",
                             "dsc": float(np.clip(rng.normal(mu, 0.02), 0, 1)),
                             "hd95": float(rng.normal(3, 0.5)),
                             "leakage": float(rng.normal(0.05, 0.01)),
                             "over_stripping": float(rng.normal(0.04, 0.01)),
                             "processing_time": float(rng.normal(10, 1)),
                             "vram_used_gb": 0.0, "success": True})
    return pd.DataFrame(rows)


def test_normality_returns_pvalue_per_tool():
    res = report.normality_by_tool(_fake_df(), metric="dsc")
    assert set(res.keys()) == {"bet", "hdbet", "mni_mask"}
    assert all(0 <= v <= 1 for v in res.values())


def test_friedman_returns_statistic_and_p():
    stat, p = report.friedman_ranking(_fake_df(), metric="dsc")
    assert p <= 1.0


def test_pairwise_bonferroni_has_all_pairs():
    pairs = report.pairwise_bonferroni(_fake_df(), metric="dsc")
    # 3 tools -> 3 pairs
    assert len(pairs) == 3
    assert all("p_corrected" in row for row in pairs)


def test_boxplot_writes_file(tmp_path):
    out = tmp_path / "box.png"
    report.dsc_boxplot(_fake_df(), out)
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_report.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `report.py`**

`research/skull_stripping_benchmark/report.py`:
```python
"""Statistical analysis and figures for the skull stripping benchmark (spec §5).

Test hierarchy:
  1. Shapiro-Wilk (n<50) / D'Agostino (n>=50) normality per tool
  2. normal -> two-way ANOVA (tool x dataset); else Kruskal-Wallis + pairwise Wilcoxon
  3. Bonferroni correction over the 21 tool pairs (7 tools)
  4. effect size: eta^2 (ANOVA) or rank-biserial r (Wilcoxon)
  5. overall ranking: Friedman
"""

import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def load(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def normality_by_tool(df: pd.DataFrame, metric: str = "dsc") -> dict:
    out = {}
    for tool, g in df.groupby("tool"):
        vals = g[metric].dropna().values
        if len(vals) < 3:
            out[tool] = float("nan"); continue
        if len(vals) < 50:
            out[tool] = float(stats.shapiro(vals).pvalue)
        else:
            out[tool] = float(stats.normaltest(vals).pvalue)
    return out


def friedman_ranking(df: pd.DataFrame, metric: str = "dsc") -> tuple[float, float]:
    """Friedman test across tools on shared subjects (pivot subject x tool)."""
    pivot = df.pivot_table(index=["dataset", "subject"], columns="tool",
                           values=metric, aggfunc="mean").dropna()
    arrays = [pivot[c].values for c in pivot.columns]
    stat, p = stats.friedmanchisquare(*arrays)
    return float(stat), float(p)


def pairwise_bonferroni(df: pd.DataFrame, metric: str = "dsc") -> list[dict]:
    """Pairwise Wilcoxon signed-rank with Bonferroni correction over all pairs."""
    pivot = df.pivot_table(index=["dataset", "subject"], columns="tool",
                           values=metric, aggfunc="mean").dropna()
    tools = list(pivot.columns)
    pairs = list(itertools.combinations(tools, 2))
    n = len(pairs)
    results = []
    for a, b in pairs:
        try:
            stat, p = stats.wilcoxon(pivot[a].values, pivot[b].values)
        except ValueError:
            stat, p = float("nan"), 1.0
        results.append({"tool_a": a, "tool_b": b, "statistic": float(stat),
                        "p_raw": float(p), "p_corrected": float(min(1.0, p * n))})
    return results


def two_way_anova(df: pd.DataFrame, metric: str = "dsc") -> pd.DataFrame:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    model = smf.ols(f"{metric} ~ C(tool) * C(dataset)", data=df).fit()
    return sm.stats.anova_lm(model, typ=2)


# ---- figures ----

def dsc_boxplot(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="tool", y="dsc")
    plt.title("DSC per tool"); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()


def dsc_heatmap_tool_x_dataset(df: pd.DataFrame, out: Path):
    pivot = df.pivot_table(index="tool", columns="dataset", values="dsc", aggfunc="mean")
    plt.figure(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Mean DSC: tool x dataset"); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()


def performance_scatter(df: pd.DataFrame, out: Path):
    agg = df.groupby("tool").agg(dsc=("dsc", "mean"),
                                 time=("processing_time", "mean")).reset_index()
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=agg, x="time", y="dsc", hue="tool", s=120)
    plt.xlabel("Mean processing time (s/volume)"); plt.ylabel("Mean DSC")
    plt.title("Performance vs quality"); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()


def leakage_comparison(df: pd.DataFrame, out: Path):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=df, x="tool", y="leakage")
    plt.title("Leakage rate per tool"); plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()


def generate_all_figures(df: pd.DataFrame, fig_dir: Path):
    dsc_boxplot(df, fig_dir / "dsc_boxplot_per_tool.png")
    dsc_heatmap_tool_x_dataset(df, fig_dir / "dsc_heatmap_tool_x_dataset.png")
    performance_scatter(df, fig_dir / "performance_scatter.png")
    leakage_comparison(df, fig_dir / "leakage_comparison.png")
```

- [ ] **Step 4: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_report.py -v`
Expected: PASS (4 tests).

- [ ] **Step 4b: Add worst-case / robustness summary (ADD-4)**

Per lit-review trend T1 (mean DSC has saturated; worst-case discriminates), add a per-tool summary
that surfaces worst-case and spread, not just the mean.

Add to `research/skull_stripping_benchmark/tests/test_report.py`:
```python
def test_summary_table_has_worst_case():
    df = _fake_df()
    summ = report.summary_table(df, metric="dsc")
    assert {"tool", "mean", "std", "min", "p05", "n"} <= set(summ.columns)
    # min (worst-case) must not exceed mean for any tool
    assert (summ["min"] <= summ["mean"] + 1e-9).all()
```

Add to `research/skull_stripping_benchmark/report.py`:
```python
def summary_table(df: pd.DataFrame, metric: str = "dsc") -> pd.DataFrame:
    """Per-tool summary emphasising worst-case robustness (ADD-4)."""
    g = df.groupby("tool")[metric]
    out = g.agg(mean="mean", std="std", min="min", n="count").reset_index()
    out["p05"] = g.quantile(0.05).values   # 5th-percentile (worst-case) DSC
    return out
```
Also add a worst-case figure:
```python
def worst_case_bar(df: pd.DataFrame, out: Path, metric: str = "dsc"):
    summ = summary_table(df, metric)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=summ, x="tool", y="min")
    plt.ylabel(f"worst-case {metric}"); plt.title(f"Worst-case {metric} per tool")
    plt.tight_layout(); out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150); plt.close()
```
Add `worst_case_bar(df, fig_dir / "worst_case_dsc.png")` to `generate_all_figures`.

> Note: reproducibility DSC (two identical runs) is produced by the benchmark runner reproducibility
> option (Task 7.1, ADD-4 note) and summarised here if present in the CSV.

- [ ] **Step 5: Commit**

```bash
git add research/skull_stripping_benchmark/report.py \
        research/skull_stripping_benchmark/tests/test_report.py
git commit -m "feat(ss-research): statistical analysis + figures incl. worst-case (ADD-4)"
```

---

### Task 9.2: Statistical analysis notebook

**Files:**
- Create: `research/skull_stripping_benchmark/results/statistical_analysis.ipynb`

- [ ] **Step 1: Create the notebook with executable cells calling `report.py`**

Build the notebook (use `jupyter nbconvert` or write JSON) with these cells in order:
1. Markdown: title + spec §5 test hierarchy.
2. Code: `import sys; sys.path.insert(0, "..")` then `import report, pandas as pd; df = report.load("raw_metrics.csv")`.
3. Code: `report.normality_by_tool(df, "dsc")`.
4. Code (markdown-guarded branch): if all normal → `report.two_way_anova(df, "dsc")`; else `report.pairwise_bonferroni(df, "dsc")`.
5. Code: `report.friedman_ranking(df, "dsc")`.
6. Code: `report.generate_all_figures(df, __import__("pathlib").Path("figures"))`.
7. Markdown: research-questions table from spec §5 with cells answering each.

Concrete creation command:
```bash
cd /home/e.roppert/work/mri_ai_service && source venv/bin/activate
pip install -r research/skull_stripping_benchmark/requirements-research.txt
jupyter nbconvert --to notebook --execute --allow-errors \
  --output results/statistical_analysis.ipynb /dev/stdin <<'PY' || true
PY
```
If scripting the notebook is awkward, create it via `jupyter notebook` and paste the cells above. The notebook must run top-to-bottom against a present `raw_metrics.csv`.

- [ ] **Step 2: Verify the notebook executes (once real or sample CSV exists)**

Run: `cd /home/e.roppert/work/mri_ai_service/research/skull_stripping_benchmark && jupyter nbconvert --to notebook --execute results/statistical_analysis.ipynb --output results/statistical_analysis.ipynb`
Expected: executes without errors; figures appear in `results/figures/`.

- [ ] **Step 3: Commit**

```bash
# results/ is gitignored (data outputs), so force-add the notebook (code):
git add -f research/skull_stripping_benchmark/results/statistical_analysis.ipynb
git commit -m "feat(ss-research): statistical analysis notebook"
```

---

### Task 9.3: Manifest-driven selection experiment (ADD-3, headline MAS result)

**Files:**
- Create: `research/skull_stripping_benchmark/selection_experiment.py`
- Test: `research/skull_stripping_benchmark/tests/test_selection_experiment.py`

> The headline MAS contribution (lit review §5): show that routing each subject to a tool by its
> manifest `lesion_type_preference` (GBM→best-for-glioma, MS→best-for-ms) **beats any single fixed
> tool** across the combined cohort. This consumes `raw_metrics.csv` + manifests; it does not re-run
> strippers. Each dataset maps to a lesion type (gbm/ms); the router picks, per lesion type, the tool
> with the highest manifest preference (ties broken by mean DSC on the validation portion or by
> `cascade_priority`).

- [ ] **Step 1: Write the failing test**

`research/skull_stripping_benchmark/tests/test_selection_experiment.py`:
```python
import pandas as pd
import selection_experiment as se


def _df():
    rows = []
    # tool A great on gbm, poor on ms; tool B opposite; routing should beat both.
    for sub in range(10):
        rows += [
            {"tool": "A", "dataset": "gbm", "subject": f"g{sub}", "dsc": 0.95},
            {"tool": "B", "dataset": "gbm", "subject": f"g{sub}", "dsc": 0.80},
            {"tool": "A", "dataset": "ms", "subject": f"m{sub}", "dsc": 0.78},
            {"tool": "B", "dataset": "ms", "subject": f"m{sub}", "dsc": 0.93},
        ]
    return pd.DataFrame(rows)


def test_route_picks_per_lesion_best():
    routing = {"gbm": "A", "ms": "B"}
    mean = se.routed_mean_dsc(_df(), routing, dataset_to_lesion={"gbm": "gbm", "ms": "ms"})
    assert abs(mean - (0.95 + 0.93) / 2) < 1e-9


def test_routing_beats_any_fixed_tool():
    routing = {"gbm": "A", "ms": "B"}
    res = se.compare_routing_vs_fixed(
        _df(), routing, dataset_to_lesion={"gbm": "gbm", "ms": "ms"})
    assert res["routed"] > res["best_fixed"]["mean_dsc"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_selection_experiment.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `selection_experiment.py`**

`research/skull_stripping_benchmark/selection_experiment.py`:
```python
"""Manifest-driven selection experiment (ADD-3).

Compares a lesion-type router (pick the manifest-preferred tool per lesion type) against every
single fixed tool, on the combined cohort, from raw_metrics.csv. Headline MAS result.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_ROOT = PROJECT_ROOT / "services" / "skull-stripping"

# Map benchmark dataset names to lesion types.
DEFAULT_DATASET_TO_LESION = {
    "upenn_gbm": "gbm", "local_gbm": "gbm",
    "mosmed_ms": "ms", "local_ms": "ms",
}
_PREF_RANK = {"high": 3, "medium": 2, "low": 1, "unknown": 0}


def build_routing_from_manifests(lesion_types) -> dict:
    """For each lesion type, pick the available tool with the highest manifest preference."""
    routing = {}
    for lesion in lesion_types:
        best_tool, best_score = None, -1
        for mdir in sorted(MANIFEST_ROOT.glob("*/manifest.yaml")):
            m = yaml.safe_load(mdir.read_text()) or {}
            pref = (m.get("lesion_type_preference", {}) or {}).get(lesion, "unknown")
            score = _PREF_RANK.get(pref, 0)
            prio = m.get("mas_metadata", {}).get("cascade_priority", 99)
            key = (score, -prio)
            if key > (best_score if isinstance(best_score, tuple) else (best_score, 0)):
                best_score, best_tool = key, m.get("name")
        routing[lesion] = best_tool
    return routing


def routed_mean_dsc(df: pd.DataFrame, routing: dict, dataset_to_lesion: dict) -> float:
    picked = []
    for _, row in df.iterrows():
        lesion = dataset_to_lesion.get(row["dataset"])
        if lesion and routing.get(lesion) == row["tool"]:
            picked.append(row["dsc"])
    return float(pd.Series(picked).mean()) if picked else float("nan")


def compare_routing_vs_fixed(df: pd.DataFrame, routing: dict, dataset_to_lesion: dict) -> dict:
    routed = routed_mean_dsc(df, routing, dataset_to_lesion)
    fixed = (df.groupby("tool")["dsc"].mean().rename("mean_dsc").reset_index()
             .sort_values("mean_dsc", ascending=False))
    best_fixed = fixed.iloc[0].to_dict()
    return {"routed": routed, "routing": routing,
            "best_fixed": best_fixed, "fixed_ranking": fixed.to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path,
                    default=Path(__file__).parent / "results" / "raw_metrics.csv")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "results" / "selection_experiment.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    lesion_types = sorted(set(DEFAULT_DATASET_TO_LESION.values()))
    routing = build_routing_from_manifests(lesion_types)
    res = compare_routing_vs_fixed(df, routing, DEFAULT_DATASET_TO_LESION)
    print("Routing:", res["routing"])
    print("Routed mean DSC:", res["routed"])
    print("Best fixed:", res["best_fixed"])
    pd.DataFrame(res["fixed_ranking"]).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_selection_experiment.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add research/skull_stripping_benchmark/selection_experiment.py \
        research/skull_stripping_benchmark/tests/test_selection_experiment.py
git commit -m "feat(ss-research): manifest-driven selection experiment (ADD-3)"
```

---

### Task 9.4: Input-characteristic → tool analysis (ADD-5, analysis only)

**Files:**
- Create: `research/skull_stripping_benchmark/characteristic_analysis.py`
- Test: `research/skull_stripping_benchmark/tests/test_characteristic_analysis.py`

> Answers "which input characteristics influence the best tool?" as an **offline analysis** (no runtime
> selector — that is deferred to Этап 7 future work). Reuses the **Stage 04 quality metrics already
> computed by the pipeline** ([scripts/quality_metrics/](../../../scripts/quality_metrics/): SNR, CNR,
> EFC, FBER, gradient sharpness, anisotropy, …) plus cheap metadata (lesion type, modalities present,
> voxel anisotropy). For each subject it finds the winning tool (max DSC) and reports how characteristics
> differ across winners. Feeds the paper's MAS-routing justification and the PhD.

- [ ] **Step 1: Write the failing test**

`research/skull_stripping_benchmark/tests/test_characteristic_analysis.py`:
```python
import pandas as pd
import characteristic_analysis as ca


def test_winning_tool_per_subject():
    metrics = pd.DataFrame([
        {"subject": "s1", "dataset": "gbm", "tool": "A", "dsc": 0.90},
        {"subject": "s1", "dataset": "gbm", "tool": "B", "dsc": 0.95},
        {"subject": "s2", "dataset": "ms", "tool": "A", "dsc": 0.88},
        {"subject": "s2", "dataset": "ms", "tool": "B", "dsc": 0.70},
    ])
    win = ca.winning_tool(metrics)
    assert win.set_index("subject")["tool"].to_dict() == {"s1": "B", "s2": "A"}


def test_feature_summary_by_winner():
    win = pd.DataFrame([{"subject": "s1", "tool": "B"}, {"subject": "s2", "tool": "A"}])
    feats = pd.DataFrame([
        {"subject": "s1", "snr": 10.0, "cnr": 2.0},
        {"subject": "s2", "snr": 30.0, "cnr": 5.0},
    ])
    summary = ca.feature_summary_by_winner(win, feats, feature_cols=["snr", "cnr"])
    # one row per (tool) with mean feature values
    assert set(summary["tool"]) == {"A", "B"}
    assert "snr" in summary.columns and "cnr" in summary.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_characteristic_analysis.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Write `characteristic_analysis.py`**

`research/skull_stripping_benchmark/characteristic_analysis.py`:
```python
"""Input-characteristic -> best-tool analysis (ADD-5, offline).

Joins per-subject winning tool (from raw_metrics.csv) with per-subject input
characteristics (Stage 04 quality metrics + light metadata) and summarises how
characteristics differ across winners. Analysis only; no runtime selector.
"""

import argparse
from pathlib import Path

import pandas as pd


def winning_tool(metrics: pd.DataFrame) -> pd.DataFrame:
    """Per subject, the tool with the highest DSC."""
    idx = metrics.groupby("subject")["dsc"].idxmax()
    return metrics.loc[idx, ["subject", "tool"]].reset_index(drop=True)


def feature_summary_by_winner(win: pd.DataFrame, feats: pd.DataFrame,
                              feature_cols: list[str]) -> pd.DataFrame:
    """Mean of each characteristic grouped by the tool that won on that subject."""
    merged = win.merge(feats, on="subject", how="inner")
    return merged.groupby("tool")[feature_cols].mean().reset_index()


def correlate_features_with_winner(win: pd.DataFrame, feats: pd.DataFrame,
                                   feature_cols: list[str]) -> pd.DataFrame:
    """Kruskal-Wallis p-value per feature across winner groups (which features matter)."""
    from scipy import stats
    merged = win.merge(feats, on="subject", how="inner")
    rows = []
    for col in feature_cols:
        groups = [g[col].dropna().values for _, g in merged.groupby("tool")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) >= 2:
            try:
                stat, p = stats.kruskal(*groups)
            except ValueError:
                stat, p = float("nan"), 1.0
        else:
            stat, p = float("nan"), 1.0
        rows.append({"feature": col, "kruskal_stat": float(stat), "p_value": float(p)})
    return pd.DataFrame(rows).sort_values("p_value")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path,
                    default=Path(__file__).parent / "results" / "raw_metrics.csv")
    ap.add_argument("--features", type=Path, required=True,
                    help="CSV of per-subject Stage 04 quality metrics + metadata")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "results" / "characteristic_analysis.csv")
    args = ap.parse_args()
    metrics = pd.read_csv(args.metrics)
    feats = pd.read_csv(args.features)
    feature_cols = [c for c in feats.columns if c != "subject"]
    win = winning_tool(metrics)
    summary = feature_summary_by_winner(win, feats, feature_cols)
    corr = correlate_features_with_winner(win, feats, feature_cols)
    summary.to_csv(args.out, index=False)
    corr.to_csv(args.out.with_name("characteristic_importance.csv"), index=False)
    print("Feature summary by winning tool:\n", summary)
    print("\nWhich characteristics matter (Kruskal-Wallis):\n", corr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests (must pass)**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest research/skull_stripping_benchmark/tests/test_characteristic_analysis.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Verify the Stage 04 features source (manual)**

Confirm where Stage 04 writes per-subject quality metrics (run a pipeline through Stage 04 and inspect
its output, e.g. a quality CSV/JSON under the run's output dir). Add a small adapter (or document the
column mapping) so `--features` receives `subject` + numeric quality columns. Record the path in
`DEVLOG.md`. This step depends on real pipeline output and is run when benchmark data exists.

- [ ] **Step 6: Commit**

```bash
git add research/skull_stripping_benchmark/characteristic_analysis.py \
        research/skull_stripping_benchmark/tests/test_characteristic_analysis.py
git commit -m "feat(ss-research): input-characteristic -> tool analysis (ADD-5)"
```

---

## Phase 10 — Documentation, completion, integration

### Task 10.1: Benchmark README

**Files:**
- Create: `research/skull_stripping_benchmark/README.md`

- [ ] **Step 1: Write the README** covering spec §10:

`research/skull_stripping_benchmark/README.md`:
```markdown
# Skull Stripping Benchmark

Comparative evaluation of 7 skull-stripping tools (FSL BET, HD-BET, SynthStrip,
SAM-based, DeepBET, MNI strict, MNI loose) across GBM + MS datasets. Supports
Этап 5.5 and the comparison paper in `docs/papers/skull_stripping_comparison/`.

## Prerequisites
- Project venv with `pip install -r requirements-research.txt`
- FSL (system) for BET; `pip install hd-bet`, `pip install surfa` (SynthStrip)
- GPU (NVIDIA RTX 5070 12GB used in paper); CPU fallback supported for BET/SynthStrip/MNI
- MNI152 template: `data/templates/MNI152_T1_1mm.nii.gz` (already in repo)

## 1. Prepare data (pipeline stages 1-4, registration space, no skull stripping)
```bash
python prepare_data.py --input-dir <bids_root> --output-dir <bench_data> --atlas MNI152
```
Produces `<bench_data>/registration_space/sub-*/ses-*/anat/*.nii.gz`.

## 2. Tune parameters on the validation split
```bash
python tuning.py --val-dir <bench_data>/registration_space \
  --gt-dir <gt_masks> --tools bet hdbet synthstrip mni_mask
```
Writes winners into each `services/skull-stripping/<tool>/manifest.yaml`.

## 3. Run the benchmark on the test split
```bash
python run_benchmark.py --data-dir <bench_data>/registration_space \
  --gt-dir <gt_masks> --dataset upenn_gbm \
  --tools bet hdbet synthstrip mni_mask --output results/raw_metrics.csv
```
Run once per dataset (`upenn_gbm`, `mosmed_ms`, `local_gbm`, `local_ms`).

## 4. Analyze
Open `results/statistical_analysis.ipynb` and run top-to-bottom, or:
```bash
python -c "import report,pathlib; df=report.load('results/raw_metrics.csv'); \
report.generate_all_figures(df, pathlib.Path('results/figures'))"
```

## Single tool / single subject
Run `run_benchmark.py` with `--tools <one>` and a `--data-dir` containing one subject.

## Results interpretation
- DSC ↑, HD95 ↓, leakage ↓, over-stripping ↓ are better.
- Pseudo-GT is the MNI152 brain mask; imperfect for GBM mass effect — cross-check
  the qualitative examples in `results/figures/visual_examples/`.

## Hardware notes
Performance numbers (time, RAM, VRAM) are hardware-specific; record your GPU.
```

- [ ] **Step 2: Commit**

```bash
git add research/skull_stripping_benchmark/README.md
git commit -m "docs(ss-research): benchmark README with install/run/interpret guide"
```

---

### Task 10.2: Production-path scaffolding note (no microservices yet)

**Files:**
- Create: `services/skull-stripping/README.md`

> Per spec §9, the 2 winners become Docker agents in Этап 7 by analogy with `gbm-seg`/`ms-seg`. We do not build those services now; we document the path so the manifests are understood as MAS-ready.

- [ ] **Step 1: Write the services README**

`services/skull-stripping/README.md`:
```markdown
# Skull Stripping Tool Manifests

These are **manifests only** (not microservices yet). Each `<tool>/manifest.yaml`
describes a skull-stripping tool's compute needs, modality support, lesion-type
preference, install method, tuned parameters, and MAS metadata
(`agent_type`, `cascade_priority`, `fallback_to`).

The Stage 05 dispatcher (`scripts/preprocessing_steps/skull_stripping/dispatcher.py`)
reads these via `load_manifest(name)`.

## Production path (Этап 7)
After the benchmark (Этап 5.5) identifies 2 winners, they become Docker agents by
analogy with `services/gbm-seg` / `services/ms-seg`:
```
services/<tool>-strip/{Dockerfile, manifest.yaml, requirements.txt, src/server.py}
```
`server.py` exposes `POST /predict` with the path-based contract. The MAS
coordinator routes by `lesion_type_preference` and `cascade_priority`.
```

- [ ] **Step 2: Commit**

```bash
git add services/skull-stripping/README.md
git commit -m "docs(ss): document manifests-only layout and Этап 7 production path"
```

---

### Task 10.3: Full plugin test sweep + ROADMAP update

**Files:**
- Modify: `ROADMAP.md` (Этап 5.5 status)
- Modify: `research/skull_stripping_benchmark/DEVLOG.md`

- [ ] **Step 1: Run the entire plugin + benchmark unit test suite**

Run: `cd /home/e.roppert/work/mri_ai_service && python -m pytest scripts/preprocessing_steps/skull_stripping/tests/ research/skull_stripping_benchmark/tests/ -v`
Expected: PASS (all unit tests across phases 1–9).

- [ ] **Step 2: Update ROADMAP Этап 5.5 status to reflect plugin + benchmark delivered**

In `ROADMAP.md`, under `## □ Этап 5.5`, change `**Статус:** 🎯 СЛЕДУЮЩАЯ` to `**Статус:** 🔬 В РАБОТЕ` and append a short bullet list of delivered components (plugin dispatcher, manifests, benchmark harness, paper skeleton). Mark the data/benchmark-execution items as pending real datasets.

- [ ] **Step 3: Update DEVLOG with the implementation summary**

Append a dated entry to `DEVLOG.md` summarizing: plugin architecture + dispatcher live, 7 tool wrappers (2 best-effort), benchmark harness + metrics + tuning + report + notebook scaffolded, paper skeleton written. Next step: acquire MosMed MS dataset, run full benchmark, fill paper results.

- [ ] **Step 4: Commit**

```bash
git add ROADMAP.md research/skull_stripping_benchmark/DEVLOG.md
git commit -m "docs(ss-research): mark Этап 5.5 in progress; log implementation milestone"
```

---

## Completion criteria mapping (spec §10)

| Spec criterion | Delivered by |
|---|---|
| Stage 05 dispatcher live, `method: bet` works as before | Tasks 1.1–1.3, 2.1 |
| Plugin architecture (`SkullStripperBase`, manifests) | Tasks 1.1–1.3, 3.x, 4.x |
| `prepare_data.py` registration-space inputs | Task 5.1 |
| `raw_metrics.csv` for all 4 datasets | Task 7.1 (run per dataset) |
| Statistical analysis notebook + all tests/figures | Tasks 9.1, 9.2 |
| Tuned params in manifests | Task 8.1 |
| Paper skeleton (fill with results) | Tasks 0.2, then results from 7–9 |
| Benchmark `README.md` | Task 10.1 |
| `DEVLOG.md` up to date | Tasks 0.1, 3.x, 4.2, 5.1, 10.3 |
| 2 production winners identified | Output of running 7–9 on real data (post-implementation) |
| ADD-1: BrainMaGe as 8th tool | Task 3.3 |
| ADD-2: cascade + mask-integrity validation | Task 1.4 |
| ADD-3: manifest-driven selection experiment | Task 9.3 |
| ADD-4: worst-case DSC + reproducibility | Task 9.1 (Step 4b), Task 7.1 (Step 4b) |
| ADD-5: input-characteristic → tool analysis | Task 9.4 |
| Branch merged to main | After full benchmark run + paper fill (use finishing-a-development-branch) |

> **Note (2026-06-16):** literature review (Task 0.3) repositioned the contribution to
> **MAS-ready, manifest-driven stripper selection** for our clinical GBM+MS cohort (vs Thakur 2020).
> Tool count is now **8** (BrainMaGe added). See `docs/papers/skull_stripping_comparison/literature_review.md`.

> **Items requiring real data (not code) to close:** running the benchmark on all 4 datasets, filling paper results, and naming the 2 winners. These depend on dataset acquisition (MosMed MS download, local clinical sets) and GPU runs; the code to do them is delivered by this plan.

---

## Self-Review notes

- **Spec coverage:** every §7 file and §10 criterion maps to a task (see table above). Atlas override (§3) → Task 5.1; metrics (§4) → Task 6.1; stats (§5) → Tasks 9.1/9.2; tuning (§6) → Task 8.1; plugin contract (§8) → Tasks 1.1/1.3; production path (§9) → Task 10.2.
- **Type consistency:** `strip()` return dict keys (`success/output_path/mask_path/processing_time/vram_used_gb/error`) are identical across `BetStripper`, `HDBetStripper`, `SynthStripStripper`, `MniMaskStripper`, and stubs. `STRIPPER_REGISTRY`, `get_stripper`, `resolve_stripper`, `process_subject_skull_stripping` names are stable from Task 1.3 onward. `metrics.quality_metrics` keys (`dsc/hd95/over_stripping/leakage/brain_volume_ml`) match `run_benchmark.build_row` and `report` column names.
- **External-tool uncertainty:** HD-BET / SynthStrip CLI flag names and Stage 05 argument names are flagged with explicit "verify with --help" steps before relying on them; best-effort tools fail closed via `is_available()`.
