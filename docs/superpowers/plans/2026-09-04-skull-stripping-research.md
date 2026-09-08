# Skull Stripping Research Implementation Plan (resumed)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish Этап 5.5 on top of the plugin already in `main`: cascade + mask validation, remaining stripper wrappers, a standalone benchmark across 4 datasets, and a filled comparison paper that names 2 production winners.

**Architecture:** Extend the existing `scripts/preprocessing_steps/skull_stripping/` package (`STRIPPERS`, `get_stripper(params)`, `process_subject_skull_stripping` in `__init__.py`, `gpu_pool.py`). Do not replace the package, do not revert production `method: hdbet`, and do not drop GPU-pool keys. Benchmark code lives in `research/skull_stripping_benchmark/` and writes into the paper skeletons under `docs/papers/skull_stripping_comparison/`.

**Tech Stack:** Python 3.12, FSL BET, HD-BET 2.x CLI, SynthStrip (`mri_synthstrip` / surfa), BrainMaGe (ADD-1), ANTs/nibabel, numpy, scipy/pingouin/statsmodels, pandas, matplotlib/seaborn, pytest, pynvml/psutil.

**Spec:** [docs/superpowers/specs/2026-06-15-skull-stripping-research-design.md](../specs/2026-06-15-skull-stripping-research-design.md)

**Frozen June plan (long samples for Phases 5–10):** [2026-06-16-skull-stripping-research.md](2026-06-16-skull-stripping-research.md)

## Global Constraints

- Python comments and docstrings in English; UI copy in Russian.
- Work on `feat/skull-stripping-research-v2` in `/home/ubuntu/mri_ai_service`. Do not recreate a git worktree.
- Do not commit `data/`, `demo_workspace/`, `*.db`, weights, `.env`, Kappa tokens, `__pycache__`.
- Register new strippers in `STRIPPERS` (`dispatcher.py`), not `STRIPPER_REGISTRY` (that name never landed on `main`).
- `get_stripper` takes a **params dict** (`method` / `fallback_method`), not a method string.
- Class name is `HdBetStripper` (`hdbet.py`), not `HDBetStripper`.
- `process_subject_skull_stripping` lives in `__init__.py` and already takes `gpu_pool=`. GPU tools (`uses_gpu = True`) must keep `acquire_device` / `resolve_devices`.
- Production config stays `method: "hdbet"` + `fallback_method: "bet"` plus `device` / `exclude_gpus` / `gpu_pool_size`. Cascade is additive; do not reset the step to BET-only.
- HD-BET 2.x has no `--threshold` CLI flag (see `services/skull-stripping/hdbet/manifest.yaml`). Do not invent it.
- `strip()` on `main` may omit `vram_used_gb`; the benchmark measures VRAM around the call. New wrappers should still return `success`, `output_path`, `mask_path`, `processing_time`, and `error` on failure.
- Lesion-type keys in new manifests: `glioblastoma` / `multiple_sclerosis` (match HD-BET manifest), not `glioma` / `ms`.
- Tests: `scripts/preprocessing_steps/skull_stripping/tests/` (already un-ignored). Benchmark tests: `research/skull_stripping_benchmark/tests/` (un-ignored 2026-09-04).
- After each working session append `research/skull_stripping_benchmark/DEVLOG.md`. Paper fills go into **both** `paper_skeleton.md` and `paper_skeleton_ru.md`.

---

## Status as of 2026-09-04

Branch `feat/skull-stripping-research-v2` = `origin/main` + research docs (spec, June plan, paper skeletons, literature review, benchmark scaffold). June plugin commits were **not** cherry-picked; `main` already has a later plugin.

| June task | Status | Where |
|---|---|---|
| 0.1 Scaffold | **done** | `research/skull_stripping_benchmark/` |
| 0.2 Paper skeletons EN/RU | **done** | `docs/papers/skull_stripping_comparison/` |
| 0.3 Literature review | **done** | `literature_review.md` |
| 1.1 Package + BET | **done on main** | `skull_stripping/{base,bet,__init__}.py` |
| 1.2 BET manifest | **done on main** | `services/skull-stripping/bet/manifest.yaml` |
| 1.3 Dispatcher + fallback | **done on main** | `dispatcher.py` `STRIPPERS` / `get_stripper(params)` |
| 1.4 Cascade + mask validation | **done on v2** (2026-09-04; hardened 2026-09-07: flags now steer the cascade, hole gate 20→2 ml, candidates no longer overwrite the input) | `validation.py`, cascade in `__init__.py` |
| 2.1 Config `method` / `fallback` / `tool_params` | **done on main** (and more: GPU keys, production `hdbet`) | `configs/preprocessing_config.yaml` |
| 3.1 HD-BET | **done on main** | `hdbet.py`, HD-BET 2.x CLI, GPU pool |
| GPU device portability | **done on main** (separate spec 2026-08-31) | `gpu_pool.py`, Stage 05 `build_pool` |
| 3.2 SynthStrip | **done on v2** (wrapper + manifest; CLI in web image via FreeSurfer) | Task B |
| 3.3 BrainMaGe | **not done** | Task C |
| 4.1 MNI strict/loose | **done on v2** (2026-09-04; prod atlas already MNI152_FSL) | Task D |
| 4.2 SAM + DeepBET stubs | **not done** | Task E |
| 5–10 Benchmark + paper fill | **not done** (scaffold only) | Tasks F–K; long samples in the June plan |

### Carried forward from the 2026-09-07 hardening

- `mni_mask` leaks onto the eyes under the production **Rigid** registration —
  an atlas mask cannot fit a head it was never scaled to. Treat it as a
  baseline, not a candidate for first place in a production cascade.
- **The spec's "DSC vs MNI152 brain mask as pseudo-GT" is unsafe as written**
  (§4 of the design spec): measured Dice is 0.84–0.87 for correct HD-BET masks
  and 0.997 for a mask containing the patient's eyes, so the metric ranks the
  broken mask first. Decide on a better reference before Phase 5 (visual
  scoring, consensus mask, or affine/nonlinear registration for the benchmark
  arm only).
- An intensity-based leakage metric was prototyped and rejected: skull and
  scalp are tissue, so "background inside the mask" does not separate the
  classes. Calibrating any future version needs volumes that still have skull —
  every file under `preprocessed/` is already masked.

**Do not implement:** replacing `skull_stripping.py` (already a package); a second HD-BET wrapper; `resolve_stripper`; production `cascade: ["bet"]`.

---

## Orientation for the implementer

- **Stage 05:** [scripts/05_preprocessing.py](../../../scripts/05_preprocessing.py) imports `process_subject_skull_stripping` and `build_pool` / `resolve_devices`. Parallel workers pass `gpu_pool=`.
- **Dispatcher:** [dispatcher.py](../../../scripts/preprocessing_steps/skull_stripping/dispatcher.py) — `STRIPPERS = {"bet", "hdbet"}`. Unknown method → `SkullStripperUnavailable`. Unavailable primary → `fallback_method` with a warning.
- **Per-subject orchestration:** [__init__.py](../../../scripts/preprocessing_steps/skull_stripping/__init__.py) `process_subject_skull_stripping`. If `stripper.uses_gpu`, pin via `acquire_device(gpu_pool)` or `resolve_devices(params)[0]`.
- **Config (do not delete these keys):** `method`, `fallback_method`, `device`, `exclude_gpus`, `gpu_pool_size`, `reference_modality`, `apply_to_all`, `cleanup`, `tool_params`.
- **BIDS files:** `sub-XXX/ses-YYY/anat/sub-XXX_ses-YYY_<mod>.nii.gz`.
- **NIfTI completeness:** Stage 05 skip uses `utils.nifti_integrity.is_complete_nifti`. `prepare_data.py` must not treat a truncated `.nii.gz` as done.

### Plugin contract (current code)

```python
class SkullStripperBase(ABC):
    name: str = "unnamed"
    uses_gpu: bool = False

    def strip(self, input_path, output_path, mask_path=None, params=None) -> dict:
        # success, output_path, mask_path, processing_time; error if failed

    def is_available(self) -> bool: ...

    @property
    def manifest(self) -> dict: ...  # services/skull-stripping/{name}/manifest.yaml
```

Register: `STRIPPERS["synthstrip"] = SynthStripStripper` in `dispatcher.py`. Set `uses_gpu = True` only if the tool must go through the GPU pool.

---

## File structure (target)

Already present: `base.py`, `bet.py`, `hdbet.py`, `dispatcher.py`, `gpu_pool.py`, `__init__.py`, manifests `bet` + `hdbet`, GPU tests.

Still to add:

```
scripts/preprocessing_steps/skull_stripping/
├── validation.py              # Task A
├── synthstrip.py              # Task B
├── brainmage.py               # Task C
├── mni_mask.py                # Task D
├── sam_bet.py                 # Task E
└── deepbet.py                 # Task E

services/skull-stripping/{synthstrip,brainmage,mni_mask,sam,deepbet}/manifest.yaml

research/skull_stripping_benchmark/
├── prepare_data.py, metrics.py, run_benchmark.py, tuning.py, report.py
├── selection_experiment.py, characteristic_analysis.py
├── README.md, tests/
└── results/                   # csv / figures / notebook (gitkeep only today)
```

---

## Phase A — Cascade + mask validation (was Task 1.4)

Wire validate-retry **without** breaking GPU pinning or today’s two-step `hdbet → bet` fallback.

**Files:**
- Create: `scripts/preprocessing_steps/skull_stripping/validation.py`
- Modify: `dispatcher.py` (`build_cascade_order`; optional `try_stripper(name)`)
- Modify: `__init__.py` `process_subject_skull_stripping` (loop cascade **inside** the existing GPU acquire)
- Modify: `configs/preprocessing_config.yaml` — add optional `cascade` / `validation` keys; **keep** GPU keys and `method: hdbet`
- Tests: `tests/test_validation.py`, `tests/test_cascade.py`

Default when `cascade` is absent: `[method, fallback_method]` if fallback set, else `[method]` — same as `get_stripper` today.

GBM mass effect can shrink/distort brain volume. Start with the June defaults (`min_ml=700`, `max_ml=1900`, `min_dominant_fraction=0.95`) but make them overridable via `params["validation"]`. If a plausible tumour mask fails volume gates, loosen via config rather than deleting validation.

- [x] **Step 1: Failing `test_validation.py`**

Use the June plan’s four cases (valid blob / empty / too large / fragmented). Import `validate_mask` from `preprocessing_steps.skull_stripping.validation`.

- [x] **Step 2: Run — expect FAIL** (`validation` missing)

`python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_validation.py -v`

- [x] **Step 3: Implement `validation.py`** as in the June plan (scipy.ndimage.label, volume in ml, dominant component).

Hard gates are **wider** than the June 700–1900 / 0.95 sample (see `DEFAULT_*` in `validation.py`): catastrophe 300–2500 ml, LCC 0.70 after dropping `< 1 ml` speckles. Review flags use 1000–1800 / 0.95 / edge-touch and do **not** retry. `fail_closed: false` is log-only.

- [x] **Step 4: Failing cascade tests**

```python
from preprocessing_steps.skull_stripping import dispatcher

def test_build_cascade_orders_method_first_then_cascade():
    order = dispatcher.build_cascade_order(
        {"method": "hdbet", "cascade": ["synthstrip", "hdbet", "bet"]})
    assert order == ["hdbet", "synthstrip", "bet"]

def test_build_cascade_falls_back_to_single_fallback_method():
    order = dispatcher.build_cascade_order({"method": "hdbet", "fallback_method": "bet"})
    assert order == ["hdbet", "bet"]

def test_build_cascade_method_only():
    assert dispatcher.build_cascade_order({"method": "bet"}) == ["bet"]
```

- [x] **Step 5: `build_cascade_order` in `dispatcher.py`**

```python
def build_cascade_order(params: dict) -> list[str]:
    method = str(params.get("method", DEFAULT_METHOD)).lower()
    chain = params.get("cascade") or (
        [params["fallback_method"]] if params.get("fallback_method") else [])
    order, seen = [], set()
    for name in [method, *chain]:
        if not name:
            continue
        name = str(name).lower()
        if name not in seen:
            order.append(name)
            seen.add(name)
    return order
```

Keep `get_stripper(params)` for callers that only need one available tool. Cascade iteration uses `STRIPPERS[name]()` + `is_available()`.

In `process_subject_skull_stripping`, replace the single `_run_strip` with: for each name in `build_cascade_order(params)`, skip unknown/unavailable, run `strip` (still inside `acquire_device` if **that** candidate `uses_gpu`; CPU tools must not hold a GPU slot), `validate_mask` if a mask was written, accept first valid, else try next. Log which name won. Preserve `method` on the result dict.

If every candidate fails: return `{success: False, error: ...}` like today.

- [x] **Step 6: Config** — add comments + optional keys, do not remove GPU block:

```yaml
      # Ordered extras after method (optional). Absent cascade still means
      # method then fallback_method.
      # cascade: ["synthstrip", "bet"]
      validation: {}   # mask-integrity; {} = adult defaults in validation.py
```

- [x] **Step 7: Tests**

`python -m pytest scripts/preprocessing_steps/skull_stripping/tests/test_validation.py scripts/preprocessing_steps/skull_stripping/tests/test_cascade.py scripts/preprocessing_steps/skull_stripping/tests/test_gpu_pool.py scripts/preprocessing_steps/skull_stripping/tests/test_gpu_dispatch.py -v`

Expected: PASS. GPU tests must still pass.

- [x] **Step 8: Commit** `feat(ss): cascade selection + mask-integrity validation`

---

## Phase B — SynthStrip (was Task 3.2)

**Files:** `synthstrip.py`, `services/skull-stripping/synthstrip/manifest.yaml`, register in `STRIPPERS`, `tests/test_synthstrip.py`.

- [x] **Step 1: Failing tests** — register via `STRIPPERS`, not `get_stripper("synthstrip")`. Availability: `shutil.which` for `mri_synthstrip`. Command builder: `-i`, `-o`, `-m`, optional `-b` / `border` from `tool_params`. `uses_gpu = False` unless you confirm a GPU CLI path.

- [x] **Step 2: Run — expect FAIL**

- [x] **Step 3: Implement wrapper** — subprocess + timeout; normalize mask to `mask_path`. Verify flags with `mri_synthstrip --help` before locking the test.

- [x] **Step 4: Manifest** — `lesion_type_preference` keys `glioblastoma` / `multiple_sclerosis`; `cascade_priority: 2`; `fallback_to: bet`.

- [x] **Step 5: Tests PASS + timeboxed install smoke. DEVLOG.**

CLI lives in the web image (`FREESURFER_HOME` / `mri_synthstrip`). `-g` exists upstream, not wired (`uses_gpu` stays False). Host venv has no CLI.

- [x] **Step 6: Commit** `feat(ss): add SynthStrip wrapper and manifest`

June plan Step 3 code is a starting point; fix registry names and `uses_gpu`.

---

## Phase C — BrainMaGe (was Task 3.3 / ADD-1)

Same wrapper pattern. Register `"brainmage"`. Timebox install (~2h). If unpackable, `is_available()` is False and the tool is documented in DEVLOG as excluded — still ship the stub + manifest so the dispatcher can skip it.

- [ ] Tests, wrapper, registry, manifest, smoke, commit `feat(ss): add BrainMaGe wrapper and manifest`

Follow June Task 3.3 body with the Global Constraints overlay.

---

## Phase D — MNI strict + loose (was Task 4.1)

Atlas baseline: `MniMaskStripper` with variant `strict` (undilated MNI152 brain mask) and `loose` (~2 mm dilation). Inputs are already in atlas space after Stage 05 registration; production atlas is **MNI152_FSL** (Kate, 2026-09-04), not SRI24.

Mask file: `data/templates/MNI152_T1_1mm_brain_mask.nii.gz` (FSL brain mask, not a threshold on the skull-on T1).

- [x] Tests for both variants, implementation, `STRIPPERS["mni_mask"]` (variant from `tool_params`), manifest.
- [ ] Commit `feat(ss): add MNI atlas-mask stripper (strict/loose)` — wait for Kate after manual test.

June Task 4.1 samples apply with registry-name overlay.

---

## Phase E — SAM + DeepBET stubs (was Task 4.2)

Best-effort, ≤2h install each. Fail closed via `is_available()`. Register `"sam"` and `"deepbet"`.

- [ ] Tests that unavailable tools are skipped, stubs, manifests, DEVLOG of install outcome, commit.

---

## Phase F — `prepare_data.py` (was Task 5.1)

Registration-space NIfTI, **skull stripping disabled**, MNI152 override for the benchmark only.

**Deltas vs June Task 5.1:**

- Paths: `/home/ubuntu/mri_ai_service`.
- Copy live `preprocessing_config.yaml` then: disable `skull_stripping`; set atlas to MNI152 (`data/templates/MNI152_T1_1mm.nii.gz`); do not strip GPU keys (unused if stripping is off).
- Skip-existing: use `utils.nifti_integrity.is_complete_nifti`, not `exists()`.
- `bias_correction.use_for_registration_only` is already true — still write the **registration-space** volume as the benchmark input (June note still stands).
- Do not point production `pipeline_config.yaml` at research outputs.

- [ ] Failing config-builder test, implement, unit tests, optional smoke on one local MS subject (BIDS id only in logs), commit.

Long sample: June plan Task 5.1.

---

## Phase G — Metrics (was Task 6.1)

`metrics.py`: DSC, HD95, over-stripping, leakage, brain volume ml vs MNI152 brain mask as pseudo-GT; timing via `perf_counter`; RAM `psutil`; VRAM `pynvml` around the strip call (do not require `strip()` to return VRAM).

- [ ] Failing tests with tiny synthetic masks, implement, commit.

Limitation for the paper (already in spec §4): atlas pseudo-GT is imperfect under GBM mass effect — qualitative review by KR remains.

---

## Phase H — Runner + tuning (was Tasks 7.1, 8.1)

`run_benchmark.py`: tools × subjects → `results/raw_metrics.csv`. ADD-4 reproducibility: optional second pass, DSC between runs.

`tuning.py`: validation split only; write tuned params into **that tool’s** `manifest.yaml` `tuned_params` / `default_params`. HD-BET: tune only what 2.x exposes (`disable_tta`, device), not `threshold`.

BET search still `fractional_intensity` 0.2–0.5; SynthStrip `--border` 0/1/2.

- [ ] Tests with fake strippers, implement, commit.

---

## Phase I — Report, notebook, ADD-3, ADD-5 (was Tasks 9.1–9.4)

`report.py`: Friedman / ANOVA or Kruskal-Wallis + Wilcoxon + Bonferroni; figures listed in spec §7.

Notebook: `results/statistical_analysis.ipynb` — `git add -f` because `results/` is gitignored.

ADD-3 `selection_experiment.py`: manifest `lesion_type_preference` vs always-BET vs always-HDBET on the test split (headline MAS result).

ADD-4: worst-case (min) DSC per tool, not only mean.

ADD-5 `characteristic_analysis.py`: Stage 04 QA features → which tool wins. Analysis only, no runtime routing (Этап 7).

Fill **both** paper skeletons when numbers exist.

- [ ] Tests, implement, commit per June 9.x with overlay.

---

## Phase J — README + production-path note (was Tasks 10.1–10.2)

Benchmark README (install, prepare, tune, run, interpret, hardware: RTX 5070 12GB as in spec, plus the GPU-pool/`exclude_gpus` story).

`services/skull-stripping/README.md`: manifests are metadata; Docker agents are Этап 7.

- [ ] Write, commit.

---

## Phase K — Sweep + ROADMAP + DEVLOG (was Task 10.3)

- [ ] `pytest` plugin tests + `research/skull_stripping_benchmark/tests/`
- [ ] ROADMAP Этап 5.5: plugin on main; benchmark/paper in progress on `feat/skull-stripping-research-v2`
- [ ] DEVLOG summary
- [ ] Commit `docs: Этап 5.5 plugin delivered, benchmark remaining`

Winners and filled Results sections need real GPU runs on the four datasets — code can be done before those runs.

---

## Completion criteria (spec §10)

| Criterion | How it closes now |
|---|---|
| Dispatcher + `method: bet` still works | Already on `main`; keep BET as fallback |
| Plugin architecture + manifests | Partial (bet, hdbet); rest = Tasks B–E |
| `prepare_data.py` | Task F |
| `raw_metrics.csv` four datasets | Task H + data access |
| Stats notebook + figures | Task I |
| Tuned params in manifests | Task H (HD-BET: TTA/device only) |
| Paper skeleton filled | Task I, both EN and RU |
| Benchmark README | Task J |
| DEVLOG current | every task + Task K |
| 2 production winners | after real benchmark, not a code commit |
| ADD-1 BrainMaGe | Task C |
| ADD-2 cascade + validation | **Task A done** on v2 (two-tier; production hdbet→bet) |
| ADD-3 selection experiment | Task I |
| ADD-4 worst-case + reproducibility | Tasks H/I |
| ADD-5 characteristic analysis | Task I |
| Merge to main | after benchmark + paper fill (`finishing-a-development-branch`) |

---

## Suggested execution order

A (cascade) → B (SynthStrip) → D (MNI baseline, no extra install) → C / E (timeboxed) → F → G → H → I → J → K.

A can proceed with only BET+HD-BET in `STRIPPERS`; later tools join the cascade via config when they exist.

---

## Self-review notes

- June Tasks 1.1–1.3 / 2.1 / 3.1 are closed by `main`; executing them again would fork the plugin.
- Cascade extends `fallback_method`; it does not replace the GPU pool.
- Contribution framing unchanged: selection + MAS cascade on clinical GBM+MS, not another glioma-only stripper bake-off (Thakur 2020).
