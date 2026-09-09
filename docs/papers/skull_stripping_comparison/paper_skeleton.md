# Comparative Evaluation of Skull-Stripping Tools for a Heterogeneous Clinical MRI Pipeline

> Status: skeleton. `<<...>>` markers are placeholders filled from benchmark results.

> Bilingual: keep this EN skeleton and `paper_skeleton_ru.md` (RU) in sync as results arrive.

## Abstract
<<One paragraph: motivation, 8 tools, 4 datasets (GBM + MS), key finding, 2 winners.>>

## 1. Introduction
- Clinical context: automated brain-lesion diagnosis pipeline (GBM + MS), multi-center.
- Problem: skull stripping currently FSL BET, chosen empirically; may not generalize, and
  prior art ranks strippers on glioma (Thakur 2020) but not on MS or a combined clinical cohort,
  and never *selects* among competing tools.
- Contributions:
  (1) a **manifest-driven, lesion-type-aware selection component** for skull stripping built as a
      **cascade of validated stripper agents** (fault-tolerant: unavailable/invalid → next agent;
      mask-integrity validation gates each step) — MAS-ready, reusable (the headline contribution);
  (2) the first **combined GBM + MS** comparison on a **heterogeneous clinical multi-center**
      cohort, including the under-studied MS arm;
  (3) a **cost/quality** analysis (GPU DL tools vs a dilated atlas mask) for production deployment;
  (4) evidence-based selection of production tools, reported with **worst-case** robustness, not
      only mean DSC.

## 2. Related Work
<<Prose from `literature_review.md`. Structure:
  - Individual tools: FSL BET (Smith 2002), HD-BET (Isensee 2019), SynthStrip (Hoopes 2022),
    deepbet (Fisch 2024, T1-only), SAM-based (2023), BrainMaGe (Thakur 2020).
  - Comparison benchmarks: Thakur 2020 (glioma, nearest competitor), pediatric T2 (2025),
    purely-synthetic (2025); context: "is brain extraction still necessary?" (2022).
  - MAS / agentic: NeuroAgent (2026), Training-Free Agentic Neuro-Radiology (2026),
    Virtual Neuroscientist (2026), co-evolving agentic (2025), mAIstro.
  Position vs each: we add MS + combined cohort + per-cohort *selection* + MAS; agentic systems
  invoke a *fixed* stripper, we *select* among competitors with validated fallback.>>

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
- Atlas: MNI152 for all datasets. Production Stage 05 already uses `atlas.name: MNI152_FSL`
  (not SRI24). The `mni_mask` plugin (strict / ~2 mm loose) is the cheap atlas baseline
  in that same space.
- Hardware: NVIDIA RTX 5070 12GB VRAM.

### 3.6 Runtime cascade and mask-integrity gate (ADD-2)

Implemented 2026-09-04 in Stage 05 (`scripts/preprocessing_steps/skull_stripping/`).
The production chain, when `cascade` is omitted, is **`method` then `fallback_method`**
— currently HD-BET → BET. A second GPU stripper is not on the clinical path: a false
retry already pays for a full HD-BET (TTA) run.

The mask check is **two-tier**, so GBM mass effect and HD-BET speckle do not inflate
runtime. Textbook adult windows are **review signals** rather than catastrophe gates —
but review is not inert: a flagged mask advances the cascade while untried tools remain
(revised 2026-09-07, see below). MNI leakage is a warning in the benchmark (Task D), not
a Stage 05 gate — atlas ≠ GT under mass effect.

| Layer | Checks | Effect |
|---|---|---|
| Catastrophe | volume 300–2500 ml; dominant component ≥ 0.70 after dropping islands `< 1 ml`; enclosed hole volume `> 2` ml; empty/missing file | `valid=False` → try next stripper |
| Review | volume 900–2000 ml; LCC ≥ 0.95; FOV `edge_touch_ratio` `> 0.05`; enclosed holes `> 0.5` ml; left/right `asymmetry` `> 0.05`; intensity `leak_fraction` `> 0.05 %` | try the next tool if one is untried; otherwise keep the mask and log `review_flags` |
| Atlas (later) | MNI loose / very-loose outside-ratio | benchmark QA, not runtime retry |

**Revision after the first production runs (2026-09-07).** As first written, review flags
were log-only. Over six runs the gate raised four flags and switched tools zero times: it
detected the defects and shipped them anyway — MNI masks containing the patient's eyes
(1838 ml, volume flag) and one with 10.71 ml of holes across 11 cavities (under the
original 20 ml catastrophe threshold). Three changes followed:

1. Review flags now advance the cascade while untried tools remain. The flagged mask is
   held in reserve and used only if nothing cleaner appears — a flagged mask still beats
   no mask. When every candidate is flagged, the earliest wins, since cascade order is
   the operator's stated preference. `validation.retry_on_review: false` restores the
   log-only behaviour.
2. The enclosed-hole catastrophe gate drops 20 ml → 2 ml. Calibration on this project's
   masks: every correct mask measures exactly 0.00 ml of enclosed holes, and cavities
   inside a brain mask are swiss cheese rather than anatomy (ventricles belong to the
   mask as 1s), so no legitimate middle ground needed protecting.
3. Candidates now write to a scratch directory and only the accepted mask is promoted.
   Stage 05 passes one directory as both `subject_dir` and `output_dir`, so a rejected
   candidate had been overwriting the very volume the next candidate must read; a retry
   would have skull-stripped an already-stripped image. Latent until (1) made retries
   possible.

Effect on the same subjects: `mni_mask` 1838 ml (eyes) → `synthstrip` 1332 ml clean;
`mni_mask` 1827 ml (eyes + holes) → `synthstrip` 1443 ml clean; a clean HD-BET mask
(1297 ml) is still accepted on the first attempt, so the gate costs nothing when the
first tool is right.

`validation.fail_closed: false` demotes volume/LCC/hole catastrophe to flags (empty/missing
still fail — the mask cannot be applied). Metrics live in `mask_metrics()` (volume, LCC,
edge-touch, bbox fill, enclosed hole volume, left/right asymmetry about the mid-sagittal
plane in atlas space, and intensity leakage — the share of mask voxels above the head's
99th intensity percentile, which is where skull and scalp fat sit on T1). Tune via `params.validation`; if a plausible tumour mask trips
a gate, loosen the numbers rather than deleting the check.

**Calibration on a random sample (2026-09-09, dropbox_33, n = 12).** Twelve seeded-random
patients were run through the production cascade with a per-subject JSON trace recording
the cascade order, the effective gates, and every attempt with its metrics and decision.
HD-BET was accepted on the first attempt for all twelve: no switches, no flags. Visual
review against the **skull-on** volume (mask outline over the original anatomy — the
stripped output cannot show leftover skull or removed brain) confirmed all twelve are
correct, with orbital contents excluded and cerebellum and brainstem retained.

The value of the sample is the distance between correct masks and the gates:

| Gate | Worst correct mask | Threshold | Margin |
|---|---|---|---|
| enclosed holes | 0.00 ml | 0.5 ml | 100 % |
| FOV edge touch | 0.000 | 0.05 | 100 % |
| intensity leakage | 0.007 % | 0.05 % | 86 % |
| volume, review band | 1677 ml | 1800 ml | **6.9 %** |
| left/right asymmetry | 0.045 | 0.05 | **10.5 %** |

Holes, edge touch and leakage are far from firing on a correct mask, which is what a gate
should look like. The volume band and the asymmetry gate are not: on twelve subjects from
a single site one mask already sits within 7 % of a threshold. (The volume band was
widened to 900–2000 ml on 2026-09-09 once the tool matrix supplied the other side of the
measurement — see below.) Because review flags now
advance the cascade, a false alarm is no longer free — it discards a correct HD-BET mask
in favour of whatever comes next. Both are inherited defaults rather than measured values,
and both need widening: a removed hemisphere scores 0.3–0.5 on asymmetry, so a gate at
0.15 keeps the separation while leaving room for genuine pathological asymmetry.

The sample carries one structural weakness worth stating: it contains twelve positives and
no negatives. Thresholds are anchored from below by correct masks and from above only by
the two defective masks of 2026-09-07. Real negatives from the same data — the same twelve
volumes stripped by every tool, `mni_mask` being the known eye-leak case — are needed
before the numbers can be called measured rather than argued.

<<Once the four-dataset run exists: fraction of subjects that retried BET, review-flag
histogram, false-retry notes from KR visual QA.>>

## 4. Results
### 4.1 Per-tool quality (table) <<auto from report.py>>
### 4.2 Tool × dataset interaction (heatmap) <<dsc_heatmap_tool_x_dataset.png>>
### 4.3 Performance vs quality trade-off <<performance_scatter.png>>
### 4.4 Leakage / over-stripping <<leakage_comparison.png>>
### 4.5 Qualitative examples <<visual_examples/>>
### 4.6 Statistical tests <<from statistical_analysis.ipynb>>
### 4.7 Agent-selection experiment (ADD-3) <<routed vs fixed; selection_experiment.csv>>
### 4.8 Characteristic → tool analysis (ADD-5) <<which characteristics matter>>

## 5. Discussion
- Limitations: atlas pseudo-GT imperfect under GBM mass effect; runtime validation
  therefore uses morphology/volume catastrophe, not DSC vs MNI.
- **Measured: DSC against the MNI152 brain mask inverts the ranking.** On our runs a
  mask containing the patient's eyes scores 0.997 against
  `MNI152_T1_1mm_brain_mask`, while correct HD-BET masks score 0.84–0.87 — the atlas
  metric would place the broken mask first. This is not mass effect alone: the atlas
  mask *is* the defective mask, since production registration is rigid (6 DOF) and an
  atlas mask cannot fit a head it was never scaled to. Any atlas-referenced DSC in §4
  needs a different reference (consensus mask, visual scoring, or an affine/nonlinear
  arm used for the benchmark only).
- **`mni_mask` is a baseline, not a production candidate.** Its leakage onto orbital
  structures is a property of applying an unscaled atlas mask under rigid registration,
  not a defect of the wrapper — worth stating explicitly so the comparison is read as a
  statement about the approach rather than the implementation.
- An intensity-based leakage metric ("background voxels inside the mask") was prototyped
  and rejected: skull and scalp are tissue, so the measure runs backwards — 0.00% air
  inside the defective mask against 2.47% inside a correct one.
- MAS implications: manifests, cascade of validated agents (unavailable/invalid → next),
  cascade priority, lesion-type preference. Integrity gate ≠ quality ranking.
- Production cascade today is HD-BET → BET only; SynthStrip/SAM join via config after
  their wrappers exist, not as a default GPU retry.
- Recommended 2 production winners + parameters.

## 6. Conclusion
<<Two winners, when to use which, future work.>>

## References
<<From literature_notes.md.>>
