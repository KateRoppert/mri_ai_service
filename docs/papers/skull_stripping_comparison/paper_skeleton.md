# Comparative Evaluation of Skull-Stripping Tools for a Heterogeneous Clinical MRI Pipeline

> Status: skeleton. `<<...>>` markers are placeholders filled from benchmark results.

## Abstract
<<One paragraph: motivation, 7 tools, 4 datasets (GBM + MS), key finding, 2 winners.>>

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
