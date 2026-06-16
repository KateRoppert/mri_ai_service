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
