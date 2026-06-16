# Analytical Literature Review — Skull Stripping Tool Selection for a Heterogeneous Clinical MRI Pipeline (MAS context)

> **Role of this document:** reference base ("опора") for the skull-stripping research (Этап 5.5)
> and the comparison paper. Compiled 2026-06-16. Covers only key, high-quality works
> (2002 baseline + 2019–2026 focus). Each entry follows the agreed extraction card:
> A. Identification · B. Method · C. Evaluation · D. Results · E. Relevance to us (incl. MAS) · F. Gap left open.
>
> **Agreed contribution framing:** prior comparison of strippers on glioma is largely done
> (Thakur 2020). Our contribution is positioned on **(1) skull-stripper selection for our specific
> heterogeneous clinical GBM+MS multi-center pipeline** and **(2) the plugin / MAS architecture**
> (config-driven, manifest-based selection with cascade/fallback). MS is an under-studied gap;
> cost/quality (GPU tool vs dilated atlas mask) is a supporting argument.

---

## 1. Review method

- **Search period:** 2019–2026 (FSL BET 2002 included as the classical baseline).
- **Venues:** NeuroImage, Human Brain Mapping, Computers in Biology and Medicine, Frontiers in
  Neuroscience, arXiv/medRxiv (for 2026 MAS preprints not yet in journals).
- **Selection:** only the strongest / most-cited / most directly relevant works, grouped as
  (a) individual tools, (b) comparison benchmarks, (c) MAS / agentic works directly relevant to
  orchestrating or selecting preprocessing tools.
- **MAS scope note:** a separate MAS-in-medicine review already exists in the project; here we add
  only works that (i) wrap skull stripping / preprocessing as agent-invoked tools, or (ii) are the
  freshest 2026 MAS works absent from the prior report.

---

## 2. Per-paper cards

### (a) Individual tools

#### 2.1 FSL BET — Smith 2002
- **A.** Smith, S.M. *Fast robust automated brain extraction.* Human Brain Mapping, 2002.
  Open tool (FSL). URL: https://doi.org/10.1002/hbm.10062 · https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/BET
- **B.** Classical, deformable surface model expanding to the brain edge. CPU-only, seconds.
  Key params `-f` (fractional intensity), `-g` (vertical gradient), `-R` (robust center).
- **C.** Validated historically on healthy adult T1/T2; not designed for pathology.
- **D.** Robust and fast; the de-facto baseline. Degrades with tumors, strong bias field, non-standard FOV.
- **E.** Our **current production default** (`-f 0.35 -g -0.1 -R`), chosen empirically. Trivially
  wrappable as an agent (CPU, no GPU). The reference every comparison must beat.
- **F.** No pathology awareness; no multi-modal fusion. Leaves open: how far modern tools beat it on
  *our* clinical GBM+MS distribution, and whether the gain justifies GPU cost.

#### 2.2 HD-BET — Isensee et al. 2019
- **A.** Isensee, F. et al. *Automated brain extraction of multisequence MRI using artificial neural
  networks.* Human Brain Mapping, 2019. Open: https://github.com/MIC-DKFZ/HD-BET
- **B.** Deep CNN (3D U-Net lineage from the nnU-Net group). Multi-sequence (T1/T1c/T2/FLAIR).
  GPU recommended (~4 GB VRAM); CPU fallback is much slower.
- **C.** Trained/validated on large multi-center data including glioma (BraTS-like distribution).
- **D.** Strong, robust on tumor data; widely adopted as the tumor-MRI default. Reported median
  DSC ~97.9% in later comparisons.
- **E.** Top candidate for our **GBM arm** and a strong MS candidate. GPU agent analogous to our
  gbm-seg/ms-seg services. Natural `cascade_priority: 1` in manifests.
- **F.** GPU dependency raises the cost/quality question vs cheaper baselines — exactly our
  supporting argument.

#### 2.3 SynthStrip — Hoopes et al. 2022
- **A.** Hoopes, A. et al. *SynthStrip: skull-stripping for any brain image.* NeuroImage, 2022.
  URL: https://arxiv.org/abs/2203.09974 · ships with FreeSurfer (`mri_synthstrip`), `surfa`.
- **B.** U-Net trained with **synthetic data** to be **modality-agnostic** (any contrast/resolution).
  CPU-capable; small GPU optional. `-b/--border` parameter controls mask tightness.
- **C.** Evaluated across many modalities/populations; outperforms ROBEX, BET, BEaST, 3DSS, etc.
- **D.** Median DSC ~97.8%; exceptionally general across contrasts.
- **E.** Strong **universal default**, especially valuable for our **MS/FLAIR** cases and any
  subject missing T1. CPU-capable → cheap agent. Likely `cascade_priority: 2`, fallback for HD-BET.
- **F.** Mean DSC slightly below specialized tools on their home turf; question is whether
  modality-agnostic generality wins on our heterogeneous cohort.

#### 2.4 deepbet — Fisch et al. 2023/2024
- **A.** Fisch, L. et al. *deepbet: Fast brain extraction of T1-weighted MRI using CNNs.*
  Computers in Biology and Medicine, 2024 (preprint https://arxiv.org/abs/2308.07003).
- **B.** **LinkNet** (modern U-Net) in a two-stage prediction. **T1-weighted only.** Very fast
  (~2 s on low-end hardware, ~10× faster than prior tools).
- **C.** 568 T1w images from 191 studies; cross-validation.
- **D.** **Median DSC 99.0%**, min >96.9%; markedly more robust to outliers than competitors
  (which dropped to ~76.5% worst-case). Outperforms the ~97.8/97.9% tools on T1.
- **E.** Excellent for **T1-only, high-throughput** use; its **worst-case robustness** result is the
  direct empirical basis for our **ADD-4 (worst-case DSC metric)**. Fast CPU agent.
- **F.** **T1-only** — not directly usable for our T1c/T2/FLAIR or MS-FLAIR streams without
  per-modality handling. Best-effort inclusion in our benchmark (T1 arm only).

#### 2.5 SAM-based brain extraction — 2023
- **A.** *SAM vs BET: A Comparative Study for Brain Extraction…* 2023,
  https://arxiv.org/abs/2304.04738 ; also SAM-for-fMRI 2024 https://arxiv.org/abs/2401.04740.
- **B.** Foundation segmentation model (Meta SAM), not trained on medical images; zero-/few-shot,
  prompt-driven. GPU-heavy.
- **C.** Compared against BET on standard MRI.
- **D.** Can match or beat BET in places; inconsistent, prompt-sensitive, not turnkey.
- **E.** Our **experimental, best-effort** tool. Interesting as a "general foundation model" data
  point but not a production candidate. Note: our prior-art comparison SAM↔BET already exists, so
  we frame SAM only as a contextual baseline, not a contribution.
- **F.** No clinical-grade reliability; the SAM↔BET comparison is published → no novelty there for us.

### (b) Comparison benchmarks

#### 2.6 Thakur et al. 2020 — BrainMaGe (the nearest GBM competitor)
- **A.** Thakur, S. et al. *Brain extraction on MRI scans in presence of diffuse glioma:
  multi-institutional performance evaluation of DL methods and robust modality-agnostic training.*
  NeuroImage, 2020. Code: https://github.com/CBICA/BrainMaGe · https://pmc.ncbi.nlm.nih.gov/articles/PMC7597856/
- **B.** Compared 2D-ResInc, 3D U-Net, 3D-Res-U-Net, FCN, **DeepMedic**; introduced
  **modality-agnostic training** (modalities fed as independent samples → network learns a
  "brain-shape prior", robust to missing modalities).
- **C.** **3,220 mpMRI scans / 805 subjects**, multi-institutional (UPenn 1,812; Thomas Jefferson 608;
  MD Anderson 100; TCIA TCGA-GBM/LGG 700). Modalities T1/T1Gd/T2/FLAIR. Pathology: diffuse glioma.
  **GT: semi-automatic + board-certified neuro-radiologist approval.** Metrics: DSC, HD95.
- **D.** 2D-ResInc best for single-modality; modality-agnostic model comparable to T1-T1 and Multi-4.
  Released as **BrainMaGe** (also in CaPTk).
- **E.** This is **our GBM arm's prior art** — and now, per **ADD-1**, **BrainMaGe becomes a tool in
  our own benchmark**, letting us position directly against it on *our* clinical cohort.
- **F.** **Glioma only, pre-operative only, non-defaced.** Explicitly untested on other diseases
  (e.g., **MS**) and on multi-center *MS* data → our combined GBM+**MS** scope is the gap. Also: no
  cost/quality framing, no agent/MAS selection.

#### 2.7 Pediatric T2 skull-stripping comparison — 2025
- **A.** *Skull stripping tools in pediatric T2-weighted MRI scans: a retrospective evaluation.*
  Frontiers in Neuroscience, 2025. https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2025.1715514/full
- **B.** Benchmark of **7 tools**: BET, ROBEX, HD-BET, HD-BET-fast, SynthStrip, SynthStrip-noCSF, d-SynthStrip.
- **C.** Pediatric **T2-weighted** scans; segmentation-quality evaluation.
- **D.** HD-BET ~1% above SynthStrip on T2 pediatric; tool ranking is population/modality dependent.
- **E.** Methodological template for a clean multi-tool comparison; reinforces that **rankings are
  population-specific** → supports our "select per cohort" thesis.
- **F.** Pediatric, T2, healthy-ish — not adult clinical tumor/MS. Different population → does not
  preempt us.

#### 2.8 Skull stripping with purely synthetic data — 2025
- **A.** *Skull stripping with purely synthetic data.* 2025, https://arxiv.org/html/2505.07159v1
- **B.** Extends the SynthStrip-style synthetic-training paradigm.
- **C./D.** Competitive DSC without real training data; confirms the synthetic/modality-agnostic trend.
- **E.** Evidence that the field's direction is generalization-by-synthesis → validates including
  SynthStrip as our universal default.
- **F.** Still benchmarks on standard distributions, not our clinical GBM+MS multi-center mix.

#### 2.9 (context) "Is brain extraction still necessary?" — 2022
- **A.** *Towards fully automated DL-based brain tumor segmentation: is brain extraction still
  necessary?* 2022, https://arxiv.org/abs/2212.07497
- **B./D.** Argues modern tumor-seg nets can tolerate skipping skull stripping in some setups.
- **E.** Important **Discussion** point: justifies *why* we still skull-strip (downstream registration,
  lobar localization, MS volumetrics, anonymization) rather than dropping it.
- **F.** Does not address tool *selection*; orthogonal to our contribution.

### (c) MAS / agentic works (the contribution angle)

#### 2.10 NeuroAgent — 2026 (closest MAS framing)
- **A.** Zhong, L. et al. *NeuroAgent: LLM Agents for Multimodal Neuroimaging Analysis and Research.*
  arXiv, May 2026. https://arxiv.org/abs/2605.06584
- **B.** **Hierarchical multi-agent** architecture with a feedback-driven **Generate-Execute-Validate**
  engine: agents generate executable preprocessing code, detect/recover runtime errors, and
  **validate output integrity**.
- **C.** sMRI/fMRI/dMRI/PET; evaluated on **1,470 ADNI** subjects.
- **D.** **84.8%** end-to-end preprocessing-step correctness (Qwen3.5-27B backend); 0.9518 AUC for AD
  classification downstream.
- **E.** **Directly informs ADD-2:** their *Validate* stage (output-integrity check) is the model for
  our **mask-integrity validation → fallback** in the dispatcher. But NeuroAgent **generates code for
  a fixed pipeline**, it does **not select among competing tools** for one step.
- **F.** **No comparative tool selection**, no manifest-driven routing by lesion type → precisely our
  niche.

#### 2.11 Training-Free Agentic Neuro-Radiology — 2026 (strippers as agent-invoked tools)
- **A.** *Agentic Large Language Models for Training-Free Neuro-Radiological Image Analysis.* 2026.
  https://arxiv.org/html/2604.16729v1
- **B.** LLM orchestrates **external specialized tools** (no intrinsic 3D processing): preprocessing
  (**skull stripping**, registration), pathology segmentation, volumetric analysis.
- **C./D.** Validated across several LLM backends; autonomous end-to-end workflows.
- **E.** Most concrete evidence that **skull stripping is now an agent-invoked tool**. Confirms our
  service-per-tool design is aligned with the field — yet again, a **fixed** stripper is called, not
  a *selected* one. Our manifest-based selection extends this pattern.
- **F.** No per-data tool selection / cascade; no quality-driven fallback among strippers.

#### 2.12 Virtual Neuroscientist — 2026
- **A.** *Towards a Virtual Neuroscientist: Autonomous Neuroimaging Analysis via Multi-Agent
  Collaboration.* 2026. https://arxiv.org/html/2605.09366v3
- **B./E.** Multi-agent collaboration building/executing/QC-ing workflows from raw data + a goal;
  delivers reusable analysis packages. Reinforces the "reusable agentic component" framing (T4) for
  our selection module.
- **F.** Workflow-level orchestration, not step-level tool selection.

#### 2.13 Co-evolving agentic medical imaging — 2025
- **A.** *A co-evolving agentic AI system for medical imaging analysis.* 2025.
  https://arxiv.org/pdf/2509.20279
- **B./E.** Self-evolving agents acquiring reusable composite skills from prior trajectories (MACRO
  lineage). Future-work inspiration: a selection agent that *learns* tool routing from accumulated
  benchmark outcomes.
- **F.** General medical imaging; no skull-stripping specialization.

#### 2.14 mAIstro — open-source multi-agent for medical imaging
- **A.** *mAIstro: open-source multi-agent system for automated end-to-end radiomics and DL model
  development for medical imaging.* https://www.sciencedirect.com/science/article/pii/S3050577125000428
- **B./E.** Practical open multi-agent reference architecture; useful as an engineering comparator for
  our MAS design and for citing an existing open multi-agent medical-imaging system.
- **F.** Radiomics/model-dev focus; not preprocessing tool selection.

> **Reference list (curated):** https://github.com/AgenticHealthAI/Awesome-AI-Agents-for-Healthcare
> (tracks 2026 agentic-healthcare advances; use to stay current).

---

## 3. Synthesis tables

### 3.1 Tools overview

| Tool | Class | Modalities | GPU | Pathology in eval | Reported DSC | In our benchmark |
|---|---|---|---|---|---|---|
| FSL BET | classical | T1/T2 | no | healthy | baseline | yes (baseline/default) |
| HD-BET | DL CNN | T1/T1c/T2/FLAIR | rec. | glioma (BraTS) | ~97.9% | yes (GBM candidate) |
| SynthStrip | DL synth/agnostic | any | optional | many | ~97.8% | yes (universal default) |
| deepbet | DL LinkNet | **T1 only** | optional | mixed/healthy | **99.0%** (T1) | yes (T1 arm, best-effort) |
| SAM-based | foundation | any (prompted) | yes | standard | variable | experimental/best-effort |
| BrainMaGe (ADD-1) | DL modality-agnostic | T1/T1c/T2/FLAIR | rec. | **glioma** | (Thakur 2020) | **yes (new, GBM)** |
| MNI strict/loose | atlas | any (registered) | no | n/a | — | yes (cost baseline) |

### 3.2 "Who already compared what" (prior art coverage)

| Prior study | Tools | Population | Covers our niche? |
|---|---|---|---|
| Thakur 2020 | DeepMedic, U-Nets, FCN | **glioma**, multi-center adult | GBM ranking — yes; **MS — no**; selection/MAS — no |
| deepbet 2023 | vs SynthStrip/HD-BET | T1, mixed | T1 ranking — yes; multimodal/MS — no |
| Pediatric 2025 | 7 tools incl. HD-BET/SynthStrip | **pediatric T2** | different population |
| SAM vs BET 2023 | SAM, BET | standard MRI | SAM context only |
| **Ours** | 8 tools + atlas | **clinical GBM+MS multi-center** | **MS gap + per-cohort selection + MAS** |

---

## 4. Gap analysis

1. **MS skull-stripping comparison is essentially unstudied** as an object of evaluation — MS papers
   treat stripping as a fixed preprocessing step. (Cards 2.6, 2.8; MS search yielded only
   lesion-segmentation work.)
2. **No combined GBM+MS clinical multi-center comparison** on one heterogeneous distribution.
3. **No data-driven tool *selection*** among competing strippers — every agentic system (2.10–2.11)
   calls a *fixed* tool. Manifest-driven, lesion-type-aware routing with cascade/fallback is unfilled.
4. **No cost/quality decision framing** (GPU DL tool vs dilated atlas mask) on a production distribution.
5. **Mean DSC saturates (~98–99%)** → worst-case robustness is the real discriminator, rarely reported.

---

## 5. Contribution statement (vs nearest competitors)

- **vs Thakur 2020 (BrainMaGe):** we do not re-rank strippers on glioma; we (i) extend to **MS** and
  **combined GBM+MS** on our clinical multi-center cohort, (ii) include **BrainMaGe itself** as a
  candidate (ADD-1), and (iii) deliver a **selection mechanism**, not just a ranking.
- **vs deepbet 2023:** we go **multimodal** (T1c/T2/FLAIR + MS-FLAIR), and adopt their key insight —
  **worst-case robustness** — as a first-class metric (ADD-4).
- **vs pediatric 2025:** different (adult clinical tumor/MS) population; we add selection + MAS.
- **vs NeuroAgent / Training-Free Agentic 2026:** they invoke a **fixed** stripper inside an agentic
  pipeline; we contribute **manifest-driven selection among competing strippers** with
  **integrity-validated fallback** (ADD-2) — a reusable MAS component, evaluated via an explicit
  **selection experiment** (ADD-3).

**Headline:** *A config-driven, manifest-based skull-stripper selection component for a heterogeneous
clinical GBM+MS pipeline — MAS-ready, with integrity-validated fallback — that beats any single fixed
tool across the combined cohort.*

---

## 6. Takeaways for our project

**Trends**
- **T1 — Mean DSC has saturated (~98–99%).** The discriminator is robustness on pathology/OOD and
  worst-case, not the mean. → emphasize leakage, over-stripping, and **worst-case DSC** (ADD-4).
- **T2 — Modality-agnostic / synthetic training dominates** (SynthStrip, BrainMaGe modality-agnostic,
  purely-synthetic 2025). → SynthStrip as universal default; important for MS-FLAIR / missing-T1.
- **T3 — Agentic 2026 systems invoke preprocessing tools but never *select* among competitors.**
  → our manifest-driven selection + cascade/fallback is a genuine, unfilled MAS niche (the headline).
- **T4 — Field moving to end-to-end agentic benchmarks** (ReX-MLE, MedMASLab). → frame our work as a
  reusable selection component, evaluated on both stripping quality and routing behavior.

**To reuse**
- BrainMaGe's **"brain-shape prior"** idea and its open model → adopted as a tool (ADD-1).
- NeuroAgent's **Generate-Execute-Validate / output-integrity** → adopted as dispatcher
  **mask-integrity validation → fallback** (ADD-2).
- Pediatric-2025's clean **multi-tool comparison methodology** → mirror its rigor.

**Adopted additions (confirmed with KR 2026-06-16)**
- **ADD-1:** BrainMaGe added as an 8th tool (wrapper + manifest).
- **ADD-2:** **cascade** selection with mask-integrity validation in the dispatcher — an ordered tool
  chain (`cascade: [hdbet, synthstrip, bet]`) unifying **fault tolerance** (unavailable tool → next
  agent) and **validate-retry** (mask fails plausible-volume + single-connected-component thresholds →
  next agent). Each stripper = an agent; first one passing validation wins.
- **ADD-3:** explicit "manifest-driven agent selection" experiment (route GBM→X, MS→Y; beat any
  single fixed tool on the combined cohort).
- **ADD-4:** worst-case / reproducibility DSC as a first-class metric.
- **ADD-5:** offline **input-characteristic → best-tool analysis**, reusing Stage 04 quality metrics
  (SNR/CNR/EFC/FBER/…) + metadata, to identify which characteristics influence tool choice. Justifies
  MAS routing; a runtime adaptive selector is deferred to Этап 7 (future work).

> **MAS framing note (KR PhD):** the runtime backbone is the **cascade of validated stripper agents**
> (ADD-2) with lesion-type routing (ADD-3); ADD-5 supplies the evidence base for *which* characteristics
> should drive a future adaptive selector. Adaptive runtime selection by characteristics is explicitly
> out of scope for this stage.

**Figures worth reproducing in the paper** (save into `figures/` when assembling):
- Thakur 2020: modality-agnostic training schematic + per-architecture DSC/HD95 table (PMC7597856).
- SynthStrip 2022: synthetic-data generation + modality-agnostic pipeline diagram (arXiv 2203.09974).
- NeuroAgent 2026: hierarchical agent / Generate-Execute-Validate diagram (arXiv 2605.06584).
- Our own: dispatcher selection/cascade diagram (to be drawn — supports the MAS contribution).
