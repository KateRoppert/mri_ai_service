# Conference Presentation Design Spec
**Paper:** Profiling a Brain MRI Pipeline for Multi-Agent System Design
**Event:** EDM (IEEE IAS-scope conference)
**Date:** 2026-05-29
**Format:** 7 min talk + 7 min Q&A · 16:9 · English slides · Russian or English delivery
**Output:** PPTX + PDF from template `/home/ubuntu/Загрузки/EDM Presentation Template.pptx`

---

## Template

- Dimensions: 19.95" × 11.22" (16:9 ✓)
- Primary accent: #00AC5A (green), title text on content slides: #008F4A
- Title slide: green left panel (8.79" wide), white area right; logo top-right
- Content slides: full-width title (#008F4A, bold), content below
- Font sizes: title ~66 pt bold; body ~28 pt

---

## Slide Structure (12 slides, ~35 sec/slide)

### 1. Title (0:00–0:30)
- **Title:** Profiling a Brain MRI Pipeline for Multi-Agent System Design
- **Authors:** E. I. Roppert · B. N. Tuchinov · E. N. Pavlovskiy
- **Affiliation:** AI Research Center, Novosibirsk State University
- **Visual:** thematic brain MRI image in green panel area

### 2. Clinical Motivation (0:30–1:05)
- Brain lesions = critical diagnostic urgency: GBM median survival 15 months; MS leads to progressive irreversible disability
- Early detection changes outcomes — automated AI diagnosis must be **fast**
- Clinical bottleneck: multi-stage processing pipeline with heterogeneous resource demands
- **IAS seed:** "This is fundamentally an automated industrial workflow with stringent latency requirements — a pattern central to Industry 4.0"
- **Addresses reviewer:** establishes clinical stakes before technical framing

### 3. System: 6-Stage Pipeline (1:05–1:45)
- Fig. 1: pipeline architecture diagram (from paper, PDF → PNG)
- 6 stages: I/O-bound (1–3) → CPU-bound (4) → CPU-heavy (5) → GPU-bound (6)
- Real deployed system: FastAPI + React + Docker + YAML config
- Dataset: 100 patients, 110 sessions, UPENN-GBM

### 4. Benchmarking Methodology (1:45–2:25)
- Controlled cache-clearing protocol: `sync` + `echo 3 > /proc/sys/vm/drop_caches`
- Why critical: without clearing, 2–4× artificial speedup in I/O stages
- 1–20 parallel workers varied per stage; 2 runs where feasible
- Two-stage dataset: 103 sessions (Stage 5), 98 sessions (Stage 6, all 4 modalities)

### 5. Stage Saturation Profiles (2:25–3:05)
- **Fig. 2:** Speedup vs. parallel processes (all 5 CPU stages)
- Table I summary (compact): saturation at 6–12; stages saturate differently
- Key message: **no single worker count is optimal across all stages**
- Stage 5 (Preprocessing): saturates at 6 workers despite heavy load — internal thread contention

### 6. Bottleneck Distribution (3:05–3:35)
- Preprocessing: **47.8%** of pipeline time; Segmentation: **44.3%** → together **92.1%**
- Critical insight: complementary resources (CPU vs GPU) → can execute **concurrently without contention**
- Visual: horizontal bar or donut showing time distribution

### 7. MAS Architecture (3:35–4:15)
- 3-tier: Broker Agent → Stage Manager Agents (×6, BDI) → Worker Processes
- BDI mapping: Beliefs = saturation profiles; Desires = minimize queue; Intentions = allocate/release
- Contract Net Protocol: Announcement → Call for bids → Award
- Diagram: clean 3-tier architecture diagram (drawn in python-pptx)

### 8. Simulation Results (4:15–4:55)
- **Fig. 3:** Pipeline makespan comparison (4 strategies × 3 batch sizes)
- Headline: **4.7× over Sequential** · **44% over Pipeline Parallel** at 100 patients
- GPU overlap eliminates 25 min GPU idle time (100 patients)
- MAS robust to overhead: 10× higher overhead → still 21% faster than Pipeline Parallel

### 9. Clinical & Operational Value (4:55–5:30)
- **Addresses "10 patients/day" reviewer concern:**
  - Correct baseline is Sequential (current clinical practice), not Pipeline Parallel
  - 20 patients: 21.4 min → 5.4 min = **4× speedup = 16 minutes saved**
  - The 19% figure compares MAS to Pipeline Parallel (already optimized) — not fair baseline
  - Latency matters: first segmentation result in ~5 min (streaming) vs 21 min
  - Multi-center & research batches: routinely 100+ patients
- **Methodology reusable:** run benchmarking script → get site-specific saturation profiles → configure MAS

### 10. vs. Workflow Orchestrators (5:30–6:00)
- **Addresses Airflow reviewer concern:**
  - Apache Airflow / Prefect / Luigi → DAG scheduling with fixed parallelism per task = **Strategy 2 (Stage-Sequential, 2.29×)**
  - Adding Pipeline Parallel allocation ≈ Strategy 3 (3.26×) — possible with manual tuning
  - MAS adds: saturation-aware dynamic reallocation → **Strategy 4 (4.7×), +44% over Airflow-class**
  - Key distinction: Airflow automates. MAS **optimizes** through empirical beliefs.
- **IAS connection (organic):**
  - Same pattern: NDT pipelines, quality control lines, predictive maintenance data flows
  - Multi-agent resource management is a core IAS topic — this is an instantiation in medical AI

### 11. Limitations & Roadmap (6:00–6:35)
- **Simulation only** → implementation in existing FastAPI/Docker infrastructure is natural next step (codebase exists, shown tonight)
- **Single dataset (UPENN-GBM)** → methodology is reusable; profiling script is portable
- **Optimistic overhead estimate** → robustness analysis shows even 10× overhead still wins
- Roadmap: (1) prototype BDI agents, (2) validate on MS dataset, (3) multi-node distributed deployment
- **Tone:** forward-looking, not defensive

### 12. Conclusion (6:35–7:00)
- 3 takeaways:
  1. Empirical saturation profiles reveal heterogeneous behavior (6–12 workers) — stage-uniform allocation wastes resources
  2. MAS with BDI agents achieves 4.7× speedup, 31% reduction vs static Pipeline Parallel
  3. Profiling methodology is site-portable; profiling data serves directly as agent beliefs
- QR code → GitHub: KateRoppert/mri_ai_service
- "Thank you" + affiliation

---

## Figures

| Figure | Source | Usage |
|--------|--------|-------|
| Fig. 1 Pipeline | `fig1_pipeline_architecture.pdf` → convert to PNG | Slide 3 |
| Fig. 2 Speedup curves | Regenerate from paper data via matplotlib (shared as image, no standalone file) | Slide 5 |
| Fig. 3 Makespan | `simulation_results/makespan_comparison.png` | Slide 8 |
| Bottleneck bar | Generate via matplotlib from Table I data | Slide 6 |
| MAS 3-tier diagram | Draw in python-pptx (shapes + arrows) | Slide 7 |

---

## Speaker Notes

Each slide gets notes in **both Russian and English**, structured as:
```
[RU] …
[EN] …
```

---

## Technical Implementation

- **Language:** Python + python-pptx
- **Charts:** matplotlib (high-res PNG, embedded)
- **Fig. 1 extraction:** `pdftoppm` or `pdf2image` (poppler)
- **PDF export:** `libreoffice --headless --convert-to pdf`
- **Output path:** `/home/ubuntu/mri_ai_service/docs/presentation/`

---

## Reviewer Concerns — Coverage Map

| Concern | Addressed on slide | How |
|---------|-------------------|-----|
| MAS not implemented | 11 | Roadmap; deployed codebase exists |
| Single dataset | 11 | Methodology reusable; portable profiling script |
| No Airflow comparison | 10 | Airflow ≡ Strategy 2 (2.29×); MAS +44% on top |
| Clinical volume / ROI | 9 | Correct baseline (Sequential); latency argument; streaming |
| Moderate novelty | 5, 7 | Empirical profiling → BDI beliefs bridge is the contribution |

---

## IAS Scope Alignment

Slide 2 (seed) + Slide 10 (explicit):
- Multi-stage AI pipelines with heterogeneous resource demands appear throughout industrial AI
- Resource optimization under heterogeneous compute = core IAS topic
- This work demonstrates the profiling → MAS design methodology that generalizes beyond medical imaging
