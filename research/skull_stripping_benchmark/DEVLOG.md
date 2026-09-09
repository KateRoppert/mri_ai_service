# Skull Stripping Benchmark — Development Log

Updated at the end of each working session: date · done · blockers · next step.
Tracks actual vs estimated timeline (estimate: 6–8 weeks part-time).

## 2026-09-09 (volume review band widened — Kate approved)

- **Done:** `DEFAULT_REVIEW_MIN_ML` 1000 → 900, `DEFAULT_REVIEW_MAX_ML`
  1800 → 2000. Config comment, paper 3.6 (EN+RU) and two regression tests
  updated. Asymmetry left at 0.05 — it caught BET on sub-049 (0.063), so it
  is earning its place.
- **Verified on the 48 real masks, not in theory.** Re-validating the whole
  tool matrix with the new defaults reproduces every previous verdict:
  hdbet 12/12 clean, synthstrip 11 clean + 1 true-positive leak, bet 3
  rejected + 1 asymmetry flag, mni_mask 12/12 flagged. The only change is
  that `mni_mask` now reports `MASK_LEAKAGE` alone instead of
  `MASK_VOLUME_TOO_LARGE;MASK_LEAKAGE` — detection is unchanged, one
  uninformative flag is gone.
- **Why it was worth doing:** the old ceiling sat 58 ml below the worst
  known-bad mask, so it separated nothing while standing ready to fire on a
  large head or a tumour with oedema — and since 2026-09-07 a review flag
  discards a correct mask in favour of the next tool.
- **Blockers:** none.
- **Next step:** decide whether the paper needs DSC at all. Consensus (and
  therefore Phase C) exists only to make a DSC ranking possible; failure
  rate, runtime and the reference-free metrics need no reference.

## 2026-09-09 (tool matrix — 12 subjects x 4 tools)

- **Done:** `run_tool_matrix.py` runs every stripper over identical input and
  records metrics + timing; `atlas_space.py` holds the resample shared with
  the contact sheet; `analyze_matrix.py` builds a leave-one-out STAPLE
  consensus and scores each tool on it. `make_contact_sheet.py` gained a
  `--matrix-dir --tool` mode. 27 unit tests.
- **Input** is the atlas-space skull-on volume, i.e. exactly what Stage 05
  hands its stripper. Tool parameters are read from the live
  `preprocessing_config.yaml`, not restated, so the comparison matches
  production. 48 runs, no tool crashed.

| tool | accepted | volume ml | max leak | mean s |
|---|---|---|---|---|
| hdbet | 12/12 | 1125–1677 | 0.007 % | 15.3 |
| synthstrip | 12/12 (1 flag) | 1332–1780 | 0.051 % | 4.0 |
| bet | 9/12 | 1371–2956 | 2.217 % | 5.2 |
| mni_mask | 0/12 clean | 1838 (constant) | 1.039 % | 0.5 |

- **Leakage is the metric that works, and it is now measured from both
  sides.** Correct HD-BET masks top out at 0.007 %; every known-bad
  `mni_mask` starts at 0.058 %. The 0.05 % threshold sits in that gap — it
  was a guess on 2026-09-07 and is now bracketed by real data.
- **The one SynthStrip flag is a true positive.** sub-042, 0.051 %, volume
  1453 ml (unremarkable). The contact sheet shows the mask bulging over a
  bright frontal-left blob. Volume would never have caught it; leakage did.
- **The volume review band is nearly useless in the range that matters.**
  Best good mask 1780 ml (synthstrip sub-039), worst known-bad mask 1838 ml
  (`mni_mask`, all subjects). 58 ml apart, and the 1800 ml gate sits inside
  that gap by luck rather than design. It flags nothing leakage does not
  already flag, while risking a false retry on any large head. Recommend
  widening to ~900–2000 ml and letting leakage carry the load.
- **Visual finding — the gate is a catastrophe detector, not a quality
  ranker, and now there is data.** Eight BET masks passed every gate but are
  visibly looser than HD-BET's, riding on a rim of CSF and dura. Leakage
  cannot see it: dura and CSF are not bright, and the metric counts voxels
  above the head's 99th intensity percentile. The three BET masks it did
  reject are visually catastrophic (whole scalp inside the mask), so the
  gate is doing its actual job.
- **Reference: consensus works mechanically but is not yet trustworthy.**
  Leave-one-out STAPLE gives synthstrip 0.937, hdbet 0.923, bet 0.899,
  mni_mask 0.858. The ranking is an artefact of size: each tool's reference,
  built from the other three, averages +225 ml against HD-BET's own masks
  and +3 ml against SynthStrip's. SynthStrip wins by sitting closest to the
  mean, and the mean is dragged up by the two poor tools. Pairwise agreement
  says the same thing more honestly — hdbet vs synthstrip 0.945 is by far
  the highest, and both agree with mni_mask at 0.84.
- **Consequence for the plan:** a consensus reference needs more good raters
  than we have. Phase C (BrainMaGe) and Phase E stop being optional extras
  and become prerequisites for any DSC number in the paper. Until then the
  defensible outputs are failure rate, runtime, and the reference-free
  metrics — not a DSC ranking.
- **Blockers:** none. **Next step:** Kate's call on widening the volume band,
  then Phase C to get the consensus a fourth credible rater.

## 2026-09-09 (calibration sample — dropbox_33, n=12)

- **Done:** seeded random sample of 12 dropbox_33 patients through stages
  01/03/05 with `validation.trace_dir` set (`run_cascade_sample.py`). One
  JSON trace per subject: cascade order, effective gates, every attempt with
  its metrics and decision. Experiment-only — production still logs one line.
- **Result: HD-BET accepted first try for all 12.** No tool switches, no
  review flags. Volumes 1125–1677 ml, holes 0.00 ml everywhere, LCC 1.000,
  edge-touch 0.000, asymmetry 0.001–0.045, leak 0.000–0.007 %.
- **`/dev/shm` fix confirmed (`1077c70`).** The previous run of this same
  sample had HD-BET fail on all 12 (`hdbet:strip_failed -> synthstrip:OK`);
  after `shm_size: 4g` it succeeds on all 12 under the same Stage 05
  parallelism. 15–18 s per subject on GPU.
- **Contact sheet:** `make_contact_sheet.py` renders every subject as a row
  of slices — 5 axial + mid coronal + mid sagittal — with the mask outline
  over the **skull-on** volume. Leftover skull and cut brain are invisible
  against the stripped output, which is why the background has to be the
  original anatomy. No re-run needed: `transformations/` keeps both the brain
  mask (it survives `cleanup: true`) and the T1→atlas affine, so the raw
  NIfTI resamples into atlas space in 0.4 s. Alignment verified — 0.0000 of
  the mask lands on background. Visual verdict: 12/12 correct, eyes and
  orbital fat excluded, cerebellum and brainstem kept.
- **Calibration finding — two review thresholds are too tight.** Margins from
  the worst correct mask to the gate: holes 100 %, edge-touch 100 %, leak
  86 %, but **volume review band only 6.9 %** (1677 ml observed vs 1800 ml
  gate) and **asymmetry 10.5 %** (0.045 vs 0.05). Since `d842861` review
  flags advance the cascade, a false alarm is no longer free — it throws away
  a good HD-BET mask for a worse one. Recommendation pending Kate: widen the
  review band to ~900–2000 ml and asymmetry to ~0.15 (a removed hemisphere
  scores 0.3–0.5, so the separation stays large).
- **Weakness of this calibration:** 12 positives, no negatives from real
  data. Thresholds are currently anchored on one side by correct masks and on
  the other by the two defective masks from 2026-09-07 (10.71 ml holes,
  0.63 %/0.19 % leak). Proposed fix: run the same 12 registered volumes
  through every tool (`bet`, `synthstrip`, `mni_mask`) — `mni_mask` is the
  known eye-leak case — giving real negatives from the same data. That is
  also most of Phase G/H.
- **Side finding (KI-058):** `scripts/data/templates/MNI152_T1_1mm.nii.gz`
  turned out to be a 39 KB XHTML page. `download_sri24_atlas()` writes any
  HTTP 200 body without checking it is NIfTI, and the filename comes from
  `atlas.name` while the URL is hardcoded to SRI24. Junk deleted, issue filed.
- **Blockers:** none.
- **Next step:** Kate's verdict on the contact sheet + the two thresholds,
  then the per-tool matrix on the same 12 subjects.

## 2026-09-04 (Task D follow-up — holes)
- **Done:** FSL `MNI152_T1_1mm_brain_mask` has ~11 ml enclosed cavities; NN resample onto an ANTs affine can add speckles. LCC of the foreground stays ~1.0, so the old gate accepted swiss-cheese masks. `MniMaskStripper` now `binary_fill_holes` after dilate. `mask_metrics` reports `hole_volume_ml` / `n_holes`; catastrophe if enclosed holes `> 20` ml, review if `> 2` ml. Tests: 53 passed.
- **Blockers:** none.
- **Next step:** Kate re-runs Stage 05 with `method: mni_mask` — brain envelope should be solid; log line includes `holes=… ml`. Then commit Task D + this fix together.

## 2026-09-04 (Task D)
- **Done:** `MniMaskStripper` (`strict` / `loose` ~2 mm), `STRIPPERS["mni_mask"]`, manifest, FSL `MNI152_T1_1mm_brain_mask.nii.gz` in `data/templates/` (not a threshold on the skull-on T1). Production atlas comment: already `MNI152_FSL` (Kate: keep MNI152 in prod). Tests: 48 passed in the plugin suite.
- **Blockers:** none.
- **Next step:** after Kate's check → commit Task D, then C/E (timeboxed BrainMaGe/SAM/DeepBET) or F (`prepare_data.py`).

## 2026-09-09 (Phase F — prepare_data.py)

- **Done:** `research/skull_stripping_benchmark/prepare_data.py` + 13 tests.
  Runs production Stage 05 with skull stripping off, keeps the
  registration-space output as the benchmark input. Config derived from the
  live `preprocessing_config.yaml` (not hand-written, so it cannot drift from
  the pipeline it characterises); skip-existing via `is_complete_nifti`.
- **June draft was stale in three places**, all caught by running it: Stage 05
  takes input/output *positionally* (draft used `--input-dir/--output-dir`);
  modalities come from `--lesion-type` via `lesion_types.yaml`, not the
  config's `modalities` key; there is no `--transform-dir` (derived as
  `output_dir.parent/transformations`).
- **Two more found only by the smoke run:** paths must be absolute and the
  subprocess must run from the project root, as production does — otherwise
  relative paths resolve against `scripts/` and ANTs cannot open the atlas.
- **Production fix (small, tested):** Stage 05 checked for FSL unconditionally,
  before looking at whether anything needed it. FSL serves exactly one step
  (BET); reorient is nibabel, bias correction and registration are ANTs. A run
  with skull stripping disabled was blocked by a dependency it never uses. Now
  conditional — an enabled step still requires FSL, since BET is the cascade's
  universal fallback. `requires_fsl()` + 3 tests.
- **Smoke:** sub-024 from `KA126_4/nifti`, 4 modalities, 147 s. Output
  182x218x182 (atlas geometry) with 6173 ml of non-zero tissue vs ~1400 ml for
  a brain-only mask — skull present. Verified visually as well.
- **Note for Phase G:** these skull-on volumes are exactly what was missing
  when the intensity-based leakage metric was prototyped on 2026-09-07 (every
  file under `preprocessed/` is already masked). If that metric is revisited,
  calibrate it here.
- **Blockers:** none.
- **Next step:** Phase G — metrics. Open question from 2026-09-07 still stands:
  DSC against MNI152 inverts the ranking, so the reference needs deciding
  before the four-dataset run.

## 2026-09-09 (environment)

- **Broke:** host `venv` stopped working — Ubuntu upgraded to 26.04, which ships
  Python 3.14 as the system interpreter and removed 3.12. The venv's packages
  were intact on disk but its interpreter (`/usr/bin/python3.12`) was gone.
- **Fixed:** installed `uv` (userspace, `~/.local/bin`, no sudo), pulled a
  standalone CPython 3.12.14, recreated the venv against it, reinstalled from
  `requirements.txt`. Kept the old tree as `venv_broken_py312` until the new
  one was verified, then removed it.
- **Why 3.12 rather than migrating to 3.14:** `antspyx` (ANTs bindings used by
  Stage 05 registration) is a compiled extension and the one most likely to
  lack wheels for a brand-new interpreter. It installed cleanly on 3.12.14.
  Also keeps the host in step with the containers (3.12.11).
- **Verified:** 254 tests pass (59 skull-stripping, 35 preprocessing, 160
  backend). The single failure, `test_dataset_mapping`, is the pre-existing
  stale-expectation one that also fails on clean `main`.
- **Trap recorded in CLAUDE.md:** recreating the venv with `python3 -m venv`
  now silently builds it on 3.14 and antspyx will not install. Use
  `uv venv --python 3.12`.
- **Next step:** Phase F (`prepare_data.py`) — unblocked.

## 2026-09-07 (Task A follow-up — validation made to steer)

- **Reported:** cascade never switched tools. Two defect classes seen on real
  runs: MNI masks containing the patient's eyes, and one with holes inside the
  brain. Both were accepted.
- **Root causes (three, one unreported):**
  1. Review flags were log-only. Six runs, four flags raised, zero switches —
     validation detected the defects and shipped them.
  2. Hole gate at 20 ml. The defective mask had 10.71 ml across 11 cavities.
  3. **The cascade overwrote its own input.** Stage 05 passes the same
     directory as `subject_dir` and `output_dir`, so a candidate writing to
     `output_path` destroyed the volume the next candidate had to read. Never
     fired only because nothing was ever rejected; fixing (1) would have
     activated it. The existing retry test missed it by using two different
     directories, which production never does.
- **Fixes:** candidates write to a scratch dir, only the accepted one is
  promoted (`75fa7d4`); review flags now advance the cascade while untried
  tools remain, with the flagged mask held in reserve and `retry_on_review`
  as an escape hatch (`d842861`); hole gate 20 → 2 ml, calibrated on this
  project's masks — every correct one measures 0.00 ml (`d842861`).
- **Verified on real runs:** KA123 `mni_mask` 1838 ml (eyes) → `synthstrip`
  1332 ml clean; KA126 `mni_mask` 1827 ml (eyes + 10.71 ml holes) →
  `synthstrip` 1443 ml clean; KA126_4 `hdbet` 1297 ml accepted first try, no
  wasted attempts. Visual check confirms eyes and cavities are gone, i.e. the
  switch improved the mask rather than just the metric.
- **Findings worth carrying into the benchmark:**
  - `mni_mask` leaks onto eyes because registration is **Rigid (6 DOF)** — an
    atlas mask cannot fit a head it was not scaled to. Not a bug in the
    wrapper; a limit of the approach under the production registration.
  - **MNI as pseudo-GT is unreliable here.** Dice against
    `MNI152_T1_1mm_brain_mask`: correct HD-BET masks 0.84–0.87, the defective
    MNI mask 0.997. The spec's DSC-vs-atlas metric would rank the broken mask
    first. Needs rethinking before Phase 5.
  - An intensity-based leakage metric was prototyped and **rejected**: skull
    and scalp are tissue, not air, so "background inside the mask" does not
    separate the classes (defective 0.00 % air vs correct 2.47 %). The volume
    review flag covers the case for now.
- **Not done:** leakage metric (deferred — the volume flag already routes both
  reported cases, and calibration needs volumes that still have skull; every
  `preprocessed` file is already masked).
- **Blockers:** none.
- **Next step:** Task C (BrainMaGe) or F (`prepare_data.py`), per Kate.

## 2026-09-04 (Task B)
- **Done:** SynthStrip plugin: `synthstrip.py`, `STRIPPERS["synthstrip"]`, manifest `services/skull-stripping/synthstrip/manifest.yaml`. CLI: `mri_synthstrip -i -o -m`, optional `-b` from `tool_params.border`. `uses_gpu = False` (does not take the HD-BET pool). Official CLI also has `-g/--gpu` — not wired. Production config unchanged (hdbet → bet). Tests: `test_synthstrip.py` + cascade/GPU still green.
- **Install smoke:** On the **host venv** `pip install surfa` still fails without `sudo python3-dev`. Stage 05 does **not** use that venv: the web image already ships FreeSurfer `mri_synthstrip` (`FREESURFER_HOME=/opt/freesurfer`, weights in `models/synthstrip.1.pt`). `shutil.which` inside the container returns `/opt/freesurfer/bin/mri_synthstrip`. `--help` matches our flags (`-i/-o/-m/-b/-g/--no-csf/--model`). A synthetic 64³ cube crashed inside SynthStrip's CC step (not a brain); real Stage 05 T1 is the smoke. Host venv `is_available()` remains False.
- **Blockers:** none for Stage 05 (FreeSurfer CLI in the web image). Host venv still has no `mri_synthstrip`.
- **Next step:** Task D (MNI baseline, no extra install).

## 2026-09-04 (evening)
- **Done:** Task A — cascade + two-tier mask validation. `validation.py` (`mask_metrics`, catastrophe vs review flags, speckle drop `< 1 ml`). `build_cascade_order` / `try_stripper`; `process_subject_skull_stripping` retries only on hard fail; GPU slot released before CPU BET. Production config: `validation: {}`, cascade omitted (= hdbet → bet). Paper §3.6 EN+RU filled with the gate design (no numbers yet). Tests: 32 passed (`test_validation`, `test_cascade`, GPU pool/dispatch).
- **Not done:** none for Task A.
- **Blockers:** none.
- **Next step:** Task B SynthStrip.

## 2026-09-04
- **Done:** Reconnected onto `origin/main` as `feat/skull-stripping-research-v2` (docs cherry-picked; June BET-only plugin dropped). Executable plan rewritten: [2026-09-04-skull-stripping-research.md](../../docs/superpowers/plans/2026-09-04-skull-stripping-research.md). June plan frozen.
- **On main already:** SkullStripperBase, BET, HD-BET 2.x, dispatcher fallback, GPU pool, production `method: hdbet`.
- **Still to do:** cascade + mask validation, SynthStrip / BrainMaGe / MNI / SAM / DeepBET, benchmark harness, paper fill.
- **Blockers:** none for Task A (cascade). MosMed + clinical sets needed later for the full CSV.
- **Next step:** Task A — cascade + `validation.py` without breaking the GPU pool.

## 2026-06-16
- **Done:** Scaffolding created (research dir, requirements, paper skeleton dirs).
- **Blockers:** none.
- **Next step:** implement plugin architecture (SkullStripperBase + BET refactor).
