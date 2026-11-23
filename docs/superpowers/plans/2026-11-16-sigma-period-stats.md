# Period Statistics for SIGMA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return a period summary of completed MRI studies by class (`normal`, `glioblastoma`, `multiple_sclerosis`) that a later sender can hand to SIGMA unchanged.

**Architecture:** `build_period_stats` reads completed `pipeline_runs` whose `completed_at` falls in a half-open UTC period, classifies each BIDS session from the report the run's `lesion_type` already writes, and returns the document from the spec. `GET /api/sigma/period-stats` parses the bounds and calls that function. Nothing is stored and nothing is sent.

**Tech Stack:** Python, FastAPI, SQLAlchemy, pytest, the existing `PipelineRun` table and on-disk JSON reports.

**Spec:** `docs/superpowers/specs/2026-11-16-sigma-period-stats-design.md`

## Global Constraints

- Document keys are `period.start`, `period.end`, `modality` = `"Mri"`, `studies_total`, `detected_total`, `by_class.normal`, `by_class.glioblastoma`, `by_class.multiple_sclerosis`. All three class keys are always present.
- Period is half-open: `start <= completed_at < end`. `end` not after `start`, or a missing/unparseable bound, is HTTP 400.
- A study is `sub-XXX_ses-YYY`. The same key in the period counts once; the later `completed_at` wins.
- GBM class comes from `*_volume_report.json` `total_tumor.voxel_count` (0 is `normal`; edema is not tumor core). MS class comes from `*_lesion_stats_report.json` `lesion_count` (0 is `normal`). Any other lesion type, a missing report, or an unreadable report is omitted and logged, not counted as `normal`.
- Tests use `tmp_path` only. No working config, no real DICOM, no network.
- One run has one `lesion_type`. The spec's example document (1 normal, 2 glioblastoma, 1 multiple sclerosis) is two runs in one period, not one run of mixed types.

## Review Focus

- A completed run with `completed_at` NULL cannot be placed in a period: it must not count and must not raise.
- `output_path` with no `segmentation` directory must not raise; its sessions are absent.
- Two reports for one session that name different classes must not be averaged or double-counted; the session is omitted.
- A naive `completed_at` from SQLite and an aware ISO bound must compare as UTC, not throw.
- Echoed `period.start` / `period.end` use a `Z` suffix, including when the query omitted the zone (treated as UTC).

---

### Task 1: Classify sessions and build the document

**Files:**
- Create: `backend/sigma_period_stats.py`
- Test: `backend/test_sigma_period_stats.py`

**Interfaces:**
- Consumes: `PipelineRun` (`run_id`, `status`, `completed_at`, `output_path`, `lesion_type`).
- Produces: `build_period_stats(db, start: datetime, end: datetime) -> dict` with the document keys above. `start` and `end` are timezone-aware UTC. The caller has already rejected a bad period.

- [x] **Step 1: Write the failing tests** named in the spec's Testing section, plus the five Review Focus cases, against a private sqlite file and report trees under `tmp_path`.

- [x] **Step 2: Run** `python -m pytest backend/test_sigma_period_stats.py -q` and confirm the import of `build_period_stats` fails.

- [x] **Step 3: Implement** `build_period_stats` in `backend/sigma_period_stats.py`. GBM glob `*_volume_report.json`. MS glob `*_lesion_stats_report.json`. Session key from directory parts `sub-*` / `ses-*`, then from the filename. Conflicting classes for one key in one run: log and omit. Later run overwrites the same key.

- [x] **Step 4: Re-run the same pytest command.** Expected: the unit tests pass. The endpoint tests stay for Task 2.

### Task 2: HTTP boundary

**Files:**
- Modify: `backend/app.py` (register the route before the frontend static mount)
- Test: `backend/test_sigma_period_stats.py`

**Interfaces:**
- Consumes: `build_period_stats`.
- Produces: `GET /api/sigma/period-stats?start=&end=` returning the document. No Kappa session.

- [x] **Step 1: Add failing endpoint tests:** the example document through `TestClient`; `400` when `end` equals `start`; `400` when `start` is absent; `400` when `start` is not a date.

- [x] **Step 2: Run those tests.** Expected: 404 or missing route.

- [x] **Step 3: Add the route.** Parse ISO 8601 (`Z` or an offset; no zone means UTC). Compare the instants, not the strings.

- [x] **Step 4: Run** `python -m pytest backend/test_sigma_period_stats.py -q`. Expected: all pass.

### Task 3: Per-run events instead of a per-run summary file

The period route stays. A finished run writes `{root_output_dir}/logs/sigma_events.json` only: one object per classified session, fields `Id`, `modality`, `timestamp`, `source`, `description`. `source` comes from the Kappa login on the runtime config (`e.roppert` → Federal Center of Neurosurgery, `test_med1` → International Tomography Center). No login leaves `source` empty.

- [x] **Step 1: Replace** the per-run summary writer with `write_run_events`. The orchestrator calls it and does not write `sigma_period_stats.json`.
- [x] **Step 2: Thread** `kappa_user_name` from the start and requeue paths through `run_pipeline_background` into the runtime YAML.
