# Period Statistics for SIGMA — Design

**Date:** 2026-11-16
**Branch:** `feat/sigma-period-stats` (from `main`, not from the Kappa-quota branch)
**Type:** New read-only summary. No transmission yet.

## Intent

The MRI module's acceptance tests (ПМИ, table 22) require an aggregated
summary for a time period: how many MRI studies were processed, and how
those studies split by finding. SIGMA, the municipal decision platform,
uses two such summaries to show a trend. This task produces that summary.
It does not send it.

The owner confirmed the scope: the period summary only, and multiple
sclerosis counts as one class of study, not as a count of foci.

## Goals

- Given a period, return one JSON document a later sender can hand to a
  SIGMA test endpoint without reshaping it.
- The document states the study count and the class counts. Two periods
  with different studies produce two documents whose counts differ.
- A study with no finding is the class `normal`. A study is not inferred
  to be normal when its result is missing.

## Non-goals

- An HTTP client, credentials, or a live SIGMA endpoint. The acceptance
  step "transmit and see the trend" waits until that endpoint exists.
- A UI control to build or send the summary.
- Tuberculosis. Named in the ПМИ object description, not implemented here.
- Counts of individual MS foci, volumes, lobes, McDonald zones.
- Patient identifiers or an organization name inside the period
  aggregate. That document is counts only. The per-run event file
  carries both, as table 20 requires.

## Constraints

- ПМИ does not give a field schema for the aggregate. It points at the
  technical specification for the per-study event only. The document
  shape below is this design's contract.
- Class labels the pipeline already uses: `glioblastoma`,
  `multiple_sclerosis`. `normal` is new and exists only in this summary.
- Reports already on disk are the evidence of a finding. Stage 06 writes
  `*_volume_report.json` beside the GBM mask (`total_tumor.voxel_count`
  is tumor core, edema excluded). Stage 08 writes
  `*_lesion_stats_report.json` for MS (`lesion_count`, components under
  5 mm³ already dropped).
- One pipeline run holds many BIDS sessions. The ПМИ counts studies, not
  runs.
- Tests write under `tmp_path` only. No working config, no real DICOM.

## Approaches

1. **Compute on read from completed runs and the reports already written.**
   A function walks runs whose `completed_at` falls in the period, reads
   each session's report, and returns the document. Nothing new is stored.
   Re-running it for the same period is stable as long as the reports and
   the run rows are.
2. **Write an outcome row at the end of stage 08, then aggregate in SQL.**
   Survives a later deletion of the report files and gives each session
   its own timestamp. Requires a schema change, a write on the pipeline
   path, and a backfill of studies already processed. More than this task
   needs.
3. **Count pipeline runs by `pipeline_runs.lesion_type`.** Rejected. A run
   is a batch. The ПМИ series is "at least three studies", including a
   normal case, and a run has no normal outcome of its own.

Approach 1 is the one this design builds. Approach 2 stays available if
transmission later needs a durable ledger.

## What a study is

A study is one BIDS session (`sub-XXX_ses-YYY`) inside a run with
`status = completed` whose `completed_at` lies in the period.

The period is half-open, UTC: `start <= completed_at < end`. Both bounds
are required. Sessions do not have their own finish time; they take the
run's `completed_at`. DICOM study dates are not used: after
de-identification they are not a reliable clock.

Subject numbers restart in each Kappa dataset and, for a run with no
dataset, in `local:<lesion_type>`. The identity of a study is that scope
plus `sub-XXX_ses-YYY`. A requeue stays inside one scope, so it counts
once, from the run with the latest `completed_at`. The same BIDS label
in two datasets counts twice. A session that appears only in runs
outside the period is absent.

A session with no readable class report is left out. It is not `normal`.
That covers a run marked completed while a session never reached
segmentation.

## How the class is decided

The run's `lesion_type` selects which report to read.

- `glioblastoma`: `normal` when `total_tumor.voxel_count` is 0. Otherwise
  `glioblastoma`. Edema alone stays `normal`, matching the report's own
  definition of tumor core.
- `multiple_sclerosis`: `normal` when `lesion_count` is 0. Otherwise
  `multiple_sclerosis`. The 5 mm³ cutoff already applied inside
  `lesion_stats` is the cutoff here. Foci are not summed into the
  document.
- Any other `lesion_type`, or a report that cannot be parsed, or a report
  missing the field above: the session is left out, and the omission is
  logged with the run id and the session key. It does not fail the whole
  summary.

`studies_total` is the number of sessions that received a class.
`detected_total` is `studies_total` minus `normal`.

## The document

```json
{
  "period": {"start": "2026-11-01T00:00:00Z", "end": "2026-11-16T00:00:00Z"},
  "modality": "Mri",
  "studies_total": 4,
  "detected_total": 3,
  "by_class": {
    "normal": 1,
    "glioblastoma": 2,
    "multiple_sclerosis": 1
  }
}
```

`modality` is the literal `Mri` from the ПМИ event contract, so a later
sender does not invent it. `by_class` always contains all three keys,
with `0` where the period had none. `detected_total` is the "how many
findings" figure; `by_class` without `normal` is the split by type.

No patient id, no organization, no volumes, no list of sessions.

## How it is reached

`GET /api/sigma/period-stats?start=...&end=...`

`start` and `end` are ISO 8601. A missing or unordered bound is `400`.
The response is the document. Nothing is written. No Kappa session is
required: the figures come from local runs.

The route is the "initiate formation" step of table 22. Transmission
stays a separate call that does not exist yet. The route does not write
a file.

## The per-run event file

A successful pipeline run writes one event per classified session to
`{root_output_dir}/logs/sigma_events.json`. There is no second summary
file in that folder: the period route above is the aggregate. A failed
or dry run does not write the file. A failure to write it does not fail
the run.

```json
[
  {
    "Id": "sub-001",
    "modality": "Mri",
    "timestamp": "2026-11-16T09:40:00Z",
    "source": "Federal Center of Neurosurgery",
    "description": "glioblastoma"
  }
]
```

`Id` is the depersonalized patient, `sub-XXX`. Two sessions of one
patient are two events with the same `Id`. `timestamp` is the run's
finish, ISO 8601 UTC. `description` is `normal`, `glioblastoma`, or
`multiple sclerosis`. `source` is the organization of the Kappa login
that started the run:

| Login | `source` |
|---|---|
| `e.roppert` | Federal Center of Neurosurgery |
| `test_med1` | International Tomography Center |

Kappa's own organization name is not used. A run with no login, or a
login that is not in this table, still gets an event, with `source`
empty. A CLI run has no login. The login travels in the runtime config
as `general.kappa_user_name`, set when the run is started or resumed. A
resume records who resumed it.

## Testing

Temp database and temp report trees, no network:

- three sessions in one completed run (normal GBM, GBM with tumor core,
  MS with `lesion_count > 0`) produce the counts above;
- an MS report with `lesion_count == 0` counts as `normal`;
- a GBM report whose only non-zero class is edema counts as `normal`;
- a second run of the same session key inside the period does not raise
  the total; the later run's class wins;
- a completed run outside the period, a non-completed run inside it, and
  a session with no report do not count;
- two different periods over a fixture with a different mix return
  different `by_class` values;
- the endpoint returns that document, and `400` when `end` is not after
  `start`.

## Left to the builder

- Module and function names, provided the endpoint and the document keys
  stay as written.
- How session keys are parsed, as long as `sub-XXX_ses-YYY` from the
  existing segmentation layout is the identity.
- Log wording for a skipped session.
