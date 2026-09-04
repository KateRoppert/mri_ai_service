# Pipeline Stop & Resume — Design

## Context

A pipeline run currently has exactly two ways to end: it finishes, or it fails. There
is no way for an operator to end one deliberately. The only kill path in the codebase
is the timeout branch of `run_pipeline_background`, which calls `_kill_process_tree`
from inside the same function that started the process — nothing outside can reach it,
because the `Popen` object lives only in that function's local scope.

That gap matters more now than it used to. Runs are long (a full batch is tens of
minutes to hours), they hold the machine's single GPU for their whole duration, and
since the barguzin deployment (`docker-compose.barguzin.yml`) a second person can start
one remotely. An operator who realises at patient 3 of 40 that they picked the wrong
input folder currently has no recourse short of restarting the container.

The owner asked for a stop button that preserves results already produced. Working
through the cases during design surfaced a second requirement — resuming a stopped run
rather than recomputing it — and a latent correctness bug that stopping would otherwise
make much easier to hit.

## Cases the design serves

Five situations motivate stopping, and they split into two groups that need different
treatment:

| Case | Why stop | Resume appropriate? |
|---|---|---|
| GPU needed for something urgent | External | Yes — inputs and settings unchanged |
| Machine going down for reboot | External | Yes — same |
| Run appears hung | External | Yes, once integrity is verifiable |
| Wrong input folder or lesion type | The run itself is wrong | **No** — start over |
| Poor quality noticed on first patients | The run itself is wrong | **No** — settings will change first |

The distinction drives the resume design below. In the first group the run is fine and
the environment is not; continuing is exactly right. In the second the operator stops
*because something about the run is wrong*, and will typically change configuration
before running again — so resuming silently produces a dataset whose first patients were
processed one way and the rest another, with nothing recording the difference.

## Design

### Stop semantics: immediate

Stopping kills the process group immediately rather than draining the current stage or
the current patient. Chosen deliberately over a graceful variant: draining requires the
orchestrator to poll a stop flag at stage or patient boundaries, and the wait is
unbounded from the operator's point of view — a segmentation stage mid-patient can take
minutes, a full stage far longer. An operator who has decided to stop wants the GPU back
now.

The cost of immediacy is that files being written at that moment are left truncated.
That is addressed by integrity checking below rather than by trying to avoid it.

### Process registry

`PipelineManager` gains an in-memory registry mapping `run_id` to the running `Popen`.
`start_pipeline` registers, `run_pipeline_background` deregisters in a `finally` so that
normal completion, failure and timeout all clean up.

An in-memory registry is sufficient here and a database PID column is not, because the
pipeline is a child process of the backend inside the same container: restarting the
backend kills the run too. There is no scenario where a run outlives the registry that
holds it.

The one consequence to handle: after a backend restart the database may still show a run
as `running` while no process exists. Stop must not lie about this. When a stop request
finds no registry entry for a run the database calls active, it marks the run `stopped`
anyway (with a note that the process was already gone) rather than returning an error —
the process is provably dead, so the database was wrong and stop fixes it.

Killing reuses the existing `_kill_process_tree`, which signals the whole process group.
This is why `start_pipeline` passes `start_new_session=True`; killing only the
orchestrator would orphan the running stage subprocess (KI-052).

### Run state

A new terminal status `stopped`, alongside `completed` and `failed`. It is deliberately
not `failed`: a stopped run is not a defect, and conflating the two would corrupt any
reading of the history — how often runs fail is a health signal, how often operators
stop them is not.

Two columns on `pipeline_runs`:

- `stopped_at_stage INTEGER` — the stage in flight when the kill landed, which the
  resume logic and the UI both need.
- `stopped_by VARCHAR` — the Kappa user who pressed the button. Cheap to record now that
  more than one person can drive the service, and unrecoverable after the fact.

### Integrity checking

The bug stopping would otherwise expose: `check_subject_processed` (Stage 05) and its
equivalents in Stages 04 and 06 decide a unit of work is done **by the existence of its
output file**. A truncated file passes that test. With `skip_existing` on — the default,
and what resume relies on — a rerun would skip the truncated patient as complete and
carry the damaged volume forward into segmentation and the clinical report.

Existence is replaced by completeness. For a `.nii.gz`, gzip stores the uncompressed
size in the last four bytes (`ISIZE`), and the NIfTI header gives shape and dtype;
their product is the expected size. Comparing the two detects truncation without
decompressing anything.

Measured on a representative 5.1 MB file (29 MB uncompressed):

| Check | Cost per file |
|---|---|
| `exists()` (current) | 0.003 ms |
| NIfTI header via nibabel | 1.57 ms |
| Full data read | 56 ms |
| **Header + gzip trailer** | **0.076 ms** |

Two limits, both acceptable and both worth stating in the code:

- `ISIZE` is stored modulo 4 GiB. Our volumes are ~30 MB; a file large enough to alias
  would break the check, so the helper documents the ceiling.
- The trailer confirms length, not content — corruption that preserves length (a bad
  disk sector) is not caught. Detecting that requires a full CRC pass at 56 ms per file,
  and it is not the failure mode stopping produces. Truncation is, and truncation is
  caught reliably.

The helper lives in `utils/nifti_integrity.py` — used by pipeline stages, which cannot
import from `backend/`. Non-gzip `.nii` falls back to comparing the file size against
the header's expectation. Anything unreadable counts as incomplete: the cost of
recomputing a good file is minutes, the cost of trusting a bad one is a wrong clinical
report.

### Resume

Resume is the existing `requeue` endpoint, which already reruns on the same
input/output paths, relies on `skip_existing`, links runs via `parent_run_id`, and
refuses to start when another run is active on the same output path. It currently
rejects only `PENDING`/`RUNNING`, so `stopped` passes without change.

What must be added is protection against the second group of cases. Today
`create_runtime_config` re-reads `pipeline_config.yaml` on every start, and
`cleanup_runtime_config` deletes the snapshot afterwards — so a resume silently adopts
whatever the configuration says at that moment. Combined with the fact that the active
segmentation model lives outside that file entirely (in each service's
`server_config.yaml`, reported via `/api/segmentation/active-model`), a resume after a
model switch can mix two label schemes in one output directory.

Three additions:

1. **Keep the snapshot.** A stopped run's runtime config is retained rather than
   deleted, and its path recorded in the existing `config_path` column.
2. **Compare before resuming.** `POST /api/pipeline-runs/{run_id}/requeue` compares the
   snapshot against the current config and the current active model, returning a list of
   differences in human terms ("skull stripping: bet → hdbet").
3. **Let the operator choose.** With differences present, the UI shows them and offers
   two paths: resume on the saved settings (`use_snapshot: true`, passing the retained
   config to the orchestrator) or start fresh. With no differences, resume proceeds
   without a prompt.

This keeps the common cases frictionless and makes the dangerous ones visible at the
moment they matter, rather than discoverable later in a report that looks fine.

### API

```
POST /api/pipeline-runs/{run_id}/stop
  -> 200 {run_id, status: "stopped", stopped_at_stage, patients_completed}
  -> 404 unknown run
  -> 409 run already in a terminal state

POST /api/pipeline-runs/{run_id}/requeue        (existing, extended)
  body: {use_snapshot?: bool}
  -> 200 as today
  -> 409 {differences: [{setting, was, now}]} when settings changed and
         use_snapshot was not specified
```

Stop emits a WebSocket event on the run's channel so every open tab reflects the new
state without polling.

### Interface

`ProgressMonitor` gains a **Stop** button beside the progress display, shown only while
a run is active. It opens a confirmation dialog stating what is done and what is lost —
patients completed, stage about to be interrupted — because the action is irreversible
and may land an hour into a run.

In the run history a stopped run reads as stopped, not failed, and carries a **Resume**
button. Pressing it either resumes directly or, when the backend reports differences,
shows them and asks which way to proceed.

## Testing

Integrity helper: a complete file passes; a file truncated mid-data fails; an empty file
fails; a `.nii` (non-gzip) file is handled; a file whose header is unreadable fails
closed.

Stop: the process group is gone after a stop (not merely the orchestrator); status,
stage and user are recorded; stopping a run whose process has already exited still marks
it stopped rather than erroring; stopping an already-terminal run returns 409.

Resume: identical settings resume without a prompt; a changed skull stripper is reported
as a difference; a changed segmentation model is reported; `use_snapshot` passes the
retained config to the orchestrator.

Integration: a run stopped mid-stage and resumed produces the same set of outputs as an
uninterrupted run, and patients whose files were truncated at the moment of the kill are
recomputed rather than skipped.

## Out of scope

Graceful (drain-then-stop) shutdown — rejected above in favour of immediate. Pausing a
run without ending it: a paused run holds the GPU, which defeats the main reason for
stopping. Stopping an individual patient within a run. Resuming a run whose input
directory has changed since it started — the design does not detect that, and the
existing `skip_existing` behaviour applies.

## Related

- KI-052 — orphaned stage subprocesses, why `_kill_process_tree` kills the group
- KI-057 — hardcoded Kappa `dataset_id`, relevant because resume re-enters the upload path
- `docs/superpowers/specs/2026-08-14-requeue-progress-linking-design.md` — `parent_run_id` semantics
