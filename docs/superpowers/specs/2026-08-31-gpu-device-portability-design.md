# GPU Device Portability for Stage 05 Skull Stripping — Design

**Date:** 2026-08-31
**Branch:** `fix/gpu-device-portability` (from `main`)
**Type:** Bugfix + portability
**Related:** KI-056 (second workstation), ROADMAP Этап 7 (Resource-агент / GPU pool)

## Problem

Run `demo_workspace/31_08_1557` hung at Stage 05, step 4/4 (skull stripping),
and froze the entire desktop, requiring a hard reboot.

Root cause (from `logs/stages/stage_05_preprocessing.log:130-144`): Stage 05
ran in parallel mode with 2 workers. Both workers launched HD-BET on
`device=cuda` within 4 seconds of each other. Two concurrent nnU-Net
inferences with test-time augmentation (TTA) landed on **GPU 0**, which on
this workstation also drives the display (Xorg + gnome-shell, per
`nvidia-smi`). VRAM/driver contention deadlocked the GPU → graphics froze →
reboot. Skull stripping never finished, so the run appeared to "hang".

Contributing factors, each independently fixable:
- **No GPU serialization** across parallel Stage-05 workers. Grep for
  `lock|semaphore` in `scripts/preprocessing_steps/skull_stripping/` finds none.
- **Uses the display GPU.** `configs/preprocessing_config.yaml` sets
  `method: hdbet` with no `device`, so `hdbet.py:66` defaults to `"cuda"` = GPU 0.
- **TTA enabled on GPU** (`hdbet.py:78` only disables it for CPU) — heaviest mode.
- **`docker-compose.yml:46` reserves `count: 1`** GPU, so the container grabs
  a single (display) GPU rather than seeing the idle second card.

## Goals

- Stage 05 skull stripping must run correctly and **never freeze the display**,
  on machines with 1 GPU (GPU is also the display) or ≥2 GPUs (one may be the
  display GPU), with no per-machine code changes required.
- Behavior configurable per machine when the operator wants explicit control.
- The concurrency mechanism must generalize from 1 to K GPUs "for free" so a
  future 4-GPU host can run K concurrent HD-BET without new machinery.

## Non-Goals

- No GPU load balancer, scheduler, or cross-stage GPU arbitration. Segmentation
  (Stage 06) is a separate microservice with its own GPU pool and is out of scope.
- No change to the segmentation services.
- No attempt to reliably detect "which GPU drives the display" via display-server
  introspection (unreliable from inside a container). We use free-VRAM as a proxy
  and an explicit exclude list.

## Design (Вариант C+)

Five parts. The GPU-slot pool (part 3) is the core; it mirrors the proven
pattern already used by the segmentation services in
`services/common/service_base.py:85-89` (a blocking queue of GPU ids).

### 1. Compose — expose all GPUs
Change the web service GPU reservation from `count: 1` to `count: all` in
`docker-compose.yml` so the container sees every card on the host. Machine-
independent: on a 1-GPU host this is just the one card. GPU **selection** then
happens in the app, not in compose, so no per-machine compose override is needed.

### 2. Device selection (`device: "auto"`)
New config value `device: "auto"` (default) for the skull-stripping step.
Resolution at Stage-05 startup, in the parent process:
- Enumerate visible CUDA GPUs and their free VRAM (via `pynvml`, already a
  dependency of `services/common/gpu_monitor.py`; fall back to `torch.cuda` /
  `nvidia-smi` query if pynvml unavailable).
- Build the **usable GPU set**: all GPUs whose free VRAM ≥ a threshold, minus
  any in the configured `exclude_gpus` list.
- Display-GPU avoidance: when the usable set has more than one card, the pool
  simply prefers the freer cards; a machine can also list the display GPU in
  `exclude_gpus` to hard-exclude it.
- If no CUDA GPU is usable → fall back to CPU (HD-BET runs, slower), log a warning.
Explicit overrides still win: `device: "cuda:1"` / `device: "cpu"` /
`device: "cuda"` bypass auto-resolution.

### 3. Cross-process GPU-slot pool
The resolved usable GPU set seeds a **`multiprocessing.Manager().Queue`** of
`gpu_id`s, created in the Stage-05 parent and passed to worker processes
(`ProcessPoolExecutor`). The single-lock and the K-GPU cases are the same code:

```
gpu_id = gpu_pool.get()          # blocks until a slot is free
try:
    run HD-BET with device=f"cuda:{gpu_id}"
finally:
    gpu_pool.put(gpu_id)         # release slot
```

- 1 usable GPU  → pool size 1 → HD-BET serialized (today's freeze impossible).
- N usable GPUs → pool size N → up to N concurrent HD-BET, each pinned to its card.
A CPU fallback path uses a pool of size 1 (or a small fixed cap) with a sentinel
so the acquire/release shape is unchanged.

The pool wraps **only** the GPU skull-stripping call, not the whole subject —
reorient / bias / registration stay CPU-parallel across all workers as today.

### 4. Config override (per-machine)
`configs/preprocessing_config.yaml` skull_stripping params gain:
- `device: "auto"` — `auto` | `cuda` | `cuda:N` | `cpu`.
- `exclude_gpus: []` — list of GPU indices never to use (e.g. `[0]` on a
  workstation to protect the display GPU).
- `gpu_pool_size: null` — optional hard cap on concurrent GPU jobs; `null` =
  size of the usable GPU set.
Machines that want fixed behavior set these in a machine-local config; the
default (`auto`, no excludes) is safe and portable out of the box.

### 5. TTA made explicit
Add `disable_tta` as an explicit skull-stripping param (default: current
behavior — TTA on for GPU, off for CPU). With serialization in place, TTA on a
single card is safe; exposing it lets batch runs trade quality for speed.

## Per-machine behavior matrix

| Host | usable set (default auto) | pool size | outcome |
|------|---------------------------|-----------|---------|
| 1 GPU (display) | `[0]` | 1 | serialized on GPU 0; no concurrent collision → no freeze |
| 2 GPU (0=display,1=idle) | `[0,1]`, or `[1]` if `exclude_gpus:[0]` | 2 (or 1) | prefers idle card; optional hard-exclude of display GPU |
| 4 GPU compute | `[0,1,2,3]` | 4 | 4 concurrent HD-BET, one per card |
| no CUDA | `[]` | 1 (CPU) | CPU fallback, slower, still finishes |

## Interfaces / files touched

- `scripts/preprocessing_steps/skull_stripping/hdbet.py` — accept a resolved
  `device` (`cuda:N` / `cpu`) and explicit `disable_tta`.
- `scripts/preprocessing_steps/skull_stripping/dispatcher.py` — GPU device
  resolution + pool acquire/release around the strip call.
- New small helper for GPU enumeration/selection (e.g.
  `skull_stripping/gpu_select.py`), or reuse `services/common/gpu_monitor.py`
  querying utilities.
- `scripts/05_preprocessing.py` — create the `Manager().Queue` pool, pass to workers.
- `configs/preprocessing_config.yaml` (+ `_ms60.yaml`) — new params.
- `docker-compose.yml` — `count: all`.

## Testing

- Unit: device resolution (`auto` with 0/1/N GPUs, excludes, CPU fallback,
  explicit overrides) with a mocked GPU enumerator.
- Unit: pool acquire/release invariants (never exceeds pool size; releases on
  exception) using a `multiprocessing`/threaded harness.
- Regression: existing `skull_stripping/tests/` (14 tests) stay green.
- Manual verification on this 2-GPU host: a multi-subject Stage-05 run with
  `exclude_gpus: [0]` completes without touching the display GPU (watch
  `nvidia-smi`), and with default `auto` does not freeze the desktop.

## Risks

- pynvml not present in the web image → mitigate with a `torch.cuda` /
  `nvidia-smi` fallback in the enumerator; if all fail, degrade to CPU.
- `Manager().Queue` overhead is negligible at these job counts.
- `count: all` exposing GPUs the operator wanted reserved → `exclude_gpus`
  covers it at the app layer.
