"""
Abstract base class for MAS inference services.

Concrete services (gbm-seg, ms-seg, mets-seg, mgmt-classify) subclass
ServiceBase and implement two abstract methods:

    async def load_model(self) -> None
    async def run_inference(self, payload: dict, gpu_id: int) -> dict

Everything else — HTTP routes, job queue, GPU pool, GPUMonitor lifecycle,
manifest loading, logging — is provided by the base class.

This guarantees that all services in the system follow the same contract
described in docs/SPEC.md §3, and that adding a new service requires
implementing only the model-specific bits.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import traceback
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from quart import Quart, jsonify, request

from common.contracts import Job, JobStatus
from common.gpu_monitor import GPUMonitor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)


# ---------------------------------------------------------------------------
# Service base
# ---------------------------------------------------------------------------

class ServiceBase(ABC):
    """
    Base class for HTTP-served inference services in the MAS.

    Subclasses must set the following class attributes (before super().__init__):
        service_id: str          — short identifier, e.g. "gbm-seg"
        service_type: str        — "segmentation" | "classification"

    Optional class attributes with defaults:
        manifest_path: str       — path to manifest.yaml (default "manifest.yaml")
        max_content_length: int  — max upload size in bytes (default 1 GB)

    Subclasses MUST implement:
        load_model()
        run_inference(payload, gpu_id)
    """

    service_id: str = "unnamed-service"
    service_type: str = "segmentation"
    manifest_path: str = "manifest.yaml"
    max_content_length: int = 1024 * 1024 * 1024  # 1 GB

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, gpu_ids: list[int] | None = None) -> None:
        """
        Args:
            gpu_ids: list of GPU indices available for inference.
                     Defaults to [0] if not given.
        """
        self.log = logging.getLogger(self.service_id)

        # Manifest is the static description of what this service can do
        self.manifest = self._load_manifest()

        # GPU pool — same blocking-queue pattern as the original simple_server
        self.gpu_ids = gpu_ids if gpu_ids else [0]
        self._gpu_pool: queue.Queue[int] = queue.Queue()
        for gpu_id in self.gpu_ids:
            self._gpu_pool.put(gpu_id)
            self.log.info("registered gpu %d in pool", gpu_id)

        # Job bookkeeping
        self._jobs: dict[str, Job] = {}
        self._job_queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._model_ready = False

        # Quart app
        self.app = Quart(self.service_id)
        self.app.config["MAX_CONTENT_LENGTH"] = self.max_content_length
        self._register_routes()

    # -----------------------------------------------------------------------
    # Abstract methods — implemented by concrete services
    # -----------------------------------------------------------------------

    @abstractmethod
    async def load_model(self) -> None:
        """
        Called once at server startup. Load model weights into memory/GPU.

        Should raise on failure — the base class catches and logs, then
        keeps /health returning 'loading' so the orchestrator knows to wait.
        """
        raise NotImplementedError

    @abstractmethod
    async def run_inference(self, payload: dict[str, Any], gpu_id: int) -> dict[str, Any]:
        """
        Called once per /predict request. Run inference and return a result dict.

        Args:
            payload: the JSON body from POST /predict, as a dict.
                     Expected keys per docs/SPEC.md §3.1:
                         case_id, input_dir, output_dir, lesion_type, options.
            gpu_id:  GPU index to use. Already allocated by base class.

        Returns:
            A dict with at minimum these keys:
                mask_path: str           — absolute path to predicted NIfTI mask
                output_classes: dict     — {class_index: class_name}
            And optionally:
                lesion_type: str
                model: dict              — framework, task name, folds, etc.

        Base class will add 'metrics' (GPU utilization, memory, temperature,
        inference_time_sec) automatically after this method returns.
        """
        raise NotImplementedError

    # -----------------------------------------------------------------------
    # Manifest loading
    # -----------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, Any]:
        """Load static service manifest from yaml. Returns minimal dict if missing."""
        path = Path(self.manifest_path)
        if not path.exists():
            self.log.warning("manifest not found at %s, using minimal stub", path)
            return {
                "service": {
                    "id": self.service_id,
                    "type": self.service_type,
                    "version": "0.0.0",
                }
            }
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    # -----------------------------------------------------------------------
    # HTTP routes
    # -----------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self.app

        @app.before_serving
        async def _on_startup() -> None:
            """Load model once, then start the queue worker."""
            self.log.info("[%s] loading model …", self.service_id)
            try:
                await self.load_model()
                self._model_ready = True
                self.log.info("[%s] model ready", self.service_id)
            except Exception:
                self.log.exception("[%s] model load failed", self.service_id)
                self._model_ready = False
            self._install_signal_probe()
            self._worker_task = asyncio.create_task(self._worker_loop())

        @app.after_serving
        async def _on_shutdown() -> None:
            """
            Normal on `docker stop`. Unexpected during a job — it used to mean
            a forked child's SIGTERM had leaked through asyncio's inherited
            wakeup pipe; see the spawn note in serve().
            """
            self.log.warning("[%s] pid=%d shutting down (queue=%d)",
                             self.service_id, os.getpid(), self._job_queue.qsize())

        @app.route("/health", methods=["GET"])
        async def health() -> Any:
            return jsonify(
                {
                    "service_id": self.service_id,
                    "status": "ready" if self._model_ready else "loading",
                    "queue_size": self._job_queue.qsize(),
                    "gpus_in_pool": self._gpu_pool.qsize(),
                    # What the active model actually emits. Consumers (the
                    # viewer legend, reports) label classes from this instead
                    # of assuming a fixed scheme, so swapping models cannot
                    # leave the UI describing the previous one.
                    "active_model": self.describe_active_model(),
                }
            )

        @app.route("/manifest", methods=["GET"])
        async def manifest() -> Any:
            return jsonify(self.manifest)

        @app.route("/predict", methods=["POST"])
        async def predict() -> Any:
            if not self._model_ready:
                return jsonify({"error": "model not ready"}), 503
            payload = await request.get_json(force=True, silent=True) or {}

            # Basic payload validation — extended checks live in subclasses
            missing = [k for k in ("case_id", "input_dir", "output_dir") if k not in payload]
            if missing:
                return jsonify({"error": f"missing keys: {missing}"}), 400

            job = Job(payload=payload)
            self._jobs[job.job_id] = job
            await self._job_queue.put(job.job_id)
            self.log.info("[%s] queued job %s for case %s",
                          self.service_id, job.job_id, payload.get("case_id"))
            return jsonify({"job_id": job.job_id, "status": job.status.value}), 202

        @app.route("/predict/<job_id>/status", methods=["GET"])
        async def status(job_id: str) -> Any:
            job = self._jobs.get(job_id)
            if not job:
                return jsonify({"error": "unknown job"}), 404
            return jsonify(job.to_status_dict())

        @app.route("/predict/<job_id>/result", methods=["GET"])
        async def result(job_id: str) -> Any:
            job = self._jobs.get(job_id)
            if not job:
                return jsonify({"error": "unknown job"}), 404
            if job.status != JobStatus.SUCCEEDED:
                # 409: client should poll /status until succeeded
                return jsonify(job.to_status_dict()), 409
            return jsonify(job.to_result_dict())

    def describe_active_model(self) -> dict[str, Any]:
        """
        Describe the model currently loaded, for consumers that must label
        output classes correctly (viewer legend, reports). Subclasses that
        can switch between models with different class semantics override
        this. Default: nothing model-specific to report.
        """
        return {}

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def _install_signal_probe(self) -> None:
        """
        Wrap whatever signal handlers are already installed (Quart/Hypercorn
        set theirs up before serving) so an incoming signal gets logged
        before the original handler runs. Purely diagnostic: the original
        behaviour is preserved by delegating to the saved handler.

        Answers: is this process being told to stop from outside, and by
        which signal — versus tearing itself down for an internal reason.
        """
        import signal

        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT):
            try:
                original = signal.getsignal(signum)
            except (ValueError, OSError):
                continue

            def _probe(sig: int, frame: Any, _original: Any = original) -> Any:
                self.log.error("[%s] pid=%d RECEIVED SIGNAL %s (handler=%r)",
                               self.service_id, os.getpid(),
                               signal.Signals(sig).name, _original)
                if callable(_original):
                    return _original(sig, frame)
                return None

            try:
                signal.signal(signum, _probe)
            except (ValueError, OSError):
                # Not settable from this thread/platform — skip, diagnostic only.
                continue

        self.log.info("[%s] pid=%d signal probe installed", self.service_id, os.getpid())

    # -----------------------------------------------------------------------
    # Background worker
    # -----------------------------------------------------------------------

    async def _worker_loop(self) -> None:
        """
        Single-consumer loop. Pulls job_ids from the queue, executes them
        one at a time per GPU slot, manages GPUMonitor and GPU pool.
        """
        # Recorded once: every later log line compares against this so a
        # forked copy of this process announces itself instead of masquerading
        # as the real worker.
        worker_pid = os.getpid()
        self.log.info("[%s] worker started (pid=%d)", self.service_id, worker_pid)
        while True:
            job_id = await self._job_queue.get()
            job = self._jobs.get(job_id)
            if not job:
                self.log.warning("worker pulled unknown job_id %s", job_id)
                continue

            # Acquire a GPU (blocking call in a thread, to keep event loop alive)
            loop = asyncio.get_running_loop()
            iter_marker = uuid.uuid4().hex[:8]
            gpu_id = await loop.run_in_executor(None, self._gpu_pool.get)
            job.mark_running(gpu_id)
            self.log.info("[%s] iter=%s pid=%d tid=%s acquired gpu %d for job %s",
                          self.service_id, iter_marker, os.getpid(),
                          threading.get_ident(), gpu_id, job_id)

            monitor = GPUMonitor(gpu_id)
            monitor.start()
            inference_start = loop.time()
            try:
                result = await self.run_inference(job.payload, gpu_id)
                inference_time = loop.time() - inference_start

                # Attach metrics + timing collected by the base class
                metrics = monitor.stop()
                metrics["inference_time_sec"] = round(inference_time, 2)
                result.setdefault("metrics", {}).update(metrics)

                job.mark_succeeded(result)
                self.log.info("[%s] iter=%s job %s succeeded in %.1fs",
                              self.service_id, iter_marker, job_id, inference_time)
            except asyncio.CancelledError:
                # Not caught by `except Exception` (CancelledError is a
                # BaseException since 3.8). A cancellation here means this
                # event loop is being torn down mid-inference — which is
                # exactly what a forked copy of this process would do on
                # exit, and would explain a "released" line appearing while
                # the real inference is still running.
                monitor.stop()
                self.log.error("[%s] iter=%s pid=%d CANCELLED mid-inference for job %s",
                               self.service_id, iter_marker, os.getpid(), job_id)
                raise
            except Exception as e:
                monitor.stop()
                tb = traceback.format_exc()
                self.log.error("[%s] iter=%s pid=%d job %s failed: %s\n%s",
                               self.service_id, iter_marker, os.getpid(), job_id, e, tb)
                job.mark_failed(f"{type(e).__name__}: {e}")
            finally:
                self._gpu_pool.put(gpu_id)
                self.log.info("[%s] iter=%s pid=%d tid=%s released gpu %d back to pool"
                              " (worker_pid=%d%s)",
                              self.service_id, iter_marker, os.getpid(),
                              threading.get_ident(), gpu_id, worker_pid,
                              "" if os.getpid() == worker_pid else " ← FORKED COPY!")

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def serve(self,
              host: str = "0.0.0.0",
              port: int = 5000,
              debug: bool = False) -> None:
        """
        Start the Quart server. Blocks until shutdown.

        Host/port/debug can be overridden via env vars SERVICE_HOST,
        SERVICE_PORT, SERVICE_DEBUG — useful in docker-compose.
        """
        host = os.environ.get("SERVICE_HOST", host)
        port = int(os.environ.get("SERVICE_PORT", port))
        debug = os.environ.get("SERVICE_DEBUG", str(debug)).lower() == "true"

        # Force "spawn" for every child process this service (or any library
        # it calls) creates. The default on Linux is "fork", and a forked
        # child inherits asyncio's signal-wakeup pipe. When multiprocessing
        # later terminates such a child (Pool teardown sends SIGTERM), the
        # child writes the signal number into that inherited pipe — which is
        # the very pipe THIS process's event loop reads. The parent then
        # believes it received SIGTERM and shuts the app down cleanly
        # mid-inference: after_serving fires, worker tasks are cancelled,
        # exit code 0, no traceback anywhere. "spawn" execs a fresh
        # interpreter instead, so nothing inherits that pipe.
        import multiprocessing
        try:
            multiprocessing.set_start_method("spawn", force=True)
            self.log.info("[%s] multiprocessing start method: %s",
                          self.service_id, multiprocessing.get_start_method())
        except RuntimeError as e:
            self.log.warning("[%s] could not set spawn start method: %s",
                             self.service_id, e)

        self.log.info("[%s] serving on %s:%d (debug=%s)",
                      self.service_id, host, port, debug)
        # Quart.run() defaults use_reloader=True *independent of debug* — it
        # spawns a second full process (its own model load, its own GPU pool
        # with 1 token for the same physical GPU) that silently races the
        # real worker for GPU 0. Two nnUNet processes hitting one GPU
        # simultaneously crashes the whole container with no Python-level
        # error. GPU services are never edited live in the running
        # container, so the reloader serves no purpose here.
        self.app.run(host=host, port=port, debug=debug, use_reloader=False)