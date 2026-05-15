"""
Data contracts for MAS services.

Defines the request/response payloads and internal job lifecycle types.
Kept separate from service_base.py so that orchestrators, tests, and
external clients can import the types without pulling in the HTTP layer.

These contracts implement the API described in docs/SPEC.md §3.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Job status enum
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Lifecycle states of a single /predict request."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Internal job record (lives inside the service)
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """
    Internal representation of a single inference job.

    The orchestrator never sees this object directly — only the dict
    returned by Job.to_status_dict() or Job.to_result_dict() over HTTP.
    """

    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.QUEUED
    payload: dict[str, Any] = field(default_factory=dict)

    # Lifecycle timestamps (unix epoch seconds)
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None

    # Execution metadata
    gpu_id: int | None = None
    message: str = "queued"
    progress: int = 0  # 0..100, coarse-grained

    # Outputs
    result: dict[str, Any] | None = None
    error: str | None = None

    def mark_running(self, gpu_id: int) -> None:
        self.status = JobStatus.RUNNING
        self.started_at = time.time()
        self.gpu_id = gpu_id
        self.message = f"running on gpu {gpu_id}"
        self.progress = 10

    def mark_succeeded(self, result: dict[str, Any]) -> None:
        self.status = JobStatus.SUCCEEDED
        self.finished_at = time.time()
        self.result = result
        self.message = "succeeded"
        self.progress = 100

    def mark_failed(self, error: str) -> None:
        self.status = JobStatus.FAILED
        self.finished_at = time.time()
        self.error = error
        self.message = f"failed: {error[:120]}"
        self.progress = 0

    def to_status_dict(self) -> dict[str, Any]:
        """Compact view for GET /predict/{job_id}/status."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "gpu_id": self.gpu_id,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    def to_result_dict(self) -> dict[str, Any]:
        """Full view for GET /predict/{job_id}/result. Includes mask + metrics."""
        return {
            "job_id": self.job_id,
            "status": self.status.value,
            **(self.result or {}),
        }