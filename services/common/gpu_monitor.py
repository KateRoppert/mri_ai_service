"""
GPU monitoring utilities for inference services.

Provides background sampling of GPU utilization, memory usage, and temperature
during inference jobs. Used by all services that perform GPU inference to
attach resource metrics to job results.

Requires pynvml. If pynvml is unavailable, the monitor degrades gracefully
to a no-op rather than failing the service.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


class GPUMonitor:
    """
    Sample-based GPU monitor.

    Usage:
        monitor = GPUMonitor(gpu_id=0)
        monitor.start()
        # ... run inference ...
        metrics = monitor.stop()
        # metrics: {'utilization_avg', 'utilization_max',
        #          'memory_used_mb_avg', 'memory_used_mb_max',
        #          'temperature_avg', 'temperature_max'}

    If pynvml is not installed or the GPU cannot be queried, start()/stop()
    are no-ops and stop() returns an empty dict. This allows services to
    use the monitor unconditionally without try/except wrappers.
    """

    SAMPLE_INTERVAL_SEC = 0.5

    def __init__(self, gpu_id: int) -> None:
        self.gpu_id = gpu_id
        self.monitoring = False
        self.monitor_thread: threading.Thread | None = None

        self.utilization_samples: list[float] = []
        self.memory_used_samples: list[float] = []  # in MB
        self.temperature_samples: list[float] = []

        # Try to initialize pynvml lazily, log if unavailable
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._available = True
        except Exception as e:
            log.warning("pynvml not available, GPU monitoring disabled: %s", e)
            self._pynvml = None
            self._available = False

    def start(self) -> None:
        """Begin sampling in a background daemon thread."""
        if not self._available:
            return

        self.monitoring = True
        self.utilization_samples = []
        self.memory_used_samples = []
        self.temperature_samples = []

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop(self) -> dict[str, Any]:
        """Stop sampling and return aggregated metrics."""
        if not self._available:
            return {}

        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)

        return self.get_metrics()

    def _monitor_loop(self) -> None:
        try:
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.gpu_id)
        except Exception as e:
            log.error("Failed to acquire GPU handle for gpu_id=%d: %s", self.gpu_id, e)
            return

        while self.monitoring:
            try:
                utilization = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                self.utilization_samples.append(utilization.gpu)

                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                self.memory_used_samples.append(mem_info.used / 1024 / 1024)

                temperature = self._pynvml.nvmlDeviceGetTemperature(
                    handle, self._pynvml.NVML_TEMPERATURE_GPU
                )
                self.temperature_samples.append(temperature)

                time.sleep(self.SAMPLE_INTERVAL_SEC)

            except Exception as e:
                log.error("GPU sampling error: %s", e)
                break

    def get_metrics(self) -> dict[str, Any]:
        """Aggregate collected samples. Returns empty dict if no samples."""
        if not self.utilization_samples:
            return {}

        return {
            "utilization_avg": sum(self.utilization_samples) / len(self.utilization_samples),
            "utilization_max": max(self.utilization_samples),
            "memory_used_mb_avg": sum(self.memory_used_samples) / len(self.memory_used_samples),
            "memory_used_mb_max": max(self.memory_used_samples),
            "temperature_avg": sum(self.temperature_samples) / len(self.temperature_samples),
            "temperature_max": max(self.temperature_samples),
        }