"""
Performance monitoring module for benchmarking.
Collects CPU, memory, and timing metrics.
"""

import csv
import json
import shutil
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available. CPU/Memory monitoring disabled.")


@dataclass
class ExperimentMetrics:
    """Container for experiment metrics (universal for CPU and GPU workloads)."""
    # ========================================================================
    # REQUIRED FIELDS (no defaults) - must come first!
    # ========================================================================
    experiment_id: str
    timestamp: str
    total_series: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    time_per_series: float
    throughput: float  # series per second
    
    # ========================================================================
    # OPTIONAL FIELDS (with defaults) - must come after required fields!
    # ========================================================================
    # New universal fields
    stage: Optional[str] = None  # ← Теперь опциональное
    parallelism_type: Optional[str] = None  # ← Теперь опциональное
    parallelism_level: Optional[int] = None  # ← Теперь опциональное
    
    # GPU-specific parameters
    server_name: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_ids: Optional[str] = None
    
    # Legacy parameters (for backward compatibility)
    mode: Optional[str] = None  # "sequential" or "parallel"
    workers: Optional[int] = None  # number of CPU workers
    
    # System resource metrics (client-side)
    cpu_avg: Optional[float] = None
    cpu_max: Optional[float] = None
    memory_avg_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None

    # GPU metrics
    gpu_utilization_avg: Optional[float] = None
    gpu_utilization_max: Optional[float] = None
    gpu_memory_used_mb_avg: Optional[float] = None
    gpu_memory_used_mb_max: Optional[float] = None
    gpu_temperature_avg: Optional[float] = None
    gpu_temperature_max: Optional[float] = None
    
    # System-wide load. cpu_avg/memory_* above describe OUR process tree;
    # these describe the machine. Both are kept because they answer different
    # questions: the tree says whether the load is ours, the machine says
    # whether it is busy.
    #
    # MIND THE TWO SCALES — they are not comparable without dividing.
    # cpu_avg sums per-process percentages, so four busy workers read ~400%.
    # system_cpu_avg is normalised over all cores, so the same four workers on
    # a 24-thread machine read ~17%. Measured together: tree 261% avg / 406%
    # peak against machine 15%. The utilisation figure the previous paper
    # quotes (12.6% -> 31.6%) is in the machine's scale, so RQ3 compares
    # against system_cpu_avg, not cpu_avg.
    system_cpu_avg: Optional[float] = None
    system_cpu_max: Optional[float] = None
    system_memory_peak_mb: Optional[float] = None

    # I/O. The previous paper explains stages 1-3 saturating at 8 processes by
    # disk bandwidth, but inferred it from the shape of the speedup curve.
    # These measure it instead.
    iowait_avg: Optional[float] = None
    iowait_max: Optional[float] = None
    disk_read_mb: Optional[float] = None
    disk_write_mb: Optional[float] = None

    # Energy. The paper's premise is that clinical hardware is limited, so if
    # a strategy wins by keeping more of it busy, the honest form of that claim
    # includes what it costs.
    #
    # The GPU has no energy counter, so this is power sampled over the run and
    # integrated. The CPU does have one (RAPL, microjoules), which is exact —
    # when it is readable at all: energy_uj is root-only since PLATYPUS, where
    # power draw was used to recover encryption keys. cpu_energy_wh stays None
    # rather than 0.0 when it cannot be read, so a table never claims the CPU
    # spent nothing.
    gpu_power_avg_w: Optional[float] = None
    gpu_power_max_w: Optional[float] = None
    gpu_energy_wh: Optional[float] = None
    cpu_energy_wh: Optional[float] = None

    # Comparative metrics
    speedup: Optional[float] = None
    efficiency: Optional[float] = None
    
    def __post_init__(self):
        """Auto-fill new fields from legacy fields for backward compatibility."""
        # Если новые поля не заданы, заполняем из старых
        if self.stage is None:
            self.stage = "unknown"  # Дефолтное значение
        
        if self.parallelism_type is None:
            self.parallelism_type = "cpu_workers"  # По умолчанию CPU
        
        if self.parallelism_level is None:
            # Берем из workers если есть
            self.parallelism_level = self.workers if self.workers else 1
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


RAPL_ROOT = "/sys/class/powercap"
# Sampling nvidia-smi costs ~50-100 ms, so it runs on a slower cadence than the
# rest: a GPU's power does not swing meaningfully inside two seconds, and the
# subprocess should not show up in the CPU numbers we are collecting.
GPU_POWER_EVERY_N_TICKS = 4


def _rapl_energy_uj(root=RAPL_ROOT) -> Optional[int]:
    """Total CPU package energy in microjoules, or None if unreadable.

    Only top-level domains (intel-rapl:N) are summed. Their subdomains
    (intel-rapl:N:M — cores, uncore, dram) are parts of the same package, so
    adding both would count the same joules twice.
    """
    from pathlib import Path as _Path
    total = None
    try:
        domains = sorted(_Path(root).glob("intel-rapl:[0-9]"))
    except OSError:
        return None
    for domain in domains:
        try:
            value = int((domain / "energy_uj").read_text().strip())
        except (OSError, ValueError):
            continue  # no permission on this domain, or it went away
        total = value if total is None else total + value
    return total


def _rapl_max_range_uj(root=RAPL_ROOT) -> Optional[int]:
    """Where the counter wraps, summed over the same domains."""
    from pathlib import Path as _Path
    total = None
    try:
        domains = sorted(_Path(root).glob("intel-rapl:[0-9]"))
    except OSError:
        return None
    for domain in domains:
        try:
            value = int((domain / "max_energy_range_uj").read_text().strip())
        except (OSError, ValueError):
            continue
        total = value if total is None else total + value
    return total


def _gpu_power_watts() -> Optional[float]:
    """Power drawn by every visible GPU, summed, or None.

    Summed because a run spends what all its cards spend — barguzin has two.
    Laptop cards often report [N/A] for power.draw; cards that do are skipped,
    and None comes back only when nothing reported at all.
    """
    try:
        import pynvml  # present in the service images, not in the host venv
        pynvml.nvmlInit()
        try:
            total = 0.0
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                total += pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            if total > 0:
                return total
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    try:
        if not shutil.which("nvidia-smi"):
            return None
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return None
        values = []
        for line in result.stdout.strip().splitlines():
            try:
                values.append(float(line.strip()))
            except ValueError:
                continue  # [N/A]
        return sum(values) if values else None
    except Exception:
        return None


class PerformanceMonitor:
    """Monitors system resources during execution."""
    
    def __init__(self, enabled: bool = True):
        """
        Initialize monitor.
        
        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled and PSUTIL_AVAILABLE
        self.monitoring = False
        self.monitor_thread = None
        
        # Metrics storage. cpu_samples/memory_samples are the PROCESS TREE
        # (this process plus its workers), not this process alone — the stages
        # do their work in ProcessPoolExecutor children while the parent waits.
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.system_cpu_samples: List[float] = []
        self.system_memory_samples: List[float] = []
        self.iowait_samples: List[float] = []
        self._disk_at_start = None
        self._disk_latest = None
        self._procs = {}
        self.gpu_power_samples: List[float] = []
        self._rapl_root = RAPL_ROOT
        self._rapl_last = None
        self._rapl_max_range = None
        self._cpu_energy_uj = 0
        self._cpu_energy_available = False
        self._tick = 0
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
        if enabled and not PSUTIL_AVAILABLE:
            print("Warning: Performance monitoring requested but psutil not installed.")
    
    def start(self):
        """Start monitoring."""
        if not self.enabled:
            return
        
        self.monitoring = True
        self.cpu_samples = []
        self.memory_samples = []
        self.system_cpu_samples = []
        self.system_memory_samples = []
        self.iowait_samples = []
        self._disk_at_start = self._disk_counters()
        self._disk_latest = self._disk_at_start
        self._procs = {}
        self.gpu_power_samples = []
        self._rapl_last = _rapl_energy_uj(self._rapl_root)
        self._rapl_max_range = _rapl_max_range_uj(self._rapl_root)
        self._cpu_energy_uj = 0
        self._cpu_energy_available = self._rapl_last is not None
        self._tick = 0
        # Prime the per-process baseline: the first cpu_percent() call always
        # returns 0.0, it only sets the reference point for the next one.
        self._root = psutil.Process()
        self._root.cpu_percent()
        psutil.cpu_percent()
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop(self):
        """Stop monitoring."""
        if not self.enabled:
            return
        
        self.monitoring = False
        self.end_time = time.time()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    @staticmethod
    def _disk_counters():
        """Cumulative disk counters, or None where they are not available.

        Inside an unprivileged container psutil can return None; the run must
        still produce its other metrics.
        """
        try:
            return psutil.disk_io_counters()
        except Exception:
            return None

    def _tree(self) -> List:
        """This process and every worker it spawned, as CACHED objects.

        The cache is the point. `children(recursive=True)` hands back NEW
        Process objects on every call, and the first `cpu_percent()` on an
        object always returns 0.0 — it only sets the reference point for the
        next call. Rebuilding the list each tick therefore reported 0% for
        every worker, which is how the parent-only defect almost survived
        its own fix (measured: 2% for a tree burning four cores).

        Membership is still refreshed each tick, because a pool replaces its
        workers over the life of a stage.
        """
        try:
            current = {proc.pid: proc for proc in self._root.children(recursive=True)}
        except Exception:
            current = {}
        current[self._root.pid] = self._root

        for pid, proc in current.items():
            if pid not in self._procs:
                try:
                    proc.cpu_percent()  # prime the baseline, value discarded
                except Exception:
                    continue
                self._procs[pid] = proc

        for pid in [pid for pid in self._procs if pid not in current]:
            del self._procs[pid]

        return list(self._procs.values())

    def _sample(self):
        """One sampling tick.

        Kept separate from the loop so it can be exercised without threads —
        a sampler that silently measures the wrong process is exactly the
        failure this split guards against.
        """
        tree_cpu = 0.0
        tree_mem = 0.0
        for proc in self._tree():
            try:
                tree_cpu += proc.cpu_percent()
                tree_mem += proc.memory_info().rss
            except Exception:
                continue  # a worker may exit between listing and reading
        self.cpu_samples.append(tree_cpu)
        self.memory_samples.append(tree_mem / 1024 / 1024)

        try:
            self.system_cpu_samples.append(psutil.cpu_percent())
            vm = psutil.virtual_memory()
            self.system_memory_samples.append((vm.total - vm.available) / 1024 / 1024)
        except Exception:
            pass

        try:
            self.iowait_samples.append(
                getattr(psutil.cpu_times_percent(), "iowait", 0.0) or 0.0)
        except Exception:
            pass

        latest = self._disk_counters()
        if latest is not None:
            self._disk_latest = latest

        self._sample_energy()

    def _sample_energy(self):
        """CPU energy from the RAPL counter; GPU power on a slower cadence."""
        self._tick += 1

        if self._cpu_energy_available:
            now = _rapl_energy_uj(self._rapl_root)
            if now is not None and self._rapl_last is not None:
                delta = now - self._rapl_last
                if delta < 0 and self._rapl_max_range:
                    # The counter is a ring: at 100 W it wraps in roughly 45
                    # minutes, and a matrix run is longer than that.
                    delta += self._rapl_max_range
                if delta >= 0:
                    self._cpu_energy_uj += delta
            if now is not None:
                self._rapl_last = now

        if self._tick % GPU_POWER_EVERY_N_TICKS == 1:
            watts = _gpu_power_watts()
            if watts is not None:
                self.gpu_power_samples.append(watts)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.monitoring:
            try:
                self._sample()
                time.sleep(0.5)  # Sample every 0.5 seconds
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def get_metrics(self) -> Dict[str, Optional[float]]:
        """
        Get collected metrics.
        
        Returns:
            Dictionary with metrics
        """
        # `is not None`, not truthiness: a start_time of 0.0 is a valid
        # instant, and `if self.start_time` silently discards the duration.
        duration = (self.end_time - self.start_time
                    if self.start_time is not None and self.end_time is not None
                    else None)

        def agg(samples, how):
            return how(samples) if samples else None

        gpu_power_avg = (sum(self.gpu_power_samples) / len(self.gpu_power_samples)
                         if self.gpu_power_samples else None)
        # No energy counter on the card, so power is integrated over the run.
        # Sampling is near-uniform, which makes mean x duration a sound
        # estimator; it is also the form a reader can check by hand.
        gpu_energy_wh = (gpu_power_avg * duration / 3600.0
                         if gpu_power_avg is not None and duration else None)
        cpu_energy_wh = (self._cpu_energy_uj / 1e6 / 3600.0
                         if self._cpu_energy_available else None)

        read_mb = write_mb = None
        if self._disk_at_start is not None and self._disk_latest is not None:
            read_mb = (self._disk_latest.read_bytes
                       - self._disk_at_start.read_bytes) / 1024 / 1024
            write_mb = (self._disk_latest.write_bytes
                        - self._disk_at_start.write_bytes) / 1024 / 1024

        if not self.enabled or not self.cpu_samples:
            return {
                'cpu_avg': None,
                'cpu_max': None,
                'memory_avg_mb': None,
                'memory_peak_mb': None,
                'system_cpu_avg': None,
                'system_cpu_max': None,
                'system_memory_peak_mb': None,
                'iowait_avg': None,
                'iowait_max': None,
                'disk_read_mb': read_mb,
                'disk_write_mb': write_mb,
                'gpu_power_avg_w': gpu_power_avg,
                'gpu_power_max_w': (max(self.gpu_power_samples)
                                    if self.gpu_power_samples else None),
                'gpu_energy_wh': gpu_energy_wh,
                'cpu_energy_wh': cpu_energy_wh,
                'duration': duration,
            }

        return {
            'cpu_avg': sum(self.cpu_samples) / len(self.cpu_samples),
            'cpu_max': max(self.cpu_samples),
            'memory_avg_mb': sum(self.memory_samples) / len(self.memory_samples),
            'memory_peak_mb': max(self.memory_samples),
            'system_cpu_avg': agg(self.system_cpu_samples,
                                  lambda s: sum(s) / len(s)),
            'system_cpu_max': agg(self.system_cpu_samples, max),
            'system_memory_peak_mb': agg(self.system_memory_samples, max),
            'iowait_avg': agg(self.iowait_samples, lambda s: sum(s) / len(s)),
            'iowait_max': agg(self.iowait_samples, max),
            'disk_read_mb': read_mb,
            'disk_write_mb': write_mb,
            'gpu_power_avg_w': gpu_power_avg,
            'gpu_power_max_w': (max(self.gpu_power_samples)
                                if self.gpu_power_samples else None),
            'gpu_energy_wh': gpu_energy_wh,
            'cpu_energy_wh': cpu_energy_wh,
            'duration': duration,
        }


class BenchmarkLogger:
    """Handles saving benchmark results."""
    
    def __init__(self, results_dir: Path):
        """
        Initialize logger.
        
        Args:
            results_dir: Directory to save results
        """
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_file = self.results_dir / "metrics.csv"
        self.summary_file = self.results_dir / "summary.json"
        
        # Initialize CSV if doesn't exist
        if not self.csv_file.exists():
            self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        headers = [
            'experiment_id', 'timestamp', 'stage',
            'parallelism_type', 'parallelism_level',
            'server_name', 'gpu_count', 'gpu_ids',
            'mode', 'workers',  # legacy fields
            'total_series', 'successful', 'failed', 'skipped',
            'total_time', 'time_per_series', 'throughput',
            'cpu_avg', 'cpu_max', 'memory_avg_mb', 'memory_peak_mb',
            'gpu_utilization_avg', 'gpu_utilization_max',
            'gpu_memory_used_mb_avg', 'gpu_memory_used_mb_max',
            'gpu_temperature_avg', 'gpu_temperature_max',
            'system_cpu_avg', 'system_cpu_max', 'system_memory_peak_mb',
            'iowait_avg', 'iowait_max', 'disk_read_mb', 'disk_write_mb',
            'gpu_power_avg_w', 'gpu_power_max_w', 'gpu_energy_wh',
            'cpu_energy_wh',
            'speedup', 'efficiency'
        ]
        
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_metrics(self, metrics: ExperimentMetrics):
        """
        Log metrics to CSV.
        
        Args:
            metrics: Experiment metrics
        """
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.experiment_id,
                metrics.timestamp,
                metrics.stage,
                metrics.parallelism_type,
                metrics.parallelism_level,
                metrics.server_name or "",
                metrics.gpu_count or "",
                metrics.gpu_ids or "",
                metrics.mode or "",
                metrics.workers or "",
                metrics.total_series,
                metrics.successful,
                metrics.failed,
                metrics.skipped,
                f"{metrics.total_time:.3f}",
                f"{metrics.time_per_series:.3f}",
                f"{metrics.throughput:.3f}",
                f"{metrics.cpu_avg:.2f}" if metrics.cpu_avg else "",
                f"{metrics.cpu_max:.2f}" if metrics.cpu_max else "",
                f"{metrics.memory_avg_mb:.2f}" if metrics.memory_avg_mb else "",
                f"{metrics.memory_peak_mb:.2f}" if metrics.memory_peak_mb else "",
                f"{metrics.gpu_utilization_avg:.2f}" if metrics.gpu_utilization_avg else "",
                f"{metrics.gpu_utilization_max:.2f}" if metrics.gpu_utilization_max else "",
                f"{metrics.gpu_memory_used_mb_avg:.2f}" if metrics.gpu_memory_used_mb_avg else "",
                f"{metrics.gpu_memory_used_mb_max:.2f}" if metrics.gpu_memory_used_mb_max else "",
                f"{metrics.gpu_temperature_avg:.2f}" if metrics.gpu_temperature_avg else "",
                f"{metrics.gpu_temperature_max:.2f}" if metrics.gpu_temperature_max else "",
                # `is not None`, not truthiness: a measured 0.0 (idle disk, no
                # iowait) is a result, and `if 0.0` would write it as blank.
                f"{metrics.system_cpu_avg:.2f}" if metrics.system_cpu_avg is not None else "",
                f"{metrics.system_cpu_max:.2f}" if metrics.system_cpu_max is not None else "",
                f"{metrics.system_memory_peak_mb:.2f}" if metrics.system_memory_peak_mb is not None else "",
                f"{metrics.iowait_avg:.3f}" if metrics.iowait_avg is not None else "",
                f"{metrics.iowait_max:.3f}" if metrics.iowait_max is not None else "",
                f"{metrics.disk_read_mb:.2f}" if metrics.disk_read_mb is not None else "",
                f"{metrics.disk_write_mb:.2f}" if metrics.disk_write_mb is not None else "",
                f"{metrics.gpu_power_avg_w:.2f}" if metrics.gpu_power_avg_w is not None else "",
                f"{metrics.gpu_power_max_w:.2f}" if metrics.gpu_power_max_w is not None else "",
                f"{metrics.gpu_energy_wh:.4f}" if metrics.gpu_energy_wh is not None else "",
                f"{metrics.cpu_energy_wh:.4f}" if metrics.cpu_energy_wh is not None else "",
                f"{metrics.speedup:.3f}" if metrics.speedup else "",
                f"{metrics.efficiency:.3f}" if metrics.efficiency else ""
            ])
        
        # Save individual experiment JSON
        experiment_file = self.results_dir / f"{metrics.experiment_id}.json"
        with open(experiment_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        
        print(f"✓ Metrics saved to {self.csv_file}")
    
    def load_all_metrics(self) -> List[ExperimentMetrics]:
        """
        Load all metrics from CSV.
        
        Returns:
            List of ExperimentMetrics
        """
        metrics_list = []
        
        if not self.csv_file.exists():
            return metrics_list
        
        with open(self.csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert strings to appropriate types
                # Handle both new and legacy CSV formats
                metrics = ExperimentMetrics(
                    experiment_id=row['experiment_id'],
                    timestamp=row['timestamp'],
                    stage=row.get('stage', 'preprocessing'),
                    parallelism_type=row.get('parallelism_type', 'cpu_workers'),
                    parallelism_level=int(row.get('parallelism_level', row.get('workers', 1))),
                    server_name=row.get('server_name') or None,
                    gpu_count=int(row['gpu_count']) if row.get('gpu_count') else None,
                    gpu_ids=row.get('gpu_ids') or None,
                    mode=row.get('mode') or None,
                    workers=int(row['workers']) if row.get('workers') else None,
                    total_series=int(row['total_series']),
                    successful=int(row['successful']),
                    failed=int(row['failed']),
                    skipped=int(row['skipped']),
                    total_time=float(row['total_time']),
                    time_per_series=float(row['time_per_series']),
                    throughput=float(row['throughput']),
                    cpu_avg=float(row['cpu_avg']) if row.get('cpu_avg') else None,
                    cpu_max=float(row['cpu_max']) if row.get('cpu_max') else None,
                    memory_avg_mb=float(row['memory_avg_mb']) if row.get('memory_avg_mb') else None,
                    memory_peak_mb=float(row['memory_peak_mb']) if row.get('memory_peak_mb') else None,
                    gpu_utilization_avg=float(row['gpu_utilization_avg']) if row.get('gpu_utilization_avg') else None,
                    gpu_utilization_max=float(row['gpu_utilization_max']) if row.get('gpu_utilization_max') else None,
                    gpu_memory_used_mb_avg=float(row['gpu_memory_used_mb_avg']) if row.get('gpu_memory_used_mb_avg') else None,
                    gpu_memory_used_mb_max=float(row['gpu_memory_used_mb_max']) if row.get('gpu_memory_used_mb_max') else None,
                    gpu_temperature_avg=float(row['gpu_temperature_avg']) if row.get('gpu_temperature_avg') else None,
                    gpu_temperature_max=float(row['gpu_temperature_max']) if row.get('gpu_temperature_max') else None,
                    speedup=float(row['speedup']) if row.get('speedup') else None,
                    efficiency=float(row['efficiency']) if row.get('efficiency') else None
                )
                metrics_list.append(metrics)
        
        return metrics_list
    
    def get_baseline_time(self, stage: str = None, server_name: str = None) -> Optional[float]:
        """
        Get baseline time per series for comparison.
        
        For CPU workflows: sequential mode with 1 worker
        For GPU workflows: 1 concurrent request on specific server
        
        Args:
            stage: Filter by stage (optional)
            server_name: Filter by server name (for GPU workflows)
        
        Returns:
            Baseline time or None if not found
        """
        metrics_list = self.load_all_metrics()
        
        for metrics in metrics_list:
            # Filter by stage if specified
            if stage and metrics.stage != stage:
                continue
            
            # Filter by server_name if specified (for GPU benchmarks)
            if server_name and metrics.server_name != server_name:
                continue
            
            # Check if this is a baseline experiment
            is_baseline = False
            
            if metrics.parallelism_type == 'cpu_workers':
                # CPU baseline: sequential mode with 1 worker
                is_baseline = (metrics.mode == 'sequential' and metrics.parallelism_level == 1)
            elif metrics.parallelism_type == 'gpu_concurrent':
                # GPU baseline: 1 concurrent request
                is_baseline = (metrics.parallelism_level == 1)
            
            if is_baseline:
                return metrics.time_per_series
        
        return None