"""
Performance monitoring module for benchmarking.
Collects CPU, memory, and timing metrics.
"""

import csv
import json
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
    stage: str  # "preprocessing", "segmentation", etc.
    parallelism_type: str  # "cpu_workers" or "gpu_concurrent"
    parallelism_level: int  # number of workers or max_concurrent requests
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
    # GPU-specific parameters
    server_name: Optional[str] = None  # "barguzin", "cube", "tunka"
    gpu_count: Optional[int] = None  # number of GPUs on server
    gpu_ids: Optional[str] = None  # e.g., "0,1,2"
    
    # Legacy parameters (for backward compatibility with preprocessing)
    mode: Optional[str] = None  # "sequential" or "parallel"
    workers: Optional[int] = None  # number of CPU workers
    
    # System resource metrics (client-side)
    cpu_avg: Optional[float] = None
    cpu_max: Optional[float] = None
    memory_avg_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    
    # Comparative metrics
    speedup: Optional[float] = None  # vs baseline
    efficiency: Optional[float] = None  # speedup / parallelism_level
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


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
        
        # Metrics storage
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
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
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        process = psutil.Process()
        
        while self.monitoring:
            try:
                # CPU usage (percentage)
                cpu_percent = process.cpu_percent(interval=0.1)
                self.cpu_samples.append(cpu_percent)
                
                # Memory usage (MB)
                memory_mb = process.memory_info().rss / 1024 / 1024
                self.memory_samples.append(memory_mb)
                
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
        if not self.enabled or not self.cpu_samples:
            return {
                'cpu_avg': None,
                'cpu_max': None,
                'memory_avg_mb': None,
                'memory_peak_mb': None,
                'duration': self.end_time - self.start_time if self.start_time and self.end_time else None
            }
        
        return {
            'cpu_avg': sum(self.cpu_samples) / len(self.cpu_samples),
            'cpu_max': max(self.cpu_samples),
            'memory_avg_mb': sum(self.memory_samples) / len(self.memory_samples),
            'memory_peak_mb': max(self.memory_samples),
            'duration': self.end_time - self.start_time if self.start_time and self.end_time else None
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
                    stage=row.get('stage', 'preprocessing'),  # default for old CSVs
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