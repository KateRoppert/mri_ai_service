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
    """Container for experiment metrics."""
    experiment_id: str
    timestamp: str
    mode: str  # sequential or parallel
    workers: int
    total_series: int
    successful: int
    failed: int
    skipped: int
    total_time: float
    time_per_series: float
    throughput: float  # series per second
    cpu_avg: Optional[float] = None
    cpu_max: Optional[float] = None
    memory_avg_mb: Optional[float] = None
    memory_peak_mb: Optional[float] = None
    speedup: Optional[float] = None  # vs baseline
    efficiency: Optional[float] = None  # speedup / workers
    
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
            'experiment_id', 'timestamp', 'mode', 'workers',
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
                metrics.mode,
                metrics.workers,
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
                metrics = ExperimentMetrics(
                    experiment_id=row['experiment_id'],
                    timestamp=row['timestamp'],
                    mode=row['mode'],
                    workers=int(row['workers']),
                    total_series=int(row['total_series']),
                    successful=int(row['successful']),
                    failed=int(row['failed']),
                    skipped=int(row['skipped']),
                    total_time=float(row['total_time']),
                    time_per_series=float(row['time_per_series']),
                    throughput=float(row['throughput']),
                    cpu_avg=float(row['cpu_avg']) if row['cpu_avg'] else None,
                    cpu_max=float(row['cpu_max']) if row['cpu_max'] else None,
                    memory_avg_mb=float(row['memory_avg_mb']) if row['memory_avg_mb'] else None,
                    memory_peak_mb=float(row['memory_peak_mb']) if row['memory_peak_mb'] else None,
                    speedup=float(row['speedup']) if row['speedup'] else None,
                    efficiency=float(row['efficiency']) if row['efficiency'] else None
                )
                metrics_list.append(metrics)
        
        return metrics_list
    
    def get_baseline_time(self) -> Optional[float]:
        """
        Get baseline (sequential, 1 worker) time per series.
        
        Returns:
            Baseline time or None if not found
        """
        metrics_list = self.load_all_metrics()
        
        for metrics in metrics_list:
            if metrics.mode == 'sequential' and metrics.workers == 1:
                return metrics.time_per_series
        
        return None