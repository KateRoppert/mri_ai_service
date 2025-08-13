"""
Performance Profiling Module for MRI Segmentation Pipeline
Provides decorators and utilities for monitoring execution time, memory usage, and system resources.
"""

import time
import psutil
import tracemalloc
import functools
import logging
import json
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import cProfile
import pstats
import io
from contextlib import contextmanager
from collections import defaultdict
import numpy as np


@dataclass
class MetricRecord:
    """Single metric record."""
    timestamp: float
    function_name: str
    module_name: str
    execution_time: float
    memory_start: float
    memory_end: float
    memory_delta: float
    cpu_percent: float
    args_info: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    

class PerformanceProfiler:
    """
    Centralized performance profiler for the entire pipeline.
    
    Usage:
        profiler = PerformanceProfiler(enabled=True)
        
        @profiler.measure
        def my_function():
            pass
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern to ensure one profiler instance."""
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, 
                 enabled: bool = True,
                 log_slow_operations: float = 1.0,  # Log ops slower than 1 second
                 memory_snapshots: bool = True,
                 output_dir: Optional[str] = None):
        """Initialize profiler with configuration."""
        if hasattr(self, '_initialized'):
            return
            
        self.enabled = enabled
        self.log_slow_operations = log_slow_operations
        self.memory_snapshots = memory_snapshots
        self.output_dir = Path(output_dir) if output_dir else Path('profiling_reports')
        
        self.metrics: List[MetricRecord] = []
        self.phase_timings: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.memory_checkpoints: List[Dict[str, Any]] = []
        
        self._start_time = None
        self._start_memory = None
        self._tracemalloc_started = False
        
        # Setup logging
        self.logger = logging.getLogger('performance_profiler')
        
        # Create output directory
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        self._initialized = True
    
    def start_session(self, session_name: str = "default"):
        """Start a new profiling session."""
        if not self.enabled:
            return
            
        self.session_name = session_name
        self._start_time = time.time()
        
        # Memory tracking
        process = psutil.Process()
        self._start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        if self.memory_snapshots and not self._tracemalloc_started:
            tracemalloc.start()
            self._tracemalloc_started = True
            
        self.logger.info(f"Started profiling session: {session_name}")
    
    def measure(self, 
                func: Optional[Callable] = None, 
                *, 
                capture_args: bool = False,
                log_result: bool = False) -> Callable:
        """
        Decorator to measure function performance.
        
        Args:
            func: Function to measure
            capture_args: Whether to capture function arguments
            log_result: Whether to log function result
        """
        def decorator(f):
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                if not self.enabled:
                    return f(*args, **kwargs)
                
                # Prepare metadata
                module_name = f.__module__
                function_name = f.__qualname__
                
                # Capture args info if requested
                args_info = None
                if capture_args:
                    args_info = self._capture_args_info(args, kwargs)
                
                # Start measurements
                start_time = time.time()
                process = psutil.Process()
                start_memory = process.memory_info().rss / 1024 / 1024
                cpu_before = process.cpu_percent(interval=None)
                
                # Execute function
                error = None
                result = None
                try:
                    result = f(*args, **kwargs)
                except Exception as e:
                    error = str(e)
                    raise
                finally:
                    # End measurements
                    end_time = time.time()
                    end_memory = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent(interval=None)
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    
                    # Create metric record
                    metric = MetricRecord(
                        timestamp=end_time,
                        function_name=function_name,
                        module_name=module_name,
                        execution_time=execution_time,
                        memory_start=start_memory,
                        memory_end=end_memory,
                        memory_delta=memory_delta,
                        cpu_percent=cpu_percent,
                        args_info=args_info,
                        error=error
                    )
                    
                    self.metrics.append(metric)
                    
                    # Log slow operations
                    if execution_time > self.log_slow_operations:
                        self.logger.warning(
                            f"Slow operation: {function_name} took {execution_time:.2f}s "
                            f"(Memory: {memory_delta:+.2f}MB)"
                        )
                    
                    # Log result if requested
                    if log_result and result is not None:
                        self.logger.debug(f"{function_name} result: {result}")
                
                return result
            
            return wrapper
        
        if func is None:
            return decorator
        else:
            return decorator(func)
    
    @contextmanager
    def measure_block(self, block_name: str):
        """Context manager for measuring code blocks."""
        if not self.enabled:
            yield
            return
            
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss / 1024 / 1024
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            metric = MetricRecord(
                timestamp=end_time,
                function_name=block_name,
                module_name="code_block",
                execution_time=execution_time,
                memory_start=start_memory,
                memory_end=end_memory,
                memory_delta=memory_delta,
                cpu_percent=0.0
            )
            
            self.metrics.append(metric)
            
            if execution_time > self.log_slow_operations:
                self.logger.warning(
                    f"Slow block: {block_name} took {execution_time:.2f}s"
                )
    
    def record_phase(self, phase_name: str, action: str = 'toggle'):
        """Record phase timing (start/end/toggle)."""
        if not self.enabled:
            return
            
        current_time = time.time()
        
        if action == 'start' or (action == 'toggle' and phase_name not in self.phase_timings):
            self.phase_timings[phase_name]['start'] = current_time
        elif action == 'end' or (action == 'toggle' and 'start' in self.phase_timings[phase_name]):
            if 'start' in self.phase_timings[phase_name]:
                self.phase_timings[phase_name]['end'] = current_time
                self.phase_timings[phase_name]['duration'] = (
                    current_time - self.phase_timings[phase_name]['start']
                )
    
    def memory_checkpoint(self, label: str):
        """Take a memory snapshot at specific point."""
        if not self.enabled:
            return
            
        process = psutil.Process()
        memory_info = process.memory_info()
        
        checkpoint = {
            'label': label,
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent()
        }
        
        self.memory_checkpoints.append(checkpoint)
        
        self.logger.info(
            f"Memory checkpoint [{label}]: "
            f"RSS={checkpoint['rss_mb']:.2f}MB ({checkpoint['percent']:.1f}%)"
        )
    
    def get_memory_profile(self, top_n: int = 10) -> List[str]:
        """Get top memory allocations."""
        if not self._tracemalloc_started:
            return ["Memory profiling not enabled"]
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        results = []
        for stat in top_stats[:top_n]:
            results.append(str(stat))
            
        return results
    
    def _capture_args_info(self, args: tuple, kwargs: dict) -> Dict[str, Any]:
        """Safely capture function arguments information."""
        info = {}
        
        # Capture positional args
        if args:
            info['args_count'] = len(args)
            # Capture safe representations
            safe_args = []
            for arg in args[:3]:  # Limit to first 3 args
                if isinstance(arg, (str, int, float, bool)):
                    safe_args.append(arg)
                elif isinstance(arg, (list, tuple)):
                    safe_args.append(f"{type(arg).__name__}(len={len(arg)})")
                elif isinstance(arg, dict):
                    safe_args.append(f"dict(keys={len(arg)})")
                elif hasattr(arg, '__class__'):
                    safe_args.append(f"{arg.__class__.__name__}")
            info['args_preview'] = safe_args
        
        # Capture kwargs keys
        if kwargs:
            info['kwargs_keys'] = list(kwargs.keys())[:5]  # Limit to 5 keys
            
        return info
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.enabled or not self._start_time:
            return {"error": "Profiling not started or disabled"}
        
        total_time = time.time() - self._start_time
        process = psutil.Process()
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Analyze metrics
        function_stats = defaultdict(lambda: {
            'count': 0, 'total_time': 0, 'avg_time': 0, 
            'max_time': 0, 'total_memory': 0
        })
        
        for metric in self.metrics:
            stats = function_stats[metric.function_name]
            stats['count'] += 1
            stats['total_time'] += metric.execution_time
            stats['max_time'] = max(stats['max_time'], metric.execution_time)
            stats['total_memory'] += abs(metric.memory_delta)
        
        # Calculate averages
        for stats in function_stats.values():
            if stats['count'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['count']
        
        # Sort by total time
        sorted_functions = sorted(
            function_stats.items(), 
            key=lambda x: x[1]['total_time'], 
            reverse=True
        )
        
        report = {
            'session_name': getattr(self, 'session_name', 'default'),
            'total_execution_time': total_time,
            'memory_usage': {
                'start_mb': self._start_memory,
                'end_mb': final_memory,
                'delta_mb': final_memory - self._start_memory
            },
            'phase_timings': dict(self.phase_timings),
            'function_statistics': dict(sorted_functions[:20]),  # Top 20
            'memory_checkpoints': self.memory_checkpoints,
            'slow_operations': [
                asdict(m) for m in self.metrics 
                if m.execution_time > self.log_slow_operations
            ],
            'memory_profile': self.get_memory_profile() if self.memory_snapshots else []
        }
        
        return report
    
    def save_report(self, filename: Optional[str] = None):
        """Save performance report to file."""
        if not self.enabled:
            return None
            
        report = self.generate_report()
        
        # Generate filename
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{self.session_name}_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        self.logger.info(f"Performance report saved to: {filepath}")
        
        # Also generate human-readable summary
        summary_path = filepath.with_suffix('.txt')
        with open(summary_path, 'w') as f:
            f.write(self._format_report_summary(report))
            
        return filepath
    
    def _format_report_summary(self, report: Dict[str, Any]) -> str:
        """Format report as human-readable summary."""
        lines = [
            "="*70,
            f"PERFORMANCE REPORT - {report['session_name']}",
            "="*70,
            f"Total execution time: {report['total_execution_time']:.2f}s",
            f"Memory: {report['memory_usage']['start_mb']:.2f}MB → "
            f"{report['memory_usage']['end_mb']:.2f}MB "
            f"(Δ{report['memory_usage']['delta_mb']:+.2f}MB)",
            "",
            "TOP FUNCTIONS BY TIME:",
        ]
        
        for func, stats in list(report['function_statistics'].items())[:10]:
            lines.append(
                f"  {func}: {stats['total_time']:.2f}s total, "
                f"{stats['avg_time']:.3f}s avg ({stats['count']} calls)"
            )
        
        if report['phase_timings']:
            lines.extend(["", "PHASE TIMINGS:"])
            total = report['total_execution_time']
            for phase, timing in report['phase_timings'].items():
                if 'duration' in timing:
                    pct = (timing['duration'] / total) * 100
                    lines.append(f"  {phase}: {timing['duration']:.2f}s ({pct:.1f}%)")
        
        return "\n".join(lines)
    
    def profile_with_cprofile(self, func: Callable, *args, **kwargs):
        """Run function with cProfile for detailed analysis."""
        if not self.enabled:
            return func(*args, **kwargs)
            
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()
            
            # Save detailed profile
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            profile_path = self.output_dir / f"cprofile_{func.__name__}_{timestamp}.prof"
            profiler.dump_stats(str(profile_path))
            
            # Generate readable stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(50)
            
            stats_path = profile_path.with_suffix('.txt')
            with open(stats_path, 'w') as f:
                f.write(s.getvalue())
                
            self.logger.info(f"Detailed profile saved to: {profile_path}")
            
        return result
    
    def reset(self):
        """Reset all metrics for a new session."""
        self.metrics.clear()
        self.phase_timings.clear()
        self.memory_checkpoints.clear()
        self._start_time = None
        self._start_memory = None


# Глобальный экземпляр профайлера
profiler = PerformanceProfiler(enabled=False)  # По умолчанию выключен


# Удобные декораторы для прямого использования
measure = profiler.measure
measure_block = profiler.measure_block


# Функция для настройки профилирования из других модулей
def setup_profiling(enabled: bool = True, 
                   output_dir: str = 'profiling_reports',
                   log_slow_operations: float = 1.0) -> PerformanceProfiler:
    """Setup and configure global profiler instance."""
    global profiler
    profiler.enabled = enabled
    profiler.output_dir = Path(output_dir)
    profiler.log_slow_operations = log_slow_operations
    
    if enabled:
        profiler.output_dir.mkdir(parents=True, exist_ok=True)
        
    return profiler