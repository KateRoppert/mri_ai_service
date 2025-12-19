#!/usr/bin/env python3
"""
Test script for modified performance_monitor.py
Place this file in the same directory as performance_monitor.py and run:
    python test_performance_monitor.py

This will test:
1. Creating CPU metrics (preprocessing style)
2. Creating GPU metrics (segmentation style)
3. Saving to CSV and loading back
4. Finding baseline times
5. Backward compatibility
"""

from pathlib import Path
from datetime import datetime
from performance_monitor import ExperimentMetrics, BenchmarkLogger


def test_cpu_metrics():
    """Test creating CPU-style metrics (preprocessing)"""
    print("=" * 70)
    print("TEST 1: Creating CPU Metrics (Preprocessing Style)")
    print("=" * 70)
    
    metrics = ExperimentMetrics(
        experiment_id="test_preproc_seq_w1",
        timestamp=datetime.now().isoformat(),
        stage="preprocessing",
        parallelism_type="cpu_workers",
        parallelism_level=1,
        mode="sequential",
        workers=1,
        total_series=10,
        successful=9,
        failed=1,
        skipped=0,
        total_time=100.5,
        time_per_series=10.05,
        throughput=0.0995,
        cpu_avg=45.2,
        cpu_max=78.5,
        memory_avg_mb=1024.5,
        memory_peak_mb=1536.8,
        speedup=1.0,
        efficiency=1.0
    )
    
    print(f"✓ Created CPU metrics: {metrics.experiment_id}")
    print(f"  Stage: {metrics.stage}")
    print(f"  Parallelism: {metrics.parallelism_type} = {metrics.parallelism_level}")
    print(f"  Legacy fields: mode={metrics.mode}, workers={metrics.workers}")
    print(f"  GPU fields: server={metrics.server_name}, gpus={metrics.gpu_count}")
    print()
    return metrics


def test_gpu_metrics():
    """Test creating GPU-style metrics (segmentation)"""
    print("=" * 70)
    print("TEST 2: Creating GPU Metrics (Segmentation Style)")
    print("=" * 70)
    
    metrics = ExperimentMetrics(
        experiment_id="test_seg_barguzin_gpu2_c4",
        timestamp=datetime.now().isoformat(),
        stage="segmentation",
        parallelism_type="gpu_concurrent",
        parallelism_level=4,
        server_name="barguzin",
        gpu_count=2,
        gpu_ids="0,1",
        total_series=20,
        successful=18,
        failed=2,
        skipped=0,
        total_time=150.8,
        time_per_series=7.54,
        throughput=0.1326,
        cpu_avg=12.5,
        cpu_max=25.8,
        memory_avg_mb=512.3,
        memory_peak_mb=768.9,
        speedup=2.5,
        efficiency=0.625
    )
    
    print(f"✓ Created GPU metrics: {metrics.experiment_id}")
    print(f"  Stage: {metrics.stage}")
    print(f"  Parallelism: {metrics.parallelism_type} = {metrics.parallelism_level}")
    print(f"  Server: {metrics.server_name}")
    print(f"  GPUs: {metrics.gpu_count} (IDs: {metrics.gpu_ids})")
    print(f"  Legacy fields: mode={metrics.mode}, workers={metrics.workers}")
    print()
    return metrics


def test_save_and_load():
    """Test saving to CSV and loading back"""
    print("=" * 70)
    print("TEST 3: Saving and Loading from CSV")
    print("=" * 70)
    
    # Create test directory
    test_dir = Path("./test_benchmark_results")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    logger = BenchmarkLogger(test_dir)
    print(f"✓ Created BenchmarkLogger at {test_dir}")
    
    # Create and save CPU baseline
    cpu_baseline = ExperimentMetrics(
        experiment_id="cpu_baseline_w1",
        timestamp=datetime.now().isoformat(),
        stage="preprocessing",
        parallelism_type="cpu_workers",
        parallelism_level=1,
        mode="sequential",
        workers=1,
        total_series=10,
        successful=10,
        failed=0,
        skipped=0,
        total_time=100.0,
        time_per_series=10.0,
        throughput=0.1,
        speedup=1.0,
        efficiency=1.0
    )
    logger.log_metrics(cpu_baseline)
    print(f"✓ Saved CPU baseline (1 worker)")
    
    # Create and save CPU parallel
    cpu_parallel = ExperimentMetrics(
        experiment_id="cpu_parallel_w4",
        timestamp=datetime.now().isoformat(),
        stage="preprocessing",
        parallelism_type="cpu_workers",
        parallelism_level=4,
        mode="parallel",
        workers=4,
        total_series=10,
        successful=10,
        failed=0,
        skipped=0,
        total_time=30.0,
        time_per_series=3.0,
        throughput=0.333,
        speedup=3.33,
        efficiency=0.833
    )
    logger.log_metrics(cpu_parallel)
    print(f"✓ Saved CPU parallel (4 workers)")
    
    # Create and save GPU baseline (barguzin)
    gpu_baseline_barguzin = ExperimentMetrics(
        experiment_id="gpu_baseline_barguzin_c1",
        timestamp=datetime.now().isoformat(),
        stage="segmentation",
        parallelism_type="gpu_concurrent",
        parallelism_level=1,
        server_name="barguzin",
        gpu_count=2,
        gpu_ids="0,1",
        total_series=20,
        successful=20,
        failed=0,
        skipped=0,
        total_time=200.0,
        time_per_series=10.0,
        throughput=0.1,
        speedup=1.0,
        efficiency=1.0
    )
    logger.log_metrics(gpu_baseline_barguzin)
    print(f"✓ Saved GPU baseline (barguzin, 1 concurrent)")
    
    # Create and save GPU parallel (barguzin)
    gpu_parallel_barguzin = ExperimentMetrics(
        experiment_id="gpu_parallel_barguzin_c8",
        timestamp=datetime.now().isoformat(),
        stage="segmentation",
        parallelism_type="gpu_concurrent",
        parallelism_level=8,
        server_name="barguzin",
        gpu_count=2,
        gpu_ids="0,1",
        total_series=20,
        successful=20,
        failed=0,
        skipped=0,
        total_time=60.0,
        time_per_series=3.0,
        throughput=0.333,
        speedup=3.33,
        efficiency=0.416
    )
    logger.log_metrics(gpu_parallel_barguzin)
    print(f"✓ Saved GPU parallel (barguzin, 8 concurrent)")
    
    # Save GPU baseline on different server (cube)
    gpu_baseline_cube = ExperimentMetrics(
        experiment_id="gpu_baseline_cube_c1",
        timestamp=datetime.now().isoformat(),
        stage="segmentation",
        parallelism_type="gpu_concurrent",
        parallelism_level=1,
        server_name="cube",
        gpu_count=4,
        gpu_ids="0,1,2,3",
        total_series=20,
        successful=20,
        failed=0,
        skipped=0,
        total_time=180.0,
        time_per_series=9.0,
        throughput=0.111,
        speedup=1.0,
        efficiency=1.0
    )
    logger.log_metrics(gpu_baseline_cube)
    print(f"✓ Saved GPU baseline (cube, 1 concurrent)")
    
    # Load back
    print()
    print("─" * 70)
    print("Loading metrics back from CSV...")
    loaded_metrics = logger.load_all_metrics()
    print(f"✓ Loaded {len(loaded_metrics)} metrics from CSV")
    print()
    
    # Display loaded metrics
    for i, m in enumerate(loaded_metrics, 1):
        print(f"  [{i}] {m.experiment_id}")
        print(f"      Stage: {m.stage}, Type: {m.parallelism_type}, Level: {m.parallelism_level}")
        if m.server_name:
            print(f"      Server: {m.server_name}, GPUs: {m.gpu_count}")
        if m.mode:
            print(f"      Legacy: mode={m.mode}, workers={m.workers}")
    
    print()
    return logger


def test_baseline_finding(logger):
    """Test finding baselines"""
    print("=" * 70)
    print("TEST 4: Finding Baseline Times")
    print("=" * 70)
    
    # Test CPU baseline
    cpu_baseline = logger.get_baseline_time(stage="preprocessing")
    if cpu_baseline:
        print(f"✓ CPU baseline (preprocessing): {cpu_baseline:.2f}s per series")
    else:
        print(f"✗ CPU baseline not found")
    
    # Test GPU baseline for barguzin
    gpu_baseline_barguzin = logger.get_baseline_time(stage="segmentation", server_name="barguzin")
    if gpu_baseline_barguzin:
        print(f"✓ GPU baseline (segmentation, barguzin): {gpu_baseline_barguzin:.2f}s per series")
    else:
        print(f"✗ GPU baseline for barguzin not found")
    
    # Test GPU baseline for cube
    gpu_baseline_cube = logger.get_baseline_time(stage="segmentation", server_name="cube")
    if gpu_baseline_cube:
        print(f"✓ GPU baseline (segmentation, cube): {gpu_baseline_cube:.2f}s per series")
    else:
        print(f"✗ GPU baseline for cube not found")
    
    # Test without filters
    any_baseline = logger.get_baseline_time()
    if any_baseline:
        print(f"✓ First baseline (no filters): {any_baseline:.2f}s per series")
    else:
        print(f"✗ No baseline found")
    
    print()


def test_csv_format():
    """Check CSV file format"""
    print("=" * 70)
    print("TEST 5: CSV File Format Check")
    print("=" * 70)
    
    csv_file = Path("./test_benchmark_results/metrics.csv")
    
    if not csv_file.exists():
        print(f"✗ CSV file not found: {csv_file}")
        return
    
    print(f"Reading CSV file: {csv_file}")
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        print(f"✓ Total lines: {len(lines)}")
        print(f"\nHeader:")
        print(f"  {lines[0].strip()}")
        
        if len(lines) > 1:
            print(f"\nFirst data row:")
            print(f"  {lines[1].strip()}")
        
        if len(lines) > 2:
            print(f"\nLast data row:")
            print(f"  {lines[-1].strip()}")
    
    print()


def test_to_dict():
    """Test to_dict() method"""
    print("=" * 70)
    print("TEST 6: to_dict() Method")
    print("=" * 70)
    
    metrics = ExperimentMetrics(
        experiment_id="test_dict",
        timestamp=datetime.now().isoformat(),
        stage="segmentation",
        parallelism_type="gpu_concurrent",
        parallelism_level=4,
        server_name="barguzin",
        gpu_count=2,
        gpu_ids="0,1",
        total_series=10,
        successful=10,
        failed=0,
        skipped=0,
        total_time=50.0,
        time_per_series=5.0,
        throughput=0.2
    )
    
    data = metrics.to_dict()
    print(f"✓ Converted to dict with {len(data)} keys")
    print(f"  Keys: {', '.join(sorted(data.keys())[:5])}...")
    
    # Check all new fields are present
    new_fields = ['stage', 'parallelism_type', 'parallelism_level', 
                  'server_name', 'gpu_count', 'gpu_ids']
    missing = [f for f in new_fields if f not in data]
    
    if missing:
        print(f"✗ Missing fields: {', '.join(missing)}")
    else:
        print(f"✓ All new fields present in dict")
    
    print()


def main():
    """Run all tests"""
    print()
    print("=" * 70)
    print("TESTING MODIFIED PERFORMANCE_MONITOR.PY")
    print("=" * 70)
    print()
    
    try:
        # Test 1: CPU metrics
        test_cpu_metrics()
        
        # Test 2: GPU metrics
        test_gpu_metrics()
        
        # Test 3: Save and load
        logger = test_save_and_load()
        
        # Test 4: Baseline finding
        test_baseline_finding(logger)
        
        # Test 5: CSV format
        test_csv_format()
        
        # Test 6: to_dict method
        test_to_dict()
        
        print("=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)
        print()
        print(f"Test results saved in: ./test_benchmark_results/")
        print(f"  - metrics.csv (main CSV file)")
        print(f"  - *.json (individual experiment files)")
        print()
        print("You can check the CSV file to see the new columns:")
        print("  - stage, parallelism_type, parallelism_level")
        print("  - server_name, gpu_count, gpu_ids")
        print()
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ TEST FAILED!")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())