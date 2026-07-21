#!/usr/bin/env python3
"""
Inverse Transform: bring segmentation masks back to native patient space.

For each modality, applies the appropriate inverse chain:
  T1:   invert [t1_to_atlas]
  T1c:  invert [t1_to_atlas, t1c_to_t1]
  T2:   invert [t1_to_atlas, t2_to_t1]
  T2fl: invert [t1_to_atlas, t2fl_to_t1]

Usage:
    python 07_inverse_transform.py <input_dir> <output_dir> [options]
"""

import argparse
import logging
import sys
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from preprocessing_steps.registration import inverse_transform_subject_masks

logger = logging.getLogger(__name__)


def _plan_workers_for_inputs(input_files, requested, cpu_cap=None, budget_bytes=None):
    """Memory-aware worker count for this stage (see utils.resource_planner)."""
    from utils.resource_planner import plan_stage_workers
    return plan_stage_workers(
        "stage_07_inverse_transform", input_files, requested,
        cpu_cap=cpu_cap, budget_bytes=budget_bytes,
    )


def setup_logging(log_file: Optional[Path] = None, level: str = "INFO"):
    """Setup logging configuration."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level))

    if root_logger.handlers:
        root_logger.handlers = []

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, level))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root_logger.addHandler(fh)


def find_masks(segmentation_dir: Path, max_subjects: Optional[int] = None
               ) -> List[Tuple[Path, str, str]]:
    """
    Find all segmentation masks in BIDS structure.
    
    Returns:
        List of (mask_path, subject_id, session_id)
    """
    masks = []
    seen_subjects = set()

    for mask_path in sorted(segmentation_dir.rglob("*_segmask.nii.gz")):
        # Skip native masks (already processed)
        if "_native_" in mask_path.name:
            continue

        # Parse subject/session from directory parts (exclude filename)
        # segmentation/sub-001/ses-001/anat/sub-001_ses-001_segmask.nii.gz
        subject_id = None
        session_id = None
        for p in mask_path.parent.parts:
            if p.startswith("sub-"):
                subject_id = p
            elif p.startswith("ses-"):
                session_id = p

        if not subject_id or not session_id:
            logger.warning(f"Cannot parse subject/session from {mask_path}, skipping")
            continue

        if max_subjects and subject_id not in seen_subjects:
            if len(seen_subjects) >= max_subjects:
                continue
            seen_subjects.add(subject_id)

        masks.append((mask_path, subject_id, session_id))

    return masks


def process_one_mask(
    mask_path: Path,
    subject_id: str,
    session_id: str,
    nifti_dir: Path,
    transform_dir: Path,
    output_dir: Path,
    reference_modality: str,
    modalities: List[str],
    lesion_type: str,
) -> Dict:
    """Process a single mask — wrapper for parallel execution."""
    try:
        result = inverse_transform_subject_masks(
            mask_path=mask_path,
            nifti_dir=nifti_dir,
            transform_dir=transform_dir,
            output_dir=output_dir,
            subject_id=subject_id,
            session_id=session_id,
            reference_modality=reference_modality,
            modalities=modalities,
            lesion_type=lesion_type,
        )

        # Early error return (no per-modality results)
        if "error" in result and "success" in result:
            return {
                "subject_id": subject_id,
                "session_id": session_id,
                "success": False,
                "modalities_ok": 0,
                "modalities_total": len(modalities),
                "error": result["error"]
            }

        successful = sum(1 for r in result.values() if r.get("success"))
        total = len(modalities)

        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "success": successful > 0,
            "modalities_ok": successful,
            "modalities_total": total,
            "details": result
        }

    except Exception as e:
        logger.error(f"Failed {subject_id}/{session_id}: {e}")
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "success": False,
            "error": str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Inverse-transform segmentation masks to native patient space'
    )

    parser.add_argument("input_dir", type=Path,
                        help="Segmentation directory with atlas-space masks")
    parser.add_argument("output_dir", type=Path,
                        help="Output directory for native-space masks")

    parser.add_argument("--log_file", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--mode", choices=["sequential", "parallel"],
                        default="sequential")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip masks that already have native versions")

    parser.add_argument("--nifti-dir", type=Path, default=None,
                        help="NIfTI directory with native images (default: sibling of input)")
    parser.add_argument("--transform-dir", type=Path, default=None,
                        help="Transformations directory (default: sibling of input)")
    parser.add_argument("--reference-modality", type=str, default="t1",
                        help="Modality registered directly to atlas (default: t1)")
    parser.add_argument(
        "--lesion-type",
        type=str,
        default="glioblastoma",
        choices=["glioblastoma", "multiple_sclerosis"],
        help="Lesion type to process (determines output subfolder and inference service)"
    )

    args = parser.parse_args()

    setup_logging(args.log_file)

    # Infer sibling directories if not specified
    root_output = args.input_dir.parent  # segmentation -> root_output_dir
    nifti_dir = args.nifti_dir or (root_output / "nifti")
    transform_dir = args.transform_dir or (root_output / "transformations")

    # Validate paths
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    if not nifti_dir.exists():
        logger.error(f"NIfTI directory not found: {nifti_dir}")
        return 1

    if not transform_dir.exists():
        logger.error(f"Transformations directory not found: {transform_dir}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["t1", "t1c", "t2", "t2fl"]

    logger.info("=" * 70)
    logger.info("INVERSE TRANSFORM: MASKS TO NATIVE SPACE")
    logger.info("=" * 70)
    logger.info(f"Input (segmentation): {args.input_dir}")
    logger.info(f"Output:               {args.output_dir}")
    logger.info(f"NIfTI (native):       {nifti_dir}")
    logger.info(f"Transforms:           {transform_dir}")
    logger.info(f"Reference modality:   {args.reference_modality}")
    logger.info(f"Mode:                 {args.mode} (max workers: {args.workers})")

    # Find masks
    masks = find_masks(args.input_dir, args.max_subjects)

    if not masks:
        logger.error("No segmentation masks found")
        return 1

    logger.info(f"Found {len(masks)} mask(s) to process")

    total_found = len(masks)

    # Skip existing if requested
    if args.skip_existing:
        filtered = []
        for mask_path, subj, sess in masks:
            mask_stem = mask_path.name.replace("_segmask.nii.gz", "")
            # Native masks go into the same lesion_type subfolder as the source mask
            # (Stage 2 contract — keeps multi-model outputs from colliding)
            # Skip-existing check: look in the lesion_type subfolder
            out_subdir = args.output_dir / subj / sess / "anat" / args.lesion_type
            existing = list(out_subdir.glob(f"{mask_stem}_segmask_native_*.nii.gz"))
            if existing:
                logger.info(f"  Skipping {subj}/{sess}: {len(existing)} native masks exist")
            else:
                filtered.append((mask_path, subj, sess))
        masks = filtered
        logger.info(f"After skip-existing: {len(masks)} mask(s) to process")

    if not masks:
        logger.info("All masks already processed")
        return 0

    # Auto-tune parallelism.
    # apply_transforms uses ~4 ANTs threads per call (lighter than full registration).
    # Reserve 2 cores for the OS/IDE on workstation machines.
    _OS_RESERVED_CORES = 2
    cpu_count = os.cpu_count() or 4
    effective_cpu = max(4, cpu_count - _OS_RESERVED_CORES)
    if args.mode == "sequential":
        actual_workers = 1
        threads = effective_cpu
        logger.info(f"Workers: 1 | Threads: {threads} (all effective cores, sequential)")
    else:
        workers_by_tasks = min(args.workers, len(masks))
        workers_by_cpu = max(1, effective_cpu // 4)
        # Memory driver for stage 07 is the native reference image each mask is
        # warped into (loaded by inverse_transform_mask), not the small
        # atlas-space mask itself. Build those native reference paths from the
        # already-assembled work list and feed the CPU cap in alongside them so
        # the planner takes the min of ceiling/CPU/memory.
        _input_refs = [
            nifti_dir / subj / sess / "anat" / f"{subj}_{sess}_{args.reference_modality}.nii.gz"
            for (mask_path, subj, sess) in masks
        ]
        _plan = _plan_workers_for_inputs(
            _input_refs, requested=workers_by_tasks, cpu_cap=workers_by_cpu,
        )
        actual_workers = _plan.actual_workers
        threads = max(1, effective_cpu // actual_workers)
        logger.info(
            f"Workers: {actual_workers} — {_plan.reason} | "
            f"Threads per worker: {threads} | "
            f"Total threads: {actual_workers * threads}/{cpu_count} "
            f"(reserved {_OS_RESERVED_CORES} for OS)"
        )
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(threads)
    os.environ["ANTS_NUMBER_OF_THREADS"] = str(threads)

    # Performance monitoring
    monitor = None
    if args.benchmark:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start()

    start_time = time.time()
    successful = 0
    failed = 0
    all_results = []

    if args.mode == "sequential":
        for idx, (mask_path, subj, sess) in enumerate(masks, 1):
            logger.info(f"\n[{idx}/{len(masks)}] {subj}/{sess}")
            result = process_one_mask(
                mask_path, subj, sess,
                nifti_dir, transform_dir, args.output_dir,
                args.reference_modality, modalities,
                args.lesion_type,
            )
            all_results.append(result)
            if result["success"]:
                successful += 1
                logger.info(f"  ✓ {result['modalities_ok']}/{result['modalities_total']} modalities")
            else:
                failed += 1
                logger.error(f"  ✗ Failed: {result.get('error', 'unknown')}")
    else:
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
            future_map = {
                executor.submit(
                    process_one_mask,
                    mask_path, subj, sess,
                    nifti_dir, transform_dir, args.output_dir,
                    args.reference_modality, modalities,
                    args.lesion_type,
                ): (subj, sess)
                for mask_path, subj, sess in masks
            }

            for idx, future in enumerate(as_completed(future_map), 1):
                subj, sess = future_map[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    if result["success"]:
                        successful += 1
                        logger.info(f"✓ [{idx}/{len(masks)}] {subj}/{sess}")
                    else:
                        failed += 1
                        logger.error(f"✗ [{idx}/{len(masks)}] {subj}/{sess}")
                except Exception as e:
                    failed += 1
                    logger.error(f"✗ [{idx}/{len(masks)}] {subj}/{sess}: {e}")

    total_time = time.time() - start_time

    # Benchmark
    if args.benchmark and monitor:
        monitor.stop()
        system_metrics = monitor.get_metrics()

        experiment_id = f"inverse_{args.mode}_w{actual_workers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        benchmark_logger = BenchmarkLogger(args.results_dir)

        metrics = ExperimentMetrics(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            mode=args.mode,
            workers=actual_workers,
            total_series=len(masks),
            successful=successful,
            failed=failed,
            skipped=total_found - len(masks),
            total_time=total_time,
            time_per_series=total_time / len(masks) if masks else 0,
            throughput=len(masks) / total_time if total_time > 0 else 0,
            cpu_avg=system_metrics.get("cpu_avg"),
            cpu_max=system_metrics.get("cpu_max"),
            memory_avg_mb=system_metrics.get("memory_avg_mb"),
            memory_peak_mb=system_metrics.get("memory_peak_mb"),
        )
        benchmark_logger.log_metrics(metrics)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("INVERSE TRANSFORM COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total masks:  {len(masks)}")
    logger.info(f"Successful:   {successful}")
    logger.info(f"Failed:       {failed}")
    logger.info(f"Total time:   {total_time:.1f}s")
    if successful > 0:
        logger.info(f"Avg time:     {total_time / len(masks):.1f}s per mask")
    logger.info("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())