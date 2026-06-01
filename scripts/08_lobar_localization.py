#!/usr/bin/env python3
"""
Lobar Localization: determine anatomical location of lesions by brain lobe.

Overlays segmentation masks with a lobar atlas to compute per-lobe
volumes for each lesion class (NCR, ED, NET, ET).

Usage:
    python 08_lobar_localization.py <input_dir> <output_dir> [options]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml
from performance_monitor import PerformanceMonitor, BenchmarkLogger, ExperimentMetrics
from lobar_analysis import LobarAnalyzer

logger = logging.getLogger(__name__)

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


def resolve_atlas_path(
    lobar_config: Dict,
    preprocessing_config: Dict,
    project_root: Path
) -> Path:
    """
    Determine which lobar atlas file to use based on the template
    selected in preprocessing_config.
    
    Args:
        lobar_config: Parsed lobar_atlas_config.yaml
        preprocessing_config: Parsed preprocessing_config.yaml
        project_root: Project root for resolving relative paths
        
    Returns:
        Absolute path to the correct lobar atlas NIfTI
    """
    template_name = preprocessing_config.get("atlas", {}).get("name", "SRI24")
    templates = lobar_config.get("templates", {})

    if template_name not in templates:
        raise FileNotFoundError(
            f"No lobar atlas registered for template '{template_name}'. "
            f"Available: {list(templates.keys())}"
        )

    atlas_rel = templates[template_name]["file"]
    atlas_path = project_root / atlas_rel

    if not atlas_path.exists():
        raise FileNotFoundError(
            f"Lobar atlas file not found: {atlas_path}. "
            f"Run register_lobar_atlas.py to generate it."
        )

    logger.info(f"Template: {template_name} -> atlas: {atlas_path.name}")
    return atlas_path


def find_masks(segmentation_dir: Path, max_subjects: Optional[int] = None,
               lesion_type: Optional[str] = None) -> List[Tuple[Path, str, str]]:
    """Find all segmentation masks (excluding native masks)."""
    masks = []
    seen_subjects = set()
    # Known lesion-type subfolder names used in the new directory structure
    _known_lesion_types = {'glioblastoma', 'multiple_sclerosis'}

    for mask_path in sorted(segmentation_dir.rglob("*_segmask.nii.gz")):
        if "_native_" in mask_path.name:
            continue

        # If new-structure path contains a lesion-type folder, filter by it.
        # Old-structure paths (no lesion-type folder) are always included.
        if lesion_type:
            path_parts = set(mask_path.parent.parts)
            if path_parts & _known_lesion_types and lesion_type not in path_parts:
                continue

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
    output_dir: Path,
    atlas_path: Path,
    mapping_path: Path,
    seg_classes: Dict,
    lesion_type: str,
    analyzer: Optional["LobarAnalyzer"] = None,
) -> Dict:
    """Process a single mask — wrapper for parallel execution.

    analyzer: pre-loaded LobarAnalyzer to reuse in sequential mode;
              if None a new one is created (needed per-process in parallel mode).
    """
    try:
        if analyzer is None:
            analyzer = LobarAnalyzer(atlas_path, mapping_path, seg_classes)
        report = analyzer.analyze_mask(mask_path)

        if report is None:
            return {
                "subject_id": subject_id,
                "session_id": session_id,
                "success": False,
                "error": "Analysis returned None"
            }

        # Save report
        mask_stem = mask_path.name.replace("_segmask.nii.gz", "")
        report_name = f"{mask_stem}_lobar_report.json"
        # Lobar report goes into the same lesion_type subfolder as the source mask
        report_path = (output_dir / subject_id / session_id / "anat" / lesion_type / report_name)

        # Add patient/session info to report
        report["patient_id"] = subject_id
        report["session_id"] = session_id

        analyzer.save_report(report, report_path)

        affected = len(report.get("lobes", {}))
        return {
            "subject_id": subject_id,
            "session_id": session_id,
            "success": True,
            "affected_lobes": affected,
            "report_path": str(report_path)
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
        description='Lobar localization of brain lesions'
    )

    parser.add_argument("input_dir", type=Path,
                        help="Segmentation directory with atlas-space masks")
    parser.add_argument("output_dir", type=Path,
                        help="Output directory for localization reports")

    parser.add_argument("--log_file", type=Path, default=None)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--mode", choices=["sequential", "parallel"],
                        default="sequential")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--results_dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip masks that already have lobar reports")

    parser.add_argument("--config", type=Path, required=True,
                        help="Path to lobar_atlas_config.yaml")
    parser.add_argument("--preprocessing-config", type=Path, required=True,
                        help="Path to preprocessing_config.yaml (to read atlas.name)")
    parser.add_argument(
        "--lesion-type",
        type=str,
        default="glioblastoma",
        choices=["glioblastoma", "multiple_sclerosis"],
        help="Lesion type to process (determines output subfolder and inference service)"
    )

    args = parser.parse_args()

    setup_logging(args.log_file)

    # Validate inputs
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    if not args.config.exists():
        logger.error(f"Lobar config not found: {args.config}")
        return 1

    if not args.preprocessing_config.exists():
        logger.error(f"Preprocessing config not found: {args.preprocessing_config}")
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    with open(args.config, 'r') as f:
        lobar_config = yaml.safe_load(f)

    with open(args.preprocessing_config, 'r') as f:
        preprocessing_config = yaml.safe_load(f)

    # Resolve paths
    project_root = Path(args.config).resolve().parent.parent  # configs/ -> project root
    atlas_path = resolve_atlas_path(lobar_config, preprocessing_config, project_root)

    mapping_rel = lobar_config.get("mapping_file", "")
    mapping_path = project_root / mapping_rel
    if not mapping_path.exists():
        logger.error(f"Mapping file not found: {mapping_path}")
        return 1

    # Parse segmentation classes from config
    seg_classes = {}
    for label_str, cls_info in lobar_config.get("segmentation_classes", {}).items():
        seg_classes[int(label_str)] = cls_info

    logger.info("=" * 70)
    logger.info("LOBAR LOCALIZATION ANALYSIS")
    logger.info("=" * 70)
    logger.info(f"Input (segmentation): {args.input_dir}")
    logger.info(f"Output:               {args.output_dir}")
    logger.info(f"Atlas:                {atlas_path.name}")
    logger.info(f"Mapping:              {mapping_path.name}")
    logger.info(f"Mode:                 {args.mode}, workers: {args.workers}")

    # Find masks
    masks = find_masks(args.input_dir, args.max_subjects, args.lesion_type)

    if not masks:
        logger.error("No segmentation masks found")
        return 1

    logger.info(f"Found {len(masks)} mask(s) to process")

    # Skip existing
    skipped = 0
    if args.skip_existing:
        filtered = []
        for mask_path, subj, sess in masks:
            mask_stem = mask_path.name.replace("_segmask.nii.gz", "")
            report_path = args.output_dir / subj / sess / "anat" / args.lesion_type / f"{mask_stem}_lobar_report.json"
            if report_path.exists():
                skipped += 1
                logger.info(f"  Skipping {subj}/{sess}: report exists")
            else:
                filtered.append((mask_path, subj, sess))
        masks = filtered
        logger.info(f"After skip-existing: {len(masks)} mask(s) to process")

    if not masks:
        logger.info("All masks already processed")
        return 0

    # Performance monitoring
    monitor = None
    if args.benchmark:
        monitor = PerformanceMonitor(enabled=True)
        monitor.start()

    start_time = time.time()
    successful = 0
    failed = 0

    if args.mode == "sequential":
        # Load atlas once and reuse across all masks
        shared_analyzer = LobarAnalyzer(atlas_path, mapping_path, seg_classes)
        for idx, (mask_path, subj, sess) in enumerate(masks, 1):
            logger.info(f"\n[{idx}/{len(masks)}] {subj}/{sess}")
            result = process_one_mask(
                mask_path, subj, sess, args.output_dir,
                atlas_path, mapping_path, seg_classes,
                args.lesion_type,
                analyzer=shared_analyzer,
            )
            if result["success"]:
                successful += 1
                logger.info(f"  ✓ {result['affected_lobes']} lobes affected")
            else:
                failed += 1
                logger.error(f"  ✗ {result.get('error', 'unknown')}")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    process_one_mask,
                    mask_path, subj, sess, args.output_dir,
                    atlas_path, mapping_path, seg_classes,
                    args.lesion_type,
                ): (subj, sess)
                for mask_path, subj, sess in masks
            }

            for idx, future in enumerate(as_completed(future_map), 1):
                subj, sess = future_map[future]
                try:
                    result = future.result()
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
        benchmark_logger = BenchmarkLogger(args.results_dir)

        metrics = ExperimentMetrics(
            experiment_id=f"lobar_{args.mode}_w{args.workers}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            mode=args.mode,
            workers=args.workers if args.mode == "parallel" else 1,
            total_series=len(masks),
            successful=successful,
            failed=failed,
            skipped=skipped,
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
    logger.info("LOBAR LOCALIZATION COMPLETED")
    logger.info("=" * 70)
    logger.info(f"Total masks:  {len(masks)}")
    logger.info(f"Successful:   {successful}")
    logger.info(f"Failed:       {failed}")
    logger.info(f"Total time:   {total_time:.1f}s")
    logger.info("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())