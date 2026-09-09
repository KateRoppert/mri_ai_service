"""Build the benchmark's inputs: pipeline stages up to registration, skull on.

Every stripper in the comparison must see the *same* volume, otherwise the
numbers describe the preprocessing rather than the tools. So this runs the
production Stage 05 with skull stripping switched off and keeps the
registration-space output — brain still inside a skull, already in atlas
space, which is where the masks are compared.

Usage:
    python prepare_data.py --input-dir <bids_root> --output-dir <bench_data> \
        [--lesion-type glioblastoma] [--max-subjects N] [--dry-run]

Notes on Stage 05's actual CLI (verified 2026-09-09, differs from the June
draft): input and output directories are POSITIONAL, there is no
--transform-dir flag (Stage 05 derives it as output_dir.parent/transformations),
and the modality set comes from --lesion-type via lesion_types.yaml rather
than the config's `modalities` key.
"""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import yaml

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_CONFIG = PROJECT_ROOT / "configs" / "preprocessing_config.yaml"
STAGE05 = PROJECT_ROOT / "scripts" / "05_preprocessing.py"
MNI_TEMPLATE = PROJECT_ROOT / "data" / "templates" / "MNI152_T1_1mm.nii.gz"

sys.path.insert(0, str(PROJECT_ROOT))
from utils.nifti_integrity import is_complete_nifti  # noqa: E402


def build_benchmark_config(live_config: Path = LIVE_CONFIG) -> dict:
    """Production config with skull stripping off and the atlas pinned to MNI152.

    Derived from the live file rather than written from scratch: the benchmark
    is supposed to characterise the pipeline as it actually runs, and a
    hand-maintained copy drifts away from it silently.
    """
    with open(live_config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # The comparison happens in atlas space, and mni_mask is only meaningful
    # there. Production already uses MNI152_FSL, so this is a guard against a
    # future switch back to SRI24 rather than a change today.
    cfg.setdefault("atlas", {})["name"] = "MNI152_FSL"
    cfg["atlas"].setdefault("cache_dir", "data/templates")

    for step in cfg.get("steps", []):
        name = step.get("name")
        if name == "skull_stripping":
            # The whole point: benchmark inputs keep their skull.
            step["enabled"] = False
        elif name == "bias_correction":
            # Keep it a registration aid only, so the volume the strippers see
            # still has its original intensities.
            step.setdefault("params", {})["use_for_registration_only"] = True

    return cfg


def write_config(cfg: dict, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    logger.info("Benchmark config written to %s", dest)
    return dest


def build_stage05_command(
    input_dir: Path,
    output_dir: Path,
    config_path: Path,
    lesion_type: str,
    max_subjects: Optional[int],
) -> List[str]:
    """Argv for Stage 05. See the module docstring for why it looks like this.

    Paths are resolved to absolute: run_stage05 executes with cwd=scripts/, so
    anything relative from the caller would resolve against the wrong
    directory.
    """
    cmd = [
        sys.executable,
        str(STAGE05),
        str(Path(input_dir).resolve()),          # positional
        str(Path(output_dir).resolve()),         # positional
        "--config", str(Path(config_path).resolve()),
        "--lesion-type", lesion_type,
    ]
    if max_subjects is not None:
        cmd += ["--max-subjects", str(max_subjects)]
    return cmd


def subject_is_prepared(
    output_dir: Path, subject_id: str, session_id: str, modalities: List[str]
) -> bool:
    """Has this subject already been prepared, and is the output usable?

    Completeness rather than existence: a half-written volume left by an
    interrupted run passes exists() and would otherwise be benchmarked as if
    it were sound.
    """
    anat = output_dir / subject_id / session_id / "anat"
    if not anat.is_dir():
        return False
    found = False
    for mod in modalities:
        vol = anat / f"{subject_id}_{session_id}_{mod}.nii.gz"
        if not vol.exists():
            continue
        found = True
        if not is_complete_nifti(vol):
            logger.warning("Incomplete output, will redo: %s", vol)
            return False
    return found


def run_stage05(cmd: List[str]) -> None:
    """Run Stage 05 the way production does — from the project root.

    The config carries relative paths (`atlas.cache_dir: data/templates`) that
    only resolve there; pipeline_manager sets cwd=pipeline_root and the
    orchestrator inherits it.
    """
    logger.info("Running Stage 05: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path,
                    help="BIDS root with NIfTI volumes (Stage 03 output)")
    ap.add_argument("--output-dir", required=True, type=Path,
                    help="Where benchmark inputs are written")
    ap.add_argument("--lesion-type", default="glioblastoma",
                    choices=["glioblastoma", "multiple_sclerosis"],
                    help="Decides the modality set via lesion_types.yaml")
    ap.add_argument("--max-subjects", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write the config and print the command, run nothing")
    args = ap.parse_args()

    if not MNI_TEMPLATE.exists():
        raise SystemExit(f"MNI152 template missing: {MNI_TEMPLATE}")
    if not STAGE05.is_file():
        raise SystemExit(f"Stage 05 script missing: {STAGE05}")

    reg_space = args.output_dir / "registration_space"
    reg_space.mkdir(parents=True, exist_ok=True)

    cfg = build_benchmark_config()
    cfg_path = write_config(cfg, args.output_dir / "benchmark_preprocessing.yaml")

    cmd = build_stage05_command(
        input_dir=args.input_dir,
        output_dir=reg_space,
        config_path=cfg_path,
        lesion_type=args.lesion_type,
        max_subjects=args.max_subjects,
    )

    if args.dry_run:
        logger.info("Dry run — would execute: %s", " ".join(cmd))
        return

    run_stage05(cmd)
    # Stage 05 puts transformations beside its output directory.
    logger.info("Benchmark inputs (registration space, skull on): %s", reg_space)
    logger.info("Transformations: %s", reg_space.parent / "transformations")


if __name__ == "__main__":
    main()
