"""Run the production cascade over a sample of patients, collecting traces.

Purpose is calibration, not production: the gates were set from a handful of
masks, so we need many decisions with their numbers attached before trusting
any threshold — and before deciding whether cascade output is good enough to
serve as reference masks for the benchmark.

Runs stages 01 (DICOM to BIDS), 03 (NIfTI) and 05 (preprocessing + cascade).
Stages 02, 04 and 06-08 are off: quality assessment is expensive and
segmentation is irrelevant to how a brain mask is made.

Usage:
    python run_cascade_sample.py --input-dir data/dropbox_33/117-152 \
        --output-dir /home/ubuntu/ss_experiment/dropbox33 \
        [--sample 12] [--seed 20260909] [--lesion-type glioblastoma]

Must run where FSL, ANTs, HD-BET and SynthStrip live — i.e. inside the web
container, not the host venv. The script only builds configs and invokes the
orchestrator; see DEVLOG for the docker exec wrapper.
"""

import argparse
import copy
import logging
import random
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG = PROJECT_ROOT / "pipeline_config.yaml"
PREPROCESSING_CONFIG = PROJECT_ROOT / "configs" / "preprocessing_config.yaml"
ORCHESTRATOR = PROJECT_ROOT / "orchestrator.py"

# Stages that say something about how a brain mask is produced.
STAGES_WANTED = {"01", "03", "05"}


def pick_sample(input_dir: Path, n: int, seed: int) -> list:
    """A reproducible random subset of patient directories.

    Random rather than first-N: the first patients in a dropbox are not a
    sample, they are whoever was uploaded first, and defects may well
    correlate with acquisition batch.
    """
    patients = sorted(p.name for p in input_dir.iterdir() if p.is_dir())
    if n >= len(patients):
        return patients
    return sorted(random.Random(seed).sample(patients, n))


def stage_all_patients(input_dir: Path, patients: list, staging: Path) -> Path:
    """Symlink the chosen patients into a directory the pipeline can consume.

    Symlinks rather than copies: these are gigabytes of DICOM per patient.
    """
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for name in patients:
        (staging / name).symlink_to((input_dir / name).resolve())
    logger.info("Staged %d patients in %s", len(patients), staging)
    return staging


def build_preprocessing_config(trace_dir: Path, dest: Path) -> Path:
    """Production preprocessing config plus cascade tracing."""
    cfg = yaml.safe_load(PREPROCESSING_CONFIG.read_text(encoding="utf-8"))
    for step in cfg.get("steps", []):
        if step.get("name") == "skull_stripping":
            params = step.setdefault("params", {})
            validation = dict(params.get("validation") or {})
            validation["trace_dir"] = str(trace_dir)
            params["validation"] = validation
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return dest


def build_pipeline_config(input_dir: Path, output_dir: Path,
                          preprocessing_cfg: Path, lesion_type: str,
                          dest: Path) -> Path:
    """Master config with only the stages this experiment needs."""
    cfg = yaml.safe_load(PIPELINE_CONFIG.read_text(encoding="utf-8"))
    cfg["general"]["root_input_dir"] = str(input_dir)
    cfg["general"]["root_output_dir"] = str(output_dir)
    cfg["general"]["max_subjects"] = None
    cfg["general"]["lesion_type"] = lesion_type

    for key, stage in list(cfg.get("stages", {}).items()):
        if not isinstance(stage, dict):
            continue
        number = "".join(ch for ch in str(key) if ch.isdigit())[:2]
        stage["enabled"] = number in STAGES_WANTED
        # Stage scripts are resolved relative to the CONFIG's directory
        # (utils/config_loader.py), and this config does not live in the repo,
        # so relative paths would be looked up beside it.
        script = stage.get("script")
        if script and not Path(script).is_absolute():
            stage["script"] = str((PROJECT_ROOT / script).resolve())
        if number == "05":
            stage.setdefault("params", {})["config"] = str(preprocessing_cfg)

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--sample", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--lesion-type", default="glioblastoma",
                    choices=["glioblastoma", "multiple_sclerosis"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    patients = pick_sample(args.input_dir, args.sample, args.seed)
    logger.info("Sample (seed=%d): %s", args.seed, ", ".join(patients))

    staging = args.output_dir / "input_sample"
    trace_dir = args.output_dir / "traces"
    prep_cfg = build_preprocessing_config(trace_dir,
                                          args.output_dir / "preprocessing.yaml")
    pipe_cfg = build_pipeline_config(staging, args.output_dir / "run",
                                     prep_cfg, args.lesion_type,
                                     args.output_dir / "pipeline.yaml")

    if args.dry_run:
        logger.info("Dry run — configs written to %s", args.output_dir)
        return

    stage_all_patients(args.input_dir, patients, staging)
    cmd = [sys.executable, str(ORCHESTRATOR), "--config", str(pipe_cfg)]
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    logger.info("Traces: %s", trace_dir)


if __name__ == "__main__":
    main()
