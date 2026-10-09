#!/usr/bin/env python3
"""
Run the skull-strip-order experiment (KI-062): every set, both orders, twice.

Each run is the full pipeline (stages 01-08) through orchestrator.py in a
throwaway brain-lesion-web:demo container that mounts this worktree's code,
joins the compose network (stage 06 reaches the segmentation services) and
has --memory 9g. The container has its own empty DB, so runs neither touch
the real BIDS numbering nor upload to Kappa.

Usage (host, from the worktree):
    python3 research/ss_order_experiment/run_experiment.py [--only SET] [--dry-run]

Resumable: a finished run leaves DONE in its output folder and is skipped.
Spec: docs/superpowers/specs/2026-10-07-skull-strip-before-registration-design.md
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

WORKTREE = Path(__file__).resolve().parents[2]
HOME_REPO = Path("/home/demob/mri_ai_service")           # same path in containers
DATA = HOME_REPO / "data"
EXP = HOME_REPO / "demo_workspace" / "exp_ss_order"
IMAGE = "brain-lesion-web:demo"
NETWORK = "mri_ai_service_default"


def pick(src: Path, n: int, seed: int) -> list:
    names = sorted(p.name for p in src.iterdir() if p.is_dir())
    return sorted(random.Random(seed).sample(names, n)) if n < len(names) else names


def sets() -> dict:
    dropbox = DATA / "dropbox_33" / "117-152"
    return {
        # name: (lesion_type, [patient dirs])
        "ms_sibbms": ("multiple_sclerosis",
                      [DATA / "SibBMS" / p for p in ("P000010", "P000063", "P000067")]),
        "ms_ms5": ("multiple_sclerosis", [DATA / "MS_5" / "P000915"]),
        # first six of the skull-stripping research's calibration sample (seed 20260909)
        "gbm_dropbox33": ("glioblastoma",
                          [dropbox / p for p in pick(dropbox, 12, 20260909)[:6]]),
        "gbm_bo": ("glioblastoma",
                   [DATA / "clinical_dicom" / "BO" / p
                    for p in pick(DATA / "clinical_dicom" / "BO", 6, 20261007)]),
        "gbm_upenn": ("glioblastoma",
                      sorted((DATA / "UPENN-GBM" / "dicom").iterdir())),
    }


def stage_input(name: str, patients: list) -> Path:
    staging = EXP / "input" / name
    staging.mkdir(parents=True, exist_ok=True)
    for p in patients:
        link = staging / p.name
        if not link.exists():
            link.symlink_to(p)        # /home/demob/... resolves on host and in containers
    return staging


def runtime_config(name: str, lesion: str, order: str, run: int, staging: Path) -> Path:
    out = EXP / order / f"run{run}" / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((WORKTREE / "pipeline_config.yaml").read_text(encoding="utf-8"))
    cfg["general"].update(root_input_dir=str(staging), root_output_dir=str(out),
                          lesion_type=lesion, max_subjects=None, skip_existing=True)
    cfg["general"].pop("benchmark_results_dir", None)
    pre = str(EXP / "configs" / f"preprocessing_{order}.yaml")
    for stage in cfg["stages"].values():
        # absolute, as backend/pipeline_manager does: relative script paths
        # resolve against the config's own folder
        if "script" in stage and not str(stage["script"]).startswith("/"):
            stage["script"] = f"/app/{stage['script']}"
        args = stage.get("args", {})
        args.pop("results_dir", None)
        args.pop("results-dir", None)
        if "preprocessing-config" in args:
            args["preprocessing-config"] = pre
    cfg["stages"]["stage_05_preprocessing"]["args"]["config"] = pre
    path = out / "pipeline_runtime.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


def docker_cmd(config: Path, network: str = NETWORK) -> list:
    w = str(WORKTREE)
    return [
        "docker", "run", "--rm", "--gpus", "all", "--memory", "9g", "--memory-swap", "9g",
        "--shm-size", "2g", "--network", network, "--entrypoint", "python",
        "-v", "/media/storage/home:/home",
        "-v", f"{w}/scripts:/app/scripts:ro", "-v", f"{w}/utils:/app/utils:ro",
        "-v", f"{w}/configs:/app/configs:ro", "-v", f"{w}/orchestrator.py:/app/orchestrator.py:ro",
        "-v", f"{HOME_REPO}/data/templates:/app/data/templates:ro",
        "-v", f"{HOME_REPO}/backend/data/hdbet_weights:/root/hd-bet_params",
        "-w", "/app", IMAGE, "orchestrator.py", "--config", str(config),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", help="run only these set names")
    ap.add_argument("--orders", default="after,before")
    ap.add_argument("--runs", default="1,2")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--network", default=NETWORK,
                    help="docker network whose service-ms-seg / service-gbm-seg to use")
    a = ap.parse_args()

    all_sets = sets()
    names = a.only or list(all_sets)
    for run in (int(r) for r in a.runs.split(",")):
        for order in a.orders.split(","):
            for name in names:
                lesion, patients = all_sets[name]
                staging = stage_input(name, patients)
                config = runtime_config(name, lesion, order, run, staging)
                done = config.parent / "DONE"
                if done.exists():
                    print(f"skip  {order}/run{run}/{name}", flush=True)
                    continue
                print(f"start {order}/run{run}/{name} ({len(patients)} patients, {lesion})", flush=True)
                if a.dry_run:
                    print("  " + " ".join(docker_cmd(config, a.network)))
                    continue
                t = time.time()
                rc = subprocess.run(docker_cmd(config, a.network), stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL).returncode
                minutes = (time.time() - t) / 60
                print(f"end   {order}/run{run}/{name} rc={rc} {minutes:.1f} min", flush=True)
                if rc == 0:
                    done.write_text(f"{minutes:.1f} min\n")


if __name__ == "__main__":
    sys.exit(main())
