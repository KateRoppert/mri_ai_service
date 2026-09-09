#!/usr/bin/env python3
"""
Run every skull stripper over the same subjects and record what each produced.

The cascade run answers "what does production do"; it cannot answer "was that
the right choice", because a cascade only ever shows you its winner. On the
dropbox_33 sample HD-BET won all twelve, so that run contains twelve positive
examples and no negatives at all — nothing to calibrate the gates against from
above, and nothing to compare against.

This runs the tools side by side on identical input, which yields three things
at once:

  * real negatives from real data (`mni_mask` is the known eye-leak case),
    so thresholds can be set from both sides instead of argued;
  * the material for a consensus reference — with no manual ground truth,
    the defensible reference for scoring tool X is the agreement of the
    *other* tools, and that needs every tool's mask;
  * the first rows of the benchmark table itself.

Input is the atlas-space skull-on volume rebuilt by `atlas_space.py`, which is
exactly what Stage 05 hands its stripper: registration has happened, stripping
has not. No Stage 05 re-run is needed.

Tool parameters come from the live `preprocessing_config.yaml` rather than
being restated here, so the comparison reflects the settings production
actually uses. Each tool reads the shared `tool_params` block and ignores keys
that are not its own — the same contract Stage 05 relies on.

Usage (inside the web container — that is where FSL, FreeSurfer and HD-BET
live):
    python run_tool_matrix.py \
        --run-dir    /path/to/run \
        --traces-dir /path/to/traces \
        --out-dir    /path/to/matrix
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atlas_space import DEFAULT_ATLAS, alignment_ok, head_paths, to_atlas_space  # noqa: E402
from scripts.preprocessing_steps.skull_stripping.dispatcher import STRIPPERS  # noqa: E402
from scripts.preprocessing_steps.skull_stripping.validation import validate_mask  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_TOOLS = ["hdbet", "synthstrip", "bet", "mni_mask"]

# Columns beyond the identifying ones. Fixed order so the CSV stays diffable
# across runs and a new metric cannot silently reshuffle existing columns.
METRIC_COLUMNS = [
    "mask_volume_ml", "lcc_ratio", "n_components", "n_components_kept",
    "hole_volume_ml", "n_holes", "edge_touch_ratio", "bbox_fill_ratio",
    "asymmetry", "leak_fraction", "leak_ml",
]


def load_tool_params(config_path: Path) -> dict:
    """The `tool_params` block Stage 05 passes to whichever tool is active."""
    config = yaml.safe_load(config_path.read_text())
    for step in config.get("steps", []):
        if step.get("name") == "skull_stripping":
            return dict(step.get("params", {}).get("tool_params", {}))
    return {}


def subjects_from_traces(traces_dir: Path):
    for trace_file in sorted(traces_dir.glob("*.json")):
        trace = json.loads(trace_file.read_text())
        yield trace["subject"], trace["session"]


def prepare_head(run_dir: Path, subject: str, session: str, modality: str,
                 atlas: Path, heads_dir: Path) -> Path:
    """Atlas-space skull-on volume for one subject, written once and reused by
    every tool. Reused across invocations too — this is the slow part only the
    first time."""
    out = heads_dir / f"{subject}_{session}_{modality}.nii.gz"
    if out.exists():
        return out
    raw, affine, mask = head_paths(run_dir, subject, session, modality)
    missing = [p for p in (raw, affine) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{subject}: missing {[p.name for p in missing]}")
    head = to_atlas_space(raw, affine, atlas, out_path=out)
    # The affine and the mask must come from the same Stage 05 run; if they do
    # not, every metric below is measured against the wrong anatomy. Cheap to
    # check, and silent corruption here would be very hard to notice later.
    if mask.exists():
        stray = alignment_ok(head, nib.load(str(mask)).get_fdata() > 0)
        if stray > 0.01:
            raise RuntimeError(
                f"{subject}: {stray:.1%} of the Stage 05 mask lands on "
                f"background after resampling — affine and mask disagree"
            )
    return out


def run_one(tool: str, head_path: Path, out_dir: Path, tool_params: dict,
            subject: str, session: str) -> dict:
    """One tool on one subject. Failures are recorded, not raised: a tool that
    cannot run on a subject is itself a benchmark result."""
    row = {"subject": subject, "session": session, "tool": tool}
    stripper = STRIPPERS[tool]()

    if not stripper.is_available():
        row.update(success=False, error="tool not available on this machine",
                   seconds=0.0)
        return row

    stripped = out_dir / tool / f"{subject}_{session}_stripped.nii.gz"
    mask = out_dir / tool / f"{subject}_{session}_mask.nii.gz"
    mask.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    result = stripper.strip(head_path, stripped, mask, tool_params)
    row["seconds"] = round(time.perf_counter() - start, 2)

    if not result.get("success"):
        row.update(success=False, error=(result.get("error") or "")[:300])
        return row

    row["success"] = True
    row["error"] = ""
    produced = Path(result.get("mask_path") or mask)
    # validate_mask returns a flat dict — metrics are spread into it, not
    # nested. (The cascade trace nests them; that shape is the trace's, not
    # this function's.)
    check = validate_mask(produced, image_path=head_path)
    row.update({k: check.get(k) for k in METRIC_COLUMNS})
    row["valid"] = check["valid"]
    row["reason"] = check["reason"]
    row["review_flags"] = ";".join(check["review_flags"])
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tools", nargs="+", default=DEFAULT_TOOLS)
    parser.add_argument("--modality", default="t1")
    parser.add_argument("--atlas", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--config", type=Path,
                        default=PROJECT_ROOT / "configs" / "preprocessing_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    unknown = [t for t in args.tools if t not in STRIPPERS]
    if unknown:
        raise SystemExit(f"Unknown tools: {unknown}. Known: {sorted(STRIPPERS)}")

    tool_params = load_tool_params(args.config)
    logger.info("Tool params from %s: %s", args.config.name, tool_params)

    heads_dir = args.out_dir / "heads"
    masks_dir = args.out_dir / "masks"
    rows = []

    for subject, session in subjects_from_traces(args.traces_dir):
        try:
            head = prepare_head(args.run_dir, subject, session, args.modality,
                                args.atlas, heads_dir)
        except (FileNotFoundError, RuntimeError) as exc:
            logger.warning("%s: skipped — %s", subject, exc)
            continue

        for tool in args.tools:
            row = run_one(tool, head, masks_dir, tool_params, subject, session)
            rows.append(row)
            if row.get("success"):
                # A hard rejection carries no review flags, so printing only
                # those would show the worst masks as unremarkable.
                verdict = row["review_flags"] or "ok"
                if not row["valid"]:
                    verdict = f"REJECTED: {row['reason']}"
                logger.info("%s %-10s %6.1f ml  holes %4.2f  leak %.3f %%  %5.1f s  %s",
                            subject, tool, row["mask_volume_ml"] or 0,
                            row["hole_volume_ml"] or 0,
                            (row["leak_fraction"] or 0) * 100, row["seconds"],
                            verdict)
            else:
                logger.warning("%s %-10s FAILED: %s", subject, tool, row["error"])

    csv_path = args.out_dir / "tool_matrix.csv"
    columns = (["subject", "session", "tool", "success", "error", "seconds"]
               + METRIC_COLUMNS + ["valid", "reason", "review_flags"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("%d rows -> %s", len(rows), csv_path)


if __name__ == "__main__":
    main()
