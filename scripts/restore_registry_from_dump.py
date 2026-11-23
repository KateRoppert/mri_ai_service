#!/usr/bin/env python3
"""Restore registry rows a backfill repointed at another dataset.

The 19.11.2026 backfill of run test500 into Kappa dataset 375 repointed the
rows of 319 studies that until then addressed datasets 347 and 348: the
registry kept one row per study_hash and register_patient overwrote its
kappa_dataset_id / kappa_entity_id (KI-084). Their old values were dumped
before the write, to backend/data/registry-before-375-*.json.

Now that a study can hold one row per dataset, the old rows can come back:
the row for 375 occupies a different pair, so re-registering the dumped
values creates a second row rather than overwriting anything.

One-off, not part of the schema migration: it depends on one particular dump
file and must be run deliberately. Dry run by default.

Usage:
    python scripts/restore_registry_from_dump.py backend/data/registry-before-375-<stamp>.json
    python scripts/restore_registry_from_dump.py <dump> --apply
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def restore(dump_path: Union[str, Path], apply: bool = False) -> dict:
    """Re-create the dumped rows. Returns a report of what it did."""
    from patient_registry import find_by_study_hash, register_patient

    rows = json.loads(Path(dump_path).read_text(encoding="utf-8"))
    report = {"in_dump": len(rows), "would_restore": 0, "restored": 0,
              "skipped": 0, "already_there": 0}

    for row in rows:
        study_hash = row.get("study_hash")
        dataset_id = row.get("kappa_dataset_id")
        if not study_hash or not dataset_id:
            # Nothing to address the row by — restoring it would be a guess.
            report["skipped"] += 1
            continue

        present = {r.get("kappa_dataset_id")
                   for r in find_by_study_hash(study_hash)}
        if dataset_id in present:
            report["already_there"] += 1
            continue

        report["would_restore"] += 1
        if not apply:
            continue

        register_patient(
            bids_id=row.get("bids_id") or "",
            study_hash=study_hash,
            original_patient_id=row.get("original_patient_id") or "",
            patient_name=row.get("patient_name") or "",
            scan_date=row.get("scan_date") or "",
            study_instance_uid=row.get("study_instance_uid") or "",
            kappa_entity_id=row.get("kappa_entity_id"),
            kappa_dataset_id=dataset_id,
            pipeline_run_id=row.get("pipeline_run_id") or "",
            lesion_type=row.get("lesion_type") or "",
            preprocessing_id=row.get("preprocessing_id") or "",
        )
        report["restored"] += 1

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="registry-before-*.json")
    parser.add_argument("--apply", action="store_true",
                        help="Without it the script only reports")
    args = parser.parse_args()

    report = restore(args.dump, apply=args.apply)
    print("готово:" if args.apply else "сухой прогон:")
    for key, value in report.items():
        print("  {:16s} {}".format(key, value))


if __name__ == "__main__":
    main()
