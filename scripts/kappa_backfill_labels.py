#!/usr/bin/env python3
"""
Bring a Kappa dataset's entities up to what Kappa now needs for statuses
3 (Labeled) and 5 (Verified) — KI-075.

Since October 2026 Kappa refuses both unless the entity has
dsEntityInfo.labels and its main mask is an `output` file. Entities uploaded
before the fix have neither, so they stay at status 2 and the validation tab
cannot verify them.

Per entity, only what is missing:
  - labels: written into dsEntityInfo (classes of the lesion type's mask,
    configs/lesion_types.yaml kappa_labels);
  - main mask (`*_segmask.nii.gz`) as output: downloaded from Kappa and
    re-uploaded in place with file_category=output (same file id) — works for
    entities uploaded from any machine;
  - --drop-stubs: delete the `empty.json` stubs old status changes left (KI-036);
  - status 2 -> 3 once the above is in place.

Dry run by default: prints what it would do and changes nothing.

Run inside the web container (needs a live Kappa session of the user):
    docker exec -w /app mri_ai_service-web-1 \\
        python scripts/kappa_backfill_labels.py --dataset 353 --user 26 [--apply] [--drop-stubs]
"""
import argparse
import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import kappa_client  # noqa: E402
import kappa_entity_files as kef  # noqa: E402
from utils.config_loader import load_lesion_type_config  # noqa: E402


def _info(entity):
    info = entity.get("dsEntityInfo") or {}
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except ValueError:
            info = {}
    return info if isinstance(info, dict) else {}


def _labels_for(info, default_lesion):
    lesion = info.get("lesion_type") or default_lesion
    try:
        labels = load_lesion_type_config(lesion).get("kappa_labels")
    except KeyError:
        labels = None
    return list(labels) if labels else [lesion] if lesion else None


def plan_entity(entity, default_lesion=None, drop_stubs=False):
    """What this entity lacks. Pure: no I/O."""
    info = _info(entity)
    files = entity.get("files") or []
    plan = {"entity_id": entity.get("dsEntityId"), "name": entity.get("dsEntityName"),
            "status": entity.get("dsEntityStatus")}
    if not info.get("labels"):
        plan["labels"] = _labels_for(info, default_lesion)
    masks = [f for f in files if kef.file_category(f.get("fileName") or "") == "output"
             and (f.get("fileCategory") or "").lower() != "output"]
    if masks:
        plan["masks_to_output"] = [(f["fileId"], f["fileName"]) for f in masks]
    if not any(kef.file_category(f.get("fileName") or "") == "output" for f in files):
        plan["no_mask"] = True          # cannot become Labeled: nothing to mark
    if drop_stubs:
        stubs = [f["fileId"] for f in files if f.get("fileName") == "empty.json"]
        if stubs:
            plan["stubs"] = stubs
    plan["to_labeled"] = plan["status"] == 2 and not plan.get("no_mask")
    return plan


def _needs_work(plan):
    return any(k in plan for k in ("labels", "masks_to_output", "stubs")) or plan["to_labeled"]


async def apply_plan(plan, entity, dataset_id, s):
    tok, uid, ut = s["kappa_token"], s["user_id"], s["user_type_id"]
    eid = plan["entity_id"]
    done, errors = [], []

    if plan.get("labels"):
        info = {**_info(entity), "labels": plan["labels"]}
        if await kef.update_entity_info(tok, uid, ut, dataset_id, eid, info,
                                        remark="KI-075: labels"):
            done.append("labels")
        else:
            errors.append("labels")

    for file_id, name in plan.get("masks_to_output", []):
        data = await kappa_client.download_entity_file(tok, uid, ut, dataset_id, file_id)
        if data is None:
            errors.append(f"download {name}")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            if await kef.patch_file(tok, uid, ut, dataset_id, eid, file_id, path):
                done.append(f"{name} -> output")
            else:
                errors.append(f"patch {name}")

    if plan.get("stubs"):
        job = await kef.delete_files(tok, uid, ut, dataset_id, plan["stubs"])
        verdict = await kef.wait_for_job(tok, dataset_id, job) if job else "failed"
        (done if verdict == "succeeded" else errors).append(
            f"{len(plan['stubs'])} empty.json: {verdict}")

    if plan["to_labeled"] and not errors:
        ok, msg = await kappa_client.set_entity_status(tok, uid, ut, dataset_id, eid, 3,
                                                       remark="KI-075 backfill")
        (done if ok else errors).append("status 3" if ok else f"status 3: {msg}")
    return done, errors


async def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", type=int, required=True)
    ap.add_argument("--user", type=int, required=True, help="Kappa user id (live session)")
    ap.add_argument("--lesion-type", default=None,
                    help="for entities whose info has no lesion_type")
    ap.add_argument("--apply", action="store_true", help="change Kappa (default: dry run)")
    ap.add_argument("--drop-stubs", action="store_true", help="also delete empty.json stubs")
    args = ap.parse_args(argv)

    from kappa_auth import find_live_session_for_user
    s = find_live_session_for_user(args.user)
    if not s:
        sys.exit(f"No live Kappa session for user {args.user} — log in first")

    entities = None
    for _ in range(4):
        entities = await kappa_client.get_dataset_entities(
            token=s["kappa_token"], user_id=s["user_id"],
            user_type_id=s["user_type_id"], dataset_id=args.dataset)
        if entities is not None:
            break
    if entities is None:
        sys.exit(f"Kappa did not return dataset {args.dataset}")

    by_id = {e["dsEntityId"]: e for e in entities}
    plans = [plan_entity(e, args.lesion_type, args.drop_stubs) for e in entities]
    todo = [p for p in plans if _needs_work(p)]
    print(f"dataset {args.dataset}: {len(entities)} entities, {len(todo)} need work"
          f"{'' if args.apply else ' (dry run)'}")
    for p in todo:
        parts = []
        if p.get("labels"):
            parts.append(f"labels={p['labels']}")
        if p.get("masks_to_output"):
            parts.append(f"{len(p['masks_to_output'])} mask(s)->output")
        if p.get("stubs"):
            parts.append(f"drop {len(p['stubs'])} empty.json")
        if p["to_labeled"]:
            parts.append("status 2->3")
        if p.get("no_mask"):
            parts.append("NO MASK — stays at its status")
        line = f"  {p['name']} (status {p['status']}): {', '.join(parts)}"
        if args.apply:
            done, errors = await apply_plan(p, by_id[p["entity_id"]], args.dataset, s)
            line += f"  -> done {done}" + (f", FAILED {errors}" if errors else "")
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
