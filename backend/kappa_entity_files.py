"""Replacing, adding and deleting the files of a Kappa entity (API v2).

The rest of the service talks to v1 (`kappa_client.py`). v1 cannot replace
anything: its `replace_entity_file` appends, and uploading a file under a
name that already exists produces a second file with that name — verified
against a live throwaway dataset. v2 can, so this feature lives here.

This module is the ONLY place the v2 URL appears. Mixing two API versions
through one file would be something a reader finds out by accident; a
separate module makes the boundary a thing you have to walk through.
"""
import asyncio
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

KAPPA_DATA_URL_V2 = "https://kappa.nsu.ru:8061/data-micro-services/v2"

# v2 takes identity from the bearer token, so user_id/user_type_id are not in
# the URLs. They stay in the signatures so these calls read like every other
# Kappa function in the codebase.


def _headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def list_entity_files(
    token: str, user_id: int, user_type_id: int,
    dataset_id: int, entity_id: str,
) -> List[Dict[str, Any]]:
    """The entity's files as [{"fileId", "fileName"}]. Empty on any failure."""
    url = f"{KAPPA_DATA_URL_V2}/datasets/datasetEntities/{dataset_id}"
    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.get(url, headers=_headers(token))
        if not response.is_success:
            logger.warning("v2 list entities failed: status=%s, body=%s",
                           response.status_code, response.text[:300])
            return []
        for entity in response.json() or []:
            if entity.get("dsEntityId") == entity_id:
                return list(entity.get("files") or [])
        logger.warning("Entity %s not found in dataset %s", entity_id, dataset_id)
        return []
    except Exception as exc:  # noqa: BLE001 — callers decide, see module docstring
        logger.exception("v2 list entity files failed: %s", exc)
        return []


async def patch_file(
    token: str, user_id: int, user_type_id: int,
    dataset_id: int, entity_id: str, file_id: str, path: Path,
) -> bool:
    """Replace one file's contents in place. The file id is preserved."""
    url = f"{KAPPA_DATA_URL_V2}/datasets/{dataset_id}/{entity_id}/{file_id}"
    try:
        with open(path, "rb") as fh:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, read=300.0, write=300.0),
                verify=False,
            ) as client:
                response = await client.patch(
                    url, headers=_headers(token),
                    files={"updated_entity_file":
                           (path.name, fh, "application/gzip")},
                )
        if response.is_success:
            logger.info("Replaced %s in entity %s", path.name, entity_id)
            return True
        logger.warning("v2 patch failed for %s: status=%s, body=%s",
                       path.name, response.status_code, response.text[:300])
        return False
    except Exception as exc:  # noqa: BLE001
        logger.exception("v2 patch failed for %s: %s", path.name, exc)
        return False


async def add_files(
    token: str, user_id: int, user_type_id: int,
    dataset_id: int, entity_id: str, paths: List[Path],
) -> bool:
    """Add files the entity does not have yet."""
    if not paths:
        return True
    url = (f"{KAPPA_DATA_URL_V2}/datasets/datasetEntities/files"
           f"/{dataset_id}/{entity_id}")
    handles = []
    try:
        files = []
        for p in paths:
            fh = open(p, "rb")
            handles.append(fh)
            files.append(("files", (p.name, fh, "application/gzip")))
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, read=300.0, write=300.0),
            verify=False,
        ) as client:
            response = await client.post(url, headers=_headers(token), files=files)
        if response.is_success:
            logger.info("Added %d file(s) to entity %s", len(paths), entity_id)
            return True
        logger.warning("v2 add files failed: status=%s, body=%s",
                       response.status_code, response.text[:300])
        return False
    except Exception as exc:  # noqa: BLE001
        logger.exception("v2 add files failed: %s", exc)
        return False
    finally:
        for fh in handles:
            fh.close()


async def delete_files(
    token: str, user_id: int, user_type_id: int,
    dataset_id: int, file_ids: List[str],
) -> Optional[str]:
    """Enqueue deletion of specific files. Returns the job id to poll."""
    if not file_ids:
        return None
    url = f"{KAPPA_DATA_URL_V2}/datasets/datasetEntities/files"
    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            response = await client.request(
                "DELETE", url, headers=_headers(token), json=list(file_ids),
            )
        if response.is_success:
            return (response.json() or {}).get("jobId")
        logger.warning("v2 delete files failed: status=%s, body=%s",
                       response.status_code, response.text[:300])
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("v2 delete files failed: %s", exc)
        return None


async def wait_for_job(
    token: str, dataset_id: int, job_id: str,
    timeout: float = 60.0, interval: float = 2.0,
) -> str:
    """Poll an async job. Returns "succeeded", "failed" or "running".

    "running" means the wait ran out — we do not know how it ended. Reporting
    either outcome would be inventing a fact about someone else's system.
    """
    url = (f"{KAPPA_DATA_URL_V2}/datasets/datasetEntities"
           f"/bulk-mutation/jobs/{job_id}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
                response = await client.get(url, headers=_headers(token),
                                            params={"datasetId": dataset_id})
            if response.is_success:
                status = (response.json() or {}).get("status")
                if status in ("succeeded", "completed"):
                    return "succeeded"
                if status in ("failed", "cancelled"):
                    return "failed"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Polling job %s failed: %s", job_id, exc)
        await asyncio.sleep(interval)
    logger.warning("Job %s still running after %.0fs", job_id, timeout)
    return "running"


# Expert masks are versioned into the filename by the Slicer flow
# ("{base}_segmask_v{N}.nii.gz"). This pattern is the one app.py already uses
# to read those version numbers back — the same rule, not a second guess at
# it. They are not part of a recomputed set, and deleting them would throw
# away a specialist's work on the first replacement.
EXPERT_MASK_RE = re.compile(r"_segmask_v\d+\.nii\.gz$")


async def replace_entity_contents(
    token: str, user_id: int, user_type_id: int,
    dataset_id: int, entity_id: str, files: List[Path],
) -> Dict[str, Any]:
    """Make the entity hold exactly `files`, plus whatever expert masks it had.

    Matched by filename: same name is replaced in place (keeping its id), a
    name only in the new set is added, a name only in the entity is deleted.

    Raises ValueError on an empty `files`: that means discovery found
    nothing, and emptying the entity because of a failure upstream is the
    one outcome worth refusing outright.
    """
    if not files:
        # Reaching here with nothing means discovery failed, and carrying on
        # would delete the entity's contents on the strength of that failure.
        raise ValueError(
            f"Refusing to replace entity {entity_id} with an empty file set"
        )

    existing = await list_entity_files(
        token, user_id, user_type_id, dataset_id, entity_id)
    by_name = {f.get("fileName"): f.get("fileId") for f in existing}
    wanted = {p.name: p for p in files}

    patched = 0
    failed: List[str] = []

    for name, path in wanted.items():
        file_id = by_name.get(name)
        if file_id is None:
            continue
        if await patch_file(token, user_id, user_type_id,
                            dataset_id, entity_id, file_id, path):
            patched += 1
        else:
            failed.append(name)

    to_add = [p for name, p in wanted.items() if name not in by_name]
    added = 0
    if to_add:
        if await add_files(token, user_id, user_type_id,
                           dataset_id, entity_id, to_add):
            added = len(to_add)
        else:
            failed.extend(p.name for p in to_add)

    stale_ids = [
        file_id for name, file_id in by_name.items()
        if name not in wanted and not EXPERT_MASK_RE.search(name or "")
    ]
    delete_job = None
    if stale_ids:
        job_id = await delete_files(token, user_id, user_type_id,
                                    dataset_id, stale_ids)
        delete_job = (
            await wait_for_job(token, dataset_id, job_id) if job_id else "failed"
        )

    return {
        "patched": patched,
        "added": added,
        "deleted": len(stale_ids),
        "failed": failed,
        "delete_job": delete_job,
    }
