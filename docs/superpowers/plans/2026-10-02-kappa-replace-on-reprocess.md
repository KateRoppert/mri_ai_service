# Replacing a Reprocessed Patient in Kappa — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A patient reprocessed after the doctor corrected their modality set can have their Kappa entity's contents replaced, on explicit confirmation, without losing expert masks.

**Architecture:** A new v2-only module replaces an entity's files by diffing filenames — patch what matches, add what is new, delete what is gone, never touch expert masks. The uploader reports such sessions as `supersedes` instead of silently skipping them as duplicates, which routes them through the `needs_attention` machinery that already exists.

**Tech Stack:** Python 3.12, FastAPI, httpx, pytest, React 19 + Ant Design.

**Spec:** `docs/superpowers/specs/2026-10-02-kappa-replace-on-reprocess-design.md`

## Global Constraints

- **The v2 base URL is `https://kappa.nsu.ru:8061/data-micro-services/v2`.** It lives in the new module only. `kappa_client.py` stays on v1 and is not modified by this plan.
- **v2 URLs carry no `{user_id}/{user_type_id}`** — identity comes from the bearer token. Keep both as Python parameters anyway, so the call sites look like every other Kappa function.
- **Expert masks are never deleted.** Any file matching `*_segmask_v<N>.nii.gz` is exempt from the "absent from the new set → delete" rule.
- **Deletion is asynchronous**: `202` + `jobId`, polled at `GET /datasets/datasetEntities/bulk-mutation/jobs/{jobId}?datasetId=` **every 2 seconds for at most 60**. On timeout report the job as still running — never as succeeded or failed.
- **Never re-read the entity listing to confirm a delete.** It was observed returning a deleted entity for about a second after its job reported `succeeded`.
- **A partial replacement is reported, not hidden.** The result says which files were replaced and which were not; the session stays `needs_attention` and the action may simply be repeated.
- **No Kappa call in the test suite.** Every one is mocked.
- **No new delivery status.** `supersedes_kappa` is a *reason* inside the existing `needs_attention`.
- Code comments in English, operator-facing strings in Russian.
- Commit style: conventional commits, ending with:
  `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`

## File Structure

| File | Responsibility |
|---|---|
| `backend/kappa_entity_files.py` (new) | Every v2 call this feature needs, and the filename diff that drives them. The only place the v2 URL appears. |
| `backend/kappa_uploader.py` | Reports a superseding session as `supersedes` instead of a skipped duplicate, and holds the one definition of which files make up an entity. |
| `backend/kappa_delivery.py` | Maps `supersedes` to `needs_attention` / `supersedes_kappa`. |
| `backend/kappa_run_log.py` | A Russian label for the new reason. |
| `backend/database.py` | `pipeline_runs.reprocessed_sessions` and its migration. |
| `backend/app.py` | Requeue records what it purged; the replace endpoint; the expert-mask count on blocked sessions. |
| `backend/pipeline_monitor.py`, `backend/kappa_delivery_worker.py` | Pass the run's superseding sessions to the uploader. |
| `backend/models.py` | `ReplaceEntityResponse`; the new reason and the expert-mask count on `KappaBlockedSession`. |
| `backend/registry_models.py` | One docstring the probe disproved. |
| `frontend/src/components/PipelineHistory.jsx`, `frontend/src/services/api.js` | The «Заменить в Kappa» action and its confirmation. |

Dependency order is strict, 1 through 10. Tasks 9 (a docstring the probe
disproved) and 10 (verification) are the only ones that depend on nothing
new.

## Before You Start

```bash
cd /home/ubuntu/mri_ai_service
git checkout feat/kappa-replace-on-reprocess   # already exists, from main
source venv/bin/activate
python -m pytest backend/ -q    # record the number; it must not drop
```

**Do not `git add -A`.** `configs/kappa_datasets.yaml` and `pipeline_config.yaml` are intentionally dirty and must stay uncommitted. Stage by explicit path.

---

### Task 1: The v2 calls

**Files:**
- Create: `backend/kappa_entity_files.py`
- Test: `backend/test_kappa_entity_files.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `KAPPA_DATA_URL_V2: str`
  - `async list_entity_files(token, user_id, user_type_id, dataset_id, entity_id) -> list[dict]` — each `{"fileId", "fileName"}`
  - `async patch_file(token, user_id, user_type_id, dataset_id, entity_id, file_id, path) -> bool`
  - `async add_files(token, user_id, user_type_id, dataset_id, entity_id, paths) -> bool`
  - `async delete_files(token, user_id, user_type_id, dataset_id, file_ids) -> str | None` — the job id
  - `async wait_for_job(token, dataset_id, job_id, timeout=60.0, interval=2.0) -> str` — `"succeeded" | "failed" | "running"`

**Background:** every function here mirrors the shape used throughout
`kappa_client.py` — `httpx.AsyncClient(timeout=..., verify=False)`, log and
return a falsy value on failure rather than raising. Read
`kappa_client.get_dataset_entities` (`backend/kappa_client.py:452`) first and
follow it.

- [ ] **Step 1: Write the failing test**

Create `backend/test_kappa_entity_files.py`:

```python
"""The v2 calls that can actually replace a file.

v1's replace_entity_file appends despite its name — verified live. These
exist because v2 can genuinely replace, delete and add.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import kappa_entity_files as kef


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


class _Client:
    """Records every call so a test can assert on the verb and the URL."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def _record(self, verb, url, **kw):
        self.calls.append((verb, url, kw))
        r = self.responses.pop(0)
        return r

    async def get(self, url, **kw):
        return await self._record("GET", url, **kw)

    async def patch(self, url, **kw):
        return await self._record("PATCH", url, **kw)

    async def post(self, url, **kw):
        return await self._record("POST", url, **kw)

    async def request(self, verb, url, **kw):
        return await self._record(verb, url, **kw)


def _install(monkeypatch, responses):
    client = _Client(responses)
    monkeypatch.setattr(kef.httpx, "AsyncClient", lambda **kw: client)
    return client


@pytest.mark.asyncio
async def test_lists_files_of_an_entity(monkeypatch):
    payload = [{"dsEntityId": "e1",
                "files": [{"fileId": "f1", "fileName": "a.nii.gz"},
                          {"fileId": "f2", "fileName": "b.nii.gz"}]}]
    client = _install(monkeypatch, [_Resp(200, payload)])

    files = await kef.list_entity_files("tok", 26, 1, 355, "e1")

    assert [f["fileName"] for f in files] == ["a.nii.gz", "b.nii.gz"]
    verb, url, _ = client.calls[0]
    assert verb == "GET"
    # v2 URLs carry no user_id/user_type_id — identity is in the token.
    assert url.endswith("/datasets/datasetEntities/355")
    assert "/26/1/" not in url


@pytest.mark.asyncio
async def test_patch_targets_the_file_id(monkeypatch, tmp_path):
    f = tmp_path / "a.nii.gz"
    f.write_bytes(b"new")
    client = _install(monkeypatch, [_Resp(200, None, "File updated successfully.")])

    assert await kef.patch_file("tok", 26, 1, 355, "e1", "f1", f) is True

    verb, url, kw = client.calls[0]
    assert verb == "PATCH"
    assert url.endswith("/datasets/355/e1/f1")
    assert "updated_entity_file" in kw["files"]


@pytest.mark.asyncio
async def test_a_refused_patch_is_false_not_an_exception(monkeypatch, tmp_path):
    """Callers decide what a failure means; they cannot do that through a
    traceback thrown from inside a loop over files."""
    f = tmp_path / "a.nii.gz"
    f.write_bytes(b"new")
    _install(monkeypatch, [_Resp(403, None, "forbidden")])

    assert await kef.patch_file("tok", 26, 1, 355, "e1", "f1", f) is False


@pytest.mark.asyncio
async def test_delete_returns_the_job_id(monkeypatch):
    client = _install(monkeypatch, [_Resp(202, {"jobId": "job-7"})])

    job = await kef.delete_files("tok", 26, 1, 355, ["f1", "f2"])

    assert job == "job-7"
    verb, url, kw = client.calls[0]
    assert verb == "DELETE"
    assert url.endswith("/datasets/datasetEntities/files")
    assert kw["json"] == ["f1", "f2"]


@pytest.mark.asyncio
async def test_wait_for_job_reports_success(monkeypatch):
    _install(monkeypatch, [
        _Resp(200, {"status": "queued"}),
        _Resp(200, {"status": "succeeded"}),
    ])
    monkeypatch.setattr(kef.asyncio, "sleep", _no_sleep)

    assert await kef.wait_for_job("tok", 355, "job-7") == "succeeded"


@pytest.mark.asyncio
async def test_wait_for_job_reports_failure(monkeypatch):
    _install(monkeypatch, [_Resp(200, {"status": "failed"})])
    monkeypatch.setattr(kef.asyncio, "sleep", _no_sleep)

    assert await kef.wait_for_job("tok", 355, "job-7") == "failed"


@pytest.mark.asyncio
async def test_an_empty_set_issues_no_call(monkeypatch):
    """A caller with nothing to add or delete must not produce a request.
    An empty DELETE body is the kind of thing a server is free to read as
    "all of them"."""
    client = _install(monkeypatch, [])

    assert await kef.add_files("tok", 26, 1, 355, "e1", []) is True
    assert await kef.delete_files("tok", 26, 1, 355, []) is None
    assert client.calls == []


@pytest.mark.asyncio
async def test_wait_for_job_gives_up_without_claiming_an_outcome(monkeypatch):
    """A timeout means we do not know. Reporting success or failure would be
    inventing a fact about someone else's system."""
    _install(monkeypatch, [_Resp(200, {"status": "queued"})] * 50)
    monkeypatch.setattr(kef.asyncio, "sleep", _no_sleep)

    assert await kef.wait_for_job("tok", 355, "job-7", timeout=4.0) == "running"


async def _no_sleep(_seconds):
    return None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_entity_files.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kappa_entity_files'`.

- [ ] **Step 3: Write the module**

Create `backend/kappa_entity_files.py`:

```python
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
        logger.warning("Entity %s not found in dataset %d", entity_id, dataset_id)
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
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0,
                                                               write=300.0),
                                         verify=False) as client:
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
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=300.0,
                                                           write=300.0),
                                     verify=False) as client:
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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_kappa_entity_files.py -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_entity_files.py backend/test_kappa_entity_files.py
git commit -m "feat(kappa): v2 calls that can actually replace a file

v1 cannot. Its replace_entity_file appends despite the name, and a file
uploaded under an existing name becomes a second file with that name —
verified against a live throwaway dataset. v2's PATCH replaces in place and
keeps the file id.

Kept to its own module so the version boundary is something you walk
through rather than discover. A timeout while polling an async job reports
'running', never an outcome we did not observe.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Replacing an entity's contents

**Files:**
- Modify: `backend/kappa_entity_files.py`
- Test: `backend/test_kappa_entity_files.py` (extend)

**Interfaces:**
- Consumes: `list_entity_files`, `patch_file`, `add_files`, `delete_files`, `wait_for_job`
- Produces:
  - `EXPERT_MASK_RE: re.Pattern` — matches `*_segmask_v<N>.nii.gz`
  - `async replace_entity_contents(token, user_id, user_type_id, dataset_id, entity_id, files: list[Path]) -> dict` with keys `patched`, `added`, `deleted`, `failed` (list of filenames), `delete_job` (`"succeeded" | "failed" | "running" | None`). Raises `ValueError` on an empty `files`.

- [ ] **Step 1: Write the failing test**

Append to `backend/test_kappa_entity_files.py`:

```python
class _Recorder:
    """Stands in for the whole v2 surface so the diff can be tested alone."""

    def __init__(self, existing):
        self.existing = existing
        self.patched, self.added, self.deleted = [], [], []

    async def list_entity_files(self, *a, **k):
        return self.existing

    async def patch_file(self, token, uid, utid, ds, eid, file_id, path):
        self.patched.append((file_id, path.name))
        return True

    async def add_files(self, token, uid, utid, ds, eid, paths):
        self.added.extend(p.name for p in paths)
        return True

    async def delete_files(self, token, uid, utid, ds, file_ids):
        self.deleted.extend(file_ids)
        return "job-1"

    async def wait_for_job(self, *a, **k):
        return "succeeded"


def _wire(monkeypatch, rec):
    for name in ("list_entity_files", "patch_file", "add_files",
                 "delete_files", "wait_for_job"):
        monkeypatch.setattr(kef, name, getattr(rec, name))


def _paths(tmp_path, *names):
    out = []
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x")
        out.append(p)
    return out


@pytest.mark.asyncio
async def test_matching_names_are_patched_not_re_added(monkeypatch, tmp_path):
    rec = _Recorder([{"fileId": "f1", "fileName": "t1.nii.gz"},
                     {"fileId": "f2", "fileName": "t2.nii.gz"}])
    _wire(monkeypatch, rec)

    result = await kef.replace_entity_contents(
        "tok", 26, 1, 355, "e1", _paths(tmp_path, "t1.nii.gz", "t2.nii.gz"))

    assert sorted(rec.patched) == [("f1", "t1.nii.gz"), ("f2", "t2.nii.gz")]
    assert rec.added == [] and rec.deleted == []
    assert result["patched"] == 2


@pytest.mark.asyncio
async def test_new_names_are_added_and_vanished_ones_deleted(monkeypatch, tmp_path):
    rec = _Recorder([{"fileId": "f1", "fileName": "t1.nii.gz"},
                     {"fileId": "f2", "fileName": "t1c.nii.gz"}])
    _wire(monkeypatch, rec)

    result = await kef.replace_entity_contents(
        "tok", 26, 1, 355, "e1", _paths(tmp_path, "t1.nii.gz", "t2fl.nii.gz"))

    assert rec.patched == [("f1", "t1.nii.gz")]
    assert rec.added == ["t2fl.nii.gz"]
    assert rec.deleted == ["f2"]          # t1c is gone from the new set
    assert result == {"patched": 1, "added": 1, "deleted": 1,
                      "failed": [], "delete_job": "succeeded"}


@pytest.mark.asyncio
async def test_expert_masks_are_never_deleted(monkeypatch, tmp_path):
    """Without this exemption the first replacement wipes every expert edit
    on that patient — they are not part of a recomputed set."""
    rec = _Recorder([
        {"fileId": "f1", "fileName": "sub-002_ses-001_t1.nii.gz"},
        {"fileId": "m1", "fileName": "sub-002_ses-001_segmask_v2.nii.gz"},
        {"fileId": "m2", "fileName": "sub-002_ses-001_segmask_v3.nii.gz"},
    ])
    _wire(monkeypatch, rec)

    await kef.replace_entity_contents(
        "tok", 26, 1, 355, "e1", _paths(tmp_path, "sub-002_ses-001_t1.nii.gz"))

    assert rec.deleted == []
    assert rec.patched == [("f1", "sub-002_ses-001_t1.nii.gz")]


@pytest.mark.asyncio
async def test_a_failed_file_is_named_not_swallowed(monkeypatch, tmp_path):
    """A partial replacement must be visible: the session stays blocked and
    the operator can repeat the action, which is idempotent."""
    rec = _Recorder([{"fileId": "f1", "fileName": "t1.nii.gz"}])
    _wire(monkeypatch, rec)

    async def _fail(*a, **k):
        return False
    monkeypatch.setattr(kef, "patch_file", _fail)

    result = await kef.replace_entity_contents(
        "tok", 26, 1, 355, "e1", _paths(tmp_path, "t1.nii.gz"))

    assert result["patched"] == 0
    assert result["failed"] == ["t1.nii.gz"]


@pytest.mark.asyncio
async def test_an_empty_replacement_is_refused(monkeypatch, tmp_path):
    """An empty set means discovery found nothing — a bug upstream, not a
    request to empty the entity. A function that deletes data in someone
    else's system must not depend on its caller being careful."""
    rec = _Recorder([{"fileId": "f1", "fileName": "t1.nii.gz"}])
    _wire(monkeypatch, rec)

    with pytest.raises(ValueError):
        await kef.replace_entity_contents("tok", 26, 1, 355, "e1", [])

    assert rec.deleted == [] and rec.patched == []


@pytest.mark.asyncio
async def test_an_unknown_entity_changes_nothing(monkeypatch, tmp_path):
    rec = _Recorder([])
    _wire(monkeypatch, rec)

    result = await kef.replace_entity_contents(
        "tok", 26, 1, 355, "e1", _paths(tmp_path, "t1.nii.gz"))

    # Nothing to match against, so everything is an addition — and crucially
    # nothing was deleted on the strength of an empty listing.
    assert rec.deleted == []
    assert result["added"] == 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_entity_files.py -q`
Expected: FAIL — `AttributeError: module 'kappa_entity_files' has no attribute 'replace_entity_contents'`.

- [ ] **Step 3: Write the function**

Append to `backend/kappa_entity_files.py` (and add `import re` at the top):

```python
# Expert masks are versioned into the filename by the Slicer flow
# ("{base}_segmask_v{N}.nii.gz"). This pattern is the one app.py:2115 already
# uses to read those version numbers back — the same rule, not a second guess
# at it. They are not part of a recomputed set, and deleting them would throw
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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_kappa_entity_files.py -q`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_entity_files.py backend/test_kappa_entity_files.py
git commit -m "feat(kappa): replace an entity's contents by diffing filenames

Same name is replaced in place and keeps its file id; a new name is added;
a name no longer in the set is deleted. Expert masks are exempt from that
last rule — they are not part of a recomputed set, and without the
exemption the first replacement would wipe every expert edit.

A file that fails is named in the result rather than swallowed, so a
partial replacement stays visible and can simply be repeated.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Remembering which sessions supersede Kappa

**Files:**
- Modify: `backend/database.py` (model ~line 75, `init_db` ~line 318, new migration beside `_migrate_add_kappa_delivery`)
- Modify: `backend/app.py` (requeue, the purge block added by the previous feature)
- Test: `backend/test_app_requeue_endpoint.py` (extend)

**Interfaces:**
- Consumes: `session_artifacts.purge_sessions_marked_for_reprocess` (returns `{"sub-001/ses-001": [...]}`)
- Produces:
  - `PipelineRun.reprocessed_sessions: Optional[str]` — JSON list of `"sub-NNN_ses-NNN"`
  - `create_pipeline_run(..., reprocessed_sessions: Optional[str] = None)`
  - `superseding_sessions(run) -> set[str]` in `backend/database.py`

**Background:** `needs_reprocess` cannot answer "does this supersede Kappa?" —
it is cleared during the purge, long before upload. The purge already knows
exactly which sessions it cleared; this carries that knowledge forward.

- [ ] **Step 1: Write the failing test**

Append to `backend/test_app_requeue_endpoint.py`:

```python
def test_requeue_records_what_it_purged_on_the_new_run():
    """needs_reprocess is cleared by the purge itself, so by upload time
    nothing would remember that these sessions supersede what Kappa holds."""
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()
    captured = {}

    def _create(db, **kwargs):
        captured.update(kwargs)
        return new_run

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", side_effect=_create), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               return_value={"sub-002/ses-001": [], "sub-002/ses-002": []}):
        response = client.post("/api/pipeline-runs/orig-run/requeue")

    assert response.status_code == 200
    import json as _json
    assert sorted(_json.loads(captured["reprocessed_sessions"])) == [
        "sub-002_ses-001", "sub-002_ses-002",
    ]


def test_requeue_without_a_purge_records_nothing():
    original = _fake_original_run(status="completed")
    new_run = _fake_new_run()
    captured = {}

    def _create(db, **kwargs):
        captured.update(kwargs)
        return new_run

    with patch("app.get_pipeline_run", return_value=original), \
         patch("app.get_active_run_by_output_path", return_value=None), \
         patch("app.create_pipeline_run", side_effect=_create), \
         patch("app.run_pipeline_background"), \
         patch("app.pipeline_monitor.start_monitoring", new=AsyncMock()), \
         patch("session_artifacts.purge_sessions_marked_for_reprocess",
               return_value={}):
        client.post("/api/pipeline-runs/orig-run/requeue")

    assert captured.get("reprocessed_sessions") is None
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_app_requeue_endpoint.py -q`
Expected: FAIL — `KeyError: 'reprocessed_sessions'`.

- [ ] **Step 3: Add the column, its migration and a reader**

In `backend/database.py`, after `kappa_user_id` in the model:

```python
    # Sessions this run rebuilt because the doctor corrected their modality
    # set. JSON list of "sub-NNN_ses-NNN". Their results in Kappa were
    # computed from a set now known to be wrong, so they supersede rather
    # than duplicate — see kappa_uploader.
    reprocessed_sessions = Column(Text, nullable=True)
```

Append the migration beside the others and register it in `init_db()`:

```python
def _migrate_add_reprocessed_sessions():
    """Add pipeline_runs.reprocessed_sessions if it is not there yet."""
    import sqlalchemy
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]
        if 'reprocessed_sessions' not in cols:
            conn.execute(sqlalchemy.text(
                "ALTER TABLE pipeline_runs ADD COLUMN reprocessed_sessions TEXT"
            ))
            conn.commit()
```

```python
    _migrate_add_kappa_delivery()
    _migrate_add_reprocessed_sessions()
```

Add the parameter to `create_pipeline_run` and pass it to the constructor:

```python
    reprocessed_sessions: Optional[str] = None,
```
```python
        reprocessed_sessions=reprocessed_sessions,
```

And a reader at the end of the module:

```python
def superseding_sessions(run: PipelineRun) -> set:
    """Session keys whose Kappa contents this run supersedes. Never raises:
    a corrupt value must not stop an upload."""
    raw = getattr(run, "reprocessed_sessions", None)
    if not raw:
        return set()
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return set()
    return {str(x) for x in parsed} if isinstance(parsed, list) else set()
```

- [ ] **Step 4: Record the purge on the new run**

In `backend/app.py`, in `requeue_pipeline_run`, the purge block becomes:

```python
    # Сессии, у которых врач поменял набор модальностей, надо пересчитать.
    # skip_existing пропускает всё, у чего уже есть результаты, поэтому без
    # удаления исправление просто не дошло бы до данных.
    purged_sessions = None
    try:
        from session_artifacts import purge_sessions_marked_for_reprocess
        purged = purge_sessions_marked_for_reprocess(original_run.output_path)
        if purged:
            logger.info("Переобработка: очищено сессий — %d", len(purged))
            # Запоминаем на новом прогоне: к моменту выгрузки флаг
            # needs_reprocess уже снят, и иначе никто не вспомнит, что эти
            # сессии вытесняют лежащее в Kappa, а не дублируют его.
            purged_sessions = json.dumps(
                [key.replace("/", "_") for key in purged]
            )
    except Exception as e:  # noqa: BLE001 — запуск важнее уборки
        logger.error("Не удалось очистить помеченные сессии: %s", e)
```

and `create_pipeline_run(...)` gains `reprocessed_sessions=purged_sessions,`.

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_app_requeue_endpoint.py -q`
Expected: all pass.

Run: `python -m pytest backend/ -q`
Expected: no drop from the baseline.

- [ ] **Step 6: Commit**

```bash
git add backend/database.py backend/app.py backend/test_app_requeue_endpoint.py
git commit -m "feat(kappa): remember which sessions supersede what Kappa holds

needs_reprocess is cleared by the purge itself, long before upload, so by
the time it matters nothing remembers that these sessions were rebuilt from
a corrected modality set. The purge already knows exactly which ones; this
carries that forward on the run it creates.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The uploader reports `supersedes`

**Files:**
- Modify: `backend/kappa_uploader.py` (constructor, and the duplicate branch at `~:115`)
- Modify: `backend/pipeline_monitor.py` (`_create_kappa_uploader`), `backend/kappa_delivery_worker.py` (`build_uploader`)
- Test: `backend/test_kappa_uploader_supersedes.py` (create)

**Interfaces:**
- Consumes: `database.superseding_sessions(run) -> set[str]`
- Produces: `KappaUploader(..., superseding_sessions: set[str] = frozenset())`; a per-session result `{"session": ..., "success": False, "error": "supersedes", "message": ...}`

**Background:** read `upload_results` (`backend/kappa_uploader.py:62`) first. The
duplicate branch is the one that currently reports success for a reprocessed
patient, which is how a correction silently fails to reach Kappa.

- [ ] **Step 1: Write the failing test**

Create `backend/test_kappa_uploader_supersedes.py`:

```python
"""A reprocessed session must not pass as a duplicate.

study_hash covers PatientID:StudyInstanceUID — the DICOM study, not the
images — so a rerun on a corrected modality set hashes the same and the
uploader reports it delivered. The correction then never reaches Kappa, and
nothing says so.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from kappa_uploader import KappaUploader


def _uploader(tmp_path, superseding=frozenset()):
    return KappaUploader(
        run_id="r1", output_path=str(tmp_path), token="tok",
        user_id=26, user_type_id=1, lesion_type="glioblastoma",
        preprocessing_config_path=str(tmp_path / "cfg.yaml"),
        dataset_id=351, superseding_sessions=superseding,
    )


@pytest.mark.asyncio
async def test_a_superseding_session_is_reported_not_skipped(tmp_path, monkeypatch):
    up = _uploader(tmp_path, superseding={"sub-002_ses-001"})
    monkeypatch.setattr(up, "_resolve_dataset_id", _async(351))
    monkeypatch.setattr(up, "_bind_pending_scope", lambda ds: None)
    monkeypatch.setattr(up, "_discover_sessions",
                        lambda: {"sub-002_ses-001": {"preprocessed": [], "masks": []}})
    monkeypatch.setattr(up, "_compute_study_hash", lambda data: "h1")
    monkeypatch.setattr(up, "_get_existing_study_hashes", _async({"h1"}))
    monkeypatch.setattr(up, "_get_existing_entity_names", _async({"sub-002_ses-001"}))

    result = await up.upload_results()

    entry = result["sessions"][0]
    assert entry["error"] == "supersedes"
    assert entry["success"] is False


@pytest.mark.asyncio
async def test_an_ordinary_duplicate_still_passes(tmp_path, monkeypatch):
    """Nothing changes for a session nobody reprocessed."""
    up = _uploader(tmp_path)
    monkeypatch.setattr(up, "_resolve_dataset_id", _async(351))
    monkeypatch.setattr(up, "_bind_pending_scope", lambda ds: None)
    monkeypatch.setattr(up, "_discover_sessions",
                        lambda: {"sub-002_ses-001": {"preprocessed": [], "masks": []}})
    monkeypatch.setattr(up, "_compute_study_hash", lambda data: "h1")
    monkeypatch.setattr(up, "_get_existing_study_hashes", _async({"h1"}))
    monkeypatch.setattr(up, "_get_existing_entity_names", _async({"sub-002_ses-001"}))
    monkeypatch.setattr(up, "_reconcile_session", _async("e1"))

    result = await up.upload_results()

    assert result["sessions"][0]["error"] != "supersedes"


def _async(value):
    async def _inner(*a, **k):
        return value
    return _inner
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_uploader_supersedes.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'superseding_sessions'`.

- [ ] **Step 3: Teach the uploader**

Add the parameter to `KappaUploader.__init__` (keep it last, defaulted):

```python
        superseding_sessions=frozenset(),
```
```python
        # Sessions rebuilt from a corrected modality set. They hash the same
        # as what Kappa holds (study_hash covers the DICOM study, not the
        # images), so without this they would pass as duplicates and the
        # correction would never arrive.
        self.superseding_sessions = set(superseding_sessions or ())
```

In `upload_results`, immediately before the existing duplicate branch
(`if study_hash and study_hash in existing_hashes:`):

```python
            if (study_hash and study_hash in existing_hashes
                    and session_key in self.superseding_sessions):
                logger.warning(
                    "Session %s supersedes what dataset %d holds — needs a "
                    "human decision, not a silent skip", session_key, dataset_id,
                )
                results.append({
                    "session": session_key,
                    "success": False,
                    "error": "supersedes",
                    "message": (
                        f"{session_key} пересчитан с исправленным набором "
                        f"модальностей — в Kappa лежит прежняя версия"
                    ),
                })
                continue
```

- [ ] **Step 4: Thread the run's list into both construction sites**

In `backend/pipeline_monitor.py`, `_create_kappa_uploader`, where `dataset_id`
is read from the run, read the list too and pass it:

```python
            from database import superseding_sessions as _superseding
            db = _DBSessionLocal()
            try:
                run = _get_pipeline_run(db, run_id)
                dataset_id = run.kappa_dataset_id if run else None
                supersedes = _superseding(run) if run else set()
            finally:
                db.close()
```
```python
                dataset_id=dataset_id,
                superseding_sessions=supersedes,
```

In `backend/kappa_delivery_worker.py`, `build_uploader` — aliased on import
for the same reason as above, so the keyword and the function it calls are not
the same word:

```python
    from database import superseding_sessions as _superseding
```
```python
        dataset_id=run.kappa_dataset_id,
        superseding_sessions=_superseding(run),
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_kappa_uploader_supersedes.py -q`
Expected: 2 passed.

Run: `python -m pytest backend/ -q`
Expected: no drop from the baseline.

- [ ] **Step 6: Commit**

```bash
git add backend/kappa_uploader.py backend/pipeline_monitor.py \
        backend/kappa_delivery_worker.py backend/test_kappa_uploader_supersedes.py
git commit -m "feat(kappa): a reprocessed session is not a duplicate

study_hash covers the DICOM study, not the images, so a rerun on a
corrected modality set hashes identically and was reported delivered. The
correction never reached Kappa and nothing said so.

Such a session is now reported as 'supersedes' — a decision for a human,
not a silent skip. Sessions nobody reprocessed behave exactly as before.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Routing it through the existing states

**Files:**
- Modify: `backend/kappa_delivery.py:25` (`_BLOCKING`)
- Modify: `backend/kappa_run_log.py:24` (`_SESSION_ERRORS`), `:37` (`_REASONS`)
- Test: `backend/test_kappa_delivery.py` (extend)

**Interfaces:**
- Consumes: the `supersedes` per-session error from Task 4.
- Produces: `classify(...)` → `needs_attention` with `detail["reason"] == "supersedes_kappa"`.

**Background:** no fourth status. The column tag, the summary banner and the
per-run modal already handle `needs_attention`; a new reason needs a label and
an action, not a state that the history, the worker and the summary must each
learn about.

- [ ] **Step 1: Write the failing test**

Append to `backend/test_kappa_delivery.py`:

```python
def test_a_superseding_session_needs_a_human_not_a_retry():
    """Overwriting data in Kappa is not something to retry into."""
    result = {"dataset_id": 351, "uploaded": 0, "total": 1,
              "sessions": [_fail("sub-002_ses-001", "supersedes")]}
    out = classify(result, None, {}, NOW)

    assert out["status"] == "needs_attention"
    assert out["detail"]["reason"] == "supersedes_kappa"
    assert out["next_attempt"] is None
    assert out["detail"]["blocked"][0]["session"] == "sub-002_ses-001"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_delivery.py -q`
Expected: FAIL — the session is treated as an ordinary transient failure, so
status is `pending`.

- [ ] **Step 3: Add the reason**

In `backend/kappa_delivery.py`:

```python
_BLOCKING = {
    "name_clash": "name_clash",
    "no files": "missing_files",
    # Retrying cannot help: the upload is correctly refusing to overwrite
    # Kappa without a human saying so.
    "supersedes": "supersedes_kappa",
}
```

In `backend/kappa_run_log.py`:

```python
    "supersedes": "в Kappa прежняя версия — нужно подтвердить замену",
```
```python
    "supersedes_kappa": "в Kappa прежняя версия",
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_kappa_delivery.py backend/test_kappa_run_log.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_delivery.py backend/kappa_run_log.py \
        backend/test_kappa_delivery.py
git commit -m "feat(kappa): route a superseding session through needs_attention

No fourth status: the column tag, the summary banner and the per-run modal
already handle needs_attention, so a new reason needs a label and an action
rather than a state the history, worker and summary must each learn.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: One definition of "the files that make an entity"

**Files:**
- Modify: `backend/kappa_uploader.py` (`_upload_session`, ~`:470-495`)
- Test: `backend/test_kappa_uploader_supersedes.py` (extend)

**Interfaces:**
- Consumes: the `session_data` dict `_discover_sessions` produces.
- Produces: `session_file_paths(session_data: dict) -> list[Path]` — module-level
  in `kappa_uploader.py`.

**Background:** the replace endpoint needs the same file list the upload
builds. Writing a second rule there is how the two drift: add a report to one
and the other keeps sending the old set. `_upload_session` already decides it —
preprocessed volumes, the main mask but not `_native_`, and the labelled lesion
mask when it exists. Lift that out and call it from both.

- [ ] **Step 1: Write the failing test**

Append to `backend/test_kappa_uploader_supersedes.py`:

```python
def test_the_entity_file_rule_lives_in_one_place(tmp_path):
    """Both the upload and the replacement must send the same set. Two
    copies of this rule drift the moment one gains a file."""
    from kappa_uploader import session_file_paths

    session_data = {
        "preprocessed": [tmp_path / "t1.nii.gz", tmp_path / "t2.nii.gz"],
        "masks": [tmp_path / "sub-002_ses-001_segmask.nii.gz",
                  tmp_path / "sub-002_ses-001_segmask_native_t1.nii.gz"],
        "lesion_labels_mask": tmp_path / "labels.nii.gz",
    }

    names = [p.name for p in session_file_paths(session_data)]

    assert "t1.nii.gz" in names and "t2.nii.gz" in names
    assert "sub-002_ses-001_segmask.nii.gz" in names
    assert "labels.nii.gz" in names
    # The native mask lives in the patient's own space and is not part of
    # the entity — it is the one thing deliberately left out.
    assert not any("_native_" in n for n in names)


def test_no_labels_mask_is_simply_absent(tmp_path):
    from kappa_uploader import session_file_paths

    paths = session_file_paths({
        "preprocessed": [tmp_path / "t1.nii.gz"],
        "masks": [],
        "lesion_labels_mask": None,
    })
    assert [p.name for p in paths] == ["t1.nii.gz"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_kappa_uploader_supersedes.py -q`
Expected: FAIL — `ImportError: cannot import name 'session_file_paths'`.

- [ ] **Step 3: Lift the rule out of `_upload_session`**

Add at module level in `backend/kappa_uploader.py`, after `logger`:

```python
def session_file_paths(session_data: Dict[str, Any]) -> List[Path]:
    """The files that make up one session's Kappa entity.

    Both the upload and the replacement need this set. Keeping one
    definition is what stops them drifting apart — a report added to the
    entity has to reach a replaced entity too.

    The native-space mask is deliberately excluded: it lives in the
    patient's own geometry and is not part of what the dataset holds.
    """
    paths = list(session_data["preprocessed"])
    paths.extend(m for m in session_data["masks"] if "_native_" not in m.name)
    if session_data.get("lesion_labels_mask"):
        paths.append(session_data["lesion_labels_mask"])
    return paths
```

Then replace the opening of `_upload_session` (the block from
`# Собираем файлы: preprocessed + основная маска` down to and including the
`lesion_labels_mask` append) with:

```python
        # Собираем файлы: preprocessed + основная маска + labels.
        # Правило одно для выгрузки и для замены — см. session_file_paths.
        file_paths = session_file_paths(session_data)
```

Leave the `if not file_paths:` guard that follows exactly as it is.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest backend/test_kappa_uploader_supersedes.py backend/test_kappa_uploader.py -q`
Expected: all pass. The existing uploader tests are what prove the extraction
changed no behaviour.

- [ ] **Step 5: Commit**

```bash
git add backend/kappa_uploader.py backend/test_kappa_uploader_supersedes.py
git commit -m "refactor(kappa): one definition of an entity's file set

The replacement needs the same files the upload sends. A second copy of
the rule drifts the moment one of them gains a report, so the rule moves
out of _upload_session and both call it.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: The replace endpoint

**Files:**
- Modify: `backend/models.py` (after `AssignmentResponse`), `backend/app.py` (after
  `retry_kappa_upload`, ~`:2920`)
- Test: `backend/test_app_replace_endpoint.py` (create)

**Interfaces:**
- Consumes: `kappa_entity_files.replace_entity_contents`,
  `kappa_uploader.session_file_paths`, `database.superseding_sessions`,
  `patient_registry.find_by_bids_id`,
  `pipeline_monitor._create_kappa_uploader`
- Produces:
  - `models.ReplaceEntityResponse` with `patched: int`, `added: int`,
    `deleted: int`, `failed: List[str]`, `delete_job: Optional[str]`
  - `POST /api/kappa/replace-entity/{run_id}/{patient_id}/{session_id}?kappa_session_id=<id>`

**Background:** two decisions that look arbitrary and are not.

- **The query parameter is `kappa_session_id`, not `session_id`.** The path
  already has a `session_id` — the BIDS session — and FastAPI cannot bind two
  parameters to one name. `kappa_session_id` is the name the other half of the
  Kappa endpoints already use (`app.py:26`, `:352`).
- **The uploader is built with `pipeline_monitor._create_kappa_uploader`**, the
  same call `retry_kappa_upload` makes (`app.py:2882`). It resolves the Kappa
  session, checks the preprocessing config, and carries the token — so the
  endpoint does not re-derive any of it, and `_discover_sessions()` on it finds
  the files with the layout logic that already works. Declare the parameter as
  a bare `str`, like `retry_kappa_upload(run_id: str, session_id: str)` does;
  `Query` is not imported in this module.

- [ ] **Step 1: Write the failing test**

Create `backend/test_app_replace_endpoint.py`:

```python
"""Replacing an entity's contents, on purpose and only on purpose."""
import json
import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, "backend")


class _Run:
    run_id = "run-1"
    output_path = "/tmp/run"
    lesion_type = "glioblastoma"
    kappa_dataset_id = 351
    reprocessed_sessions = json.dumps(["sub-002_ses-001"])


class _Uploader:
    token = "t"
    user_id = 26
    user_type_id = 1

    def _discover_sessions(self):
        return {"sub-002_ses-001": {
            "preprocessed": [Path("/tmp/run/t1.nii.gz")],
            "masks": [], "lesion_labels_mask": None,
        }}


@pytest.mark.asyncio
async def test_refuses_a_session_that_supersedes_nothing(monkeypatch):
    """This endpoint overwrites data in someone else's system. It must not
    be a general purpose tool reachable by guessing a URL."""
    import app

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())

    with pytest.raises(HTTPException) as caught:
        await app.replace_kappa_entity(
            "run-1", "sub-009", "ses-001", kappa_session_id="sid", db=None)

    assert caught.value.status_code == 400
    assert "не помечена" in caught.value.detail


@pytest.mark.asyncio
async def test_replaces_and_reports_the_counts(monkeypatch):
    import app
    import kappa_entity_files as kef

    seen = {}

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: _Uploader())

    def _find(bids_id, dataset_ids=None):
        seen["dataset_ids"] = dataset_ids
        return [{"kappa_entity_id": "e1"}]
    monkeypatch.setattr(app, "find_by_bids_id", _find)

    async def _replace(**kwargs):
        seen["files"] = [p.name for p in kwargs["files"]]
        seen["entity_id"] = kwargs["entity_id"]
        return {"patched": 3, "added": 1, "deleted": 1,
                "failed": [], "delete_job": "succeeded"}
    monkeypatch.setattr(kef, "replace_entity_contents", _replace)
    # app.py imports kappa_run_log inside the endpoint (house style), so
    # the real module is what the local import resolves.
    monkeypatch.setattr("kappa_run_log.append", lambda *a, **k: None)

    result = await app.replace_kappa_entity(
        "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)

    assert (result.patched, result.added, result.deleted) == (3, 1, 1)
    assert result.failed == []
    assert seen["entity_id"] == "e1"
    assert seen["files"] == ["t1.nii.gz"]
    # sub-NNN is unique only WITHIN a dataset; an unscoped lookup would
    # report another account's patient as this one.
    assert seen["dataset_ids"] == {351}


@pytest.mark.asyncio
async def test_refuses_without_a_kappa_session(monkeypatch):
    import app

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: None)

    with pytest.raises(HTTPException) as caught:
        await app.replace_kappa_entity(
            "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)
    assert caught.value.status_code == 401


@pytest.mark.asyncio
async def test_refuses_when_kappa_has_no_such_entity(monkeypatch):
    """Nothing to replace is not the same as a replacement that did
    nothing, and the operator needs to be able to tell them apart."""
    import app

    monkeypatch.setattr(app, "get_pipeline_run", lambda db, rid: _Run())
    monkeypatch.setattr(app.pipeline_monitor, "_create_kappa_uploader",
                        lambda *a, **k: _Uploader())
    monkeypatch.setattr(app, "find_by_bids_id", lambda *a, **k: [])

    with pytest.raises(HTTPException) as caught:
        await app.replace_kappa_entity(
            "run-1", "sub-002", "ses-001", kappa_session_id="sid", db=None)
    assert caught.value.status_code == 404

```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest backend/test_app_replace_endpoint.py -q`
Expected: FAIL — `AttributeError: module 'app' has no attribute 'replace_kappa_entity'`.

- [ ] **Step 3: Add the response model**

In `backend/models.py`:

```python
class ReplaceEntityResponse(BaseModel):
    """Результат замены содержимого сущности в Kappa."""
    patched: int = Field(0, description="Файлов заменено на месте")
    added: int = Field(0, description="Файлов добавлено")
    deleted: int = Field(0, description="Файлов удалено")
    failed: List[str] = Field(default_factory=list, description="Что не удалось")
    delete_job: Optional[str] = Field(
        None, description="succeeded | failed | running — итог удаления"
    )
```

Also widen the now-stale `reason` description on `KappaBlockedSession`
(`models.py:396`), which still lists only two of the three reasons:

```python
    reason: str = Field(
        ..., description="name_clash | missing_files | supersedes_kappa"
    )
```

- [ ] **Step 4: Add the endpoint**

In `backend/app.py`, after `retry_kappa_upload` ends. Add
`ReplaceEntityResponse` to the `models` import list; `kappa_run_log` is
imported inside the function, which is how every other call site in this
module does it:

```python
@app.post(
    "/api/kappa/replace-entity/{run_id}/{patient_id}/{session_id}",
    response_model=ReplaceEntityResponse,
)
async def replace_kappa_entity(
    run_id: str,
    patient_id: str,
    session_id: str,
    kappa_session_id: str,
    db: Session = Depends(get_db),
):
    """Заменить содержимое сущности пациента в Kappa результатами переобработки.

    Параметр сессии Kappa называется kappa_session_id, а не session_id:
    session_id в пути — это BIDS-сессия, и два параметра с одним именем
    FastAPI связать не может.
    """
    import kappa_entity_files
    import kappa_run_log
    from database import superseding_sessions
    from kappa_uploader import session_file_paths

    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Запуск не найден")

    session_key = f"{patient_id}_{session_id}"
    # Эндпоинт перезаписывает данные в чужой системе. Он должен работать
    # только для сессии, которую переобработали, а не для любой, чей адрес
    # удалось угадать.
    if session_key not in superseding_sessions(run):
        raise HTTPException(
            status_code=400,
            detail=f"Сессия {session_key} не помечена как вытесняющая — "
                   f"заменять нечего",
        )

    # Тот же путь, которым пользуется retry-upload: разрешает сессию Kappa,
    # проверяет конфиг предобработки и знает раскладку файлов прогона.
    uploader = pipeline_monitor._create_kappa_uploader(
        run_id, run.output_path, kappa_session_id,
        getattr(run, "lesion_type", None) or "glioblastoma",
    )
    if uploader is None:
        raise HTTPException(
            status_code=401,
            detail="Сессия Kappa не найдена/истекла, или не найден конфиг "
                   "препроцессинга. Войдите в Kappa заново.",
        )

    records = find_by_bids_id(session_key, {run.kappa_dataset_id}) or []
    entity_id = next((r.get("kappa_entity_id") for r in records
                      if r.get("kappa_entity_id")), None)
    if not entity_id:
        raise HTTPException(
            status_code=404,
            detail=f"В Kappa нет записи для {session_key} — заменять нечего",
        )

    session_data = uploader._discover_sessions().get(session_key)
    files = session_file_paths(session_data) if session_data else []
    if not files:
        raise HTTPException(
            status_code=404,
            detail=f"На диске нет файлов {session_key} — нечем заменять",
        )

    result = await kappa_entity_files.replace_entity_contents(
        token=uploader.token,
        user_id=uploader.user_id,
        user_type_id=uploader.user_type_id,
        dataset_id=run.kappa_dataset_id,
        entity_id=entity_id,
        files=files,
    )

    kappa_run_log.append(
        run.output_path,
        f"Замена в Kappa для {session_key}: заменено {result['patched']}, "
        f"добавлено {result['added']}, удалено {result['deleted']}"
        + (f", не удалось: {', '.join(result['failed'])}"
           if result["failed"] else ""),
    )
    logger.info("Kappa entity replaced for %s: %s", session_key, result)
    return ReplaceEntityResponse(**result)
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest backend/test_app_replace_endpoint.py -q`
Expected: 4 passed.

Run: `python -c "import sys; sys.path.insert(0,'backend'); import app"`
Expected: no traceback.

Run: `python -m pytest backend/ -q`
Expected: no drop from the baseline.

- [ ] **Step 6: Commit**

```bash
git add backend/models.py backend/app.py backend/test_app_replace_endpoint.py
git commit -m "feat(kappa): endpoint that replaces a reprocessed patient

Refuses any session the run did not mark as superseding: this overwrites
data in someone else's system and must not be a general purpose tool
reachable by guessing a URL. The registry lookup is scoped to the run's
dataset, because sub-NNN is unique only within one.

Reports what was patched, added, deleted and what failed, so a partial
replacement stays visible rather than rounded up to success.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: The confirmation

**Files:**
- Modify: `frontend/src/services/api.js` (after `retryKappaUpload`, `:373`),
  `frontend/src/components/PipelineHistory.jsx` (imports `:5`, state `:34`,
  the blocked `List` `:514-525`)
- Modify: `backend/app.py` (`_delivery_status`, `:859-881`), `backend/models.py`

**Interfaces:**
- Consumes: `POST /api/kappa/replace-entity/...`; `blocked[].expert_masks`
  from the history response.
- Produces: no backend-facing interfaces.

**Background:** the expert-mask count cannot come from `classify`, which is
pure and has no database. `_delivery_status` has the run and already runs per
row, so looking it up there — and only for a blocked session whose reason is
`supersedes_kappa` — is the cheap place.

- [ ] **Step 1: Carry the expert-mask count on blocked sessions**

In `backend/models.py`, `KappaBlockedSession` gains:

```python
    expert_masks: int = Field(
        0, description="Сколько экспертных масок у сессии — они переживут замену"
    )
```

In `backend/app.py`, inside `_delivery_status`, before the
`KappaDeliveryStatus(...)` return:

```python
    def _expert_masks(session_key: str) -> int:
        """How many expert masks this session has. They survive a
        replacement, but they were drawn on the superseded data, so the
        operator is told before confirming."""
        dataset_id = getattr(run, "kappa_dataset_id", None)
        if not dataset_id or not session_key:
            return 0
        from mask_service import get_mask_history
        records = find_by_bids_id(session_key, {dataset_id}) or []
        entity_id = next((r.get("kappa_entity_id") for r in records
                          if r.get("kappa_entity_id")), None)
        if not entity_id:
            return 0
        return sum(1 for v in get_mask_history(entity_id)
                   if v.get("source") == "expert")
```

and each blocked entry gains, inside the existing list comprehension:

```python
                expert_masks=(
                    _expert_masks(b.get("session") or "")
                    if b.get("reason") == "supersedes_kappa" else 0
                ),
```

The `reason` guard is what keeps this from adding two queries per row to
every history page: only a superseding session pays for it.

- [ ] **Step 2: Add the API helper**

In `frontend/src/services/api.js`, after `retryKappaUpload`:

```javascript
/**
 * Заменить содержимое сущности пациента в Kappa результатами переобработки.
 */
export const replaceKappaEntity = async (runId, patientId, sessionId) => {
  const response = await apiClient.post(
    `/kappa/replace-entity/${runId}/${patientId}/${sessionId}`,
    null,
    { params: { kappa_session_id: localStorage.getItem('kappa_session_id') } },
  );
  return response.data;
};
```

Add `replaceKappaEntity` to the default-export object at the end of the file.

- [ ] **Step 3: Offer it in the delivery modal**

In `frontend/src/components/PipelineHistory.jsx`:

Add `Popconfirm` to the antd import on line 5 — it is **not** currently
imported — and `replaceKappaEntity` to the `../services/api` import.

Add the state beside `retrying` (line 34) and the handler beside the others:

```javascript
  const [replacing, setReplacing] = useState(null);

  const handleReplace = async (blocked) => {
    const [patient, session] = (blocked.session || '').split('_');
    setReplacing(blocked.session);
    try {
      const r = await replaceKappaEntity(
        deliveryDetail.run_id, patient, session);
      if (r.failed?.length) {
        message.warning(
          `Заменено ${r.patched}, но не удалось: ${r.failed.join(', ')}. `
          + 'Можно повторить — замена идемпотентна.', 8,
        );
      } else if (r.delete_job === 'running') {
        message.warning(
          `Заменено файлов: ${r.patched}. Удаление лишних ещё идёт в Kappa — `
          + 'проверьте сущность через минуту.', 8,
        );
      } else {
        message.success(
          `Заменено файлов: ${r.patched}, добавлено ${r.added}, `
          + `удалено ${r.deleted}`,
        );
      }
      setDeliveryDetail(null);
      fetchHistory();
    } catch (e) {
      message.error(e?.response?.data?.detail || 'Не удалось заменить в Kappa');
    } finally {
      setReplacing(null);
    }
  };
```

Replace the blocked `List`'s `renderItem` (line 519-523) with:

```jsx
                renderItem={(b) => (
                  <List.Item
                    actions={b.reason === 'supersedes_kappa' ? [
                      <Popconfirm
                        key="replace"
                        title="Заменить версию в Kappa?"
                        description={(
                          <div style={{ maxWidth: 360 }}>
                            Файлы пациента будут перезаписаны результатами
                            новой обработки. Прежняя версия не сохранится.
                            {b.expert_masks > 0 && (
                              <div style={{ marginTop: 8 }}>
                                У пациента есть экспертные маски
                                {` (${b.expert_masks})`}. Они останутся, но
                                нарисованы по прежним данным.
                              </div>
                            )}
                          </div>
                        )}
                        okText="Заменить"
                        cancelText="Отмена"
                        onConfirm={() => handleReplace(b)}
                      >
                        <Button size="small" danger
                                loading={replacing === b.session}>
                          Заменить в Kappa
                        </Button>
                      </Popconfirm>,
                    ] : []}
                  >
                    <strong>{b.session}</strong>: {b.message || b.reason}
                  </List.Item>
                )}
```

- [ ] **Step 4: Lint and build**

Run: `cd frontend && npm run lint`
Expected: no new errors. The baseline is pre-existing errors in other files;
`PipelineHistory.jsx` must not appear.

Run: `cd frontend && npm run build`
Expected: builds.

- [ ] **Step 5: Run the backend suite**

Run: `python -m pytest backend/ -q`
Expected: no drop from the baseline.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/services/api.js frontend/src/components/PipelineHistory.jsx \
        backend/models.py backend/app.py
git commit -m "feat(frontend): confirm before replacing a patient in Kappa

Overwriting data in someone else's system is not something to do on a
click, so the action states plainly that the previous version will not
survive — and, when the patient has expert masks, that those remain but
were drawn on the superseded data.

A partial replacement says so and notes that repeating it is safe; a
delete job still running says that too, rather than implying it finished.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Correct the docstring the probe disproved

**Files:**
- Modify: `backend/registry_models.py:110-115`

**Background:** `MaskVersion`'s docstring says "В Каппе хранится только
актуальная (последняя), здесь — все". The probe showed the opposite: v1's
upload appends, so every expert mask stays in Kappa. The behaviour is what we
want; the comment is simply false, and it is exactly the kind of false premise
that makes the next person design something wrong.

- [ ] **Step 1: Fix it**

```python
class MaskVersion(Base):
    """
    История версий масок сегментации.
    Каждая запись — одна версия маски для сущности.

    В Каппе остаются ВСЕ версии: загрузка файла там добавляет, а не
    заменяет (проверено на живом API — файл с тем же именем создаёт вторую
    запись с другим id). Для экспертных правок это и нужно: каждая версия —
    отдельное мнение специалиста, а не исправление ошибки. Имя файла несёт
    номер версии (`..._segmask_v{N}.nii.gz`), поэтому они различимы и в
    интерфейсе Каппы.
    """
```

- [ ] **Step 2: Verify nothing depended on the old claim**

```bash
grep -rn "только актуальная\|только последн" backend/ || echo "больше нигде не утверждается"
```

- [ ] **Step 3: Commit**

```bash
git add backend/registry_models.py
git commit -m "docs(masks): Kappa keeps every version, not only the latest

The docstring claimed the opposite. Uploading a file to an entity appends —
a same-named file becomes a second record with a different id, verified
against the live API. For expert edits that is what we want, but a false
premise in a model docstring is how the next person designs the wrong
thing.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: End-to-end verification

**Files:** none — this task changes no code.

- [ ] **Step 1: Run everything**

```bash
cd /home/ubuntu/mri_ai_service && source venv/bin/activate
python -m pytest backend/ -q
FILES=$(ls test_*.py | grep -v '^test_stage05_fixes.py$' | tr '\n' ' ')
python -m pytest $FILES -q
cd frontend && npm run lint
```

Expected: backend green; the root suite shows the same pre-existing failures as
`main` (10 failed, 1 error) and no others; lint unchanged.

- [ ] **Step 2: Rebuild the image**

```bash
docker compose --profile full up --build -d
```

- [ ] **Step 3: Confirm the migration ran**

```bash
sqlite3 backend/data/brain_lesion.db "PRAGMA table_info(pipeline_runs);" \
  | grep reprocessed_sessions
```

- [ ] **Step 4: Reprocess, then replace**

Take a patient already delivered to Kappa. Change their modality set, save,
requeue, and let the run finish.

Expected: the run ends `needs_attention`, the column tag reads «в Kappa прежняя
версия», and the modal offers «Заменить в Kappa».

Note the entity's file count and ids first:

```bash
# подставьте свой dataset_id и entity_id
curl -sk -H "Authorization: Bearer $TOKEN" \
  "https://kappa.nsu.ru:8061/data-micro-services/v2/datasets/datasetEntities/<DS>" \
  | python3 -m json.tool | grep -E 'fileName|fileId'
```

Confirm, then run the same command again.

Expected: **the file count did not grow**, the file ids are unchanged, and any
`*_segmask_v*.nii.gz` are still present.

- [ ] **Step 5: Report**

Report what you saw at each step, including anything that did not match. Do not
mark this task complete on a partial verification.
