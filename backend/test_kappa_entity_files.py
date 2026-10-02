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
        return self.responses.pop(0)

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


async def _no_sleep(_seconds):
    return None


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
    # sleep is mocked out, so the loop spins against the real clock: keep
    # the deadline short or this one test costs as much as the whole suite.
    _install(monkeypatch, [_Resp(200, {"status": "queued"})] * 50)
    monkeypatch.setattr(kef.asyncio, "sleep", _no_sleep)

    assert await kef.wait_for_job("tok", 355, "job-7", timeout=0.1) == "running"


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
