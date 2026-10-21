"""KI-075: Kappa statuses 3 and 5 need dsEntityInfo.labels and an output mask.

Probed live 2026-10-21 (throwaway dataset 357): both conditions are needed,
both can be set when the entity is created, and a v2 PUT without files
changes the status without the empty.json the v1 call left behind (KI-036).
"""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent))

import kappa_client
import kappa_entity_files as kef
import kappa_uploader

REPO = Path(__file__).resolve().parents[1]


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = json.dumps(self._body)

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._body


class _Client:
    def __init__(self, log, resp):
        self.log, self.resp = log, resp

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def _record(self, method, url, **kw):
        files = kw.get("files")
        if isinstance(files, list):
            files = [(f[0], f[1][0]) for f in files]
        elif isinstance(files, dict):
            files = {k: v[0] for k, v in files.items()}
        self.log.append({"method": method, "url": url, "data": kw.get("data"),
                         "params": kw.get("params"), "files": files})
        return self.resp

    async def put(self, url, **kw):
        return await self._record("PUT", url, **kw)

    async def post(self, url, **kw):
        return await self._record("POST", url, **kw)

    async def patch(self, url, **kw):
        return await self._record("PATCH", url, **kw)


def _fake_http(monkeypatch, module, resp=None):
    log = []
    monkeypatch.setattr(module, "kappa_client",
                        lambda **kw: _Client(log, resp or _Resp(200, {"ds_entity_id": "e1"})))
    return log


@pytest.mark.parametrize("name, category", [
    ("sub-001_ses-001_t1_segmask.nii.gz", "output"),
    ("sub-001_ses-001_t1c_segmask.nii.gz", "output"),
    ("sub-001_ses-001_t1_segmask_labels.nii.gz", "input"),
    ("sub-001_ses-001_t1_segmask_native_t1.nii.gz", "input"),
    ("sub-001_ses-001_t1_segmask_v2.nii.gz", "input"),
    ("sub-001_ses-001_t2fl.nii.gz", "input"),
])
def test_only_the_main_mask_is_an_output(name, category):
    assert kef.file_category(name) == category


@pytest.mark.asyncio
async def test_status_is_a_v2_put_without_files(monkeypatch):
    log = _fake_http(monkeypatch, kappa_client)

    ok, msg = await kappa_client.set_entity_status("tok", 26, 3, 357, "e1", status=5)

    assert ok and msg == ""
    (call,) = log
    assert call["method"] == "PUT"
    assert call["url"].endswith("/v2/datasets/datasetEntities/357/e1")
    assert call["files"] == []          # no empty.json stub (KI-036)
    assert json.loads(call["data"]["update_dataset_entity"])["dsEntityStatus"] == 5


@pytest.mark.asyncio
async def test_a_refused_status_returns_kappas_reason(monkeypatch):
    _fake_http(monkeypatch, kappa_client, _Resp(422, {
        "detail": "Cannot mark as verified; missing required fields: labels"}))

    ok, msg = await kappa_client.set_entity_status("tok", 26, 3, 357, "e1", status=5)

    assert ok is False
    assert msg == "Cannot mark as verified; missing required fields: labels"
    assert await kappa_client.update_entity_status("tok", 26, 3, 357, "e1", 5) is False


@pytest.mark.asyncio
async def test_upload_sends_files_category(monkeypatch, tmp_path):
    log = _fake_http(monkeypatch, kappa_client)
    files = []
    for n in ("s_t1.nii.gz", "s_t1_segmask.nii.gz"):
        (tmp_path / n).write_bytes(b"x")
        files.append(tmp_path / n)

    await kappa_client.upload_entity("tok", 26, 3, 357, "s", files, {"labels": ["lesion"]},
                                     files_category={"s_t1.nii.gz": "input",
                                                     "s_t1_segmask.nii.gz": "output"})

    body = json.loads(log[0]["data"]["new_dataset_entity"])
    assert body["filesCategory"] == {"s_t1.nii.gz": "input", "s_t1_segmask.nii.gz": "output"}
    assert body["dsEntityInfo"]["labels"] == ["lesion"]


@pytest.mark.asyncio
async def test_patch_marks_the_main_mask_as_output(monkeypatch, tmp_path):
    log = _fake_http(monkeypatch, kef)
    mask = tmp_path / "s_t1_segmask.nii.gz"
    mask.write_bytes(b"x")

    assert await kef.patch_file("tok", 26, 3, 357, "e1", "f1", mask)

    assert log[0]["params"] == {"file_category": "output"}


@pytest.mark.asyncio
async def test_add_files_sends_outputs_and_inputs_separately(monkeypatch, tmp_path):
    """Kappa applies one category to the whole request."""
    log = _fake_http(monkeypatch, kef)
    paths = []
    for n in ("s_t1c.nii.gz", "s_t1c_segmask.nii.gz", "s_t1c_segmask_labels.nii.gz"):
        (tmp_path / n).write_bytes(b"x")
        paths.append(tmp_path / n)

    assert await kef.add_files("tok", 26, 3, 357, "e1", paths)

    sent = {c["params"]["file_category"]: sorted(n for _, n in c["files"]) for c in log}
    assert sent == {"input": ["s_t1c.nii.gz", "s_t1c_segmask_labels.nii.gz"],
                    "output": ["s_t1c_segmask.nii.gz"]}


def _uploader(lesion):
    cfg = str(REPO / "configs" / "preprocessing_config.yaml")
    return kappa_uploader.KappaUploader(
        run_id="r", output_path="/tmp", token="t", user_id=26, user_type_id=3,
        lesion_type=lesion, preprocessing_config_path=cfg, dataset_id=357)


@pytest.mark.parametrize("lesion, labels", [
    ("multiple_sclerosis", ["lesion"]),
    ("glioblastoma", ["ed", "net", "et", "ncr"]),
])
def test_entity_info_carries_the_mask_classes(lesion, labels):
    up = _uploader(lesion)
    session = {"preprocessed": [Path("/o/s_t1.nii.gz")], "masks": [Path("/o/s_t1_segmask.nii.gz")],
               "quality_reports": [], "volume_report": None, "lobar_report": None,
               "lesion_stats_report": None, "mcdonald_report": None, "lesion_labels_mask": None}
    up._compute_study_hash = lambda data: "h"

    assert up._build_entity_info("s", session)["labels"] == labels


def test_kappa_labels_match_the_segmentation_manifests():
    """The web image has no services/, so lesion_types.yaml carries a copy of
    each service's mask classes. They must not drift apart."""
    lesion_types = yaml.safe_load((REPO / "configs" / "lesion_types.yaml").read_text())
    manifests = {}
    for path in (REPO / "services").glob("*/manifest.yaml"):
        m = yaml.safe_load(path.read_text())
        lesions = _find(m, "lesion_types") or []
        classes = _find(m, "classes")
        for lesion in lesions:
            if isinstance(classes, dict):
                manifests[lesion] = [v for k, v in sorted(classes.items()) if int(k) != 0]
    assert manifests, "no segmentation manifest with classes found"
    for lesion, classes in manifests.items():
        assert lesion_types[lesion]["kappa_labels"] == classes, lesion


def _find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for v in node.values():
            found = _find(v, key)
            if found is not None:
                return found
    return None


@pytest.mark.asyncio
async def test_validation_reports_a_refused_status(monkeypatch):
    import app
    import kappa_auth
    import validation_service

    monkeypatch.setattr(kappa_auth, "get_session", lambda sid, **k: {
        "kappa_token": "t", "user_id": 26, "user_type_id": 3,
        "first_name": "K", "last_name": "R"})
    monkeypatch.setattr(validation_service, "record_action", lambda **k: None)
    monkeypatch.setattr(validation_service, "get_current_votes",
                        lambda eid: {"confirms_count": 2, "total_votes": 2})

    async def refuse(**kw):
        return False, "Cannot mark as verified; missing required fields: labels"

    monkeypatch.setattr(kappa_client, "set_entity_status", refuse)

    req = app.ValidationActionRequest(session_id="s", entity_id="e1", dataset_id=357,
                                      action="confirm")
    out = await app.validation_action(req)

    assert out["kappa_status"] == 5
    assert out["kappa_status_updated"] is False
    assert "labels" in out["kappa_message"]
