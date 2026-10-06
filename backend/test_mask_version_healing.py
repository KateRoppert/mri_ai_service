"""The AI mask must not depend on a local file that moved.

mask_versions stores a local path and a Kappa file id. For AI masks the id
was never recorded, so the local path was the only route — and
register_ai_mask froze that path at first registration and never refreshed
it. A reprocessed patient, or a run directory that got cleaned up, left the
mask marked «недоступна» in the validation tab even though Kappa holds it.

Observed live on entity 4d72111f (2026-10-05): v1 pointed into
demo_workspace/input/sibms-21_07_1810/, a run folder deleted long before,
while Kappa served the very same mask as file 7626f912 for Slicer.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import mask_service


def _fresh(entity_id, tmp_path, name="sub-002_ses-002_t1_segmask.nii.gz"):
    """A v1 row whose file exists, as a first upload leaves it."""
    old = tmp_path / "run-july" / name
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(b"old")
    mask_service.register_ai_mask(
        entity_id=entity_id, dataset_id=158, file_path=str(old),
    )
    return old


def test_re_registering_refreshes_the_path(tmp_path):
    """A requeue regenerates the mask under a new run directory. Keeping the
    first path means pointing at data the pipeline has already replaced."""
    entity = "heal-path"
    _fresh(entity, tmp_path)

    new = tmp_path / "run-october" / "sub-002_ses-002_t1_segmask.nii.gz"
    new.parent.mkdir(parents=True, exist_ok=True)
    new.write_bytes(b"new")

    mask_service.register_ai_mask(
        entity_id=entity, dataset_id=158, file_path=str(new),
    )

    history = mask_service.get_mask_history(entity)
    assert len(history) == 1, "создалась вторая версия 1 вместо обновления"
    assert history[0]["file_path"] == str(new)


def test_re_registering_records_a_kappa_file_id(tmp_path):
    entity = "heal-id"
    _fresh(entity, tmp_path)

    mask_service.register_ai_mask(
        entity_id=entity, dataset_id=158,
        file_path=str(tmp_path / "x.nii.gz"), kappa_file_id="7626f912",
    )

    assert mask_service.get_mask_history(entity)[0]["kappa_file_id"] == "7626f912"


def test_an_existing_kappa_file_id_is_not_erased(tmp_path):
    """A later registration without an id must not throw away the route to
    Kappa we already have — that would undo the healing."""
    entity = "heal-keep"
    _fresh(entity, tmp_path)
    mask_service.register_ai_mask(
        entity_id=entity, dataset_id=158,
        file_path=str(tmp_path / "x.nii.gz"), kappa_file_id="7626f912",
    )

    mask_service.register_ai_mask(
        entity_id=entity, dataset_id=158, file_path=str(tmp_path / "y.nii.gz"),
    )

    assert mask_service.get_mask_history(entity)[0]["kappa_file_id"] == "7626f912"


def test_set_kappa_file_id_backfills_one_row(tmp_path):
    """Used to heal rows that predate this: the id is resolved from Kappa by
    filename and written back, so the next view proxies from Kappa."""
    entity = "heal-backfill"
    _fresh(entity, tmp_path)

    assert mask_service.set_kappa_file_id(entity, 1, "7626f912") is True
    assert mask_service.get_mask_history(entity)[0]["kappa_file_id"] == "7626f912"


def test_set_kappa_file_id_on_a_missing_version_is_false():
    assert mask_service.set_kappa_file_id("nope", 99, "x") is False


# --- Availability answered by Kappa, not by a local file --------------------

@pytest.mark.asyncio
async def test_a_mask_in_kappa_is_available_even_with_no_local_file(monkeypatch, tmp_path):
    """The validation tab serves masks out of Kappa; the local path is only a
    fallback. An AI mask with no recorded file id and a vanished local file
    was reported «недоступна» while Kappa held it all along."""
    import app

    entity = "avail-from-kappa"
    gone = tmp_path / "deleted-run" / "sub-002_ses-002_t1_segmask.nii.gz"
    gone.parent.mkdir(parents=True, exist_ok=True)
    gone.write_bytes(b"x")
    mask_service.register_ai_mask(
        entity_id=entity, dataset_id=158, file_path=str(gone))
    gone.unlink()                      # the run folder got cleaned up

    # The endpoint imports get_session inside the function (house style), so
    # the real module is what the local import resolves.
    monkeypatch.setattr("kappa_auth.get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26,
                                     "user_type_id": 1})

    async def _details(**kwargs):
        return {"files": [
            {"fileId": "7626f912",
             "fileName": "sub-002_ses-002_t1_segmask.nii.gz"},
        ]}
    monkeypatch.setattr("kappa_client.get_entity_details", _details)

    out = await app.get_mask_versions(entity, session_id="sid")

    v1 = out["versions"][0]
    assert v1["available"] is True
    # And the id is written back, so the next view proxies from Kappa
    # without asking it for the listing again.
    assert mask_service.get_mask_history(entity)[0]["kappa_file_id"] == "7626f912"


@pytest.mark.asyncio
async def test_a_mask_in_neither_place_is_still_unavailable(monkeypatch, tmp_path):
    """The red tag has to keep meaning something."""
    import app

    entity = "avail-nowhere"
    gone = tmp_path / "x" / "sub-003_ses-001_t1_segmask.nii.gz"
    gone.parent.mkdir(parents=True, exist_ok=True)
    gone.write_bytes(b"x")
    mask_service.register_ai_mask(
        entity_id=entity, dataset_id=158, file_path=str(gone))
    gone.unlink()

    # The endpoint imports get_session inside the function (house style), so
    # the real module is what the local import resolves.
    monkeypatch.setattr("kappa_auth.get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26,
                                     "user_type_id": 1})

    async def _details(**kwargs):
        return {"files": []}
    monkeypatch.setattr("kappa_client.get_entity_details", _details)

    out = await app.get_mask_versions(entity, session_id="sid")
    assert out["versions"][0]["available"] is False


@pytest.mark.asyncio
async def test_unreachable_kappa_falls_back_to_what_we_know(monkeypatch, tmp_path):
    """Asking Kappa is an improvement, not a dependency: the modal must still
    open when Kappa is down."""
    import app

    entity = "avail-offline"
    here = tmp_path / "run" / "sub-004_ses-001_t1_segmask.nii.gz"
    here.parent.mkdir(parents=True, exist_ok=True)
    here.write_bytes(b"x")
    mask_service.register_ai_mask(
        entity_id=entity, dataset_id=158, file_path=str(here))

    # The endpoint imports get_session inside the function (house style), so
    # the real module is what the local import resolves.
    monkeypatch.setattr("kappa_auth.get_session",
                        lambda sid: {"kappa_token": "t", "user_id": 26,
                                     "user_type_id": 1})

    async def _boom(**kwargs):
        raise RuntimeError("Kappa недоступна")
    monkeypatch.setattr("kappa_client.get_entity_details", _boom)

    out = await app.get_mask_versions(entity, session_id="sid")
    assert out["versions"][0]["available"] is True   # local file is there
