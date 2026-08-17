"""Tests for Kappa → Slicer cache file selection and materialize.

Production change that would fail these tests: picking a native/labels
mask, ignoring *_segmask_vN, or re-downloading a cache file of matching size.
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from slicer_workspace import (
    MaskMissingError,
    materialize_from_kappa,
    parse_bids_id,
    select_mask_filename,
)


def test_select_mask_prefers_highest_versioned_segmask():
    names = [
        "sub-001_ses-001_t1.nii.gz",
        "sub-001_ses-001_t1_segmask.nii.gz",
        "sub-001_ses-001_t1_segmask_v2.nii.gz",
        "sub-001_ses-001_t1_segmask_v3.nii.gz",
        "sub-001_ses-001_t1_segmask_native_t1.nii.gz",
        "sub-001_ses-001_t1_segmask_labels.nii.gz",
    ]
    assert select_mask_filename(names, prediction_files=["sub-001_ses-001_t1_segmask.nii.gz"]) == (
        "sub-001_ses-001_t1_segmask_v3.nii.gz"
    )


def test_select_mask_falls_back_to_prediction_file_when_no_versioned():
    names = [
        "sub-001_ses-001_t1.nii.gz",
        "sub-001_ses-001_t1_segmask.nii.gz",
        "sub-001_ses-001_t1_segmask_native_t1.nii.gz",
    ]
    assert select_mask_filename(
        names, prediction_files=["sub-001_ses-001_t1_segmask.nii.gz"]
    ) == "sub-001_ses-001_t1_segmask.nii.gz"


def test_select_mask_raises_when_only_native_or_labels():
    names = [
        "sub-001_ses-001_t1_segmask_native_t1.nii.gz",
        "sub-001_ses-001_t1_segmask_labels.nii.gz",
    ]
    with pytest.raises(MaskMissingError):
        select_mask_filename(names, prediction_files=[])


def test_select_mask_requested_version_picks_that_file_not_highest():
    names = [
        "sub-001_ses-001_t1_segmask.nii.gz",
        "sub-001_ses-001_t1_segmask_v2.nii.gz",
        "sub-001_ses-001_t1_segmask_v3.nii.gz",
        "sub-001_ses-001_t1_segmask_native_t1.nii.gz",
    ]
    assert select_mask_filename(names, requested_version=2) == (
        "sub-001_ses-001_t1_segmask_v2.nii.gz"
    )


def test_select_mask_requested_version_1_is_unversioned_ai_mask():
    names = [
        "sub-001_ses-001_t1_segmask.nii.gz",
        "sub-001_ses-001_t1_segmask_v2.nii.gz",
        "sub-001_ses-001_t1_segmask_v3.nii.gz",
    ]
    assert select_mask_filename(names, requested_version=1) == (
        "sub-001_ses-001_t1_segmask.nii.gz"
    )


def test_select_mask_requested_version_missing_raises():
    names = [
        "sub-001_ses-001_t1_segmask.nii.gz",
        "sub-001_ses-001_t1_segmask_v2.nii.gz",
    ]
    with pytest.raises(MaskMissingError, match="версии 5"):
        select_mask_filename(names, requested_version=5)


def test_parse_bids_id_splits_patient_and_session():
    assert parse_bids_id("sub-001_ses-002") == ("sub-001", "ses-002")


def test_parse_bids_id_without_session():
    assert parse_bids_id("sub-001") == ("sub-001", "")


def _entity_details():
    return {
        "dsEntityInfo": {
            "bids_id": "sub-001_ses-002",
            "lesion_type": "multiple_sclerosis",
            "data_files": [
                "sub-001_ses-002_t1.nii.gz",
                "sub-001_ses-002_t2fl.nii.gz",
            ],
            "prediction_files": ["sub-001_ses-002_t1_segmask.nii.gz"],
        },
        "files": [
            {"fileName": "sub-001_ses-002_t1.nii.gz", "fileId": "vol-t1", "fileSize": 4},
            {"fileName": "sub-001_ses-002_t2fl.nii.gz", "fileId": "vol-fl", "fileSize": 4},
            {"fileName": "sub-001_ses-002_t1_segmask.nii.gz", "fileId": "mask-ai", "fileSize": 3},
            {"fileName": "sub-001_ses-002_t1_segmask_v2.nii.gz", "fileId": "mask-v2", "fileSize": 5},
            {"fileName": "sub-001_ses-002_t1_segmask_native_t1.nii.gz", "fileId": "mask-nat", "fileSize": 3},
            {"fileName": "volume_report.json", "fileId": "json-1", "fileSize": 2},
        ],
    }


def test_materialize_downloads_volumes_and_highest_versioned_mask(tmp_path):
    session = {"kappa_token": "t", "user_id": 1, "user_type_id": 2}
    downloads = []

    async def fake_download(token, user_id, user_type_id, dataset_id, file_id):
        downloads.append(file_id)
        return f"data-{file_id}".encode()

    async def _run():
        with patch("slicer_workspace.get_entity_details", new=AsyncMock(return_value=_entity_details())), \
             patch("slicer_workspace.download_entity_file", new=fake_download):
            ws = await materialize_from_kappa(
                session, dataset_id=158, entity_id="ent-1", cache_root=tmp_path
            )
        return ws

    ws = asyncio.run(_run())
    assert ws.patient_id == "sub-001"
    assert ws.session_id == "ses-002"
    assert ws.lesion_type == "multiple_sclerosis"
    assert Path(ws.mask_path).name == "sub-001_ses-002_t1_segmask_v2.nii.gz"
    assert Path(ws.mask_path).exists()
    assert {Path(p).name for p in ws.image_paths} == {
        "sub-001_ses-002_t1.nii.gz",
        "sub-001_ses-002_t2fl.nii.gz",
    }
    assert "mask-nat" not in downloads
    assert "json-1" not in downloads
    assert "mask-v2" in downloads
    assert "mask-ai" not in downloads


def test_materialize_requested_version_downloads_that_mask_not_highest(tmp_path):
    session = {"kappa_token": "t", "user_id": 1, "user_type_id": 2}
    downloads = []

    async def fake_download(token, user_id, user_type_id, dataset_id, file_id):
        downloads.append(file_id)
        return f"data-{file_id}".encode()

    async def _run():
        with patch("slicer_workspace.get_entity_details", new=AsyncMock(return_value=_entity_details())), \
             patch("slicer_workspace.download_entity_file", new=fake_download):
            ws = await materialize_from_kappa(
                session,
                dataset_id=158,
                entity_id="ent-1",
                cache_root=tmp_path,
                requested_version=1,
            )
        return ws

    ws = asyncio.run(_run())
    assert Path(ws.mask_path).name == "sub-001_ses-002_t1_segmask.nii.gz"
    assert "mask-ai" in downloads
    assert "mask-v2" not in downloads


def test_materialize_skips_download_when_cache_size_matches(tmp_path):
    session = {"kappa_token": "t", "user_id": 1, "user_type_id": 2}
    cache_dir = tmp_path / "ent-1"
    cache_dir.mkdir()
    details = _entity_details()
    for f in details["files"]:
        if f["fileId"] in {"vol-t1", "vol-fl", "mask-v2"}:
            (cache_dir / f["fileName"]).write_bytes(b"x" * f["fileSize"])

    downloads = []

    async def fake_download(*args, **kwargs):
        downloads.append(kwargs.get("file_id") or args[-1])
        return b"new"

    async def _run():
        with patch("slicer_workspace.get_entity_details", new=AsyncMock(return_value=details)), \
             patch("slicer_workspace.download_entity_file", new=fake_download):
            await materialize_from_kappa(
                session, dataset_id=158, entity_id="ent-1", cache_root=tmp_path
            )

    asyncio.run(_run())
    assert downloads == []
