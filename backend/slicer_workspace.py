"""Materialize Kappa entity files into a host-visible cache for 3D Slicer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import json
import logging
import re

from kappa_client import download_entity_file, get_entity_details

logger = logging.getLogger(__name__)
_VERSIONED_SEGMASK = re.compile(r"_segmask_v(\d+)\.nii\.gz$", re.IGNORECASE)


class MaskMissingError(LookupError):
    """Entity has no atlas-space segmask suitable for Slicer."""


@dataclass
class SlicerWorkspace:
    image_paths: List[str]
    mask_path: str
    cache_dir: str
    patient_id: str
    session_id: str
    lesion_type: str


def parse_bids_id(bids_id: str) -> Tuple[str, str]:
    bids = bids_id or ""
    ses_idx = bids.find("_ses-")
    if ses_idx != -1:
        return bids[:ses_idx], bids[ses_idx + 1 :]
    return bids, ""


def _is_atlas_segmask(name: str) -> bool:
    if not name.endswith(".nii.gz"):
        return False
    if "_native_" in name or "_labels" in name:
        return False
    return "_segmask" in name


def select_mask_filename(
    file_names: Sequence[str],
    prediction_files: Optional[Iterable[str]] = None,
    requested_version: Optional[int] = None,
) -> str:
    atlas = [n for n in file_names if _is_atlas_segmask(n)]
    versioned: List[Tuple[int, str]] = []
    unversioned: List[str] = []
    for name in atlas:
        match = _VERSIONED_SEGMASK.search(name)
        if match:
            versioned.append((int(match.group(1)), name))
        else:
            unversioned.append(name)

    if requested_version is not None:
        if requested_version == 1:
            predictions = list(prediction_files or [])
            for name in predictions:
                if name in unversioned:
                    return name
            if unversioned:
                return unversioned[0]
            for ver, name in versioned:
                if ver == 1:
                    return name
            raise MaskMissingError("в сущности нет маски версии 1")
        for ver, name in versioned:
            if ver == requested_version:
                return name
        raise MaskMissingError(f"в сущности нет маски версии {requested_version}")

    if versioned:
        versioned.sort(key=lambda item: item[0])
        return versioned[-1][1]

    predictions = list(prediction_files or [])
    for name in predictions:
        if name in atlas or _is_atlas_segmask(name):
            return name
    if atlas:
        return atlas[0]
    raise MaskMissingError("в сущности нет маски")


def entity_cache_dir(cache_root: Path, entity_id: str) -> Path:
    return Path(cache_root) / entity_id


def _file_size_from_kappa(entry: Dict[str, Any]) -> Optional[int]:
    for key in ("fileSize", "file_size", "size"):
        value = entry.get(key)
        if isinstance(value, int) and value >= 0:
            return value
    return None


async def materialize_from_kappa(
    session: Dict[str, Any],
    dataset_id: int,
    entity_id: str,
    cache_root: Path,
    requested_version: Optional[int] = None,
) -> SlicerWorkspace:
    details = await get_entity_details(
        token=session["kappa_token"],
        user_id=session["user_id"],
        user_type_id=session["user_type_id"],
        dataset_id=dataset_id,
        entity_id=entity_id,
    )
    if not details:
        raise MaskMissingError("сущность не найдена в Kappa")

    info = details.get("dsEntityInfo") or {}
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except Exception:
            info = {}

    files = details.get("files") or []
    by_name = {f.get("fileName"): f for f in files if f.get("fileName")}
    data_files = list(info.get("data_files") or [])
    prediction_files = list(info.get("prediction_files") or [])
    mask_name = select_mask_filename(
        list(by_name.keys()),
        prediction_files=prediction_files,
        requested_version=requested_version,
    )

    wanted = list(data_files) + [mask_name]
    cache_dir = entity_cache_dir(cache_root, entity_id)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for name in wanted:
        meta = by_name.get(name)
        if not meta or not meta.get("fileId"):
            raise MaskMissingError(f"файл не найден в сущности: {name}")
        dest = cache_dir / name
        expected_size = _file_size_from_kappa(meta)
        if dest.exists() and expected_size is not None and dest.stat().st_size == expected_size:
            logger.info("Slicer cache hit: entity=%s file=%s", entity_id, name)
            continue
        content = await download_entity_file(
            token=session["kappa_token"],
            user_id=session["user_id"],
            user_type_id=session["user_type_id"],
            dataset_id=dataset_id,
            file_id=meta["fileId"],
        )
        if content is None:
            raise ConnectionError(f"не удалось скачать {name}")
        dest.write_bytes(content)
        logger.info("Slicer cache write: entity=%s file=%s bytes=%d", entity_id, name, len(content))

    patient_id, session_id = parse_bids_id(info.get("bids_id") or "")
    return SlicerWorkspace(
        image_paths=[str(cache_dir / name) for name in data_files],
        mask_path=str(cache_dir / mask_name),
        cache_dir=str(cache_dir),
        patient_id=patient_id,
        session_id=session_id,
        lesion_type=info.get("lesion_type") or "glioblastoma",
    )
