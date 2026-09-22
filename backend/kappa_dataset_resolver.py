"""
One place that answers "which Kappa dataset does this run belong to, and what
number comes next in it".

Both ends of a run need this: the start, to fix the numbering scope before
Stage 01 issues any id, and the upload, to send entities to the dataset those
ids were issued for. It used to live inside KappaUploader, which is why the
two ends could disagree when `current` was repointed mid-run.
"""

import logging
import re
from typing import Optional

from kappa_client import create_dataset, get_dataset_entities
from kappa_dataset_mapping import get_dataset_id, set_dataset_id

logger = logging.getLogger(__name__)

_SUBJECT_RE = re.compile(r"^sub-(\d+)")


def highest_subject_number(entities: list) -> int:
    """Highest sub-NNN among a dataset's entity names (0 if there are none).

    Entity names are the session keys we upload ("sub-001_ses-001"), so the
    dataset itself tells us what numbers are taken — including ones issued on
    another machine.
    """
    highest = 0
    for entity in entities or []:
        name = (entity or {}).get("dsEntityName") or ""
        match = _SUBJECT_RE.match(name)
        if match:
            highest = max(highest, int(match.group(1)))
    return highest


async def resolve_or_create(
    token: str,
    user_id: int,
    user_type_id: int,
    lesion_type: str,
    preprocessing_id: str,
    create: bool = True,
) -> Optional[int]:
    """The dataset for this (user, lesion, preprocessing), creating one if the
    mapping has none and `create` is set."""
    dataset_id = get_dataset_id(user_id, lesion_type, preprocessing_id)
    if dataset_id is not None:
        return dataset_id
    if not create:
        return None

    short_id = preprocessing_id[:8]
    new_id = await create_dataset(
        token=token,
        user_id=user_id,
        user_type_id=user_type_id,
        dataset_name=f"{lesion_type}_{short_id}",
        dataset_short_info=f"Lesion: {lesion_type}, Preprocessing: {preprocessing_id}",
        dataset_type=1,
        # Kappa requires datasetTags to include at least one predefined ML
        # tag. "Image Segmentation" is the applicable one.
        dataset_tags=f"Image Segmentation,mri,{lesion_type}",
    )
    if new_id is not None:
        set_dataset_id(user_id, lesion_type, preprocessing_id, new_id)
        logger.info("New dataset created: id=%d", new_id)
    return new_id


async def dataset_floor(
    token: str, user_id: int, user_type_id: int, dataset_id: int
) -> Optional[int]:
    """Highest subject number the dataset already holds, or None if Kappa
    could not be asked. None means unknown — never treat it as 0."""
    try:
        entities = await get_dataset_entities(
            token=token, user_id=user_id,
            user_type_id=user_type_id, dataset_id=dataset_id,
        )
    except Exception as exc:
        logger.warning("Could not read dataset %s from Kappa: %s", dataset_id, exc)
        return None
    if entities is None:
        return None
    return highest_subject_number(entities)
