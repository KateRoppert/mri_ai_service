"""
Decide the BIDS numbering scope for a run, once, at start.

Stage 01 issues subject ids before anything is uploaded, so the scope has to
be fixed before the pipeline starts. Resolving it here — rather than at
upload, where it used to happen — is also what keeps numbering and upload
pointed at the same Kappa dataset even if `current` in
configs/kappa_datasets.yaml is repointed while the run is in flight.

See docs/superpowers/specs/2026-09-21-bids-numbering-per-dataset-design.md.
"""

import logging
from typing import Optional, Tuple

from kappa_auth import get_session
from kappa_dataset_resolver import dataset_floor, resolve_or_create
from pipeline_monitor import PREPROCESSING_CONFIG
from preprocessing_version import compute_preprocessing_id
from utils.bids_allocator import dataset_scope, local_scope, pending_scope, set_floor

logger = logging.getLogger(__name__)


async def scope_for_run(
    run_id: str, lesion_type: str, kappa_session_id: Optional[str]
) -> Tuple[str, Optional[int], Optional[str]]:
    """(scope, dataset_id, warning) for a run about to start.

    Never raises: a run must start even when Kappa is down. `warning` is a
    user-facing message to surface in the start response when set; None means
    nothing worth telling the operator happened.
    """
    if not kappa_session_id:
        return local_scope(lesion_type), None, None

    session = get_session(kappa_session_id)
    if not session:
        return local_scope(lesion_type), None, None

    preprocessing_id = compute_preprocessing_id(str(PREPROCESSING_CONFIG))
    try:
        dataset_id = await resolve_or_create(
            token=session["kappa_token"], user_id=session["user_id"],
            user_type_id=session["user_type_id"], lesion_type=lesion_type,
            preprocessing_id=preprocessing_id,
        )
    except Exception as exc:
        logger.warning("Kappa unreachable at run start: %s", exc)
        return (pending_scope(run_id), None,
                "Kappa недоступна: номера будут закреплены за датасетом при выгрузке")

    if dataset_id is None:
        return (pending_scope(run_id), None,
                "Датасет в Kappa не создан: будет создан при выгрузке")

    scope = dataset_scope(dataset_id)
    floor = await dataset_floor(
        token=session["kappa_token"], user_id=session["user_id"],
        user_type_id=session["user_type_id"], dataset_id=dataset_id,
    )
    if floor is not None:
        set_floor(scope, floor)
    return scope, dataset_id, None


def scope_from_dataset_id(dataset_id: Optional[int], lesion_type: str) -> str:
    """Scope for a resumed/requeued run: no Kappa call, just the parent run's
    already-fixed dataset (or local numbering if it never had one)."""
    if dataset_id is not None:
        return dataset_scope(dataset_id)
    return local_scope(lesion_type)
