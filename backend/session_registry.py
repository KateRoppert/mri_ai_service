"""Registering a run's sessions locally, the moment the run finishes.

KI-059: `GET /api/longitudinal/{patient}` builds its points from
`patient_registry`, and those rows used to be written ONLY by
KappaUploader._register_patient — after a successful upload. A run with no
Kappa session, or one whose upload failed, never got rows, so the doctor
saw no dynamics at all even though every stage-08 report was on disk.

A session exists because the pipeline produced it. Its Kappa ids are an
attribute acquired later, not its reason to exist. So the row is written
here, and the uploader's own register_patient call later fills in
`kappa_entity_id` — it upserts by study_hash, which both sides compute
through session_identity precisely so they land on the same row.
"""
import logging
from typing import Any

from patient_registry import register_patient

import session_identity

logger = logging.getLogger(__name__)


def register_run_sessions(run: Any) -> int:
    """Write a `patient_registry` row for every session this run produced.

    Returns how many were registered. Never raises: this is bookkeeping
    around a run that has already succeeded, and a registry problem must not
    be reported as a pipeline failure.
    """
    output_path = getattr(run, "output_path", None)
    if not output_path:
        return 0

    sessions = session_identity.discover_preprocessed_sessions(output_path)
    if not sessions:
        return 0

    lesion_type = getattr(run, "lesion_type", None) or ""
    # Fixed at run start by numbering.py when there is a Kappa session, so we
    # know it well before any upload. It matters: _resolve_longitudinal_records
    # scopes a bids_id lookup by dataset, and a row left NULL here would be
    # invisible to it. NULL stays NULL for CLI runs, where nothing scopes.
    dataset_id = getattr(run, "kappa_dataset_id", None)
    run_id = getattr(run, "run_id", "") or ""

    registered = 0
    for session_key, preprocessed in sorted(sessions.items()):
        study_hash = session_identity.study_hash(output_path, preprocessed)
        if not study_hash:
            # No stable identity, and register_patient keys on it — a row
            # under an empty hash would collide with every other such row.
            logger.warning(
                "Реестр: %s пропущена, нет метаданных для study_hash", session_key,
            )
            continue

        fields = session_identity.registry_fields(output_path, preprocessed)
        try:
            register_patient(
                bids_id=session_key,
                study_hash=study_hash,
                original_patient_id=fields.get("PatientID", ""),
                patient_name=fields.get("PatientName", ""),
                scan_date=fields.get("StudyDate", ""),
                study_instance_uid=fields.get("StudyInstanceUID", ""),
                kappa_entity_id=None,
                kappa_dataset_id=dataset_id,
                pipeline_run_id=run_id,
                lesion_type=lesion_type,
            )
            registered += 1
        except Exception as exc:  # noqa: BLE001 — see docstring
            logger.error(
                "Реестр: не удалось записать %s: %s", session_key, exc,
            )

    if registered:
        logger.info(
            "Реестр: записано сессий — %d (прогон %s)", registered, run_id,
        )
    return registered
