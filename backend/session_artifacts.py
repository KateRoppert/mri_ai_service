"""Removing one session's stage outputs so the next run rebuilds it.

`skip_existing` makes every stage skip a patient whose output already
exists. That is what makes a correction pointless on its own: the doctor
fixes the modality set and the pipeline skips right past it. Deleting the
session's outputs is how the correction reaches the data.

What is deleted is only ever output. `bids_organized/` holds the corrected
assignment and is the pipeline's input, so it stays.
"""
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

# Every stage writes per-patient output as {stage}/{sub-XXX}/{ses-YYY}/.
STAGE_DIRS = (
    "metadata",
    "nifti",
    "preprocessed",
    "quality_reports",
    "segmentation",
    "transformations",
)

_PATIENT_RE = re.compile(r"^sub-\d+$")
_SESSION_RE = re.compile(r"^ses-\d+$")


def delete_session_artifacts(
    output_path: str, patient_id: str, session_id: str
) -> List[str]:
    """Delete one session's output under every stage directory.

    Returns the paths actually removed. Raises ValueError for identifiers
    that are not plain BIDS ids — these come out of a JSON file, and a
    delete must not be steerable by its contents.
    """
    if not _PATIENT_RE.match(patient_id or ""):
        raise ValueError(f"Invalid patient_id: {patient_id!r}")
    if not _SESSION_RE.match(session_id or ""):
        raise ValueError(f"Invalid session_id: {session_id!r}")

    base = Path(output_path)
    removed: List[str] = []

    for stage in STAGE_DIRS:
        session_dir = base / stage / patient_id / session_id
        if not session_dir.is_dir():
            continue
        shutil.rmtree(session_dir)
        removed.append(str(session_dir))

        # Drop the patient directory too once its last session is gone, so
        # the tree does not accumulate empty shells.
        patient_dir = base / stage / patient_id
        if patient_dir.is_dir() and not any(patient_dir.iterdir()):
            patient_dir.rmdir()

    if removed:
        logger.info(
            "Удалены результаты %s/%s для переобработки: %d папок",
            patient_id, session_id, len(removed),
        )
    return removed


def purge_sessions_marked_for_reprocess(output_path: str) -> Dict[str, List[str]]:
    """Delete the outputs of every session flagged needs_reprocess, and clear
    the flags. Returns {"sub-001/ses-001": [removed paths]}.

    A session with no outputs yet simply yields an empty list — the flag is
    set whenever the assignment changed, without asking whether the session
    was ever processed, so a no-op here is expected rather than exceptional.
    """
    mapping_file = Path(output_path) / "bids_organized" / "dataset_mapping.json"
    if not mapping_file.exists():
        return {}

    try:
        with open(mapping_file, "r", encoding="utf-8") as fh:
            mapping = json.load(fh)
    except (OSError, ValueError) as exc:
        logger.error("Не удалось прочитать %s: %s", mapping_file, exc)
        return {}

    purged: Dict[str, List[str]] = {}
    changed = False

    for patient_id, patient in (mapping.get("patients") or {}).items():
        for session_id, session in (patient.get("sessions") or {}).items():
            if not session.get("needs_reprocess"):
                continue
            try:
                purged[f"{patient_id}/{session_id}"] = delete_session_artifacts(
                    output_path, patient_id, session_id
                )
            except ValueError as exc:
                logger.error("Пропущена сессия с некорректным идентификатором: %s", exc)
                continue
            session["needs_reprocess"] = False
            changed = True

    if changed:
        with open(mapping_file, "w", encoding="utf-8") as fh:
            json.dump(mapping, fh, indent=2, ensure_ascii=False)

    return purged
