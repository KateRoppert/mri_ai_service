"""A session's DICOM identity, read off disk.

Two things need it and must agree: the Kappa uploader, which uses the study
hash to decide whether a session is already in the dataset, and the local
registrar, which writes `patient_registry` rows when a run finishes.
`register_patient` upserts by that hash, so if the two computed it
differently the upload would create a second row instead of filling in the
first — and the longitudinal view would show one session twice.

The two consumers select differently, and that is kept exactly as it was:
`study_hash` skips a metadata file whose PatientID or StudyInstanceUID is
empty and keeps looking, while `registry_fields` takes the first readable
file as it is. Only the WALK is shared. Unifying the selection would have
changed the hash — the dedup key for every upload — for the sake of
tidiness, and that key has already caused one incident when it came back
None.
"""
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)


def extract_session_key(filepath: Path) -> Optional[str]:
    """`sub-XXX_ses-YYY` from a file's DIRECTORIES, never from its name."""
    sub = None
    ses = None
    for part in filepath.parts[:-1]:
        if part.startswith("sub-"):
            sub = part
        elif part.startswith("ses-"):
            ses = part
    if sub and ses:
        return f"{sub}_{ses}"
    return None


def iter_identifications(
    output_path: Path, preprocessed: List[Path]
) -> Iterator[Dict[str, Any]]:
    """The `identification` block of every metadata file for this session.

    Walks the preprocessed volumes and, for each, the metadata written
    beside it by stage 01 (`metadata/{sub}/{ses}/anat/{modality}/*.json`).
    An unreadable file is skipped, not raised: metadata is a convenience
    here, and one bad file must not hide the rest.
    """
    metadata_dir = Path(output_path) / "metadata"
    if not metadata_dir.exists():
        return

    for prep_file in preprocessed or []:
        session_key = extract_session_key(prep_file)
        if not session_key:
            continue
        sub, ses = session_key.split("_", 1)
        modality = prep_file.stem.replace(".nii", "").split("_")[-1]

        meta_dir = metadata_dir / sub / ses / "anat" / modality
        if not meta_dir.exists():
            continue
        for meta_file in meta_dir.glob("*.json"):
            try:
                with open(meta_file, "r") as fh:
                    meta = json.load(fh)
            except Exception as exc:  # noqa: BLE001 — see docstring
                logger.warning("Failed to read metadata %s: %s", meta_file, exc)
                continue
            yield meta.get("identification", {}) or {}


def study_hash(output_path: Path, preprocessed: List[Path]) -> Optional[str]:
    """Stable identity of the STUDY: sha256 of PatientID:StudyInstanceUID.

    Note this identifies the DICOM study, not the images — reprocessing the
    same study with a corrected modality set gives the same hash, which is
    why a reprocessed patient supersedes rather than duplicates in Kappa.
    """
    for ident in iter_identifications(output_path, preprocessed):
        patient_id = ident.get("PatientID", "")
        study_uid = ident.get("StudyInstanceUID", "")
        if patient_id and study_uid:
            raw = f"{patient_id}:{study_uid}"
            return hashlib.sha256(raw.encode()).hexdigest()[:16]
    return None


def registry_fields(output_path: Path, preprocessed: List[Path]) -> Dict[str, str]:
    """The four fields `patient_registry` keeps, from the first readable file."""
    for ident in iter_identifications(output_path, preprocessed):
        # Some exports write PatientName as a list of characters.
        patient_name = ident.get("PatientName", "")
        if isinstance(patient_name, list):
            patient_name = "".join(str(c) for c in patient_name)
        return {
            "PatientID": ident.get("PatientID", ""),
            "PatientName": patient_name,
            "StudyDate": ident.get("StudyDate", ""),
            "StudyInstanceUID": ident.get("StudyInstanceUID", ""),
        }
    return {}


def discover_preprocessed_sessions(output_path: Path) -> Dict[str, List[Path]]:
    """{session_key: [preprocessed volumes]} for a run's output.

    Keyed the same way KappaUploader._discover_sessions keys its own map, so
    the registry and the upload cannot disagree about which sessions exist.
    Deliberately does NOT require segmentation output: a session that was
    preprocessed and then failed later still exists and still belongs in the
    registry.
    """
    preprocessed_dir = Path(output_path) / "preprocessed"
    if not preprocessed_dir.exists():
        return {}

    sessions: Dict[str, List[Path]] = {}
    for nifti in sorted(preprocessed_dir.rglob("*.nii.gz")):
        key = extract_session_key(nifti)
        if key:
            sessions.setdefault(key, []).append(nifti)
    return sessions
