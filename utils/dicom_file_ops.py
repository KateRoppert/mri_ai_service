"""Shared DICOM file operations used by both stage 01 (01_reorganize_folders.py)
and the backend API (incomplete-patient manual relabel endpoint)."""
import shutil
from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING

import pydicom

if TYPE_CHECKING:
    from scripts.metadata_extractor import MetadataExtractor

MODALITY_BIDS_SUFFIX: Dict[str, str] = {
    't1':   'T1w',
    't1c':  'T1wCE',
    't2':   'T2w',
    't2fl': 'FLAIR',
}


def is_dicom_file(path: Path) -> bool:
    """
    True if `path` is a DICOM file, independent of extension (KI-029).

    Real clinical exports are not always named ``*.dcm`` — some vendors
    write extensionless files (e.g. ``23831328``), others use ``.IMA`` or
    ``.dicom``. Extension alone is unreliable both ways: it misses files
    without ``.dcm``, and (in theory) could match a non-DICOM file someone
    renamed to ``.dcm``.

    Fast path: files already named ``*.dcm`` are trusted directly, so
    datasets that already use this convention (the common case) pay no
    extra I/O. Anything else is checked via the standard DICOM Part-10
    magic marker: a 128-byte preamble followed by ``b'DICM'`` at offset
    128. This is authoritative — unrelated files (README.txt, DICOMDIR,
    thumbnails) will not carry it by coincidence — and correctly
    recognizes extensionless DICOM and other extensions uniformly.
    """
    if path.suffix.lower() == '.dcm':
        return True
    try:
        with open(path, 'rb') as f:
            header = f.read(132)
    except OSError:
        return False
    return len(header) == 132 and header[128:132] == b'DICM'


def find_dicom_files(directory: Path, recursive: bool = True) -> List[Path]:
    """All DICOM files under `directory`, any extension or none (see is_dicom_file)."""
    pattern = '**/*' if recursive else '*'
    return sorted(f for f in directory.glob(pattern) if f.is_file() and is_dicom_file(f))


def copy_and_anonymize_series(
    source_files: List[Path],
    target_dir: Path,
    patient_id: str,
    session_id: str,
    modality: str,
    metadata_extractor: Optional['MetadataExtractor'] = None,
    logger=None,
) -> int:
    """
    Copy DICOM files to target_dir with BIDS naming, anonymizing each file
    if metadata_extractor is provided (else a plain copy). Caller is
    responsible for calling find_dicom_files() to build source_files and
    for creating target_dir beforehand.

    Returns:
        Number of files successfully copied.
    """
    bids_suffix = MODALITY_BIDS_SUFFIX.get(modality, modality.upper())
    copied = 0
    for idx, source_file in enumerate(source_files, 1):
        target_name = f"{patient_id}_{session_id}_{bids_suffix}_{idx:04d}.dcm"
        target_path = target_dir / target_name
        try:
            if metadata_extractor:
                dcm = pydicom.dcmread(str(source_file), force=True)
                removed = metadata_extractor.anonymize_dicom(dcm)
                dcm.save_as(str(target_path))
                if idx == 1 and logger:
                    logger.info(
                        f"    Anonymized: removed {len(removed)} tags: "
                        f"{', '.join(removed)}"
                    )
            else:
                shutil.copy2(source_file, target_path)
            copied += 1
        except Exception as e:
            if logger:
                logger.error(f"Failed to process {source_file}: {e}")
    return copied
