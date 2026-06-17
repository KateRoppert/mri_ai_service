"""Config-driven selection of a skull stripper, with availability-based fallback,
plus the per-subject orchestration used by Stage 05."""

import logging
from pathlib import Path

from .base import SkullStripperBase, apply_brain_mask
from .bet import BetStripper

logger = logging.getLogger(__name__)

# method string -> class. Extended as tools are added (hdbet, synthstrip, ...).
STRIPPER_REGISTRY: dict[str, type[SkullStripperBase]] = {
    "bet": BetStripper,
}


def get_stripper(method: str) -> SkullStripperBase:
    """Instantiate the stripper registered under `method`."""
    cls = STRIPPER_REGISTRY.get(method)
    if cls is None:
        raise ValueError(
            f"Unknown skull stripping method '{method}'. "
            f"Available: {sorted(STRIPPER_REGISTRY)}"
        )
    return cls()


def resolve_stripper(params: dict) -> SkullStripperBase:
    """Return the primary stripper if available, else the fallback. Raise if neither."""
    method = params.get("method", "bet")
    primary = get_stripper(method)
    if primary.is_available():
        return primary

    fallback_method = params.get("fallback_method")
    if not fallback_method or fallback_method == method:
        raise RuntimeError(
            f"Skull stripper '{method}' unavailable and no usable fallback configured."
        )
    logger.warning(
        f"Skull stripper '{method}' unavailable; "
        f"falling back to '{fallback_method}'."
    )
    try:
        fallback = get_stripper(fallback_method)
    except ValueError as exc:
        raise RuntimeError(
            f"Skull stripper '{method}' unavailable and fallback '{fallback_method}' "
            f"is not registered. {exc}"
        ) from exc
    if not fallback.is_available():
        raise RuntimeError(
            f"Neither '{method}' nor fallback '{fallback_method}' is available."
        )
    return fallback


def process_subject_skull_stripping(subject_dir: Path, output_dir: Path,
                                    transform_dir: Path, modalities: list,
                                    params: dict) -> dict:
    """Create a brain mask on the reference modality via the configured stripper,
    then apply it to the remaining modalities. Signature matches Stage 05."""
    results = {}
    subject_id = subject_dir.parent.parent.name
    session_id = subject_dir.parent.name
    logger.info(f"Processing {subject_id}/{session_id} - Skull Stripping")

    stripper = resolve_stripper(params)
    logger.info(f"Using skull stripper: {stripper.name}")

    reference_modality = params.get("reference_modality", "t1c")
    ref_pattern = f"{subject_id}_{session_id}_{reference_modality}.nii.gz"
    ref_files = list(subject_dir.glob(ref_pattern))
    if not ref_files:
        msg = f"Reference modality {reference_modality} not found"
        logger.error(msg)
        return {"success": False, "error": msg}
    ref_file = ref_files[0]

    ref_output = output_dir / subject_id / session_id / "anat" / ref_pattern
    mask_pattern = f"{subject_id}_{session_id}_brain_mask.nii.gz"
    mask_path = transform_dir / subject_id / session_id / "anat" / mask_pattern
    mask_path.parent.mkdir(parents=True, exist_ok=True)

    strip_result = stripper.strip(ref_file, ref_output, mask_path, params)
    results[reference_modality] = strip_result
    if not strip_result.get("success"):
        logger.error(f"Failed to create brain mask on {reference_modality}")
        return results

    if params.get("apply_to_all", True):
        for modality in modalities:
            if modality == reference_modality:
                continue
            modal_pattern = f"{subject_id}_{session_id}_{modality}.nii.gz"
            modal_files = list(subject_dir.glob(modal_pattern))
            if not modal_files:
                logger.warning(f"Modality {modality} not found, skipping")
                results[modality] = {"success": False, "error": "File not found"}
                continue
            modal_output = output_dir / subject_id / session_id / "anat" / modal_pattern
            results[modality] = apply_brain_mask(modal_files[0], mask_path, modal_output)

    if params.get("cleanup", True):
        for pattern in ["*_mesh.vtk", "*_skull.nii.gz", "*_outskin_mesh.off"]:
            for temp_file in subject_dir.parent.rglob(pattern):
                try:
                    temp_file.unlink()
                except OSError:
                    pass
    return results
