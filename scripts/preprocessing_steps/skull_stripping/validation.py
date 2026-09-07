"""Mask-integrity checks for skull-stripping outputs (ADD-2).

Two layers, because a false cascade retry on GBM (mass effect, HD-BET
speckle) pays for a full extra tool run:

* **Hard fail** (`valid=False`) — catastrophe only: missing/empty mask,
  volume outside a wide adult range, no dominant component after
  dropping islands smaller than ``speckle_ml``, or enclosed cavities
  (black holes inside an otherwise connected brain). This is what the
  cascade uses to try the next stripper.
* **Review flags** — tighter windows (volume, LCC, FOV-edge touch, small
  holes) logged for QA. They do not reject the mask. Set
  ``fail_closed=False`` to also demote the hard volume/LCC/hole gates to
  flags (empty/missing still fail).

Defaults are wide on purpose. Tune via ``params["validation"]`` rather
than deleting the gate. MNI *leakage* (skull left on) remains a
benchmark warning, not a Stage 05 retry. Enclosed holes are the opposite
failure and *do* retry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import nibabel as nib
import numpy as np
from scipy.ndimage import label

logger = logging.getLogger(__name__)

# Catastrophe gates (cascade retry). Adult brain is ~1100–1500 ml; GBM mass
# effect and residual skull can sit well outside a textbook window without
# being an empty or whole-head mask.
DEFAULT_MIN_ML = 300.0
DEFAULT_MAX_ML = 2500.0
DEFAULT_MIN_DOMINANT_FRACTION = 0.70
DEFAULT_SPECKLE_ML = 1.0

# Review-only windows (log, do not retry). Inspired by the private337 QA
# pipeline (review 1000–1800 ml, morphology / edge-touch as warnings).
DEFAULT_REVIEW_MIN_ML = 1000.0
DEFAULT_REVIEW_MAX_ML = 1800.0
DEFAULT_REVIEW_MIN_DOMINANT_FRACTION = 0.95
DEFAULT_REVIEW_MAX_EDGE_TOUCH = 0.05
# Enclosed background cavities (mask==0 completely surrounded by brain).
# Adult ventricles belong *inside* a skull-strip mask as 1s; holes here
# are swiss cheese, not CSF — so there is no legitimate middle ground and
# the gate can sit low. Calibrated on this project's runs: every correct
# mask measured 0.00 ml of holes, while the defective MNI mask
# (KA126/sub-024) had 10.71 ml across 11 cavities and sailed through the
# previous 20 ml gate.
DEFAULT_MAX_HOLE_ML = 2.0
DEFAULT_REVIEW_MAX_HOLE_ML = 0.5

PathLike = Union[str, Path]


def _voxel_ml(img: nib.Nifti1Image) -> float:
    zooms = img.header.get_zooms()[:3]
    return float(np.prod(zooms)) / 1000.0


def _empty_metrics() -> Dict[str, Any]:
    return {
        "mask_voxels": 0,
        "mask_volume_ml": 0.0,
        "lcc_ratio": 0.0,
        "dominant_fraction": 0.0,
        "edge_touch_ratio": 0.0,
        "bbox_fill_ratio": 0.0,
        "n_components": 0,
        "n_components_kept": 0,
        "hole_volume_ml": 0.0,
        "n_holes": 0,
    }


def _enclosed_holes(data: np.ndarray, voxel_ml: float) -> tuple[float, int]:
    """Background components that do not touch the FOV face.

    These are cavities inside the brain mask (swiss cheese), not the
    outside-the-head air and not sulci that open to the exterior.
    """
    labeled, n_bg = label(~data)
    if n_bg == 0:
        return 0.0, 0
    edge = np.zeros_like(data, dtype=bool)
    edge[0, :, :] = True
    edge[-1, :, :] = True
    edge[:, 0, :] = True
    edge[:, -1, :] = True
    edge[:, :, 0] = True
    edge[:, :, -1] = True
    touching = np.zeros(n_bg + 1, dtype=bool)
    touching[np.unique(labeled[edge])] = True
    touching[0] = False
    sizes = np.bincount(labeled.ravel(), minlength=n_bg + 1)
    interior = ~touching
    interior[0] = False
    n_holes = int(interior.sum())
    hole_ml = float(sizes[interior].sum() * voxel_ml)
    return hole_ml, n_holes


def mask_metrics(
    mask_path: PathLike,
    speckle_ml: float = DEFAULT_SPECKLE_ML,
) -> Dict[str, Any]:
    """Morphology of a brain mask (volume, LCC after speckles, FOV edge).

    ``lcc_ratio`` / ``dominant_fraction`` ignore connected components smaller
    than ``speckle_ml`` so HD-BET islands do not look like fragmentation.
    """
    path = Path(mask_path)
    if not path.is_file():
        return {"mask_exists": False, **_empty_metrics()}

    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj) > 0
    voxel_ml = _voxel_ml(img)
    voxels = int(data.sum())
    volume_ml = float(voxels * voxel_ml)

    if voxels == 0:
        return {"mask_exists": True, **_empty_metrics()}

    labeled, n_components = label(data)
    sizes = np.bincount(labeled.ravel())[1:]
    speckle_voxels = speckle_ml / voxel_ml if voxel_ml > 0 else 0.0
    kept = sizes[sizes >= speckle_voxels] if speckle_voxels > 0 else sizes
    dominant_fraction = float(kept.max() / kept.sum()) if kept.size else 0.0

    edge = np.zeros_like(data, dtype=bool)
    edge[0, :, :] = True
    edge[-1, :, :] = True
    edge[:, 0, :] = True
    edge[:, -1, :] = True
    edge[:, :, 0] = True
    edge[:, :, -1] = True
    edge_touch_ratio = float(np.logical_and(data, edge).sum() / voxels)

    coords = np.argwhere(data)
    mn = coords.min(0)
    mx = coords.max(0) + 1
    bbox_fill_ratio = float(voxels / max(int(np.prod(mx - mn)), 1))
    hole_volume_ml, n_holes = _enclosed_holes(data, voxel_ml)

    return {
        "mask_exists": True,
        "mask_voxels": voxels,
        "mask_volume_ml": volume_ml,
        "lcc_ratio": dominant_fraction,
        "dominant_fraction": dominant_fraction,
        "edge_touch_ratio": edge_touch_ratio,
        "bbox_fill_ratio": bbox_fill_ratio,
        "n_components": int(n_components),
        "n_components_kept": int(kept.size),
        "hole_volume_ml": hole_volume_ml,
        "n_holes": n_holes,
    }


def validate_mask(
    mask_path: PathLike,
    min_ml: float = DEFAULT_MIN_ML,
    max_ml: float = DEFAULT_MAX_ML,
    min_dominant_fraction: float = DEFAULT_MIN_DOMINANT_FRACTION,
    speckle_ml: float = DEFAULT_SPECKLE_ML,
    fail_closed: bool = True,
    review_min_ml: float = DEFAULT_REVIEW_MIN_ML,
    review_max_ml: float = DEFAULT_REVIEW_MAX_ML,
    review_min_dominant_fraction: float = DEFAULT_REVIEW_MIN_DOMINANT_FRACTION,
    review_max_edge_touch_ratio: Optional[float] = DEFAULT_REVIEW_MAX_EDGE_TOUCH,
    max_hole_ml: float = DEFAULT_MAX_HOLE_ML,
    review_max_hole_ml: float = DEFAULT_REVIEW_MAX_HOLE_ML,
    **_ignored: Any,
) -> Dict[str, Any]:
    """Return validity, reason, metrics, and review flags.

    ``fail_closed=True`` (default): volume / LCC / enclosed-hole catastrophe
    rejects the mask and the cascade may retry. ``fail_closed=False``: those
    become review flags; only missing/empty masks still fail (cannot apply
    them). Extra kwargs are ignored so ``params["validation"]`` can be
    splatted in.
    """
    metrics = mask_metrics(mask_path, speckle_ml=speckle_ml)
    volume_ml = float(metrics["mask_volume_ml"])
    dominant_fraction = float(metrics["dominant_fraction"])
    hole_volume_ml = float(metrics.get("hole_volume_ml") or 0.0)
    review_flags: list[str] = []

    result: Dict[str, Any] = {
        "valid": True,
        "reason": "ok",
        "volume_ml": volume_ml,
        "dominant_fraction": dominant_fraction,
        "review_flags": review_flags,
        **metrics,
    }

    if not metrics["mask_exists"] or volume_ml <= 0:
        result["valid"] = False
        result["reason"] = (
            "volume: mask file missing"
            if not metrics["mask_exists"]
            else "volume: empty mask"
        )
        return result

    volume_fail = volume_ml < min_ml or volume_ml > max_ml
    component_fail = (
        metrics["n_components_kept"] == 0
        or dominant_fraction < min_dominant_fraction
    )
    hole_fail = hole_volume_ml > max_hole_ml

    if volume_fail:
        hard_reason = f"volume {volume_ml:.0f}ml out of [{min_ml},{max_ml}]"
        if fail_closed:
            result["valid"] = False
            result["reason"] = hard_reason
            return result
        review_flags.append("MASK_VOLUME_OUT_OF_HARD_RANGE")
        result["reason"] = hard_reason

    if component_fail:
        hard_reason = (
            f"fragmented: dominant component {dominant_fraction:.2f} "
            f"< {min_dominant_fraction}"
        )
        if fail_closed:
            result["valid"] = False
            result["reason"] = hard_reason
            return result
        review_flags.append("MASK_FRAGMENTED")
        result["reason"] = hard_reason

    if hole_fail:
        n_holes = int(metrics.get("n_holes") or 0)
        hard_reason = (
            f"holes {hole_volume_ml:.1f}ml in {n_holes} cavities "
            f"(max {max_hole_ml:.0f}ml)"
        )
        if fail_closed:
            result["valid"] = False
            result["reason"] = hard_reason
            return result
        review_flags.append("MASK_HOLES")
        result["reason"] = hard_reason

    if volume_ml < review_min_ml:
        review_flags.append("MASK_VOLUME_TOO_SMALL")
    if volume_ml > review_max_ml:
        review_flags.append("MASK_VOLUME_TOO_LARGE")
    if dominant_fraction < review_min_dominant_fraction:
        review_flags.append("LOW_LCC_RATIO")
    if (
        review_max_edge_touch_ratio is not None
        and metrics["edge_touch_ratio"] > review_max_edge_touch_ratio
    ):
        review_flags.append("EDGE_TOUCH")
    if hole_volume_ml > review_max_hole_ml and "MASK_HOLES" not in review_flags:
        review_flags.append("MASK_HOLES")

    if review_flags:
        logger.debug(
            "Skull-strip mask review flags for %s: %s (volume=%.1f ml, "
            "dominant=%.3f)",
            mask_path,
            ";".join(review_flags),
            volume_ml,
            dominant_fraction,
        )

    return result


def format_mask_check_line(check: Dict[str, Any]) -> str:
    """One-line metrics for Stage 05 logs."""
    flags = check.get("review_flags") or []
    flag_s = ";".join(flags) if flags else "none"
    volume = float(check.get("volume_ml") or 0.0)
    lcc = float(check.get("dominant_fraction") or 0.0)
    edge = float(check.get("edge_touch_ratio") or 0.0)
    n_comp = int(check.get("n_components") or 0)
    n_kept = int(check.get("n_components_kept") or 0)
    hole_ml = float(check.get("hole_volume_ml") or 0.0)
    n_holes = int(check.get("n_holes") or 0)
    return (
        f"volume={volume:.1f} ml, LCC={lcc:.3f}, edge_touch={edge:.3f}, "
        f"holes={hole_ml:.1f} ml (n={n_holes}), "
        f"components={n_comp} (kept={n_kept}), valid={check.get('valid')}, "
        f"reason={check.get('reason', '')}, review_flags={flag_s}"
    )


def cascade_decision_message(
    *,
    name: str,
    accepted: bool,
    remaining: list,
    check: Optional[Dict[str, Any]] = None,
    strip_failed: bool = False,
    validation_enabled: bool = True,
    review_retry: bool = False,
) -> str:
    """Explain ACCEPT / REJECT and whether the next stripper will run."""
    rest = [str(x) for x in remaining if x]
    next_name = rest[0] if rest else None
    flags = (check or {}).get("review_flags") or []
    flag_s = ";".join(flags) if flags else "none"

    if not validation_enabled and accepted:
        tail = (
            f"not trying remaining {rest}"
            if rest
            else "cascade exhausted"
        )
        return (
            f"Cascade decision: ACCEPT {name!r} — validation disabled; {tail}"
        )

    if strip_failed:
        if next_name:
            return (
                f"Cascade decision: REJECT {name!r} — strip() failed; "
                f"trying next: {next_name}"
            )
        return (
            f"Cascade decision: REJECT {name!r} — strip() failed; "
            f"no further tools in cascade"
        )

    if accepted:
        review_note = ""
        if flags:
            review_note = (
                f" with review_flags={flag_s} (review is log-only, not a retry)"
            )
        if rest:
            return (
                f"Cascade decision: ACCEPT {name!r} — hard gates passed"
                f"{review_note}; not trying remaining {rest}"
            )
        return (
            f"Cascade decision: ACCEPT {name!r} — hard gates passed"
            f"{review_note}; no further tools in cascade"
        )

    if review_retry:
        # Not a catastrophe — the mask cleared the hard gates but raised
        # flags, and something untried might do better.
        if next_name:
            return (
                f"Cascade decision: REJECT {name!r} — review flags {flag_s}; "
                f"trying next: {next_name} (kept in reserve if nothing is cleaner)"
            )
        return (
            f"Cascade decision: ACCEPT {name!r} — review flags {flag_s}; "
            f"no further tools in cascade"
        )

    reason = (check or {}).get("reason", "invalid mask")
    if next_name:
        return (
            f"Cascade decision: REJECT {name!r} — catastrophe ({reason}); "
            f"trying next: {next_name}"
        )
    return (
        f"Cascade decision: REJECT {name!r} — catastrophe ({reason}); "
        f"no further tools in cascade"
    )


def validation_gate_banner(vcfg: Optional[Dict[str, Any]] = None) -> str:
    """Thresholds actually used, so a run log is interpretable without the code."""
    cfg = dict(vcfg or {})
    if not cfg.get("enabled", True):
        return "Mask validation: disabled (accept first successful strip)"
    min_ml = float(cfg.get("min_ml", DEFAULT_MIN_ML))
    max_ml = float(cfg.get("max_ml", DEFAULT_MAX_ML))
    min_lcc = float(cfg.get("min_dominant_fraction", DEFAULT_MIN_DOMINANT_FRACTION))
    speckle = float(cfg.get("speckle_ml", DEFAULT_SPECKLE_ML))
    max_hole = float(cfg.get("max_hole_ml", DEFAULT_MAX_HOLE_ML))
    rmin = float(cfg.get("review_min_ml", DEFAULT_REVIEW_MIN_ML))
    rmax = float(cfg.get("review_max_ml", DEFAULT_REVIEW_MAX_ML))
    fail_closed = cfg.get("fail_closed", True)
    return (
        f"Mask validation gates: hard volume [{min_ml:.0f}, {max_ml:.0f}] ml, "
        f"min LCC {min_lcc:.2f} (islands < {speckle:.1f} ml ignored), "
        f"max enclosed holes {max_hole:.0f} ml; "
        f"review volume [{rmin:.0f}, {rmax:.0f}] ml; fail_closed={fail_closed}"
    )
