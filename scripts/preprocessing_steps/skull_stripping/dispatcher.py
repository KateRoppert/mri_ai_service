"""
Config → skull stripper instance.

Single place where the tool is chosen, so switching tools is one line in
preprocessing_config.yaml:

    params:
      method: "hdbet"
      fallback_method: "bet"

Per the Этап 5.5 spec §8: if the requested tool is not available on this
machine, fall back with a warning rather than failing the run — a missing
GPU or an uninstalled package should degrade quality, not stop the
pipeline.
"""

import logging
from typing import Any, Dict, Optional

from .base import SkullStripperBase
from .bet import BetStripper
from .hdbet import HdBetStripper

logger = logging.getLogger(__name__)

# Registry of implemented tools. The research spec lists more (SynthStrip,
# MNI masks, SAM, DeepBET); they land here as they are implemented.
STRIPPERS = {
    "bet": BetStripper,
    "hdbet": HdBetStripper,
}

DEFAULT_METHOD = "bet"


class SkullStripperUnavailable(RuntimeError):
    """Neither the requested tool nor its fallback can run here."""


def get_stripper(params: Optional[Dict[str, Any]] = None) -> SkullStripperBase:
    """
    Return the stripper to use for this run.

    Resolution order:
      1. `method` from params, if that tool is available
      2. `fallback_method`, if set and available (logged as a warning)
      3. raise SkullStripperUnavailable

    Availability is checked here, once per subject, rather than inside
    strip() — a tool that cannot run should be swapped out before any
    files are touched.
    """
    params = params or {}
    method = str(params.get("method", DEFAULT_METHOD)).lower()
    fallback = params.get("fallback_method")

    if method not in STRIPPERS:
        known = ", ".join(sorted(STRIPPERS))
        raise SkullStripperUnavailable(
            f"unknown skull stripping method {method!r}; known methods: {known}"
        )

    stripper = STRIPPERS[method]()
    if stripper.is_available():
        logger.info("Skull stripping method: %s", method)
        return stripper

    if not fallback:
        raise SkullStripperUnavailable(
            f"skull stripping method {method!r} is not available on this machine "
            f"and no fallback_method is configured"
        )

    fallback = str(fallback).lower()
    if fallback not in STRIPPERS:
        known = ", ".join(sorted(STRIPPERS))
        raise SkullStripperUnavailable(
            f"{method!r} unavailable and fallback_method {fallback!r} is unknown; "
            f"known methods: {known}"
        )

    fallback_stripper = STRIPPERS[fallback]()
    if not fallback_stripper.is_available():
        raise SkullStripperUnavailable(
            f"neither {method!r} nor fallback {fallback!r} is available on this machine"
        )

    logger.warning(
        "Skull stripping method %r is not available here — falling back to %r. "
        "Masks will differ from a %r run.", method, fallback, method
    )
    return fallback_stripper


def get_tool_params(params: Optional[Dict[str, Any]] = None,
                    stripper: Optional[SkullStripperBase] = None) -> Dict[str, Any]:
    """
    Tool-specific parameters for the selected stripper.

    Reads `tool_params` when present (the spec's per-tool block, populated
    from benchmark tuning). Falls back to the flat legacy keys so existing
    BET configs — which put fractional_intensity/vertical_gradient directly
    under params — keep working untouched.
    """
    params = params or {}
    tool_params = dict(params.get("tool_params") or {})

    legacy_keys = ("fractional_intensity", "vertical_gradient")
    for key in legacy_keys:
        if key in params and key not in tool_params:
            tool_params[key] = params[key]

    return tool_params
