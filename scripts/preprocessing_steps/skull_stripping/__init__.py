"""Skull stripping plugin package.

Backward-compatible facade: Stage 05 imports setup_fsl_environment,
check_fsl_installed, and process_subject_skull_stripping from here.
"""

from .base import (
    SkullStripperBase,
    apply_brain_mask,
    compare_before_after_stripping,
    load_manifest,
)
from .bet import (
    BetStripper,
    check_fsl_installed,
    get_bet_command,
    get_fsl_env,
    run_bet,
    setup_fsl_environment,
)
# Import dispatcher last to avoid circular imports (dispatcher imports from base and bet).
from .dispatcher import (
    STRIPPER_REGISTRY,
    get_stripper,
    process_subject_skull_stripping,
    resolve_stripper,
)

__all__ = [
    "SkullStripperBase",
    "apply_brain_mask",
    "compare_before_after_stripping",
    "load_manifest",
    "BetStripper",
    "check_fsl_installed",
    "get_bet_command",
    "get_fsl_env",
    "run_bet",
    "setup_fsl_environment",
    "STRIPPER_REGISTRY",
    "get_stripper",
    "process_subject_skull_stripping",
    "resolve_stripper",
]
