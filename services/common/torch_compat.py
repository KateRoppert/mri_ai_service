"""
Torch compatibility shims shared by all segmentation services.

enable_legacy_checkpoint_loading() monkey-patches torch.load to force
weights_only=False, which is required for loading nnUNet v1 (gbm-seg) and
CATMIL (ms-seg) checkpoints that pre-date PyTorch 2.6's safe-loading default.
"""

from typing import Any


def enable_legacy_checkpoint_loading() -> None:
    """
    Patch torch.load so that old nnUNet / CATMIL checkpoints load without error
    on PyTorch >= 2.6 (which switched the default to weights_only=True).

    Also registers numpy.core.multiarray.scalar as a safe global if the
    torch.serialization API supports it (PyTorch >= 2.4).

    Call once at service startup, before any model weights are loaded.
    """
    try:
        import numpy
        import torch

        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([numpy.core.multiarray.scalar])

        _original_torch_load = torch.load

        def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_torch_load  # type: ignore[assignment]

    except Exception as e:
        print(f"WARNING: could not patch torch.load: {e}")
