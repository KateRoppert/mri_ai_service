"""
Tests for services/common/torch_compat.py (KI-008).

torch is not installed in the project venv (service-level dep), so torch and
numpy are mocked in sys.modules at function-call time. This is the right place
to patch them because the function does `import torch` lazily (inside the
function body), not at module import time.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJ_ROOT = Path(__file__).parent
TORCH_COMPAT = PROJ_ROOT / "services" / "common" / "torch_compat.py"


def _load_module():
    """Load a fresh copy of torch_compat without needing torch at import time."""
    name = f"torch_compat_{id(object())}"
    spec = importlib.util.spec_from_file_location(name, TORCH_COMPAT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_mock_torch(has_add_safe_globals: bool = True) -> MagicMock:
    mt = MagicMock(name="torch")
    mt.load = MagicMock(name="original_torch_load")
    if not has_add_safe_globals:
        del mt.serialization.add_safe_globals
    return mt


# ---------------------------------------------------------------------------
# Module-level import — torch must NOT be required at import time
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_importable_without_torch(self):
        """Module imports cleanly even when torch is absent from sys.modules."""
        with patch.dict(sys.modules, {"torch": None, "numpy": None}):
            mod = _load_module()
        assert callable(mod.enable_legacy_checkpoint_loading)


# ---------------------------------------------------------------------------
# enable_legacy_checkpoint_loading() — patching behaviour
# ---------------------------------------------------------------------------

class TestEnableLegacyCheckpointLoading:
    def test_torch_load_is_replaced(self):
        mt = _make_mock_torch()
        original_load = mt.load
        mod = _load_module()

        with patch.dict(sys.modules, {"torch": mt, "numpy": MagicMock()}):
            mod.enable_legacy_checkpoint_loading()

        assert mt.load is not original_load, "torch.load must be replaced by the patch"

    def test_patched_load_forces_weights_only_false(self):
        mt = _make_mock_torch()
        original_load = mt.load
        mod = _load_module()

        with patch.dict(sys.modules, {"torch": mt, "numpy": MagicMock()}):
            mod.enable_legacy_checkpoint_loading()

        mt.load("/model.pth")
        original_load.assert_called_once_with("/model.pth", weights_only=False)

    def test_patched_load_preserves_other_kwargs(self):
        mt = _make_mock_torch()
        original_load = mt.load
        mod = _load_module()

        with patch.dict(sys.modules, {"torch": mt, "numpy": MagicMock()}):
            mod.enable_legacy_checkpoint_loading()

        mt.load("/model.pth", map_location="cpu")
        original_load.assert_called_once_with(
            "/model.pth", map_location="cpu", weights_only=False
        )

    def test_explicit_weights_only_true_overridden_to_false(self):
        mt = _make_mock_torch()
        original_load = mt.load
        mod = _load_module()

        with patch.dict(sys.modules, {"torch": mt, "numpy": MagicMock()}):
            mod.enable_legacy_checkpoint_loading()

        mt.load("/model.pth", weights_only=True)
        _, kwargs = original_load.call_args
        assert kwargs["weights_only"] is False

    def test_add_safe_globals_called_when_api_available(self):
        mt = _make_mock_torch(has_add_safe_globals=True)
        mock_numpy = MagicMock()
        mod = _load_module()

        with patch.dict(sys.modules, {"torch": mt, "numpy": mock_numpy}):
            mod.enable_legacy_checkpoint_loading()

        mt.serialization.add_safe_globals.assert_called_once()

    def test_add_safe_globals_skipped_on_old_pytorch(self):
        """PyTorch < 2.4 without add_safe_globals must not crash."""
        mt = _make_mock_torch(has_add_safe_globals=False)
        mod = _load_module()

        with patch.dict(sys.modules, {"torch": mt, "numpy": MagicMock()}):
            mod.enable_legacy_checkpoint_loading()  # must not raise

    def test_exception_during_patch_prints_warning_not_raises(self, capsys):
        """If anything goes wrong during patching, WARNING is printed, no exception."""
        mt = _make_mock_torch(has_add_safe_globals=True)
        # add_safe_globals is called inside the try block — raising there triggers except
        mt.serialization.add_safe_globals.side_effect = RuntimeError("serialization boom")
        mod = _load_module()

        with patch.dict(sys.modules, {"torch": mt, "numpy": MagicMock()}):
            mod.enable_legacy_checkpoint_loading()  # must not raise

        assert "WARNING" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Import path — the 'from common.torch_compat import ...' in service_server.py
# ---------------------------------------------------------------------------

class TestImportPathResolution:
    def test_importable_as_common_torch_compat_from_services_dir(self):
        """
        Both service_server.py files do:
            from common.torch_compat import enable_legacy_checkpoint_loading
        PYTHONPATH in Docker is /app = services root. Simulate that.
        """
        services_dir = str(PROJ_ROOT / "services")
        saved = sys.path[:]
        try:
            if services_dir not in sys.path:
                sys.path.insert(0, services_dir)
            spec = importlib.util.find_spec("common.torch_compat")
            assert spec is not None, \
                "common.torch_compat must be resolvable when services/ is on sys.path"
        finally:
            sys.path[:] = saved
