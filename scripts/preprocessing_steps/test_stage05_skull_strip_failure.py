"""
KI-049: a failed skull strip fails the session.

In the after_registration order a failed mask only set
steps.skull_stripping.success = False; the step logged a tick, the session
counted as a success, and the registered images went on with the skull.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

_spec = importlib.util.spec_from_file_location("stage05", SCRIPTS / "05_preprocessing.py")
stage05 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage05)
check = stage05._require_skull_strip_success


def test_all_modalities_stripped_passes():
    check({"t1": {"success": True}, "t2": {"success": True}})


def test_failed_reference_mask_raises_with_the_tools_message():
    with pytest.raises(RuntimeError) as e:
        check({"t1": {"success": False, "error": "BET failed with return code 1: "
                                                  "Error: input image /a b/t1 not valid"}})
    assert "t1" in str(e.value) and "not valid" in str(e.value)


def test_a_modality_left_unmasked_raises():
    with pytest.raises(RuntimeError) as e:
        check({"t1": {"success": True}, "t2fl": {"success": False, "error": "File not found"}})
    assert "t2fl" in str(e.value)


def test_unavailable_tool_shape_raises_its_error():
    """process_subject_skull_stripping returns {"success": False, "error": …}
    when no tool is available; that used to die with AttributeError."""
    with pytest.raises(RuntimeError) as e:
        check({"success": False, "error": "no skull stripper available"})
    assert "no skull stripper available" in str(e.value)


def test_empty_result_raises():
    with pytest.raises(RuntimeError):
        check({})
