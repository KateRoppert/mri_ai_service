"""Validating a desired modality set, and working out what has to change."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from session_assignment import AssignmentError, plan_changes, validate

REQUIRED = ["t1", "t1c", "t2", "t2fl"]


def _session(series=None, excluded=(), status="incomplete"):
    return {
        "status": status,
        "series": series or {},
        "excluded_series": [
            {"original_path": p, "series_description": "d",
             "slice_count": 10, "detected_modality": None,
             "reason": "unrecognized"}
            for p in excluded
        ],
    }


def test_accepts_a_set_drawn_from_this_session():
    session = _session(series={"t1": {"original_path": "/raw/a"}},
                       excluded=["/raw/b"])
    validate(session, {"t1": "/raw/a", "t1c": "/raw/b"}, REQUIRED)


def test_rejects_a_path_from_outside_this_session():
    """A stale screen can send a path belonging to another patient."""
    session = _session(series={"t1": {"original_path": "/raw/a"}})
    with pytest.raises(AssignmentError, match="не принадлежит"):
        validate(session, {"t1": "/raw/somebody-else"}, REQUIRED)


def test_rejects_a_modality_this_lesion_type_does_not_use():
    """MS has no t1c. Offering the slot at all was the bug; accepting it
    would write a modality nothing downstream expects."""
    session = _session(excluded=["/raw/b"])
    with pytest.raises(AssignmentError, match="t1c"):
        validate(session, {"t1c": "/raw/b"}, ["t1", "t2", "t2fl"])


def test_rejects_one_series_assigned_to_two_modalities():
    session = _session(excluded=["/raw/b"])
    with pytest.raises(AssignmentError, match="дважды"):
        validate(session, {"t1": "/raw/b", "t2": "/raw/b"}, REQUIRED)


@pytest.mark.parametrize("status", ["discarded", "merged"])
def test_refuses_sessions_whose_fate_is_already_decided(status):
    """Discarding and merging are decisions already taken. Quietly
    re-opening them would undo a choice nobody asked to undo."""
    session = _session(excluded=["/raw/b"], status=status)
    with pytest.raises(AssignmentError, match=status):
        validate(session, {"t1": "/raw/b"}, REQUIRED)


def test_plan_leaves_an_unchanged_modality_alone():
    """Without this every save would re-copy every DICOM in the session."""
    session = _session(series={"t1": {"original_path": "/raw/a"}})
    changes = plan_changes(session, {"t1": "/raw/a"})

    assert changes.assign == {}
    assert changes.clear == ()
    assert changes.unchanged == ("t1",)
    assert changes.is_empty()


def test_plan_reports_a_replacement():
    session = _session(series={"t1": {"original_path": "/raw/a"}},
                       excluded=["/raw/b"])
    changes = plan_changes(session, {"t1": "/raw/b"})

    assert changes.assign == {"t1": "/raw/b"}
    assert changes.clear == ()
    assert not changes.is_empty()


def test_plan_reports_a_new_fill():
    session = _session(excluded=["/raw/b"])
    changes = plan_changes(session, {"t2": "/raw/b"})

    assert changes.assign == {"t2": "/raw/b"}


def test_plan_reports_a_clearing():
    """"This is not the t1c" is a legitimate thing for a doctor to say."""
    session = _session(series={"t1": {"original_path": "/raw/a"},
                               "t2": {"original_path": "/raw/c"}})
    changes = plan_changes(session, {"t1": "/raw/a"})

    assert changes.clear == ("t2",)
    assert changes.unchanged == ("t1",)
    assert not changes.is_empty()
