"""Reading the list of sessions a run supersedes in Kappa.

The value is JSON in a text column, so it can be anything by the time it is
read: written by an older version, truncated, hand-edited during a repair.
None of that may stop an upload, which is why this function never raises.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from database import superseding_sessions


@pytest.mark.parametrize("raw,expected", [
    ('["sub-002_ses-001"]', {"sub-002_ses-001"}),
    ('["sub-002_ses-001", "sub-002_ses-002"]',
     {"sub-002_ses-001", "sub-002_ses-002"}),
    ("[]", set()),
    (None, set()),          # the column's default — most runs
    ("", set()),
    ("[1, 2]", {"1", "2"}),  # numbers coerce rather than blow up
])
def test_reads_what_it_can(raw, expected):
    assert superseding_sessions(SimpleNamespace(reprocessed_sessions=raw)) == expected


@pytest.mark.parametrize("raw", [
    "not json at all",
    '["sub-002_ses-001"',   # truncated write
    '{"sub-002_ses-001": true}',  # an object, not a list
    "42",
])
def test_a_corrupt_value_is_empty_not_an_exception(raw):
    """An upload must not be stopped by a column nobody can parse. Empty
    means 'nothing supersedes', which degrades to the old behaviour — the
    session is reported as a duplicate — rather than to a failed run."""
    assert superseding_sessions(SimpleNamespace(reprocessed_sessions=raw)) == set()


def test_a_run_from_before_the_column_existed():
    """getattr, not attribute access: an old row loaded by an older model
    has no such attribute at all."""
    assert superseding_sessions(SimpleNamespace()) == set()
