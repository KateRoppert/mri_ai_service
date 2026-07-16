"""
Tests for registry↔allocator BIDS-id consistency.

register_patient must update bids_id when a patient is re-registered under a new
allocator subject, and reconcile_registry_bids_ids must bring existing stale rows
back in line. Both are what keep the longitudinal view pointing at the right
patient after the persistent allocator changes numbering.

Runs against an isolated temp DB (DATABASE_URL set before importing the modules
that bind SessionLocal at import time). Run standalone:
    DATABASE_URL=... python -m pytest backend/test_registry_reconcile.py
"""

import os
import sys
import tempfile
from pathlib import Path

_TMP_DB = Path(tempfile.mkdtemp()) / "test_registry.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from patient_registry import (
    register_patient,
    find_by_bids_subject,
    reconcile_registry_bids_ids,
    _resync_bids_id,
    ensure_tables,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    from database import SessionLocal
    from registry_models import PatientRegistry
    ensure_tables()
    db = SessionLocal()
    db.query(PatientRegistry).delete()
    db.commit()
    db.close()
    yield


# --- _resync_bids_id (pure) --------------------------------------------------

def test_resync_replaces_subject_keeps_session():
    assert _resync_bids_id("sub-001_ses-002", "sub-004") == "sub-004_ses-002"


def test_resync_without_session_returns_subject():
    assert _resync_bids_id("sub-001", "sub-004") == "sub-004"
    assert _resync_bids_id("", "sub-004") == "sub-004"


# --- register_patient updates bids_id ---------------------------------------

def test_reregistration_updates_bids_id():
    # First run numbered this patient sub-001.
    register_patient(bids_id="sub-001_ses-001", study_hash="h1",
                     original_patient_id="P000067", lesion_type="multiple_sclerosis")
    # A later run, via the allocator, numbers the same study sub-004.
    register_patient(bids_id="sub-004_ses-001", study_hash="h1",
                     original_patient_id="P000067", lesion_type="multiple_sclerosis")

    assert find_by_bids_subject("sub-001") == []            # stale id gone
    under_004 = find_by_bids_subject("sub-004")
    assert len(under_004) == 1
    assert under_004[0]["original_patient_id"] == "P000067"


# --- reconcile_registry_bids_ids --------------------------------------------

def test_reconcile_fixes_stale_subject():
    # Registry holds the incident's stale numbering for P000067 (should be sub-004).
    register_patient(bids_id="sub-003_ses-001", study_hash="h1",
                     original_patient_id="P000067", lesion_type="multiple_sclerosis")
    register_patient(bids_id="sub-003_ses-002", study_hash="h2",
                     original_patient_id="P000067", lesion_type="multiple_sclerosis")

    n = reconcile_registry_bids_ids({"multiple_sclerosis": {"P000067": "sub-004"}})

    assert n == 2
    fixed = {r["bids_id"] for r in find_by_bids_subject("sub-004")}
    assert fixed == {"sub-004_ses-001", "sub-004_ses-002"}
    assert find_by_bids_subject("sub-003") == []


def test_reconcile_is_idempotent_and_skips_unknown():
    register_patient(bids_id="sub-004_ses-001", study_hash="h1",
                     original_patient_id="P000067", lesion_type="multiple_sclerosis")
    # A patient with no allocator entry must be left alone.
    register_patient(bids_id="sub-009_ses-001", study_hash="h2",
                     original_patient_id="P_LEGACY", lesion_type="multiple_sclerosis")

    n = reconcile_registry_bids_ids({"multiple_sclerosis": {"P000067": "sub-004"}})

    assert n == 0  # already correct + no allocator entry for the legacy patient
    assert len(find_by_bids_subject("sub-004")) == 1
    assert len(find_by_bids_subject("sub-009")) == 1


def test_reconcile_resolves_collision():
    # The original incident: two patients both stuck at sub-001.
    register_patient(bids_id="sub-001_ses-001", study_hash="hA",
                     original_patient_id="P000915", lesion_type="multiple_sclerosis")
    register_patient(bids_id="sub-001_ses-001", study_hash="hB",
                     original_patient_id="P000010", lesion_type="multiple_sclerosis")

    n = reconcile_registry_bids_ids(
        {"multiple_sclerosis": {"P000915": "sub-001", "P000010": "sub-002"}}
    )

    assert n == 1  # only P000010 moves
    assert find_by_bids_subject("sub-001")[0]["original_patient_id"] == "P000915"
    assert find_by_bids_subject("sub-002")[0]["original_patient_id"] == "P000010"
