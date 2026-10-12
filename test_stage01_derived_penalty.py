"""The derived-reconstruction penalty is a tie-break, not a decider.

Introduced in e4f592d with a sound intent: among candidates for the same
modality, a direct acquisition should beat a reconstruction of one. The
valuable half of that commit was making DERIVED series visible at all —
before it, detect_modality returned None for them and a patient whose only
t1c was a reconstruction showed that modality as missing
(clinical_dicom/BO-1).

The penalty itself was set to 0.1, which is ten times the entire dynamic
range of the resolution factor (0.5-2.0). Wherever it applied it therefore
decided alone, overruling the one real quality signal in the system.
Measured over the six datasets in data/: of 52 modality slots with more
than one candidate it changed the winner in two, and in both it chose the
worse series — a 4 mm 2D T1 over a 1 mm isotropic MPRAGE in one of them.

Its intent needs it only to break ties, so it is now 0.8.
"""
import importlib.util
import logging
import sys
from pathlib import Path

import pytest

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "reorg01_dp", SCRIPTS_DIR / "01_reorganize_folders.py")
reorg = importlib.util.module_from_spec(_spec)
sys.modules["reorg01_dp"] = reorg
_spec.loader.exec_module(reorg)

from utils.config_loader import load_series_scoring_config


@pytest.fixture
def scoring():
    return load_series_scoring_config()


@pytest.fixture
def dedup(scoring):
    return reorg.SeriesDeduplicator(
        logging.getLogger("test_derived_penalty"), scoring_config=scoring)


def _series(name, thickness, derived, slices):
    return reorg.SeriesInfo(
        original_path=Path(f"/raw/{name}"),
        patient_id="TEST",
        date="20260101",
        modality="t1",
        slice_count=slices,
        series_description=name,
        slice_thickness_mm=thickness,
        is_derived_reconstruction=derived,
    )


def test_a_thin_derived_series_beats_a_thick_primary_one(dedup, monkeypatch):
    """The outcome this change is for. Real dropbox_33 case: a 4 mm 2D T1
    was winning over a 1 mm isotropic MPRAGE purely because the MPRAGE's
    distortion-uncorrected variant is tagged DERIVED."""
    thin_derived = _series("t1_mprage_fs_sag_1mm_iso_ND", 1.0, True, 176)
    thick_primary = _series("t1_fl2d_sag_4mm", 4.0, False, 40)
    # slice_count is recomputed from disk by _select_best_series; these
    # series have no files, so keep what the fixtures declare.
    monkeypatch.setattr(reorg, "find_dicom_files",
                        lambda p: [p / f"{i}.dcm" for i in range(
                            176 if "mprage" in p.name else 40)])

    winner = dedup._select_best_series([thick_primary, thin_derived], "t1")

    assert winner is thin_derived, (
        "4-мм срез побеждает 1-мм изотропный — штраф за derived "
        "перебивает разрешение"
    )


def test_a_primary_beats_a_derived_of_the_same_quality(dedup, monkeypatch):
    """The original intent, preserved: when nothing else separates them, the
    direct acquisition wins. This is the ND-versus-DIS2D case the penalty
    was actually written for."""
    primary = _series("t1_mprage_sag_1mm_iso", 1.0, False, 176)
    derived = _series("t1_mprage_sag_1mm_iso_ND", 1.0, True, 176)
    monkeypatch.setattr(reorg, "find_dicom_files",
                        lambda p: [p / f"{i}.dcm" for i in range(176)])

    winner = dedup._select_best_series([derived, primary], "t1")

    assert winner is primary


def test_a_lone_derived_candidate_still_wins(dedup, monkeypatch):
    """What e4f592d actually fixed, and what no penalty value can undo: with
    one candidate there is nothing to compare against, so the modality
    resolves instead of being reported missing (clinical_dicom/BO-1)."""
    derived = _series("MDC_T1_SPAIR_VIBE_SAG_1mm_MPR", 1.0, True, 120)
    monkeypatch.setattr(reorg, "find_dicom_files",
                        lambda p: [p / f"{i}.dcm" for i in range(120)])

    winner = dedup._select_best_series([derived], "t1")

    assert winner is derived


def test_the_penalty_cannot_outweigh_the_resolution_signal(scoring):
    """The arithmetic behind all of the above, pinned so a future edit of
    either number has to face it: the penalty must not exceed the span of
    the resolution factor, or it decides alone wherever it applies."""
    penalty = scoring["derived_reconstruction_penalty"]
    res = scoring["resolution_scoring"]
    span = res["max_factor"] / res["min_factor"]

    assert penalty > 1.0 / span, (
        f"штраф {penalty} сильнее всего диапазона разрешения "
        f"({res['min_factor']}–{res['max_factor']}, разброс {span}x) — "
        f"он будет решать единолично"
    )
