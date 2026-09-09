"""Tests for the tool-matrix runner and its analysis.

The heavy parts here are integration territory (real strippers, real volumes,
FSL/FreeSurfer/HD-BET in the container). What is unit-testable is the small
set of pure functions that decide *what gets compared with what* — and those
are exactly where a silent mistake would corrupt every downstream number
without ever raising.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

import analyze_matrix
import atlas_space
import make_contact_sheet
import run_tool_matrix


# ---------------------------------------------------------------------------
# Dice
# ---------------------------------------------------------------------------

def test_dice_identical_masks_is_one():
    mask = np.zeros((10, 10, 10), bool)
    mask[2:8, 2:8, 2:8] = True
    assert analyze_matrix.dice(mask, mask) == pytest.approx(1.0)


def test_dice_disjoint_masks_is_zero():
    left = np.zeros((10, 10, 10), bool)
    right = np.zeros((10, 10, 10), bool)
    left[0:3] = True
    right[7:10] = True
    assert analyze_matrix.dice(left, right) == pytest.approx(0.0)


def test_dice_half_overlap():
    """A mask twice the size of another, fully containing it: 2*1/(1+2)."""
    small = np.zeros((10, 10, 10), bool)
    big = np.zeros((10, 10, 10), bool)
    small[0:2] = True
    big[0:4] = True
    assert analyze_matrix.dice(small, big) == pytest.approx(2 / 3)


def test_dice_of_two_empty_masks_is_nan_not_one():
    """Two empty masks agree on nothing measurable. Returning 1.0 here would
    quietly award a perfect score to a tool that produced no mask at all."""
    empty = np.zeros((5, 5, 5), bool)
    assert np.isnan(analyze_matrix.dice(empty, empty))


# ---------------------------------------------------------------------------
# Alignment guard
# ---------------------------------------------------------------------------

def test_alignment_ok_zero_when_mask_sits_on_tissue():
    head = np.ones((8, 8, 8))
    mask = np.zeros((8, 8, 8), bool)
    mask[2:6, 2:6, 2:6] = True
    assert atlas_space.alignment_ok(head, mask) == pytest.approx(0.0)


def test_alignment_ok_reports_share_on_background():
    """Half the mask over background must read 0.5 — this is the guard that
    catches an affine and a mask coming from different Stage 05 runs."""
    head = np.zeros((10, 10, 10))
    head[5:] = 100.0
    mask = np.ones((10, 10, 10), bool)
    assert atlas_space.alignment_ok(head, mask) == pytest.approx(0.5)


def test_alignment_ok_empty_mask_is_worst_case():
    head = np.ones((4, 4, 4))
    assert atlas_space.alignment_ok(head, np.zeros((4, 4, 4), bool)) == 1.0


# ---------------------------------------------------------------------------
# Tool params come from the live config, not from a copy
# ---------------------------------------------------------------------------

def test_load_tool_params_reads_the_skull_stripping_step(tmp_path: Path):
    config = tmp_path / "prep.yaml"
    config.write_text(yaml.safe_dump({
        "steps": [
            {"name": "registration", "params": {"tool_params": {"wrong": 1}}},
            {"name": "skull_stripping",
             "params": {"tool_params": {"device": "cuda",
                                        "fractional_intensity": 0.35}}},
        ]
    }))
    params = run_tool_matrix.load_tool_params(config)

    assert params == {"device": "cuda", "fractional_intensity": 0.35}


def test_load_tool_params_without_the_step_is_empty(tmp_path: Path):
    config = tmp_path / "prep.yaml"
    config.write_text(yaml.safe_dump({"steps": [{"name": "reorient"}]}))

    assert run_tool_matrix.load_tool_params(config) == {}


def test_production_config_still_exposes_tool_params():
    """Guards the real file: the matrix is only comparable to production while
    it reads the same parameters production uses."""
    params = run_tool_matrix.load_tool_params(
        atlas_space.PROJECT_ROOT / "configs" / "preprocessing_config.yaml")

    assert "device" in params
    assert "fractional_intensity" in params


# ---------------------------------------------------------------------------
# Contact sheet geometry
# ---------------------------------------------------------------------------

def test_outline_is_the_boundary_only():
    mask = np.zeros((10, 10), bool)
    mask[3:7, 3:7] = True
    edge = make_contact_sheet.outline(mask)

    assert edge.sum() == 12          # 4x4 block minus its 2x2 interior
    assert not edge[4, 4]            # interior stays clear
    assert edge[3, 3]                # corner is on the boundary


def test_outline_of_empty_mask_is_empty():
    assert not make_contact_sheet.outline(np.zeros((6, 6), bool)).any()


def test_axial_levels_reach_below_the_mask():
    """The lowest panel has to sit under the mask, or leftover orbital tissue
    is never on screen — the defect the sheet exists to catch."""
    mask = np.zeros((20, 20, 100), bool)
    mask[5:15, 5:15, 40:80] = True
    levels = make_contact_sheet.pick_axial_levels(mask)

    assert min(levels) < 40
    assert max(levels) <= 80
    assert len(levels) == len(make_contact_sheet.AXIAL_FRACTIONS)


def test_axial_levels_survive_an_empty_mask():
    """A failed strip must not crash the whole sheet."""
    levels = make_contact_sheet.pick_axial_levels(np.zeros((10, 10, 10), bool))

    assert all(0 <= z < 10 for z in levels)
