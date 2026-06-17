"""Stage 05 imports these three names; they must survive the package refactor."""


def test_stage05_imports_resolve():
    from preprocessing_steps.skull_stripping import (
        setup_fsl_environment,
        check_fsl_installed,
        process_subject_skull_stripping,
    )
    assert callable(setup_fsl_environment)
    assert callable(check_fsl_installed)
    assert callable(process_subject_skull_stripping)


def test_legacy_helpers_still_exported():
    from preprocessing_steps.skull_stripping import (
        run_bet,
        apply_brain_mask,
        compare_before_after_stripping,
    )
    assert callable(run_bet)
    assert callable(apply_brain_mask)
    assert callable(compare_before_after_stripping)


import pytest
from preprocessing_steps.skull_stripping import SkullStripperBase, BetStripper


def test_cannot_instantiate_abstract_base():
    with pytest.raises(TypeError):
        SkullStripperBase()


def test_bet_stripper_name():
    assert BetStripper().name == "bet"


def test_bet_stripper_unavailable_when_fsl_missing(monkeypatch):
    monkeypatch.setattr("preprocessing_steps.skull_stripping.bet.check_fsl_installed",
                        lambda: False)
    assert BetStripper().is_available() is False
