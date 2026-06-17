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
