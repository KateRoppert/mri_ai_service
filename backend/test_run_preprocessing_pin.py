"""preprocessing_id закрепляется в момент старта прогона, а не выгрузки.

KI-085: расчёт из живого конфига при выгрузке метит данные настройками,
которые действуют сейчас, а не теми, с которыми считали.
"""
import yaml

from database import SessionLocal, create_pipeline_run, get_pipeline_run
from preprocessing_version import compute_preprocessing_id


def _config(path, sigma):
    """A preprocessing config whose functional part depends on `sigma`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "atlas": {"name": "MNI152", "filename": "mni.nii.gz"},
        "modalities": ["t1", "t1c"],
        "steps": [{"name": "smooth", "enabled": True,
                   "params": {"sigma": sigma}}],
    }), encoding="utf-8")
    return path


def test_create_pipeline_run_stores_preprocessing_id(tmp_path):
    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db,
            input_path=str(tmp_path / "in"),
            output_path=str(tmp_path / "out"),
            preprocessing_id="deadbeef",
        )
        run_id = run.run_id
    finally:
        db.close()

    db = SessionLocal()
    try:
        assert get_pipeline_run(db, run_id).preprocessing_id == "deadbeef"
    finally:
        db.close()


def test_create_pipeline_run_defaults_to_none(tmp_path):
    """Старые пути (CLI, orchestrator) не передают идентификатор."""
    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db,
            input_path=str(tmp_path / "in2"),
            output_path=str(tmp_path / "out2"),
        )
        assert run.preprocessing_id is None
    finally:
        db.close()


def test_pin_preprocessing_id_reads_the_given_config(tmp_path):
    import numbering

    path = _config(tmp_path / "c.yaml", sigma=1.5)
    assert numbering.pin_preprocessing_id(path) == \
        compute_preprocessing_id(str(path))


def test_pin_preprocessing_id_is_none_without_a_config(tmp_path):
    """Отсутствие конфига не должно валить старт прогона."""
    import numbering

    assert numbering.pin_preprocessing_id(tmp_path / "нет.yaml") is None


def test_different_configs_give_different_ids(tmp_path):
    """Иначе тест на закрепление ничего не доказывает."""
    import numbering

    a = numbering.pin_preprocessing_id(
        _config(tmp_path / "a" / "c.yaml", sigma=1.0))
    b = numbering.pin_preprocessing_id(
        _config(tmp_path / "b" / "c.yaml", sigma=2.0))
    assert a is not None and b is not None and a != b


def test_pinning_also_records_what_the_id_means(tmp_path):
    """Закрепить метку, по которой нельзя узнать настройки, бессмысленно."""
    import numbering
    from preprocessing_version import get_version_config

    path = _config(tmp_path / "recorded" / "c.yaml", sigma=4.25)
    prep_id = numbering.pin_preprocessing_id(path)

    stored = get_version_config(prep_id)
    assert stored is not None
    assert stored["steps"][0]["params"]["sigma"] == 4.25
