"""KappaUploader берёт preprocessing_id с прогона, а файл — только как фолбэк.

Тест на сам дефект KI-085: прогон стартовал при одном конфиге, выгружается
при другом, а в реестр обязан попасть идентификатор СТАРТА.
"""
import pytest
import yaml

from kappa_uploader import KappaUploader
from preprocessing_version import compute_preprocessing_id


def _config(path, sigma):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "atlas": {"name": "MNI152", "filename": "mni.nii.gz"},
        "modalities": ["t1"],
        "steps": [{"name": "smooth", "enabled": True,
                   "params": {"sigma": sigma}}],
    }), encoding="utf-8")
    return path


def _uploader(tmp_path, **kwargs):
    return KappaUploader(
        run_id="run-1",
        output_path=str(tmp_path / "out"),
        token="t",
        user_id=1,
        user_type_id=3,
        lesion_type="glioblastoma",
        **kwargs,
    )


def test_pinned_id_wins_over_the_live_config(tmp_path):
    at_start = compute_preprocessing_id(
        str(_config(tmp_path / "start" / "c.yaml", sigma=1.0)))
    live = _config(tmp_path / "live" / "c.yaml", sigma=9.0)

    uploader = _uploader(
        tmp_path,
        preprocessing_config_path=str(live),
        preprocessing_id=at_start,
    )
    assert uploader.preprocessing_id == at_start
    assert uploader.preprocessing_id != compute_preprocessing_id(str(live))


def test_pinned_id_means_the_config_file_is_never_read(tmp_path):
    """Жёсткая проверка: путь заведомо несуществующий, конструктор обязан пройти."""
    uploader = _uploader(
        tmp_path,
        preprocessing_config_path=str(tmp_path / "нет-такого.yaml"),
        preprocessing_id="deadbeef",
    )
    assert uploader.preprocessing_id == "deadbeef"


def test_without_a_pinned_id_the_file_is_used(tmp_path):
    """Путь деградации для прогонов, стартовавших до этой правки."""
    live = _config(tmp_path / "legacy" / "c.yaml", sigma=3.0)
    uploader = _uploader(tmp_path, preprocessing_config_path=str(live))
    assert uploader.preprocessing_id == compute_preprocessing_id(str(live))


def test_no_pinned_id_and_no_file_still_fails_loudly(tmp_path):
    """Молча проглотить отсутствие и файла, и метки нельзя."""
    with pytest.raises(OSError):
        _uploader(tmp_path,
                  preprocessing_config_path=str(tmp_path / "нет.yaml"))
