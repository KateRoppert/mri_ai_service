"""Развёртка по параллелизму: планирование, возобновление, точка насыщения."""
import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sweep_stage import (build_cell_config, cell_key, done_cells, plan_cells,
                         saturation_point, throughput_rows)


BASE = {
    "general": {"root_input_dir": "/in", "root_output_dir": "/out",
                "skip_existing": True, "max_subjects": 3},
    "stages": {
        "stage_01_reorganize": {"enabled": True, "args": {"workers": 8, "mode": "parallel"}},
        "stage_05_preprocessing": {"enabled": True, "args": {"workers": 2, "mode": "sequential"}},
        "stage_08_anatomical_analysis": {"enabled": False, "args": {"workers": 4}},
    },
}


# ── планирование и возобновление ──────────────────────────────────────────


def test_repeats_are_separate_cells_not_averaged_in_place():
    """Разброс между повторами — сам результат: прошлая статья показывала его
    полосой на графике, а не прятала в среднее."""
    cells = plan_cells("stage_05_preprocessing", [1, 4], repeats=2)

    assert len(cells) == 4
    assert [c["repeat"] for c in cells] == [1, 2, 1, 2]


def test_resume_skips_only_what_succeeded(tmp_path):
    """Прерванная развёртка должна дочитать, а не начать заново — но
    упавшую клетку надо пересчитать, а не считать сделанной."""
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps({"stage": "s", "workers": 1, "repeat": 1, "ok": True}) + "\n"
        + json.dumps({"stage": "s", "workers": 2, "repeat": 1, "ok": False}) + "\n"
        + "{битый json\n"          # обрыв на середине строки при падении
        + '"валидный json, но не запись"\n'
        + "\n",
        encoding="utf-8")

    done = done_cells(results)

    assert done == {"s|w1|r1"}


def test_a_missing_results_file_means_nothing_is_done(tmp_path):
    assert done_cells(tmp_path / "absent.jsonl") == set()


# ── конфиг клетки ─────────────────────────────────────────────────────────


def test_only_the_target_stage_is_enabled(tmp_path):
    cfg = build_cell_config(BASE, "stage_05_preprocessing", 4, tmp_path, None)

    enabled = {n for n, s in cfg["stages"].items() if s["enabled"]}
    assert enabled == {"stage_05_preprocessing"}


def test_the_worker_count_and_parallel_mode_are_forced(tmp_path):
    """Последовательная ветка этапа — другой путь исполнения; подмешивать её
    в кривую нельзя даже при одном воркере."""
    cfg = build_cell_config(BASE, "stage_05_preprocessing", 1, tmp_path, None)

    args = cfg["stages"]["stage_05_preprocessing"]["args"]
    assert args["workers"] == 1
    assert args["mode"] == "parallel"


def test_skip_existing_is_switched_off(tmp_path):
    """Иначе вторая клетка «пропустит» готовую работу и окажется мгновенной —
    самая дорогая ошибка измерения, какую здесь можно сделать."""
    cfg = build_cell_config(BASE, "stage_05_preprocessing", 2, tmp_path, None)

    assert cfg["general"]["skip_existing"] is False


def test_the_base_config_is_not_mutated(tmp_path):
    build_cell_config(BASE, "stage_05_preprocessing", 16, tmp_path, 99)

    assert BASE["stages"]["stage_05_preprocessing"]["args"]["workers"] == 2
    assert BASE["general"]["max_subjects"] == 3


def test_an_unknown_stage_is_refused(tmp_path):
    with pytest.raises(KeyError):
        build_cell_config(BASE, "stage_99_nonexistent", 1, tmp_path, None)


def test_no_limit_means_the_key_is_removed(tmp_path):
    """`max_subjects` из локального конфига не должен молча урезать развёртку."""
    cfg = build_cell_config(BASE, "stage_05_preprocessing", 1, tmp_path, None)

    assert "max_subjects" not in cfg["general"]


def test_the_cell_config_is_valid_yaml(tmp_path):
    cfg = build_cell_config(BASE, "stage_01_reorganize", 6, tmp_path, 10)

    assert yaml.safe_load(yaml.safe_dump(cfg)) == cfg


# ── кривая и точка насыщения ──────────────────────────────────────────────


def _rows(pairs):
    return [{"stage": "s", "workers": w, "repeat": 1, "ok": True,
             "sessions_done": 100, "wall_seconds": sec} for w, sec in pairs]


def test_throughput_uses_the_median_of_repeats():
    rows = [{"stage": "s", "workers": 4, "repeat": r, "ok": True,
             "sessions_done": 10, "wall_seconds": sec}
            for r, sec in enumerate([10.0, 11.0, 100.0], 1)]

    curve = throughput_rows(rows)

    assert curve[0]["seconds_median"] == 11.0  # не 40.3
    assert curve[0]["seconds_max"] - curve[0]["seconds_min"] == 90.0


def test_failed_cells_are_left_out_of_the_curve():
    rows = _rows([(1, 100.0), (2, 50.0)])
    rows.append({"stage": "s", "workers": 4, "repeat": 1, "ok": False,
                 "sessions_done": None, "wall_seconds": 1.0})

    assert [r["workers"] for r in throughput_rows(rows)] == [1, 2]


def test_saturation_is_the_last_level_before_the_gain_drops():
    """Определение из прошлой статьи: наименьшее число процессов, после
    которого прирост меньше 5%."""
    # 100 -> 50 -> 34 -> 33 с: прирост на 8 воркерах всего 3%
    curve = throughput_rows(_rows([(1, 100.0), (2, 50.0), (4, 34.0), (8, 33.0)]))

    assert saturation_point(curve) == 4


def test_linear_scaling_saturates_at_the_last_measured_level():
    curve = throughput_rows(_rows([(1, 100.0), (2, 50.0), (4, 25.0)]))

    assert saturation_point(curve) == 4


def test_a_stage_that_does_not_scale_saturates_at_one():
    curve = throughput_rows(_rows([(1, 100.0), (2, 99.0), (4, 98.0)]))

    assert saturation_point(curve) == 1


def test_a_single_level_has_no_saturation_point():
    assert saturation_point(throughput_rows(_rows([(1, 100.0)]))) is None


def test_the_cell_key_is_stable():
    assert cell_key({"stage": "s", "workers": 4, "repeat": 2}) == "s|w4|r2"


def test_the_stages_own_skip_flag_is_cleared_too(tmp_path):
    """Самая дорогая ошибка измерения: `general.skip_existing` оркестратор
    только пишет в лог, а решает собственный ключ этапа в `args`. С ним
    вторая клетка пропустила бы готовую работу и оказалась мгновенной.

    Написания разные у разных этапов (`skip-existing` у 04–08, `force` у
    01–04), и передаются они буква в букву (KI-071)."""
    base = {
        "general": {"root_input_dir": "/in", "root_output_dir": "/out"},
        "stages": {
            "stage_08_anatomical_analysis": {
                "enabled": True,
                "args": {"workers": 8, "skip-existing": True},
            },
            "stage_01_reorganize": {
                "enabled": True, "args": {"workers": 8, "force": False},
            },
        },
    }

    eight = build_cell_config(base, "stage_08_anatomical_analysis", 2,
                              tmp_path, None)
    one = build_cell_config(base, "stage_01_reorganize", 2, tmp_path, None)

    assert eight["stages"]["stage_08_anatomical_analysis"]["args"]["skip-existing"] is False
    assert one["stages"]["stage_01_reorganize"]["args"]["force"] is True


def test_the_script_path_is_made_absolute(tmp_path):
    """Конфиг клетки лежит в каталоге результатов, а `script` разрешается
    относительно каталога конфига — относительный путь искал бы `scripts/`
    рядом с результатами."""
    base = {
        "general": {"root_input_dir": "/in", "root_output_dir": "/out"},
        "stages": {"stage_05_preprocessing": {
            "enabled": True, "script": "scripts/05_preprocessing.py",
            "args": {"workers": 1}}},
    }

    cfg = build_cell_config(base, "stage_05_preprocessing", 1, tmp_path, None)
    script = cfg["stages"]["stage_05_preprocessing"]["script"]

    assert Path(script).is_absolute()
    assert Path(script).exists(), script


def test_the_environment_identifies_the_machine_and_the_code():
    """Числа без этого артефакта ничего не значат: Блок 0 требует
    зафиксировать железо и версии до любого прогона. Имя хоста тут
    недостаточно — на рабочем ноуте `platform.node()` даёт бесполезную "1"."""
    from sweep_stage import environment

    env = environment()

    assert env["cpu_model"], "модель процессора обязательна"
    assert env["os_cpu_count"] and env["cpu_budget"]
    assert env["git_commit"] and len(env["git_commit"]) == 40
    assert isinstance(env["git_dirty"], bool)
    for optional in ("gpus", "antspyx", "fsl_bet", "dcm2niix", "nvidia_driver"):
        assert optional in env, f"ключ {optional} должен быть, пусть и None"
