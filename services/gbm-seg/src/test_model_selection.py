"""
Tests for model preset selection (services/gbm-seg/src/model_selection.py).
Pure config logic — no GPU, no nnUNet install required.
"""
import os

import pytest

from model_selection import (
    build_v2_model_folder,
    channel_order_from_dataset_json,
    get_label_map,
    load_server_config,
    normalize_channel_name,
    resolve_active_model,
    resolve_channel_order,
    v1_base_path,
)


def test_resolve_active_model_picks_named_preset():
    config = {
        "active_model": "v2_meta_segresnet",
        "models": {
            "v1_task115": {
                "framework": "nnunetv1",
                "nnunet_models": "/app/nnUNetv1_data",
                "task_name": "Task115_AllData5foldsMeta",
            },
            "v2_meta_segresnet": {
                "framework": "nnunetv2",
                "nnunet_models": "/app/nnUNetv2_data",
                "dataset_name": "Dataset111_Meta",
                "trainer": "nnUNetTrainerSegResNet",
                "plans": "nnUNetPlans",
                "configuration": "3d_fullres",
                "checkpoint": "checkpoint_final.pth",
            },
        },
    }

    name, preset = resolve_active_model(config)

    assert name == "v2_meta_segresnet"
    assert preset["framework"] == "nnunetv2"
    assert preset["dataset_name"] == "Dataset111_Meta"


def test_resolve_active_model_raises_on_unknown_active_model():
    config = {
        "active_model": "does_not_exist",
        "models": {"v1_task115": {"framework": "nnunetv1"}},
    }

    with pytest.raises(ValueError, match="does_not_exist"):
        resolve_active_model(config)


def test_resolve_active_model_falls_back_to_legacy_flat_config():
    # A pre-presets config (no `models`/`active_model` keys).
    config = {
        "nnunet_models": "/app/nnUNetv1_data",
        "task_name": "Task115_AllData5foldsMeta",
    }

    name, preset = resolve_active_model(config)

    assert name == "default"
    assert preset["framework"] == "nnunetv1"
    assert preset["task_name"] == "Task115_AllData5foldsMeta"


def test_build_v2_model_folder_matches_nnunet_v2_layout():
    preset = {
        "nnunet_models": "/app/nnUNetv2_data",
        "dataset_name": "Dataset111_Meta",
        "trainer": "nnUNetTrainerSegResNet",
        "plans": "nnUNetPlans",
        "configuration": "3d_fullres",
    }

    folder = build_v2_model_folder(preset)

    assert folder == (
        "/app/nnUNetv2_data/Dataset111_Meta/"
        "nnUNetTrainerSegResNet__nnUNetPlans__3d_fullres"
    )


def test_v1_base_path_finds_v1_preset_even_when_v2_is_active():
    config = {
        "active_model": "v2_meta_segresnet",
        "models": {
            "v1_task115": {
                "framework": "nnunetv1",
                "nnunet_models": "/app/nnUNetv1_data",
            },
            "v2_meta_segresnet": {
                "framework": "nnunetv2",
                "nnunet_models": "/app/nnUNetv2_data",
            },
        },
    }

    assert v1_base_path(config) == "/app/nnUNetv1_data"


def test_v1_base_path_defaults_when_no_v1_preset_exists():
    config = {
        "active_model": "v2_meta_segresnet",
        "models": {
            "v2_meta_segresnet": {
                "framework": "nnunetv2",
                "nnunet_models": "/app/nnUNetv2_data",
            },
        },
    }

    assert v1_base_path(config) == "/app/nnUNetv1_data"


def test_load_server_config_returns_empty_dict_for_missing_file():
    assert load_server_config("/nonexistent/path/server_config.yaml") == {}


def test_load_server_config_reads_real_file(tmp_path):
    config_path = tmp_path / "server_config.yaml"
    config_path.write_text(
        "server:\n"
        "  active_model: v1_task115\n"
        "  models:\n"
        "    v1_task115:\n"
        "      framework: nnunetv1\n"
    )

    config = load_server_config(str(config_path))

    assert config["active_model"] == "v1_task115"
    assert config["models"]["v1_task115"]["framework"] == "nnunetv1"


# ---------------------------------------------------------------------------
# Input channel order
# ---------------------------------------------------------------------------

def test_channel_order_read_from_dataset_json():
    # Dataset111_Meta's real channel_names — deliberately NOT the historical
    # t1/t1c/t2/flair order the pipeline used to hardcode.
    dataset_json = {
        "channel_names": {"0": "T1C", "1": "T1N", "2": "T2F", "3": "T2W"},
    }

    assert channel_order_from_dataset_json(dataset_json) == ["t1c", "t1", "t2fl", "t2"]


def test_channel_order_sorts_by_index_not_string():
    # "10" must not sort before "2" — indices are numeric.
    dataset_json = {
        "channel_names": {"0": "T1", "1": "T1C", "2": "T2", "3": "FLAIR"},
    }

    assert channel_order_from_dataset_json(dataset_json) == ["t1", "t1c", "t2", "t2fl"]


def test_normalize_channel_name_accepts_known_aliases():
    assert normalize_channel_name("T1CE") == "t1c"
    assert normalize_channel_name("t1gd") == "t1c"
    assert normalize_channel_name("FLAIR") == "t2fl"
    assert normalize_channel_name("T2W") == "t2"
    assert normalize_channel_name("T1N") == "t1"


def test_normalize_channel_name_rejects_unknown():
    with pytest.raises(ValueError, match="unknown modality"):
        normalize_channel_name("DWI")


def test_preset_channel_order_wins_over_dataset_json():
    preset = {"channel_order": ["t1", "t1c", "t2", "t2fl"]}
    dataset_json = {"channel_names": {"0": "T1C", "1": "T1N", "2": "T2F", "3": "T2W"}}

    assert resolve_channel_order(preset, dataset_json) == ["t1", "t1c", "t2", "t2fl"]


def test_v1_preset_keeps_historical_channel_order():
    # The v1 model has no dataset.json; its behaviour must not change.
    preset = {"framework": "nnunetv1", "channel_order": ["t1", "t1c", "t2", "t2fl"]}

    assert resolve_channel_order(preset, None) == ["t1", "t1c", "t2", "t2fl"]


def test_channel_order_falls_back_to_historical_default():
    assert resolve_channel_order({}, None) == ["t1", "t1c", "t2", "t2fl"]


# ---------------------------------------------------------------------------
# Output label mapping
# ---------------------------------------------------------------------------

def test_label_map_parsed_from_preset():
    preset = {"label_map": {1: 1, 2: 2, 3: 4}}

    assert get_label_map(preset) == {1: 1, 2: 2, 3: 4}


def test_label_map_coerces_yaml_string_keys():
    preset = {"label_map": {"1": "1", "2": "2", "3": "4"}}

    assert get_label_map(preset) == {1: 1, 2: 2, 3: 4}


def test_canonical_model_has_no_label_map():
    # The v1 preset declares none — its output must pass through untouched.
    assert get_label_map({"framework": "nnunetv1"}) == {}
