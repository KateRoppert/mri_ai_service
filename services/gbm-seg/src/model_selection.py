"""
Model preset selection for the GBM segmentation service.

Pure config logic, no torch/quart/nnUNet imports — kept separate from
service_server.py so it can be unit-tested without a GPU or the nnUNet
Docker image.
"""
from __future__ import annotations

import os
from typing import Any


def load_server_config(config_path: str) -> dict[str, Any]:
    """Load server_config.yaml. Returns {} if the file is missing/unreadable."""
    import yaml

    if not os.path.exists(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("server", {})
    except Exception as e:
        print(f"WARNING: failed to load {config_path}: {e}")
        return {}


def resolve_active_model(server_config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """
    Pick the active model preset from server_config["models"].

    Returns (preset_name, preset_dict). Falls back to a synthetic v1 preset
    built from legacy top-level keys (nnunet_models/task_name) when the
    config predates the `models`/`active_model` schema, so old configs keep
    working unchanged.
    """
    models = server_config.get("models")
    if not models:
        return "default", {
            "framework": "nnunetv1",
            "nnunet_models": server_config.get("nnunet_models", "/app/nnUNetv1_data"),
            "task_name": server_config.get("task_name", "Task115_AllData5foldsMeta"),
            "folds": server_config.get("folds", [0, 1, 2, 3, 4]),
        }

    active_name = server_config.get("active_model")
    if not active_name or active_name not in models:
        raise ValueError(
            f"server_config.yaml: active_model={active_name!r} not found in "
            f"models ({sorted(models)})"
        )
    return active_name, models[active_name]


def v1_base_path(server_config: dict[str, Any]) -> str:
    """
    nnUNet v1's RESULTS_FOLDER, resolved independently of which model is
    currently active — the v1 inference module is only imported lazily when
    a v1 preset actually runs, but the env vars must exist at import time
    regardless (nnUNet v1 reads them at module load).
    """
    models = server_config.get("models")
    if not models:
        return server_config.get("nnunet_models", "/app/nnUNetv1_data")
    for preset in models.values():
        if preset.get("framework") == "nnunetv1":
            return preset.get("nnunet_models", "/app/nnUNetv1_data")
    return "/app/nnUNetv1_data"


def build_v2_model_folder(preset: dict[str, Any]) -> str:
    """Build the nnUNet v2 trainer/plans/configuration folder from a preset."""
    return os.path.join(
        preset["nnunet_models"],
        preset["dataset_name"],
        f"{preset['trainer']}__{preset['plans']}__{preset['configuration']}",
    )


# ---------------------------------------------------------------------------
# Input channel order
# ---------------------------------------------------------------------------
# Every model declares which modality belongs on which input channel. Feeding
# them in the wrong order does not fail loudly — the model just produces
# garbage (observed: 107 predicted voxels instead of 58123 on the same case).
# The v1 model predates dataset.json here, so its order stays declared in the
# preset; v2 models carry channel_names in dataset.json and are read from there.

# Modality names as they appear in dataset.json channel_names, mapped to the
# internal modality keys used by _resolve_modalities (configs/lesion_types.yaml).
_CHANNEL_NAME_ALIASES = {
    # contrast-enhanced T1
    "t1c": "t1c", "t1ce": "t1c", "t1gd": "t1c", "t1-gd": "t1c", "t1_gd": "t1c",
    # native (non-contrast) T1
    "t1": "t1", "t1n": "t1", "t1w": "t1",
    # FLAIR
    "t2f": "t2fl", "flair": "t2fl", "t2flair": "t2fl", "t2-flair": "t2fl",
    "t2_flair": "t2fl", "t2fl": "t2fl",
    # T2
    "t2": "t2", "t2w": "t2",
}

DEFAULT_V1_CHANNEL_ORDER = ["t1", "t1c", "t2", "t2fl"]


def normalize_channel_name(name: str) -> str:
    """Map a dataset.json channel name (e.g. "T1C", "T2F") to a modality key."""
    key = str(name).strip().lower().replace(" ", "")
    if key not in _CHANNEL_NAME_ALIASES:
        raise ValueError(
            f"unknown modality name in dataset.json channel_names: {name!r}. "
            f"Add it to _CHANNEL_NAME_ALIASES in model_selection.py."
        )
    return _CHANNEL_NAME_ALIASES[key]


def channel_order_from_dataset_json(dataset_json: dict[str, Any]) -> list[str]:
    """
    Read channel_names from a nnUNet v2 dataset.json and return modality keys
    ordered by channel index: index 0 first, then 1, 2, …
    """
    channel_names = dataset_json.get("channel_names") or dataset_json.get("modality")
    if not channel_names:
        raise ValueError("dataset.json has no channel_names")
    ordered = sorted(channel_names.items(), key=lambda kv: int(kv[0]))
    return [normalize_channel_name(name) for _, name in ordered]


def resolve_channel_order(preset: dict[str, Any],
                          dataset_json: dict[str, Any] | None = None) -> list[str]:
    """
    Decide which modality goes on which input channel for the active preset.

    Order of precedence:
      1. `channel_order` declared explicitly in the preset (escape hatch)
      2. channel_names from the model's own dataset.json (v2 — authoritative)
      3. the historical v1 order (t1, t1c, t2, t2fl)
    """
    explicit = preset.get("channel_order")
    if explicit:
        return [normalize_channel_name(n) for n in explicit]
    if dataset_json:
        return channel_order_from_dataset_json(dataset_json)
    return list(DEFAULT_V1_CHANNEL_ORDER)


# ---------------------------------------------------------------------------
# Output label mapping
# ---------------------------------------------------------------------------
# The rest of the pipeline (compute_volumes, lobar analysis, Kappa reports,
# the viewer and Slicer colours) is written against one canonical scheme:
#     1 = NCR, 2 = ED, 3 = NET, 4 = ET
# A model that predicts a different scheme declares `label_map` in its preset
# so its raw output is translated into the canonical one before anything
# downstream sees it. A preset without `label_map` is already canonical and
# passes through untouched — which is what keeps the v1 model behaving
# exactly as before.


def get_label_map(preset: dict[str, Any]) -> dict[int, int]:
    """
    Return {raw_label: canonical_label} for the preset. Empty dict = identity
    (model already emits the canonical scheme).
    """
    raw = preset.get("label_map") or {}
    return {int(k): int(v) for k, v in raw.items()}
