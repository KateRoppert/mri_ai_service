#!/usr/bin/env python3
"""
GBM segmentation service.

Concrete service for glioblastoma segmentation, running whichever model
preset is active in server_config.yaml (nnUNet v1 or nnUNet v2 — see
model_selection.py). Inherits the HTTP/queue/GPU infrastructure from
common.service_base.ServiceBase and implements only:

    load_model()         — verifies the active preset's weights are mounted
    run_inference()      — runs inference on requested files

"""

from __future__ import annotations

import asyncio
import glob
import hashlib
import os
import shutil
import sys
import traceback
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import torch
from quart import jsonify, request, send_file, send_from_directory
from werkzeug.utils import secure_filename
import aiofiles

# Make services/common/ visible to imports (PYTHONPATH set in Dockerfile)
from common.service_base import ServiceBase


# ============================================================================
# Torch & nnUNet v1 environment setup
# ============================================================================
# These globals affect the entire process and must run before any nnUNet
# import. The block is kept verbatim from the previous simple_server.py.

# Ampere optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# CUDA device ordering by PCI bus (so GPU indices match nvidia-smi)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from common.torch_compat import enable_legacy_checkpoint_loading
enable_legacy_checkpoint_loading()


from model_selection import (
    build_v2_model_folder as _build_v2_model_folder,
    get_label_map as _get_label_map,
    load_server_config as _load_server_config_file,
    resolve_active_model as _resolve_active_model,
    resolve_channel_order as _resolve_channel_order,
    v1_base_path as _v1_base_path,
)


def _load_server_config() -> dict[str, Any]:
    """Load server_config.yaml from the script directory or SERVER_CONFIG env var."""
    config_path = os.getenv("SERVER_CONFIG")
    if not config_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "server_config.yaml")

    config = _load_server_config_file(config_path)
    if config:
        return config

    # Fallback defaults (legacy flat v1 config, no presets)
    return {
        "nnunet_path": "/app/nnUNet",
        "nnunet_models": "/app/nnUNetv1_data",
        "task_name": "Task115_AllData5foldsMeta",
        "gpu_ids": [0],
        "host": "0.0.0.0",
        "port": 5000,
        "debug": False,
        "max_parallel_tasks": 1,
    }


_server_config = _load_server_config()
_active_model_name, _active_model = _resolve_active_model(_server_config)

# nnUNet v1 import path + RESULTS_FOLDER env var must be set before importing
# the inference module
_nnunet_path = _server_config.get("nnunet_path", "/app/nnUNet")
if _nnunet_path not in sys.path:
    sys.path.insert(0, _nnunet_path)

_nnunet_v1_base = _v1_base_path(_server_config)
os.environ["nnUNet_raw_data_base"] = os.path.join(_nnunet_v1_base, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(_nnunet_v1_base, "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = _nnunet_v1_base

# nnUNet v2 custom trainer classes (e.g. nnUNetTrainerSegResNet) aren't part
# of the base nnunetv2 package — they live here so nnUNet v2's
# recursive_find_trainer_class_by_name() can discover them via nnUNet_extTrainer.
os.environ["nnUNet_extTrainer"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "external_trainers"
)

# TMP directory used by run_inference for nnUNet-convention staging
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TMP_DIR = os.path.join(_BASE_DIR, _server_config.get("tmp_dir", "data/tmp"))
os.makedirs(_TMP_DIR, exist_ok=True)


# ============================================================================
# nnUNet v1 inference wrapper
# ============================================================================

def _prepare_files_for_unet(files: dict[str, str], prefix: str,
                            channel_order: list[str]) -> tuple[str, str]:
    """
    Stage the modality files under nnUNet's naming convention, placing each
    modality on the channel index the active model expects:
        prefix_0000.nii.gz  <- channel_order[0]
        prefix_0001.nii.gz  <- channel_order[1]
        ...
    Returns (input_folder, output_folder) inside TMP_DIR.
    """
    rnd_suffix = hashlib.sha1(os.urandom(512)).hexdigest()[:10]
    rnd_str = prefix + rnd_suffix

    in_path = os.path.join(_TMP_DIR, f"{rnd_str}_in")
    out_path = os.path.join(_TMP_DIR, f"{rnd_str}_out")
    os.mkdir(in_path)
    os.mkdir(out_path)

    for idx, modality in enumerate(channel_order):
        shutil.move(files[modality], os.path.join(in_path, f"{prefix}{idx:04d}.nii.gz"))

    return in_path, out_path


def _apply_label_map(mask_path: Path, label_map: dict[int, int]) -> dict[str, int]:
    """
    Rewrite a mask's raw model labels into the pipeline's canonical scheme
    in place. No-op when label_map is empty (model already canonical).

    Returns the canonical label histogram, for logging.
    """
    import SimpleITK as sitk
    import numpy as np

    img = sitk.ReadImage(str(mask_path))
    arr = sitk.GetArrayFromImage(img)

    if label_map:
        # Build into a fresh array so overlapping source/target labels
        # (e.g. 3->4 while 1->1) cannot clobber each other mid-remap.
        remapped = np.zeros_like(arr)
        for raw, canonical in label_map.items():
            remapped[arr == raw] = canonical
        arr = remapped

        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(img)
        sitk.WriteImage(out, str(mask_path))

    values, counts = np.unique(arr[arr > 0], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def _run_nnunet_sync(gpu_id: int,
                     in_path: str,
                     out_path: str,
                     use_tta: bool,
                     folds: tuple[int, ...],
                     task_name: str) -> str:
    """
    Synchronous nnUNet v1 call. Must run inside a thread executor — it sets
    the CUDA device for the calling thread and invokes inference.predict_for_api.
    Returns empty string on success, error message otherwise.
    """
    torch.cuda.set_device(gpu_id)
    import inference as nnUNet_inference
    return nnUNet_inference.predict_for_api(
        in_path, out_path, use_tta, folds, task_name
    )


def _run_nnunetv2_sync(gpu_id: int,
                       in_path: str,
                       out_path: str,
                       use_tta: bool,
                       folds: tuple[int, ...],
                       model_folder: str,
                       checkpoint: str) -> str:
    """
    Synchronous nnUNet v2 call — mirrors _run_nnunet_sync. Must run inside a
    thread executor for the same reason (sets CUDA device for the calling
    thread before invoking prediction).
    """
    torch.cuda.set_device(gpu_id)
    import nnUNetv2_inference
    result = nnUNetv2_inference.predict_for_api(
        in_path, out_path, use_tta, folds, model_folder, checkpoint
    )
    # nnUNetPredictor's own CUDA context/allocator state can still be
    # settling when predict_from_files returns — the multiprocess
    # preprocessing/export workers it spawns tear down asynchronously.
    # Force a synchronous release before the GPU token goes back in the
    # pool, so the next queued job doesn't start on top of memory nnUNet
    # hasn't actually freed yet (observed: back-to-back jobs with zero gap
    # crash the whole process with no Python-level error).
    torch.cuda.synchronize(gpu_id)
    torch.cuda.empty_cache()
    return result


# ============================================================================
# Service class
# ============================================================================

class GbmSegService(ServiceBase):
    """
    Glioblastoma segmentation service. Wraps nnUNet v1 inference behind
    the unified ServiceBase contract.
    """

    service_id = "gbm-seg"
    service_type = "segmentation"
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "manifest.yaml"
    )

    def __init__(self) -> None:
        gpu_ids = _server_config.get("gpu_ids", [0])
        super().__init__(gpu_ids=gpu_ids)
        self._model_preset_name = _active_model_name
        self._model = _active_model
        self._framework = self._model.get("framework", "nnunetv1")
        self._default_folds = tuple(self._model.get("folds", [0, 1, 2, 3, 4]))
        # Raw model labels -> the pipeline's canonical 1=NCR/2=ED/3=NET/4=ET.
        # Empty for a model that is already canonical (the v1 preset).
        self._label_map = _get_label_map(self._model)
        # Filled in load_model(): which modality goes on which input channel.
        self._channel_order: list[str] = []
        if self._framework == "nnunetv2":
            self._model_folder = _build_v2_model_folder(self._model)
            self._checkpoint = self._model.get("checkpoint", "checkpoint_final.pth")
        else:
            self._task_name = self._model.get("task_name", "Task115_AllData5foldsMeta")

    # -----------------------------------------------------------------------
    # Abstract methods from ServiceBase
    # -----------------------------------------------------------------------

    async def load_model(self) -> None:
        """Verify model weights are accessible for the active preset."""
        if self._framework == "nnunetv2":
            if not os.path.isdir(self._model_folder):
                raise FileNotFoundError(
                    f"nnUNet v2 weights not found at {self._model_folder}. "
                    "Is the nnUNetv2_data volume mounted correctly?"
                )
            folds = [f for f in os.listdir(self._model_folder) if f.startswith("fold_")]
            if not folds:
                raise FileNotFoundError(f"No fold_* directories under {self._model_folder}")
            missing = [
                f for f in sorted(folds)
                if not os.path.isfile(os.path.join(self._model_folder, f, self._checkpoint))
            ]
            if missing:
                raise FileNotFoundError(
                    f"Checkpoint {self._checkpoint} missing in folds: {missing}"
                )
            self.log.info("model preset=%s: found %d fold(s) at %s (checkpoint=%s)",
                          self._model_preset_name, len(folds), self._model_folder,
                          self._checkpoint)

            # Channel order comes from the model's own dataset.json — the
            # model is the authority on which modality it expects where.
            import json
            dataset_json_path = os.path.join(self._model_folder, "dataset.json")
            dataset_json = None
            if os.path.isfile(dataset_json_path):
                with open(dataset_json_path, encoding="utf-8") as f:
                    dataset_json = json.load(f)
            self._channel_order = _resolve_channel_order(self._model, dataset_json)
            self.log.info("model preset=%s: input channel order %s | label_map %s",
                          self._model_preset_name, self._channel_order,
                          self._label_map or "identity (already canonical)")
            return

        # nnUNet v1: loads lazily per call, just verify weights are on disk.
        task_path = os.path.join(
            os.environ["RESULTS_FOLDER"],
            "nnUNet", "3d_fullres",
            self._task_name,
            "nnUNetTrainerV2__nnUNetPlansv2.1",
        )
        if not os.path.isdir(task_path):
            raise FileNotFoundError(
                f"nnUNet v1 weights not found at {task_path}. "
                "Is the nnUNetv1_data volume mounted correctly?"
            )

        folds = [f for f in os.listdir(task_path) if f.startswith("fold_")]
        if not folds:
            raise FileNotFoundError(f"No fold_* directories under {task_path}")

        # v1 has no dataset.json — order comes from the preset (or the
        # historical default), preserving this model's original behaviour.
        self._channel_order = _resolve_channel_order(self._model, None)
        self.log.info("model preset=%s: found %d fold(s) for task %s: %s | "
                      "input channel order %s | label_map %s",
                      self._model_preset_name, len(folds), self._task_name, sorted(folds),
                      self._channel_order,
                      self._label_map or "identity (already canonical)")

    def describe_active_model(self) -> dict[str, Any]:
        """
        Report what the active preset's canonical labels actually contain.

        Labels are always canonical (1=NCR, 2=ED, 3=NET, 4=ET) by the time
        anything downstream sees them, but their *content* differs per model:
        the v2 region model cannot separate necrosis from non-enhancing
        tumour, so both land in label 1 and label 3 stays empty. The viewer
        legend reads this so it never mislabels whichever model is loaded.
        """
        merges_ncr_net = self._label_map.get(1) == 1 and 3 not in self._label_map.values()
        return {
            "preset": self._model_preset_name,
            "framework": self._framework,
            "merged_ncr_net": merges_ncr_net,
            "class_labels_ru": {
                "1": "Некроз + неусиливающаяся опухоль (NCR/NET)" if merges_ncr_net
                     else "Некротическое ядро (NCR)",
                "2": "Отёк (ED)" if merges_ncr_net
                     else "Отёк (ED)",
                "3": "" if merges_ncr_net else "Неусиливающаяся опухоль (NET)",
                "4": "Усиливающаяся опухоль (ET)",
            },
        }

    async def run_inference(self, payload: dict[str, Any], gpu_id: int) -> dict[str, Any]:
        """
        Run segmentation with the active model preset (nnUNet v1 or v2).
        Expects new /predict contract payload:
            case_id, input_dir, output_dir, lesion_type, options.

        Reads four modality files from input_dir, runs inference,
        writes the resulting mask to output_dir.
        """
        case_id = payload["case_id"]
        input_dir = Path(payload["input_dir"])
        output_dir = Path(payload["output_dir"])
        options = payload.get("options", {}) or {}

        # Default options
        use_tta = bool(options.get("use_tta", False))
        folds = tuple(options.get("folds", self._default_folds))

        # Validate input modalities exist
        if not input_dir.is_dir():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")
        modality_paths = self._resolve_modalities(input_dir, case_id)

        # Create output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare nnUNet-convention staging dir
        prefix = f"{case_id}_"
        # Copy (not move) source files so the original input_dir is preserved
        staged = {k: str(self._stage_file(v, _TMP_DIR)) for k, v in modality_paths.items()}
        in_path, out_path = _prepare_files_for_unet(staged, prefix, self._channel_order)

        # Run nnUNet in a thread executor
        loop = asyncio.get_running_loop()
        if self._framework == "nnunetv2":
            func = partial(_run_nnunetv2_sync, gpu_id, in_path, out_path,
                           use_tta, folds, self._model_folder, self._checkpoint)
        else:
            func = partial(_run_nnunet_sync, gpu_id, in_path, out_path,
                           use_tta, folds, self._task_name)
        err = await loop.run_in_executor(None, func)
        if err:
            raise RuntimeError(f"nnUNet inference failed: {err}")

        # Move result to output_dir/mask.nii.gz
        result_files = glob.glob(os.path.join(out_path, "*.nii.gz"))
        if not result_files:
            raise RuntimeError(f"nnUNet produced no .nii.gz in {out_path}")
        mask_path = output_dir / "mask.nii.gz"
        shutil.move(result_files[0], mask_path)

        # Translate the model's raw labels into the canonical scheme the rest
        # of the pipeline reads (no-op for an already-canonical model).
        histogram = _apply_label_map(mask_path, self._label_map)
        self.log.info("case=%s preset=%s canonical labels: %s",
                      case_id, self._model_preset_name, histogram or "empty mask")

        # Clean up staging directories
        shutil.rmtree(in_path, ignore_errors=True)
        shutil.rmtree(out_path, ignore_errors=True)

        return {
            "mask_path": str(mask_path),
            # Canonical scheme, matching scripts/compute_volumes.py LABEL_MAP.
            # Labels are already translated above, so this holds for every
            # preset regardless of what the model itself emitted.
            "output_classes": {
                "0": "background",
                "1": "ncr",  # necrotic core (v2 preset: necrosis + non-enhancing)
                "2": "ed",   # peritumoral edema
                "3": "net",  # non-enhancing tumor (empty under the v2 preset)
                "4": "et",   # enhancing tumor
            },
            "lesion_type": "glioblastoma",
            "model": {
                "framework": self._framework,
                "preset": self._model_preset_name,
                "task": self._task_name if self._framework != "nnunetv2" else self._model_folder,
                "folds_used": list(folds),
                "tta": use_tta,
                "channel_order": list(self._channel_order),
                "label_map": {str(k): v for k, v in self._label_map.items()},
            },
            "label_histogram": {str(k): v for k, v in histogram.items()},
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _resolve_modalities(self, input_dir: Path, case_id: str) -> dict[str, str]:
        """
        Locate four modality files in input_dir.

        Naming conventions (in order of preference):
          1. {case_id}_t1.nii.gz, _t1c.nii.gz, _t2.nii.gz, _t2fl.nii.gz
          2. t1.nii.gz, t1c.nii.gz, t2.nii.gz, t2fl.nii.gz
          3. *_T1.nii.gz, *_T1c.nii.gz, *_T2.nii.gz, *_FLAIR.nii.gz
        """
        # Try strategy 1
        # Modality file matching supports several conventions:
        #   - lower-case short:  *_t1.nii.gz, *_t1c.nii.gz, ...
        #   - upper-case short:  *_T1.nii.gz, *_T1c.nii.gz, ...
        #   - BIDS:              *_T1w.nii.gz, *_ce-gd_T1w.nii.gz, *_T2w.nii.gz, *_FLAIR.nii.gz
        # `case_id` is the case identifier, files may or may not be prefixed with it.
        # We exclude T1c matches from T1 results by checking that 'c' or 'ce-' does
        # not follow T1/T1w (since glob is too coarse for this).

        def find(patterns: list[str]) -> list[Path]:
            results: list[Path] = []
            for p in patterns:
                results.extend(input_dir.glob(p))
            # Deduplicate while preserving order
            seen: set[Path] = set()
            deduped = []
            for path in results:
                if path not in seen:
                    seen.add(path)
                    deduped.append(path)
            return deduped

        # Find T1c first — we'll use it to exclude from T1 results
        t1c_matches = find([
            f"*{case_id}*t1c.nii.gz",
            f"*{case_id}*T1c.nii.gz",
            f"*{case_id}*T1ce.nii.gz",
            f"*{case_id}*ce-gd_T1w.nii.gz",
            f"*{case_id}*ce-*T1w.nii.gz",
        ])
        t1c_set = set(t1c_matches)

        t1_matches = [
            p for p in find([
                f"*{case_id}*t1.nii.gz",
                f"*{case_id}*T1.nii.gz",
                f"*{case_id}*T1w.nii.gz",
            ])
            if p not in t1c_set  # exclude contrast-enhanced T1
        ]

        candidates = {
            "t1":   t1_matches,
            "t1c":  t1c_matches,
            "t2":   find([
                f"*{case_id}*t2.nii.gz",
                f"*{case_id}*T2.nii.gz",
                f"*{case_id}*T2w.nii.gz",
            ]),
            "t2fl": find([
                f"*{case_id}*t2fl.nii.gz",
                f"*{case_id}*FLAIR.nii.gz",
                f"*{case_id}*flair.nii.gz",
            ]),
        }
        result = {}
        for key, matches in candidates.items():
            if not matches:
                raise FileNotFoundError(
                    f"no {key} modality found in {input_dir} "
                    f"(case_id={case_id})"
                )
            result[key] = str(matches[0])
        return result

    @staticmethod
    def _stage_file(src: str, tmp_dir: str) -> Path:
        """Copy a source file to tmp_dir under a unique name. Returns staged path."""
        src_path = Path(src)
        staged = Path(tmp_dir) / f"{uuid.uuid4().hex[:8]}_{src_path.name}"
        shutil.copy2(src_path, staged)
        return staged

# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    GbmSegService().serve(
        host=_server_config.get("host", "0.0.0.0"),
        port=int(_server_config.get("port", 5000)),
        debug=bool(_server_config.get("debug", False)),
    )