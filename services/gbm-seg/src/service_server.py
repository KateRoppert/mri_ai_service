#!/usr/bin/env python3
"""
GBM segmentation service.

Concrete service for glioblastoma segmentation using nnUNet v1
(Task115_AllData5foldsMeta). Inherits the HTTP/queue/GPU infrastructure
from common.service_base.ServiceBase and implements only:

    load_model()         — verifies weights are mounted and accessible
    run_inference()      — runs nnUNet v1 inference on requested files
    
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
import yaml
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


def _load_server_config() -> dict[str, Any]:
    """Load server_config.yaml from the script directory or SERVER_CONFIG env var."""
    config_path = os.getenv("SERVER_CONFIG")
    if not config_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, "server_config.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("server", {})
        except Exception as e:
            print(f"WARNING: failed to load {config_path}: {e}")

    # Fallback defaults
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

# nnUNet v1 import path + RESULTS_FOLDER env var must be set before importing
# the inference module
_nnunet_path = _server_config.get("nnunet_path", "/app/nnUNet")
if _nnunet_path not in sys.path:
    sys.path.insert(0, _nnunet_path)

_nnunet_v1_base = _server_config.get("nnunet_models", "/app/nnUNetv1_data")
os.environ["nnUNet_raw_data_base"] = os.path.join(_nnunet_v1_base, "nnUNet_raw")
os.environ["nnUNet_preprocessed"] = os.path.join(_nnunet_v1_base, "nnUNet_preprocessed")
os.environ["RESULTS_FOLDER"] = _nnunet_v1_base

# TMP directory used by run_inference for nnUNet-convention staging
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TMP_DIR = os.path.join(_BASE_DIR, _server_config.get("tmp_dir", "data/tmp"))
os.makedirs(_TMP_DIR, exist_ok=True)


# ============================================================================
# nnUNet v1 inference wrapper
# ============================================================================

def _prepare_files_for_unet(files: dict[str, str], prefix: str) -> tuple[str, str]:
    """
    Rename four modality files into nnUNet v1 convention:
        prefix_0000.nii.gz  (T1)
        prefix_0001.nii.gz  (T1c)
        prefix_0002.nii.gz  (T2)
        prefix_0003.nii.gz  (T2-FLAIR)
    Returns (input_folder, output_folder) inside TMP_DIR.
    """
    rnd_suffix = hashlib.sha1(os.urandom(512)).hexdigest()[:10]
    rnd_str = prefix + rnd_suffix

    in_path = os.path.join(_TMP_DIR, f"{rnd_str}_in")
    out_path = os.path.join(_TMP_DIR, f"{rnd_str}_out")
    os.mkdir(in_path)
    os.mkdir(out_path)

    shutil.move(files["t1"],   os.path.join(in_path, f"{prefix}0000.nii.gz"))
    shutil.move(files["t1c"],  os.path.join(in_path, f"{prefix}0001.nii.gz"))
    shutil.move(files["t2"],   os.path.join(in_path, f"{prefix}0002.nii.gz"))
    shutil.move(files["t2fl"], os.path.join(in_path, f"{prefix}0003.nii.gz"))

    return in_path, out_path


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
        self._task_name = _server_config.get("task_name", "Task115_AllData5foldsMeta")

    # -----------------------------------------------------------------------
    # Abstract methods from ServiceBase
    # -----------------------------------------------------------------------

    async def load_model(self) -> None:
        """Verify model weights are accessible. nnUNet v1 loads lazily per call."""
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

        self.log.info("found %d fold(s) for task %s: %s",
                      len(folds), self._task_name, sorted(folds))

    async def run_inference(self, payload: dict[str, Any], gpu_id: int) -> dict[str, Any]:
        """
        Run nnUNet v1 segmentation. Expects new /predict contract payload:
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
        folds = tuple(options.get("folds", [0, 1, 2, 3, 4]))

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
        in_path, out_path = _prepare_files_for_unet(staged, prefix)

        # Run nnUNet in a thread executor
        loop = asyncio.get_running_loop()
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

        # Clean up staging directories
        shutil.rmtree(in_path, ignore_errors=True)
        shutil.rmtree(out_path, ignore_errors=True)

        return {
            "mask_path": str(mask_path),
            "output_classes": {
                "0": "background",
                "1": "ed",   # edema
                "2": "net",  # non-enhancing tumor
                "3": "et",   # enhancing tumor
                "4": "ncr",  # necrotic core
            },
            "lesion_type": "glioblastoma",
            "model": {
                "framework": "nnunetv1",
                "task": self._task_name,
                "folds_used": list(folds),
                "tta": use_tta,
            },
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