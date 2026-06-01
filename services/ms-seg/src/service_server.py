#!/usr/bin/env python3
"""
Multiple Sclerosis segmentation service.

Concrete service for MS lesion segmentation using nnUNet v2 with a custom
trainer (CATMIL by default) from Luu's SmallLesionMRI fork. The fork
REPLACES stock nnunetv2 inside this container; do not pip-install the
original nnunetv2 here.

This file is a SKELETON. Stage 3.1 only provides the infrastructure
(HTTP server, manifest, health endpoint). The actual nnUNet v2 predictor
wiring is filled in by Stage 3.2.

Skeleton behaviour:
  load_model()    — checks weights are mounted and the fork is importable;
                    does NOT instantiate any predictor yet.
  run_inference() — raises NotImplementedError; calls to /predict will
                    be queued by ServiceBase and fail with a clear error.

After Stage 3.1:
  docker compose up -d service-ms-seg → container should come up.
  curl :5001/health   → {"status": "ready"} (model_ready will be true once
                        the basic load_model passes).
  curl :5001/manifest → returns the manifest from Stage 3.1.
"""

from __future__ import annotations

import asyncio
import glob
import os
import shutil
import sys
import uuid
from functools import partial
from pathlib import Path
from typing import Any

import torch
import yaml

def _load_server_config() -> dict[str, Any]:
    """Load server_config.yaml from the script directory or SERVER_CONFIG env var."""
    config_path = os.getenv("SERVER_CONFIG")
    if not config_path:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(os.path.dirname(script_dir), "server_config.yaml")

    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return data.get("server", {})
        except Exception as e:
            print(f"WARNING: failed to load {config_path}: {e}")

    # Fallback defaults
    return {
        "nnunet_results": "/app/nnUNet_results",
        "dataset_name": "Dataset333_MSLesSeg",
        "trainer": "nnUNetTrainerCATMIL",
        "plans_identifier": "nnUNetPlans",
        "configuration": "3d_fullres",
        "gpu_ids": [0],
        "host": "0.0.0.0",
        "port": 5001,
        "debug": False,
    }


_server_config = _load_server_config()

# nnUNet v2 env vars — must be set before importing the fork's nnunetv2
os.environ["nnUNet_results"] = _server_config.get("nnunet_results", "/app/nnUNet_results")
# nnUNet v2 also reads these but we don't use them at inference time:
os.environ.setdefault("nnUNet_raw", "/app/nnUNet_raw")
os.environ.setdefault("nnUNet_preprocessed", "/app/nnUNet_preprocessed")

# ============================================================================
# Torch optimizations (Ampere / Tensor cores)
# ============================================================================
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from common.torch_compat import enable_legacy_checkpoint_loading
enable_legacy_checkpoint_loading()


# Make services/common/ visible (PYTHONPATH=/app set in Dockerfile)
from common.service_base import ServiceBase


# ============================================================================
# Service class
# ============================================================================

class MsSegService(ServiceBase):
    """
    MS lesion segmentation service. Wraps nnUNet v2 inference (Luu fork)
    behind the unified ServiceBase contract.

    NOTE: Stage 3.1 skeleton — run_inference() is not implemented yet.
    """

    service_id = "ms-seg"
    service_type = "segmentation"
    manifest_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "manifest.yaml"
    )

    def __init__(self) -> None:
        gpu_ids = _server_config.get("gpu_ids", [0])
        super().__init__(gpu_ids=gpu_ids)
        self._dataset_name = _server_config.get("dataset_name", "Dataset333_MSLesSeg")
        self._default_trainer = _server_config.get("trainer", "nnUNetTrainerCATMIL")
        self._plans_identifier = _server_config.get("plans_identifier", "nnUNetPlans")
        self._configuration = _server_config.get("configuration", "3d_fullres")

        # Staging directory for nnUNet-convention temporary files during inference.
        # Each request creates a unique sub-directory here.
        self._tmp_dir = Path("/app/data/tmp")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        # Predictor instance, populated by load_model() on startup
        self._predictor: Any = None  # nnUNetPredictor — type-imported lazily

    async def load_model(self) -> None:
        """
        Verify the fork is importable and weights are mounted at the expected
        path. Does NOT instantiate the predictor yet — that happens in
        Stage 3.2 inside run_inference (or as a cached singleton).
        """
        # 1. Importability check — the fork registers as 'nnunetv2'
        try:
            import nnunetv2  # noqa: F401
            self.log.info("nnunetv2 (fork) import OK from %s",
                          os.path.dirname(nnunetv2.__file__))
        except Exception as e:
            raise RuntimeError(
                f"Cannot import nnunetv2 (Luu fork). "
                f"Is SmallLesionMRI/slsseg properly installed? Error: {e}"
            )

        # 2. Trainer class importability check
        trainer_module = (
            f"nnunetv2.training.nnUNetTrainer.{self._default_trainer}"
        )
        try:
            __import__(trainer_module)
            self.log.info("Trainer module %s import OK", trainer_module)
        except Exception as e:
            self.log.warning(
                "Default trainer %s not importable: %s. "
                "Service will start but /predict with this trainer will fail.",
                self._default_trainer, e
            )

        # 3. Weights directory check
        results_root = Path(os.environ["nnUNet_results"])
        dataset_dir = results_root / self._dataset_name
        if not dataset_dir.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {dataset_dir}. "
                f"Is the volume mount correct? "
                f"Expected: services/ms-seg/nnUNet_results → /app/nnUNet_results"
            )

        # 4. List available trainer configurations
        available = sorted(
            d.name for d in dataset_dir.iterdir()
            if d.is_dir() and "__" in d.name and not d.name.startswith(".")
        )
        self.log.info(
            "Found %d trainer configuration(s) for %s: %s",
            len(available), self._dataset_name, available
        )

        # 5. Default trainer/plans/config folder existence check
        default_folder_name = (
            f"{self._default_trainer}__{self._plans_identifier}__{self._configuration}"
        )
        default_folder = dataset_dir / default_folder_name
        if not default_folder.is_dir():
            raise FileNotFoundError(
                f"Default trainer folder not found: {default_folder}. "
                f"Check server_config.yaml (trainer/plans/configuration values)."
            )

        # 6. Fold directories
        folds = sorted(
            f.name for f in default_folder.iterdir()
            if f.is_dir() and f.name.startswith("fold_")
        )
        self.log.info("Default trainer %s has folds: %s", default_folder_name, folds)

        # 7. Instantiate predictor — loads weights to GPU once, reused across requests
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

        # Use the first GPU from our pool. nnUNet v2 picks device at init,
        # so for multi-GPU we'd need one predictor per GPU. Single-GPU here.
        device = torch.device("cuda", self.gpu_ids[0]) if torch.cuda.is_available() else torch.device("cpu")
        self.log.info("Initializing nnUNetPredictor on device %s", device)

        self._predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,  # do preprocessing + postprocessing on GPU
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )

        self._predictor.initialize_from_trained_model_folder(
            model_training_output_dir=str(default_folder),
            use_folds=tuple(_server_config.get("folds", [0, 1, 2, 3, 4])),
            checkpoint_name="checkpoint_final.pth",
        )

        self.log.info("Predictor initialized; %s ready on %s",
                      self._default_trainer, device)

    async def run_inference(self, payload: dict[str, Any], gpu_id: int) -> dict[str, Any]:
        """
        Run nnUNet v2 inference (CATMIL) for one MS case.

        Expects payload per SPEC.md §3.1:
            case_id, input_dir, output_dir, lesion_type, options.

        Reads three modality files (T1, T2, FLAIR) from input_dir,
        writes a binary lesion mask to output_dir/mask.nii.gz.
        """
        case_id = payload["case_id"]
        input_dir = Path(payload["input_dir"])
        output_dir = Path(payload["output_dir"])
        options = payload.get("options", {}) or {}

        # Only CATMIL is wired up for now; explicit error if someone asks for another
        requested_trainer = options.get("trainer", self._default_trainer)
        if requested_trainer != self._default_trainer:
            raise NotImplementedError(
                f"Only {self._default_trainer} is supported in this version. "
                f"Requested: {requested_trainer}. Multi-trainer dispatch is planned."
            )

        if not input_dir.is_dir():
            raise FileNotFoundError(f"input_dir not found: {input_dir}")

        # Find three modality files (T1, T2, FLAIR)
        modality_paths = self._resolve_modalities(input_dir, case_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Stage files with nnUNet v2 channel-suffix convention
        # nnUNet v2 expects: case_identifier_0000.nii.gz, _0001.nii.gz, _0002.nii.gz
        # Channels per manifest.yaml: T1=0, T2=1, FLAIR=2
        staging_root = self._tmp_dir / f"req_{uuid.uuid4().hex[:8]}"
        staging_in = staging_root / "in"
        staging_out = staging_root / "out"
        staging_in.mkdir(parents=True, exist_ok=True)
        staging_out.mkdir(parents=True, exist_ok=True)

        # Use a simple stable identifier inside staging so we know what to look for
        stage_id = "case"
        shutil.copy2(modality_paths["t1"],    staging_in / f"{stage_id}_0000.nii.gz")
        shutil.copy2(modality_paths["t2"],    staging_in / f"{stage_id}_0001.nii.gz")
        shutil.copy2(modality_paths["flair"], staging_in / f"{stage_id}_0002.nii.gz")

        # Run predictor in a thread executor (it's a sync blocking call internally)
        loop = asyncio.get_running_loop()
        func = partial(self._predict_sync, gpu_id, staging_in, staging_out)
        try:
            await loop.run_in_executor(None, func)

            # Predictor writes case.nii.gz in staging_out
            result_files = glob.glob(str(staging_out / "*.nii.gz"))
            if not result_files:
                raise RuntimeError(f"nnUNetPredictor produced no .nii.gz in {staging_out}")

            mask_path = output_dir / "mask.nii.gz"
            shutil.move(result_files[0], mask_path)

        finally:
            # Always clean up staging — even on error
            shutil.rmtree(staging_root, ignore_errors=True)

        return {
            "mask_path": str(mask_path),
            "output_classes": {
                "0": "background",
                "1": "lesion",
            },
            "lesion_type": "multiple_sclerosis",
            "model": {
                "framework": "nnunetv2",
                "framework_source": "luumsk/SmallLesionMRI",
                "dataset_name": self._dataset_name,
                "trainer": self._default_trainer,
                "plans": self._plans_identifier,
                "configuration": self._configuration,
                "folds_used": _server_config.get("folds", [0, 1, 2, 3, 4]),
            },
        }

    def _predict_sync(self, gpu_id: int, input_folder: Path, output_folder: Path) -> None:
        """
        Synchronous predictor invocation — runs inside a thread executor.
        Sets CUDA device for the calling thread before launching inference.
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        self._predictor.predict_from_files(
            list_of_lists_or_source_folder=str(input_folder),
            output_folder_or_list_of_truncated_output_files=str(output_folder),
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            num_parts=1,
            part_id=0,
        )

    def _resolve_modalities(self, input_dir: Path, case_id: str) -> dict[str, Path]:
        """
        Locate three modality files in input_dir.

        Supports several naming conventions:
            - lower-case short:  *_t1.nii.gz, *_t2.nii.gz, *_flair.nii.gz, *_t2fl.nii.gz
            - upper-case short:  *_T1.nii.gz, *_T2.nii.gz, *_FLAIR.nii.gz
            - BIDS:              *_T1w.nii.gz, *_T2w.nii.gz, *_FLAIR.nii.gz

        Returns dict {modality_key: path}, where modality_key is t1/t2/flair.
        Raises FileNotFoundError if any modality is missing.
        """
        def find(patterns: list[str]) -> list[Path]:
            results: list[Path] = []
            for p in patterns:
                results.extend(input_dir.glob(p))
            seen: set[Path] = set()
            deduped: list[Path] = []
            for path in results:
                if path not in seen:
                    seen.add(path)
                    deduped.append(path)
            return deduped

        # Find T1c first to exclude from T1 — some pipelines include contrast T1 alongside T1
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
            if p not in t1c_set
        ]

        candidates = {
            "t1": t1_matches,
            "t2": find([
                f"*{case_id}*t2.nii.gz",
                f"*{case_id}*T2.nii.gz",
                f"*{case_id}*T2w.nii.gz",
            ]),
            "flair": find([
                f"*{case_id}*t2fl.nii.gz",
                f"*{case_id}*FLAIR.nii.gz",
                f"*{case_id}*flair.nii.gz",
            ]),
        }

        result = {}
        for key, matches in candidates.items():
            if not matches:
                raise FileNotFoundError(
                    f"No {key} modality found in {input_dir} (case_id={case_id})"
                )
            result[key] = matches[0]
        return result


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    MsSegService().serve(
        host=_server_config.get("host", "0.0.0.0"),
        port=int(_server_config.get("port", 5001)),
        debug=bool(_server_config.get("debug", False)),
    )