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

import os
import sys
from pathlib import Path
from typing import Any

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

    async def run_inference(self, payload: dict[str, Any], gpu_id: int) -> dict[str, Any]:
        """
        STAGE 3.1 SKELETON: not implemented.
        Filled in by Stage 3.2 with the actual nnUNet v2 predictor pipeline.
        """
        raise NotImplementedError(
            "ms-seg run_inference is not implemented yet (Stage 3.1 skeleton). "
            "Will be added in Stage 3.2."
        )


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    MsSegService().serve(
        host=_server_config.get("host", "0.0.0.0"),
        port=int(_server_config.get("port", 5001)),
        debug=bool(_server_config.get("debug", False)),
    )