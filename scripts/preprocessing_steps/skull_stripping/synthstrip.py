"""
SynthStrip wrapper (Hoopes et al. 2022) via the ``mri_synthstrip`` CLI.

Modality-agnostic brain extraction (surfa / FreeSurfer). Invoked as a
subprocess so Stage 05 does not import the SynthStrip stack in-process.
CPU by default: it must not occupy the GPU pool used by HD-BET.
"""

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import SkullStripperBase

logger = logging.getLogger(__name__)

_TIMEOUT_SEC = 900


class SynthStripStripper(SkullStripperBase):
    """SynthStrip — synthetic-data, modality-agnostic brain extraction."""

    name = "synthstrip"
    uses_gpu = False

    def is_available(self) -> bool:
        return shutil.which("mri_synthstrip") is not None

    def build_command(
        self,
        input_path: Path,
        output_path: Path,
        mask_path: Optional[Path] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> list:
        params = params or {}
        nested = params.get("tool_params")
        nested = nested if isinstance(nested, dict) else {}
        border = params.get("border", nested.get("border"))

        cmd = [
            "mri_synthstrip",
            "-i", str(input_path),
            "-o", str(output_path),
        ]
        if mask_path is not None:
            cmd += ["-m", str(mask_path)]
        if border is not None:
            cmd += ["-b", str(border)]
        # Official CLI also has -g/--gpu and --no-csf. GPU is not wired here:
        # this plugin stays off the HD-BET device pool (uses_gpu = False).
        return cmd

    def strip(
        self,
        input_path: Path,
        output_path: Path,
        mask_path: Optional[Path] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        params = params or {}
        start_time = time.time()

        try:
            if not self.is_available():
                raise RuntimeError(
                    "mri_synthstrip not found on PATH. Install with "
                    "`pip install surfa` (see services/skull-stripping/synthstrip)."
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if mask_path is not None:
                mask_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = self.build_command(input_path, output_path, mask_path, params)
            logger.info("Running SynthStrip on %s", input_path.name)
            logger.debug("SynthStrip command: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SEC,
                env=os.environ.copy(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"SynthStrip failed with return code {result.returncode}: "
                    f"{(result.stderr or result.stdout or '')[:500]}"
                )
            if not output_path.exists():
                raise RuntimeError(
                    f"SynthStrip reported success but produced no output at {output_path}"
                )

            mask_created = mask_path is not None and mask_path.exists()
            if mask_path is not None and not mask_created:
                logger.warning("SynthStrip produced no mask at %s", mask_path)

            processing_time = time.time() - start_time
            logger.info("SynthStrip completed in %.2f seconds", processing_time)
            return {
                "success": True,
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_created else None,
                "processing_time": processing_time,
            }
        except subprocess.TimeoutExpired:
            logger.error("SynthStrip timeout on %s", input_path.name)
            return {
                "success": False,
                "error": f"SynthStrip timeout (exceeded {_TIMEOUT_SEC // 60} minutes)",
            }
        except Exception as e:
            logger.error("Error running SynthStrip on %s: %s", input_path.name, e)
            return {"success": False, "error": str(e)}
