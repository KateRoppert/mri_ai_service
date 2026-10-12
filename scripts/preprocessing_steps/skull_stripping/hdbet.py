"""
HD-BET skull stripper (https://github.com/MIC-DKFZ/HD-BET).

Deep-learning brain extraction, trained on multi-modal tumour MRI — the
reason it is worth having alongside BET for this pipeline's data. Runs on
GPU when one is visible, falls back to CPU otherwise (much slower, but it
does finish).

Invoked through its CLI rather than its Python API: HD-BET 2.x pulls in
nnunetv2 and configures torch at import time, and doing that inside the
preprocessing worker process would drag GPU/CUDA state into a stage that
is otherwise pure CPU. A subprocess keeps that contained, and matches how
BET is already called here.

The subprocess is `hdbet_runner.py` run by the interpreter of the installed
`hd-bet` CLI, not the CLI itself: the CLI starts ~13 torch processes for one
file (5.1 GB); the runner uses the same predictor with one worker (~2 GB),
same outputs. See hdbet_runner.py.
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

# HD-BET writes the stripped image to the path given by -o, and the mask
# alongside it with this suffix appended before the extension.
_MASK_SUFFIX = "_bet.nii.gz"

# Brain extraction on CPU is minutes per volume, not seconds — the timeout
# has to accommodate the slow path, not just the GPU one.
_TIMEOUT_SEC = 1800

_RUNNER = Path(__file__).with_name("hdbet_runner.py")


def _hdbet_python() -> Optional[str]:
    """Interpreter HD-BET is installed in, from the `hd-bet` script's shebang
    (in the image HD-BET lives in FSL's Python, not the pipeline's)."""
    cli = shutil.which("hd-bet")
    if cli is None:
        return None
    try:
        with open(cli, "rb") as f:
            first = f.readline().decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].split()
    if not parts:
        return None
    # "#!/usr/bin/env python3" -> resolve the named interpreter
    if os.path.basename(parts[0]) == "env" and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0] if os.access(parts[0], os.X_OK) else None


class HdBetStripper(SkullStripperBase):
    """HD-BET — deep learning brain extraction."""

    name = "hdbet"
    uses_gpu = True

    def is_available(self) -> bool:
        """HD-BET is usable if its CLI is on PATH and its interpreter is known."""
        return _hdbet_python() is not None

    def strip(self,
              input_path: Path,
              output_path: Path,
              mask_path: Optional[Path] = None,
              params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        params = params or {}
        start_time = time.time()

        try:
            if not self.is_available():
                raise RuntimeError(
                    "HD-BET not found on PATH. Install it with "
                    "`pip install hd-bet` (see web.Dockerfile)."
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            # torch device string — "cuda", "cuda:0", "cpu" or "mps".
            # Note: HD-BET 1.x took a bare GPU index ("0"); 2.x rejects that
            # with "Invalid device string".
            device = str(params.get("device", "cuda"))
            # The runner always keeps the brain mask (the CLI needed
            # --save_bet_mask): the remaining modalities need it.
            cmd = [
                _hdbet_python(), str(_RUNNER),
                "-i", str(input_path),
                "-o", str(output_path),
                "-device", device,
            ]
            # Test-time augmentation: better masks, several times slower.
            # HD-BET recommends disabling it on CPU; keep that choice explicit.
            if params.get("disable_tta", device.startswith("cpu")):
                cmd.append("--disable_tta")

            logger.info("Running HD-BET on %s (device=%s)", input_path.name, device)
            logger.debug("HD-BET command: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SEC,
                env=os.environ.copy(),
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"HD-BET failed with return code {result.returncode}: "
                    f"{(result.stderr or result.stdout or '')[-1500:]}"
                )

            if not output_path.exists():
                raise RuntimeError(
                    f"HD-BET reported success but produced no output at {output_path}"
                )

            produced_mask = self._locate_mask(output_path)
            mask_created = False
            if produced_mask is not None:
                if mask_path is not None and produced_mask != mask_path:
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(produced_mask), str(mask_path))
                    logger.info("Saved brain mask to %s", mask_path)
                else:
                    mask_path = produced_mask
                    logger.info("Brain mask created at %s", produced_mask)
                mask_created = True
            else:
                logger.warning("HD-BET produced no mask next to %s", output_path)

            processing_time = time.time() - start_time
            logger.info("HD-BET completed in %.2f seconds", processing_time)

            return {
                "success": True,
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_created else None,
                "processing_time": processing_time,
            }

        except subprocess.TimeoutExpired:
            logger.error("HD-BET timeout on %s", input_path.name)
            return {
                "success": False,
                "error": f"HD-BET timeout (exceeded {_TIMEOUT_SEC // 60} minutes)",
            }
        except Exception as e:
            logger.error("Error running HD-BET on %s: %s", input_path.name, e)
            return {"success": False, "error": str(e)}

    @staticmethod
    def _locate_mask(output_path: Path) -> Optional[Path]:
        """
        Find the mask HD-BET wrote next to the stripped image. The exact
        name has varied between HD-BET versions, so probe the known
        spellings instead of assuming one.
        """
        base = output_path.name
        for ext in (".nii.gz", ".nii"):
            if base.endswith(ext):
                base = base[: -len(ext)]
                break

        candidates = [
            output_path.parent / f"{base}{_MASK_SUFFIX}",
            output_path.parent / f"{base}_mask.nii.gz",
            output_path.parent / f"{base}_bet_mask.nii.gz",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
