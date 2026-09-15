"""
deepbet wrapper (Fisch et al. 2023) via the ``deepbet-cli`` command.

A small 3D U-Net trained on ~2500 T1 scans. It is the cheapest tool in the
cascade by a wide margin — measured at 2.5 s per volume on CPU here, against
4 s for SynthStrip and 15 s for HD-BET on a GPU — because the network is tiny
and its weights ship inside the wheel, so there is no download and nothing to
mount (contrast HD-BET, which hardcodes ~/hd-bet_params).

Invoked as a subprocess like the other wrappers, so Stage 05 never imports a
torch model in-process.
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

_TIMEOUT_SEC = 600


def _tail(text: str, limit: int = 1500) -> str:
    """Keep the END of a failing tool's output, not the start.

    A Python traceback states the actual cause on its last line; truncating
    from the front threw exactly that away and left only the import chain —
    which is how an HD-BET failure across a whole batch stayed unexplained.
    """
    text = text.strip()
    if len(text) <= limit:
        return text
    return "...(truncated)... " + text[-limit:]


class DeepBetStripper(SkullStripperBase):
    """deepbet — fast CPU brain extraction with bundled weights."""

    name = "deepbet"
    # CPU is the normal mode here: a 2.5 s job has nothing to gain from a
    # card, and taking a GPU pool slot would stall HD-BET, which does.
    uses_gpu = False

    def is_available(self) -> bool:
        return shutil.which("deepbet-cli") is not None

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

        def option(key):
            return params.get(key, nested.get(key))

        cmd = [
            "deepbet-cli",
            "-i", str(input_path),
            "-o", str(output_path),
        ]
        if mask_path is not None:
            # Despite the help text saying "Mask folder", a file path here
            # produces a file — verified on a real volume before wiring it up.
            cmd += ["-m", str(mask_path)]

        threshold = option("threshold")
        if threshold is not None:
            cmd += ["-t", str(threshold)]
        n_dilate = option("n_dilate")
        if n_dilate is not None:
            cmd += ["-d", str(n_dilate)]

        # Careful: `-g` is the short form of `--no_gpu`, i.e. it *disables*
        # the GPU. Reading it as "use gpu" would put this tool on the card
        # and starve HD-BET. Default is CPU; `use_gpu: true` opts in.
        if not option("use_gpu"):
            cmd.append("-g")
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
                    "deepbet-cli not found on PATH. Install with "
                    "`pip install deepbet` (see services/skull-stripping/deepbet)."
                )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if mask_path is not None:
                mask_path.parent.mkdir(parents=True, exist_ok=True)

            cmd = self.build_command(input_path, output_path, mask_path, params)
            logger.info("Running deepbet on %s", input_path.name)
            logger.debug("deepbet command: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_TIMEOUT_SEC,
                env=os.environ.copy(),
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"deepbet failed with return code {result.returncode}: "
                    f"{_tail(result.stderr or result.stdout or '')}"
                )
            if not output_path.exists():
                raise RuntimeError(
                    f"deepbet reported success but produced no output at {output_path}"
                )

            mask_created = mask_path is not None and mask_path.exists()
            if mask_path is not None and not mask_created:
                logger.warning("deepbet produced no mask at %s", mask_path)

            processing_time = time.time() - start_time
            logger.info("deepbet completed in %.2f seconds", processing_time)
            return {
                "success": True,
                "output_path": str(output_path),
                "mask_path": str(mask_path) if mask_created else None,
                "processing_time": processing_time,
            }
        except subprocess.TimeoutExpired:
            logger.error("deepbet timeout on %s", input_path.name)
            return {
                "success": False,
                "error": f"deepbet timeout (exceeded {_TIMEOUT_SEC // 60} minutes)",
            }
        except Exception as e:
            logger.error("Error running deepbet on %s: %s", input_path.name, e)
            return {"success": False, "error": str(e)}
