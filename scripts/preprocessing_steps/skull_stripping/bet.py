"""
FSL BET skull stripper.

The module-level FSL setup and run_bet() below are carried over verbatim
from the original scripts/preprocessing_steps/skull_stripping.py so BET
behaves exactly as before the plugin split. Only the class wrapper at the
bottom is new.
"""

import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import nibabel as nib
import numpy as np
import os

logger = logging.getLogger(__name__)

# Global FSL configuration
FSL_DIR = None
FSL_BIN_DIR = None


def setup_fsl_environment(fsl_dir: str = None):
    """
    Setup FSL environment variables.
    
    Args:
        fsl_dir: Path to FSL installation directory (e.g., /usr/local/fsl or /path/to/fsl/share/fsl)
                 If None, assumes FSL is in PATH
    """
    global FSL_DIR, FSL_BIN_DIR
    
    if fsl_dir and fsl_dir.strip():
        fsl_path = Path(fsl_dir)
        
        # If path points to a file (like bet or fsl executable), get parent directory
        if fsl_path.is_file():
            fsl_path = fsl_path.parent
        
        # Now fsl_path should be the bin directory, so go up one level to get FSL_DIR
        if fsl_path.name == "bin":
            FSL_DIR = fsl_path.parent
            FSL_BIN_DIR = fsl_path
        else:
            # Assume it's the root FSL directory
            FSL_DIR = fsl_path
            FSL_BIN_DIR = FSL_DIR / "bin"
        
        # Verify bin directory exists
        if not FSL_BIN_DIR.exists():
            logger.warning(f"FSL bin directory not found: {FSL_BIN_DIR} — falling back to system PATH")
            FSL_DIR = None
            FSL_BIN_DIR = None
            return

        # Verify bet exists
        bet_path = FSL_BIN_DIR / "bet"
        if not bet_path.exists():
            logger.warning(f"BET executable not found: {bet_path} — falling back to system PATH")
            FSL_DIR = None
            FSL_BIN_DIR = None
            return
        
        logger.info(f"✓ FSL configured successfully")
        logger.info(f"  FSLDIR: {FSL_DIR}")
        logger.info(f"  BIN: {FSL_BIN_DIR}")
        logger.info(f"  BET: {bet_path}")
    else:
        FSL_DIR = None
        FSL_BIN_DIR = None
        logger.info("Using FSL from system PATH")

def get_fsl_env() -> dict:
    """
    Get environment variables for FSL commands.
    
    Returns:
        dict: Environment variables
    """
    env = os.environ.copy()
    
    if FSL_DIR is not None:
        env['FSLDIR'] = str(FSL_DIR)
        env['FSLOUTPUTTYPE'] = 'NIFTI_GZ'
        
        # Add FSL bin to PATH
        if FSL_BIN_DIR is not None:
            current_path = env.get('PATH', '')
            env['PATH'] = f"{FSL_BIN_DIR}:{current_path}"
        
        # Add FSL lib to LD_LIBRARY_PATH if exists
        fsl_lib = FSL_DIR / "lib"
        if fsl_lib.exists():
            current_ld_path = env.get('LD_LIBRARY_PATH', '')
            env['LD_LIBRARY_PATH'] = f"{fsl_lib}:{current_ld_path}"
    
    return env


def get_bet_command() -> str:
    """
    Get the BET command path.
    
    Returns:
        str: Path to BET executable
    """
    if FSL_BIN_DIR is not None:
        return str(FSL_BIN_DIR / "bet")
    else:
        return "bet"


def check_fsl_installed() -> bool:
    """
    Check if FSL is installed and available.
    
    Returns:
        bool: True if FSL is available
    """
    try:
        bet_cmd = get_bet_command()
        env = get_fsl_env()
        
        logger.debug(f"Testing BET command: {bet_cmd}")
        logger.debug(f"Environment FSLDIR: {env.get('FSLDIR', 'not set')}")
        logger.debug(f"Environment PATH: {env.get('PATH', 'not set')[:200]}...")
        
        result = subprocess.run(
            [bet_cmd, '-help'],
            capture_output=True,
            timeout=5,
            env=env,
            text=True
        )
        
        logger.debug(f"BET return code: {result.returncode}")
        logger.debug(f"BET stdout: {result.stdout[:200] if result.stdout else 'empty'}")
        logger.debug(f"BET stderr: {result.stderr[:200] if result.stderr else 'empty'}")
        
        # BET returns code 1 for -help, but outputs usage info
        # Check if output contains expected BET usage text
        output = result.stdout + result.stderr
        is_valid = "Usage:" in output and "bet" in output.lower()
        
        if is_valid:
            logger.debug("✓ BET is working correctly")
        else:
            logger.debug("✗ BET output doesn't contain expected usage text")
        
        return is_valid
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"FSL check timeout: {e}")
        return False
    except FileNotFoundError as e:
        logger.error(f"FSL BET executable not found: {e}")
        return False
    except Exception as e:
        logger.error(f"FSL check failed with exception: {e}")
        return False


def run_bet(
    input_path: Path,
    output_path: Path,
    mask_path: Path = None,
    fractional_intensity: float = 0.5,
    vertical_gradient: float = 0.0,
    generate_mask: bool = True
) -> dict:
    """
    Run FSL BET for brain extraction.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save skull-stripped image
        mask_path: Path to save brain mask (optional)
        fractional_intensity: Fractional intensity threshold (0-1), default 0.5
        vertical_gradient: Vertical gradient in fractional intensity threshold (-1 to 1)
        generate_mask: Whether to generate brain mask
    
    Returns:
        dict: Information about the skull stripping
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"Running BET on {input_path.name}")
        
        # Check if FSL is installed
        if not check_fsl_installed():
            raise RuntimeError(
                f"FSL BET not found. "
                f"FSL_DIR: {FSL_DIR}, FSL_BIN_DIR: {FSL_BIN_DIR}. "
                f"Please check FSL installation path in config."
            )
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get BET command and environment
        bet_cmd = get_bet_command()
        env = get_fsl_env()

        # KI-049: `bet` is a shell wrapper that splits its arguments on
        # spaces — under ".../Рабочий стол/..." it failed with "input image
        # not valid". It runs in a private temp dir (system temp, no spaces)
        # on a symlink to the input; results are moved to the caller's paths.
        work = Path(tempfile.mkdtemp(prefix="bet_"))
        try:
            work_in = work / "in.nii.gz"
            work_in.symlink_to(Path(input_path).resolve())
            work_out = work / "out"

            # Build BET command
            cmd = [
                bet_cmd,
                str(work_in),
                str(work_out),
                '-f', str(fractional_intensity),
                '-g', str(vertical_gradient)
            ]

            # Add mask generation flag
            if generate_mask:
                cmd.append('-m')

            # Add robust brain center estimation
            cmd.append('-R')

            logger.debug(f"BET command: {' '.join(cmd)}")
            logger.debug(f"FSLDIR: {env.get('FSLDIR', 'not set')}")

            # Run BET with FSL environment
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
                env=env
            )

            if result.returncode != 0:
                # BET prints its own error to stdout and leaves stderr empty;
                # stderr alone gave "return code 1: " and nothing else.
                said = " ".join(
                    part.strip() for part in (result.stdout, result.stderr) if part and part.strip()
                )
                raise RuntimeError(
                    f"BET failed with return code {result.returncode}: {said[:500]}"
                )

            logger.debug(f"BET output: {result.stdout}")

            produced = work / "out.nii.gz"
            produced_mask = work / "out_mask.nii.gz"
            if not produced.exists():
                raise RuntimeError(
                    f"BET exited 0 but did not write the stripped image for {input_path.name}"
                )
            if generate_mask and not produced_mask.exists():
                raise RuntimeError(
                    f"BET exited 0 but did not write the brain mask for {input_path.name}"
                )

            shutil.move(str(produced), str(output_path))
            mask_created = False
            if generate_mask:
                if mask_path is None:
                    base = output_path.name
                    for ext in (".nii.gz", ".nii"):
                        if base.endswith(ext):
                            base = base[: -len(ext)]
                            break
                    mask_path = output_path.parent / f"{base}_mask.nii.gz"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(produced_mask), str(mask_path))
                logger.info(f"Saved brain mask to {mask_path}")
                mask_created = True
        finally:
            shutil.rmtree(work, ignore_errors=True)
        
        processing_time = time.time() - start_time
        logger.info(f"BET completed in {processing_time:.2f} seconds")
        
        return {
            "success": True,
            "output_path": str(output_path),
            "mask_path": str(mask_path) if mask_created else None,
            "processing_time": processing_time
        }
        
    except subprocess.TimeoutExpired:
        logger.error(f"BET timeout on {input_path.name}")
        return {
            "success": False,
            "error": "BET timeout (exceeded 5 minutes)"
        }
    except Exception as e:
        logger.error(f"Error running BET on {input_path.name}: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }


# ---------------------------------------------------------------------------
# Plugin wrapper
# ---------------------------------------------------------------------------

from .base import SkullStripperBase  # noqa: E402  (kept below the verbatim block)


class BetStripper(SkullStripperBase):
    """FSL BET — the historical default, classical algorithm, CPU-only."""

    name = "bet"

    def is_available(self) -> bool:
        return check_fsl_installed()

    def strip(self, input_path, output_path, mask_path=None, params=None):
        params = params or {}
        return run_bet(
            input_path=input_path,
            output_path=output_path,
            mask_path=mask_path,
            # Same defaults the pipeline has always passed through.
            fractional_intensity=params.get("fractional_intensity", 0.5),
            vertical_gradient=params.get("vertical_gradient", 0.0),
            generate_mask=True,
        )
