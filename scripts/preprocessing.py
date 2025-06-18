"""Refactored MRI Preprocessing Pipeline.

This script implements a configurable, sequential preprocessing pipeline for NIfTI
images, following a BIDS-like structure. It has been refactored using OOP
principles (SOLID, DRY, KISS) and the Strategy design pattern to enhance
modularity, maintainability, and extensibility.

Key Components:
- PreprocessingPipeline: The main orchestrator that finds files and manages the
  overall process.
- FileProcessor: The "Context" in the Strategy pattern. Manages the state and
  execution for a single input file.
- BaseProcessingStep: The abstract "Strategy" interface. Defines the common
  execution logic for all steps, including logging, error handling, and parameter
  saving. This adheres to the DRY principle.
- Concrete Step Classes (e.g., NormalizationStep, RegistrationStep): The
  "Concrete Strategies". Each class encapsulates the logic for one specific
  preprocessing tool (SimpleITK, ANTsPy, FSL BET). This adheres to the Single
  Responsibility Principle.
- Config, DependencyChecker, and utils: Helper classes and functions to
  encapsulate configuration loading, dependency checks, and other utilities.
"""
import os
import sys
import glob
import shutil
import json
import logging
import tempfile
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from abc import ABC, abstractmethod
import numpy as np
import yaml
import SimpleITK as sitk
from nipype.interfaces.fsl import BET, FSLCommand
from nipype.interfaces.base import Undefined
import ants

# --- Logger Setup ---
# A single logger instance is configured once and used throughout the application.
logger = logging.getLogger(__name__)

# --- Utility Functions & Classes ---

def setup_main_logging(log_file_path: Path, console_level: str = "INFO"):
    """Configures the main application logger."""
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s'
    )
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    try:
        ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    try:
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.critical(f"Could not set up file logger at {log_file_path}: {e}")

def save_parameters(params_dict: dict, output_path: Path):
    """Saves a dictionary of parameters to a JSON file, handling non-serializable types."""
    def make_serializable(obj):
        if isinstance(obj, Path):
            return str(obj.resolve())
        if isinstance(obj, (np.integer)):
            return int(obj)
        if isinstance(obj, (np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime, FSLCommand)):
             return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(params_dict, f, indent=4, ensure_ascii=False, default=make_serializable)
    except Exception as e:
        logger.error(f"Failed to save parameters to {output_path}: {e}", exc_info=True)


class DependencyChecker:
    """Encapsulates checks for external command-line tools."""
    def __init__(self):
        self._fsl_present = None
        self._freesurfer_present = None

    @property
    def is_fsl_present(self) -> bool:
        """Checks for FSL 'bet' command availability, caching the result."""
        if self._fsl_present is not None:
            return self._fsl_present
        try:
            FSLCommand.check_fsl()
            logger.info("FSL dependency check: OK (found via Nipype).")
            self._fsl_present = True
        except Exception:
            if shutil.which("bet"):
                logger.info("FSL dependency check: OK (found 'bet' in PATH).")
                self._fsl_present = True
            else:
                logger.error("FSL dependency check: FAILED. 'bet' command not found.")
                self._fsl_present = False
        return self._fsl_present
    
    @property
    def is_freesurfer_present(self) -> bool:
        """Checks for Freesurfer availability by checking FREESURFER_HOME."""
        if self._freesurfer_present is not None:
            return self._freesurfer_present

        if os.getenv("FREESURFER_HOME"):
            logger.info("Freesurfer dependency check: OK (FREESURFER_HOME is set).")
            self._freesurfer_present = True
        else:
            logger.error("Freesurfer dependency check: FAILED. FREESURFER_HOME environment variable is not set.")
            self._freesurfer_present = False
        return self._freesurfer_present


class Config:
    """Handles loading and providing access to the YAML configuration file."""
    def __init__(self, config_path: Path):
        self.config_path = config_path
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._data = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
        except Exception as e:
            logger.critical(f"Failed to load or parse config {config_path}: {e}")
            raise ValueError(f"Configuration error in {config_path}") from e

    def get_step_params(self, step_name: str) -> dict:
        """Gets parameters for a specific preprocessing step."""
        return self._data.get('steps', {}).get(step_name, {})

    @property
    def keep_intermediate_files(self) -> bool:
        """Returns whether to keep intermediate files after processing."""
        return self._data.get('steps', {}).get('preprocessing', {}).get('keep_intermediate_files', False)


# --- Strategy Pattern Implementation ---

class BaseProcessingStep(ABC):
    """
    Abstract Base Class for a preprocessing step (Strategy Interface).

    This class defines the template for executing a step, handling common logic
    such as logging, error handling, parameter saving, and file management.
    Concrete steps must implement the `_run_step` method.
    """
    def __init__(self, processor: 'FileProcessor', step_name: str, params: dict):
        self.processor = processor
        self.step_name = step_name
        self.params = params
        self.run_params = {**params}
        self.step_output_path = self.processor.temp_dir / f"{self.processor.file_stem}_{step_name}.nii.gz"
        self.params_json_path = self.processor.transform_dir / f"{step_name}_params.json"

    def execute(self) -> bool:
        """
        Executes the processing step using a template method pattern.

        Returns:
            bool: True if the step was successful or skipped, False on error.
        """
        is_enabled = str(self.params.get('enabled', 'true')).lower() in ['true', '1', 'yes']
        if not is_enabled:
            logger.info(f"  Step '{self.step_name}': SKIPPED (disabled in config).")
            self.processor.update_step_summary(self.step_name, self.run_params, status="skipped")
            save_parameters(self.processor.overall_summary, self.processor.summary_json_path)
            return True # Skipped is not a failure

        logger.info(f"  Running step: '{self.step_name}' for {self.processor.input_file.name}")
        try:
            self._run_step()
            logger.info(f"  Step '{self.step_name}': SUCCESS.")
            self.processor.update_step_summary(self.step_name, self.run_params, status="success")
            self.processor.current_input_path = self.step_output_path
            return True
        except Exception as e:
            logger.error(f"  Step '{self.step_name}': FAILED. Reason: {e}", exc_info=True)
            self.run_params["error"] = str(e)
            self.processor.update_step_summary(self.step_name, self.run_params, status="failed")
            return False
        finally:
            save_parameters(self.processor.overall_summary, self.processor.summary_json_path)

    @abstractmethod
    def _run_step(self):
        """
        The core logic for the specific processing step. To be implemented by subclasses.
        This method should perform the processing and save the result to `self.step_output_path`.
        """
        pass

class IntensityNormalizationStep(BaseProcessingStep):
    """Concrete step for histogram matching normalization."""
    def __init__(self, processor: 'FileProcessor', params: dict, dependency_checker: DependencyChecker):
        super().__init__(processor, "1_intensity_normalization", params)
        self.run_params["template_file"] = self.processor.template_path

    def _run_step(self):
        template_img = sitk.ReadImage(str(self.processor.template_path.resolve()), sitk.sitkFloat32)
        input_img = sitk.ReadImage(str(self.processor.current_input_path.resolve()), sitk.sitkFloat32)
        
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(self.params.get('histogram_levels', 1024))
        matcher.SetNumberOfMatchPoints(self.params.get('match_points', 7))
        
        normalized_img = matcher.Execute(input_img, template_img)
        sitk.WriteImage(normalized_img, str(self.step_output_path.resolve()))

class BiasFieldCorrectionStep(BaseProcessingStep):
    """Concrete step for N4 bias field correction."""

    def __init__(self, processor: 'FileProcessor', params: dict, dependency_checker: DependencyChecker):
        super().__init__(processor, "2_bias_field_correction", params)

    def _run_step(self):
        input_img = sitk.ReadImage(str(self.processor.current_input_path.resolve()), sitk.sitkFloat32)
        
        # Create a mask to focus correction on the head
        mask_img = sitk.OtsuThreshold(input_img, 0, 1)
        
        shrink_factor = self.params.get('sitk_shrinkFactor', 4)
        shrunk_img = sitk.Shrink(input_img, [shrink_factor] * input_img.GetDimension())
        shrunk_mask = sitk.Shrink(mask_img, [shrink_factor] * input_img.GetDimension())
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        n_iterations = self.params.get('sitk_numberOfIterations', [50] * shrink_factor)
        
        # Adjust iterations list to match shrink factor
        if len(n_iterations) != shrink_factor:
            n_iterations = (n_iterations * shrink_factor)[:shrink_factor]
            
        corrector.SetMaximumNumberOfIterations(n_iterations)
        
        if self.params.get('sitk_convergenceThreshold', 0.0) > 0:
            corrector.SetConvergenceThreshold(self.params.get('sitk_convergenceThreshold'))
            
        corrector.Execute(shrunk_img, shrunk_mask)

        log_bias_field = corrector.GetLogBiasFieldAsImage(input_img)
        corrected_image = input_img / sitk.Exp(log_bias_field)
        
        sitk.WriteImage(corrected_image, str(self.step_output_path.resolve()))

        # Save the bias field for inspection
        bias_field_path = self.processor.transform_dir / "bias_field.nii.gz"
        sitk.WriteImage(sitk.Exp(log_bias_field), str(bias_field_path))
        self.run_params["output_bias_field_path"] = bias_field_path


class RegistrationStep(BaseProcessingStep):
    """Concrete step for ANTsPy registration."""
    def __init__(self, processor: 'FileProcessor', params: dict, dependency_checker: DependencyChecker):
        super().__init__(processor, "3_registration", params)
        self.transform_prefix = str(self.processor.transform_dir / f"{self.processor.file_stem}_reg_")
        self.run_params.update({
            "template_file": self.processor.template_path,
            "output_prefix": self.transform_prefix
        })

    def _run_step(self):
        fixed_img = ants.image_read(str(self.processor.template_path.resolve()))
        moving_img = ants.image_read(str(self.processor.current_input_path.resolve()))

        transform_type = self.params.get('ants_transform_type', 'SyN')
        
        reg_result = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform=transform_type,
            outprefix=str(Path(self.transform_prefix).resolve()),
            verbose=False
        )
        
        ants.image_write(reg_result['warpedmovout'], str(self.step_output_path.resolve()))
        
        self.run_params["forward_transforms_paths"] = reg_result.get('fwdtransforms', [])
        self.run_params["inverse_transforms_paths"] = reg_result.get('invtransforms', [])

class SkullStrippingStep(BaseProcessingStep):
    """Concrete step for FSL BET skull stripping."""
    def __init__(self, processor: 'FileProcessor', params: dict, dependency_checker: DependencyChecker):
        super().__init__(processor, "4_skull_stripping", params)
        self.dependency_checker = dependency_checker
        # This step produces the final output, not an intermediate one.
        self.step_output_path = self.processor.final_output_path

    def _run_step(self):
        if not self.dependency_checker.is_fsl_present:
            raise RuntimeError("FSL 'bet' command is not available. This step cannot proceed.")
        
        # Use a temporary name for BET output before moving to final destination
        bet_temp_base = self.processor.temp_dir / f"{self.processor.file_stem}_bet_temp"

        bet = BET()
        bet.inputs.in_file = str(self.processor.current_input_path.resolve())
        bet.inputs.out_file = str(bet_temp_base.resolve())
        bet.inputs.frac = self.params.get('bet_fractional_intensity_threshold', 0.5)
        bet.inputs.robust = self.params.get('bet_robust', True)
        bet.inputs.mask = True
        bet.inputs.output_type = 'NIFTI_GZ'
        if self.params.get('bet_options'):
            bet.inputs.args = self.params.get('bet_options')
        
        logger.debug(f"    Executing FSL BET command: {bet.cmdline}")
        result = bet.run()

        # Layer 1: Check the command's exit code FIRST. This is the most reliable failure indicator.
        if result.runtime.returncode != 0:
            error_message = f"FSL BET command failed with a non-zero exit code ({result.runtime.returncode})."
            stdout = result.runtime.stdout or "(stdout not captured)"
            stderr = result.runtime.stderr or "(stderr not captured)"
            error_message += f"\n\n--- FSL BET STDOUT ---\n{stdout.strip()}"
            error_message += f"\n\n--- FSL BET STDERR ---\n{stderr.strip()}\n"
            raise RuntimeError(error_message)

        # Layer 2: If the command succeeded, verify the file's existence using the explicitly constructed path.
        expected_stripped_file = Path(f"{bet_temp_base.resolve()}.nii.gz")
        if not expected_stripped_file.exists():
            raise FileNotFoundError(
                f"FSL BET command reported success (exit code 0), but the expected output file was not found: "
                f"{expected_stripped_file}\n"
                "This could indicate a permissions issue or a problem with the output directory."
            )
        
        # Now that we've verified the file, we can proceed.
        expected_mask_file = Path(f"{bet_temp_base.resolve()}_mask.nii.gz")
        
        self.step_output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(expected_stripped_file), str(self.step_output_path))
        self.run_params["output_stripped_path"] = self.step_output_path
        
        if expected_mask_file.exists():
            final_mask_path = self.processor.transform_dir / f"{self.processor.file_stem}_brain_mask.nii.gz"
            shutil.move(str(expected_mask_file), str(final_mask_path))
            self.run_params["output_mask_path"] = final_mask_path
        else:
            logger.warning(f"  FSL BET did not produce a mask file at {expected_mask_file}")
            self.run_params["output_mask_path"] = None

class FreesurferSkullStrippingStep(BaseProcessingStep):
    """
    Performs skull stripping by running the full Freesurfer `recon-all -all`
    pipeline, which produces a high-quality result.
    NOTE: This is computationally intensive and can take several hours per subject.
    """
    def __init__(self, processor: 'FileProcessor', params: dict, dependency_checker: DependencyChecker):
        super().__init__(processor, "skull_stripping", params)
        self.dependency_checker = dependency_checker
        self.step_output_path = self.processor.final_output_path

    def _run_step(self):
        if not self.dependency_checker.is_freesurfer_present:
            raise RuntimeError("Freesurfer is not available. This step cannot proceed.")

        input_filename = self.processor.current_input_path.name
        
        # This check is still critical, as recon-all is T1-based.
        if "T1w" in input_filename and "ce" not in input_filename:
            logger.info(f"    Detected T1w image. Running full Freesurfer `recon-all -all` pipeline.")
            
            temp_fs_subjects_dir = self.processor.temp_dir / "fs_subjects"
            temp_fs_subjects_dir.mkdir()
            subject_id = self.processor.file_stem

            # --- START OF THE FIX ---
            # Build the command to run the full pipeline.
            command = [
                "recon-all",
                "-i", str(self.processor.current_input_path.resolve()),
                "-subjid", subject_id,
                "-sd", str(temp_fs_subjects_dir.resolve()),
                "-all"  # Use the full pipeline directive
            ]
            # --- END OF THE FIX ---

            logger.debug(f"    Executing Freesurfer command: {' '.join(command)}")

            result = subprocess.run(command, capture_output=True, text=True, check=False)

            if result.returncode != 0:
                error_message = f"Freesurfer recon-all command failed with exit code {result.returncode}."
                error_message += f"\n--- Freesurfer STDOUT ---\n{result.stdout.strip()}"
                error_message += f"\n\n--- Freesurfer STDERR ---\n{result.stderr.strip()}\n"
                raise RuntimeError(error_message)

            # After -all, the primary skull-stripped output is brain.mgz
            expected_stripped_mgz = temp_fs_subjects_dir / subject_id / "mri" / "brain.mgz"
            if not expected_stripped_mgz.exists():
                raise FileNotFoundError(f"Freesurfer ran successfully, but the primary output (brain.mgz) was not found at {expected_stripped_mgz}")

            logger.debug(f"    recon-all complete. Converting {expected_stripped_mgz} to NIfTI format.")
            stripped_image_sitk = sitk.ReadImage(str(expected_stripped_mgz.resolve()))
            sitk.WriteImage(stripped_image_sitk, str(self.step_output_path.resolve()))
            self.run_params["output_stripped_path"] = self.step_output_path
            
            # The brainmask.mgz file is the binary mask, which is useful for inspection.
            expected_mask_mgz = temp_fs_subjects_dir / subject_id / "mri" / "brainmask.mgz"
            if expected_mask_mgz.exists():
                final_mask_path = self.processor.transform_dir / f"{self.processor.file_stem}_brain_mask.nii.gz"
                mask_image_sitk = sitk.ReadImage(str(expected_mask_mgz.resolve()))
                sitk.WriteImage(mask_image_sitk, str(final_mask_path.resolve()))
                self.run_params["output_mask_path"] = final_mask_path
            else:
                logger.warning("Could not find the binary brain mask (brainmask.mgz) for inspection.")
        else:
            logger.info(f"    Skipping Freesurfer execution for non-T1w image: {input_filename}. Copying file to output.")
            shutil.copy(str(self.processor.current_input_path.resolve()), str(self.step_output_path.resolve()))

class SynthStripSkullStrippingStep(BaseProcessingStep):
    """Performs skull stripping using Freesurfer's mri_synthstrip."""
    def __init__(self, processor: 'FileProcessor', params: dict, dependency_checker: DependencyChecker):
        super().__init__(processor, "skull_stripping", params)
        self.dependency_checker = dependency_checker
        self.step_output_path = self.processor.final_output_path

    def _run_step(self):
        if not self.dependency_checker.is_freesurfer_present:
            raise RuntimeError("SynthStrip requires Freesurfer, which is not available.")

        # The output of this tool is the final stripped brain.
        # We point it to a temporary file before moving it.
        stripped_temp_path = self.processor.temp_dir / f"{self.processor.file_stem}_synth_stripped.nii.gz"
        mask_temp_path = self.processor.temp_dir / f"{self.processor.file_stem}_synth_mask.nii.gz"

        # 1. Build the command as a list of arguments
        command = [
            "mri_synthstrip",
            "-i", str(self.processor.current_input_path.resolve()),
            "-o", str(stripped_temp_path.resolve())
        ]
        
        # 2. Add optional mask output from the config file
        save_mask = self.params.get('save_mask', False)
        if save_mask:
            command.extend(["-m", str(mask_temp_path.resolve())])
        
        # 3. Add the border parameter if it's specified in the config
        if 'border' in self.params:
            border_val = self.params['border']
            logger.info(f"    Using custom border of {border_val} voxels for SynthStrip.")
            command.extend(["-b", str(border_val)])

        # 4. Add other optional parameters from the config file
        if 'gpu_device' in self.params:
            logger.info(f"    Using GPU device {self.params['gpu_device']} for SynthStrip.")
            command.extend(["-g", "-d", str(self.params['gpu_device'])])

        logger.debug(f"    Executing SynthStrip command: {' '.join(command)}")

        # 5. Run the command using subprocess
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        # 6. Robust error checking
        if result.returncode != 0:
            error_message = f"mri_synthstrip command failed with exit code {result.returncode}."
            error_message += f"\n--- SynthStrip STDOUT ---\n{result.stdout.strip()}"
            error_message += f"\n\n--- SynthStrip STDERR ---\n{result.stderr.strip()}\n"
            raise RuntimeError(error_message)

        # 7. Verify and move the primary output
        if not stripped_temp_path.exists():
            raise FileNotFoundError(f"SynthStrip ran successfully, but the output file was not found at {stripped_temp_path}")
        
        shutil.move(str(stripped_temp_path), str(self.step_output_path.resolve()))
        self.run_params["output_stripped_path"] = self.step_output_path
        
        # 8. Move the mask if it was created
        if save_mask:
            if mask_temp_path.exists():
                final_mask_path = self.processor.transform_dir / f"{self.processor.file_stem}_brain_mask.nii.gz"
                shutil.move(str(mask_temp_path), str(final_mask_path.resolve()))
                self.run_params["output_mask_path"] = final_mask_path
            else:
                logger.warning(f"SynthStrip was asked to save a mask, but it was not found at {mask_temp_path}")
                self.run_params["output_mask_path"] = None
        else:
            self.run_params["output_mask_path"] = None

class StepFactory:
    """A factory for creating preprocessing step objects based on a method name."""
    def __init__(self):
        self._registry = {}

    def register_step(self, method_name: str, step_class: type):
        """Registers a step class with a corresponding method name from the config."""
        logger.debug(f"Registering method '{method_name}' with class {step_class.__name__}")
        self._registry[method_name] = step_class

    def create_step(self, step_name: str, processor: 'FileProcessor'):
        """Creates a step instance based on the configuration."""
        step_config = processor.preprocessing_config.get(step_name)
        if not step_config:
            return None  # Step not defined in the config for this run

        method = step_config.get("method")
        if not method:
            raise ValueError(f"Configuration for step '{step_name}' is missing the required 'method' key.")

        step_class = self._registry.get(method)
        if not step_class:
            raise ValueError(f"Unknown method '{method}' for step '{step_name}'. Is it registered in the factory?")
        
        # The dependency_checker is passed to all steps for consistency.
        return step_class(processor, step_config, processor.dependency_checker)
    
# ===================================================================
# STEP FACTORY REGISTRATION
# -------------------------------------------------------------------
# This is the central registry for all available preprocessing tools.
# To add a new tool (e.g., for skull stripping):
# 1. Create a new class inheriting from BaseProcessingStep.
# 2. Register it here with the 'method' name from the config.yaml.
# ===================================================================

step_factory = StepFactory()
step_factory.register_step("HistogramMatching", IntensityNormalizationStep)
step_factory.register_step("N4BiasFieldCorrection", BiasFieldCorrectionStep)
step_factory.register_step("ANTsPy", RegistrationStep)
step_factory.register_step("FSL_BET", SkullStrippingStep)
step_factory.register_step("Freesurfer", FreesurferSkullStrippingStep)
step_factory.register_step("SynthStrip", SynthStripSkullStrippingStep)

# ===================================================================

# --- Context and Orchestrator Classes ---

class FileProcessor:
    """
    Manages the entire preprocessing workflow for a single NIfTI file (Context).
    """
    def __init__(self, input_file: Path, config: Config, pipeline_paths: dict, checker: DependencyChecker):
        self.input_file = input_file
        self.config = config
        self.paths = pipeline_paths
        self.dependency_checker = checker
        self.template_path = self.paths['template_path']
        self.preprocessing_config = self.config.get_step_params("preprocessing")
        
        self.file_stem = self.input_file.name.replace(".nii.gz", "").replace(".nii", "")
        self.current_input_path = self.input_file
        
        # Setup paths
        relative_path = self.input_file.relative_to(self.paths['input_root'])
        self.final_output_path = self.paths['output_prep_root'] / relative_path
        self.transform_dir = self.paths['output_tfm_root'] / relative_path.parent / self.file_stem
        self.summary_json_path = self.transform_dir / f"{self.file_stem}_processing_summary.json"
        
        self.temp_dir = None # To be created in run()
        self.overall_summary = {
            "input_file": self.input_file,
            "processing_timestamp": datetime.now().isoformat(),
            "steps_parameters": {}
        }

    def update_step_summary(self, step_name: str, params: dict, status: str):
        """Updates the summary dictionary for the current file."""
        self.overall_summary["steps_parameters"][step_name] = {"status": status, **params}
        
    def run(self) -> bool:
        """
        Executes the full pipeline of steps for the file.

        Returns:
            bool: True if all steps succeeded, False otherwise.
        """
        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.transform_dir.mkdir(parents=True, exist_ok=True)

        log_file_path = self.transform_dir / f"{self.file_stem}_log.txt"
        file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        file_handler.setFormatter(logger.handlers[0].formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

        all_steps_succeeded = False
        try:
            with tempfile.TemporaryDirectory(prefix=f"{self.file_stem}_", dir=self.transform_dir) as temp_dir_str:
                self.temp_dir = Path(temp_dir_str)
                logger.info(f"Processing in temporary directory: {self.temp_dir}")
                
                steps_to_run = []
                # Define the required order of execution
                step_order = [
                    "intensity_normalization",
                    "bias_field_correction",
                    "registration",
                    "skull_stripping"
                ]

                for step_name in step_order:
                    # Ask the factory to create the correct step object
                    step = step_factory.create_step(step_name, self)
                    if step:
                        steps_to_run.append(step)
                
                logger.info(f"Pipeline for {self.file_stem} will run {len(steps_to_run)} steps.")

                for step in steps_to_run:
                    if not step.execute():
                        raise RuntimeError(f"Pipeline stopped due to failure in step '{step.step_name}'.")

                all_steps_succeeded = True
                self.overall_summary["processing_status"] = "success"
        
        except Exception as e:
            logger.error(f"--- FAILED processing file {self.input_file.name}. Reason: {e}")
            self.overall_summary["processing_status"] = "failed"
            self.overall_summary["error_message"] = str(e)

        finally:
            # Finalize and save summary
            save_parameters(self.overall_summary, self.summary_json_path)
            
            # Clean up intermediate files if configured to do so
            if self.temp_dir and self.temp_dir.exists() and not self.config.keep_intermediate_files and all_steps_succeeded:
                logger.debug(f"Removing temporary directory: {self.temp_dir}")
                # The 'with' statement handles this automatically.
                pass
            elif self.temp_dir:
                 logger.info(f"Intermediate files kept at: {self.temp_dir}")
            
            logger.removeHandler(file_handler)
            file_handler.close()

        return all_steps_succeeded


class PreprocessingPipeline:
    """
    Main orchestrator for the entire preprocessing workflow.
    """
    def __init__(self, input_root: Path, output_prep_root: Path, output_tfm_root: Path, template_path: Path, config_path: Path):
        self.paths = {
            "input_root": input_root,
            "output_prep_root": output_prep_root,
            "output_tfm_root": output_tfm_root,
            "template_path": template_path,
        }
        self.config = Config(config_path)
        self.dependency_checker = DependencyChecker()

    def _find_nifti_files(self) -> list[Path]:
        """Finds all NIfTI files in the input directory based on BIDS-like patterns."""
        patterns = ['sub-*/ses-*/anat/*.nii.gz', 'sub-*/ses-*/anat/*.nii',
                    'sub-*/anat/*.nii.gz', 'sub-*/anat/*.nii']
        found_files = set()
        for pattern in patterns:
            found_files.update(self.paths['input_root'].glob(pattern))
        
        if not found_files:
            logger.warning(f"No NIfTI files found in {self.paths['input_root']} matching BIDS patterns.")
        else:
            logger.info(f"Found {len(found_files)} NIfTI files to process.")
        return sorted(list(found_files))

    def run(self):
        """Runs the entire preprocessing pipeline for all found files."""
        logger.info("=" * 50)
        logger.info("Starting Preprocessing Pipeline")
        for key, path in self.paths.items():
            logger.info(f"  {key.replace('_', ' ').title()}: {path.resolve()}")
        logger.info(f"  Keep Intermediate Files: {self.config.keep_intermediate_files}")
        logger.info("=" * 50)
        
        nifti_files = self._find_nifti_files()
        if not nifti_files:
            return

        success_count, error_count = 0, 0
        for nifti_file in nifti_files:
            logger.info(f"--- Processing file: {nifti_file.relative_to(self.paths['input_root'])} ---")
            processor = FileProcessor(nifti_file, self.config, self.paths, self.dependency_checker)
            if processor.run():
                success_count += 1
                logger.info(f"--- SUCCESS: Finished processing {nifti_file.name} ---")
            else:
                error_count += 1
                logger.error(f"--- FAILED: Finished processing {nifti_file.name} with errors ---")
        
        logger.info("=" * 50)
        logger.info("Preprocessing Pipeline Finished")
        logger.info(f"  Total files found: {len(nifti_files)}")
        logger.info(f"  Successfully processed: {success_count}")
        logger.info(f"  Failed: {error_count}")
        logger.info("=" * 50)


# --- Main Execution Block ---
def main():
    """Parses arguments and initiates the pipeline."""
    parser = argparse.ArgumentParser(
        description="OOP-Refactored NIfTI MRI Preprocessing Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", required=True, type=Path, help="Input BIDS-like NIfTI directory.")
    parser.add_argument("--output_prep_dir", required=True, type=Path, help="Output directory for preprocessed files.")
    parser.add_argument("--output_transform_dir", required=True, type=Path, help="Output directory for transforms and logs.")
    parser.add_argument("--template_path", required=True, type=Path, help="Path to the MRI template file.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the YAML pipeline configuration file.")
    parser.add_argument("--log_file", default=None, type=Path, help="Path to main log file. Defaults to 'main_log.txt' in transform dir.")
    parser.add_argument("--console_log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Console logging level.")
    args = parser.parse_args()

    log_path = args.log_file or args.output_transform_dir / "preprocessing_main.log"
    setup_main_logging(log_path, args.console_log_level)

    try:
        pipeline = PreprocessingPipeline(
            input_root=args.input_dir,
            output_prep_root=args.output_prep_dir,
            output_tfm_root=args.output_transform_dir,
            template_path=args.template_path,
            config_path=args.config
        )
        pipeline.run()
        sys.exit(0)
    except (FileNotFoundError, ValueError, KeyError) as e:
        logger.critical(f"A critical configuration or file error occurred: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()