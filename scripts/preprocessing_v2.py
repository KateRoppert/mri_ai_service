"""Refactored MRI Preprocessing Pipeline with Modular Steps - Version 3.

This version fixes template path handling and configuration parsing.
"""
import os
import sys
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
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import standalone module interfaces
try:
    from intensity_normalization import IntensityNormalizationStep
    from bias_field_correction import BiasFieldCorrectionStep  
    from registration import RegistrationStep
    from skull_stripping import SkullStrippingStep
    STANDALONE_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Some standalone modules not available: {e}")
    STANDALONE_MODULES_AVAILABLE = False

# --- Logger Setup ---
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
        if isinstance(obj, datetime):
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
        self._ants_present = None

    @property
    def is_fsl_present(self) -> bool:
        """Checks for FSL 'bet' command availability, caching the result."""
        if self._fsl_present is not None:
            return self._fsl_present
        
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
    
    @property
    def is_ants_present(self) -> bool:
        """Checks for ANTs availability."""
        if self._ants_present is not None:
            return self._ants_present
        
        try:
            import ants
            self._ants_present = True
            logger.info("ANTs dependency check: OK (antspyx available).")
        except ImportError:
            self._ants_present = False
            logger.error("ANTs dependency check: FAILED. antspyx not installed.")
        
        return self._ants_present


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
        """Gets parameters for a specific preprocessing step from steps.preprocessing section."""
        preprocessing_config = self._data.get('steps', {}).get('preprocessing', {})
        return preprocessing_config.get(step_name, {})
    
    def get_preprocessing_config(self) -> dict:
        """Gets the entire preprocessing configuration."""
        return self._data.get('steps', {}).get('preprocessing', {})

    @property
    def keep_intermediate_files(self) -> bool:
        """Returns whether to keep intermediate files after processing."""
        return self._data.get('steps', {}).get('preprocessing', {}).get('keep_intermediate_files', False)
    
    def get_enabled_steps(self) -> List[str]:
        """Returns names of all enabled steps in correct order"""
        step_order = ["intensity_normalization", "bias_field_correction", 
                     "registration", "skull_stripping"]
        preprocessing_config = self.get_preprocessing_config()
        return [
            step_name for step_name in step_order
            if self.is_step_enabled(step_name)
        ]
    
    def is_step_enabled(self, step_name: str) -> bool:
        """Check if a preprocessing step is enabled."""
        params = self.get_step_params(step_name)
        return str(params.get('enabled', 'true')).lower() in ['true', '1', 'yes']
    
    def get_template_path(self) -> Optional[Path]:
        """Get template path from config if available."""
        # Try paths.template_path first
        if 'paths' in self._data and 'template_path' in self._data['paths']:
            return Path(self._data['paths']['template_path'])
        
        # Try registration section
        reg_config = self.get_step_params('registration')
        if 'template_path' in reg_config:
            return Path(reg_config['template_path'])
        
        return None


# --- Base Processing Step ---

class BaseProcessingStep(ABC):
    """
    Abstract Base Class for a preprocessing step.
    This is used for steps that are NOT using standalone modules.
    """
    def __init__(self, processor: 'FileProcessor', step_name: str, params: dict):
        self.processor = processor
        self.step_name = step_name
        self.params = params
        self.run_params = {**params}
        self.step_output_path = self.processor.temp_dir / f"{self.processor.file_stem}_{step_name}.nii.gz"
        self.params_json_path = self.processor.transform_dir / f"{step_name}_params.json"

    def execute(self) -> bool:
        """Executes the processing step using a template method pattern."""
        is_enabled = str(self.params.get('enabled', 'true')).lower() in ['true', '1', 'yes']
        if not is_enabled:
            logger.info(f"  Step '{self.step_name}': SKIPPED (disabled in config).")
            self.processor.update_step_summary(self.step_name, self.run_params, status="skipped")
            save_parameters(self.processor.overall_summary, self.processor.summary_json_path)
            return True

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
        """The core logic for the specific processing step."""
        pass


class StepFactory:
    """A factory for creating preprocessing step objects."""
    def __init__(self):
        self._registry = {}

    def register_step(self, method_name: str, step_class: type):
        """Registers a step class with a corresponding method name from the config."""
        logger.debug(f"Registering method '{method_name}' with class {step_class.__name__}")
        self._registry[method_name] = step_class

    def create_step(self, step_name: str, processor: 'FileProcessor'):
        """Creates a step instance based on the configuration."""
        step_config = processor.config.get_step_params(step_name)
        
        if not step_config:
            return None  # Step not defined in the config

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
# Register all available preprocessing steps here.
# The standalone modules provide their own step classes.
# ===================================================================

step_factory = StepFactory()

# Register standalone module steps if available
if STANDALONE_MODULES_AVAILABLE:
    # Intensity normalization methods
    step_factory.register_step("HistogramMatching", IntensityNormalizationStep)
    step_factory.register_step("ZScore", IntensityNormalizationStep)
    step_factory.register_step("WhiteStripe", IntensityNormalizationStep)
    
    # Bias field correction methods
    step_factory.register_step("N4BiasFieldCorrection", BiasFieldCorrectionStep)
    step_factory.register_step("Standard", BiasFieldCorrectionStep)
    step_factory.register_step("ModalitySpecific", BiasFieldCorrectionStep)
    step_factory.register_step("T1Based", BiasFieldCorrectionStep)
    
    # Registration methods
    step_factory.register_step("ANTsPy", RegistrationStep)
    step_factory.register_step("ANTs", RegistrationStep)
    
    # Skull stripping methods
    step_factory.register_step("FSL_BET", SkullStrippingStep)
    step_factory.register_step("FreeSurfer", SkullStrippingStep)
    step_factory.register_step("SynthStrip", SkullStrippingStep)
else:
    logger.warning("Standalone modules not available. Pipeline functionality will be limited.")

# ===================================================================

# --- Context and Orchestrator Classes ---

class FileProcessor:
    """Manages the entire preprocessing workflow for a single NIfTI file."""
    def __init__(self, input_file: Path, config: Config, pipeline_paths: dict, checker: DependencyChecker):
        self.input_file = input_file
        self.config = config
        self.paths = pipeline_paths
        self.dependency_checker = checker
        self.template_path = self.paths.get('template_path')
        
        # Extract subject and session IDs from filename
        self._extract_ids_from_filename()
        
        self.file_stem = self.input_file.name.replace(".nii.gz", "").replace(".nii", "")
        self.current_input_path = self.input_file

        self.paths['output_prep_root'].mkdir(parents=True, exist_ok=True)
        
        # Setup paths
        relative_path = self.input_file.relative_to(self.paths['input_root']).parent
        self.final_output_dir = self.paths['output_prep_root'] / relative_path
        self.final_output_path = self.final_output_dir / self.input_file.name
        
        self.transform_dir = self.paths['output_tfm_root'] / relative_path / self.file_stem
        self.summary_json_path = self.transform_dir / f"{self.file_stem}_processing_summary.json"
        
        self.temp_dir = None
        self.overall_summary = {
            "input_file": self.input_file,
            "processing_timestamp": datetime.now().isoformat(),
            "steps_parameters": {}
        }
        
        # For BIDS compatibility
        self.bids_root = self.paths['input_root']
    
    def _extract_ids_from_filename(self):
        """Extract subject and session IDs from BIDS-style filename."""
        parts = self.input_file.parts
        self.subject_id = None
        self.session_id = None
        
        for part in parts:
            if part.startswith('sub-'):
                self.subject_id = part[4:]  # Remove 'sub-' prefix
            elif part.startswith('ses-'):
                self.session_id = part[4:]  # Remove 'ses-' prefix
        
        # If not found in path, try filename
        if not self.subject_id:
            filename = self.input_file.name
            if 'sub-' in filename:
                sub_start = filename.index('sub-') + 4
                sub_end = filename.index('_', sub_start) if '_' in filename[sub_start:] else len(filename)
                self.subject_id = filename[sub_start:sub_end]

    def update_step_summary(self, step_name: str, params: dict, status: str):
        """Updates the summary dictionary for the current file."""
        self.overall_summary["steps_parameters"][step_name] = {"status": status, **params}
        
    def run(self) -> bool:
        """Executes the pipeline for one file with proper result saving"""
        try:
            # Ensure directories exist
            self.final_output_dir.mkdir(parents=True, exist_ok=True)
            self.transform_dir.mkdir(parents=True, exist_ok=True)

            # Setup logging
            log_file_path = self.transform_dir / f"{self.file_stem}_log.txt"
            file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
            file_handler.setFormatter(logger.handlers[0].formatter)
            logger.addHandler(file_handler)

            all_steps_succeeded = False
            
            with tempfile.TemporaryDirectory(prefix=f"{self.file_stem}_", dir=self.transform_dir) as temp_dir_str:
                self.temp_dir = Path(temp_dir_str)
                
                # Get enabled steps in correct order
                step_order = ["intensity_normalization", "bias_field_correction", 
                            "registration", "skull_stripping"]
                enabled_steps = []
                
                for step_name in step_order:
                    step = step_factory.create_step(step_name, self)
                    if step and step.params.get('enabled', True):
                        enabled_steps.append(step)
                
                logger.info(f"Will execute {len(enabled_steps)} steps")
                
                # Execute all enabled steps
                for step in enabled_steps:
                    if hasattr(step, 'execute'):
                        # Old-style step (BaseProcessingStep)
                        if not step.execute():
                            raise RuntimeError(f"Step {step.step_name} failed")
                    elif hasattr(step, 'run_step'):
                        # New-style step (standalone module)
                        try:
                            step.run_step()
                            self.update_step_summary(step.step_name, step.params, status="success")
                        except Exception as e:
                            self.update_step_summary(step.step_name, step.params, status="failed")
                            raise RuntimeError(f"Step {step.step_name} failed: {e}")
                    else:
                        raise RuntimeError(f"Step {step} does not have execute() or run_step() method")
                
                # Save final result if at least one step was executed
                if enabled_steps:
                    # If skull stripping was the last step, it already saved to final location
                    if not any(hasattr(step, 'step_name') and step.step_name == '4_skull_stripping' 
                             for step in enabled_steps[-1:]):
                        final_path = self.final_output_dir / self.input_file.name
                        final_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(self.current_input_path), str(final_path))
                        logger.info(f"Final result saved to {final_path}")
                
                all_steps_succeeded = True
                
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}", exc_info=True)
            self.overall_summary["processing_status"] = "failed"
            self.overall_summary["error_message"] = str(e)
        finally:
            # Save processing summary
            save_parameters(self.overall_summary, self.summary_json_path)
            
            # Clean up
            if 'file_handler' in locals():
                logger.removeHandler(file_handler)
                file_handler.close()
                
            return all_steps_succeeded


class PreprocessingPipeline:
    """Main orchestrator for the entire preprocessing workflow."""
    def __init__(self, input_root: Path, output_prep_root: Path, output_tfm_root: Path, 
                 template_path: Optional[Path], config_path: Path):
        self.config = Config(config_path)
        
        # Handle template path - command line takes precedence over config
        if template_path:
            final_template_path = template_path
        else:
            # Try to get from config
            final_template_path = self.config.get_template_path()
            if not final_template_path:
                raise ValueError("Template path must be provided either via --template_path or in the configuration file")
        
        self.paths = {
            "input_root": input_root,
            "output_prep_root": output_prep_root,
            "output_tfm_root": output_tfm_root,
            "template_path": final_template_path,
        }
        
        self.dependency_checker = DependencyChecker()

    def _find_nifti_files(self) -> List[Path]:
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
        logger.info("Using modular standalone steps")
        for key, path in self.paths.items():
            logger.info(f"  {key.replace('_', ' ').title()}: {path.resolve()}")
        logger.info(f"  Keep Intermediate Files: {self.config.keep_intermediate_files}")
        logger.info("=" * 50)
        
        # Check which modules are available
        if not STANDALONE_MODULES_AVAILABLE:
            logger.error("Standalone preprocessing modules not found!")
            logger.error("Please ensure intensity_normalization.py, bias_field_correction.py,")
            logger.error("registration.py, and skull_stripping.py are in the same directory.")
            return
        
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
        description="Modular NIfTI MRI Preprocessing Pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", type=Path, help="Input BIDS-like NIfTI directory.")
    parser.add_argument("output_prep_dir", type=Path, help="Output directory for preprocessed files.")
    parser.add_argument("output_transform_dir", type=Path, help="Output directory for transforms and logs.")
    parser.add_argument("--template_path", type=Path, 
                       help="Path to the MRI template file. If not provided, will use template_path from config.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the YAML pipeline configuration file.")
    parser.add_argument("--log_file", default=None, type=Path, help="Path to main log file.")
    parser.add_argument("--console_log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                       help="Console logging level.")
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