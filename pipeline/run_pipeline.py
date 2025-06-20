import argparse
import yaml
import subprocess
import sys
import os
from pathlib import Path
import logging
from datetime import datetime
import uuid
import zipfile
import shutil
import requests 

# --- Logger Setup ---
logger = logging.getLogger("Pipeline")
USER_LOG_PREFIX = "PIPELINE_USER_MSG:"

def setup_logging(log_file: Path, console_level: str):
    logger.setLevel(logging.DEBUG)
    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [Pipeline] %(message)s')
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    try:
        ch_level = getattr(logging, console_level.upper())
    except AttributeError:
        ch_level = logging.INFO
        logger.warning(f"Invalid console log level '{console_level}'. Using INFO.")
    ch.setLevel(ch_level)
    ch.setFormatter(log_formatter)
    logger.addHandler(ch)

    # File handler
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.error(f"Failed to configure file logging at {log_file}: {e}")

# --- Core Data Management Classes ---

class Config:
    def __init__(self, path: Path):
        self.path = path
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self._data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load or parse config {path}") from e
        
    def get_path(self, key: str) -> Path | None:
        return Path(self._data['paths'][key]) if key in self._data.get('paths', {}) else None
    
    def get_exec(self, key: str, default: str) -> str:
        return self._data.get('executables', {}).get(key, default)
    
    def get_step_config(self, step_name: str) -> dict:
        return self._data.get('steps', {}).get(step_name, {})

class PipelineWorkspace:
    def __init__(self, base_output_dir: Path, run_id: str, config: Config):
        self.run_id = run_id
        self.run_output_dir = base_output_dir / run_id
        sd = config.get_step_config('paths').get('subdirs', {})
        self.logs_dir = self.run_output_dir / sd.get('logs', 'logs')
        self.bids_dicom_dir = self.run_output_dir / sd.get('bids_dicom', 'bids_dicom')
        self.dicom_checks_dir = self.run_output_dir / sd.get('dicom_checks', 'dicom_checks')
        self.dicom_meta_dir = self.run_output_dir / sd.get('dicom_meta', 'dicom_meta')
        self.bids_nifti_dir = self.run_output_dir / sd.get('bids_nifti', 'bids_nifti')
        self.validation_reports_dir = self.run_output_dir / sd.get('validation_reports', 'validation_reports')
        self.fast_qc_reports_dir = self.run_output_dir / sd.get('fast_qc_reports', 'fast_qc_reports')
        self.mriqc_output_dir = self.run_output_dir / sd.get('mriqc_output', 'mriqc_output')
        self.mriqc_interpret_dir = self.run_output_dir / sd.get('mriqc_interpret', 'mriqc_interpret')
        self.transforms_dir = self.run_output_dir / sd.get('transforms', 'transforms')
        self.preprocessed_dir = self.run_output_dir / sd.get('preprocessed', 'preprocessed')
        self.segmentation_dir = self.run_output_dir / sd.get('segmentation_masks', 'segmentation_masks')

    def create_directories(self):
        logger.info(f"{USER_LOG_PREFIX} Creating output directory structure...")
        for path in self.__dict__.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)

class PipelineInputHandler:
    def __init__(self, cli_input_path_str: str | None, config_input_path: Path | None, run_dir: Path):
        self.cli_input_path_str = cli_input_path_str
        self.config_input_path = config_input_path
        self.run_dir = run_dir

    def prepare(self) -> Path:
        path_str = self.cli_input_path_str or self.config_input_path
        if not path_str: raise ValueError("Input data directory is not specified in CLI arguments or config file.")
        path = Path(path_str).resolve()
        logger.info(f"Using input data path: {path}")
        if not path.exists(): raise FileNotFoundError(f"Input data path does not exist: {path}")
        if path.is_file() and path.name.lower().endswith('.zip'):
            extraction_dir = self.run_dir / "input_raw_data"
            extraction_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"ZIP archive detected. Extracting to: {extraction_dir}")
            with zipfile.ZipFile(path, 'r') as zip_ref: zip_ref.extractall(extraction_dir)
            #items = list(extraction_dir.iterdir())
            #return items[0] if len(items) == 1 and items[0].is_dir() else extraction_dir
            return extraction_dir
        elif path.is_dir(): return path
        else: raise ValueError(f"Input path is not a valid directory or .zip file: {path}")

# --- Step Abstraction ---

class Step:
    """Abstract Base Class for a pipeline step."""
    def __init__(self, step_name: str, workspace: PipelineWorkspace, config: Config):
        self.name = step_name
        self.workspace = workspace
        self.config = config
        self.step_config = self.config.get_step_config(step_name.lower().replace(' ', '_'))
        self.python_executable = sys.executable
        self.scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
        self.fatal_on_error = True

    def execute(self) -> bool:
        # Check if this step has a custom is_enabled method
        if hasattr(self, 'is_enabled') and callable(getattr(self, 'is_enabled')):
            if not self.is_enabled():
                logger.info(f"{USER_LOG_PREFIX} Skipped - {self.name} (disabled by custom logic).")
                return True
            
        elif not self.step_config.get('enabled', True):
            logger.info(f"{USER_LOG_PREFIX} Skipped - {self.name} (disabled in config).")
            return True
        
        logger.info(f"{USER_LOG_PREFIX} Starting step - {self.name}...")
        log_path = self.workspace.logs_dir / f"{self.name.lower().replace(' ', '_')}.log"
        try:
            command = self._build_command()
            logger.debug(f"Executing command: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            if result.stdout: logger.debug(f"Stdout from '{self.name}':\n{result.stdout.strip()}")
            if result.stderr: logger.warning(f"Stderr from '{self.name}':\n{result.stderr.strip()}")
            self._post_execute_on_success()
            logger.info(f"{USER_LOG_PREFIX} Success - {self.name}.")
            return True
        except subprocess.CalledProcessError as e:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"COMMAND: {' '.join(e.cmd)}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
            logger.error(f"{USER_LOG_PREFIX} ERROR - {self.name}. See log: {log_path.name}")
            logger.critical(f"Step '{self.name}' failed with code {e.returncode}. Log saved to {log_path.name}")
            return self.fatal_on_error is False
        except Exception as e:
            logger.error(f"{USER_LOG_PREFIX} CRITICAL ERROR - {self.name}: An unexpected error occurred.")
            logger.exception(e)
            return False
        
    def _build_command(self) -> list[str]: raise NotImplementedError
    def _post_execute_on_success(self): pass # Hook for post-run actions

class ReorganizeStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config, input_dir: Path):
        super().__init__("Reorganize_to_BIDS_DICOM", workspace, config)
        self.input_dir = input_dir
        self.action = self.step_config.get('action', 'copy').lower()

    def _build_command(self) -> list[str]:
        return [self.python_executable, str(self.scripts_dir / "reorganize_folders.py"),
                str(self.input_dir),
                str(self.workspace.bids_dicom_dir),
                "--action", self.action]
    
    def _post_execute_on_success(self):
        if self.action == 'move' and self.input_dir.exists():
            logger.info(f"Action was 'move'. Cleaning up source directory: {self.input_dir}")
            shutil.rmtree(self.input_dir)

class DICOMCheckStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("DICOM_Standard_Check", workspace, config)

    def _build_command(self) -> list[str]:
        return [self.python_executable, str(self.scripts_dir / "dicom_standard_check.py"),
                str(self.workspace.bids_dicom_dir),
                str(self.workspace.dicom_checks_dir),
                "--dciodvfy_path", self.config.get_exec('dciodvfy', 'dciodvfy')]

class ExtractMetadataStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("Extract_DICOM_Metadata", workspace, config)

    def _build_command(self) -> list[str]:
        return [self.python_executable, str(self.scripts_dir / "extract_metadata.py"),
                str(self.workspace.bids_dicom_dir),
                str(self.workspace.dicom_meta_dir)]
    
class DicomToNiftiStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("Convert_DICOM_to_NIfTI", workspace, config)
    def _build_command(self) -> list[str]:
        return [self.python_executable, str(self.scripts_dir / "convert_dicom_to_nifti.py"),
                str(self.workspace.bids_dicom_dir),
                str(self.workspace.bids_nifti_dir),
                "--dcm2niix_path", self.config.get_exec('dcm2niix', 'dcm2niix')]

class BIDSValidationStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("BIDS_Validation", workspace, config)
        self.fatal_on_error = False 
        # A validation error should not stop the pipeline
    def _build_command(self) -> list[str]:
        return [self.python_executable, str(self.scripts_dir / "bids_validation.py"),
                str(self.workspace.bids_nifti_dir), 
                str(self.workspace.validation_reports_dir),
                "--validator_path", self.config.get_exec('bids_validator', 'bids-validator')]
    
    def _post_execute_on_success(self):
        report_file = self.workspace.validation_reports_dir / "bids_validator_report.txt"
        if report_file.exists():
            content = report_file.read_text()
            if "error" in content.lower():
                logger.warning(f"{USER_LOG_PREFIX} BIDS Validation found errors. Please review the report.")

class MRIQCServerTriggerStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("MRIQC_Server_Trigger", workspace, config)
        self.mriqc_config = self.config.get_step_config('mriqc')

    def _is_enabled(self) -> bool:
        return (self.mriqc_config.get('enabled', False) and
                self.mriqc_config.get('run_on_server', False) and
                self.mriqc_config.get('run_on_server_auto_trigger', False))
    
    def execute(self) -> bool:
        if not self._is_enabled():
            logger.info(f"Skipping {self.name}: conditions not met in config.")
            return True
        logger.info(f"{USER_LOG_PREFIX} Initiating automatic MRIQC run on server...")
        trigger_url = f"{self.config.get_exec('flask_api_base_url', 'http://127.0.0.1:5001')}/trigger_mriqc_auto/{self.workspace.run_id}"
        try:
            if not self.workspace.bids_nifti_dir.is_dir() or not any(self.workspace.bids_nifti_dir.iterdir()):
                logger.warning(f"Cannot trigger server MRIQC: BIDS NIfTI directory is empty or missing.")
                return True
            response = requests.post(trigger_url, timeout=20)
            if response.status_code in [200, 202]:
                logger.info(f"{USER_LOG_PREFIX} Server trigger successful: {response.json().get('message', '')}")
            else:
                logger.error(f"{USER_LOG_PREFIX} Failed to trigger server MRIQC. Status: {response.status_code}, Info: {response.text}")
        except requests.RequestException as e:
            logger.error(f"{USER_LOG_PREFIX} Failed to connect to server for MRIQC trigger: {e}")
        return True # Non-fatal

class MRIQCLocalStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("MRIQC_Local_Analysis", workspace, config)
        self.mriqc_config = self.config.get_step_config('mriqc')

    def _is_enabled(self) -> bool:
        return (self.mriqc_config.get('enabled', False) and not self.mriqc_config.get('run_on_server', False))
    
    def execute(self) -> bool:
        if not self._is_enabled():
            logger.info(f"Skipping {self.name}: conditions not met in config.")
            return True
        
        logger.info(f"{USER_LOG_PREFIX} Starting step - {self.name}...")
        log_path = self.workspace.logs_dir / f"{self.name.lower().replace(' ', '_')}.log"
        
        try:
            command = self._build_command()
            logger.debug(f"Executing command: {' '.join(command)}")
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            
            if result.stdout: 
                logger.debug(f"Stdout from '{self.name}':\n{result.stdout.strip()}")
            if result.stderr: 
                logger.warning(f"Stderr from '{self.name}':\n{result.stderr.strip()}")
            
            self._post_execute_on_success()
            logger.info(f"{USER_LOG_PREFIX} Success - {self.name}.")
            return True
        
        except subprocess.CalledProcessError as e:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"COMMAND: {' '.join(e.cmd)}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
            logger.error(f"{USER_LOG_PREFIX} ERROR - {self.name}. See log: {log_path.name}")
            logger.critical(f"Step '{self.name}' failed with code {e.returncode}. Log saved to {log_path.name}")
            return self.fatal_on_error is False
        
        except Exception as e:
            logger.error(f"{USER_LOG_PREFIX} CRITICAL ERROR - {self.name}: An unexpected error occurred.")
            logger.exception(e)
            return False
    
    def _build_command(self) -> list:
        cfg = self.mriqc_config
        return [self.python_executable, str(self.scripts_dir / "mriqc_quality.py"),
                str(self.workspace.bids_nifti_dir),
                str(self.workspace.mriqc_output_dir),
                "--mriqc_path", self.config.get_exec('mriqc_local_exec', 'mriqc'),
                "--report_type", str(cfg.get('report_type', 'both')),
                "--n_procs", str(cfg.get('n_procs', 1)),
                "--n_threads", str(cfg.get('n_threads', 1)),
                "--mem_gb", str(cfg.get('mem_gb', 4))]

class MRIQCInterpretationStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("MRIQC_Interpretation", workspace, config)
        self.mriqc_config = config.get_step_config('mriqc')
        self.interpret_config = config.get_step_config('mriqc_interpretation')

    def _is_enabled(self) -> bool:
        return (self.mriqc_config.get('enabled', False) and
                not self.mriqc_config.get('run_on_server', False) and
                self.interpret_config.get('enabled', False))
    
    def execute(self) -> bool:
        if not self._is_enabled():
            logger.info(f"Skipping {self.name}: conditions not met in config.")
            return True
        if not self.workspace.mriqc_output_dir.is_dir() or not any(self.workspace.mriqc_output_dir.iterdir()):
            logger.warning(f"{USER_LOG_PREFIX} Skipped - {self.name}: MRIQC output directory is empty.")
            return True
        return super().execute()
    
    def _build_command(self) -> list:
        return [self.python_executable, str(self.scripts_dir / "mriqc_interpretation.py"),
                str(self.workspace.mriqc_output_dir),
                str(self.workspace.mriqc_interpret_dir)]

class FastQCMetricsStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("Fast_Quality_Metrics", workspace, config)

    def _build_command(self) -> list[str]:
        return [self.python_executable, str(self.scripts_dir / "quality_metrics.py"),
                str(self.workspace.bids_nifti_dir),
                str(self.workspace.fast_qc_reports_dir)]

class PreprocessingStep(Step):
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("Preprocessing", workspace, config)
    def _build_command(self) -> list[str]:
        return [self.python_executable, str(self.scripts_dir / "preprocessing.py"),
                str(self.workspace.bids_nifti_dir), 
                str(self.workspace.preprocessed_dir),
                str(self.workspace.transforms_dir),
                "--template_path", str(self.config.get_path('template_path')), 
                "--config", str(self.config.path)]

class SegmentationStep(Step):
    """Concrete step for running segmentation.py for all subjects/sessions."""
    def __init__(self, workspace: PipelineWorkspace, config: Config):
        super().__init__("AI_Segmentation", workspace, config)

    def execute(self) -> bool:
        """
        Custom execution logic for segmentation, which finds all modalities for
        each session and calls the segmentation script.
        """
        logger.info(f"{USER_LOG_PREFIX} Starting step - {self.name}...")
        
        seg_config = self.config.get_step_config('segmentation')
        if not seg_config.get('enabled', False):
            logger.info(f"{USER_LOG_PREFIX} Skipped - {self.name} (disabled in config).")
            return True

        session_files = self._find_session_files()
        if not session_files:
            logger.warning(f"No subject/session folders found in {self.workspace.preprocessed_dir} to segment.")
            return True

        error_count = 0
        for (subj, ses), files_map in session_files.items():
            client_id = f"{subj}_{ses}"
            logger.info(f"--- Preparing segmentation for {client_id} ---")
            
            base_name_for_output = f"{subj}_{ses}"
            output_mask_filename = f"{base_name_for_output}_segmask.nii.gz"

            output_mask_subfolder = self.workspace.segmentation_dir / subj / ses / "anat"
            output_mask_subfolder.mkdir(parents=True, exist_ok=True)
            output_mask_file_path = output_mask_subfolder / output_mask_filename

            step_log_path = self.workspace.logs_dir / f"segmentation_{client_id}.log"
            
            try:
                command = self._build_command_for_session(files_map, output_mask_file_path, client_id)
                
                # Check that we have at least one file to send before running
                if not any(arg.startswith('--input') for arg in command):
                    logger.error(f"For {client_id}, no valid modality files were found to process. Skipping.")
                    error_count += 1
                    continue

                logger.debug(f"Executing command: {' '.join(command)}")
                # Use a generic run_step function to handle subprocess logic
                self._run_single_segmentation(command, client_id, step_log_path)
                logger.info(f"--- Segmentation successful for {client_id} ---")

            except Exception as e:
                logger.error(f"--- Segmentation FAILED for {client_id} ---")
                logger.exception(e)
                error_count += 1

        if error_count > 0:
            logger.error(f"{USER_LOG_PREFIX} ERROR - {self.name} completed with {error_count} failures.")
            return False
        
        logger.info(f"{USER_LOG_PREFIX} Success - {self.name}.")
        return True

    def _find_session_files(self) -> dict:
        """Groups preprocessed files by subject and session."""
        sessions = {}
        mod_map = self.config.get_step_config('segmentation').get('modality_input_map', {})
        if not mod_map:
            logger.warning("`modality_input_map` is not defined in the config under segmentation. Cannot find files.")
            return sessions

        for subj_dir in self.workspace.preprocessed_dir.glob("sub-*"):
            if not subj_dir.is_dir(): continue
            for ses_dir in subj_dir.glob("ses-*"):
                if not ses_dir.is_dir() or not (ses_dir / "anat").exists(): continue
                key = (subj_dir.name, ses_dir.name)
                sessions[key] = {}
                for mod_key, mod_id in mod_map.items():
                    found_files = list((ses_dir / "anat").glob(f"*{mod_id}*.nii*"))
                    if found_files:
                        # Add logic to avoid mistaking T1c for T1
                        if mod_key == 't1' and 't1c' in mod_map:
                            if mod_map['t1c'] in found_files[0].name:
                                continue # This is a T1c file, skip for T1 key
                        sessions[key][mod_key] = found_files[0]
        return sessions

    def _build_command_for_session(self, files_map, output_mask, client_id) -> list[str]:
        """Builds the command for a single segmentation run."""
        command = [
            self.python_executable,
            str(self.scripts_dir / "segmentation.py"),
            "--output_mask", str(output_mask),
            "--config", str(self.config.path),
            "--client_id", client_id,
        ]
        if files_map.get('t1'): command.extend(["--input_t1", str(files_map['t1'])])
        if files_map.get('t1c'): command.extend(["--input_t1ce", str(files_map['t1c'])])
        if files_map.get('t2'): command.extend(["--input_t2", str(files_map['t2'])])
        if files_map.get('flair'): command.extend(["--input_flair", str(files_map['flair'])])
        return command

    def _run_single_segmentation(self, command: list[str], client_id: str, log_path: Path):
        """Helper to run one instance of the segmentation script and handle its output."""
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8')
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(f"COMMAND: {' '.join(e.cmd)}\n\nSTDOUT:\n{e.stdout}\n\nSTDERR:\n{e.stderr}")
            # Re-raise the exception to be caught by the main execute loop
            raise RuntimeError(f"Segmentation for {client_id} failed. See log: {log_path.name}") from e

# --- Main Orchestrator ---

class PipelineRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = Config(Path(args.config))
        run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_output_dir = Path(args.output_base_dir or self.config.get_path('output_base_dir'))
        self.workspace = PipelineWorkspace(base_output_dir, run_id, self.config)
        log_file = self.workspace.logs_dir / "pipeline.log"
        setup_logging(log_file, args.console_log_level)
        logger.info(f"Pipeline run starting with ID: {run_id}")

    def run(self):
        try:
            self.workspace.create_directories()
            input_handler = PipelineInputHandler(
                self.args.input_data_dir, self.config.get_path('raw_input_dir'), self.workspace.run_output_dir)
            effective_input_dir = input_handler.prepare()

            logger.info("=" * 60)
            logger.info(f"Effective Input Directory: {effective_input_dir}")
            logger.info(f"Run Output Directory: {self.workspace.run_output_dir}")
            logger.info("=" * 60)

            steps_to_run = [
                ReorganizeStep(self.workspace, self.config, effective_input_dir),
                DICOMCheckStep(self.workspace, self.config),
                ExtractMetadataStep(self.workspace, self.config),
                DicomToNiftiStep(self.workspace, self.config),
                BIDSValidationStep(self.workspace, self.config),
                MRIQCServerTriggerStep(self.workspace, self.config),
                MRIQCLocalStep(self.workspace, self.config),
                MRIQCInterpretationStep(self.workspace, self.config),
                FastQCMetricsStep(self.workspace, self.config),
                PreprocessingStep(self.workspace, self.config),
                SegmentationStep(self.workspace, self.config),
            ]

            for step in steps_to_run:
                if not step.execute():
                    logger.critical(f"Pipeline aborted due to failure in step: {step.name}")
                    sys.exit(1)

            logger.info(f"{USER_LOG_PREFIX} Success - Pipeline completed all steps.")
        except Exception as e:
            logger.error(f"{USER_LOG_PREFIX} A critical error occurred during the pipeline execution.")
            logger.critical("Pipeline failed with an unhandled exception.", exc_info=True)
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main pipeline orchestrator.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="Path to the main YAML config file.")
    parser.add_argument("--run_id", help="Unique ID for this run. If not provided, one will be generated.")
    parser.add_argument("--input_data_dir", help="Override input path from config. Can be a directory or ZIP file.")
    parser.add_argument("--output_base_dir", help="Override base output directory from config.")
    parser.add_argument("--console_log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console logging level.")
    args = parser.parse_args()
    try:
        runner = PipelineRunner(args)
        runner.run()
        sys.exit(0)
    except (ValueError, FileNotFoundError) as e:
        logger.critical(f"Setup failed: {e}")
        sys.exit(1)
    except Exception:
        logger.critical("A fatal exception occurred during pipeline initialization.", exc_info=True)
        sys.exit(1)