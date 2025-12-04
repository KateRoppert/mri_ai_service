# -*- coding: utf-8 -*-
"""
Batch MRI segmentation script for BIDS-formatted datasets.

This script processes multiple subjects/sessions from a BIDS directory structure,
sending NIfTI files to a segmentation server and organizing outputs in BIDS format.

Key components:
- Config: Manages YAML configuration including modality mappings
- SubjectSession: Data structure for a single subject/session
- BIDSScanner: Discovers and validates subjects in BIDS directories
- SegmentationInput: Validates complete modality sets
- SegmentationClient: Handles HTTP communication with segmentation server
- SegmentationRunner: Orchestrates batch processing workflow
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import yaml
import requests
from dataclasses import dataclass
from contextlib import ExitStack
from collections import defaultdict

# --- Logger Setup ---
logger = logging.getLogger(__name__)

def setup_logging(log_file: Optional[Path] = None, console_level: str = "INFO"):
    """Configures the main application logger."""
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(name)s] %(message)s')

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    try:
        ch.setLevel(getattr(logging, console_level.upper()))
    except AttributeError:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, encoding='utf-8', mode='a')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        except Exception as e:
            logger.error(f"Failed to set up file logger at {log_file}: {e}")

# --- Core Classes ---

class Config:
    """Handles loading and providing access to the YAML configuration file."""
    def __init__(self, config_path: Path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self._data = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
        except Exception as e:
            logger.critical(f"Failed to load or parse config {config_path}: {e}")
            raise

    def get_server_url(self) -> str:
        """Retrieves the AIAA server URL from the config."""
        url = self._data.get('executables', {}).get('aiaa_server_url')
        if not url:
            raise ValueError("'executables.aiaa_server_url' not found in config.")
        return url

    def get_model_name(self) -> str:
        """Retrieves the segmentation model name from the config."""
        model = self._data.get('segmentation', {}).get('model_name')
        if not model:
            raise ValueError("'segmentation.model_name' not found in config.")
        return model

    def get_modality_map(self) -> dict[str, str]:
        """
        Retrieves the modality input mapping from the config.
        
        Returns:
            Dictionary mapping server keys to BIDS suffixes.
            Example: {"t1": "T1w", "t1c": "ce-gd_T1w", "t2": "T2w", "flair": "FLAIR"}
        """
        modality_map = self._data.get('segmentation', {}).get('modality_input_map')
        if not modality_map:
            raise ValueError("'segmentation.modality_input_map' not found in config.")
        
        # Validate that all required keys are present
        required_keys = {"t1", "t1c", "t2", "flair"}
        if not required_keys.issubset(modality_map.keys()):
            missing = required_keys - modality_map.keys()
            raise ValueError(f"Missing required modality keys in config: {missing}")
        
        return modality_map


@dataclass
class SubjectSession:
    """
    Data structure for a single subject/session with all required information.
    """
    subject_id: str
    session_id: Optional[str]
    modality_files: dict[str, Path]  # Keys: t1, t1c, t2, flair
    output_mask_path: Path
    
    def get_identifier(self) -> str:
        """Returns a human-readable identifier for logging."""
        if self.session_id:
            return f"{self.subject_id}_{self.session_id}"
        return self.subject_id
    
    def has_all_modalities(self) -> bool:
        """Checks if all required modalities are present."""
        required = {"t1", "t1c", "t2", "flair"}
        return required.issubset(self.modality_files.keys())


class BIDSScanner:
    """
    Scans a BIDS directory structure and discovers subject/session combinations.
    """
    def __init__(self, input_dir: Path, output_dir: Path, modality_map: dict[str, str]):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.modality_map = modality_map
        
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    
    def scan(self, max_subjects: Optional[int] = None) -> list[SubjectSession]:
        """
        Scans the input directory and returns a list of SubjectSession objects.
        
        Args:
            max_subjects: Maximum number of subjects to process (for testing)
        
        Returns:
            List of SubjectSession objects with complete metadata
        """
        sessions = []
        subject_dirs = sorted([d for d in self.input_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('sub-')])
        
        logger.info(f"Found {len(subject_dirs)} subject directories in {self.input_dir}")
        
        for subject_dir in subject_dirs:
            if max_subjects and len(sessions) >= max_subjects:
                logger.info(f"Reached max_subjects limit ({max_subjects}). Stopping scan.")
                break
            
            subject_id = subject_dir.name
            sessions.extend(self._scan_subject(subject_dir, subject_id))
        
        logger.info(f"Total sessions discovered: {len(sessions)}")
        return sessions
    
    def _scan_subject(self, subject_dir: Path, subject_id: str) -> list[SubjectSession]:
        """Scans a single subject directory for sessions."""
        sessions = []
        
        # Check for session subdirectories
        session_dirs = sorted([d for d in subject_dir.iterdir() 
                              if d.is_dir() and d.name.startswith('ses-')])
        
        if session_dirs:
            # Process each session
            for session_dir in session_dirs:
                session_id = session_dir.name
                anat_dir = session_dir / "anat"
                
                if not anat_dir.exists():
                    logger.warning(f"No 'anat' directory found for {subject_id}/{session_id}")
                    continue
                
                session = self._create_session(subject_id, session_id, anat_dir)
                if session:
                    sessions.append(session)
        else:
            # No sessions - check for direct anat directory
            anat_dir = subject_dir / "anat"
            if anat_dir.exists():
                session = self._create_session(subject_id, None, anat_dir)
                if session:
                    sessions.append(session)
            else:
                logger.warning(f"No 'anat' directory found for {subject_id}")
        
        return sessions
    
    def _create_session(
        self, 
        subject_id: str, 
        session_id: Optional[str], 
        anat_dir: Path
    ) -> Optional[SubjectSession]:
        """
        Creates a SubjectSession object by finding all modality files.
        
        Returns None if not all modalities are found.
        """
        identifier = f"{subject_id}_{session_id}" if session_id else subject_id
        
        # Find files for each modality
        modality_files = {}
        for server_key, bids_suffix in self.modality_map.items():
            # Pattern: sub-XXX_ses-YYY_<bids_suffix>.nii.gz or sub-XXX_<bids_suffix>.nii.gz
            if session_id:
                pattern = f"{subject_id}_{session_id}_{bids_suffix}.nii.gz"
            else:
                pattern = f"{subject_id}_{bids_suffix}.nii.gz"
            
            matches = list(anat_dir.glob(pattern))
            
            if matches:
                modality_files[server_key] = matches[0]
            else:
                logger.debug(f"{identifier}: Missing modality '{server_key}' (pattern: {pattern})")
        
        # Check if we have all required modalities
        required = {"t1", "t1c", "t2", "flair"}
        if not required.issubset(modality_files.keys()):
            missing = required - modality_files.keys()
            logger.warning(f"⚠️  Skipping {identifier}: Missing modalities {missing}")
            return None
        
        # Determine output path (mirrors BIDS structure)
        if session_id:
            output_anat_dir = self.output_dir / subject_id / session_id / "anat"
            base_filename = f"{subject_id}_{session_id}"
        else:
            output_anat_dir = self.output_dir / subject_id / "anat"
            base_filename = subject_id
        
        # Pick one modality file to derive the output name from (e.g., T1w)
        reference_file = modality_files["t1"]
        # Extract the full suffix after subject/session ID
        # e.g., sub-001_ses-01_T1w.nii.gz -> T1w.nii.gz
        ref_name = reference_file.name
        suffix_start = ref_name.find(self.modality_map["t1"])
        if suffix_start != -1:
            original_suffix = ref_name[suffix_start:]
            # Insert _segmask before .nii.gz
            output_name = original_suffix.replace('.nii.gz', '_segmask.nii.gz')
        else:
            # Fallback if pattern not found
            output_name = f"{self.modality_map['t1']}_segmask.nii.gz"
        
        output_mask_path = output_anat_dir / f"{base_filename}_{output_name}"
        
        logger.debug(f"✓ {identifier}: All modalities found")
        
        return SubjectSession(
            subject_id=subject_id,
            session_id=session_id,
            modality_files=modality_files,
            output_mask_path=output_mask_path
        )


@dataclass
class SegmentationInput:
    """
    Validates and prepares input files for segmentation.
    No fallback logic - all modalities must be present.
    """
    t1: Path
    t1c: Path
    t2: Path
    flair: Path
    
    def validate(self) -> bool:
        """Validates that all files exist."""
        all_files = [self.t1, self.t1c, self.t2, self.flair]
        
        for file_path in all_files:
            if not file_path.exists():
                logger.error(f"Required file not found: {file_path}")
                return False
        
        return True
    
    def prepare_for_server(self) -> dict[str, Path]:
        """Returns a dictionary ready for server submission."""
        if not self.validate():
            raise FileNotFoundError("Cannot prepare files for server: validation failed")
        
        files = {
            "t1": self.t1,
            "t1c": self.t1c,
            "t2": self.t2,
            "flair": self.flair,
        }
        
        logger.debug("Files prepared for server:")
        for mod, path in files.items():
            logger.debug(f"  - {mod.upper()}: {path.name}")
        
        return files


class SegmentationClient:
    """Handles all HTTP communication with the segmentation server."""
    def __init__(self, server_url: str, timeout: int = 1200):
        self.server_url = server_url
        self.timeout = timeout

    def segment(
        self,
        files_to_send: dict[str, Path],
        model_name: str,
        client_id: str,
        output_path: Path
    ) -> bool:
        """
        Sends files to the server for segmentation and saves the resulting mask.
        
        Returns:
            True on success, False on failure.
        """
        inference_url = f"{self.server_url}/v1/inference?net={model_name}&client_id={client_id}"
        logger.debug(f"Sending request to: {inference_url}")

        try:
            # ExitStack cleanly manages opening multiple files
            with ExitStack() as stack:
                files_multipart = {
                    f't1_{files_to_send["t1"].name}': (
                        files_to_send["t1"].name, 
                        stack.enter_context(open(files_to_send["t1"], 'rb'))
                    ),
                    f't1c_{files_to_send["t1c"].name}': (
                        files_to_send["t1c"].name,
                        stack.enter_context(open(files_to_send["t1c"], 'rb'))
                    ),
                    f't2_{files_to_send["t2"].name}': (
                        files_to_send["t2"].name,
                        stack.enter_context(open(files_to_send["t2"], 'rb'))
                    ),
                    f't2flair_{files_to_send["flair"].name}': (
                        files_to_send["flair"].name,
                        stack.enter_context(open(files_to_send["flair"], 'rb'))
                    ),
                }

                logger.info("  → Uploading files to server...")
                response = requests.post(inference_url, files=files_multipart, timeout=self.timeout)
                response.raise_for_status()

                logger.info("  → Saving segmentation mask...")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f_mask:
                    f_mask.write(response.content)
                
                logger.info(f"  ✓ Success: {output_path}")
                return True

        except requests.exceptions.Timeout:
            logger.error(f"  ✗ Request timed out after {self.timeout} seconds")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"  ✗ Connection error: {e}")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"  ✗ HTTP Error {e.response.status_code}: {e.response.reason}")
            if hasattr(e.response, 'text'):
                logger.error(f"  Server response: {e.response.text[:500]}")
            return False
        except Exception as e:
            logger.exception(f"  ✗ Unexpected error during segmentation: {e}")
            return False


@dataclass
class ProcessingStats:
    """Tracks processing statistics."""
    total: int = 0
    successful: int = 0
    skipped: int = 0
    failed: int = 0
    
    def log_summary(self):
        """Logs a summary of processing statistics."""
        logger.info("=" * 60)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total sessions discovered: {self.total}")
        logger.info(f"Successfully processed:    {self.successful}")
        logger.info(f"Failed:                    {self.failed}")
        logger.info(f"Skipped (incomplete):      {self.skipped}")
        logger.info("=" * 60)


class SegmentationRunner:
    """Orchestrates the batch segmentation workflow."""
    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args
        self.stats = ProcessingStats()

    def run(self) -> bool:
        """Executes the full batch segmentation process."""
        logger.info("=" * 60)
        logger.info("BATCH MRI SEGMENTATION - STARTING")
        logger.info("=" * 60)
        logger.info(f"Input directory:  {self.args.input_dir}")
        logger.info(f"Output directory: {self.args.output_dir}")
        logger.info(f"Config file:      {self.args.config}")
        
        try:
            # Scan for subjects/sessions
            scanner = BIDSScanner(
                input_dir=self.args.input_dir,
                output_dir=self.args.output_dir,
                modality_map=self.config.get_modality_map()
            )
            
            sessions = scanner.scan(max_subjects=self.args.max_subjects)
            
            if not sessions:
                logger.warning("No valid sessions found to process!")
                return False
            
            self.stats.total = len(sessions)
            
            # Initialize segmentation client
            client = SegmentationClient(server_url=self.config.get_server_url())
            model_name = self.config.get_model_name()
            
            # Process each session
            logger.info("=" * 60)
            logger.info("PROCESSING SESSIONS")
            logger.info("=" * 60)
            
            for idx, session in enumerate(sessions, 1):
                identifier = session.get_identifier()
                logger.info(f"[{idx}/{self.stats.total}] Processing: {identifier}")
                
                try:
                    # Prepare input
                    seg_input = SegmentationInput(
                        t1=session.modality_files["t1"],
                        t1c=session.modality_files["t1c"],
                        t2=session.modality_files["t2"],
                        flair=session.modality_files["flair"]
                    )
                    
                    files_to_send = seg_input.prepare_for_server()
                    
                    # Generate unique client ID
                    client_id = f"{identifier}_{idx}"
                    
                    # Perform segmentation
                    success = client.segment(
                        files_to_send=files_to_send,
                        model_name=model_name,
                        client_id=client_id,
                        output_path=session.output_mask_path
                    )
                    
                    if success:
                        self.stats.successful += 1
                    else:
                        self.stats.failed += 1
                        logger.error(f"  Failed to process {identifier}")
                
                except Exception as e:
                    self.stats.failed += 1
                    logger.error(f"  ✗ Error processing {identifier}: {e}")
                    logger.exception(f"  Full traceback for {identifier}:")
            
            # Log final statistics
            self.stats.log_summary()
            
            return self.stats.failed == 0
        
        except Exception as e:
            logger.critical(f"Critical error in batch processing: {e}")
            logger.exception("Full traceback:")
            return False


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch MRI segmentation for BIDS-formatted datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with preprocessed NIfTI files in BIDS format"
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for segmentation masks"
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to log file (default: output_dir/segmentation.log)"
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=None,
        help="Maximum number of subjects to process (for testing)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "segmentation_config.yaml",
        help="Path to segmentation configuration file"
    )
    parser.add_argument(
        "--console-log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level"
    )
    
    args = parser.parse_args()

    # Determine log file path
    if args.log_file:
        log_file = args.log_file
    else:
        log_file = args.output_dir / "segmentation.log"

    setup_logging(log_file, args.console_log_level)

    try:
        app_config = Config(args.config)
        runner = SegmentationRunner(config=app_config, args=args)
        is_successful = runner.run()
        sys.exit(0 if is_successful else 1)
        
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.critical("A fatal exception occurred.", exc_info=True)
        sys.exit(1)