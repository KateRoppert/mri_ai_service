# -*- coding: utf-8 -*-
import os
import subprocess
import shutil
import argparse
import logging
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Generator, Dict

# =============================================================================
# 1. SETUP & UTILITIES
# =============================================================================

logger = logging.getLogger(__name__)

def setup_logging(log_file: Path):
    """Configures logging to both console (INFO) and file (DEBUG)."""
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()

    log_file.parent.mkdir(exist_ok=True, parents=True)
    
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s] - %(message)s'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================

@dataclass
class ConversionTask:
    """Represents a single DICOM series to be converted."""
    source_dir: Path
    subject: str
    session: str
    raw_modality: str # The original folder name, e.g., 't1c'

# =============================================================================
# 3. CONCRETE COMPONENTS (Single Responsibility Classes)
# =============================================================================

class TaskFinder:
    """Finds valid conversion tasks within a BIDS-like DICOM directory."""
    def find(self, input_root: Path) -> Generator[ConversionTask, None, None]:
        """Yields ConversionTask objects for each valid DICOM series found."""
        logger.info(f"Scanning for conversion tasks in '{input_root}'...")
        for root, _, files in os.walk(input_root):
            if any(f.lower().endswith('.dcm') for f in files):
                task = self._create_task_from_path(Path(root), input_root)
                if task:
                    logger.info(f"Found task: sub-{task.subject}, ses-{task.session}, mod-{task.raw_modality}")
                    yield task

    def _create_task_from_path(self, dir_path: Path, root_path: Path) -> ConversionTask | None:
        """Validates path structure and creates a task if valid."""
        try:
            rel_path_parts = dir_path.relative_to(root_path).parts
            # Expects: ('sub-xxx', 'ses-yyy', 'anat', 'modality_folder')
            if len(rel_path_parts) == 4 and rel_path_parts[2] == 'anat':
                return ConversionTask(
                    source_dir=dir_path,
                    subject=rel_path_parts[0].replace('sub-', ''),
                    session=rel_path_parts[1].replace('ses-', ''),
                    raw_modality=rel_path_parts[3]
                )
            else:
                logger.warning(f"Path does not match expected BIDS structure 'sub/ses/anat/modality': {dir_path}")
                return None
        except ValueError:
            logger.warning(f"Could not determine relative path for {dir_path}. Skipping.")
            return None

class BidsNamer:
    """A strategy for determining BIDS-compliant names."""
    MODALITY_MAP: Dict[str, str] = {
        't1c': 'ce-GAD_T1w',
        't1gd': 'ce-GAD_T1w',
        'contrast': 'ce-GAD_T1w',
        't1': 'T1w',
        't2fl': 'FLAIR',
        'flair': 'FLAIR',
        't2': 'T2w',
        # Add new mappings here easily
    }

    def get_bids_modality(self, raw_modality: str) -> str | None:
        """Maps a raw folder name to a BIDS modality entity."""
        modality_lower = raw_modality.lower()
        for key, value in self.MODALITY_MAP.items():
            if key in modality_lower:
                return value
        logger.warning(f"Could not map raw modality '{raw_modality}' to a BIDS entity.")
        return None

    def get_bids_filename_base(self, task: ConversionTask) -> str | None:
        """Constructs the base BIDS filename (without extension)."""
        bids_modality = self.get_bids_modality(task.raw_modality)
        if not bids_modality:
            return None
        return f"sub-{task.subject}_ses-{task.session}_{bids_modality}"

class Dcm2niixRunner:
    """A strategy for running the dcm2niix conversion tool."""
    def __init__(self, executable_path: str, timeout: int = 300):
        self.executable_path = self._find_executable(executable_path)
        self.timeout = timeout

    def _find_executable(self, path: str) -> str:
        """Validates that the dcm2niix executable can be found."""
        found_path = shutil.which(path)
        if not found_path:
            raise FileNotFoundError(f"dcm2niix executable not found at '{path}' or in system PATH.")
        logger.info(f"dcm2niix executable found at: {found_path}")
        return found_path

    def run(self, task: ConversionTask, output_dir: Path, temp_filename: str) -> bool:
        """Runs dcm2niix for a given task."""
        cmd = [
            self.executable_path,
            '-z', 'y',                # Enable gzip compression
            '-o', str(output_dir),   # Set output directory
            '-f', temp_filename,     # Set temporary output filename format
            str(task.source_dir)     # Set input directory
        ]
        logger.debug(f"  Executing command: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            if result.stdout:
                logger.debug(f"  dcm2niix stdout:\n{result.stdout.strip()}")
            if result.stderr:
                logger.debug(f"  dcm2niix stderr:\n{result.stderr.strip()}")
            
            if result.returncode != 0:
                logger.error(f"dcm2niix failed for {task.source_dir} with exit code {result.returncode}.")
                return False
            return True
        except subprocess.TimeoutExpired:
            logger.error(f"dcm2niix timed out for {task.source_dir}.")
            return False
        except Exception as e:
            logger.error(f"Unexpected error running dcm2niix for {task.source_dir}: {e}", exc_info=True)
            return False

class BidsFileRenamer:
    """Handles renaming of converter output files to BIDS format."""
    def rename(self, output_dir: Path, temp_filename_base: str, final_filename_base: str) -> bool:
        """Finds temporary files and renames them to their final BIDS names."""
        logger.info(f"  Renaming output files from '{temp_filename_base}' to '{final_filename_base}'...")
        renamed_nii = self._rename_file(output_dir, temp_filename_base, final_filename_base, ".nii.gz")
        self._rename_file(output_dir, temp_filename_base, final_filename_base, ".json")
        
        if not renamed_nii:
            logger.error(f"Failed to find and rename primary .nii.gz file for base '{temp_filename_base}'")
            return False
        return True

    def _rename_file(self, directory: Path, temp_base: str, final_base: str, extension: str) -> bool:
        """Helper to find and rename a single file type."""
        # dcm2niix might add suffixes (_e1, _ph), so we glob
        for temp_file in directory.glob(f"{temp_base}*{extension}"):
            final_path = directory / f"{final_base}{extension}"
            logger.debug(f"    Renaming '{temp_file.name}' to '{final_path.name}'")
            try:
                shutil.move(str(temp_file), str(final_path))
                return True # Assume we only need to rename the first one found
            except OSError as e:
                logger.error(f"    Failed to rename {temp_file}: {e}")
                return False
        return False # File not found

# =============================================================================
# 4. ORCHESTRATOR / FACADE
# =============================================================================

class DicomToNiftiConverter:
    """Orchestrates the DICOM to NIfTI conversion process."""
    def __init__(
        self,
        task_finder: TaskFinder,
        namer: BidsNamer,
        runner: Dcm2niixRunner,
        renamer: BidsFileRenamer
    ):
        self.task_finder = task_finder
        self.namer = namer
        self.runner = runner
        self.renamer = renamer

    def convert_all(self, input_dir: Path, output_dir: Path):
        """Facade method to run the entire conversion process."""
        stats = {'converted': 0, 'skipped': 0, 'failed': 0}
        
        for task in self.task_finder.find(input_dir):
            final_filename_base = self.namer.get_bids_filename_base(task)
            if not final_filename_base:
                stats['skipped'] += 1
                continue
                
            # Create a unique temporary filename to avoid collisions
            temp_filename_base = f"tmp_{task.subject}_{task.session}_{task.raw_modality}"
            
            # The output directory for this specific task
            task_output_dir = output_dir / f"sub-{task.subject}" / f"ses-{task.session}" / "anat"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            
            run_success = self.runner.run(task, task_output_dir, temp_filename_base)
            
            if not run_success:
                stats['failed'] += 1
                continue

            rename_success = self.renamer.rename(task_output_dir, temp_filename_base, final_filename_base)

            if rename_success:
                stats['converted'] += 1
            else:
                stats['failed'] += 1
        
        self._log_summary(stats)

    def _log_summary(self, stats: Dict[str, int]):
        logger.info("=" * 60)
        logger.info("Conversion Summary")
        logger.info(f"  Successfully converted: {stats['converted']} series")
        logger.info(f"  Skipped (no BIDS match): {stats['skipped']} series")
        logger.info(f"  Failed conversions/renames: {stats['failed']} series")
        logger.info("=" * 60)

# =============================================================================
# 5. SCRIPT ENTRYPOINT (Composition Root)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Converts DICOM series from a BIDS-like structure to NIfTI using dcm2niix.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", type=Path, help="Input directory with BIDS-like DICOM data.")
    parser.add_argument("output_dir", type=Path, help="Output directory for BIDS NIfTI data.")
    parser.add_argument("--dcm2niix_path", default='dcm2niix', help="Path to dcm2niix executable.")
    parser.add_argument("--log_file", type=Path, default=None, help="Path for the log file.")
    args = parser.parse_args()

    log_file = args.log_file or (args.output_dir / 'convert_dicom_to_nifti.log')
    setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("DICOM to NIfTI Converter Initializing")
    logger.info(f"Input:  {args.input_dir.resolve()}")
    logger.info(f"Output: {args.output_dir.resolve()}")
    logger.info(f"Log:    {log_file.resolve()}")
    logger.info("=" * 60)

    try:
        # --- 1. Assemble the components (Dependency Injection) ---
        task_finder = TaskFinder()
        bids_namer = BidsNamer()
        dcm2niix_runner = Dcm2niixRunner(executable_path=args.dcm2niix_path)
        file_renamer = BidsFileRenamer()
        
        # --- 2. Create the orchestrator and inject dependencies ---
        converter = DicomToNiftiConverter(
            task_finder=task_finder,
            namer=bids_namer,
            runner=dcm2niix_runner,
            renamer=file_renamer
        )
        
        # --- 3. Run the process through the simple facade ---
        converter.convert_all(args.input_dir, args.output_dir)

        logger.info("Script finished successfully.")
        sys.exit(0)

    except FileNotFoundError as e:
        logger.critical(f"A required file or executable was not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical("An unexpected critical error occurred.", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()