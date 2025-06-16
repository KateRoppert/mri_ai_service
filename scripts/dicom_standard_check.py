# -*- coding: utf-8 -*-
import os
import subprocess
import argparse
import logging
import sys
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

# --- Global Logger Setup ---
logger = logging.getLogger(__name__)

# =============================================================================
# 1. DATA CLASSES (for clear, structured data)
# =============================================================================

@dataclass
class Config:
    """Holds all script configuration."""
    input_dir: Path
    output_dir: Path
    dciodvfy_path: str
    log_file: Path
    timeout: int = 60

@dataclass
class ValidationResult:
    """Represents the result of a single DICOM file validation."""
    source_file: Path
    report_file: Path
    success: bool
    errors: int = 0
    warnings: int = 0
    report_content: str = ""
    run_error: bool = False

# =============================================================================
# 2. COMPONENT CLASSES (adhering to Single Responsibility Principle)
# =============================================================================

class DciodvfyRunner:
    """
    Responsible for finding and running the dciodvfy executable.
    This class's only job is to interact with the external process.
    """
    def __init__(self, executable_path: str, timeout: int):
        self.timeout = timeout
        self.executable_path = self._find_executable(executable_path)

    def _find_executable(self, path: str) -> str:
        """Validates and finds the full path to the executable."""
        found_path = shutil.which(path)
        if found_path is None:
            msg = f"dciodvfy executable not found at '{path}' or in system PATH."
            logger.error(msg)
            raise FileNotFoundError(msg)
        logger.debug(f"Found dciodvfy executable at: {found_path}")
        return found_path

    def run_on_file(self, file_path: Path) -> Tuple[bool, str, str]:
        """
        Runs dciodvfy on a single file.

        Returns:
            A tuple of (success, stdout, stderr).
            'success' is False if the process fails to run (e.g., timeout).
        """
        cmd = [self.executable_path, str(file_path)]
        logger.debug(f"  Executing command: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False  # dciodvfy returns non-zero on warnings, so we can't check
            )
            return True, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            msg = f"Timeout error ({self.timeout}s) while processing {file_path}"
            logger.error(f"  {msg}")
            return False, "", msg
        except Exception as e:
            msg = f"Unexpected error running dciodvfy on {file_path}: {e}"
            logger.error(f"  {msg}", exc_info=True)
            return False, "", msg


class ReportManager:
    """
    Responsible for parsing, generating, and saving all reports.
    This class handles all string formatting and file I/O for reports.
    """
    @staticmethod
    def parse_output(output: str) -> Tuple[int, int]:
        """Parses stdout/stderr to count errors and warnings."""
        errors = 0
        warnings = 0
        for line in output.splitlines():
            clean_line = line.strip()
            if clean_line.startswith("Error"):
                errors += 1
            elif clean_line.startswith("Warning"):
                warnings += 1
        return errors, warnings

    @staticmethod
    def generate_file_report_content(
        source_file: Path,
        stdout: str,
        stderr: str,
        run_successful: bool
    ) -> str:
        """Generates the detailed text content for a single file's report."""
        if not run_successful:
            # stderr contains the error message from the runner
            return f"--- FAILED to run dciodvfy on {source_file} ---\n\n{stderr}"

        # Note: Including the full path can be an information security risk if reports are shared.
        # For internal use, it is standard and useful for debugging.
        return (
            f"--- dciodvfy Report for: {source_file} ---\n"
            f"--- Command: dciodvfy {source_file} ---\n\n"
            "--- STDOUT ---\n"
            f"{stdout or '[No stdout output]'}\n\n"
            "--- STDERR ---\n"
            f"{stderr or '[No stderr output]'}\n\n"
            "--- End of Report ---"
        )

    def generate_directory_summary(
        self,
        results: List[ValidationResult],
        relative_dir: Path
    ) -> str:
        """Generates the summary.txt content for a directory."""
        total_files = len(results)
        total_errors = sum(r.errors for r in results)
        total_warnings = sum(r.warnings for r in results)
        files_with_errors = sum(1 for r in results if r.errors > 0)
        files_with_warnings = sum(1 for r in results if r.warnings > 0 and r.errors == 0)
        files_failed = sum(1 for r in results if r.run_error)

        summary_lines = [
            f"DICOM Standard Validation Summary for Directory: {relative_dir}",
            "=" * (45 + len(str(relative_dir))),
            f"Total files processed: {total_files}",
            f"  Files with errors: {files_with_errors} (Total {total_errors} errors)",
            f"  Files with warnings only: {files_with_warnings} (Total {total_warnings} warnings)",
            f"  Files that failed to run: {files_failed}" if files_failed > 0 else "",
            "\n--- File Details ---"
        ]

        for res in sorted(results, key=lambda r: r.source_file.name):
            if res.run_error:
                status = f"❌ RUN-ERROR (see {res.report_file.name})"
            elif res.errors > 0:
                status = f"❗ ERROR ({res.errors} errors, {res.warnings} warnings)"
            elif res.warnings > 0:
                status = f"⚠️ WARNING ({res.warnings} warnings)"
            else:
                status = "✅ OK"
            summary_lines.append(f"  {res.source_file.name}: {status}")

        return "\n".join(filter(None, summary_lines))

    def save_report(self, content: str, path: Path):
        """Saves content to a file with error handling."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
            logger.debug(f"    Report saved to: {path}")
        except OSError as e:
            logger.error(f"    Failed to write report file {path}: {e}")

# =============================================================================
# 3. ORCHESTRATOR CLASS (using Dependency Injection)
# =============================================================================

class DicomValidator:
    """
    Orchestrates the validation process by coordinating the runner and reporter.
    """
    def __init__(self, config: Config, runner: DciodvfyRunner, reporter: ReportManager):
        self.config = config
        self.runner = runner
        self.reporter = reporter
        self._validate_paths()

    def _validate_paths(self):
        """Ensures input directory exists and output can be created."""
        if not self.config.input_dir.is_dir():
            msg = f"Input directory not found: {self.config.input_dir}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        
        try:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            msg = f"Cannot create output directory {self.config.output_dir}: {e}"
            logger.error(msg)
            raise

    def validate_all(self):
        """Main method to start the validation process for all files."""
        logger.info(f"Starting DICOM validation in '{self.config.input_dir}'")
        logger.info(f"Reports will be saved in '{self.config.output_dir}'")
        
        for root, _, files in os.walk(self.config.input_dir):
            root_path = Path(root)
            dicom_files = [f for f in files if f.lower().endswith(".dcm")]

            if not dicom_files:
                continue
            
            logger.info(f"Processing directory: {root_path.relative_to(self.config.input_dir)}")
            dir_results: List[ValidationResult] = []

            for filename in dicom_files:
                file_path = root_path / filename
                result = self._process_one_file(file_path)
                dir_results.append(result)
            
            self._create_directory_summary(root_path, dir_results)

        logger.info("DICOM validation process completed.")

    def _process_one_file(self, file_path: Path) -> ValidationResult:
        """Processes a single DICOM file."""
        logger.debug(f"  Validating file: {file_path}")
        
        relative_path = file_path.relative_to(self.config.input_dir)
        report_file = self.config.output_dir / relative_path.with_name(f"{file_path.stem}_report.txt")

        success, stdout, stderr = self.runner.run_on_file(file_path)
        
        report_content = self.reporter.generate_file_report_content(file_path, stdout, stderr, success)
        self.reporter.save_report(report_content, report_file)
        
        if not success:
            return ValidationResult(
                source_file=file_path,
                report_file=report_file,
                success=False,
                report_content=report_content,
                run_error=True
            )
        
        errors, warnings = self.reporter.parse_output(stdout + stderr)
        return ValidationResult(
            source_file=file_path,
            report_file=report_file,
            success=(errors == 0),
            errors=errors,
            warnings=warnings,
            report_content=report_content,
            run_error=False
        )

    def _create_directory_summary(self, dir_path: Path, results: List[ValidationResult]):
        """Creates and saves the summary report for a single directory."""
        if not results:
            return
            
        relative_dir = dir_path.relative_to(self.config.input_dir)
        summary_content = self.reporter.generate_directory_summary(results, relative_dir)
        
        summary_file = self.config.output_dir / relative_dir / "summary.txt"
        logger.debug(f"  Writing directory summary: {summary_file}")
        self.reporter.save_report(summary_content, summary_file)

# =============================================================================
# 4. SCRIPT ENTRYPOINT
# =============================================================================

def setup_logging(log_file: Path):
    """Configures logging to both console (INFO) and file (DEBUG)."""
    logger.setLevel(logging.DEBUG)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    log_file.parent.mkdir(exist_ok=True, parents=True)
    
    # File handler (DEBUG)
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(module)s.%(funcName)s] - %(message)s'))
    logger.addHandler(fh)
    
    # Console handler (INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

def main():
    """Parses arguments and runs the validation process."""
    parser = argparse.ArgumentParser(
        description="Run dciodvfy to validate DICOM files against the standard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input directory with DICOM data (e.g., in BIDS format)."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory to save dciodvfy reports."
    )
    parser.add_argument(
        "--dciodvfy_path",
        default='dciodvfy',
        help="Path or name of the dciodvfy executable."
    )
    parser.add_argument(
        "--log_file",
        type=Path,
        default=None,
        help="Path to log file. Defaults to 'dicom_validation.log' inside the output directory."
    )
    args = parser.parse_args()

    log_file = args.log_file or (args.output_dir / 'dicom_validation.log')
    
    try:
        setup_logging(log_file)
    except Exception as e:
        print(f"FATAL: Could not set up logging at '{log_file}': {e}", file=sys.stderr)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("DICOM Standard Check Initializing")
    logger.info(f"Input Directory:  {args.input_dir.resolve()}")
    logger.info(f"Output Directory: {args.output_dir.resolve()}")
    logger.info(f"Log File:         {log_file.resolve()}")
    logger.info("=" * 60)

    try:
        # 1. Create dependencies
        runner = DciodvfyRunner(executable_path=args.dciodvfy_path, timeout=60)
        reporter = ReportManager()
        
        # 2. Assemble the main object (Dependency Injection)
        config = Config(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dciodvfy_path=args.dciodvfy_path,
            log_file=log_file
        )
        validator = DicomValidator(config, runner, reporter)

        # 3. Run the process
        validator.validate_all()

        logger.info("Script finished successfully.")
        sys.exit(0)

    except FileNotFoundError as e:
        logger.critical(f"A required file or directory was not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.critical("An unexpected error occurred.", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main() 