from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# --- Global Logger Setup ---
# We configure the logger once, and other modules/classes can get it by name.
logger = logging.getLogger(__name__)


def setup_logging(log_file_path: Path | str | None = None) -> None:
    """
    Configures logging to stream to console (INFO) and optionally to a file (DEBUG).

    This function is idempotent; it clears existing handlers before adding new ones.

    Args:
        log_file_path: The path to the file where logs should be saved.
    """
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Console Handler (always on, at INFO level)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (optional)
    if log_file_path:
        try:
            log_path = Path(log_file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.debug(f"File logging configured at: {log_path.resolve()}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to configure file logging at '{log_file_path}': {e}", exc_info=False)


# --- Data Structures ---

@dataclass(frozen=True)
class ValidationResult:
    """
    A simple, immutable data structure to hold the results of a validation run.
    This avoids passing multiple loose variables (bool, str, str, int) around.
    """
    is_success: bool
    return_code: int
    output: str
    command: str


# --- Core Components (Abstractions and Implementations) ---

class IDescriptorGenerator(ABC):
    """
    Interface (Abstract Base Class) for generating a dataset_description.json file.

    This defines the contract for any class that creates this file, allowing
    for different generation strategies (e.g., from a template, from arguments).
    This follows the Open/Closed Principle.
    """
    @abstractmethod
    def create(self, bids_dir: Path) -> bool:
        """
        Creates the dataset_description.json file in the given directory.

        Args:
            bids_dir: The root directory of the BIDS dataset.

        Returns:
            True if the file was created successfully, False otherwise.
        """
        pass


class DefaultDescriptorGenerator(IDescriptorGenerator):
    """
    A concrete implementation that creates a default dataset_description.json.
    """
    def get_content(self) -> dict:
        """Provides the default content for the JSON file. Easy to override."""
        return {
            "Name": "Brain Tumour MRI Dataset (Processed)",
            "BIDSVersion": "1.8.0",
            "License": "CC0",
            "Authors": ["Your Name/Lab Name"],
            "DatasetDOI": "10.1234/example.doi"
        }

    def create(self, bids_dir: Path) -> bool:
        """Creates the file with default content."""
        logger.info(f"Ensuring dataset_description.json exists in '{bids_dir}'")
        try:
            bids_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Could not create BIDS directory {bids_dir}: {e}")
            return False

        description_content = self.get_content()
        output_file = bids_dir / "dataset_description.json"
        
        logger.debug(f"Content for {output_file}:\n{json.dumps(description_content, indent=2)}")

        try:
            with output_file.open('w', encoding='utf-8') as f:
                json.dump(description_content, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully created/updated {output_file}")
            return True
        except OSError as e:
            logger.error(f"Failed to write to {output_file}: {e}")
            return False


class IValidator(ABC):
    """
    Interface (Abstract Base Class) for a dataset validator.

    This defines the contract for any validation tool, allowing us to swap out
    `bids-validator` for another tool in the future without changing the pipeline.
    This is a prime example of the Strategy Pattern and Open/Closed Principle.
    """
    @abstractmethod
    def validate(self, bids_dir: Path) -> ValidationResult:
        """
        Runs validation on the given directory.

        Args:
            bids_dir: The directory to validate.

        Returns:
            A ValidationResult object containing the outcome.
        """
        pass


class BidsValidator(IValidator):
    """
    A concrete validator that uses the external 'bids-validator' CLI tool.
    This class is responsible ONLY for finding and running the validator.
    """
    def __init__(self, validator_path: str = 'bids-validator', timeout: int = 600):
        self.validator_path_str = validator_path
        self.timeout = timeout
        self.executable_path: str | None = None

    def _find_executable(self) -> bool:
        """Locates the bids-validator executable. Caches the result."""
        if self.executable_path:
            return True
        
        found_path = shutil.which(self.validator_path_str)
        if not found_path:
            logger.error(f"Executable '{self.validator_path_str}' not found in PATH.")
            logger.error("Please install it, e.g., with 'npm install -g bids-validator'")
            return False
        
        self.executable_path = found_path
        logger.info(f"Found bids-validator executable at: {self.executable_path}")
        return True

    def validate(self, bids_dir: Path) -> ValidationResult:
        """Executes bids-validator and returns the results."""
        if not self._find_executable():
            raise FileNotFoundError(f"Validator executable '{self.validator_path_str}' not found.")
        
        if not bids_dir.is_dir():
            raise FileNotFoundError(f"BIDS directory does not exist: {bids_dir}")

        # Non-None assertion is safe due to the check above
        cmd = [self.executable_path, str(bids_dir)]
        command_str = ' '.join(cmd)
        logger.info(f"Running validation command: {command_str}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,  # We handle the return code ourselves
                timeout=self.timeout
            )
            full_output = (result.stdout or "") + (result.stderr or "")
            # bids-validator exits with 0 only on zero errors and zero warnings.
            # For our purpose, success means the script ran and we got a result.
            # The *interpretation* of the result is handled by the pipeline.
            return ValidationResult(
                is_success=(result.returncode == 0),
                return_code=result.returncode,
                output=full_output.strip(),
                command=command_str
            )
        except subprocess.TimeoutExpired:
            logger.error(f"Validation command timed out after {self.timeout} seconds.")
            return ValidationResult(
                is_success=False, return_code=-1, 
                output="Error: Validation process timed out.", command=command_str
            )
        except Exception as e:
            logger.exception("An unexpected error occurred during validation.")
            return ValidationResult(
                is_success=False, return_code=-1, 
                output=f"An unexpected Python error occurred: {e}", command=command_str
            )


# --- Orchestrator / Facade ---

class BidsValidationPipeline:
    """
    Orchestrates the validation process. Acts as a Facade.

    This class coordinates the different components (descriptor generator, validator)
    and handles high-level logic like logging results and saving reports.
    It relies on injected dependencies, making it flexible and testable.
    """
    def __init__(self,
                 bids_dir: Path,
                 output_dir: Path,
                 descriptor_generator: IDescriptorGenerator,
                 validator: IValidator):
        self.bids_dir = bids_dir
        self.output_dir = output_dir
        self.descriptor_generator = descriptor_generator
        self.validator = validator

    def _save_report(self, result: ValidationResult) -> None:
        """Saves the detailed validation output to a report file."""
        report_file = self.output_dir / "bids_validator_report.txt"
        logger.debug(f"Attempting to save validation report to {report_file}")
        
        report_content = (
            f"--- BIDS Validator Report for {self.bids_dir.resolve()} ---\n"
            f"Command: {result.command}\n"
            f"Return Code: {result.return_code}\n\n"
            f"--- Validator Output ---\n"
            f"{result.output or '[No output received]'}\n"
            f"--- End Report ---"
        )
        
        try:
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(report_content, encoding='utf-8')
            logger.info(f"Validation report saved to: {report_file}")
        except OSError as e:
            logger.error(f"Failed to write validation report to {report_file}: {e}")

    def run(self) -> bool:
        """
        Executes the full validation pipeline.

        1. Creates the dataset_description.json file.
        2. Runs the BIDS validator.
        3. Logs the outcome and saves a report.

        Returns:
            True if the BIDS dataset is valid (no errors or warnings), False otherwise.
        """
        logger.info("="*50)
        logger.info("Starting BIDS Validation Pipeline")
        logger.info(f"  BIDS Directory: {self.bids_dir.resolve()}")
        logger.info(f"  Output Directory: {self.output_dir.resolve()}")
        logger.info("="*50)

        # Step 1: Create dataset_description.json
        if not self.descriptor_generator.create(self.bids_dir):
            logger.error("Failed to create dataset_description.json. Halting pipeline.")
            return False

        # Step 2: Run validation
        validation_result = self.validator.validate(self.bids_dir)

        # Step 3: Interpret and report results
        self._save_report(validation_result)

        if validation_result.is_success:
            logger.info("BIDS validation completed successfully (Exit Code 0). Dataset is valid.")
            return True
        else:
            logger.error(f"BIDS validation finished with issues (Exit Code {validation_result.return_code}).")
            # Heuristic to distinguish warnings from errors for better logging
            output_lower = validation_result.output.lower()
            if "error" in output_lower:
                logger.error("Validation failed with one or more ERRORS.")
            elif "warning" in output_lower:
                logger.warning("Validation passed with one or more WARNINGS.")
            else:
                logger.warning("Validation failed for an unknown reason. Check the report.")
            
            # We return False because any non-zero exit code indicates the dataset
            # is not perfectly BIDS compliant.
            return False


# --- Main execution block ---

def main(argv: Sequence[str] | None = None) -> int:
    """
    Parses command-line arguments and runs the validation pipeline.

    Returns:
        0 on success (dataset is valid), 1 on failure or error.
    """
    parser = argparse.ArgumentParser(
        description="Creates dataset_description.json and runs bids-validator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "bids_dir", type=Path,
        help="Root directory of the BIDS dataset."
    )
    parser.add_argument(
        "output_dir", type=Path,
        help="Directory to save logs and validation reports."
    )
    parser.add_argument(
        "--validator_path", default='bids-validator',
        help="Path or name of the bids-validator executable."
    )
    parser.add_argument(
        "--log_file", default=None,
        help="Custom path for the log file. Defaults to 'bids_validation.log' inside --output_dir."
    )
    args = parser.parse_args(argv)

    # --- Setup Logging ---
    log_file = args.log_file or (args.output_dir / 'bids_validation.log')
    setup_logging(log_file)

    try:
        # --- Dependency Injection: Create and configure components ---
        descriptor_generator = DefaultDescriptorGenerator()
        validator = BidsValidator(validator_path=args.validator_path)
        
        pipeline = BidsValidationPipeline(
            bids_dir=args.bids_dir,
            output_dir=args.output_dir,
            descriptor_generator=descriptor_generator,
            validator=validator
        )
        
        # --- Run the pipeline ---
        is_valid = pipeline.run()
        
        if is_valid:
            logger.info("Pipeline finished. The BIDS dataset is valid.")
            return 0
        else:
            logger.warning("Pipeline finished. The BIDS dataset has validation issues.")
            return 1 # Signal failure to calling scripts/CI

    except FileNotFoundError as e:
        logger.error(f"A required file or directory was not found: {e}", exc_info=True)
        return 1
    except Exception:
        logger.exception("An unhandled exception occurred in the main pipeline.")
        return 1


if __name__ == '__main__':
    sys.exit(main())