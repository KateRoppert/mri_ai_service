# -*- coding: utf-8 -*-
"""
A refactored script to run MRI segmentation by sending NIfTI files to a server.

This script demonstrates SOLID principles by separating responsibilities into distinct
classes:
- Config: Manages loading and accessing the YAML configuration.
- SegmentationInput: Encapsulates the logic for managing and preparing the
  four input modalities for segmentation.
- SegmentationClient: Handles all HTTP communication with the segmentation server.
- SegmentationRunner: The main orchestrator that uses the other components to
  execute the workflow.
"""
import argparse
import logging
import sys
from pathlib import Path
import yaml
import os
import requests
from dataclasses import dataclass, field
from contextlib import ExitStack

# --- Logger Setup ---
logger = logging.getLogger(__name__)

def setup_logging(log_file: Path, console_level: str = "INFO"):
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
        model = self._data.get('steps', {}).get('segmentation', {}).get('model_name')
        if not model:
            raise ValueError("'steps.segmentation.model_name' not found in config.")
        return model

@dataclass
class SegmentationInput:
    """Manages the set of input files for a single segmentation session."""
    t1: Path | None = None
    t1c: Path | None = None
    t2: Path | None = None
    flair: Path | None = None

    def _get_fallback_file(self) -> Path:
        """
        Finds the highest-priority valid file to use for filling empty slots.
        Priority: T1c > T1 > T2 > FLAIR.
        """
        modality_map = {"t1c": self.t1c, "t1": self.t1, "t2": self.t2, "flair": self.flair}
        for mod in ["t1c", "t1", "t2", "flair"]:
            file_path = modality_map[mod]
            if file_path and file_path.is_file():
                logger.info(f"'{mod}' is available. Selected '{file_path.name}' as the fallback file.")
                return file_path
            elif file_path:
                logger.warning(f"Path provided for '{mod}' but file not found: {file_path}")

        raise FileNotFoundError("No valid input files were provided for segmentation.")

    def prepare_for_server(self) -> dict[str, Path]:
        """
        Creates a dictionary of 4 files ready to be sent to the server,
        using the fallback logic for any missing files.
        """
        fallback_file = self._get_fallback_file()
        
        final_files = {
            "t1": self.t1 if self.t1 and self.t1.is_file() else fallback_file,
            "t1c": self.t1c if self.t1c and self.t1c.is_file() else fallback_file,
            "t2": self.t2 if self.t2 and self.t2.is_file() else fallback_file,
            "flair": self.flair if self.flair and self.flair.is_file() else fallback_file,
        }

        logger.info("Final file set prepared for server submission:")
        for mod, path in final_files.items():
            logger.info(f"  - {mod.upper()}: {path.name}")
            
        return final_files

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
        logger.info(f"Sending segmentation request to: {inference_url}")

        try:
            # ExitStack cleanly manages opening multiple files
            with ExitStack() as stack:
                files_multipart = {
                    f't1_{files_to_send["t1"].name}': (files_to_send["t1"].name, stack.enter_context(open(files_to_send["t1"], 'rb'))),
                    f't1c_{files_to_send["t1c"].name}': (files_to_send["t1c"].name, stack.enter_context(open(files_to_send["t1c"], 'rb'))),
                    f't2_{files_to_send["t2"].name}': (files_to_send["t2"].name, stack.enter_context(open(files_to_send["t2"], 'rb'))),
                    f't2flair_{files_to_send["flair"].name}': (files_to_send["flair"].name, stack.enter_context(open(files_to_send["flair"], 'rb'))),
                }

                logger.info("Uploading files for segmentation...")
                response = requests.post(inference_url, files=files_multipart, timeout=self.timeout)
                response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

                logger.info("Received response from server. Saving segmentation mask.")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, 'wb') as f_mask:
                    f_mask.write(response.content)
                
                logger.info(f"Segmentation mask successfully saved to: {output_path}")
                return True

        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds.")
            return False
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to the server at {self.server_url}. Error: {e}")
            return False
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP Error {e.response.status_code}: {e.response.reason}")
            logger.error(f"Server response: {e.response.text[:500]}")
            return False
        except Exception as e:
            logger.exception(f"An unexpected error occurred during segmentation: {e}")
            return False

class SegmentationRunner:
    """Orchestrates the entire segmentation workflow."""
    def __init__(self, config: Config, args: argparse.Namespace):
        self.config = config
        self.args = args

    def run(self):
        """Executes the full segmentation process."""
        logger.info("=" * 50)
        logger.info("Starting MRI Segmentation Script")
        try:
            segmentation_input = SegmentationInput(
                t1=Path(self.args.input_t1) if self.args.input_t1 else None,
                t1c=Path(self.args.input_t1ce) if self.args.input_t1ce else None,
                t2=Path(self.args.input_t2) if self.args.input_t2 else None,
                flair=Path(self.args.input_flair) if self.args.input_flair else None,
            )
            
            files_to_process = segmentation_input.prepare_for_server()
            
            client = SegmentationClient(server_url=self.config.get_server_url())
            
            success = client.segment(
                files_to_send=files_to_process,
                model_name=self.config.get_model_name(),
                client_id=self.args.client_id,
                output_path=Path(self.args.output_mask)
            )

            if success:
                logger.info("Segmentation workflow completed successfully.")
                return True
            else:
                logger.error("Segmentation workflow failed.")
                return False

        except FileNotFoundError as e:
            logger.error(f"Critical error: {e}")
            return False
        except Exception:
            logger.exception("An unexpected critical error stopped the workflow.")
            return False
        finally:
            logger.info("=" * 50)


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run server-based MRI segmentation with dynamic file selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_t1", help="Path to the preprocessed T1 NIfTI file.")
    parser.add_argument("--input_t1ce", help="Path to the preprocessed T1c NIfTI file.")
    parser.add_argument("--input_t2", help="Path to the preprocessed T2 NIfTI file.")
    parser.add_argument("--input_flair", help="Path to the preprocessed T2-FLAIR NIfTI file.")
    parser.add_argument("--output_mask", required=True, help="Path to save the output segmentation mask.")
    parser.add_argument("--config", required=True, help="Path to the main pipeline YAML config file.")
    parser.add_argument("--client_id", required=True, help="A unique identifier for this client/run.")
    parser.add_argument("--log_file", help="Path to the log file. Defaults to a file next to the output mask.")
    parser.add_argument("--console_log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Console logging level.")
    args = parser.parse_args()

    # Determine log file path
    output_mask_path = Path(args.output_mask)
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = output_mask_path.parent / f"{output_mask_path.stem.replace('_segmask', '')}_segmentation.log"

    setup_logging(log_file, args.console_log_level)

    try:
        app_config = Config(Path(args.config))
        runner = SegmentationRunner(config=app_config, args=args)
        is_successful = runner.run()
        sys.exit(0 if is_successful else 1)
        
    except (FileNotFoundError, ValueError) as e:
        logger.critical(f"Configuration error: {e}")
        sys.exit(1)
    except Exception:
        logger.critical("A fatal exception occurred.", exc_info=True)
        sys.exit(1)