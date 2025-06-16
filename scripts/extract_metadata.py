import os
import pydicom
import json
import argparse
import logging
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Generator

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
# 2. CORE INTERFACES (Strategy Pattern Contracts)
# =============================================================================

class IMetadataCleaner(ABC):
    """Interface for a strategy that cleans metadata dictionaries."""
    @abstractmethod
    def clean(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans a metadata dictionary, returning the cleaned version."""
        pass

class IMetadataWriter(ABC):
    """Interface for a strategy that writes metadata to a destination."""
    @abstractmethod
    def write(self, metadata: Dict[str, Any], source_path: Path, output_dir: Path):
        """Writes the metadata to a file or other destination."""
        pass

# =============================================================================
# 3. CONCRETE COMPONENTS (Implementations of Responsibilities)
# =============================================================================

class FileFinder:
    """Finds files in a directory based on a given extension."""
    @staticmethod
    def find(directory: Path, extension: str) -> Generator[Path, None, None]:
        """Yields all files in a directory with a specific extension."""
        if not directory.is_dir():
            logger.error(f"Input path is not a directory: {directory}")
            return
        
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(extension):
                    yield Path(root) / file

class DicomReader:
    """Reads a DICOM file's header."""
    def read(self, file_path: Path) -> Dict[str, Any] | None:
        """
        Reads a DICOM file header and converts it to a dictionary.
        Returns None if the file is not a valid DICOM.
        """
        try:
            # stop_before_pixels=True is critical for performance
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            return ds.to_json_dict()
        except pydicom.errors.InvalidDicomError:
            logger.warning(f"Skipping invalid DICOM file: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to read DICOM file {file_path}: {e}", exc_info=True)
            return None

class JsonSafeCleaner(IMetadataCleaner):
    """A concrete cleaning strategy to make metadata JSON-serializable."""
    def clean(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively cleans a dictionary from a pydicom.to_json_dict() call."""
        cleaned_meta = {}
        for tag, element in metadata.items():
            if isinstance(element, dict) and 'vr' in element:
                cleaned_meta[tag] = self._clean_element(element)
            else:
                # Handle unexpected structures gracefully
                logger.warning(f"Tag {tag} has an unexpected structure. Attempting general clean.")
                cleaned_meta[tag] = self._clean_value(element)
        return cleaned_meta

    def _clean_element(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """Cleans a single DICOM element dictionary."""
        vr = element.get('vr')
        value = element.get('Value')

        if vr == 'SQ': # Handle sequences recursively
            cleaned_value = [self.clean(item) for item in value] if value else []
        elif value is not None:
            cleaned_value = self._clean_value(value)
        else:
            cleaned_value = None
        
        return {'vr': vr, 'Value': cleaned_value}

    def _clean_value(self, value: Any) -> Any:
        """Recursively cleans a value to be JSON-safe."""
        if isinstance(value, list):
            return [self._clean_value(item) for item in value]
        if isinstance(value, dict):
            return {k: self._clean_value(v) for k, v in value.items()}
        if isinstance(value, bytes):
            return "<binary_data_removed>"
        
        # Attempt to return as-is, but fallback to string representation
        try:
            json.dumps(value)
            return value
        except TypeError:
            logger.debug(f"Unserializable type {type(value)} converted to string.")
            return str(value)

class JsonWriter(IMetadataWriter):
    """A concrete writing strategy that saves metadata as a .json file."""
    def write(self, metadata: Dict[str, Any], source_path: Path, output_dir: Path):
        """Saves the metadata dictionary to a JSON file, mirroring the source structure."""
        try:
            relative_path = source_path.relative_to(self.input_root)
            json_filename = source_path.stem + '_meta.json'
            output_path = output_dir / relative_path.parent / json_filename

            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            logger.debug(f"  Successfully wrote metadata to {output_path}")

        except Exception as e:
            logger.error(f"  Failed to write JSON for {source_path}: {e}", exc_info=True)

    def set_input_root(self, input_root: Path):
        """Sets the root directory to calculate relative paths correctly."""
        self.input_root = input_root

# =============================================================================
# 4. ORCHESTRATOR / FACADE
# =============================================================================

class MetadataExtractor:
    """Orchestrates the metadata extraction process using injected components."""
    def __init__(
        self,
        reader: DicomReader,
        cleaner: IMetadataCleaner,
        writer: IMetadataWriter
    ):
        self.reader = reader
        self.cleaner = cleaner
        self.writer = writer
    
    def process_directory(self, input_dir: Path, output_dir: Path):
        """
        Processes all DICOM files in a directory using the configured components.
        This is the main Facade method.
        """
        logger.info(f"Starting metadata extraction from '{input_dir}' to '{output_dir}'.")
        
        # A small hack for the writer to know the root for relative path calculation.
        # A more advanced solution might involve passing a "context" object.
        if hasattr(self.writer, 'set_input_root'):
            self.writer.set_input_root(input_dir)
            
        processed_count = 0
        failed_count = 0

        for file_path in FileFinder.find(input_dir, extension=".dcm"):
            logger.debug(f"Processing file: {file_path}")
            
            raw_meta = self.reader.read(file_path)
            if raw_meta is None:
                failed_count += 1
                continue
                
            cleaned_meta = self.cleaner.clean(raw_meta)
            self.writer.write(cleaned_meta, file_path, output_dir)
            
            processed_count += 1

        logger.info("Extraction complete.")
        logger.info(f"  Successfully processed: {processed_count} files")
        if failed_count > 0:
            logger.warning(f"  Failed to process: {failed_count} files")


# =============================================================================
# 5. SCRIPT ENTRYPOINT (Composition Root)
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extracts and cleans DICOM metadata, saving it to JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", type=Path, help="Input directory with DICOM files.")
    parser.add_argument("output_dir", type=Path, help="Output directory for JSON metadata.")
    parser.add_argument("--log_file", type=Path, default=None, help="Path to log file.")
    args = parser.parse_args()

    log_file = args.log_file or (args.output_dir / 'extract_metadata.log')
    setup_logging(log_file)

    logger.info("="*60)
    logger.info("DICOM Metadata Extractor Initializing")
    logger.info(f"Input:  {args.input_dir.resolve()}")
    logger.info(f"Output: {args.output_dir.resolve()}")
    logger.info(f"Log:    {log_file.resolve()}")
    logger.info("="*60)

    try:
        # --- 1. Assemble the components (Dependency Injection) ---
        # Here we choose our concrete strategies.
        dicom_reader = DicomReader()
        metadata_cleaner = JsonSafeCleaner()
        metadata_writer = JsonWriter()

        # --- 2. Create the orchestrator and inject dependencies ---
        extractor = MetadataExtractor(
            reader=dicom_reader,
            cleaner=metadata_cleaner,
            writer=metadata_writer
        )

        # --- 3. Run the process through the simple facade ---
        extractor.process_directory(args.input_dir, args.output_dir)

        logger.info("Script finished successfully.")
        sys.exit(0)

    except Exception as e:
        logger.critical("An unexpected critical error occurred.", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()