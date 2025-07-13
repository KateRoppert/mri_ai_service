"""
FSL BET Skull Stripping Script for BIDS-structured NIfTI files.

This script performs skull stripping using FSL BET on preprocessed images.
"""

import argparse
import logging
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List
from nipype.interfaces.fsl import BET, FSLCommand

# Setup logging
logger = logging.getLogger(__name__)


def setup_logging(log_file: Path, console_level: str = "INFO"):
    """Configure logging for the script."""
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
    )
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


class DependencyChecker:
    """Check for required dependencies."""
    
    @staticmethod
    def check_fsl() -> bool:
        """Check if FSL is available."""
        try:
            FSLCommand.check_fsl()
            logger.info("FSL dependency check: OK (found via Nipype)")
            return True
        except Exception:
            if shutil.which("bet"):
                logger.info("FSL dependency check: OK (found 'bet' in PATH)")
                return True
            else:
                logger.error("FSL dependency check: FAILED. 'bet' command not found")
                return False


class BIDSFileManager:
    """Manages BIDS file structure and discovery."""
    
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.files = self._discover_files()
    
    def _discover_files(self) -> List[Dict]:
        """Discover all NIfTI files in BIDS structure."""
        files = []
        
        # Pattern to find anat files
        patterns = [
            "sub-*/ses-*/anat/*.nii.gz",
            "sub-*/ses-*/anat/*.nii",
            "sub-*/anat/*.nii.gz",
            "sub-*/anat/*.nii"
        ]
        
        for pattern in patterns:
            for file_path in self.input_dir.glob(pattern):
                # Parse BIDS structure
                parts = file_path.relative_to(self.input_dir).parts
                
                subject = parts[0] if parts[0].startswith("sub-") else None
                session = parts[1] if len(parts) > 2 and parts[1].startswith("ses-") else None
                
                if subject:
                    files.append({
                        "path": file_path,
                        "subject": subject,
                        "session": session,
                        "filename": file_path.name,
                        "modality": self._get_modality(file_path)
                    })
        
        return files
    
    def _get_modality(self, file_path: Path) -> str:
        """Detect modality from filename."""
        filename = file_path.name.lower()
        
        if "flair" in filename:
            return "FLAIR"
        elif "t2w" in filename:
            return "T2"
        elif "t1w" in filename:
            if "ce" in filename or "gad" in filename:
                return "T1CE"
            else:
                return "T1"
        else:
            return "UNKNOWN"


class SkullStripper:
    """Handles skull stripping operations using FSL BET."""
    
    def __init__(self, frac: float = 0.5, robust: bool = True, **kwargs):
        self.frac = frac
        self.robust = robust
        self.extra_options = kwargs
    
    def strip_skull(self, input_file: Path, output_file: Path, 
                   save_mask: bool = True) -> Dict:
        """Perform skull stripping on a single file."""
        logger.info(f"Skull stripping: {input_file.name}")
        
        # Setup BET
        bet = BET()
        bet.inputs.in_file = str(input_file)
        bet.inputs.out_file = str(output_file)
        bet.inputs.frac = self.frac
        bet.inputs.robust = self.robust
        bet.inputs.mask = save_mask
        bet.inputs.output_type = 'NIFTI_GZ'
        
        # Add any extra options
        if self.extra_options.get('bet_options'):
            bet.inputs.args = self.extra_options['bet_options']
        
        try:
            # Run BET
            logger.debug(f"Executing command: {bet.cmdline}")
            result = bet.run()
            
            # Check for success
            if result.runtime.returncode != 0:
                raise RuntimeError(f"BET failed with exit code {result.runtime.returncode}")
            
            # Verify output exists
            if not output_file.exists():
                raise FileNotFoundError(f"BET output not found: {output_file}")
            
            # Prepare result
            result_dict = {
                "status": "success",
                "input": str(input_file),
                "output": str(output_file),
                "parameters": {
                    "frac": self.frac,
                    "robust": self.robust
                }
            }
            
            # Check for mask file
            if save_mask:
                mask_file = Path(str(output_file).replace('.nii.gz', '_mask.nii.gz'))
                if mask_file.exists():
                    result_dict["mask"] = str(mask_file)
                else:
                    logger.warning(f"Mask file not found: {mask_file}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Skull stripping failed for {input_file.name}: {e}")
            return {
                "status": "failed",
                "input": str(input_file),
                "error": str(e)
            }


class SkullStrippingPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, input_dir: Path, output_dir: Path, masks_dir: Path, **bet_params):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.masks_dir = masks_dir
        self.file_manager = BIDSFileManager(input_dir)
        self.skull_stripper = SkullStripper(**bet_params)
    
    def process_file(self, file_info: Dict) -> Dict:
        """Process a single file."""
        input_file = file_info["path"]
        
        # Create output paths maintaining BIDS structure
        if file_info["session"]:
            output_path = self.output_dir / file_info["subject"] / file_info["session"] / "anat"
            mask_path = self.masks_dir / file_info["subject"] / file_info["session"] / "anat"
        else:
            output_path = self.output_dir / file_info["subject"] / "anat"
            mask_path = self.masks_dir / file_info["subject"] / "anat"
        
        output_path.mkdir(parents=True, exist_ok=True)
        mask_path.mkdir(parents=True, exist_ok=True)
        
        # Define output file
        output_file = output_path / file_info["filename"]
        
        # Process file
        result = self.skull_stripper.strip_skull(
            input_file, output_file, save_mask=True
        )
        
        # Move mask if created
        if result.get("status") == "success" and result.get("mask"):
            mask_src = Path(result["mask"])
            mask_dst = mask_path / mask_src.name
            shutil.move(str(mask_src), str(mask_dst))
            result["mask"] = str(mask_dst)
        
        result.update({
            "subject": file_info["subject"],
            "session": file_info["session"],
            "modality": file_info["modality"]
        })
        
        return result
    
    def run(self) -> List[Dict]:
        """Run the skull stripping pipeline."""
        logger.info(f"Starting skull stripping pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Masks directory: {self.masks_dir}")
        logger.info(f"Found {len(self.file_manager.files)} files to process")
        
        results = []
        
        for file_info in self.file_manager.files:
            logger.info(f"Processing: {file_info['subject']}/{file_info['session'] or 'no-session'}/{file_info['filename']}")
            
            result = self.process_file(file_info)
            results.append(result)
        
        # Summary
        successful = sum(1 for r in results if r.get("status") == "success")
        logger.info(f"Skull stripping complete: {successful}/{len(results)} files processed successfully")
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FSL BET skull stripping for BIDS-structured NIfTI files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_dir", type=Path, help="Input BIDS directory")
    parser.add_argument("output_dir", type=Path, help="Output BIDS directory")
    parser.add_argument("--masks_dir", type=Path, help="Directory for brain masks")
    parser.add_argument("--frac", type=float, default=0.5, help="BET fractional intensity threshold")
    parser.add_argument("--robust", action="store_true", help="Use robust brain centre estimation")
    parser.add_argument("--bet_options", type=str, help="Additional BET command line options")
    parser.add_argument("--log_file", type=Path, help="Path to log file")
    parser.add_argument("--summary_file", type=Path, help="Path to save summary JSON")
    
    args = parser.parse_args()
    
    # Default masks directory
    if not args.masks_dir:
        args.masks_dir = args.output_dir.parent / "brain_masks"
    
    # Setup logging
    log_file = args.log_file or args.output_dir / "skull_stripping.log"
    setup_logging(log_file)
    
    try:
        # Check dependencies
        if not DependencyChecker.check_fsl():
            logger.error("FSL is not available. Cannot proceed.")
            sys.exit(1)
        
        # Create pipeline and run
        bet_params = {
            "frac": args.frac,
            "robust": args.robust
        }
        if args.bet_options:
            bet_params["bet_options"] = args.bet_options
        
        pipeline = SkullStrippingPipeline(
            args.input_dir,
            args.output_dir,
            args.masks_dir,
            **bet_params
        )
        
        results = pipeline.run()
        
        # Save summary if requested
        if args.summary_file:
            args.summary_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Exit with appropriate code
        if all(r.get("status") == "success" for r in results):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()