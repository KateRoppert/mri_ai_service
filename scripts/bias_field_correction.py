"""
Bias Field Correction Script for BIDS-structured NIfTI files.

Implements N4 bias field correction with configurable parameters.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List
import SimpleITK as sitk

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


class BiasFieldCorrector:
    """Handles N4 bias field correction."""
    
    def __init__(self, shrink_factor: int = 4, 
                 n_iterations: List[int] = None,
                 convergence_threshold: float = 0.001,
                 save_bias_field: bool = True):
        self.shrink_factor = shrink_factor
        self.n_iterations = n_iterations or [50, 50, 50, 50]
        self.convergence_threshold = convergence_threshold
        self.save_bias_field = save_bias_field
    
    def correct_bias(self, input_file: Path, output_file: Path, 
                    bias_field_file: Path = None) -> Dict:
        """Perform bias field correction on a single file."""
        logger.info(f"Correcting bias field: {input_file.name}")
        
        try:
            # Read image
            image = sitk.ReadImage(str(input_file), sitk.sitkFloat32)
            
            # Create mask using Otsu thresholding
            mask = sitk.OtsuThreshold(image, 0, 1)
            
            # Shrink for faster processing
            shrunk_image = sitk.Shrink(image, [self.shrink_factor] * image.GetDimension())
            shrunk_mask = sitk.Shrink(mask, [self.shrink_factor] * image.GetDimension())
            
            # Configure N4 bias field correction
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            
            # Adjust iterations to match shrink factor
            n_iterations = self.n_iterations
            if len(n_iterations) != self.shrink_factor:
                n_iterations = (n_iterations * self.shrink_factor)[:self.shrink_factor]
            
            corrector.SetMaximumNumberOfIterations(n_iterations)
            corrector.SetConvergenceThreshold(self.convergence_threshold)
            
            # Execute correction
            corrector.Execute(shrunk_image, shrunk_mask)
            
            # Get bias field
            log_bias_field = corrector.GetLogBiasFieldAsImage(image)
            bias_field = sitk.Exp(log_bias_field)
            
            # Apply correction
            corrected_image = image / bias_field
            
            # Save corrected image
            output_file.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(corrected_image, str(output_file))
            
            result = {
                "status": "success",
                "input": str(input_file),
                "output": str(output_file),
                "parameters": {
                    "shrink_factor": self.shrink_factor,
                    "n_iterations": n_iterations,
                    "convergence_threshold": self.convergence_threshold
                }
            }
            
            # Save bias field if requested
            if self.save_bias_field and bias_field_file:
                bias_field_file.parent.mkdir(parents=True, exist_ok=True)
                sitk.WriteImage(bias_field, str(bias_field_file))
                result["bias_field"] = str(bias_field_file)
                logger.debug(f"Saved bias field to: {bias_field_file}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to correct bias field for {input_file.name}: {e}")
            return {
                "status": "failed",
                "input": str(input_file),
                "error": str(e)
            }


class BiasFieldCorrectionPipeline:
    """Main pipeline for bias field correction."""
    
    def __init__(self, corrector: BiasFieldCorrector):
        self.corrector = corrector
    
    def process_directory(self, input_dir: Path, output_dir: Path, 
                         bias_fields_dir: Path = None) -> List[Dict]:
        """Process all files in directory maintaining BIDS structure."""
        results = []
        
        # Find all NIfTI files
        patterns = [
            "sub-*/ses-*/anat/*.nii.gz",
            "sub-*/ses-*/anat/*.nii",
            "sub-*/anat/*.nii.gz",
            "sub-*/anat/*.nii"
        ]
        
        files = []
        for pattern in patterns:
            files.extend(input_dir.glob(pattern))
        
        logger.info(f"Found {len(files)} files to process")
        
        for input_file in files:
            # Maintain directory structure
            relative_path = input_file.relative_to(input_dir)
            output_file = output_dir / relative_path
            
            # Bias field path
            bias_field_file = None
            if bias_fields_dir and self.corrector.save_bias_field:
                bias_field_path = bias_fields_dir / relative_path.parent
                bias_field_file = bias_field_path / f"{input_file.stem}_bias_field.nii.gz"
            
            result = self.corrector.correct_bias(
                input_file, output_file, bias_field_file
            )
            results.append(result)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="N4 bias field correction for BIDS-structured NIfTI files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_dir", type=Path, help="Input BIDS directory")
    parser.add_argument("output_dir", type=Path, help="Output BIDS directory")
    parser.add_argument("--method", default="N4BiasFieldCorrection",
                       help="Bias correction method (currently only N4)")
    parser.add_argument("--shrink_factor", type=int, default=4,
                       help="Shrink factor for faster processing")
    parser.add_argument("--n_iterations", type=int, nargs='+', 
                       default=[50, 50, 50, 50],
                       help="Number of iterations per level")
    parser.add_argument("--convergence_threshold", type=float, default=0.001,
                       help="Convergence threshold")
    parser.add_argument("--bias_fields_dir", type=Path,
                       help="Directory to save bias fields")
    parser.add_argument("--no_save_bias_field", action="store_true",
                       help="Don't save bias field images")
    parser.add_argument("--log_file", type=Path, help="Path to log file")
    parser.add_argument("--summary_file", type=Path, help="Path to save summary JSON")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or args.output_dir / "bias_correction.log"
    setup_logging(log_file)
    
    try:
        # Create corrector
        corrector = BiasFieldCorrector(
            shrink_factor=args.shrink_factor,
            n_iterations=args.n_iterations,
            convergence_threshold=args.convergence_threshold,
            save_bias_field=not args.no_save_bias_field
        )
        
        # Create pipeline
        pipeline = BiasFieldCorrectionPipeline(corrector)
        
        # Process files
        results = pipeline.process_directory(
            args.input_dir, 
            args.output_dir,
            args.bias_fields_dir
        )
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        logger.info(f"Bias correction complete: {successful}/{len(results)} files processed successfully")
        
        # Save summary if requested
        if args.summary_file:
            args.summary_file.parent.mkdir(parents=True, exist_ok=True)
            with open(args.summary_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        
        # Exit code
        if all(r["status"] == "success" for r in results):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()