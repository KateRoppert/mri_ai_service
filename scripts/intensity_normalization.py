"""
Intensity Normalization Script for BIDS-structured NIfTI files.

Supports multiple normalization methods:
- Histogram Matching
- Z-Score normalization
- WhiteStripe normalization
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import SimpleITK as sitk
from scipy.stats import gaussian_kde

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


class NormalizationMethod:
    """Base class for normalization methods."""
    
    def normalize(self, image: sitk.Image, **kwargs) -> sitk.Image:
        """Normalize the image."""
        raise NotImplementedError


class HistogramMatchingNormalization(NormalizationMethod):
    """Histogram matching normalization."""
    
    def __init__(self, template_path: Path, histogram_levels: int = 1024, 
                 match_points: int = 7):
        self.template = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
        self.histogram_levels = histogram_levels
        self.match_points = match_points
    
    def normalize(self, image: sitk.Image, **kwargs) -> sitk.Image:
        """Normalize using histogram matching."""
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(self.histogram_levels)
        matcher.SetNumberOfMatchPoints(self.match_points)
        
        return matcher.Execute(image, self.template)


class ZScoreNormalization(NormalizationMethod):
    """Z-score normalization."""
    
    def __init__(self, exclude_zeros: bool = True):
        self.exclude_zeros = exclude_zeros
    
    def normalize(self, image: sitk.Image, **kwargs) -> sitk.Image:
        """Normalize to zero mean and unit variance."""
        array = sitk.GetArrayFromImage(image)
        
        if self.exclude_zeros:
            mask = array > 0
            if np.sum(mask) == 0:
                raise ValueError("No non-zero voxels found")
            mean = array[mask].mean()
            std = array[mask].std()
        else:
            mean = array.mean()
            std = array.std()
        
        if std == 0:
            logger.warning("Standard deviation is zero, returning original image")
            return image
        
        normalized_array = (array - mean) / std
        
        normalized_img = sitk.GetImageFromArray(normalized_array)
        normalized_img.CopyInformation(image)
        
        return normalized_img


class WhiteStripeNormalization(NormalizationMethod):
    """WhiteStripe normalization."""
    
    def __init__(self, n_sd: float = 2.0, threshold_at_zero: bool = True):
        self.n_sd = n_sd
        self.threshold_at_zero = threshold_at_zero
    
    def normalize(self, image: sitk.Image, filename: str = "", **kwargs) -> sitk.Image:
        """Normalize using WhiteStripe method."""
        modality = self._detect_modality(filename)
        array = sitk.GetArrayFromImage(image)
        
        # Apply threshold if enabled
        if self.threshold_at_zero:
            mask = array > 0
            if np.sum(mask) == 0:
                raise ValueError("No non-zero voxels found")
            working_array = array[mask]
        else:
            working_array = array.flatten()
        
        # Find WhiteStripe peak
        peak, window = self._find_whitestripe_peak(working_array, modality)
        ws_mask = (working_array >= window[0]) & (working_array <= window[1])
        
        if np.sum(ws_mask) == 0:
            raise ValueError("No voxels found in WhiteStripe window")
        
        # Calculate normalization parameters
        ws_mean = np.mean(working_array[ws_mask])
        ws_std = np.std(working_array[ws_mask])
        
        if ws_std == 0:
            logger.warning("WhiteStripe std is zero, returning original image")
            return image
        
        # Apply normalization
        normalized_array = (array - ws_mean) / ws_std
        
        normalized_img = sitk.GetImageFromArray(normalized_array)
        normalized_img.CopyInformation(image)
        
        return normalized_img
    
    def _detect_modality(self, filename: str) -> str:
        """Detect modality from filename."""
        filename_lower = filename.lower()
        
        if 'flair' in filename_lower:
            return 'FLAIR'
        elif 't2w' in filename_lower:
            return 'T2'
        elif 't1w' in filename_lower:
            if 'ce' in filename_lower or 'gad' in filename_lower:
                return 'T1CE'
            else:
                return 'T1'
        else:
            return 'T1'
    
    def _find_whitestripe_peak(self, data: np.ndarray, modality: str) -> Tuple[float, Tuple[float, float]]:
        """Find peak and window for WhiteStripe."""
        kde = gaussian_kde(data)
        x = np.linspace(np.min(data), np.max(data), 1000)
        y = kde(x)
        peak = x[np.argmax(y)]
        
        std = np.std(data)
        if modality in ['T1', 'T1CE']:
            window = (peak, peak + self.n_sd * std)
        else:  # T2/FLAIR
            window = (peak - self.n_sd * std, peak + self.n_sd * std)
        
        return peak, window


class IntensityNormalizationPipeline:
    """Main pipeline for intensity normalization."""
    
    def __init__(self, method: str, **method_params):
        self.method = method
        self.normalizer = self._create_normalizer(method, **method_params)
    
    def _create_normalizer(self, method: str, **params) -> NormalizationMethod:
        """Create the appropriate normalizer."""
        if method.lower() == "histogrammatching":
            if 'template_path' not in params:
                raise ValueError("HistogramMatching requires template_path")
            return HistogramMatchingNormalization(
                params['template_path'],
                params.get('histogram_levels', 1024),
                params.get('match_points', 7)
            )
        elif method.lower() == "zscore":
            return ZScoreNormalization(params.get('exclude_zeros', True))
        elif method.lower() == "whitestripe":
            return WhiteStripeNormalization(
                params.get('n_sd', 2.0),
                params.get('threshold_at_zero', True)
            )
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def process_file(self, input_file: Path, output_file: Path) -> Dict:
        """Process a single file."""
        logger.info(f"Normalizing: {input_file.name}")
        
        try:
            # Read image
            image = sitk.ReadImage(str(input_file), sitk.sitkFloat32)
            
            # Normalize
            normalized = self.normalizer.normalize(
                image, 
                filename=input_file.name
            )
            
            # Save
            output_file.parent.mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(normalized, str(output_file))
            
            return {
                "status": "success",
                "input": str(input_file),
                "output": str(output_file),
                "method": self.method
            }
            
        except Exception as e:
            logger.error(f"Failed to normalize {input_file.name}: {e}")
            return {
                "status": "failed",
                "input": str(input_file),
                "error": str(e),
                "method": self.method
            }
    
    def process_directory(self, input_dir: Path, output_dir: Path) -> List[Dict]:
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
            
            result = self.process_file(input_file, output_file)
            results.append(result)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Intensity normalization for BIDS-structured NIfTI files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_dir", type=Path, help="Input BIDS directory")
    parser.add_argument("output_dir", type=Path, help="Output BIDS directory")
    parser.add_argument("--method", required=True, 
                       choices=["HistogramMatching", "ZScore", "WhiteStripe"],
                       help="Normalization method")
    parser.add_argument("--template_path", type=Path, 
                       help="Template path (required for HistogramMatching)")
    parser.add_argument("--histogram_levels", type=int, default=1024,
                       help="Histogram levels for HistogramMatching")
    parser.add_argument("--match_points", type=int, default=7,
                       help="Match points for HistogramMatching")
    parser.add_argument("--exclude_zeros", action="store_true",
                       help="Exclude zero voxels in ZScore normalization")
    parser.add_argument("--n_sd", type=float, default=2.0,
                       help="Number of standard deviations for WhiteStripe")
    parser.add_argument("--log_file", type=Path, help="Path to log file")
    parser.add_argument("--summary_file", type=Path, help="Path to save summary JSON")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or args.output_dir / "intensity_normalization.log"
    setup_logging(log_file)
    
    try:
        # Prepare method parameters
        method_params = {}
        
        if args.method == "HistogramMatching":
            if not args.template_path:
                raise ValueError("HistogramMatching requires --template_path")
            method_params.update({
                'template_path': args.template_path,
                'histogram_levels': args.histogram_levels,
                'match_points': args.match_points
            })
        elif args.method == "ZScore":
            method_params['exclude_zeros'] = args.exclude_zeros
        elif args.method == "WhiteStripe":
            method_params['n_sd'] = args.n_sd
        
        # Create pipeline
        pipeline = IntensityNormalizationPipeline(args.method, **method_params)
        
        # Process files
        results = pipeline.process_directory(args.input_dir, args.output_dir)
        
        # Summary
        successful = sum(1 for r in results if r["status"] == "success")
        logger.info(f"Normalization complete: {successful}/{len(results)} files processed successfully")
        
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