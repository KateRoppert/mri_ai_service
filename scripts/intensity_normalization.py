#!/usr/bin/env python3
"""
Intensity Normalization Module

This module provides various intensity normalization strategies for medical images.
Can be used as a standalone script or imported as part of a preprocessing pipeline.

Usage:
    python intensity_normalization.py --input input.nii.gz --output output.nii.gz --method HistogramMatching --template template.nii.gz
    python intensity_normalization.py --input input.nii.gz --output output.nii.gz --method ZScore
    python intensity_normalization.py --input input.nii.gz --output output.nii.gz --method WhiteStripe
"""

import argparse
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import SimpleITK as sitk
from scipy.stats import gaussian_kde
from datetime import datetime 
import sys 

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Utility Functions & Classes ---

def setup_main_logging(log_file_path: Path, console_level: str = "INFO"):
    """Configures the main application logger."""
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s.%(funcName)s] %(message)s'
    )
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
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except Exception as e:
        logger.critical(f"Could not set up file logger at {log_file_path}: {e}")

def save_parameters(params_dict: dict, output_path: Path):
    """Saves a dictionary of parameters to a JSON file, handling non-serializable types."""
    def make_serializable(obj):
        if isinstance(obj, Path):
            return str(obj.resolve())
        if isinstance(obj, (np.integer)):
            return int(obj)
        if isinstance(obj, (np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (datetime)):
             return str(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    # try:
    #     output_path.parent.mkdir(parents=True, exist_ok=True)
    #     with open(output_path, 'w', encoding='utf-8') as f:
    #         json.dump(params_dict, f, indent=4, ensure_ascii=False, default=make_serializable)
    # except Exception as e:
    #     logger.error(f"Failed to save parameters to {output_path}: {e}", exc_info=True)


class IntensityNormalizationStrategy(ABC):
    """Abstract base class for intensity normalization strategies."""
    
    @abstractmethod
    def normalize(self, input_img: sitk.Image, params: dict) -> sitk.Image:
        """Normalize the input image according to the specific strategy."""
        pass


class HistogramMatchingStrategy(IntensityNormalizationStrategy):
    """Normalization using histogram matching to a template."""
    
    def normalize(self, input_img: sitk.Image, params: dict) -> sitk.Image:
        template_path = params.get('template_path')
        if not template_path:
            raise ValueError("HistogramMatching requires 'template_path' parameter")
        
        if not Path(template_path).exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        template_img = sitk.ReadImage(str(template_path), sitk.sitkFloat32)
        
        matcher = sitk.HistogramMatchingImageFilter()
        matcher.SetNumberOfHistogramLevels(params.get('histogram_levels', 1024))
        matcher.SetNumberOfMatchPoints(params.get('match_points', 7))
        
        logger.info(f"Applying histogram matching with {params.get('histogram_levels', 1024)} levels")
        return matcher.Execute(input_img, template_img)


class ZScoreStrategy(IntensityNormalizationStrategy):
    """Normalization using z-score (mean=0, std=1)."""
    
    def normalize(self, input_img: sitk.Image, params: dict) -> sitk.Image:
        # Convert to numpy array for calculations
        array = sitk.GetArrayFromImage(input_img)
        
        # Check if we should exclude zeros (default: False to match your config)
        exclude_zeros = params.get('exclude_zeros', False)
        
        if exclude_zeros:
            # Calculate mean and std only for non-zero voxels (assuming background is 0)
            mask = array > 0
            if np.sum(mask) == 0:
                raise ValueError("No non-zero voxels found in the image")
            
            mean = array[mask].mean()
            std = array[mask].std()
            
            if std == 0:
                raise ValueError("Standard deviation is zero, cannot perform z-score normalization")
            
            logger.info(f"Applying z-score normalization excluding zeros (mean={mean:.3f}, std={std:.3f})")
            
            # Apply z-score normalization only to non-zero voxels
            normalized_array = np.zeros_like(array)
            normalized_array[mask] = (array[mask] - mean) / std
        else:
            # Calculate mean and std for all voxels
            mean = array.mean()
            std = array.std()
            
            if std == 0:
                raise ValueError("Standard deviation is zero, cannot perform z-score normalization")
            
            logger.info(f"Applying z-score normalization including zeros (mean={mean:.3f}, std={std:.3f})")
            
            # Apply z-score normalization to all voxels
            normalized_array = (array - mean) / std
        
        # Convert back to SimpleITK image
        normalized_img = sitk.GetImageFromArray(normalized_array)
        normalized_img.CopyInformation(input_img)
        
        return normalized_img


class WhiteStripeStrategy(IntensityNormalizationStrategy):
    """
    WhiteStripe normalization with automatic modality detection from filename
    and configurable parameters.
    """
    
    def normalize(self, input_img: sitk.Image, params: dict) -> sitk.Image:
        # Get parameters from config (with defaults)
        n_sd = params.get('n_sd', 2.0)
        threshold_at_zero = params.get('threshold_at_zero', True)
        input_filename = params.get('input_filename', '')
        
        # Detect modality from filename
        modality = self._detect_modality_from_filename(input_filename)
        logger.info(f"Detected modality: {modality}")
        
        # Convert to numpy array
        img_array = sitk.GetArrayFromImage(input_img)
        
        # Apply threshold if enabled
        if threshold_at_zero:
            mask = img_array > 0
            if np.sum(mask) == 0:
                raise ValueError("No non-zero voxels found after thresholding")
            working_array = img_array[mask]
        else:
            working_array = img_array.flatten()
        
        # Find WhiteStripe voxels
        peak, window = self._find_whitestripe_peak(working_array, modality, n_sd)
        ws_mask = (working_array >= window[0]) & (working_array <= window[1])
        
        if np.sum(ws_mask) == 0:
            raise ValueError("WhiteStripe failed: no voxels selected in the stripe")
        
        # Calculate normalization parameters
        ws_mean, ws_std = np.mean(working_array[ws_mask]), np.std(working_array[ws_mask])
        
        if ws_std == 0:
            raise ValueError("WhiteStripe standard deviation is zero")
        
        logger.info(f"WhiteStripe normalization: peak={peak:.3f}, window=({window[0]:.3f}, {window[1]:.3f})")
        logger.info(f"Selected {np.sum(ws_mask)} voxels (mean={ws_mean:.3f}, std={ws_std:.3f})")
        
        # Apply normalization to entire image
        normalized_array = (img_array - ws_mean) / ws_std
        
        # Convert back to SimpleITK
        normalized_img = sitk.GetImageFromArray(normalized_array)
        normalized_img.CopyInformation(input_img)
        
        return normalized_img
    
    def _detect_modality_from_filename(self, filename: str) -> str:
        """Detect modality from BIDS-style filename"""
        filename_lower = filename.lower()
        
        if 'flair' in filename_lower or 't2f' in filename_lower:
            return 'FLAIR'
        elif 't2w' in filename_lower:
            return 'T2'
        elif 't1' in filename_lower:
            if 'gad' in filename_lower or 'ce-' in filename_lower or 't1c' in filename_lower:
                return 'T1CE'
            else:
                return 'T1'
        else:
            return 'T1'  # Default fallback
        
    def _find_whitestripe_peak(self, data: np.ndarray, modality: str, n_sd: float) -> Tuple[float, Tuple[float, float]]:
        """Find peak and window for WhiteStripe"""
        # Kernel density estimation
        kde = gaussian_kde(data)
        x = np.linspace(np.min(data), np.max(data), 1000)
        y = kde(x)
        peak = x[np.argmax(y)]
        
        # Set window based on modality
        std = np.std(data)
        if modality in ['T1', 'T1CE']:
            window = (peak, peak + n_sd * std)  # Right tail for T1
        else:  # T2/FLAIR
            window = (peak - n_sd * std, peak + n_sd * std)  # Symmetric for T2/FLAIR
        
        return peak, window


class IntensityNormalizer:
    """Main class for intensity normalization operations."""
    
    STRATEGIES = {
        "HistogramMatching": HistogramMatchingStrategy,
        "ZScore": ZScoreStrategy,
        "WhiteStripe": WhiteStripeStrategy
    }
    
    def __init__(self, method: str = "HistogramMatching"):
        if method not in self.STRATEGIES:
            raise ValueError(f"Unknown intensity normalization method: {method}. "
                           f"Available methods: {list(self.STRATEGIES.keys())}")
        
        self.method = method
        self.strategy = self.STRATEGIES[method]()
    
    def normalize_image(self, input_path: Path, output_path: Path, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Normalize an image using the selected strategy.
        
        Args:
            input_path: Path to input image
            output_path: Path for output image
            params: Parameters for normalization strategy
            
        Returns:
            bool: True if successful, False otherwise
        """
        if params is None:
            params = {}
        
        try:
            # Ensure input file exists
            if not input_path.exists():
                raise FileNotFoundError(f"Input file not found: {input_path}")
            
            # Create output directory if it doesn't exist
            #output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Load image
            logger.info(f"Loading image: {input_path}")
            input_img = sitk.ReadImage(str(input_path), sitk.sitkFloat32)
            
            # Add filename to params for WhiteStripe modality detection
            params['input_filename'] = input_path.name
            
            # Apply normalization
            logger.info(f"Applying {self.method} normalization")
            normalized_img = self.strategy.normalize(input_img, params)
            
            # Save result
            logger.info(f"Saving normalized image: {output_path}")
            sitk.WriteImage(normalized_img, str(output_path))
            
            logger.info("Intensity normalization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Intensity normalization failed: {e}")
            return False


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file."""
    try:
        with open(config_path, 'r') as f:
            content = f.read().strip()
            
        if not content:
            logger.warning(f"Config file is empty: {config_path}")
            return {}
            
        # Try to determine file type by extension
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML is required to read YAML files. Install with: pip install PyYAML")
            return yaml.safe_load(content)
        elif config_path.suffix.lower() == '.json':
            return json.loads(content)
        else:
            # Try YAML first, then JSON
            if YAML_AVAILABLE:
                try:
                    return yaml.safe_load(content)
                except yaml.YAMLError:
                    pass
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                if YAML_AVAILABLE:
                    return yaml.safe_load(content)
                else:
                    raise ValueError("Could not parse config file. Install PyYAML for YAML support: pip install PyYAML")
                    
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return {}


def extract_intensity_config(config: Dict[str, Any], step_name: str) -> Dict[str, Any]:
    """
    Extract intensity normalization configuration from the loaded config.
    This function handles the specific structure of your YAML file.
    """
    intensity_config = {}
    
    # First, try to get from preprocessing.intensity_normalization
    steps = config.get('steps', {})
    preprocessing = steps.get('preprocessing', {})
    if 'intensity_normalization' in preprocessing:
        intensity_config = preprocessing['intensity_normalization'].copy()
        logger.info("Found intensity_normalization config in preprocessing section")
    
    # If not found, try other locations
    if not intensity_config:
        # Try steps structure
        steps_config = config.get('steps', {})
        if step_name in steps_config:
            intensity_config = steps_config[step_name].copy()
            logger.info(f"Found config in steps.{step_name}")
        elif 'intensity_normalization' in steps_config:
            intensity_config = steps_config['intensity_normalization'].copy()
            logger.info("Found config in steps.intensity_normalization")
        else:
            # Try direct access
            if step_name in config:
                intensity_config = config[step_name].copy()
                logger.info(f"Found config in {step_name}")
            elif 'intensity_normalization' in config:
                intensity_config = config['intensity_normalization'].copy()
                logger.info("Found config in intensity_normalization")
    
    # Log what we found
    if intensity_config:
        logger.info(f"Loaded intensity normalization config: {intensity_config}")
    else:
        logger.warning("No intensity normalization config found in any expected location")
    
    return intensity_config

# Pipeline Integration Interface
class IntensityNormalizationStep:
    """Pipeline integration interface for intensity normalization"""
    
    def __init__(self, processor, params: dict, dependency_checker=None):
        self.processor = processor
        self.params = params
        self.dependency_checker = dependency_checker
        self.step_name = "1_intensity_normalization"
        
        # Get method from params
        self.method = params.get('method', 'HistogramMatching')
        
        # Initialize normalizer
        self.normalizer = IntensityNormalizer(method=self.method)
        
        # Handle template path for HistogramMatching
        if self.method == 'HistogramMatching':
            # Check if template_path is in params
            if 'template_path' not in self.params:
                # Try to get from processor
                if hasattr(processor, 'template_path') and processor.template_path:
                    self.params['template_path'] = str(processor.template_path)
                # Try to get from main config paths section
                elif hasattr(processor, 'config'):
                    main_config = processor.config._data
                    if 'paths' in main_config and 'template_path' in main_config['paths']:
                        self.params['template_path'] = main_config['paths']['template_path']
                
            if 'template_path' not in self.params:
                raise ValueError("HistogramMatching method requires template_path")
    
    def run_step(self):
        """Execute intensity normalization step in pipeline context"""
        try:
            # Prepare output path
            output_path = self.processor.temp_dir / f"{self.processor.file_stem}_{self.step_name}.nii.gz"
            
            # Run normalization
            success = self.normalizer.normalize_image(
                input_path=self.processor.current_input_path,
                output_path=output_path,
                params=self.params
            )
            
            if not success:
                raise RuntimeError("Intensity normalization failed")
            
            # Update processor state
            self.processor.current_input_path = output_path
            
            # Store results in processor if needed
            if hasattr(self.processor, 'intensity_normalization_results'):
                self.processor.intensity_normalization_results = {
                    'method': self.method,
                    'input_path': str(self.processor.current_input_path),
                    'output_path': str(output_path),
                    'success': success
                }
            
        except Exception as e:
            logger.error(f"Intensity normalization step failed: {e}")
            raise

def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="Intensity Normalization for Medical Images")
    parser.add_argument("--input", "-i", type=Path, required=True,
                       help="Input image file path")
    parser.add_argument("--output", "-o", type=Path, required=True,
                       help="Output image file path")
    parser.add_argument("--method", "-m", choices=["HistogramMatching", "ZScore", "WhiteStripe"],
                       help="Normalization method (overrides config)")
    parser.add_argument("--template", "-t", type=Path,
                       help="Template image for histogram matching (overrides config)")
    parser.add_argument("--config", "-c", type=Path,
                       help="Configuration file (JSON or YAML)")
    parser.add_argument("--histogram-levels", type=int,
                       help="Number of histogram levels for histogram matching (overrides config)")
    parser.add_argument("--match-points", type=int,
                       help="Number of match points for histogram matching (overrides config)")
    parser.add_argument("--n-sd", type=float,
                       help="Number of standard deviations for WhiteStripe (overrides config)")
    parser.add_argument("--no-threshold", action="store_true",
                       help="Disable thresholding at zero for WhiteStripe (overrides config)")
    parser.add_argument("--exclude-zeros", action="store_true",
                       help="Exclude zero voxels from Z-Score calculation (overrides config)")
    parser.add_argument("--include-zeros", action="store_true",
                       help="Include zero voxels in Z-Score calculation (overrides config)")
    parser.add_argument("--step-name", default="intensity_normalization",
                       help="Step name to look for in config file")
    parser.add_argument("--log_file", default=None, type=Path, 
                        help="Path to log file. Defaults to 'intensity_norm.log' in output dir.")
    
    args = parser.parse_args()

    log_path = args.log_file or args.output / "intensity_norm.log"
    setup_main_logging(log_path)
    
    # Load config if provided
    config = {}
    intensity_config = {}
    if args.config and args.config.exists():
        config = load_config(args.config)
        intensity_config = extract_intensity_config(config, args.step_name)
    
    # Get method from config or command line, with fallback to default
    method = args.method or intensity_config.get('method', 'ZScore')
    
    # Start with config parameters (create a copy to avoid modifying original)
    params = intensity_config.copy()
    
    # Remove non-parameter keys that shouldn't be passed to strategies
    config_only_keys = ['enabled', 'method']
    for key in config_only_keys:
        params.pop(key, None)
    
    # Override with command line arguments if provided
    if args.histogram_levels is not None:
        params['histogram_levels'] = args.histogram_levels
    if args.match_points is not None:
        params['match_points'] = args.match_points
    if args.n_sd is not None:
        params['n_sd'] = args.n_sd
    if args.no_threshold:
        params['threshold_at_zero'] = False
    if args.exclude_zeros:
        params['exclude_zeros'] = True
    if args.include_zeros:
        params['exclude_zeros'] = False
    if args.template:
        params['template_path'] = str(args.template)
    
    # Set defaults for parameters not specified in config or command line
    if 'histogram_levels' not in params:
        params['histogram_levels'] = 1024
    if 'match_points' not in params:
        params['match_points'] = 7
    if 'n_sd' not in params:
        params['n_sd'] = 2.0
    if 'threshold_at_zero' not in params:
        params['threshold_at_zero'] = True
    if 'exclude_zeros' not in params:
        params['exclude_zeros'] = False
    
    # Check if method requires template
    if method == "HistogramMatching" and 'template_path' not in params:
        logger.error("HistogramMatching method requires --template argument or template_path in config")
        return False
    
    # Log final parameters
    logger.info(f"Using method: {method}")
    logger.info(f"Final parameters: {params}")
    
    # Initialize normalizer and process
    normalizer = IntensityNormalizer(method=method)
    success = normalizer.normalize_image(args.input, args.output, params)
    
    return success


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)