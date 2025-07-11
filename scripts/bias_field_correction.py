#!/usr/bin/env python3
"""
Standalone Bias Field Correction Script for Brain MRI Preprocessing

This script provides modular bias field correction with multiple strategies:
- Standard N4ITK correction
- Modality-specific correction with optimized parameters
- T1-based correction for co-registered images

Author: Your Name
Date: 2025
"""

import argparse
import json
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml

try:
    import SimpleITK as sitk
    import numpy as np
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages: pip install SimpleITK numpy")
    sys.exit(1)


@dataclass
class ModalityParams:
    """Parameters for modality-specific bias field correction"""
    shrink_factor: int = 4
    n_iterations: List[int] = None
    convergence_threshold: float = 0.001
    smoothing_factor: float = 0.15
    spline_order: int = 3
    number_of_control_points: int = 4
    use_brain_mask: bool = False
    
    def __post_init__(self):
        if self.n_iterations is None:
            self.n_iterations = [50, 50, 50, 50]


class BiasFieldValidator:
    """Validator for bias field correction quality assessment"""
    
    def validate_correction(self, original_img: sitk.Image, corrected_img: sitk.Image, 
                          bias_field: sitk.Image, output_dir: Path, modality: str) -> Dict:
        """Validate bias field correction quality"""
        try:
            # Calculate coefficient of variation improvement
            original_array = sitk.GetArrayFromImage(original_img)
            corrected_array = sitk.GetArrayFromImage(corrected_img)
            
            # Mask out background
            mask = original_array > np.percentile(original_array[original_array > 0], 5)
            
            original_cv = np.std(original_array[mask]) / np.mean(original_array[mask])
            corrected_cv = np.std(corrected_array[mask]) / np.mean(corrected_array[mask])
            
            cv_improvement = original_cv - corrected_cv
            
            # Calculate bias field statistics
            bias_array = sitk.GetArrayFromImage(bias_field)
            bias_stats = {
                'min': float(np.min(bias_array)),
                'max': float(np.max(bias_array)),
                'mean': float(np.mean(bias_array)),
                'std': float(np.std(bias_array))
            }
            
            metrics = {
                'modality': modality,
                'original_cv': float(original_cv),
                'corrected_cv': float(corrected_cv),
                'cv_improvement': float(cv_improvement),
                'bias_field_stats': bias_stats,
                'correction_quality': 'good' if cv_improvement > 0.01 else 'moderate' if cv_improvement > 0 else 'poor'
            }
            
            return metrics
            
        except Exception as e:
            logging.warning(f"Validation failed for {modality}: {e}")
            return {'modality': modality, 'validation_error': str(e)}


class BiasFieldCorrectionStrategy(ABC):
    """Abstract base class for bias field correction strategies"""
    
    @abstractmethod
    def correct_bias(self, input_img: sitk.Image, params: dict) -> Tuple[sitk.Image, sitk.Image]:
        """
        Correct bias field in the input image
        Returns: (corrected_image, bias_field)
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging"""
        pass


class StandardBiasFieldStrategy(BiasFieldCorrectionStrategy):
    """Standard N4ITK with same parameters for all modalities"""
    
    def correct_bias(self, input_img: sitk.Image, params: dict) -> Tuple[sitk.Image, sitk.Image]:
        # Create mask
        mask_img = self._create_mask(input_img, params)
        
        # Use config parameters with fallback to defaults
        shrink_factor = params.get('sitk_shrinkFactor', params.get('shrink_factor', 4))
        shrunk_img = sitk.Shrink(input_img, [shrink_factor] * input_img.GetDimension())
        shrunk_mask = sitk.Shrink(mask_img, [shrink_factor] * input_img.GetDimension())
        
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(params.get('sitk_numberOfIterations', params.get('n_iterations', [50] * shrink_factor)))
        corrector.SetConvergenceThreshold(params.get('sitk_convergenceThreshold', params.get('convergence_threshold', 0.001)))
        corrector.SetBiasFieldFullWidthAtHalfMaximum(params.get('smoothing_factor', 0.15))
        
        corrector.Execute(shrunk_img, shrunk_mask)
        
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_img)
        bias_field = sitk.Exp(log_bias_field)
        corrected_image = input_img / bias_field
        
        return corrected_image, bias_field
    
    def get_strategy_name(self) -> str:
        return "StandardN4ITK"
    
    def _create_mask(self, input_img: sitk.Image, params: dict) -> sitk.Image:
        """Create brain mask using Otsu thresholding"""
        return sitk.OtsuThreshold(input_img, 0, 1)


class ModalitySpecificBiasFieldStrategy(BiasFieldCorrectionStrategy):
    """Modality-specific N4ITK with optimized parameters per modality"""
    
    def __init__(self):
        self.modality_params = self._get_default_modality_params()
    
    def _get_default_modality_params(self) -> Dict[str, ModalityParams]:
        """Get default parameters optimized for each modality"""
        return {
            'T1': ModalityParams(
                shrink_factor=4,
                n_iterations=[50, 50, 50, 50],
                convergence_threshold=0.001,
                smoothing_factor=0.15,
                use_brain_mask=False
            ),
            'T1CE': ModalityParams(
                shrink_factor=4,
                n_iterations=[50, 50, 30, 20],  # Less aggressive for contrast
                convergence_threshold=0.001,
                smoothing_factor=0.20,  # More smoothing for contrast artifacts
                use_brain_mask=True  # Mask enhancing lesions
            ),
            'T2': ModalityParams(
                shrink_factor=4,
                n_iterations=[50, 50, 50, 50],
                convergence_threshold=0.001,
                smoothing_factor=0.15,
                use_brain_mask=False
            ),
            'FLAIR': ModalityParams(
                shrink_factor=3,  # Less downsampling for noisy FLAIR
                n_iterations=[60, 50, 40, 30],  # More iterations
                convergence_threshold=0.0005,  # Stricter convergence
                smoothing_factor=0.25,  # More smoothing for noise
                use_brain_mask=True  # Better with brain extraction
            )
        }
    
    def correct_bias(self, input_img: sitk.Image, params: dict) -> Tuple[sitk.Image, sitk.Image]:
        modality = params.get('modality', 'T1')
        modal_params = self.modality_params.get(modality, self.modality_params['T1'])
        
        # Override with config parameters if provided
        if 'modality_specific_params' in params and modality in params['modality_specific_params']:
            config_params = params['modality_specific_params'][modality]
            for key, value in config_params.items():
                if hasattr(modal_params, key):
                    setattr(modal_params, key, value)
        
        # Create appropriate mask
        mask_img = self._create_modality_specific_mask(input_img, modality, modal_params)
        
        # Apply shrinking
        shrunk_img = sitk.Shrink(input_img, [modal_params.shrink_factor] * input_img.GetDimension())
        shrunk_mask = sitk.Shrink(mask_img, [modal_params.shrink_factor] * input_img.GetDimension())
        
        # Configure N4ITK
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(modal_params.n_iterations)
        corrector.SetConvergenceThreshold(modal_params.convergence_threshold)
        corrector.SetBiasFieldFullWidthAtHalfMaximum(modal_params.smoothing_factor)
        corrector.SetSplineOrder(modal_params.spline_order)
        
        corrector.Execute(shrunk_img, shrunk_mask)
        
        log_bias_field = corrector.GetLogBiasFieldAsImage(input_img)
        bias_field = sitk.Exp(log_bias_field)
        corrected_image = input_img / bias_field
        
        return corrected_image, bias_field
    
    def get_strategy_name(self) -> str:
        return "ModalitySpecificN4ITK"
    
    def _create_modality_specific_mask(self, input_img: sitk.Image, modality: str, modal_params: ModalityParams) -> sitk.Image:
        """Create modality-specific mask"""
        if modal_params.use_brain_mask:
            # More sophisticated brain extraction for FLAIR and T1CE
            return self._create_brain_mask(input_img, modality)
        else:
            # Simple Otsu thresholding
            return sitk.OtsuThreshold(input_img, 0, 1)
    
    def _create_brain_mask(self, input_img: sitk.Image, modality: str) -> sitk.Image:
        """Create brain mask with morphological operations"""
        # Otsu threshold
        mask = sitk.OtsuThreshold(input_img, 0, 1)
        
        # Morphological operations to clean up mask
        kernel_radius = 2
        mask = sitk.BinaryMorphologicalClosing(mask, [kernel_radius] * input_img.GetDimension())
        mask = sitk.BinaryFillhole(mask)
        
        # For T1CE, additional processing to handle contrast enhancement
        if modality == 'T1CE':
            # Erosion to reduce impact of enhancing lesions
            mask = sitk.BinaryErode(mask, [1] * input_img.GetDimension())
        
        return mask


class T1BasedCorrectionStrategy(BiasFieldCorrectionStrategy):
    """Compute T1 bias field first, then use for all co-registered modalities"""
    
    def __init__(self):
        self.t1_bias_field = None
        self.t1_params = ModalityParams(
            shrink_factor=4,
            n_iterations=[50, 50, 50, 50],
            convergence_threshold=0.001,
            smoothing_factor=0.15,
            use_brain_mask=False
        )
    
    def correct_bias(self, input_img: sitk.Image, params: dict) -> Tuple[sitk.Image, sitk.Image]:
        modality = params.get('modality', 'T1')
        
        if modality == 'T1':
            # Compute T1 bias field
            bias_field = self._compute_t1_bias_field(input_img, params)
            self.t1_bias_field = bias_field
            corrected_image = input_img / bias_field
        else:
            # Use T1 bias field for other modalities
            if self.t1_bias_field is None:
                # Load from saved path if available
                t1_bias_path = params.get('t1_bias_field_path')
                if t1_bias_path and Path(t1_bias_path).exists():
                    self.t1_bias_field = sitk.ReadImage(str(t1_bias_path))
                else:
                    raise ValueError("T1 bias field not computed yet. Process T1 modality first.")
            
            bias_field = self.t1_bias_field
            corrected_image = input_img / bias_field
        
        return corrected_image, bias_field
    
    def _compute_t1_bias_field(self, t1_img: sitk.Image, params: dict) -> sitk.Image:
        """Compute bias field from T1 image"""
        # Create mask
        mask_img = sitk.OtsuThreshold(t1_img, 0, 1)
        
        # Use config parameters with fallback to defaults
        shrink_factor = params.get('sitk_shrinkFactor', params.get('shrink_factor', self.t1_params.shrink_factor))
        
        # Apply shrinking
        shrunk_img = sitk.Shrink(t1_img, [shrink_factor] * t1_img.GetDimension())
        shrunk_mask = sitk.Shrink(mask_img, [shrink_factor] * t1_img.GetDimension())
        
        # Configure N4ITK
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(params.get('sitk_numberOfIterations', params.get('n_iterations', self.t1_params.n_iterations)))
        corrector.SetConvergenceThreshold(params.get('sitk_convergenceThreshold', params.get('convergence_threshold', self.t1_params.convergence_threshold)))
        corrector.SetBiasFieldFullWidthAtHalfMaximum(params.get('smoothing_factor', self.t1_params.smoothing_factor))
        
        corrector.Execute(shrunk_img, shrunk_mask)
        
        log_bias_field = corrector.GetLogBiasFieldAsImage(t1_img)
        return sitk.Exp(log_bias_field)
    
    def get_strategy_name(self) -> str:
        return "T1BasedCorrection"


class BiasFieldProcessor:
    """Main processor for bias field correction with logging and validation"""
    
    STRATEGIES = {
        "standard": StandardBiasFieldStrategy,
        "N4BiasFieldCorrection": StandardBiasFieldStrategy,  # Backward compatibility
        "modality_specific": ModalitySpecificBiasFieldStrategy,
        "t1_based": T1BasedCorrectionStrategy
    }
    
    def __init__(self, strategy: str = "modality_specific", validate_correction: bool = True):
        """
        Initialize bias field processor
        
        Args:
            strategy: Correction strategy ('standard', 'modality_specific', 't1_based')
            validate_correction: Whether to validate correction quality
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown bias correction strategy: {strategy}. Available: {list(self.STRATEGIES.keys())}")
        
        self.strategy = self.STRATEGIES[strategy]()
        self.validator = BiasFieldValidator() if validate_correction else None
        self.validate_correction = validate_correction
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def process_image(self, input_path: Union[str, Path], output_path: Union[str, Path], 
                     transform_dir: Union[str, Path], modality: Optional[str] = None,
                     config_params: Optional[Dict] = None) -> Dict:
        """
        Process a single image for bias field correction
        
        Args:
            input_path: Path to input NIfTI image
            output_path: Path to save corrected image
            transform_dir: Directory to save transformations and logs
            modality: Image modality ('T1', 'T2', 'FLAIR', 'T1CE')
            config_params: Additional configuration parameters
            
        Returns:
            Dictionary with processing results and paths
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        transform_dir = Path(transform_dir)
        
        # Create directories
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect modality if not provided
        if modality is None:
            modality = self._detect_modality_from_filename(str(input_path))
        
        self.logger.info(f"Processing {input_path.name} with {self.strategy.get_strategy_name()} strategy")
        self.logger.info(f"Detected modality: {modality}")
        
        try:
            # Load image
            input_img = sitk.ReadImage(str(input_path.resolve()), sitk.sitkFloat32)
            
            # Prepare parameters
            correction_params = {
                'modality': modality,
                'input_filename': str(input_path),
                **(config_params or {})
            }
            
            # Run bias field correction
            corrected_img, bias_field = self.strategy.correct_bias(input_img, correction_params)
            
            # Save corrected image
            sitk.WriteImage(corrected_img, str(output_path.resolve()))
            self.logger.info(f"Corrected image saved to: {output_path}")
            
            # Save bias field
            bias_field_path = transform_dir / f"bias_field_{modality.lower()}.nii.gz"
            sitk.WriteImage(bias_field, str(bias_field_path))
            self.logger.info(f"Bias field saved to: {bias_field_path}")
            
            # Prepare results
            results = {
                'input_path': str(input_path),
                'output_path': str(output_path),
                'bias_field_path': str(bias_field_path),
                'modality': modality,
                'strategy': self.strategy.get_strategy_name(),
                'success': True
            }
            
            # Validation
            if self.validate_correction:
                validation_dir = transform_dir / "validation"
                validation_dir.mkdir(exist_ok=True)
                
                metrics = self.validator.validate_correction(
                    input_img, corrected_img, bias_field, 
                    validation_dir, modality
                )
                
                # Save metrics
                metrics_path = validation_dir / f"{modality}_bias_correction_metrics.json"
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                results['validation_metrics'] = metrics
                results['metrics_path'] = str(metrics_path)
                
                cv_improvement = metrics.get('cv_improvement', 0)
                self.logger.info(f"Bias correction validation completed. CV improvement: {cv_improvement:.3f}")
            
            return results
            
        except Exception as e:
            error_msg = f"Error processing {input_path.name}: {str(e)}"
            self.logger.error(error_msg)
            return {
                'input_path': str(input_path),
                'output_path': str(output_path),
                'modality': modality,
                'strategy': self.strategy.get_strategy_name(),
                'success': False,
                'error': error_msg
            }
    
    def _detect_modality_from_filename(self, filename: str) -> str:
        """Detect modality from filename"""
        filename_lower = Path(filename).name.lower()
        
        # Common BIDS patterns
        if 'flair' in filename_lower or 't2f' in filename_lower:
            return 'FLAIR'
        elif 't2w' in filename_lower:
            return 'T2'
        elif 'ce-gd_t1w' in filename_lower or ('ce' in filename_lower and 't1w' in filename_lower) or 't1c' in filename_lower:
            return 'T1CE'
        elif 't1w' in filename_lower or 't1n' in filename_lower:
            return 'T1'
        else:
            self.logger.warning(f"Could not detect modality from filename: {filename}. Using T1 as default.")
            return 'T1'


def setup_logging(log_path: Optional[Path] = None, console_level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, console_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if log path provided
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logging.info(f"Logging setup complete. Log file: {log_path}")
    else:
        logging.info("Logging setup complete. Console only.")


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Error loading config file {config_path}: {e}")
        return {}


def extract_bias_field_config(config: Dict) -> Dict:
    """Extract bias field correction configuration from nested config structure"""
    try:
        # Navigate through the nested structure
        steps = config.get('steps', {})
        preprocessing = steps.get('preprocessing', {})
        bias_field_config = preprocessing.get('bias_field_correction', {})
        
        # Log what we found for debugging
        logging.debug(f"Extracted bias field config: {bias_field_config}")
        
        return bias_field_config
    except Exception as e:
        logging.warning(f"Error extracting bias field configuration: {e}")
        return {}
    

# Add this section to the end of bias_field_correction.py, before the main() function

# Pipeline Integration Interface
class BiasFieldCorrectionStep:
    """Pipeline integration interface for bias field correction"""
    
    def __init__(self, processor, params: dict, dependency_checker=None):
        self.processor = processor
        self.params = params
        self.dependency_checker = dependency_checker
        self.step_name = "2_bias_field_correction"
        
        # Get strategy from params
        # Map config names to internal strategy names
        strategy_mapping = {
            'N4BiasFieldCorrection': 'standard',
            'Standard': 'standard',
            'ModalitySpecific': 'modality_specific',
            'T1Based': 't1_based'
        }
        
        config_strategy = params.get('strategy', params.get('method', 'N4BiasFieldCorrection'))
        self.strategy = strategy_mapping.get(config_strategy, 'standard')
        self.validate_correction = params.get('validate_correction', True)
        
        # Initialize processor
        self.bias_processor = BiasFieldProcessor(
            strategy=self.strategy,
            validate_correction=self.validate_correction
        )
    
    def run_step(self):
        """Execute bias field correction step in pipeline context"""
        try:
            # Prepare output path
            output_path = self.processor.temp_dir / f"{self.processor.file_stem}_{self.step_name}.nii.gz"
            
            # Detect modality from filename
            modality = self._detect_modality_from_filename(str(self.processor.current_input_path))
            
            # Process the image
            results = self.bias_processor.process_image(
                input_path=self.processor.current_input_path,
                output_path=output_path,
                transform_dir=self.processor.transform_dir,
                modality=modality,
                config_params=self.params
            )
            
            if not results['success']:
                raise RuntimeError(f"Bias field correction failed: {results.get('error', 'Unknown error')}")
            
            # Update processor state
            self.processor.current_input_path = output_path
            
            # Store results in processor
            if hasattr(self.processor, 'bias_correction_results'):
                self.processor.bias_correction_results = results
            
        except Exception as e:
            logging.error(f"Bias field correction step failed: {e}")
            raise
    
    def _detect_modality_from_filename(self, filename: str) -> str:
        """Detect modality from filename"""
        filename_lower = Path(filename).name.lower()
        
        if 'flair' in filename_lower:
            return 'FLAIR'
        elif 't2w' in filename_lower:
            return 'T2'
        elif 'ce-gd_t1w' in filename_lower or ('ce' in filename_lower and 't1w' in filename_lower):
            return 'T1CE'
        elif 't1w' in filename_lower:
            return 'T1'
        else:
            return 'T1'  # Default fallback


def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(
        description="Standalone Bias Field Correction for Brain MRI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", type=Path, help="Input NIfTI image path")
    parser.add_argument("output_path", type=Path, help="Output corrected image path")
    parser.add_argument("transform_dir", type=Path, help="Directory for transforms and logs")
    
    parser.add_argument("--strategy", type=str, default="modality_specific",
                       choices=["standard", "modality_specific", "t1_based"],
                       help="Bias correction strategy")
    parser.add_argument("--modality", type=str, choices=["T1", "T2", "FLAIR", "T1CE"],
                       help="Image modality (auto-detected if not specified)")
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--validate", action="store_true", default=True,
                       help="Validate correction quality")
    parser.add_argument("--log_file", type=Path, help="Log file path")
    parser.add_argument("--console_log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Console logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_path = args.log_file or args.transform_dir / "bias_field_correction.log"
    setup_logging(log_path, args.console_log_level)
    
    # Load and extract configuration
    config_params = {}
    if args.config:
        full_config = load_config(args.config)
        bias_field_config = extract_bias_field_config(full_config)
        
        # Use config parameters, with command line args taking precedence
        strategy = args.strategy
        if 'strategy' in bias_field_config:
            # Map config strategy names to our internal names
            strategy_mapping = {
                'Standard': 'standard',
                'ModalitySpecific': 'modality_specific', 
                'T1Based': 't1_based'
            }
            config_strategy = bias_field_config['strategy']
            if config_strategy in strategy_mapping:
                strategy = strategy_mapping[config_strategy]
                logging.info(f"Using strategy from config: {config_strategy} -> {strategy}")
        
        # Override validation setting from config if present
        validate_correction = args.validate
        if 'validate_correction' in bias_field_config:
            validate_correction = bias_field_config['validate_correction']
            logging.info(f"Using validation setting from config: {validate_correction}")
        
        # Pass all bias field config parameters
        config_params = bias_field_config
        logging.info(f"Loaded config parameters: {list(config_params.keys())}")
    else:
        strategy = args.strategy
        validate_correction = args.validate

    # Initialize processor
    processor = BiasFieldProcessor(
        strategy=strategy,
        validate_correction=validate_correction
    )
    
    # Process image
    results = processor.process_image(
        input_path=args.input_path,
        output_path=args.output_path,
        transform_dir=args.transform_dir,
        modality=args.modality,
        config_params=config_params
    )
    
    # Save results
    results_path = args.transform_dir / "bias_correction_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if results['success']:
        logging.info("Bias field correction completed successfully")
        logging.info(f"Results saved to: {results_path}")
    else:
        logging.error("Bias field correction failed")
        sys.exit(1)


if __name__ == "__main__":
    main()