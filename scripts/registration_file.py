"""
Standalone Brain MRI Registration Tool
Registers a single image to a template without requiring BIDS structure
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import yaml
import json
from dataclasses import dataclass, field, asdict
import subprocess
import tempfile
import shutil
import os
from datetime import datetime

# Global logger will be configured later
logger = logging.getLogger(__name__)


def setup_logging(log_file: Optional[Path] = None, verbose: bool = False) -> None:
    """Setup logging to both console and optionally to file"""
    
    # Clear any existing handlers
    logger.handlers.clear()
    logging.getLogger().handlers.clear()
    
    # Set logging level
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    
    # File handler if log file specified
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    # Configure module logger
    logger.setLevel(logging.DEBUG)


@dataclass
class RegistrationResult:
    """Data class to hold registration results"""
    registered_image_path: Path
    transform_paths: List[Path] = field(default_factory=list)
    inverse_transform_paths: List[Path] = field(default_factory=list)
    metric_value: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'registered_image_path': str(self.registered_image_path),
            'transform_paths': [str(p) for p in self.transform_paths],
            'inverse_transform_paths': [str(p) for p in self.inverse_transform_paths],
            'metric_value': self.metric_value,
            'success': self.success,
            'error_message': self.error_message
        }


@dataclass
class RegistrationConfig:
    """Configuration for registration parameters"""
    tool: str = "ANTs"
    similarity_metric: str = "MI"
    interpolation: str = "Linear"
    transform_type: str = "SyN"
    iterations: List[int] = field(default_factory=lambda: [1000, 500, 250, 100])
    shrink_factors: List[int] = field(default_factory=lambda: [8, 4, 2, 1])
    smoothing_sigmas: List[float] = field(default_factory=lambda: [3, 2, 1, 0])
    sampling_percentage: float = 0.25
    histogram_bins: int = 32
    learning_rate: float = 0.1
    convergence_threshold: float = 1e-6
    # Template paths from config
    template_path: Optional[str] = None
    template_mask_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, config_path: Path) -> 'RegistrationConfig':
        """Load configuration from YAML file"""
        logger.info(f"Loading configuration from: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract registration config
        reg_config = config_data.get('registration', {})
        
        # Create instance with defaults, then update with config
        instance = cls()
        
        # Log default values
        logger.info("Using default registration parameters:")
        logger.info(f"  tool: {instance.tool}")
        logger.info(f"  similarity_metric: {instance.similarity_metric}")
        logger.info(f"  interpolation: {instance.interpolation}")
        logger.info(f"  transform_type: {instance.transform_type}")
        logger.info(f"  iterations: {instance.iterations}")
        logger.info(f"  shrink_factors: {instance.shrink_factors}")
        logger.info(f"  smoothing_sigmas: {instance.smoothing_sigmas}")
        logger.info(f"  sampling_percentage: {instance.sampling_percentage}")
        logger.info(f"  histogram_bins: {instance.histogram_bins}")
        logger.info(f"  learning_rate: {instance.learning_rate}")
        logger.info(f"  convergence_threshold: {instance.convergence_threshold}")
        
        # Update with config file values and log overrides
        config_overrides = []
        for key, value in reg_config.items():
            if hasattr(instance, key):
                old_value = getattr(instance, key)
                setattr(instance, key, value)
                if old_value != value:
                    config_overrides.append(f"  {key}: {old_value} -> {value}")
        
        if config_overrides:
            logger.info("Configuration file overrides:")
            for override in config_overrides:
                logger.info(override)
        else:
            logger.info("No parameters overridden by configuration file")
        
        return instance
    
    def get_template_path(self) -> Path:
        """Get template path, raise error if not set"""
        if not self.template_path:
            raise ValueError("template_path not specified in configuration file")
        
        template_path = Path(self.template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        logger.info(f"Using template from config: {template_path}")
        return template_path


class RegistrationTool:
    """Registration tool implementation"""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.validate_dependencies()
    
    def validate_dependencies(self):
        """Validate tool dependencies"""
        logger.info(f"Validating dependencies for tool: {self.config.tool}")
        
        if self.config.tool.upper() == "ANTS":
            self._validate_ants()
            logger.info("ANTs dependencies validated successfully")
        elif self.config.tool.upper() == "FSL":
            self._validate_fsl()
            logger.info("FSL dependencies validated successfully")
        else:
            raise ValueError(f"Unsupported registration tool: {self.config.tool}")
    
    def _validate_ants(self):
        """Validate ANTs dependencies"""
        try:
            import ants
            self.ants = ants
            logger.debug("ANTs Python package imported successfully")
        except ImportError:
            raise ImportError("ANTs Python package (antspyx) required but not installed")
        
        required_binaries = ['antsRegistration', 'antsApplyTransforms']
        for binary in required_binaries:
            if shutil.which(binary) is None:
                raise RuntimeError(f"ANTs binary '{binary}' not found in PATH")
            else:
                logger.debug(f"Found ANTs binary: {binary}")
    
    def _validate_fsl(self):
        """Validate FSL dependencies"""
        required_binaries = ['flirt', 'fnirt', 'applywarp']
        for binary in required_binaries:
            if shutil.which(binary) is None:
                raise RuntimeError(f"FSL binary '{binary}' not found in PATH")
            else:
                logger.debug(f"Found FSL binary: {binary}")
        
        if 'FSLDIR' not in os.environ:
            raise RuntimeError("FSLDIR environment variable not set")
        else:
            logger.debug(f"FSLDIR environment variable: {os.environ['FSLDIR']}")
    
    def register_to_template(self, moving_image: Path, template: Path, 
                           output_path: Path, transform_dir: Path) -> RegistrationResult:
        """Register image to template"""
        logger.info(f"Starting registration using {self.config.tool}")
        logger.info(f"Moving image: {moving_image}")
        logger.info(f"Template: {template}")
        logger.info(f"Output path: {output_path}")
        logger.info(f"Transform directory: {transform_dir}")
        
        # Create transform directory
        transform_dir.mkdir(parents=True, exist_ok=True)
        
        if self.config.tool.upper() == "ANTS":
            return self._ants_register(moving_image, template, output_path, transform_dir)
        elif self.config.tool.upper() == "FSL":
            return self._fsl_register(moving_image, template, output_path, transform_dir)
    
    def _ants_register(self, moving_image: Path, template: Path, 
                      output_path: Path, transform_dir: Path) -> RegistrationResult:
        """ANTs registration implementation"""
        try:
            logger.info("Loading images with ANTs")
            
            fixed_img = self.ants.image_read(str(template))
            moving_img = self.ants.image_read(str(moving_image))
            
            logger.debug(f"Fixed image shape: {fixed_img.shape}")
            logger.debug(f"Moving image shape: {moving_img.shape}")
            
            # Determine transform type
            if self.config.transform_type.lower() == "rigid":
                type_of_transform = "Rigid"
            elif self.config.transform_type.lower() == "affine":
                type_of_transform = "Affine"
            else:
                type_of_transform = "SyN"
            
            logger.info(f"Using transform type: {type_of_transform} (from config: {self.config.transform_type})")
            
            # Create output prefix for transforms
            output_prefix = transform_dir / "transform"
            
            logger.info("Starting ANTs registration with parameters:")
            logger.info(f"  similarity_metric: {self.config.similarity_metric}")
            logger.info(f"  interpolation: {self.config.interpolation}")
            logger.info(f"  iterations: {self.config.iterations}")
            logger.info(f"  shrink_factors: {self.config.shrink_factors}")
            logger.info(f"  smoothing_sigmas: {self.config.smoothing_sigmas}")
            
            # Perform registration
            reg_result = self.ants.registration(
                fixed=fixed_img,
                moving=moving_img,
                type_of_transform=type_of_transform,
                outprefix=str(output_prefix),
                verbose=True
            )
            
            # Save result to specified output path
            self.ants.image_write(reg_result['warpedmovout'], str(output_path))
            
            logger.info(f"ANTs registration completed successfully")
            logger.info(f"Registered image saved to: {output_path}")
            
            # Log transform files
            fwd_transforms = reg_result.get('fwdtransforms', [])
            inv_transforms = reg_result.get('invtransforms', [])
            
            logger.info(f"Forward transforms: {len(fwd_transforms)} files")
            for i, t in enumerate(fwd_transforms):
                logger.info(f"  [{i}] {t}")
            
            logger.info(f"Inverse transforms: {len(inv_transforms)} files")
            for i, t in enumerate(inv_transforms):
                logger.info(f"  [{i}] {t}")
            
            return RegistrationResult(
                registered_image_path=output_path,
                transform_paths=[Path(t) for t in fwd_transforms],
                inverse_transform_paths=[Path(t) for t in inv_transforms],
                success=True
            )
            
        except Exception as e:
            logger.error(f"ANTs registration failed: {e}")
            logger.exception("Full traceback:")
            return RegistrationResult(
                registered_image_path=output_path,
                success=False,
                error_message=str(e)
            )
    
    def _fsl_register(self, moving_image: Path, template: Path, 
                     output_path: Path, transform_dir: Path) -> RegistrationResult:
        """FSL registration implementation"""
        try:
            logger.info("Starting FSL registration")
            
            linear_matrix = transform_dir / "linear_transform.mat"
            
            # Linear registration with FLIRT
            flirt_cmd = [
                "flirt",
                "-in", str(moving_image),
                "-ref", str(template),
                "-out", str(output_path),
                "-omat", str(linear_matrix),
                "-cost", "mutualinfo",
                "-dof", "12"
            ]
            
            logger.info("Running FLIRT with parameters:")
            logger.info(f"  cost function: mutualinfo")
            logger.info(f"  degrees of freedom: 12")
            logger.debug(f"FLIRT command: {' '.join(flirt_cmd)}")
            
            result = subprocess.run(flirt_cmd, check=True, capture_output=True, text=True)
            logger.debug(f"FLIRT stdout: {result.stdout}")
            
            transform_paths = [linear_matrix]
            
            # Non-linear registration if requested
            if self.config.transform_type.lower() in ["syn", "nonlinear"]:
                logger.info("Running non-linear registration with FNIRT")
                warp_field = transform_dir / "nonlinear_warp.nii.gz"
                
                fnirt_cmd = [
                    "fnirt",
                    f"--in={moving_image}",
                    f"--ref={template}",
                    f"--aff={linear_matrix}",
                    f"--iout={output_path}",
                    f"--cout={warp_field}"
                ]
                
                logger.debug(f"FNIRT command: {' '.join(fnirt_cmd)}")
                result = subprocess.run(fnirt_cmd, check=True, capture_output=True, text=True)
                logger.debug(f"FNIRT stdout: {result.stdout}")
                
                transform_paths.append(warp_field)
            
            logger.info(f"FSL registration completed successfully")
            logger.info(f"Registered image saved to: {output_path}")
            
            logger.info(f"Transform files: {len(transform_paths)} files")
            for i, t in enumerate(transform_paths):
                logger.info(f"  [{i}] {t}")
            
            return RegistrationResult(
                registered_image_path=output_path,
                transform_paths=transform_paths,
                success=True
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FSL registration failed: {e}")
            logger.error(f"stderr: {e.stderr}")
            return RegistrationResult(
                registered_image_path=output_path,
                success=False,
                error_message=f"{e}\nstderr: {e.stderr}"
            )


def create_default_config():
    """Create default registration configuration file"""
    default_config = {
        'registration': {
            'tool': 'ANTs',
            'similarity_metric': 'MI',
            'interpolation': 'Linear',
            'transform_type': 'SyN',
            'iterations': [1000, 500, 250, 100],
            'shrink_factors': [8, 4, 2, 1],
            'smoothing_sigmas': [3.0, 2.0, 1.0, 0.0],
            'sampling_percentage': 0.25,
            'histogram_bins': 32,
            'learning_rate': 0.1,
            'convergence_threshold': 1e-6,
            # Template paths - USER MUST UPDATE THESE
            'template_path': '/path/to/your/template.nii.gz',
            'template_mask_path': '/path/to/your/template_mask.nii.gz'
        }
    }
    
    return default_config


def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(
        description="Standalone Brain MRI Registration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic registration
  python standalone_registration.py input.nii.gz output.nii.gz /path/to/transforms --config config.yaml

  # With custom log file
  python standalone_registration.py input.nii.gz output.nii.gz /path/to/transforms \\
    --config config.yaml --log_file registration.log

  # Create default config
  python standalone_registration.py --create-config
        """
    )
    
    parser.add_argument(
        "input_path",
        type=Path,
        nargs='?',
        help="Input NIfTI image path"
    )
    
    parser.add_argument(
        "output_path",
        type=Path,
        nargs='?',
        help="Output registered image path"
    )
    
    parser.add_argument(
        "transform_dir",
        type=Path,
        nargs='?',
        help="Directory for transforms and logs"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML configuration file"
    )
    
    parser.add_argument(
        "--log_file",
        type=Path,
        help="Log file path"
    )
    
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create default configuration file"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        config_path = Path("registration_config.yaml")
        default_config = create_default_config()
        
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
        
        print(f"Default configuration created: {config_path}")
        print("IMPORTANT: Please update the template_path in the config file!")
        return
    
    # Validate required arguments
    if not all([args.input_path, args.output_path, args.transform_dir]):
        parser.error("input_path, output_path, and transform_dir are required unless using --create-config")
    
    if not args.config:
        parser.error("--config is required for registration")
    
    # Check if files exist
    if not args.input_path.exists():
        print(f"Error: Input image not found: {args.input_path}")
        sys.exit(1)
    
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        print("Use --create-config to create a default configuration file")
        sys.exit(1)
    
    try:
        # Setup logging first
        if args.log_file:
            log_file = args.log_file
        else:
            # Create default log file in transform directory
            args.transform_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = args.transform_dir / f"registration_{timestamp}.log"
        
        setup_logging(log_file, args.verbose)
        
        logger.info("="*70)
        logger.info("Standalone Brain MRI Registration Tool Started")
        logger.info("="*70)
        logger.info(f"Input image: {args.input_path}")
        logger.info(f"Output image: {args.output_path}")
        logger.info(f"Transform directory: {args.transform_dir}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Log file: {log_file}")
        logger.info(f"Verbose mode: {args.verbose}")
        
        # Load configuration
        config = RegistrationConfig.from_yaml(args.config)
        
        # Get template path from config
        template_path = config.get_template_path()
        
        # Initialize registration tool
        tool = RegistrationTool(config)
        
        # Create output directory
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Perform registration
        result = tool.register_to_template(
            args.input_path,
            template_path,
            args.output_path,
            args.transform_dir
        )
        
        # Save results
        results_file = args.transform_dir / "registration_results.json"
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        logger.info("="*70)
        logger.info("Registration Summary")
        logger.info("="*70)
        
        if result.success:
            logger.info("STATUS: SUCCESS")
            logger.info(f"Registered image: {result.registered_image_path}")
            logger.info(f"Number of transforms: {len(result.transform_paths)}")
            logger.info(f"Number of inverse transforms: {len(result.inverse_transform_paths)}")
            
            print(f"\nRegistration completed successfully!")
            print(f"✓ Registered image: {result.registered_image_path}")
            print(f"✓ Transforms saved in: {args.transform_dir}")
            print(f"✓ Results file: {results_file}")
        else:
            logger.error("STATUS: FAILED")
            logger.error(f"Error: {result.error_message}")
            
            print(f"\nRegistration failed!")
            print(f"✗ Error: {result.error_message}")
            print(f"Check log file for details: {log_file}")
            sys.exit(1)
        
        logger.info("="*70)
        logger.info("Standalone Brain MRI Registration Tool Completed")
        logger.info("="*70)
    
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Registration failed: {e}")
            logger.exception("Full traceback:")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()