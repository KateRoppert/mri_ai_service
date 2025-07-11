"""
Modular Brain MRI Registration System for BIDS datasets
Can be used standalone or imported into preprocessing pipelines
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


def setup_logging(output_dir: Path, verbose: bool = False) -> None:
    """Setup logging to both console and file in output directory"""
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f"registration_{timestamp}.log"
    
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
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log debug to file
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure module logger
    logger.setLevel(logging.DEBUG)
    
    logger.info(f"Logging initialized. Log file: {log_file}")


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
    strategy: str = "hybrid"
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
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Extract registration config
        reg_config = config_data.get('registration', {})
        
        # Create instance with defaults, then update with config
        instance = cls()
        for key, value in reg_config.items():
            if hasattr(instance, key):
                setattr(instance, key, value)
        
        return instance
    
    def get_template_path(self) -> Path:
        """Get template path, raise error if not set"""
        if not self.template_path:
            raise ValueError("template_path not specified in configuration file")
        
        template_path = Path(self.template_path)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        return template_path


class BIDSNavigator:
    """Helper class for navigating BIDS structure"""
    
    def __init__(self, bids_root: Path, output_root: Optional[Path] = None):
        self.bids_root = Path(bids_root)
        self.derivatives_root = output_root if output_root else self.bids_root.parent / "results"
    
    def get_subject_sessions(self, subject: str) -> List[str]:
        """Get all sessions for a subject"""
        subject_dir = self.bids_root / f"sub-{subject}"
        if not subject_dir.exists():
            return []
        
        sessions = []
        for item in subject_dir.iterdir():
            if item.is_dir() and item.name.startswith('ses-'):
                sessions.append(item.name[4:])  # Remove 'ses-' prefix
        
        return sessions if sessions else [None]  # None for no sessions
    
    def get_modality_files(self, subject: str, session: Optional[str] = None) -> Dict[str, Path]:
        """Get all modality files for a subject/session"""
        if session:
            anat_dir = self.bids_root / f"sub-{subject}" / f"ses-{session}" / "anat"
        else:
            anat_dir = self.bids_root / f"sub-{subject}" / "anat"
        
        if not anat_dir.exists():
            return {}
        
        modalities = {}
        for nii_file in anat_dir.glob("*.nii.gz"):
            # Parse BIDS filename to extract modality
            parts = nii_file.name.replace('.nii.gz', '').split('_')
            
            # Find modality part (it could be T1w, FLAIR, T2w, or ce-GAD_T1w)
            modality_part = None
            if 'ce-GAD_T1w' in nii_file.name:
                modality_part = 'ce-GAD_T1w'
            else:
                for part in parts:
                    if part in ['T1w', 'T2w', 'FLAIR']:
                        modality_part = part
                        break
            
            if not modality_part:
                continue
            
            # Map to standard modality names
            if modality_part == 'ce-GAD_T1w':
                modality = 'T1CE'
            elif modality_part == 'T1w':
                modality = 'T1'
            elif modality_part == 'T2w':
                modality = 'T2'
            elif modality_part == 'FLAIR':
                modality = 'FLAIR'
            else:
                continue
            
            modalities[modality] = nii_file
        
        return modalities
    
    def create_derivatives_structure(self, subject: str, session: Optional[str] = None, 
                                   step_name: str = "registration") -> Path:
        """Create derivatives directory structure"""
        if session:
            output_dir = (self.derivatives_root / 
                         f"sub-{subject}" / f"ses-{session}" / step_name)
        else:
            output_dir = (self.derivatives_root / 
                         f"sub-{subject}" / step_name)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir


class RegistrationTool:
    """Base class for registration tools"""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.validate_dependencies()
    
    def validate_dependencies(self):
        """Validate tool dependencies"""
        if self.config.tool.upper() == "ANTS":
            self._validate_ants()
        elif self.config.tool.upper() == "FSL":
            self._validate_fsl()
    
    def _validate_ants(self):
        """Validate ANTs dependencies"""
        try:
            import ants
            self.ants = ants
        except ImportError:
            raise ImportError("ANTs Python package (antspyx) required but not installed")
        
        required_binaries = ['antsRegistration', 'antsApplyTransforms']
        for binary in required_binaries:
            if shutil.which(binary) is None:
                raise RuntimeError(f"ANTs binary '{binary}' not found in PATH")
    
    def _validate_fsl(self):
        """Validate FSL dependencies"""
        required_binaries = ['flirt', 'fnirt', 'applywarp']
        for binary in required_binaries:
            if shutil.which(binary) is None:
                raise RuntimeError(f"FSL binary '{binary}' not found in PATH")
        
        if 'FSLDIR' not in os.environ:
            raise RuntimeError("FSLDIR environment variable not set")
    
    def register_to_template(self, moving_image: Path, template: Path, 
                           output_prefix: Path) -> RegistrationResult:
        """Register image to template"""
        if self.config.tool.upper() == "ANTS":
            return self._ants_register(moving_image, template, output_prefix)
        elif self.config.tool.upper() == "FSL":
            return self._fsl_register(moving_image, template, output_prefix)
    
    def _ants_register(self, moving_image: Path, template: Path, 
                      output_prefix: Path) -> RegistrationResult:
        """ANTs registration implementation"""
        try:
            logger.info(f"Starting ANTs registration: {moving_image.name} -> {template.name}")
            
            fixed_img = self.ants.image_read(str(template))
            moving_img = self.ants.image_read(str(moving_image))
            
            # Determine transform type
            if self.config.transform_type.lower() == "rigid":
                type_of_transform = "Rigid"
            elif self.config.transform_type.lower() == "affine":
                type_of_transform = "Affine"
            else:
                type_of_transform = "SyN"
            
            logger.debug(f"Using transform type: {type_of_transform}")
            
            # Perform registration
            reg_result = self.ants.registration(
                fixed=fixed_img,
                moving=moving_img,
                type_of_transform=type_of_transform,
                outprefix=str(output_prefix),
                verbose=True
            )
            
            # Save result
            output_path = output_prefix.parent / f"{output_prefix.name}Warped.nii.gz"
            self.ants.image_write(reg_result['warpedmovout'], str(output_path))
            
            logger.info(f"ANTs registration completed successfully: {output_path}")
            
            return RegistrationResult(
                registered_image_path=output_path,
                transform_paths=[Path(t) for t in reg_result.get('fwdtransforms', [])],
                inverse_transform_paths=[Path(t) for t in reg_result.get('invtransforms', [])],
                success=True
            )
            
        except Exception as e:
            logger.error(f"ANTs registration failed: {e}")
            return RegistrationResult(
                registered_image_path=Path(""),
                success=False,
                error_message=str(e)
            )
    
    def _fsl_register(self, moving_image: Path, template: Path, 
                     output_prefix: Path) -> RegistrationResult:
        """FSL registration implementation"""
        try:
            logger.info(f"Starting FSL registration: {moving_image.name} -> {template.name}")
            
            output_path = output_prefix.parent / f"{output_prefix.name}_registered.nii.gz"
            linear_matrix = output_prefix.parent / f"{output_prefix.name}_linear.mat"
            
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
            
            logger.debug(f"Running FLIRT command: {' '.join(flirt_cmd)}")
            subprocess.run(flirt_cmd, check=True, capture_output=True)
            
            transform_paths = [linear_matrix]
            
            # Non-linear registration if requested
            if self.config.transform_type.lower() in ["syn", "nonlinear"]:
                logger.info("Running non-linear registration with FNIRT")
                warp_field = output_prefix.parent / f"{output_prefix.name}_warp.nii.gz"
                
                fnirt_cmd = [
                    "fnirt",
                    f"--in={moving_image}",
                    f"--ref={template}",
                    f"--aff={linear_matrix}",
                    f"--iout={output_path}",
                    f"--cout={warp_field}"
                ]
                
                logger.debug(f"Running FNIRT command: {' '.join(fnirt_cmd)}")
                subprocess.run(fnirt_cmd, check=True, capture_output=True)
                transform_paths.append(warp_field)
            
            logger.info(f"FSL registration completed successfully: {output_path}")
            
            return RegistrationResult(
                registered_image_path=output_path,
                transform_paths=transform_paths,
                success=True
            )
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FSL registration failed: {e}")
            return RegistrationResult(
                registered_image_path=Path(""),
                success=False,
                error_message=str(e)
            )


class RegistrationPipeline:
    """Main registration pipeline class"""
    
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.template_path = config.get_template_path()  # Get from config
        self.tool = RegistrationTool(config)
        
        logger.info(f"Initialized registration pipeline with template: {self.template_path}")
        logger.info(f"Using tool: {config.tool}, strategy: {config.strategy}")
    
    def process_subject(self, bids_root: Path, subject: str, 
                   session: Optional[str] = None,
                   output_root: Optional[Path] = None) -> Dict[str, RegistrationResult]:
        """Process a single subject"""
        navigator = BIDSNavigator(bids_root, output_root)
        
        # Get subject's modality files
        modalities = navigator.get_modality_files(subject, session)
        if not modalities:
            raise ValueError(f"No modality files found for subject {subject}")
        
        logger.info(f"Found modalities for sub-{subject}: {list(modalities.keys())}")
        
        # Create output directory
        output_dir = navigator.create_derivatives_structure(subject, session, "registration")
        
        # Process based on strategy
        if self.config.strategy == "intra":
            return self._intra_subject_registration(modalities, output_dir)
        elif self.config.strategy == "inter":
            return self._inter_subject_registration(modalities, output_dir)
        elif self.config.strategy == "hybrid":
            return self._hybrid_registration(modalities, output_dir)
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _intra_subject_registration(self, modalities: Dict[str, Path], 
                                  output_dir: Path) -> Dict[str, RegistrationResult]:
        """Register all modalities to T1 within subject"""
        results = {}
        
        # Use T1 as reference
        reference_key = "T1" if "T1" in modalities else list(modalities.keys())[0]
        reference_image = modalities[reference_key]
        
        logger.info(f"Using {reference_key} as intra-subject reference")
        
        for modality, image_path in modalities.items():
            if modality == reference_key:
                # Copy reference image
                output_path = output_dir / f"sub-{modality}_registered.nii.gz"
                shutil.copy2(image_path, output_path)
                results[modality] = RegistrationResult(
                    registered_image_path=output_path,
                    success=True
                )
                logger.info(f"Copied reference image {modality}: {output_path}")
            else:
                # Register to reference
                output_prefix = output_dir / f"{modality}_to_{reference_key}"
                result = self.tool.register_to_template(image_path, reference_image, output_prefix)
                results[modality] = result
        
        return results
    
    def _inter_subject_registration(self, modalities: Dict[str, Path], 
                                  output_dir: Path) -> Dict[str, RegistrationResult]:
        """Register all modalities to template"""
        results = {}
        
        for modality, image_path in modalities.items():
            output_prefix = output_dir / f"{modality}_to_template"
            result = self.tool.register_to_template(image_path, self.template_path, output_prefix)
            results[modality] = result
        
        return results
    
    def _hybrid_registration(self, modalities: Dict[str, Path], 
                           output_dir: Path) -> Dict[str, RegistrationResult]:
        """Two-step registration: intra-subject then inter-subject"""
        logger.info("Starting hybrid registration (intra-subject + inter-subject)")
        
        # Step 1: Intra-subject registration
        intra_dir = output_dir / "intra_subject"
        intra_dir.mkdir(exist_ok=True)
        
        logger.info("Step 1: Intra-subject registration")
        intra_results = self._intra_subject_registration(modalities, intra_dir)
        
        # Step 2: Register aligned images to template
        inter_dir = output_dir / "final"
        inter_dir.mkdir(exist_ok=True)
        
        logger.info("Step 2: Inter-subject registration to template")
        final_results = {}
        for modality, intra_result in intra_results.items():
            if intra_result.success:
                output_prefix = inter_dir / f"{modality}_to_template"
                result = self.tool.register_to_template(
                    intra_result.registered_image_path, 
                    self.template_path, 
                    output_prefix
                )
                
                # Combine transform paths
                if result.success:
                    result.transform_paths = intra_result.transform_paths + result.transform_paths
                
                final_results[modality] = result
            else:
                final_results[modality] = intra_result
        
        return final_results
    
    def save_results(self, results: Dict[str, RegistrationResult], 
                    output_path: Path):
        """Save registration results to JSON"""
        results_dict = {
            modality: result.to_dict() 
            for modality, result in results.items()
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


def create_default_config():
    """Create default registration configuration file"""
    default_config = {
        'registration': {
            'tool': 'ANTs',
            'strategy': 'hybrid',
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

class RegistrationStep:
    """Pipeline integration interface"""
    
    def __init__(self, processor, params: dict, dependency_checker=None):
        self.processor = processor
        self.params = params
        self.step_name = "3_registration"
        
        # Create a copy of params to modify
        config_params = params.copy()
        
        # Check if template_path is provided via processor (from command line)
        if hasattr(processor, 'template_path') and processor.template_path:
            config_params['template_path'] = str(processor.template_path)
        
        # Also check in the main config paths section
        if 'template_path' not in config_params and hasattr(processor, 'config'):
            # Try to get from the main config paths section
            main_config = processor.config._data
            if 'paths' in main_config and 'template_path' in main_config['paths']:
                config_params['template_path'] = main_config['paths']['template_path']
        
        # Load config with template path included
        if 'template_path' in config_params:
            self.config = RegistrationConfig()
            self.config.template_path = config_params['template_path']
            # Update other config parameters
            for key, value in config_params.items():
                if hasattr(self.config, key) and key != 'template_path':
                    setattr(self.config, key, value)
        else:
            # Fall back to loading from config file if available
            config_path = params.get('config_path')
            if config_path and Path(config_path).exists():
                self.config = RegistrationConfig.from_yaml(Path(config_path))
            else:
                self.config = RegistrationConfig()
                # Update with params
                reg_params = params.get('registration', {})
                for key, value in reg_params.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
        
        # Initialize pipeline
        self.pipeline = RegistrationPipeline(self.config)
    
    def run_step(self):
        """Execute registration step"""
        try:
            # Extract subject info from processor
            subject_id = getattr(self.processor, 'subject_id', 'unknown')
            session_id = getattr(self.processor, 'session_id', None)
            bids_root = getattr(self.processor, 'bids_root', Path.cwd())
            
            # For preprocessing pipeline integration, we need to handle single file processing
            if hasattr(self.processor, 'current_input_path') and hasattr(self.processor, 'temp_dir'):
                # Single file mode for preprocessing pipeline
                output_path = self.processor.temp_dir / f"{self.processor.file_stem}_{self.step_name}.nii.gz"
                
                # Get template path
                template_path = Path(self.config.template_path) if self.config.template_path else self.processor.template_path
                
                # Create registration tool directly
                tool = RegistrationTool(self.config)
                
                # Register single file
                output_prefix = self.processor.transform_dir / f"{self.processor.file_stem}_reg_"
                result = tool.register_to_template(
                    self.processor.current_input_path,
                    template_path,
                    output_prefix
                )
                
                if not result.success:
                    raise RuntimeError(f"Registration failed: {result.error_message}")
                
                # Update processor state
                self.processor.current_input_path = result.registered_image_path
                
                # Store results
                if hasattr(self.processor, 'registration_results'):
                    self.processor.registration_results = result
            else:
                # Original BIDS dataset mode
                results = self.pipeline.process_subject(bids_root, subject_id, session_id)
                
                # Update processor with results
                self.processor.registration_results = results
                
                # Update current input path to registered image
                current_modality = self._infer_current_modality()
                if current_modality in results and results[current_modality].success:
                    self.processor.current_input_path = results[current_modality].registered_image_path
            
        except Exception as e:
            logger.error(f"Registration step failed: {e}")
            raise
    
    def _infer_current_modality(self) -> str:
        """Infer current modality from processor"""
        if hasattr(self.processor, 'current_input_path'):
            filename = self.processor.current_input_path.name.lower()
            if 't1ce' in filename or 't1c' in filename or 'ce-gd' in filename:
                return 'T1CE'
            elif 't1' in filename:
                return 'T1'
            elif 't2' in filename:
                return 'T2'
            elif 'flair' in filename:
                return 'FLAIR'
        return 'T1'  # Default


def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(
        description="Brain MRI Registration for BIDS datasets"
    )
    
    parser.add_argument(
        "bids_root",
        type=Path,
        help="Path to BIDS dataset root directory"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to registration configuration file (contains template path)"
    )
    
    parser.add_argument(
        "--subject",
        required=True,
        help="Subject ID (without 'sub-' prefix)"
    )
    
    parser.add_argument(
        "--session",
        help="Session ID (without 'ses-' prefix)"
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        help="Root directory for output results (default: bids_root/../results)"
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
    
    # Check if config file exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        print("Use --create-config to create a default configuration file")
        sys.exit(1)
    
    try:
        # Determine output directory for logging
        navigator = BIDSNavigator(args.bids_root, args.output_root)
        output_dir = navigator.create_derivatives_structure(
            args.subject, args.session, "registration"
        )
        
        # Setup logging BEFORE other operations
        setup_logging(output_dir, args.verbose)
        
        logger.info("="*60)
        logger.info("Brain MRI Registration Pipeline Started")
        logger.info("="*60)
        logger.info(f"BIDS root: {args.bids_root}")
        logger.info(f"Config file: {args.config}")
        logger.info(f"Subject: {args.subject}")
        logger.info(f"Session: {args.session}")
        logger.info(f"Output directory: {output_dir}")
        
        # Load configuration
        config = RegistrationConfig.from_yaml(args.config)
        logger.info(f"Loaded configuration: tool={config.tool}, strategy={config.strategy}")
        logger.info(f"Template path from config: {config.template_path}")
        
        # Initialize pipeline
        pipeline = RegistrationPipeline(config)
        
        # Process subject
        results = pipeline.process_subject(
            args.bids_root, 
            args.subject, 
            args.session,
            args.output_root
        )
        
        # Save results
        results_file = output_dir / "registration_results.json"
        pipeline.save_results(results, results_file)
        
        # Print summary
        successful = sum(1 for r in results.values() if r.success)
        total = len(results)
        
        logger.info("="*60)
        logger.info("Registration Summary")
        logger.info("="*60)
        logger.info(f"Total modalities processed: {total}")
        logger.info(f"Successfully registered: {successful}")
        logger.info(f"Failed: {total - successful}")
        
        print(f"\nRegistration completed: {successful}/{total} modalities successful")
        
        for modality, result in results.items():
            status = "✓" if result.success else "✗"
            print(f"  {status} {modality}: {result.registered_image_path}")
            if result.success:
                logger.info(f"SUCCESS - {modality}: {result.registered_image_path}")
            else:
                logger.error(f"FAILED - {modality}: {result.error_message}")
        
        logger.info("="*60)
        logger.info("Brain MRI Registration Pipeline Completed")
        logger.info("="*60)
    
    except Exception as e:
        logger.error(f"Registration failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

if __name__ == "__main__":
    main()