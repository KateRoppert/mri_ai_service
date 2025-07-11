#!/usr/bin/env python3
"""
Standalone Skull Stripping Module for Brain MRI Preprocessing

This module provides multiple skull stripping methods:
- FSL BET (Brain Extraction Tool)
- FreeSurfer (full recon-all pipeline)
- SynthStrip (FreeSurfer's deep learning method)

Can be used standalone or integrated into preprocessing pipelines.

Usage:
    python skull_stripping.py input.nii.gz output.nii.gz transform_dir --method FSL_BET
    python skull_stripping.py input.nii.gz output.nii.gz transform_dir --method SynthStrip --config config.yaml
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import tempfile
import os 

try:
    import SimpleITK as sitk
    import yaml
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages: pip install SimpleITK PyYAML")
    sys.exit(1)

# Try importing nipype (optional for FSL)
try:
    from nipype.interfaces.fsl import BET
    from nipype.interfaces.base import Undefined
    NIPYPE_AVAILABLE = True
except ImportError:
    NIPYPE_AVAILABLE = False

# Setup logging
logger = logging.getLogger(__name__)


class SkullStrippingStrategy(ABC):
    """Abstract base class for skull stripping strategies"""
    
    @abstractmethod
    def strip_skull(self, input_path: Path, output_path: Path, 
                   mask_path: Optional[Path] = None, params: Dict = None) -> Dict:
        """
        Perform skull stripping on the input image
        
        Returns:
            Dict with results including paths and success status
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging"""
        pass
    
    @abstractmethod
    def check_dependencies(self) -> Tuple[bool, str]:
        """
        Check if required dependencies are available
        Returns: (is_available, error_message)
        """
        pass


class FSLBETStrategy(SkullStrippingStrategy):
    """FSL BET skull stripping strategy"""
    
    def strip_skull(self, input_path: Path, output_path: Path, 
                   mask_path: Optional[Path] = None, params: Dict = None) -> Dict:
        params = params or {}
        
        # Use nipype if available, otherwise fall back to subprocess
        if NIPYPE_AVAILABLE:
            return self._strip_with_nipype(input_path, output_path, mask_path, params)
        else:
            return self._strip_with_subprocess(input_path, output_path, mask_path, params)
    
    def _strip_with_nipype(self, input_path: Path, output_path: Path,
                          mask_path: Optional[Path], params: Dict) -> Dict:
        """Use nipype interface for BET"""
        try:
            # Create temporary base for BET output
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_base = Path(temp_dir) / "bet_temp"
                
                bet = BET()
                bet.inputs.in_file = str(input_path.resolve())
                bet.inputs.out_file = str(temp_base.resolve())
                bet.inputs.frac = params.get('bet_fractional_intensity_threshold', 0.5)
                bet.inputs.robust = params.get('bet_robust', True)
                bet.inputs.mask = True
                bet.inputs.output_type = 'NIFTI_GZ'
                
                if params.get('bet_options'):
                    bet.inputs.args = params.get('bet_options')
                
                logger.debug(f"Executing FSL BET command: {bet.cmdline}")
                result = bet.run()
                
                # Check for success
                if result.runtime.returncode != 0:
                    error_msg = f"FSL BET failed with exit code {result.runtime.returncode}"
                    if result.runtime.stderr:
                        error_msg += f"\nSTDERR: {result.runtime.stderr}"
                    raise RuntimeError(error_msg)
                
                # Move output files
                expected_stripped = Path(f"{temp_base.resolve()}.nii.gz")
                expected_mask = Path(f"{temp_base.resolve()}_mask.nii.gz")
                
                if not expected_stripped.exists():
                    raise FileNotFoundError(f"BET output not found: {expected_stripped}")
                
                shutil.move(str(expected_stripped), str(output_path))
                
                results = {
                    'method': 'FSL_BET',
                    'output_path': str(output_path),
                    'success': True
                }
                
                if mask_path and expected_mask.exists():
                    shutil.move(str(expected_mask), str(mask_path))
                    results['mask_path'] = str(mask_path)
                
                return results
                
        except Exception as e:
            logger.error(f"FSL BET (nipype) failed: {e}")
            return {
                'method': 'FSL_BET',
                'output_path': str(output_path),
                'success': False,
                'error': str(e)
            }
    
    def _strip_with_subprocess(self, input_path: Path, output_path: Path,
                             mask_path: Optional[Path], params: Dict) -> Dict:
        """Use subprocess to call bet directly"""
        try:
            # Build command
            cmd = [
                "bet",
                str(input_path.resolve()),
                str(output_path.resolve()),
                "-f", str(params.get('bet_fractional_intensity_threshold', 0.5))
            ]
            
            if params.get('bet_robust', True):
                cmd.append("-R")
            
            if mask_path:
                cmd.append("-m")
            
            if params.get('bet_options'):
                cmd.extend(params['bet_options'].split())
            
            logger.debug(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = f"FSL BET failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                raise RuntimeError(error_msg)
            
            results = {
                'method': 'FSL_BET',
                'output_path': str(output_path),
                'success': True
            }
            
            # Check for mask file
            if mask_path:
                # BET creates mask with specific naming
                bet_mask = output_path.parent / f"{output_path.stem}_mask.nii.gz"
                if bet_mask.exists():
                    shutil.move(str(bet_mask), str(mask_path))
                    results['mask_path'] = str(mask_path)
            
            return results
            
        except Exception as e:
            logger.error(f"FSL BET (subprocess) failed: {e}")
            return {
                'method': 'FSL_BET',
                'output_path': str(output_path),
                'success': False,
                'error': str(e)
            }
    
    def get_strategy_name(self) -> str:
        return "FSL_BET"
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if FSL is available"""
        if shutil.which("bet") is None:
            return False, "FSL 'bet' command not found in PATH"
        return True, ""


class FreeSurferStrategy(SkullStrippingStrategy):
    """FreeSurfer recon-all skull stripping strategy"""
    
    def strip_skull(self, input_path: Path, output_path: Path,
                   mask_path: Optional[Path] = None, params: Dict = None) -> Dict:
        params = params or {}
        
        # Check if input is T1w (FreeSurfer requirement)
        if not self._is_t1_image(input_path):
            logger.warning(f"FreeSurfer recon-all requires T1w images. Skipping {input_path.name}")
            # Just copy the input to output
            shutil.copy2(str(input_path), str(output_path))
            return {
                'method': 'FreeSurfer',
                'output_path': str(output_path),
                'success': True,
                'warning': 'Input is not T1w, copied without processing'
            }
        
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_fs_dir = Path(temp_dir) / "freesurfer_subjects"
                temp_fs_dir.mkdir()
                subject_id = input_path.stem
                
                # Build recon-all command
                cmd = [
                    "recon-all",
                    "-i", str(input_path.resolve()),
                    "-subjid", subject_id,
                    "-sd", str(temp_fs_dir.resolve()),
                    "-all"
                ]
                
                if params.get('parallel', False):
                    cmd.extend(["-parallel", "-openmp", str(params.get('threads', 4))])
                
                logger.info(f"Running FreeSurfer recon-all (this may take several hours)...")
                logger.debug(f"Command: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    error_msg = f"FreeSurfer recon-all failed with exit code {result.returncode}"
                    if result.stderr:
                        error_msg += f"\nSTDERR: {result.stderr[-1000:]}"  # Last 1000 chars
                    raise RuntimeError(error_msg)
                
                # Extract brain.mgz
                brain_mgz = temp_fs_dir / subject_id / "mri" / "brain.mgz"
                if not brain_mgz.exists():
                    raise FileNotFoundError(f"FreeSurfer brain.mgz not found: {brain_mgz}")
                
                # Convert to NIfTI
                brain_img = sitk.ReadImage(str(brain_mgz))
                sitk.WriteImage(brain_img, str(output_path))
                
                results = {
                    'method': 'FreeSurfer',
                    'output_path': str(output_path),
                    'success': True
                }
                
                # Extract mask if requested
                if mask_path:
                    brainmask_mgz = temp_fs_dir / subject_id / "mri" / "brainmask.mgz"
                    if brainmask_mgz.exists():
                        mask_img = sitk.ReadImage(str(brainmask_mgz))
                        sitk.WriteImage(mask_img, str(mask_path))
                        results['mask_path'] = str(mask_path)
                
                return results
                
        except Exception as e:
            logger.error(f"FreeSurfer recon-all failed: {e}")
            return {
                'method': 'FreeSurfer',
                'output_path': str(output_path),
                'success': False,
                'error': str(e)
            }
    
    def _is_t1_image(self, image_path: Path) -> bool:
        """Check if image is T1-weighted based on filename"""
        filename_lower = image_path.name.lower()
        return "t1w" in filename_lower and "ce" not in filename_lower
    
    def get_strategy_name(self) -> str:
        return "FreeSurfer"
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if FreeSurfer is available"""
        if not shutil.which("recon-all"):
            return False, "FreeSurfer 'recon-all' command not found"
        if not os.environ.get("FREESURFER_HOME"):
            return False, "FREESURFER_HOME environment variable not set"
        return True, ""


class SynthStripStrategy(SkullStrippingStrategy):
    """FreeSurfer SynthStrip skull stripping strategy"""
    
    def strip_skull(self, input_path: Path, output_path: Path,
                   mask_path: Optional[Path] = None, params: Dict = None) -> Dict:
        params = params or {}
        
        try:
            # Build command
            cmd = [
                "mri_synthstrip",
                "-i", str(input_path.resolve()),
                "-o", str(output_path.resolve())
            ]
            
            # Add mask output if requested
            if mask_path or params.get('save_mask', False):
                if not mask_path:
                    mask_path = output_path.parent / f"{output_path.stem}_mask.nii.gz"
                cmd.extend(["-m", str(mask_path.resolve())])
            
            # Add border parameter
            if 'border' in params:
                cmd.extend(["-b", str(params['border'])])
            
            # Add GPU options
            if params.get('gpu', False) and 'gpu_device' in params:
                cmd.extend(["-g", "-d", str(params['gpu_device'])])
            
            logger.debug(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                error_msg = f"SynthStrip failed with exit code {result.returncode}"
                if result.stderr:
                    error_msg += f"\nSTDERR: {result.stderr}"
                raise RuntimeError(error_msg)
            
            if not output_path.exists():
                raise FileNotFoundError(f"SynthStrip output not found: {output_path}")
            
            results = {
                'method': 'SynthStrip',
                'output_path': str(output_path),
                'success': True
            }
            
            if mask_path and mask_path.exists():
                results['mask_path'] = str(mask_path)
            
            return results
            
        except Exception as e:
            logger.error(f"SynthStrip failed: {e}")
            return {
                'method': 'SynthStrip',
                'output_path': str(output_path),
                'success': False,
                'error': str(e)
            }
    
    def get_strategy_name(self) -> str:
        return "SynthStrip"
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if SynthStrip is available"""
        if not shutil.which("mri_synthstrip"):
            return False, "FreeSurfer 'mri_synthstrip' command not found"
        if not os.environ.get("FREESURFER_HOME"):
            return False, "FREESURFER_HOME environment variable not set"
        return True, ""


class SkullStripper:
    """Main skull stripping processor"""
    
    STRATEGIES = {
        "FSL_BET": FSLBETStrategy,
        "FreeSurfer": FreeSurferStrategy,
        "SynthStrip": SynthStripStrategy
    }
    
    def __init__(self, method: str = "FSL_BET"):
        """
        Initialize skull stripper
        
        Args:
            method: Skull stripping method to use
        """
        if method not in self.STRATEGIES:
            raise ValueError(f"Unknown skull stripping method: {method}. "
                           f"Available: {list(self.STRATEGIES.keys())}")
        
        self.method = method
        self.strategy = self.STRATEGIES[method]()
        
        # Check dependencies
        is_available, error_msg = self.strategy.check_dependencies()
        if not is_available:
            raise RuntimeError(f"Dependencies not met for {method}: {error_msg}")
        
        logger.info(f"Initialized {method} skull stripping strategy")
    
    def process_image(self, input_path: Union[str, Path], output_path: Union[str, Path],
                     transform_dir: Union[str, Path], save_mask: bool = True,
                     config_params: Optional[Dict] = None) -> Dict:
        """
        Process a single image for skull stripping
        
        Args:
            input_path: Path to input NIfTI image
            output_path: Path to save skull-stripped image
            transform_dir: Directory to save masks and logs
            save_mask: Whether to save the brain mask
            config_params: Additional configuration parameters
            
        Returns:
            Dictionary with processing results
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        transform_dir = Path(transform_dir)
        
        # Create directories
        output_path.parent.mkdir(parents=True, exist_ok=True)
        transform_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare mask path
        mask_path = None
        if save_mask or (config_params and config_params.get('save_mask', False)):
            mask_path = transform_dir / f"{input_path.stem}_brain_mask.nii.gz"
        
        logger.info(f"Processing {input_path.name} with {self.method}")
        
        # Run skull stripping
        results = self.strategy.strip_skull(
            input_path=input_path,
            output_path=output_path,
            mask_path=mask_path,
            params=config_params
        )
        
        # Add metadata
        results['input_path'] = str(input_path)
        results['transform_dir'] = str(transform_dir)
        
        # Save results
        results_path = transform_dir / "skull_stripping_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        if results['success']:
            logger.info(f"Skull stripping completed successfully. Output: {output_path}")
        else:
            logger.error(f"Skull stripping failed: {results.get('error', 'Unknown error')}")
        
        return results


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
        
        logger.info(f"Logging setup complete. Log file: {log_path}")


def load_config(config_path: Path) -> Dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {e}")
        return {}


def extract_skull_stripping_config(config: Dict) -> Dict:
    """Extract skull stripping configuration from nested config structure"""
    try:
        # Navigate through the nested structure
        steps = config.get('steps', {})
        preprocessing = steps.get('preprocessing', {})
        skull_stripping_config = preprocessing.get('skull_stripping', {})
        
        logger.debug(f"Extracted skull stripping config: {skull_stripping_config}")
        
        return skull_stripping_config
    except Exception as e:
        logger.warning(f"Error extracting skull stripping configuration: {e}")
        return {}


# Pipeline Integration Interface
class SkullStrippingStep:
    """Pipeline integration interface for skull stripping"""
    
    def __init__(self, processor, params: dict, dependency_checker=None):
        self.processor = processor
        self.params = params
        self.dependency_checker = dependency_checker
        self.step_name = "4_skull_stripping"
        
        # Get method from params
        self.method = params.get('method', 'FSL_BET')
        
        # Initialize skull stripper
        try:
            self.skull_stripper = SkullStripper(method=self.method)
        except RuntimeError as e:
            logger.error(f"Failed to initialize skull stripper: {e}")
            raise
    
    def run_step(self):
        """Execute skull stripping step in pipeline context"""
        try:
            # This step produces the final output
            output_path = self.processor.final_output_path
            
            # Process the image
            results = self.skull_stripper.process_image(
                input_path=self.processor.current_input_path,
                output_path=output_path,
                transform_dir=self.processor.transform_dir,
                save_mask=self.params.get('save_mask', True),
                config_params=self.params
            )
            
            if not results['success']:
                raise RuntimeError(f"Skull stripping failed: {results.get('error', 'Unknown error')}")
            
            # Update processor state
            self.processor.current_input_path = output_path
            
            # Store results in processor
            if hasattr(self.processor, 'skull_stripping_results'):
                self.processor.skull_stripping_results = results
            
        except Exception as e:
            logger.error(f"Skull stripping step failed: {e}")
            raise


def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(
        description="Standalone Skull Stripping for Brain MRI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_path", type=Path, help="Input NIfTI image path")
    parser.add_argument("output_path", type=Path, help="Output skull-stripped image path")
    parser.add_argument("transform_dir", type=Path, help="Directory for masks and logs")
    
    parser.add_argument("--method", type=str, default="FSL_BET",
                       choices=["FSL_BET", "FreeSurfer", "SynthStrip"],
                       help="Skull stripping method")
    parser.add_argument("--save-mask", action="store_true", default=True,
                       help="Save brain mask")
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--log-file", type=Path, help="Log file path")
    parser.add_argument("--console-log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Console logging level")
    
    # Method-specific options
    parser.add_argument("--frac", type=float, help="FSL BET fractional intensity threshold")
    parser.add_argument("--robust", action="store_true", help="FSL BET robust brain center estimation")
    parser.add_argument("--border", type=int, help="SynthStrip border size")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for SynthStrip")
    parser.add_argument("--gpu-device", type=int, default=0, help="GPU device ID for SynthStrip")
    
    args = parser.parse_args()
    
    # Setup logging
    log_path = args.log_file or args.transform_dir / "skull_stripping.log"
    setup_logging(log_path, args.console_log_level)
    
    # Load configuration
    config_params = {}
    if args.config:
        full_config = load_config(args.config)
        config_params = extract_skull_stripping_config(full_config)
        logger.info(f"Loaded config parameters: {list(config_params.keys())}")
    
    # Override with command line arguments
    if args.frac is not None:
        config_params['bet_fractional_intensity_threshold'] = args.frac
    if args.robust:
        config_params['bet_robust'] = True
    if args.border is not None:
        config_params['border'] = args.border
    if args.gpu:
        config_params['gpu'] = True
        config_params['gpu_device'] = args.gpu_device
    if args.save_mask is not None:
        config_params['save_mask'] = args.save_mask
    
    try:
        # Initialize and run skull stripper
        skull_stripper = SkullStripper(method=args.method)
        results = skull_stripper.process_image(
            input_path=args.input_path,
            output_path=args.output_path,
            transform_dir=args.transform_dir,
            save_mask=args.save_mask,
            config_params=config_params
        )
        
        if results['success']:
            logger.info("Skull stripping completed successfully")
            sys.exit(0)
        else:
            logger.error("Skull stripping failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()