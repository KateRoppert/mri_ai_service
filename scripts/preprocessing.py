"""
Preprocessing Orchestrator Script.

This script orchestrates various preprocessing steps in a configurable order.
"""

import argparse
import logging
import sys
import json
import subprocess
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import tempfile

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


class Config:
    """Handles configuration loading and access."""
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self._data = yaml.safe_load(f)
            
    def get_preprocessing_config(self) -> Dict:
        """Get preprocessing configuration."""
        return self._data.get('steps', {}).get('preprocessing', {})
    
    def get_step_config(self, step_name: str) -> Dict:
        """Get configuration for a specific step."""
        preprocessing = self.get_preprocessing_config()
        return preprocessing.get(step_name, {})
    
    def get_enabled_steps(self) -> List[str]:
        """Get list of enabled preprocessing steps in order."""
        preprocessing = self.get_preprocessing_config()
        step_order = preprocessing.get('step_order', [
            'intensity_normalization',
            'bias_field_correction', 
            'registration',
            'skull_stripping'
        ])
        
        # Filter only enabled steps
        enabled_steps = []
        for step in step_order:
            step_config = self.get_step_config(step)
            if step_config.get('enabled', False):
                enabled_steps.append(step)
                
        return enabled_steps
    
    def get_template_path(self) -> Path:
        """Get template path from config."""
        template_path = self._data.get('paths', {}).get('template_path')
        if not template_path:
            # Fallback to preprocessing config
            template_path = self.get_step_config('registration').get('template_path')
        return Path(template_path) if template_path else None
    
    def get_executables(self) -> Dict:
        """Get executable paths."""
        return self._data.get('executables', {})


class PreprocessingStep(ABC):
    """Abstract base class for preprocessing steps."""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.python_executable = sys.executable
        
    @abstractmethod
    def get_command(self, input_dir: Path, output_dir: Path, 
                   work_dir: Path, **kwargs) -> List[str]:
        """Build command for this step."""
        pass
    
    def execute(self, input_dir: Path, output_dir: Path, 
               work_dir: Path, **kwargs) -> Dict:
        """Execute the preprocessing step."""
        logger.info(f"Executing step: {self.name}")
        
        # Build command
        command = self.get_command(input_dir, output_dir, work_dir, **kwargs)
        
        # Log file for this step
        log_file = work_dir / f"{self.name}.log"
        
        # Add log file to command if supported
        if '--log_file' not in ' '.join(command):
            command.extend(['--log_file', str(log_file)])
        
        logger.debug(f"Command: {' '.join(command)}")
        
        # Execute command
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Step {self.name} completed successfully")
            
            return {
                "status": "success",
                "command": command,
                "log_file": str(log_file),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Step {self.name} failed with exit code {e.returncode}")
            logger.error(f"stderr: {e.stderr}")
            
            return {
                "status": "failed",
                "command": command,
                "log_file": str(log_file),
                "exit_code": e.returncode,
                "stdout": e.stdout,
                "stderr": e.stderr,
                "error": str(e)
            }


class IntensityNormalizationStep(PreprocessingStep):
    """Intensity normalization preprocessing step."""
    
    def __init__(self, config: Dict, template_path: Path):
        super().__init__("intensity_normalization", config)
        self.template_path = template_path
        
    def get_command(self, input_dir: Path, output_dir: Path, 
                   work_dir: Path, **kwargs) -> List[str]:
        # For this example, using the existing preprocessing.py script
        # In a real implementation, this would call a dedicated script
        script_path = Path(__file__).parent / "intensity_normalization.py"
        
        command = [
            self.python_executable,
            str(script_path),
            str(input_dir),
            str(output_dir),
            "--method", self.config.get('method', 'HistogramMatching'),
            "--template_path", str(self.template_path)
        ]
        
        return command


class BiasFieldCorrectionStep(PreprocessingStep):
    """Bias field correction preprocessing step."""
    
    def __init__(self, config: Dict):
        super().__init__("bias_field_correction", config)
        
    def get_command(self, input_dir: Path, output_dir: Path, 
                   work_dir: Path, **kwargs) -> List[str]:
        script_path = Path(__file__).parent / "bias_correction.py"
        
        command = [
            self.python_executable,
            str(script_path),
            str(input_dir),
            str(output_dir),
            "--method", self.config.get('method', 'N4BiasFieldCorrection')
        ]
        
        # Add method-specific parameters
        if self.config.get('sitk_shrinkFactor'):
            command.extend(['--shrink_factor', str(self.config['sitk_shrinkFactor'])])
            
        return command


class RegistrationStep(PreprocessingStep):
    """Registration preprocessing step."""
    
    def __init__(self, config: Dict, template_path: Path):
        super().__init__("registration", config)
        self.template_path = template_path
        
    def get_command(self, input_dir: Path, output_dir: Path, 
                   work_dir: Path, **kwargs) -> List[str]:
        script_path = Path(__file__).parent / "registration.py"
        transforms_dir = kwargs.get('transforms_dir', work_dir / 'transforms')
        
        command = [
            self.python_executable,
            str(script_path),
            str(input_dir),
            str(output_dir),
            str(transforms_dir),
            "--template_path", str(self.template_path),
            "--transform_type", self.config.get('ants_transform_type', 'SyN')
        ]
        
        return command


class SkullStrippingStep(PreprocessingStep):
    """Skull stripping preprocessing step."""
    
    def __init__(self, config: Dict):
        super().__init__("skull_stripping", config)
        
    def get_command(self, input_dir: Path, output_dir: Path, 
                   work_dir: Path, **kwargs) -> List[str]:
        script_path = Path(__file__).parent / "skull_stripping.py"
        masks_dir = kwargs.get('masks_dir', work_dir / 'masks')
        
        command = [
            self.python_executable,
            str(script_path),
            str(input_dir),
            str(output_dir),
            "--masks_dir", str(masks_dir),
            "--frac", str(self.config.get('bet_fractional_intensity_threshold', 0.5))
        ]
        
        if self.config.get('bet_robust', True):
            command.append('--robust')
            
        if self.config.get('bet_options'):
            command.extend(['--bet_options', self.config['bet_options']])
            
        return command


class StepFactory:
    """Factory for creating preprocessing steps."""
    
    def __init__(self, config: Config):
        self.config = config
        self.template_path = config.get_template_path()
        
    def create_step(self, step_name: str) -> Optional[PreprocessingStep]:
        """Create a preprocessing step based on name."""
        step_config = self.config.get_step_config(step_name)
        
        if step_name == "intensity_normalization":
            return IntensityNormalizationStep(step_config, self.template_path)
        elif step_name == "bias_field_correction":
            return BiasFieldCorrectionStep(step_config)
        elif step_name == "registration":
            return RegistrationStep(step_config, self.template_path)
        elif step_name == "skull_stripping":
            return SkullStrippingStep(step_config)
        else:
            logger.warning(f"Unknown preprocessing step: {step_name}")
            return None


class PreprocessingPipeline:
    """Main preprocessing pipeline orchestrator."""
    
    def __init__(self, input_dir: Path, output_dir: Path, transforms_dir: Path, 
                 config_path: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.transforms_dir = transforms_dir
        self.config = Config(config_path)
        self.step_factory = StepFactory(self.config)
        
    def run(self) -> Dict:
        """Run the preprocessing pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Preprocessing Pipeline")
        logger.info(f"Input directory: {self.input_dir}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Transforms directory: {self.transforms_dir}")
        logger.info("=" * 60)
        
        # Get enabled steps
        enabled_steps = self.config.get_enabled_steps()
        logger.info(f"Enabled steps: {', '.join(enabled_steps)}")
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.transforms_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline results
        results = {
            "timestamp": datetime.now().isoformat(),
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "steps": []
        }
        
        # Use temporary directory for intermediate results
        with tempfile.TemporaryDirectory(prefix="preprocessing_") as temp_dir:
            temp_path = Path(temp_dir)
            current_input = self.input_dir
            
            # Execute each step in sequence
            for i, step_name in enumerate(enabled_steps):
                logger.info(f"\n--- Step {i+1}/{len(enabled_steps)}: {step_name} ---")
                
                # Create step
                step = self.step_factory.create_step(step_name)
                if not step:
                    logger.error(f"Failed to create step: {step_name}")
                    continue
                
                # Determine output directory
                if i < len(enabled_steps) - 1:
                    # Intermediate step - output to temp
                    step_output = temp_path / f"step_{i+1}_{step_name}"
                else:
                    # Final step - output to final directory
                    step_output = self.output_dir
                
                # Execute step
                step_result = step.execute(
                    current_input,
                    step_output,
                    self.transforms_dir,
                    transforms_dir=self.transforms_dir / step_name,
                    masks_dir=self.transforms_dir / "masks"
                )
                
                # Record result
                results["steps"].append({
                    "name": step_name,
                    "order": i + 1,
                    **step_result
                })
                
                # Check for failure
                if step_result["status"] == "failed":
                    logger.error(f"Pipeline aborted due to failure in step: {step_name}")
                    results["status"] = "failed"
                    break
                
                # Update input for next step
                current_input = step_output
            else:
                # All steps completed successfully
                results["status"] = "success"
                logger.info("\n" + "=" * 60)
                logger.info("Preprocessing Pipeline Completed Successfully")
                logger.info("=" * 60)
        
        # Save pipeline summary
        summary_file = self.transforms_dir / "preprocessing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input_dir", type=Path, help="Input BIDS directory")
    parser.add_argument("output_dir", type=Path, help="Output directory for preprocessed files")
    parser.add_argument("transforms_dir", type=Path, help="Output directory for transforms and logs")
    parser.add_argument("--template_path", type=Path, help="Path to template file (overrides config)")
    parser.add_argument("--config", required=True, type=Path, help="Path to configuration file")
    parser.add_argument("--log_file", type=Path, help="Path to log file")
    parser.add_argument("--console_log_level", default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Console logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = args.log_file or args.transforms_dir / "preprocessing.log"
    setup_logging(log_file, args.console_log_level)
    
    try:
        # Create and run pipeline
        pipeline = PreprocessingPipeline(
            args.input_dir,
            args.output_dir,
            args.transforms_dir,
            args.config
        )
        
        results = pipeline.run()
        
        # Exit with appropriate code
        if results["status"] == "success":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()