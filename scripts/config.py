"""
Universal configuration management for MRI processing pipeline.
Each module loads and validates only its required configuration section.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
import yaml
from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)


class Config:
    """
    Universal configuration loader for the entire pipeline.
    Provides access to specific sections without loading everything into models.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self._raw_config: Dict[str, Any] = {}
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: Path) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                self._raw_config = yaml.safe_load(f) or {}
            logger.info(f"Configuration loaded from {config_path}")
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def get_section(self, *keys: str) -> Dict[str, Any]:
        """
        Get a nested configuration section by keys.
        
        Args:
            *keys: Nested keys to access the section
            
        Returns:
            Configuration section as dictionary
            
        Example:
            config.get_section('steps', 'reorganize_folders')
            config.get_section('paths', 'subdirs')
        """
        result = self._raw_config
        for key in keys:
            return result.get(key, {}) if isinstance(result, dict) else {}
    
    def get_value(self, *keys: str, default: Any = None) -> Any:
        """
        Get a specific configuration value by nested keys.
        
        Args:
            *keys: Nested keys to access the value
            default: Default value if key not found
            
        Returns:
            Configuration value or default
            
        Example:
            config.get_value('paths', 'output_base_dir', default='/tmp')
        """
        result = self._raw_config
        for key in keys[:-1]:
            if isinstance(result, dict):
                result = result.get(key, {})
            else:
                return default
        
        if isinstance(result, dict):
            return result.get(keys[-1], default)
        return default
    
    def get_paths(self) -> Dict[str, Any]:
        """Get paths configuration section."""
        return self.get_section('paths')
    
    def get_host_paths(self) -> Dict[str, Any]:
        """Get host paths configuration section."""
        return self.get_section('host_paths')
    
    def get_remote_paths(self) -> Dict[str, Any]:
        """Get remote paths configuration section."""
        return self.get_section('remote_paths')
    
    def get_step_config(self, step_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific pipeline step.
        
        Args:
            step_name: Name of the step
            
        Returns:
            Step configuration dictionary
        """
        return self.get_section('steps', step_name)
    
    def is_step_enabled(self, step_name: str) -> bool:
        """
        Check if a step is enabled.
        
        Args:
            step_name: Name of the step
            
        Returns:
            True if step is enabled or has no 'enabled' field, False otherwise
        """
        step_config = self.get_step_config(step_name)
        return step_config.get('enabled', True)
    
    def get_executables(self) -> Dict[str, str]:
        """Get executables paths."""
        return self.get_section('executables')
    
    def to_dict(self) -> Dict[str, Any]:
        """Get the entire configuration as dictionary."""
        return self._raw_config.copy()
    
    def __repr__(self) -> str:
        """String representation."""
        steps = list(self.get_section('steps').keys())
        return f"Config(path={self.config_path}, steps={steps})"


# =============================================================================
# Step-specific configuration models (with validation)
# Each module should define its own configuration model
# =============================================================================

class ActionType(str, Enum):
    """File operation actions."""
    COPY = "copy"
    MOVE = "move"
    SYMLINK = "symlink"


class ReorganizeFoldersConfig(BaseModel):
    """Configuration for reorganize_folders step."""
    
    model_config = ConfigDict(use_enum_values=True, extra='allow')
    
    action: ActionType = Field(
        default=ActionType.COPY,
        description="How to handle files (copy/move/symlink)"
    )
    
    # Add more fields as needed based on your original design
    # This is simplified for now
    
    @field_validator('action', mode='before')
    @classmethod
    def validate_action(cls, v):
        """Validate and convert action type."""
        if isinstance(v, str):
            return ActionType(v.lower())
        return v
    
    @classmethod
    def from_config(cls, config: Config) -> 'ReorganizeFoldersConfig':
        """
        Create from Config object.
        
        Args:
            config: Config object with loaded configuration
            
        Returns:
            ReorganizeFoldersConfig instance
        """
        step_config = config.get_step_config('reorganize_folders')
        return cls(**step_config)


class BiasFieldCorrectionConfig(BaseModel):
    """Configuration for bias field correction."""
    
    enabled: bool = Field(default=True)
    method: str = Field(default="N4BiasFieldCorrection")
    strategy: str = Field(default="Standard")
    validate_correction: bool = Field(default=False)
    parallel_processing: bool = Field(default=True)
    max_workers: int = Field(default=4, ge=1, le=32)
    
    # SimpleITK parameters
    sitk_shrinkFactor: int = Field(default=4, ge=1)
    sitk_numberOfIterations: List[int] = Field(default=[50, 50, 50, 50])
    sitk_convergenceThreshold: float = Field(default=0.001)
    
    # Modality-specific parameters
    modality_specific_params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    @classmethod
    def from_config(cls, config: Config) -> 'BiasFieldCorrectionConfig':
        """Create from Config object."""
        preprocessing = config.get_section('steps', 'preprocessing')
        bias_config = preprocessing.get('bias_field_correction', {})
        return cls(**bias_config)


class RegistrationConfig(BaseModel):
    """Configuration for registration step."""
    
    enabled: bool = Field(default=True)
    method: str = Field(default="ANTsPy")
    ants_transform_type: str = Field(default="SyN")
    template_path: str = Field(default="/app/mni_templates/mni_icbm152_t1_tal_nlin_sym_09a.nii")
    
    @classmethod
    def from_config(cls, config: Config) -> 'RegistrationConfig':
        """Create from Config object."""
        preprocessing = config.get_section('steps', 'preprocessing')
        registration_config = preprocessing.get('registration', {})
        return cls(**registration_config)


class SkullStrippingConfig(BaseModel):
    """Configuration for skull stripping."""
    
    enabled: bool = Field(default=True)
    method: str = Field(default="FSL_BET")
    bet_fractional_intensity_threshold: float = Field(default=0.5, ge=0, le=1)
    bet_robust: bool = Field(default=True)
    wsthresh: int = Field(default=5)
    clean_bm: bool = Field(default=True)
    save_mask: bool = Field(default=True)
    border: int = Field(default=0)
    
    @classmethod
    def from_config(cls, config: Config) -> 'SkullStrippingConfig':
        """Create from Config object."""
        preprocessing = config.get_section('steps', 'preprocessing')
        skull_config = preprocessing.get('skull_stripping', {})
        return cls(**skull_config)


class MRIQCConfig(BaseModel):
    """Configuration for MRIQC step."""
    
    enabled: bool = Field(default=False)
    run_on_server: bool = Field(default=True)
    run_on_server_auto_trigger: bool = Field(default=True)
    report_type: str = Field(default="participant")
    n_procs: int = Field(default=1, ge=1)
    n_threads: int = Field(default=1, ge=1)
    mem_gb: int = Field(default=15, ge=1)
    
    @classmethod
    def from_config(cls, config: Config) -> 'MRIQCConfig':
        """Create from Config object."""
        mriqc_config = config.get_step_config('mriqc')
        return cls(**mriqc_config)


class PathsConfig(BaseModel):
    """Configuration for paths."""
    
    raw_input_dir: str
    output_base_dir: str
    template_path: Optional[str] = None
    subdirs: Dict[str, str] = Field(default_factory=dict)
    
    @classmethod
    def from_config(cls, config: Config, use_remote: bool = False) -> 'PathsConfig':
        """
        Create from Config object.
        
        Args:
            config: Config object
            use_remote: If True, use remote_paths; otherwise use host_paths
        """
        if use_remote:
            paths_dict = config.get_remote_paths()
        else:
            paths_dict = config.get_host_paths()
        
        # Merge with container paths subdirs
        container_paths = config.get_paths()
        paths_dict['subdirs'] = container_paths.get('subdirs', {})
        
        return cls(**paths_dict)


# =============================================================================
# Helper functions for modules to use
# =============================================================================

def load_module_config(config_path: Path, module_name: str) -> Dict[str, Any]:
    """
    Helper function for modules to load only their configuration.
    
    Args:
        config_path: Path to configuration file
        module_name: Name of the module/step
        
    Returns:
        Module-specific configuration dictionary
        
    Example:
        # In reorganize_folders.py:
        config_dict = load_module_config("config.yaml", "reorganize_folders")
        config = ReorganizeFoldersConfig(**config_dict)
    """
    config = Config(config_path)
    return config.get_step_config(module_name)


def get_paths_for_module(config_path: Path, use_remote: bool = False) -> PathsConfig:
    """
    Helper function to get paths configuration.
    
    Args:
        config_path: Path to configuration file
        use_remote: Whether to use remote paths
        
    Returns:
        PathsConfig object
    """
    config = Config(config_path)
    return PathsConfig.from_config(config, use_remote)