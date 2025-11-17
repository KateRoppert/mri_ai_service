"""
Quality metrics module.
"""

from .utils import create_foreground_mask, get_foreground_background
from .snr import SNRMetric
from .cnr import CNRMetric
from .efc import EFCMetric
from .fber import FBERMetric
from .gradient_sharpness import GradientSharpnessMetric
from .voxel_anisotropy import VoxelAnisotropyMetric
from .intensity_variance import IntensityVarianceMetric
from .coefficient_of_variation import CoefficientOfVariationMetric

# List of all available metrics
AVAILABLE_METRICS = {
    'snr': SNRMetric,
    'cnr': CNRMetric, 
    'efc': EFCMetric,
    'fber': FBERMetric,
    'gradient_sharpness': GradientSharpnessMetric,
    'voxel_anisotropy': VoxelAnisotropyMetric,
    'intensity_variance': IntensityVarianceMetric,
    'coefficient_of_variation': CoefficientOfVariationMetric
}

__all__ = [
    'create_foreground_mask',
    'get_foreground_background',
    'SNRMetric',
    'CNRMetric', 
    'EFCMetric',
    'FBERMetric',
    'GradientSharpnessMetric',
    'VoxelAnisotropyMetric',
    'IntensityVarianceMetric',
    'CoefficientOfVariationMetric',
    'AVAILABLE_METRICS'
]