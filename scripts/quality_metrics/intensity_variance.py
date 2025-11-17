"""
Intensity Variance metric.
"""

import numpy as np
from .base import QualityMetric


class IntensityVarianceMetric(QualityMetric):
    """
    Intensity Variance metric.
    
    Measures the variance of intensity values in the foreground region.
    Very high variance might indicate noise or artifacts, while very low
    variance might indicate poor contrast. Moderate variance is typically optimal.
    """
    
    @property
    def name(self) -> str:
        return "intensity_variance"
    
    @property
    def higher_is_better(self) -> bool:
        return True  # Higher variance usually indicates better contrast
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate intensity variance in foreground.
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask (True = foreground)
            
        Returns:
            Variance of foreground intensities
        """
        if not np.any(fg_mask):
            return 0.0
        
        foreground = data[fg_mask]
        
        if len(foreground) == 0:
            return 0.0
        
        variance = np.var(foreground)
        return float(variance)