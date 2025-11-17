"""
Coefficient of Variation metric.
"""

import numpy as np
from .base import QualityMetric


class CoefficientOfVariationMetric(QualityMetric):
    """
    Coefficient of Variation metric.
    
    Measures the ratio of standard deviation to mean intensity in foreground.
    CV = std(foreground) / mean(foreground)
    
    This metric is normalized and helps identify noise relative to signal strength.
    Lower values generally indicate better signal stability.
    """
    
    @property
    def name(self) -> str:
        return "coefficient_of_variation"
    
    @property
    def higher_is_better(self) -> bool:
        return False  # Lower CV indicates more stable signal
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate coefficient of variation in foreground.
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask (True = foreground)
            
        Returns:
            Coefficient of variation (std/mean)
        """
        if not np.any(fg_mask):
            return 0.0
        
        foreground = data[fg_mask]
        
        if len(foreground) == 0:
            return 0.0
        
        mean_intensity = np.mean(foreground)
        std_intensity = np.std(foreground)
        
        # Avoid division by zero
        if mean_intensity == 0:
            return float('inf') if std_intensity > 0 else 0.0
        
        cv = std_intensity / mean_intensity
        return float(cv)