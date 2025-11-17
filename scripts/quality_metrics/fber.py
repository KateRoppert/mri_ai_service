"""
FBER (Foreground-Background Energy Ratio) metric.
"""

import numpy as np
from .base import QualityMetric


class FBERMetric(QualityMetric):
    """
    Foreground-Background Energy Ratio.
    
    Measures the ratio of mean squared intensity between foreground and background.
    Higher values indicate better contrast between brain and background.
    """
    
    @property
    def name(self) -> str:
        return "fber"
    
    @property
    def higher_is_better(self) -> bool:
        return True
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate FBER metric.
        
        FBER = mean(foreground^2) / mean(background^2)
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask (True = foreground)
            
        Returns:
            FBER value
        """
        if not np.any(fg_mask) or not np.any(~fg_mask):
            return 0.0
        
        foreground = data[fg_mask]
        background = data[~fg_mask]
        
        # Calculate mean squared values
        mean_fg_squared = np.mean(foreground**2)
        mean_bg_squared = np.mean(background**2)
        
        # Avoid division by zero
        if mean_bg_squared == 0:
            return float('inf') if mean_fg_squared > 0 else 0.0
        
        fber = mean_fg_squared / mean_bg_squared
        return float(fber)