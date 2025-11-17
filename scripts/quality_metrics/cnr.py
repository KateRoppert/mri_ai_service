"""
Contrast-to-Noise Ratio (CNR) metric.
"""

import numpy as np
from .base import QualityMetric


class CNRMetric(QualityMetric):
    """
    Contrast-to-Noise Ratio metric.
    
    CNR = |mean(high_intensity) - mean(low_intensity)| / std(background)
    
    Higher values indicate better tissue contrast.
    """
    
    @property
    def name(self) -> str:
        return "cnr"
    
    @property
    def higher_is_better(self) -> bool:
        return True
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate CNR.
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask
            
        Returns:
            CNR value
        """
        foreground = data[fg_mask]
        background = data[~fg_mask]
        
        if len(background) == 0 or np.std(background) == 0:
            return 0.0
        
        # Split foreground into high and low intensity regions
        fg_median = np.median(foreground)
        high_intensity = foreground[foreground > fg_median]
        low_intensity = foreground[foreground <= fg_median]
        
        if len(high_intensity) == 0 or len(low_intensity) == 0:
            return 0.0
        
        contrast = abs(np.mean(high_intensity) - np.mean(low_intensity))
        noise = np.std(background)
        
        cnr = contrast / noise
        return float(cnr)