"""
Entropy Focus Criterion (EFC) metric.
"""

import numpy as np
from .base import QualityMetric


class EFCMetric(QualityMetric):
    """
    Entropy Focus Criterion metric.
    
    Measures sharpness based on entropy of normalized histogram.
    Lower values indicate sharper images (less entropy).
    """
    
    @property
    def name(self) -> str:
        return "efc"
    
    @property
    def higher_is_better(self) -> bool:
        return False  # Lower entropy = sharper
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate EFC.
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask
            
        Returns:
            EFC value (0-1, lower is better)
        """
        foreground = data[fg_mask]
        
        # Normalize to 0-1
        fg_min = np.min(foreground)
        fg_max = np.max(foreground)
        
        if fg_max - fg_min == 0:
            return 1.0
        
        normalized = (foreground - fg_min) / (fg_max - fg_min)
        
        # Calculate histogram
        hist, _ = np.histogram(normalized, bins=256, range=(0, 1))
        hist = hist[hist > 0]  # Remove zeros
        
        # Normalize histogram
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Normalize by maximum possible entropy (log2(256))
        max_entropy = np.log2(256)
        efc = entropy / max_entropy
        
        return float(efc)