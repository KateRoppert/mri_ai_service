"""
Signal-to-Noise Ratio (SNR) metric.
"""

import numpy as np
from .base import QualityMetric


class SNRMetric(QualityMetric):
    """
    Signal-to-Noise Ratio metric.
    
    SNR = mean(foreground) / std(background)
    
    Higher values indicate better image quality (less noise).
    """
    
    @property
    def name(self) -> str:
        return "snr"
    
    @property
    def higher_is_better(self) -> bool:
        return True
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate SNR.
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask
            
        Returns:
            SNR value
        """
        foreground = data[fg_mask]
        background = data[~fg_mask]
        
        if len(background) == 0 or np.std(background) == 0:
            return 0.0
        
        signal = np.mean(foreground)
        noise = np.std(background)
        
        snr = signal / noise
        return float(snr)