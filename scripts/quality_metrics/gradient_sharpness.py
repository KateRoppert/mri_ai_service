"""
Gradient Sharpness metric.
"""

import numpy as np
from .base import QualityMetric


class GradientSharpnessMetric(QualityMetric):
    """
    Gradient Sharpness metric.
    
    Measures image sharpness using the variance of gradient magnitudes
    in the foreground region. Higher values indicate sharper images.
    """
    
    @property
    def name(self) -> str:
        return "gradient_sharpness"
    
    @property
    def higher_is_better(self) -> bool:
        return True
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate gradient sharpness metric.
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask (True = foreground)
            
        Returns:
            Gradient sharpness value (variance of gradient magnitudes)
        """
        if not np.any(fg_mask):
            return 0.0
        
        # Calculate gradients in each direction
        grad_x = np.gradient(data, axis=0)
        grad_y = np.gradient(data, axis=1) 
        grad_z = np.gradient(data, axis=2)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        # Extract foreground gradients
        fg_gradients = gradient_magnitude[fg_mask]
        
        # Return variance of gradient magnitudes
        if len(fg_gradients) == 0:
            return 0.0
        
        sharpness = np.var(fg_gradients)
        return float(sharpness)