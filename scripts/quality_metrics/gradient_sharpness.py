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

        # Compute the gradient magnitude while holding at most two full-size
        # arrays at once. The naive form
        #   np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        # keeps three gradient arrays plus squared temporaries alive
        # simultaneously — ~6 full volumes at peak, which OOM-killed workers on
        # high-resolution SibBMS data. Accumulating the sum of squares in place
        # holds only the accumulator plus one transient per-axis gradient.
        magnitude = np.gradient(data, axis=0)
        magnitude *= magnitude  # grad_x**2, in place
        for axis in (1, 2):
            grad = np.gradient(data, axis=axis)
            grad *= grad  # grad_axis**2, in place
            magnitude += grad
            del grad
        np.sqrt(magnitude, out=magnitude)  # magnitude now holds |grad|

        # Extract foreground gradients
        fg_gradients = magnitude[fg_mask]

        # Return variance of gradient magnitudes
        if len(fg_gradients) == 0:
            return 0.0

        sharpness = np.var(fg_gradients)
        return float(sharpness)