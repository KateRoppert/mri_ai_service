"""
Utility functions for quality assessment.
"""

import numpy as np

try:
    from skimage import filters
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def create_foreground_mask(data: np.ndarray, method: str = "otsu", 
                           percentile: float = 5.0) -> np.ndarray:
    """
    Create foreground mask to separate brain from background.
    
    Args:
        data: 3D image data
        method: "otsu" or "percentile"
        percentile: Percentile threshold if using percentile method
        
    Returns:
        Binary mask (True = foreground)
    """
    if method == "otsu" and SKIMAGE_AVAILABLE:
        # Otsu thresholding
        threshold = filters.threshold_otsu(data)
    else:
        # Fallback: percentile-based thresholding
        threshold = np.percentile(data[data > 0], percentile)
    
    mask = data > threshold
    return mask


def get_foreground_background(data: np.ndarray, fg_mask: np.ndarray):
    """
    Split data into foreground and background.
    
    Args:
        data: 3D image data
        fg_mask: Foreground mask
        
    Returns:
        Tuple of (foreground, background) arrays
    """
    foreground = data[fg_mask]
    background = data[~fg_mask]
    return foreground, background