"""
Voxel Anisotropy metric.
"""

import numpy as np
import nibabel as nib
from .base import QualityMetric


class VoxelAnisotropyMetric(QualityMetric):
    """
    Voxel Anisotropy metric.
    
    Measures how anisotropic (non-uniform) the voxel dimensions are.
    Lower values indicate more isotropic (uniform) voxels, which is generally better
    for processing and analysis.
    """
    
    @property
    def name(self) -> str:
        return "voxel_anisotropy"
    
    @property
    def higher_is_better(self) -> bool:
        return False  # Lower anisotropy (more isotropic) is better
    
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray, 
                  img_header=None) -> float:
        """
        Calculate voxel anisotropy metric.
        
        Anisotropy = max(voxel_size) / min(voxel_size)
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask (True = foreground) 
            img_header: NIfTI header (optional, for voxel sizes)
            
        Returns:
            Anisotropy ratio (1.0 = perfectly isotropic)
        """
        # Default voxel sizes if no header provided
        if img_header is None:
            # Assume isotropic 1mm voxels as default
            return 1.0
        
        try:
            # Get voxel dimensions from header
            if hasattr(img_header, 'get_zooms'):
                voxel_sizes = img_header.get_zooms()[:3]  # x, y, z dimensions
            else:
                # Fallback: assume isotropic
                return 1.0
            
            # Calculate anisotropy ratio
            max_size = np.max(voxel_sizes)
            min_size = np.min(voxel_sizes)
            
            if min_size == 0:
                return float('inf')
                
            anisotropy = max_size / min_size
            return float(anisotropy)
            
        except Exception:
            # Return default isotropic value on error
            return 1.0