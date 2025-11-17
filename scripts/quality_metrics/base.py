"""
Base classes and interfaces for quality metrics.
"""

from abc import ABC, abstractmethod
from typing import Dict
import numpy as np


class QualityMetric(ABC):
    """Abstract base class for quality metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name (e.g., 'snr', 'cnr')."""
        pass
    
    @property
    @abstractmethod
    def higher_is_better(self) -> bool:
        """
        Whether higher values indicate better quality.
        
        Returns:
            True if higher is better (SNR, CNR), False if lower is better (EFC)
        """
        pass
    
    @abstractmethod
    def calculate(self, data: np.ndarray, fg_mask: np.ndarray) -> float:
        """
        Calculate the metric.
        
        Args:
            data: 3D image data
            fg_mask: Foreground mask (True = foreground)
            
        Returns:
            Metric value
        """
        pass
    
    def get_info(self) -> Dict:
        """
        Get metric information.
        
        Returns:
            Dictionary with metric metadata
        """
        return {
            'name': self.name,
            'higher_is_better': self.higher_is_better
        }