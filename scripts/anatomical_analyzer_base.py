"""
Shared interface for per-lesion-type anatomical analyzers run at Stage 08.

Each lesion type gets its own analyzer (LobarAnalyzer for glioblastoma,
MSZoneAnalyzer for multiple_sclerosis) behind this common contract, so the
Stage 08 dispatcher and future lesion types (e.g. brain metastases, Этап 8)
don't need to special-case each analyzer's internals.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional


class AnatomicalAnalyzerBase(ABC):
    """Analyzes anatomical localization of lesions for one lesion type."""

    @abstractmethod
    def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
        """Analyze a single segmentation mask. Returns a report dict, or None on failure."""
        raise NotImplementedError

    @abstractmethod
    def save_report(self, report: Dict, output_path: Path) -> bool:
        """Save a report dict as JSON. Returns True on success."""
        raise NotImplementedError
