import sys
from pathlib import Path
from typing import Dict, Optional

import pytest

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from anatomical_analyzer_base import AnatomicalAnalyzerBase


def test_cannot_instantiate_base_directly():
    with pytest.raises(TypeError):
        AnatomicalAnalyzerBase()


def test_subclass_missing_methods_cannot_be_instantiated():
    class Incomplete(AnatomicalAnalyzerBase):
        def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
            return None
        # save_report intentionally not implemented

    with pytest.raises(TypeError):
        Incomplete()


def test_complete_subclass_can_be_instantiated():
    class Complete(AnatomicalAnalyzerBase):
        def analyze_mask(self, mask_path: Path) -> Optional[Dict]:
            return {"ok": True}

        def save_report(self, report: Dict, output_path: Path) -> bool:
            return True

    instance = Complete()
    assert instance.analyze_mask(Path("x")) == {"ok": True}


def test_lobar_analyzer_is_an_anatomical_analyzer():
    from lobar_analysis import LobarAnalyzer
    assert issubclass(LobarAnalyzer, AnatomicalAnalyzerBase)
