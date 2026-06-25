import sys
from pathlib import Path

import numpy as np

PROJ_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJ_ROOT / "scripts"))

from register_ms_zone_atlases import binarize_labels


def test_binarize_keeps_only_listed_labels():
    data = np.array([0, 1, 2, 3, 4, 5])
    result = binarize_labels(data, {2, 4})
    assert list(result) == [0, 0, 1, 0, 1, 0]


def test_binarize_returns_int_array():
    data = np.array([[0, 6], [6, 0]])
    result = binarize_labels(data, {6})
    assert result.dtype.kind in ("i", "u")
    assert result.tolist() == [[0, 1], [1, 0]]
