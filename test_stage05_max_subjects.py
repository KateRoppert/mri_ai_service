"""Stage 05's patient limit counts patients, not sessions.

It sliced a list of (anat_dir, subject, session) tuples, so a limit of 1 on
an MS patient with two sessions took ONE SESSION: the second never reached
preprocessing, segmentation or the longitudinal view, and the stage log
said "Limited to first 1 subjects" while meaning something else.

In a pipeline run the limit now lives only in stage 01, so this matters for
running the stage by hand — where the flag has to mean what it means in
every other stage.
"""
import importlib.util
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

_spec = importlib.util.spec_from_file_location(
    "prep05", SCRIPTS_DIR / "05_preprocessing.py")
prep = importlib.util.module_from_spec(_spec)
sys.modules["prep05"] = prep
_spec.loader.exec_module(prep)


def _tree(root: Path, layout):
    """layout: {"sub-001": ["ses-001", "ses-002"], ...}"""
    for sub, sessions in layout.items():
        for ses in sessions:
            (root / sub / ses / "anat").mkdir(parents=True, exist_ok=True)
    return root


def test_both_sessions_of_one_patient_survive_a_limit_of_one(tmp_path):
    """The exact case that lost a session on run 7a9825eb."""
    _tree(tmp_path, {"sub-002": ["ses-001", "ses-002"]})
    subjects = prep.find_subjects(tmp_path)

    limited = prep.limit_to_patients(subjects, 1)

    assert [s[2] for s in limited] == ["ses-001", "ses-002"]


def test_the_limit_counts_patients(tmp_path):
    _tree(tmp_path, {
        "sub-001": ["ses-001", "ses-002"],
        "sub-002": ["ses-001"],
        "sub-003": ["ses-001"],
    })
    subjects = prep.find_subjects(tmp_path)

    limited = prep.limit_to_patients(subjects, 2)

    assert {s[1] for s in limited} == {"sub-001", "sub-002"}
    assert len(limited) == 3, "сессии взятых пациентов должны остаться все"


def test_no_limit_keeps_everything(tmp_path):
    _tree(tmp_path, {"sub-001": ["ses-001"], "sub-002": ["ses-001"]})
    subjects = prep.find_subjects(tmp_path)

    for empty in (None, 0):
        assert prep.limit_to_patients(subjects, empty) == subjects


def test_a_limit_above_the_count_keeps_everything(tmp_path):
    _tree(tmp_path, {"sub-001": ["ses-001", "ses-002"]})
    subjects = prep.find_subjects(tmp_path)

    assert prep.limit_to_patients(subjects, 99) == subjects


def test_find_subjects_own_limit_counts_patients_too(tmp_path):
    """The parameter exists on find_subjects as well; it must not mean
    something different from the flag that feeds it."""
    _tree(tmp_path, {"sub-001": ["ses-001", "ses-002"], "sub-002": ["ses-001"]})

    found = prep.find_subjects(tmp_path, max_subjects=1)

    assert {s[1] for s in found} == {"sub-001"}
    assert len(found) == 2, "обе сессии единственного взятого пациента"
