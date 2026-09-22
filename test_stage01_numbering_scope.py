"""
Tests for IDMapper's scope-aware numbering (scripts/01_reorganize_folders.py).

Loaded by path since scripts/01_reorganize_folders.py is a digit-prefixed
standalone script, not a package module — same loading pattern as
tests/stage01/test_bids_allocator.py's _load_reorganize().
"""
import importlib.util
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).parent
SCRIPTS_DIR = PROJ_ROOT / "scripts"
sys.path.insert(0, str(PROJ_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

spec = importlib.util.spec_from_file_location(
    "reorganize_folders_numbering_scope", SCRIPTS_DIR / "01_reorganize_folders.py")
stage01 = importlib.util.module_from_spec(spec)
sys.modules["reorganize_folders_numbering_scope"] = stage01
spec.loader.exec_module(stage01)


def test_id_mapper_allocates_inside_its_scope(tmp_path):
    db = tmp_path / "alloc.db"
    mapper = stage01.IDMapper(scope="ds:337", db_path=db)
    other = stage01.IDMapper(scope="ds:349", db_path=db)

    assert mapper.get_patient_id("P001") == "sub-001"
    assert other.get_patient_id("P001") == "sub-001"


def test_id_mapper_is_stable_within_a_scope(tmp_path):
    db = tmp_path / "alloc.db"
    mapper = stage01.IDMapper(scope="ds:337", db_path=db)
    first = mapper.get_patient_id("P001")

    fresh = stage01.IDMapper(scope="ds:337", db_path=db)
    assert fresh.get_patient_id("P001") == first


def test_lesion_type_alone_still_works(tmp_path):
    """CLI runs pass only --lesion-type; they must keep numbering as before
    (their own local scope, derived from the lesion type)."""
    db = tmp_path / "alloc.db"
    mapper = stage01.IDMapper(lesion_type="glioblastoma", db_path=db)

    assert mapper.get_patient_id("P001") == "sub-001"


def test_lesion_type_alone_is_isolated_from_a_dataset_scope(tmp_path):
    """The old lesion_type-only path must not collide with a dataset scope
    for the same lesion type — they are different numbering spaces now."""
    db = tmp_path / "alloc.db"
    legacy = stage01.IDMapper(lesion_type="glioblastoma", db_path=db)
    scoped = stage01.IDMapper(scope="ds:337", db_path=db)

    assert legacy.get_patient_id("P001") == "sub-001"
    assert scoped.get_patient_id("P002") == "sub-001"


def test_no_scope_and_no_lesion_type_falls_back_to_in_memory_counter(tmp_path):
    """Legacy in-memory fallback for callers that pass neither — no
    persistence, no cross-run stability, same as before this change."""
    mapper = stage01.IDMapper()

    assert mapper.get_patient_id("P001") == "sub-001"
    assert mapper.get_patient_id("P002") == "sub-002"
