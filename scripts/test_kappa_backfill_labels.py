"""KI-075 backfill: what an entity lacks for statuses 3/5, decided without I/O."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))
_spec = importlib.util.spec_from_file_location("backfill", ROOT / "scripts" / "kappa_backfill_labels.py")
backfill = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backfill)


def _entity(files, status=2, info=None):
    return {"dsEntityId": "e1", "dsEntityName": "sub-001_ses-001", "dsEntityStatus": status,
            "dsEntityInfo": info if info is not None else {"lesion_type": "glioblastoma"},
            "files": [{"fileId": f"id-{n}", "fileName": n, "fileCategory": c} for n, c in files]}


OLD = [("sub-001_ses-001_t1c.nii.gz", "Input"), ("sub-001_ses-001_t1c_segmask.nii.gz", "Input"),
       ("sub-001_ses-001_t1c_segmask_labels.nii.gz", "Input"), ("empty.json", "Input")]


def test_old_entity_gets_labels_output_mask_and_labeled():
    p = backfill.plan_entity(_entity(OLD))
    assert p["labels"] == ["ed", "net", "et", "ncr"]
    assert p["masks_to_output"] == [("id-sub-001_ses-001_t1c_segmask.nii.gz",
                                     "sub-001_ses-001_t1c_segmask.nii.gz")]
    assert p["to_labeled"] is True
    assert "stubs" not in p


def test_stubs_only_when_asked():
    assert backfill.plan_entity(_entity(OLD), drop_stubs=True)["stubs"] == ["id-empty.json"]


def test_verified_entity_is_not_relabeled_but_gets_what_it_lacks():
    p = backfill.plan_entity(_entity(OLD, status=5))
    assert p["to_labeled"] is False
    assert "labels" in p and "masks_to_output" in p


def test_up_to_date_entity_needs_nothing():
    files = [("s_t1_segmask.nii.gz", "Output"), ("s_t1.nii.gz", "Input")]
    p = backfill.plan_entity(_entity(files, status=3,
                                     info={"lesion_type": "multiple_sclerosis", "labels": ["lesion"]}))
    assert not backfill._needs_work(p)


def test_entity_without_a_mask_cannot_become_labeled():
    p = backfill.plan_entity(_entity([("s_t1.nii.gz", "Input")]))
    assert p["no_mask"] is True and p["to_labeled"] is False


def test_lesion_type_falls_back_to_the_argument():
    p = backfill.plan_entity(_entity(OLD, info={}), default_lesion="multiple_sclerosis")
    assert p["labels"] == ["lesion"]
