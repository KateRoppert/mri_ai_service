"""Period summary for SIGMA: one class per completed BIDS session.

The document shape is the contract in
docs/superpowers/specs/2026-11-16-sigma-period-stats-design.md.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent))

from database import PipelineRun


PERIOD_START = datetime(2026, 11, 1, tzinfo=timezone.utc)
PERIOD_END = datetime(2026, 11, 16, tzinfo=timezone.utc)


def _db(tmp_path):
    engine = sqlalchemy.create_engine(f"sqlite:///{tmp_path / 'stats.db'}")
    PipelineRun.__table__.create(engine)
    return sessionmaker(bind=engine)()


def _run(db, run_id, output, lesion_type, completed_at, status="completed",
         dataset_id=None):
    db.add(PipelineRun(
        run_id=run_id,
        input_path="/in",
        output_path=str(output),
        status=status,
        lesion_type=lesion_type,
        completed_at=completed_at,
        kappa_dataset_id=dataset_id,
    ))
    db.commit()


def _gbm(root: Path, sub: str, ses: str, voxels: int, edema: int = 0):
    path = (
        root / "segmentation" / sub / ses / "anat" / "glioblastoma"
        / f"{sub}_{ses}_t1_volume_report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "total_tumor": {"voxel_count": voxels},
        "classes": {"2": {"name": "ED (edema)", "voxel_count": edema}},
    }), encoding="utf-8")


def _ms(root: Path, sub: str, ses: str, lesion_count: int):
    path = (
        root / "segmentation" / sub / ses / "anat" / "multiple_sclerosis"
        / f"{sub}_{ses}_t1_lesion_stats_report.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"lesion_count": lesion_count}), encoding="utf-8")


def _naive(moment: datetime) -> datetime:
    """SQLite returns the column the way the pipeline writes it: naive UTC."""
    return moment.astimezone(timezone.utc).replace(tzinfo=None)


def test_a_period_mix_matches_the_document(tmp_path):
    """1 normal + 2 glioblastoma in one run, 1 MS in another, same period."""
    from sigma_period_stats import build_period_stats

    gbm = tmp_path / "gbm"
    ms = tmp_path / "ms"
    _gbm(gbm, "sub-001", "ses-001", voxels=0, edema=40)
    _gbm(gbm, "sub-002", "ses-001", voxels=12)
    _gbm(gbm, "sub-003", "ses-001", voxels=4)
    _ms(ms, "sub-004", "ses-001", lesion_count=3)

    db = _db(tmp_path)
    inside = _naive(datetime(2026, 11, 10, tzinfo=timezone.utc))
    _run(db, "gbm-run", gbm, "glioblastoma", inside)
    _run(db, "ms-run", ms, "multiple_sclerosis", inside)

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)

    assert doc == {
        "period": {
            "start": "2026-11-01T00:00:00Z",
            "end": "2026-11-16T00:00:00Z",
        },
        "modality": "Mri",
        "studies_total": 4,
        "detected_total": 3,
        "by_class": {
            "normal": 1,
            "glioblastoma": 2,
            "multiple_sclerosis": 1,
        },
    }


def test_ms_with_no_counted_lesions_is_normal(tmp_path):
    from sigma_period_stats import build_period_stats

    root = tmp_path / "out"
    _ms(root, "sub-001", "ses-001", lesion_count=0)
    db = _db(tmp_path)
    _run(db, "r", root, "multiple_sclerosis", _naive(PERIOD_START))

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["by_class"] == {
        "normal": 1, "glioblastoma": 0, "multiple_sclerosis": 0,
    }


def test_edema_alone_is_normal(tmp_path):
    from sigma_period_stats import build_period_stats

    root = tmp_path / "out"
    _gbm(root, "sub-001", "ses-001", voxels=0, edema=80)
    db = _db(tmp_path)
    _run(db, "r", root, "glioblastoma", _naive(PERIOD_START))

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["by_class"]["normal"] == 1
    assert doc["by_class"]["glioblastoma"] == 0


def test_a_later_run_of_the_same_session_replaces_the_class(tmp_path):
    from sigma_period_stats import build_period_stats

    first = tmp_path / "first"
    second = tmp_path / "second"
    _gbm(first, "sub-001", "ses-001", voxels=0)
    _gbm(second, "sub-001", "ses-001", voxels=9)
    db = _db(tmp_path)
    _run(db, "early", first, "glioblastoma",
         _naive(datetime(2026, 11, 2, tzinfo=timezone.utc)))
    _run(db, "late", second, "glioblastoma",
         _naive(datetime(2026, 11, 3, tzinfo=timezone.utc)))

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["studies_total"] == 1
    assert doc["by_class"]["glioblastoma"] == 1
    assert doc["by_class"]["normal"] == 0


def test_outside_incomplete_and_unreported_sessions_are_absent(tmp_path):
    from sigma_period_stats import build_period_stats

    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    running = tmp_path / "running"
    bare = tmp_path / "bare"
    _gbm(inside, "sub-001", "ses-001", voxels=3)
    _gbm(outside, "sub-002", "ses-001", voxels=3)
    _gbm(running, "sub-003", "ses-001", voxels=3)
    (bare / "segmentation" / "sub-004" / "ses-001").mkdir(parents=True)

    db = _db(tmp_path)
    _run(db, "in", inside, "glioblastoma", _naive(PERIOD_START))
    _run(db, "out", outside, "glioblastoma",
         _naive(datetime(2026, 10, 1, tzinfo=timezone.utc)))
    _run(db, "run", running, "glioblastoma", _naive(PERIOD_START),
         status="running")
    _run(db, "bare", bare, "glioblastoma", _naive(PERIOD_START))

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["studies_total"] == 1
    assert doc["by_class"]["glioblastoma"] == 1


def test_two_periods_over_a_different_mix_differ(tmp_path):
    from sigma_period_stats import build_period_stats

    early = tmp_path / "early"
    late = tmp_path / "late"
    _gbm(early, "sub-001", "ses-001", voxels=5)
    _ms(late, "sub-002", "ses-001", lesion_count=1)
    db = _db(tmp_path)
    _run(db, "e", early, "glioblastoma",
         _naive(datetime(2026, 11, 2, tzinfo=timezone.utc)))
    _run(db, "l", late, "multiple_sclerosis",
         _naive(datetime(2026, 11, 20, tzinfo=timezone.utc)))

    november = build_period_stats(
        db, PERIOD_START, PERIOD_END,
    )
    later = build_period_stats(
        db,
        datetime(2026, 11, 16, tzinfo=timezone.utc),
        datetime(2026, 12, 1, tzinfo=timezone.utc),
    )
    assert november["by_class"] != later["by_class"]
    assert november["by_class"]["glioblastoma"] == 1
    assert later["by_class"]["multiple_sclerosis"] == 1


def test_a_completed_run_without_a_timestamp_is_not_an_error(tmp_path):
    from sigma_period_stats import build_period_stats

    root = tmp_path / "out"
    _gbm(root, "sub-001", "ses-001", voxels=4)
    db = _db(tmp_path)
    _run(db, "untimed", root, "glioblastoma", completed_at=None)

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["studies_total"] == 0


def test_a_missing_segmentation_directory_is_not_an_error(tmp_path):
    from sigma_period_stats import build_period_stats

    db = _db(tmp_path)
    _run(db, "r", tmp_path / "nowhere", "glioblastoma", _naive(PERIOD_START))

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["studies_total"] == 0
    assert doc["by_class"]["normal"] == 0


def test_conflicting_reports_for_one_session_omit_it(tmp_path):
    from sigma_period_stats import build_period_stats

    root = tmp_path / "out"
    _gbm(root, "sub-001", "ses-001", voxels=0)
    other = (
        root / "segmentation" / "sub-001" / "ses-001" / "anat" / "glioblastoma"
        / "sub-001_ses-001_t1c_volume_report.json"
    )
    other.write_text(json.dumps({"total_tumor": {"voxel_count": 8}}), encoding="utf-8")
    db = _db(tmp_path)
    _run(db, "r", root, "glioblastoma", _naive(PERIOD_START))

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["studies_total"] == 0


def test_the_same_bids_id_in_two_datasets_counts_twice(tmp_path):
    """Номер sub-001 начинается заново в каждом датасете Kappa."""
    from sigma_period_stats import build_period_stats

    first = tmp_path / "first"
    second = tmp_path / "second"
    _gbm(first, "sub-001", "ses-001", voxels=3)
    _gbm(second, "sub-001", "ses-001", voxels=7)
    db = _db(tmp_path)
    moment = _naive(PERIOD_START)
    _run(db, "a", first, "glioblastoma", moment, dataset_id=351)
    _run(db, "b", second, "glioblastoma", moment, dataset_id=349)

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["studies_total"] == 2
    assert doc["by_class"]["glioblastoma"] == 2


def test_a_requeue_inside_one_dataset_still_counts_once(tmp_path):
    from sigma_period_stats import build_period_stats

    first = tmp_path / "first"
    second = tmp_path / "second"
    _gbm(first, "sub-001", "ses-001", voxels=0)
    _gbm(second, "sub-001", "ses-001", voxels=6)
    db = _db(tmp_path)
    _run(db, "early", first, "glioblastoma",
         _naive(datetime(2026, 11, 2, tzinfo=timezone.utc)), dataset_id=351)
    _run(db, "late", second, "glioblastoma",
         _naive(datetime(2026, 11, 3, tzinfo=timezone.utc)), dataset_id=351)

    doc = build_period_stats(db, PERIOD_START, PERIOD_END)
    assert doc["studies_total"] == 1
    assert doc["by_class"]["glioblastoma"] == 1


def test_a_bound_without_a_zone_is_utc_and_echoed_with_z(tmp_path):
    from sigma_period_stats import build_period_stats

    root = tmp_path / "out"
    _gbm(root, "sub-001", "ses-001", voxels=2)
    db = _db(tmp_path)
    _run(db, "r", root, "glioblastoma",
         datetime(2026, 11, 1, 0, 0, 0))

    doc = build_period_stats(
        db,
        datetime(2026, 11, 1, 0, 0, 0),
        datetime(2026, 11, 2, 0, 0, 0),
    )
    assert doc["studies_total"] == 1
    assert doc["period"] == {
        "start": "2026-11-01T00:00:00Z",
        "end": "2026-11-02T00:00:00Z",
    }


def test_the_endpoint_returns_the_document(tmp_path):
    from fastapi.testclient import TestClient

    import app as app_module
    from database import SessionLocal

    root = tmp_path / "out"
    _gbm(root, "sub-001", "ses-001", voxels=2)
    db = SessionLocal()
    try:
        _run(db, "sigma-endpoint", root, "glioblastoma",
             _naive(datetime(2026, 11, 5, tzinfo=timezone.utc)))
        client = TestClient(app_module.app)
        response = client.get(
            "/api/sigma/period-stats",
            params={"start": "2026-11-01T00:00:00Z", "end": "2026-11-16T00:00:00Z"},
        )
    finally:
        db.query(PipelineRun).filter(PipelineRun.run_id == "sigma-endpoint").delete()
        db.commit()
        db.close()

    assert response.status_code == 200
    body = response.json()
    assert body["modality"] == "Mri"
    assert body["by_class"]["glioblastoma"] == 1
    assert body["studies_total"] == 1


@pytest.mark.parametrize(
    "params",
    [
        {"start": "2026-11-01T00:00:00Z", "end": "2026-11-01T00:00:00Z"},
        {"end": "2026-11-16T00:00:00Z"},
        {"start": "not-a-date", "end": "2026-11-16T00:00:00Z"},
    ],
)
def test_a_bad_period_is_400(params):
    from fastapi.testclient import TestClient
    import app as app_module

    response = TestClient(app_module.app).get("/api/sigma/period-stats", params=params)
    assert response.status_code == 400


def test_a_finished_run_writes_one_event_per_study(tmp_path):
    from sigma_period_stats import write_run_events

    root = tmp_path / "out"
    _gbm(root, "sub-002", "ses-001", voxels=5)
    _gbm(root, "sub-001", "ses-001", voxels=0, edema=10)
    finished = datetime(2026, 11, 16, 9, 40, tzinfo=timezone.utc)

    path = write_run_events(
        root, "glioblastoma",
        finished_at=finished, kappa_user_name="e.roppert",
    )

    assert path == root / "logs" / "sigma_events.json"
    assert not (root / "logs" / "sigma_period_stats.json").exists()
    events = json.loads(path.read_text(encoding="utf-8"))
    assert events == [
        {
            "Id": "sub-001",
            "modality": "Mri",
            "timestamp": "2026-11-16T09:40:00Z",
            "source": "Federal Center of Neurosurgery",
            "description": "normal",
        },
        {
            "Id": "sub-002",
            "modality": "Mri",
            "timestamp": "2026-11-16T09:40:00Z",
            "source": "Federal Center of Neurosurgery",
            "description": "glioblastoma",
        },
    ]


def test_the_other_account_is_the_tomography_center(tmp_path):
    from sigma_period_stats import write_run_events

    root = tmp_path / "out"
    _ms(root, "sub-010", "ses-001", lesion_count=3)
    finished = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)

    path = write_run_events(
        root, "multiple_sclerosis",
        finished_at=finished, kappa_user_name="test_med1",
    )

    events = json.loads(path.read_text(encoding="utf-8"))
    assert events == [{
        "Id": "sub-010",
        "modality": "Mri",
        "timestamp": "2026-11-16T12:00:00Z",
        "source": "International Tomography Center",
        "description": "multiple sclerosis",
    }]


def test_two_sessions_of_one_patient_share_the_patient_id(tmp_path):
    from sigma_period_stats import write_run_events

    root = tmp_path / "out"
    _gbm(root, "sub-001", "ses-002", voxels=0)
    _gbm(root, "sub-001", "ses-001", voxels=4)
    finished = datetime(2026, 11, 16, 9, 40, tzinfo=timezone.utc)

    path = write_run_events(
        root, "glioblastoma",
        finished_at=finished, kappa_user_name="e.roppert",
    )

    events = json.loads(path.read_text(encoding="utf-8"))
    assert [event["Id"] for event in events] == ["sub-001", "sub-001"]
    assert [event["description"] for event in events] == ["glioblastoma", "normal"]


def test_an_unknown_login_does_not_invent_an_organization(tmp_path):
    from sigma_period_stats import write_run_events

    root = tmp_path / "out"
    _gbm(root, "sub-001", "ses-001", voxels=1)
    finished = datetime(2026, 11, 16, 9, 40, tzinfo=timezone.utc)

    path = write_run_events(
        root, "glioblastoma", finished_at=finished, kappa_user_name=None,
    )

    events = json.loads(path.read_text(encoding="utf-8"))
    assert events[0]["source"] == ""
    assert events[0]["description"] == "glioblastoma"
