"""Finding a live Kappa session for a user, days after the run started."""
from datetime import datetime, timedelta, timezone

from database import SessionLocal
from kappa_auth import find_live_session_for_user
from registry_models import KappaSession

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=timezone.utc)


def _add(db, session_id, user_id, expiry, created):
    db.add(KappaSession(
        session_id=session_id, kappa_token=f"token-{session_id}",
        user_id=user_id, user_type_id=1, token_expiry=expiry,
        created_at=created,
    ))
    db.commit()


def _cleanup(db, *session_ids):
    for sid in session_ids:
        row = db.query(KappaSession).filter(
            KappaSession.session_id == sid
        ).first()
        if row:
            db.delete(row)
    db.commit()


def test_picks_the_newest_unexpired_session():
    db = SessionLocal()
    ids = ["test_sess_old", "test_sess_new"]
    try:
        _add(db, "test_sess_old", 990026, "2026-09-29T09:00:00.000000Z",
             NOW - timedelta(days=2))
        _add(db, "test_sess_new", 990026, "2026-09-30T09:00:00.000000Z",
             NOW - timedelta(hours=1))
        found = find_live_session_for_user(990026, now=NOW)
        assert found["kappa_token"] == "token-test_sess_new"
    finally:
        _cleanup(db, *ids)
        db.close()


def test_skips_expired_sessions():
    db = SessionLocal()
    try:
        _add(db, "test_sess_dead", 990027, "2026-09-20T09:00:00.000000Z",
             NOW - timedelta(hours=1))
        assert find_live_session_for_user(990027, now=NOW) is None
    finally:
        _cleanup(db, "test_sess_dead")
        db.close()


def test_unparseable_expiry_is_treated_as_usable():
    db = SessionLocal()
    try:
        _add(db, "test_sess_weird", 990028, "not-a-date",
             NOW - timedelta(hours=1))
        found = find_live_session_for_user(990028, now=NOW)
        assert found["kappa_token"] == "token-test_sess_weird"
    finally:
        _cleanup(db, "test_sess_weird")
        db.close()


def test_unknown_user_has_no_session():
    assert find_live_session_for_user(990029, now=NOW) is None


def test_owner_of_dataset_reads_the_mapping(tmp_path, monkeypatch):
    import kappa_dataset_mapping as kdm

    mapping = tmp_path / "kappa_datasets.yaml"
    mapping.write_text(
        "datasets:\n"
        "  26:glioblastoma:current: 350\n"
        "  52:glioblastoma:current: 349\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(kdm, "MAPPING_FILE", mapping)

    assert kdm.owner_of_dataset(350) == 26
    assert kdm.owner_of_dataset(349) == 52
    assert kdm.owner_of_dataset(999) is None
