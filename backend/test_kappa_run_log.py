"""The per-run Kappa log: informative enough to answer "what happened",
short enough that someone actually reads it."""
from datetime import datetime, timedelta, timezone

import kappa_run_log


NOW = datetime(2026, 9, 25, 16, 4, 34, tzinfo=timezone.utc)


def _read(tmp_path):
    return (tmp_path / "logs" / "kappa.log").read_text(encoding="utf-8")


def test_records_which_session_failed_and_why(tmp_path):
    result = {
        "dataset_id": 351, "uploaded": 1, "total": 2,
        "sessions": [
            {"session": "sub-001_ses-001", "success": True, "entity_id": "abc123"},
            {"session": "sub-002_ses-001", "success": False,
             "error": "name_clash", "message": "..."},
        ],
    }
    verdict = {
        "status": "needs_attention", "next_attempt": None,
        "detail": {"delivered": 1, "total": 2, "reason": "name_clash",
                   "attempts": 1},
    }
    kappa_run_log.log_attempt(tmp_path, "после прогона", result, None, verdict)

    text = _read(tmp_path)
    assert "сессий найдено 2" in text
    assert "sub-001_ses-001 — загружено (entity abc123)" in text
    assert "sub-002_ses-001 — не отправлено: номер уже занят" in text
    assert "Итог: требует внимания — 1 из 2" in text


def test_records_the_next_attempt_for_a_transient_failure(tmp_path):
    verdict = {
        "status": "pending",
        "next_attempt": NOW + timedelta(minutes=1),
        "detail": {"delivered": 0, "total": 1, "reason": "network",
                   "attempts": 1},
    }
    kappa_run_log.log_attempt(
        tmp_path, "фоновая досылка",
        {"error": "Failed to resolve dataset_id"}, None, verdict,
    )

    text = _read(tmp_path)
    assert "не удалось определить датасет" in text
    assert "Итог: ждёт — 0 из 1 (Kappa недоступна)" in text
    assert "следующая попытка" in text


def test_exception_is_named_not_dumped(tmp_path):
    """A stack trace belongs in the service log, not in the operator's."""
    verdict = {"status": "pending", "next_attempt": None,
               "detail": {"delivered": 0, "total": 0, "reason": "network",
                          "attempts": 2}}
    kappa_run_log.log_attempt(
        tmp_path, "фоновая досылка", None, ConnectionError("boom"), verdict,
    )

    text = _read(tmp_path)
    assert "сорвалась — ConnectionError" in text
    assert "Traceback" not in text
    assert "boom" not in text


def test_start_line_names_the_dataset(tmp_path):
    kappa_run_log.log_start(tmp_path, 351, None)
    assert "Датасет Kappa: 351" in _read(tmp_path)


def test_start_line_survives_an_unknown_dataset(tmp_path):
    kappa_run_log.log_start(
        tmp_path, None, "Kappa недоступна: номера закрепим при выгрузке",
    )
    text = _read(tmp_path)
    assert "ещё не определён" in text
    assert "Kappa недоступна" in text


def test_writing_never_raises_on_a_bad_path(tmp_path):
    """The output folder being gone is one of the states we report — the
    reporting must not itself explode when it happens."""
    missing = tmp_path / "gone" / "deeper"
    (tmp_path / "gone").write_text("I am a file, not a directory")
    kappa_run_log.append(missing, "что-нибудь")   # must not raise
    kappa_run_log.append(None, "и это тоже")


def test_names_the_dataset_entity_when_it_differs(tmp_path):
    """The session on disk is sub-001 but it lives in the dataset as sub-010.
    Reporting the local name asserts something false about the dataset."""
    result = {
        "dataset_id": 351, "uploaded": 1, "total": 1,
        "sessions": [{
            "session": "sub-001_ses-001", "kappa_name": "sub-010_ses-001",
            "success": True, "skipped_upload": True, "entity_id": "10b0923d",
        }],
    }
    verdict = {"status": "done", "next_attempt": None,
               "detail": {"delivered": 1, "total": 1, "attempts": 5}}
    kappa_run_log.log_attempt(tmp_path, "вручную", result, None, verdict)

    text = _read(tmp_path)
    assert "уже в датасете под именем sub-010_ses-001" in text
