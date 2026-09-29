"""Credentials held in memory so an outage does not cost a second login.

The operator types their login while Kappa is down. We cannot authenticate
yet, so we keep what they typed and do it ourselves the moment Kappa answers.

Deliberately RAM only — never the database, never a file, never a log line.
A Kappa token is already persisted, but a token expires in about a week and
is scoped to Kappa; a password does neither, and is likely reused elsewhere.
Keeping one on disk is a materially bigger promise than this feature is
worth, so the credentials die with the process (and sooner, see below).

They are dropped as soon as any of these is true:
  * the automatic login succeeded — a session exists, they are redundant;
  * Kappa answered and rejected them — retrying a wrong password is how you
    lock an account out, so a rejection discards them immediately;
  * the operator logged out;
  * CREDENTIAL_TTL_HOURS elapsed, as a backstop for a long outage.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CREDENTIAL_TTL_HOURS = 24

# {"login_id": str, "passwd": str, "stored_at": datetime} or None.
# Module-private and never rendered: no __repr__, no logging, no API field.
_held: Optional[Dict[str, Any]] = None

# Чем закончилась последняя автоматическая попытка входа. Живёт отдельно от
# _held: данные к этому моменту уже стёрты, а рассказать оператору, чем всё
# кончилось, ещё нужно.
#   {"status": "succeeded", "session_id", "user_name", "first_name", "last_name"}
#   {"status": "rejected",  "login_id"}
_outcome: Optional[Dict[str, Any]] = None


def remember(login_id: str, passwd: str) -> None:
    """Hold a login attempt that could not be completed because Kappa was
    unreachable. Never called for a rejected password."""
    global _held, _outcome
    if not login_id or not passwd:
        return
    _outcome = None                      # новая попытка — прошлый итог неактуален
    _held = {
        "login_id": login_id,
        "passwd": passwd,
        "stored_at": datetime.now(timezone.utc),
    }
    # The login id is safe to log and useful for support; the password is not.
    logger.info("Запомнен вход %s до восстановления связи с Kappa", login_id)


def forget(reason: str) -> None:
    global _held
    if _held is not None:
        logger.info("Забыты учётные данные (%s)", reason)
    _held = None


def outcome() -> Optional[Dict[str, Any]]:
    """Чем закончился автоматический вход, чтобы интерфейс мог это показать.

    Не стирается при чтении: страницу могут перезагрузить, и подхватить
    созданную сессию нужно всё равно. Сбрасывается выходом из аккаунта и
    новой попыткой входа.
    """
    return dict(_outcome) if _outcome else None


def clear_outcome() -> None:
    global _outcome
    _outcome = None


def held_login() -> Optional[str]:
    """The login id we are holding, for showing the operator who will be
    logged in. Never exposes the password."""
    _expire_if_stale()
    return _held["login_id"] if _held else None


def _expire_if_stale() -> None:
    if _held is None:
        return
    age = datetime.now(timezone.utc) - _held["stored_at"]
    if age >= timedelta(hours=CREDENTIAL_TTL_HOURS):
        forget(f"прошло больше {CREDENTIAL_TTL_HOURS} ч")


async def try_login_now() -> Optional[Dict[str, Any]]:
    """Attempt the deferred login. Returns the session dict, or None.

    Never raises: this runs inside the delivery worker, and a failed login
    must not take the loop down with it.
    """
    _expire_if_stale()
    if _held is None:
        return None

    from kappa_auth import KappaUnreachable, kappa_login

    login_id = _held["login_id"]
    try:
        result = await kappa_login(login_id, _held["passwd"])
    except KappaUnreachable:
        return None                      # still down; keep waiting
    except Exception as e:               # noqa: BLE001
        logger.error("Отложенный вход не удался: %s", e)
        return None

    global _outcome

    if result is None:
        # Kappa answered and said no. Retrying the same wrong password is
        # exactly how an account gets locked, so stop here and make the
        # operator re-enter it.
        forget("Kappa отклонила учётные данные")
        _outcome = {"status": "rejected", "login_id": login_id}
        return None

    forget("выполнен автоматический вход")
    # Сессию создал бэкенд, и браузер о ней не знает — без этого он так и
    # будет писать «войдём автоматически», хотя вход давно выполнен.
    _outcome = {
        "status": "succeeded",
        "session_id": result.get("session_id"),
        "user_name": result.get("user_name"),
        "first_name": result.get("first_name"),
        "last_name": result.get("last_name"),
    }
    logger.info("Автоматический вход в Kappa выполнен: %s", login_id)
    return result
