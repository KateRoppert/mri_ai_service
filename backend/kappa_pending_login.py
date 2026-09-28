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


def remember(login_id: str, passwd: str) -> None:
    """Hold a login attempt that could not be completed because Kappa was
    unreachable. Never called for a rejected password."""
    global _held
    if not login_id or not passwd:
        return
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

    if result is None:
        # Kappa answered and said no. Retrying the same wrong password is
        # exactly how an account gets locked, so stop here and make the
        # operator re-enter it.
        forget("Kappa отклонила учётные данные")
        return None

    forget("выполнен автоматический вход")
    logger.info("Автоматический вход в Kappa выполнен: %s", login_id)
    return result
