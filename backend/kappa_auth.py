"""
Модуль авторизации через Kappa
"""
import json
import httpx
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from database import SessionLocal
from registry_models import KappaSession

logger = logging.getLogger(__name__)


class KappaUnreachable(Exception):
    """Kappa не ответила: сеть, DNS, таймаут. Отличается от неверных
    учётных данных, и сообщение оператору должно быть другим."""

KAPPA_BASE_URL = "https://kappa.nsu.ru:8061/user-micro-services/v1"


async def kappa_login(login_id: str, passwd: str) -> Dict[str, Any]:
    """
    Авторизация в Kappa через POST /session/new.
    Возвращает данные профиля + внутренний session_id.
    """
    url = f"{KAPPA_BASE_URL}/session/new"
    payload = {
        "loginId": login_id,
        "passwd": passwd,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
            response = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        # Kappa недоступна — это не то же самое, что неверный пароль, и
        # обработать это должен вызывающий. Без перехвата исключение уходило
        # наружу, FastAPI отдавал 500, а его тело — простой текст
        # "Internal Server Error", на котором фронт спотыкался в JSON.parse.
        logger.warning("Kappa unreachable at login: %s", exc)
        raise KappaUnreachable(str(exc)) from exc

    if response.status_code != 200:
        logger.warning("Kappa login failed: status=%s, body=%s", response.status_code, response.text[:300])
        return None

    data = response.json()

    # Создаём внутреннюю сессию — храним в БД, чтобы она пережила
    # перезапуск backend-процесса (фронт держит session_id в localStorage
    # и не знает, что процесс перезапускался).
    session_id = str(uuid.uuid4())
    db = SessionLocal()
    try:
        record = KappaSession(
            session_id=session_id,
            kappa_token=data.get("token"),
            user_id=data.get("userId"),
            user_type_id=data.get("userTypeId"),
            user_name=data.get("userName"),
            first_name=data.get("firstName"),
            last_name=data.get("lastName"),
            token_expiry=data.get("tokenExpiryDate"),
            org_details=json.dumps(data.get("orgDetails")) if data.get("orgDetails") is not None else None,
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

    logger.info("Kappa login successful: user=%s, session=%s", data.get("userName"), session_id)

    return {
        "session_id": session_id,
        "user_name": data.get("userName"),
        "first_name": data.get("firstName"),
        "last_name": data.get("lastName"),
        "token_expiry": data.get("tokenExpiryDate"),
    }


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Получить данные сессии по session_id."""
    if not session_id:
        return None

    db = SessionLocal()
    try:
        record = db.query(KappaSession).filter(
            KappaSession.session_id == session_id
        ).first()

        if not record:
            return None

        return {
            "kappa_token": record.kappa_token,
            "user_id": record.user_id,
            "user_type_id": record.user_type_id,
            "user_name": record.user_name,
            "first_name": record.first_name,
            "last_name": record.last_name,
            "token_expiry": record.token_expiry,
            "org_details": json.loads(record.org_details) if record.org_details else None,
        }
    finally:
        db.close()


def find_live_session_for_user(
    user_id: int, now: Optional[datetime] = None
) -> Optional[Dict[str, Any]]:
    """The newest session for this Kappa user whose token has not expired.

    Deferred uploads outlive the session that started the run: sessions expire
    after ~7 days and a re-login issues a NEW session_id, so a remembered
    session id is dead exactly when the retry needs it. The Kappa user id is
    stable, so that is what we search by.

    A row with a NULL or unparseable token_expiry is treated as usable — the
    upload attempt is the real test, and refusing to try would strand runs
    over a change in Kappa's date formatting.
    """
    if user_id is None:
        return None

    now = now or datetime.now(timezone.utc)

    db = SessionLocal()
    try:
        rows = db.query(KappaSession).filter(
            KappaSession.user_id == user_id
        ).order_by(KappaSession.created_at.desc()).all()

        for record in rows:
            expiry = _parse_expiry(record.token_expiry)
            if expiry is not None and expiry <= now:
                continue
            return {
                "kappa_token": record.kappa_token,
                "user_id": record.user_id,
                "user_type_id": record.user_type_id,
                "user_name": record.user_name,
                "first_name": record.first_name,
                "last_name": record.last_name,
                "token_expiry": record.token_expiry,
                "org_details": (
                    json.loads(record.org_details) if record.org_details else None
                ),
            }
        return None
    finally:
        db.close()


def _parse_expiry(value: Optional[str]) -> Optional[datetime]:
    """Kappa sends e.g. "2026-09-29T09:36:18.391277Z"; Python 3.12 parses the
    trailing Z directly. None means "cannot tell", not "expired"."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def delete_session(session_id: str) -> bool:
    """Удалить сессию (logout)."""
    db = SessionLocal()
    try:
        record = db.query(KappaSession).filter(
            KappaSession.session_id == session_id
        ).first()

        if not record:
            return False

        db.delete(record)
        db.commit()
        return True
    finally:
        db.close()
