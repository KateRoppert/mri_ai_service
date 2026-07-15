"""
Модуль авторизации через Kappa
"""
import json
import httpx
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from database import SessionLocal
from registry_models import KappaSession

logger = logging.getLogger(__name__)

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

    async with httpx.AsyncClient(timeout=10.0, verify=False) as client:
        response = await client.post(url, json=payload)

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
