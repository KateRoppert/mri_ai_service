"""
Модуль авторизации через Kappa
"""
import httpx
import logging
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

KAPPA_BASE_URL = "https://kappa.nsu.ru:8061/user-micro-services/v1"

# In-memory хранилище сессий: session_id -> {token, userId, userTypeId, ...}
_sessions: Dict[str, Dict[str, Any]] = {}


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

    # Создаём внутреннюю сессию
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "kappa_token": data.get("token"),
        "user_id": data.get("userId"),
        "user_type_id": data.get("userTypeId"),
        "user_name": data.get("userName"),
        "first_name": data.get("firstName"),
        "last_name": data.get("lastName"),
        "token_expiry": data.get("tokenExpiryDate"),
        "org_details": data.get("orgDetails"),
    }

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
    return _sessions.get(session_id)


def delete_session(session_id: str) -> bool:
    """Удалить сессию (logout)."""
    if session_id in _sessions:
        del _sessions[session_id]
        return True
    return False