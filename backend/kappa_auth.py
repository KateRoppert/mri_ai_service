"""
Модуль авторизации через Kappa
"""
import json
import httpx

from kappa_http import kappa_client
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


class KappaSessionQuota(KappaUnreachable):
    """Kappa ОТВЕТИЛА, но отказалась выдать сессию — пароль она при этом не
    проверяла.

    Третий случай, которого в делении «ответила / не ответила» не было, и
    отсюда дефект: учётные данные оказывались не подтверждёнными и не
    отклонёнными, то есть не отбрасывались и повторялись фоновым воркером до
    24 часов. Если пароль неверен, каждая такая попытка уходит в Kappa — а
    `kappa_pending_login` существует именно затем, чтобы неверный пароль
    отправлялся не более одного раза.

    Наследуется от `KappaUnreachable` сознательно: для ВСЕГО остального это
    та же недоступность — офлайн-режим включается, обработку запускать можно.
    Отличается ровно одним: придерживать непроверенный пароль нельзя.
    """

# v2, потому что v1 закрыт: 13.11.2026 любой путь под
# `/user-micro-services/v1/`, включая несуществующий, стал отдавать 401 с
# пустым телом от nginx, не доходя до приложения. v2 принимает тот же запрос
# (`loginId`/`passwd`), а `UserDetails` в ответе содержит все поля, которые
# читает код ниже, включая `token` и `tokenExpiryDate` — проверено живым
# входом. Клиент данных (`kappa_client.py`) переведён на v2 отдельно, KI-082.
KAPPA_BASE_URL = "https://kappa.nsu.ru:8061/user-micro-services/v2"

# Отказы входа, означающие «сессию сейчас выдать не можем», а не «данные
# неверны». Разница существенна в обе стороны:
#
#   * неверный пароль повторять НЕЛЬЗЯ — так блокируют учётную запись;
#   * нехватку ресурса, наоборот, надо пережить: включить офлайн-режим, чтобы
#     обработку можно было запускать (пайплайн от Kappa не зависит), и войти
#     самим, когда станет можно.
#
# `TOO_MANY_ACTIVE_DEVICES` появился 16.11.2026: v2 ограничивает число
# одновременно активных сессий, а наш выход из аккаунта намеренно НЕ гасит
# сессию в Kappa, чтобы не обрывать отложенную выгрузку
# (`/api/kappa/logout`). Решение верное, но принято против v1, у которого
# лимита не было, поэтому каждый вход занимал слот на срок жизни токена.
#
# Набор задан данными, а не разбором текста: добавить код — одна строка, и
# видно, какие именно отказы считаются временными.
TRANSIENT_LOGIN_DETAILS = frozenset({"TOO_MANY_ACTIVE_DEVICES"})


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
        async with kappa_client(timeout=10.0) as client:
            response = await client.post(url, json=payload)
    except httpx.HTTPError as exc:
        # Kappa недоступна — это не то же самое, что неверный пароль, и
        # обработать это должен вызывающий. Без перехвата исключение уходило
        # наружу, FastAPI отдавал 500, а его тело — простой текст
        # "Internal Server Error", на котором фронт спотыкался в JSON.parse.
        logger.warning("Kappa unreachable at login: %s", exc)
        raise KappaUnreachable(str(exc)) from exc

    if response.status_code != 200:
        # Отказ БЕЗ разбираемого тела — это не ответ по существу. Приложение
        # Kappa объясняет отказ в JSON (`detail`), а шлюз не объясняет ничего:
        # 13.11.2026 закрытый v1 отдавал 401 с пустым телом и `text/html`, и
        # пока это считалось неверным паролем, офлайн-режим с отложенной
        # выгрузкой не включался (он срабатывает только на KappaUnreachable), а
        # работать было нельзя вообще — хотя пайплайн от Kappa не зависит.
        #
        # Обратное направление не менее важно: отказ С объяснением трактовать
        # как недоступность нельзя, иначе повторные попытки с неверным паролем
        # заблокируют учётную запись.
        try:
            detail = response.json()
        except ValueError:
            detail = None
        code = detail.get("detail") if isinstance(detail, dict) else None
        if isinstance(code, str) and code.strip() in TRANSIENT_LOGIN_DETAILS:
            # Сравнение точное, а не по подстроке: сообщение, лишь упоминающее
            # код, не должно попадать во временные.
            logger.warning(
                "Kappa не выдала сессию (%s, статус %s) — считаем временной "
                "нехваткой, а не ошибкой оператора",
                code.strip(), response.status_code,
            )
            raise KappaSessionQuota(code.strip())

        if detail is None:
            logger.warning(
                "Kappa ответила %s без JSON-тела (%d байт, content-type %s) — "
                "считаем недоступностью, а не неверным паролем",
                response.status_code, len(response.content or b""),
                response.headers.get("content-type"),
            )
            raise KappaUnreachable(
                f"Kappa отклонила запрос со статусом {response.status_code} "
                f"без объяснения — похоже на отказ шлюза, а не на проверку "
                f"учётных данных"
            )
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
        # user_id обязателен: по нему кто угодно позже находит живой токен
        # (find_live_session_for_user) и усыновляет прогоны, запущенные без
        # входа (adopt_orphan_runs). Без него усыновление тихо не делало
        # ничего — вход происходил, а данные так и не уходили.
        "user_id": data.get("userId"),
        "user_type_id": data.get("userTypeId"),
        "user_name": data.get("userName"),
        "first_name": data.get("firstName"),
        "last_name": data.get("lastName"),
        "token_expiry": data.get("tokenExpiryDate"),
    }


def get_session(
    session_id: str, *, include_expired: bool = False
) -> Optional[Dict[str, Any]]:
    """Получить данные сессии по session_id; None — если её нет или токен истёк.

    Истёкшая сессия — не сессия: иначе /api/kappa/me отвечал «вошли», и все
    запросы уходили в Kappa с мёртвым токеном, а вкладка валидации молча
    показывала пустой список (2026-10-06). Правило то же, что у
    find_live_session_for_user: NULL / неразборчивый срок считается живым.
    include_expired=True — только для выхода из аккаунта.
    """
    if not session_id:
        return None

    db = SessionLocal()
    try:
        record = db.query(KappaSession).filter(
            KappaSession.session_id == session_id
        ).first()

        if not record:
            return None

        if not include_expired:
            expiry = _parse_expiry(record.token_expiry)
            if expiry is not None and expiry <= datetime.now(timezone.utc):
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


def user_id_for_login(login_id: str) -> Optional[int]:
    """user_id по логину, из ранее сохранённых сессий. Без обращения к Kappa.

    Kappa кладёт в userName ровно тот логин, которым входили, поэтому связь
    точная, а не догадка. Нужно, когда Kappa недоступна, а определить датасет
    для нумерации уже надо: конфиг ключуется по user_id.
    """
    if not login_id:
        return None
    db = SessionLocal()
    try:
        record = db.query(KappaSession).filter(
            KappaSession.user_name == login_id
        ).order_by(KappaSession.created_at.desc()).first()
        return record.user_id if record else None
    finally:
        db.close()


def sole_known_user_id() -> Optional[int]:
    """Единственный пользователь, работавший на этой машине, если он один.

    Когда аккаунт на машине один, «чей это прогон» — не догадка, а факт.
    Если аккаунтов несколько, возвращаем None и не гадаем.
    """
    db = SessionLocal()
    try:
        ids = {
            row[0] for row in db.query(KappaSession.user_id).distinct().all()
            if row[0] is not None
        }
        return next(iter(ids)) if len(ids) == 1 else None
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
