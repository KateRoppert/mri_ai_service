"""Отказ выдать сессию не превращается в повтор непроверенного пароля.

16.11.2026 вход отказал с `403 {"detail":"TOO_MANY_ACTIVE_DEVICES"}`: v2
ограничивает число активных сессий. Правило «отказ с JSON-телом = ошибка
оператора» получило исключение, и исключение сломало инвариант, на котором
держался `kappa_pending_login`.

Деление было двоичным: Kappa **не ответила** → придержать введённое и войти
самим; Kappa **ответила и отклонила** → немедленно забыть, потому что повтор
неверного пароля блокирует учётную запись. Квота — третий случай: Kappa
ответила, но пароль НЕ проверяла. Такие данные не подтверждены и не
отклонены, поэтому не отбрасывались и повторялись воркером до 24 часов.

Отсюда `KappaSessionQuota`: для всего остального это та же недоступность
(офлайн-режим работает, обработку запускать можно), и отличается ровно одним —
непроверенный пароль не придерживается.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import app as app_module
import kappa_pending_login as pending
from kappa_auth import KappaSessionQuota, KappaUnreachable
from app import KappaLoginRequest


@pytest.fixture(autouse=True)
def _clean_pending():
    pending.forget("тест")
    pending.clear_outcome()
    yield
    pending.forget("тест")
    pending.clear_outcome()


def test_the_quota_refusal_is_still_an_outage_for_everything_else():
    """Наследование намеренное: все существующие обработчики недоступности
    должны продолжать работать без правок."""
    assert issubclass(KappaSessionQuota, KappaUnreachable)
    assert isinstance(KappaSessionQuota("TOO_MANY_ACTIVE_DEVICES"), KappaUnreachable)


# ── эндпоинт входа ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_endpoint_does_not_hold_the_password_on_a_quota_refusal(monkeypatch):
    """Главное в этой правке: придержанный непроверенный пароль воркер
    повторял бы до суток."""
    async def _quota(login_id, passwd):
        raise KappaSessionQuota("TOO_MANY_ACTIVE_DEVICES")

    monkeypatch.setattr(app_module, "kappa_login", _quota)

    with pytest.raises(app_module.HTTPException) as exc:
        await app_module.kappa_login_endpoint(
            KappaLoginRequest(login_id="e.roppert", passwd="secret"))

    assert exc.value.status_code == 503
    assert "устройств" in exc.value.detail, "оператор должен узнать причину"
    assert pending.held_login() is None, "пароль не должен быть придержан"


@pytest.mark.asyncio
async def test_the_endpoint_still_holds_the_password_when_kappa_is_silent(monkeypatch):
    """Защита от регрессии: ради этого случая механизм и написан."""
    async def _down(login_id, passwd):
        raise KappaUnreachable("соединение оборвалось")

    monkeypatch.setattr(app_module, "kappa_login", _down)

    with pytest.raises(app_module.HTTPException) as exc:
        await app_module.kappa_login_endpoint(
            KappaLoginRequest(login_id="e.roppert", passwd="secret"))

    assert exc.value.status_code == 503
    assert pending.held_login() == "e.roppert"


# ── фоновый воркер ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_worker_drops_the_password_on_a_quota_refusal(monkeypatch):
    """Данные могли быть придержаны ещё при обрыве связи, а квоту встретить
    позже — тогда отбрасывает воркер."""
    async def _quota(login_id, passwd):
        raise KappaSessionQuota("TOO_MANY_ACTIVE_DEVICES")

    monkeypatch.setattr("kappa_auth.kappa_login", _quota)
    pending.remember("e.roppert", "secret")

    assert await pending.try_login_now() is None
    assert pending.held_login() is None, "непроверенный пароль не повторяем"
    assert pending.outcome() == {"status": "quota", "login_id": "e.roppert"}


@pytest.mark.asyncio
async def test_the_worker_keeps_waiting_while_kappa_is_silent(monkeypatch):
    """Защита от регрессии: молчание — это «ещё не ответила», данные нужны."""
    async def _down(login_id, passwd):
        raise KappaUnreachable("таймаут")

    monkeypatch.setattr("kappa_auth.kappa_login", _down)
    pending.remember("e.roppert", "secret")

    assert await pending.try_login_now() is None
    assert pending.held_login() == "e.roppert"


@pytest.mark.asyncio
async def test_a_rejected_password_is_still_dropped_at_once(monkeypatch):
    """Инвариант, который и пострадал: неверный пароль уходит в Kappa не
    более одного раза."""
    calls = []

    async def _rejected(login_id, passwd):
        calls.append(login_id)
        return None

    monkeypatch.setattr("kappa_auth.kappa_login", _rejected)
    pending.remember("e.roppert", "wrong")

    assert await pending.try_login_now() is None
    assert pending.held_login() is None
    assert await pending.try_login_now() is None, "повторять нечего"
    assert len(calls) == 1, "второй отправки пароля быть не должно"


# ── освобождение слота устройства ─────────────────────────────────────────


class _Resp:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self._body = body
        self.content = b"" if body is None else b"{}"
        self.text = ""
        self.headers = {"content-type": "application/json"}

    def json(self):
        if self._body is None:
            raise ValueError("нет тела")
        return self._body


class _DelClient:
    def __init__(self, response, calls, boom=False):
        self._r, self._calls, self._boom = response, calls, boom

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def delete(self, url, **kw):
        if self._boom:
            raise RuntimeError("сеть упала")
        self._calls.append({"url": url, **kw})
        return self._r


@pytest.mark.asyncio
async def test_release_sends_delete_with_that_token(monkeypatch):
    """Списка сессий API не даёт, поэтому гасить можно только своим токеном."""
    import kappa_auth
    from kappa_auth import release_kappa_session

    calls = []
    monkeypatch.setattr(kappa_auth, "kappa_client",
                        lambda timeout: _DelClient(_Resp(204), calls))

    assert await release_kappa_session("tok-123") is True
    assert calls[0]["url"].endswith("/user-micro-services/v2/session")
    assert calls[0]["headers"]["Authorization"] == "Bearer tok-123"


@pytest.mark.asyncio
async def test_release_never_raises_into_the_caller(monkeypatch):
    """Освобождение слота — попытка, а не обязательство: срывать из-за неё
    выход из аккаунта нельзя."""
    import kappa_auth
    from kappa_auth import release_kappa_session

    monkeypatch.setattr(kappa_auth, "kappa_client",
                        lambda timeout: _DelClient(None, [], boom=True))

    assert await release_kappa_session("tok") is False


@pytest.mark.asyncio
async def test_release_without_a_token_does_not_call_kappa(monkeypatch):
    import kappa_auth
    from kappa_auth import release_kappa_session

    calls = []
    monkeypatch.setattr(kappa_auth, "kappa_client",
                        lambda timeout: _DelClient(_Resp(204), calls))

    assert await release_kappa_session("") is False
    assert calls == []


@pytest.mark.asyncio
async def test_a_refusal_to_log_out_is_reported_not_hidden(monkeypatch):
    import kappa_auth
    from kappa_auth import release_kappa_session

    monkeypatch.setattr(kappa_auth, "kappa_client",
                        lambda timeout: _DelClient(_Resp(401, {"detail": "expired"}), []))

    assert await release_kappa_session("stale") is False
