"""Вход в Kappa через v2 и отличение недоступности от неверного пароля.

13.11.2026 вход перестал работать на обеих машинах и под обоими аккаунтами.
Причина оказалась не у нас: **v1 обоих микросервисов Kappa закрыт**, живёт v2.
Любой путь под `/user-micro-services/v1/`, включая несуществующий, отдавал
401 с пустым телом и `Content-Type: text/html` — то есть отклонял nginx, не
доводя запрос до приложения. v2 при этом отвечает как приложение (JSON, 422 на
неполный запрос), сертификат прежний, а интерфейс самой Kappa пускал оба
аккаунта.

Отсюда две правки, обе здесь проверяются:

1. адрес входа переведён на v2. Форма запроса та же (`loginId`/`passwd`), и
   `UserDetails` в ответе содержит все поля, которые мы читаем, включая `token`
   и `tokenExpiryDate` — проверено живым входом;
2. отказ **без разбираемого тела** больше не считается неверным паролем.
   Приложение Kappa объясняет отказ в JSON (`detail`), шлюз не объясняет
   ничего. Пока это не различалось, недоступность внешней системы выглядела
   как ошибка оператора и блокировала работу: офлайн-режим с отложенной
   выгрузкой включается только по `KappaUnreachable`.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import kappa_auth
from kappa_auth import KappaUnreachable, kappa_login

V2_LOGIN_RESPONSE = {
    "userId": 26, "userName": "e.roppert", "firstName": "Kate",
    "middleName": None, "lastName": "", "email": "kate@example.test",
    "userTypeId": 3, "orgId": 1, "orgDetails": {"orgName": "НГУ"},
    "token": "bearer-token-value", "tokenExpiryDate": "2026-11-20T10:53:14.321672Z",
    "profileStatus": 1,
}


class _Response:
    def __init__(self, status_code, body=None, raw=b"", content_type="application/json"):
        self.status_code = status_code
        self._body = body
        self.content = raw if body is None else json.dumps(body).encode()
        self.text = self.content.decode(errors="replace")
        self.headers = {"content-type": content_type}

    def json(self):
        if self._body is None:
            raise ValueError("нет JSON-тела")
        return self._body


class _Client:
    """Подмена `kappa_client`: запоминает запрос, отдаёт заготовленный ответ."""

    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        self._calls.append({"url": url, "payload": json})
        return self._response


def _install(monkeypatch, response):
    calls = []
    monkeypatch.setattr(kappa_auth, "kappa_client",
                        lambda timeout: _Client(response, calls))
    return calls


# ── адрес ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_login_goes_to_the_v2_user_service(monkeypatch):
    """Ровно то, что сломалось: адрес входа. Тестом он раньше не был прикрыт."""
    calls = _install(monkeypatch, _Response(200, V2_LOGIN_RESPONSE))

    await kappa_login("e.roppert", "secret")

    assert calls[0]["url"].endswith("/user-micro-services/v2/session/new"), calls[0]["url"]


@pytest.mark.asyncio
async def test_the_request_shape_is_unchanged(monkeypatch):
    """v2 принимает те же поля — менять нечего, и это стоит зафиксировать."""
    calls = _install(monkeypatch, _Response(200, V2_LOGIN_RESPONSE))

    await kappa_login("e.roppert", "secret")

    assert calls[0]["payload"] == {"loginId": "e.roppert", "passwd": "secret"}


@pytest.mark.asyncio
async def test_every_field_we_store_comes_from_the_v2_response(monkeypatch):
    """`UserDetails` v2 содержит всё, что читает наш код."""
    _install(monkeypatch, _Response(200, V2_LOGIN_RESPONSE))

    result = await kappa_login("e.roppert", "secret")

    assert result["user_id"] == 26
    assert result["user_name"] == "e.roppert"
    assert result["token_expiry"] == "2026-11-20T10:53:14.321672Z"
    assert result["session_id"]


@pytest.mark.asyncio
async def test_the_token_is_not_returned_to_the_caller(monkeypatch):
    """Токен живёт в БД и подставляется по session_id; наружу он не едет."""
    _install(monkeypatch, _Response(200, V2_LOGIN_RESPONSE))

    result = await kappa_login("e.roppert", "secret")

    assert "bearer-token-value" not in json.dumps(result)


# ── отказ без объяснения = недоступность, а не неверный пароль ────────────


@pytest.mark.asyncio
async def test_a_bodyless_401_is_unavailability_not_a_wrong_password(monkeypatch):
    """Ровно то, что отдавал шлюз: 401, пустое тело, text/html.

    Пока это считалось неверным паролем, офлайн-режим не включался и работать
    было нельзя вообще, хотя пайплайн от Kappa не зависит.
    """
    _install(monkeypatch, _Response(401, None, raw=b"", content_type="text/html"))

    with pytest.raises(KappaUnreachable):
        await kappa_login("e.roppert", "secret")


@pytest.mark.asyncio
async def test_a_bodyless_403_is_unavailability_too(monkeypatch):
    """Шлюз может отвечать и 403 — объяснения в нём столько же."""
    _install(monkeypatch, _Response(403, None, raw=b"", content_type="text/html"))

    with pytest.raises(KappaUnreachable):
        await kappa_login("e.roppert", "secret")


@pytest.mark.asyncio
async def test_html_that_is_not_json_is_unavailability(monkeypatch):
    """Страница ошибки от шлюза — тоже не объяснение от приложения."""
    _install(monkeypatch, _Response(502, None, raw=b"<html>502 Bad Gateway</html>",
                                    content_type="text/html"))

    with pytest.raises(KappaUnreachable):
        await kappa_login("e.roppert", "secret")


# ── отказ с объяснением = действительно неверные данные ───────────────────


@pytest.mark.asyncio
async def test_a_json_401_really_is_a_credential_rejection(monkeypatch):
    """Приложение Kappa объясняет отказ в `detail` — это ответ по существу,
    и трактовать его как недоступность нельзя: повторные попытки с неверным
    паролем блокируют учётную запись."""
    _install(monkeypatch, _Response(401, {"detail": "Invalid credentials"}))

    assert await kappa_login("e.roppert", "wrong") is None


@pytest.mark.asyncio
async def test_a_json_403_is_a_credential_rejection(monkeypatch):
    """v2 документирует 403 как «учётная запись отключена или не активирована»."""
    _install(monkeypatch, _Response(403, {"detail": "Account is disabled"}))

    assert await kappa_login("e.roppert", "secret") is None


@pytest.mark.asyncio
async def test_a_json_422_is_not_unavailability(monkeypatch):
    """422 от v2 — ошибка валидации запроса, тоже ответ по существу."""
    _install(monkeypatch, _Response(422, {"detail": [{"loc": ["body", "loginId"]}]}))

    assert await kappa_login("e.roppert", "") is None


# ── защита от повторения ──────────────────────────────────────────────────


def _v1_user_service_literals(path: Path):
    """Строки, где адрес закрытого v1 входа стоит в КОДЕ, а не в прозе.

    Подстрочный поиск здесь не годится: история этой самой поломки описана в
    комментарии `kappa_auth.py`, и он бы на нём споткнулся — как уже спотыкался
    на докстринге в `test_kappa_http.py`.
    """
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Докстринг — это тоже строковая константа, поэтому «голые» строки-выражения
    # исключаются: код от прозы отличает именно то, что строка куда-то идёт
    # (присваивание, аргумент, f-строка), а не стоит отдельным выражением.
    # Комментарии в дерево вообще не попадают.
    prose = {id(node.value) for node in ast.walk(tree)
             if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    return sorted({
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and "user-micro-services/v1" in node.value
        and id(node) not in prose
    })


def test_no_backend_module_still_points_at_the_user_service_v1():
    """Клиент данных пока остаётся на v1 (его миграция — отдельная задача),
    но сервис пользователей там закрыт, и возврат туда означает отказ входа."""
    offenders = {
        path.name: lines
        for path in sorted(Path(__file__).parent.glob("*.py"))
        if not path.name.startswith("test_")
        and (lines := _v1_user_service_literals(path))
    }

    assert offenders == {}, f"ссылка на закрытый v1 входа: {offenders}"


def test_the_guard_detects_a_real_violation_and_ignores_prose(tmp_path):
    """Проверка, которая утверждает отсутствие нарушений, обязана уметь их
    видеть — и обязана не считать нарушением рассказ о них."""
    bad = tmp_path / "offender.py"
    bad.write_text(
        'URL = "https://kappa.nsu.ru:8061/user-micro-services/v1"\n',
        encoding="utf-8")
    prose = tmp_path / "prose.py"
    prose.write_text(
        '"""Раньше тут был user-micro-services/v1 — больше нет."""\n'
        "# и в комментарии user-micro-services/v1 не нарушение\n",
        encoding="utf-8")

    assert _v1_user_service_literals(bad) == [1]
    assert _v1_user_service_literals(prose) == []
