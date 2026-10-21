"""TLS verification for every connection to Kappa (KI-064).

`verify=False` sat in 15 places across three modules, including the POST that
carries the login and password. It was legacy: the certificate is a valid
GlobalSign one for *.nsu.ru and plain verification returns 200, measured
2026-10-21. It spread from 11 sites to 15 by copying, because each call site
built its own client — so the fix is one factory, and the test that matters
most is the one forbidding a client built anywhere else.
"""
import ast
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import kappa_http
from kappa_http import kappa_client

BACKEND = Path(__file__).parent
KAPPA_MODULES = ("kappa_auth.py", "kappa_client.py", "kappa_entity_files.py")


def test_the_factory_verifies_tls_by_default():
    with patch.object(kappa_http.httpx, "AsyncClient", MagicMock()) as mock:
        kappa_client(timeout=15.0)

    assert mock.call_args.kwargs["verify"] is True


def test_the_flag_can_turn_verification_off(monkeypatch):
    """Отключение остаётся возможным — но как явное решение, а не наследство."""
    monkeypatch.setattr(kappa_http.settings, "kappa_tls_verify", False)

    with patch.object(kappa_http.httpx, "AsyncClient", MagicMock()) as mock:
        kappa_client(timeout=15.0)

    assert mock.call_args.kwargs["verify"] is False


def test_the_flag_is_read_per_call_not_at_import(monkeypatch):
    """Иначе тесты и живая правка .env расходятся с поведением процесса."""
    with patch.object(kappa_http.httpx, "AsyncClient", MagicMock()) as mock:
        kappa_client(timeout=1.0)
        assert mock.call_args.kwargs["verify"] is True
        monkeypatch.setattr(kappa_http.settings, "kappa_tls_verify", False)
        kappa_client(timeout=1.0)
        assert mock.call_args.kwargs["verify"] is False


def test_the_factory_passes_a_timeout_object_through():
    """Среди 15 мест есть и число, и httpx.Timeout — фабрика обязана принимать оба."""
    timeout = httpx.Timeout(30.0, read=300.0, write=300.0)

    with patch.object(kappa_http.httpx, "AsyncClient", MagicMock()) as mock:
        kappa_client(timeout=timeout)

    assert mock.call_args.kwargs["timeout"] is timeout


def test_disabled_verification_is_logged(monkeypatch, caplog):
    """Машина, работающая без проверки сертификата, должна об этом сказать."""
    monkeypatch.setattr(kappa_http.settings, "kappa_tls_verify", False)
    kappa_http._warned_once = False

    with patch.object(kappa_http.httpx, "AsyncClient", MagicMock()):
        with caplog.at_level("WARNING"):
            kappa_client(timeout=1.0)

    assert any("TLS" in r.message or "сертификат" in r.message
               for r in caplog.records), caplog.text


def test_the_factory_returns_what_httpx_built():
    """Фабрика не должна подменять объект — вызывающий код делает `async with`."""
    sentinel = object()
    with patch.object(kappa_http.httpx, "AsyncClient", MagicMock(return_value=sentinel)):
        assert kappa_client(timeout=1.0) is sentinel


# ── Главный тест: защита от повторного расползания ─────────────────────────


# Проверяем КОД, а не текст: подстрочный поиск ловил бы и прозу — например
# докстринг kappa_http.py, где описана история этой самой проблемы.


def _client_constructions(path: Path):
    """Строки, где конструируется httpx.AsyncClient напрямую."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "AsyncClient"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "httpx"
    ]


def _verify_false_arguments(path: Path):
    """Строки, где в вызов передан verify=False."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "verify"
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is False
    ]


@pytest.mark.parametrize("module", KAPPA_MODULES)
def test_no_kappa_module_builds_a_client_outside_the_factory(module):
    """Проблема уже отрастала: 11 мест превратились в 15 копированием.

    Пока клиента можно сконструировать на месте, следующий модуль снова
    скопирует образец у соседа вместе с отключённой проверкой. Поэтому
    конструктор в этих файлах запрещён — только фабрика.
    """
    lines = _client_constructions(BACKEND / module)

    assert lines == [], (
        f"{module} создаёт клиента напрямую (строки {lines}) — "
        f"используйте kappa_http.kappa_client()"
    )


@pytest.mark.parametrize("module", KAPPA_MODULES)
def test_no_kappa_module_disables_verification(module):
    lines = _verify_false_arguments(BACKEND / module)
    assert lines == [], f"{module} отключает проверку сертификата в строках {lines}"


def test_only_the_factory_may_decide_verification_in_the_whole_backend():
    """Включая модули, которые появятся позже: образец копируют отовсюду.

    kappa_http сам решает значение из настройки, поэтому литерального False в
    нём нет — он тут не исключение, а просто не нарушитель.
    """
    offenders = {
        p.name: _verify_false_arguments(p)
        for p in sorted(BACKEND.glob("*.py"))
        if not p.name.startswith("test_") and _verify_false_arguments(p)
    }
    assert offenders == {}, f"verify=False остался в: {offenders}"


def test_the_guards_actually_detect_a_violation(tmp_path):
    """Тест, который проходит впустую, хуже отсутствующего.

    Защитные проверки выше утверждают, что нарушений НЕТ. Если детектор
    сломается, они продолжат зеленеть на любом коде. Поэтому проверяем, что на
    заведомом нарушении он срабатывает.
    """
    bad = tmp_path / "offender.py"
    bad.write_text(
        "import httpx\n"
        "async def go():\n"
        "    async with httpx.AsyncClient(timeout=1.0, verify=False) as c:\n"
        "        return await c.get('https://example.test')\n",
        encoding="utf-8",
    )

    assert _client_constructions(bad) == [3]
    assert _verify_false_arguments(bad) == [3]


def test_the_guards_ignore_prose(tmp_path):
    """Докстринг kappa_http.py описывает verify=False как историю — и это не
    нарушение. Подстрочный поиск спотыкался именно здесь."""
    prose = tmp_path / "prose.py"
    prose.write_text(
        '"""Раньше здесь стояло httpx.AsyncClient(verify=False) — больше нет."""\n'
        "# и в комментарии verify=False тоже не нарушение\n"
        "X = 'httpx.AsyncClient(verify=False)'\n",
        encoding="utf-8",
    )

    assert _client_constructions(prose) == []
    assert _verify_false_arguments(prose) == []
