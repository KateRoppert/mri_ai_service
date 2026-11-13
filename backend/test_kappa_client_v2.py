"""Клиент данных Kappa говорит с v2 (KI-082).

v1 обоих микросервисов закрыт 13.11.2026 (KI-081), поэтому выгрузка не
работала. Отличие v2 на всех восьми вызовах одно и то же: из пути уходят
`{user_id}/{user_type_id}` — личность берётся из токена. Форма тел и ответов
совпадает, это проверено живыми пробами ещё в октябре
(`docs/superpowers/specs/2026-10-02-kappa-replace-on-reprocess-design.md`).

Адреса раньше не были закреплены тестами ни у входа, ни здесь — именно поэтому
версия и сгнила незаметно. Теперь каждый закреплён.

Параметры `user_id`/`user_type_id` в сигнатурах сохранены: их передают
вызывающие по всему проекту (загрузчик, доставщик, `app.py`), и выкидывать их
означало бы трогать весь этот путь ради косметики. В URL они больше не идут, а
в телах запросов `userId` остался там, где он и был.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import kappa_client


class _Response:
    def __init__(self, status_code=200, body=None, content=b""):
        self.status_code = status_code
        self._body = body
        self.content = content if body is None else json.dumps(body).encode()
        self.text = self.content.decode(errors="replace")
        self.headers = {"content-type": "application/json"}

    @property
    def is_success(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._body is None:
            raise ValueError("нет JSON-тела")
        return self._body


class _Client:
    def __init__(self, response, calls):
        self._response = response
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def _record(self, method, url, **kwargs):
        self._calls.append({"method": method, "url": url, **kwargs})
        return self._response

    async def get(self, url, **kw):
        return await self._record("GET", url, **kw)

    async def post(self, url, **kw):
        return await self._record("POST", url, **kw)

    async def put(self, url, **kw):
        return await self._record("PUT", url, **kw)


def _install(monkeypatch, response):
    calls = []
    monkeypatch.setattr(kappa_client, "kappa_client",
                        lambda timeout: _Client(response, calls))
    return calls


BASE = "https://kappa.nsu.ru:8061/data-micro-services/v2"


# ── адрес каждого вызова ──────────────────────────────────────────────────


def test_the_base_url_is_v2():
    assert kappa_client.KAPPA_DATA_URL == BASE


@pytest.mark.asyncio
async def test_listing_datasets_drops_the_user_path(monkeypatch):
    calls = _install(monkeypatch, _Response(body=[{"datasetId": 1}]))

    await kappa_client.list_user_datasets("tok", 26, 3)

    assert calls[0]["url"] == f"{BASE}/datasets"


@pytest.mark.asyncio
async def test_creating_a_dataset_drops_the_user_path(monkeypatch):
    calls = _install(monkeypatch, _Response(body=[{"datasetName": "DS", "datasetId": 7}]))

    result = await kappa_client.create_dataset("tok", 26, 3, "DS")

    assert calls[0]["method"] == "POST"
    assert calls[0]["url"] == f"{BASE}/datasets/new"
    assert result == 7, "id по-прежнему добирается перечислением по имени"


@pytest.mark.asyncio
async def test_the_create_payload_still_carries_user_id(monkeypatch):
    """Из пути `userId` ушёл, но в теле он остался — так в схеме v2."""
    calls = _install(monkeypatch, _Response(body=[]))

    await kappa_client.create_dataset("tok", 26, 3, "DS", dataset_tags="t")

    assert calls[0]["json"]["userId"] == 26
    assert calls[0]["json"]["datasetName"] == "DS"


@pytest.mark.asyncio
async def test_uploading_an_entity_keeps_only_the_dataset_in_the_path(monkeypatch, tmp_path):
    calls = _install(monkeypatch, _Response(body={"dsEntityId": "e1"}))
    f = tmp_path / "a.json"
    f.write_text("{}", encoding="utf-8")

    await kappa_client.upload_entity("tok", 26, 3, 350, "sub-001", [f])

    assert calls[0]["url"] == f"{BASE}/datasets/datasetEntities/new/350"


@pytest.mark.asyncio
async def test_listing_entities_keeps_only_the_dataset_in_the_path(monkeypatch):
    calls = _install(monkeypatch, _Response(body=[{"dsEntityId": "e1"}]))

    await kappa_client.get_dataset_entities("tok", 26, 3, 350)

    assert calls[0]["url"] == f"{BASE}/datasets/datasetEntities/350"


@pytest.mark.asyncio
async def test_entity_details_keep_dataset_and_entity(monkeypatch):
    calls = _install(monkeypatch, _Response(body={"dsEntityId": "e1"}))

    await kappa_client.get_entity_details("tok", 26, 3, 350, "e1")

    assert calls[0]["url"] == f"{BASE}/datasets/datasetEntities/350/e1"


@pytest.mark.asyncio
async def test_downloading_a_file_keeps_dataset_and_file(monkeypatch):
    calls = _install(monkeypatch, _Response(content=b"nii"))

    data = await kappa_client.download_entity_file("tok", 26, 3, 350, "f9")

    assert calls[0]["url"] == f"{BASE}/datasets/datasetEntities/files/350/f9"
    assert data == b"nii"


@pytest.mark.asyncio
async def test_appending_a_file_keeps_dataset_and_entity(monkeypatch, tmp_path):
    """Тот же эндпоинт, что у обновления сущности: в v2 PUT принимает ту же
    пару `update_dataset_entity` + `files`, что и v1 — и так же ДОПИСЫВАЕТ.

    Проверено живой пробой 13.11.2026 на одноразовом датасете 376: файл под
    существующим именем дал второй файл с другим id, как и в v1.
    """
    calls = _install(monkeypatch, _Response(body={"files": [{"fileId": "f1"}]}))
    f = tmp_path / "mask.nii.gz"
    f.write_bytes(b"x")

    await kappa_client.append_entity_file("tok", 26, 3, 350, "e1", f)

    assert calls[0]["method"] == "PUT"
    assert calls[0]["url"] == f"{BASE}/datasets/datasetEntities/350/e1"


@pytest.mark.asyncio
async def test_setting_status_uses_the_module_base_url(monkeypatch):
    """Этот вызов перевели на v2 ещё в KI-036/KI-075, импортируя адрес из
    соседнего модуля. Теперь базовый адрес модуля и так v2 — импорт лишний."""
    calls = _install(monkeypatch, _Response(body={}))

    ok, _ = await kappa_client.set_entity_status("tok", 26, 3, 350, "e1", status=3)

    assert ok
    assert calls[0]["url"] == f"{BASE}/datasets/datasetEntities/350/e1"


# ── защита от возврата на закрытый v1 ─────────────────────────────────────


def _v1_literals(path: Path, needle: str):
    """Строки, где закрытый адрес стоит в КОДЕ, а не в прозе.

    Докстринги и комментарии рассказывают историю этой поломки, и подстрочный
    поиск спотыкался бы на них — как уже спотыкался дважды (KI-081).
    """
    import ast
    tree = ast.parse(path.read_text(encoding="utf-8"))
    prose = {id(n.value) for n in ast.walk(tree)
             if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)}
    return sorted({n.lineno for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)
                   and needle in n.value and id(n) not in prose})


@pytest.mark.parametrize("needle", ["data-micro-services/v1",
                                    "user-micro-services/v1"])
def test_no_backend_module_points_at_a_closed_v1_service(needle):
    offenders = {
        path.name: lines
        for path in sorted(Path(__file__).parent.glob("*.py"))
        if not path.name.startswith("test_")
        and (lines := _v1_literals(path, needle))
    }

    assert offenders == {}, f"ссылка на закрытый {needle}: {offenders}"
