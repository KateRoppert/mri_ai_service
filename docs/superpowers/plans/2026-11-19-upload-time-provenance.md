# Фиксация происхождения данных при выгрузке — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** выгрузка в Kappa записывает происхождение данных на момент прогона, а не на момент выгрузки, и одно исследование может быть известно в нескольких датасетах одновременно.

**Architecture:** два независимых изменения с общим корнем. (A) `preprocessing_id` закрепляется при старте прогона в новой колонке `pipeline_runs.preprocessing_id`, а `KappaUploader` читает закреплённое значение и считает из файла только при `NULL` — в точности как уже деградирует `kappa_dataset_id`. История версий конфига переезжает из отслеживаемого гитом JSON в таблицу SQLite. (B) ключ `patient_registry` становится парой `(study_hash, COALESCE(kappa_dataset_id, 0))`, а `register_patient` получает правило присвоения строки с `NULL`, которое сохраняет существующую передачу `session_registry` → `KappaUploader`.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy (императивные `Column`, без Alembic), SQLite 3.50.3, pytest.

**Spec:** `docs/superpowers/specs/2026-11-19-upload-time-provenance-design.md`

**Ветка:** `fix/upload-time-provenance` (уже создана, от `main`; в ней лежат спека и записи KI-084/KI-085)

## Global Constraints

- **Никогда `git add -A` и `git commit -a`.** `configs/kappa_datasets.yaml`,
  `pipeline_config.yaml` и `configs/preprocessing_config.yaml` намеренно
  держатся грязными как машинно-локальные настройки. Каждый коммит перечисляет
  пути явно.
- Комментарии в коде — на английском. Сообщения коммитов — на русском,
  conventional commits.
- Новые тесты — только `backend/test_*.py`, рядом с кодом. Папка `tests/` в
  `.gitignore`, туда не класть.
- Тесты пишут только в `tmp_path`. Ни в рабочие конфиги, ни в корень репозитория.
- Миграции — добавочные и идемпотентные, по образцу
  `backend/database.py::_migrate_add_kappa_dataset_id` (`database.py:401`).
  Alembic в проекте нет.
- Данные клинические: миграция, наткнувшаяся на конфликт, **не применяется** и
  печатает конфликты. Автоматическое слияние строк запрещено.
- `backend/conftest.py` уводит тесты в временную БД и вызывает `init_db()` и
  `ensure_tables()` при импорте — новые миграции обязаны проходить оттуда.
- Push — только по явной просьбе владельца. Коммиты по задачам в ветку делать
  по ходу.
- Перед коммитом код обязан как минимум импортироваться; прогонять
  `python -m pytest backend/ -q`.

---

## File Structure

| Файл | Что в нём меняется |
|---|---|
| `backend/database.py` | колонка `PipelineRun.preprocessing_id`, параметр `create_pipeline_run`, миграция `_migrate_add_preprocessing_id`, её вызов в `init_db` |
| `backend/numbering.py` | новый `current_preprocessing_id()` — единственное определение «из какого файла считаем» |
| `backend/app.py` | вычисление и передача `preprocessing_id` при старте прогона |
| `backend/requeue_service.py` | то же для requeue/возобновления, с учётом снимка конфига |
| `backend/kappa_uploader.py` | необязательный параметр `preprocessing_id`, расчёт из файла только при `None` |
| `backend/kappa_delivery_worker.py` | передача закреплённого значения в `build_uploader` |
| `backend/pipeline_monitor.py` | то же в `_create_kappa_uploader` |
| `backend/preprocessing_version.py` | история версий в SQLite, старый JSON — только на чтение |
| `backend/registry_models.py` | `study_hash` теряет `unique=True`; новый уникальный индекс по выражению |
| `backend/patient_registry.py` | правило upsert, миграция индекса, `find_by_study_hash` с датасетом |
| `backend/schema.sql` | пересобранный дамп схемы |
| `scripts/restore_registry_from_dump.py` | разовое восстановление строк, перенаправленных догрузкой |

Новые тесты: `backend/test_run_preprocessing_pin.py`,
`backend/test_uploader_preprocessing_source.py`,
`backend/test_preprocessing_version_store.py`,
`backend/test_registry_dataset_key.py`,
`backend/test_restore_registry_from_dump.py`.

---

## Task 1: Закрепить `preprocessing_id` на строке прогона

**Files:**
- Modify: `backend/database.py` (модель ~`60-95`, `create_pipeline_run:134-175`, `init_db:345-353`, миграции после `_migrate_add_queued_requeue`)
- Modify: `backend/numbering.py` (импорты `14-20`, новая функция после `scope_for_run`)
- Modify: `backend/app.py:548-583`
- Modify: `backend/requeue_service.py:109-123`
- Test: `backend/test_run_preprocessing_pin.py`

**Interfaces:**
- Produces: `numbering.current_preprocessing_id(config_path: Optional[Path] = None) -> Optional[str]` — идентификатор живого (или переданного) конфига предобработки; `None`, если файла нет.
- Produces: `database.create_pipeline_run(..., preprocessing_id: Optional[str] = None)`.
- Produces: `PipelineRun.preprocessing_id` — `VARCHAR NULL`, `NULL` = «прогон старше этой правки».

- [ ] **Step 1: Написать падающий тест**

Создать `backend/test_run_preprocessing_pin.py`:

```python
"""preprocessing_id закрепляется в момент старта прогона, а не выгрузки.

KI-085: расчёт из живого конфига при выгрузке метит данные настройками,
которые действуют сейчас, а не теми, с которыми считали.
"""
import yaml

from database import SessionLocal, create_pipeline_run, get_pipeline_run
from preprocessing_version import compute_preprocessing_id


def _config(tmp_path, blur: float):
    """A preprocessing config whose functional part depends on `blur`."""
    path = tmp_path / "preprocessing_config.yaml"
    path.write_text(yaml.safe_dump({
        "atlas": {"name": "MNI152", "filename": "mni.nii.gz"},
        "modalities": ["t1", "t1c"],
        "steps": [{"name": "smooth", "enabled": True,
                   "params": {"sigma": blur}}],
    }), encoding="utf-8")
    return path


def test_create_pipeline_run_stores_preprocessing_id(tmp_path):
    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db,
            input_path=str(tmp_path / "in"),
            output_path=str(tmp_path / "out"),
            preprocessing_id="deadbeef",
        )
        run_id = run.run_id
    finally:
        db.close()

    db = SessionLocal()
    try:
        assert get_pipeline_run(db, run_id).preprocessing_id == "deadbeef"
    finally:
        db.close()


def test_create_pipeline_run_defaults_to_none(tmp_path):
    """Старые пути (CLI, orchestrator) не передают идентификатор."""
    db = SessionLocal()
    try:
        run = create_pipeline_run(
            db,
            input_path=str(tmp_path / "in2"),
            output_path=str(tmp_path / "out2"),
        )
        assert run.preprocessing_id is None
    finally:
        db.close()


def test_current_preprocessing_id_reads_the_given_config(tmp_path):
    import numbering

    path = _config(tmp_path, blur=1.5)
    assert numbering.current_preprocessing_id(path) == \
        compute_preprocessing_id(str(path))


def test_current_preprocessing_id_is_none_without_a_config(tmp_path):
    """Отсутствие конфига не должно валить старт прогона."""
    import numbering

    assert numbering.current_preprocessing_id(tmp_path / "нет.yaml") is None


def test_different_configs_give_different_ids(tmp_path):
    """Иначе тест на закрепление ничего не доказывает."""
    import numbering

    a = numbering.current_preprocessing_id(_config(tmp_path / "a", blur=1.0))
    b = numbering.current_preprocessing_id(_config(tmp_path / "b", blur=2.0))
    assert a is not None and b is not None and a != b
```

Создать подкаталоги `tmp_path / "a"` и `tmp_path / "b"` внутри `_config` —
добавить в неё `path.parent.mkdir(parents=True, exist_ok=True)` до записи.

- [ ] **Step 2: Прогнать и убедиться, что падает**

Run: `cd backend && python -m pytest test_run_preprocessing_pin.py -q`
Expected: FAIL — `TypeError: create_pipeline_run() got an unexpected keyword
argument 'preprocessing_id'` и `AttributeError: module 'numbering' has no
attribute 'current_preprocessing_id'`.

- [ ] **Step 3: Колонка и миграция**

В `backend/database.py`, в модели `PipelineRun`, сразу под `kappa_dataset_id`:

```python
    # The preprocessing config this run actually used, pinned at start
    # (backend/numbering.py). Recomputing it at upload time read the LIVE
    # config file, so a deferred delivery — an outage, offline mode, a
    # backfill months later — labelled the data with today's settings
    # instead of the ones it was processed with (KI-085). NULL means the run
    # predates this column; KappaUploader then falls back to the file, the
    # same way it falls back for kappa_dataset_id.
    preprocessing_id = Column(String, nullable=True)
```

Новая миграция рядом с остальными:

```python
def _migrate_add_preprocessing_id():
    """Add pipeline_runs.preprocessing_id if it is not there yet."""
    import sqlalchemy
    with engine.connect() as conn:
        cols = [row[1] for row in conn.execute(
            sqlalchemy.text("PRAGMA table_info(pipeline_runs)")
        )]
        if 'preprocessing_id' not in cols:
            conn.execute(sqlalchemy.text(
                "ALTER TABLE pipeline_runs ADD COLUMN preprocessing_id VARCHAR"
            ))
            conn.commit()
```

Добавить `_migrate_add_preprocessing_id()` в конец `init_db()`.

В `create_pipeline_run` — параметр и его передача в конструктор:

```python
    preprocessing_id: Optional[str] = None,
```
```python
        preprocessing_id=preprocessing_id,
```

- [ ] **Step 4: `current_preprocessing_id` в numbering**

В `backend/numbering.py` добавить после `scope_for_run`:

```python
def current_preprocessing_id(
    config_path: Optional[Path] = None,
) -> Optional[str]:
    """The preprocessing id to pin on a run that starts now.

    One definition of WHICH file the id comes from, shared by the run-start
    path in app.py and the requeue path, so the two cannot drift apart. The
    file is read here, at start, which is the only moment the answer is
    certainly right: by upload time the operator may have edited it (KI-085).

    Returns None when the config is missing — a run must still start, and the
    uploader's own fallback covers a NULL id.
    """
    path = Path(config_path) if config_path is not None else PREPROCESSING_CONFIG
    if not path.is_file():
        logger.warning("Preprocessing config not found, run left unpinned: %s", path)
        return None
    try:
        return compute_preprocessing_id(str(path))
    except Exception as exc:  # noqa: BLE001 — старт важнее метки
        logger.warning("Could not compute preprocessing id from %s: %s", path, exc)
        return None
```

Добавить `from pathlib import Path` в импорты модуля.

- [ ] **Step 5: Прогнать тесты задачи**

Run: `cd backend && python -m pytest test_run_preprocessing_pin.py -q`
Expected: PASS (5 тестов).

- [ ] **Step 6: Передать идентификатор при старте прогона**

В `backend/app.py`, **сразу после** вызова `numbering.scope_for_run` (строка
~548-550) и **до** второго `_refuse_if_path_busy`:

```python
    # Pinned here, next to the dataset, and for the same reason: by upload
    # time the live preprocessing config may be a different file (KI-085).
    run_preprocessing_id = numbering.current_preprocessing_id()
```

> **ВАЖНО:** между вторым `_refuse_if_path_busy(db, output_path)` и
> `create_pipeline_run` вставлять **ничего** нельзя. Там намеренно закрыта
> гонка двух одновременных старта («Again, with the row write as the very next
> statement», `app.py:566-569`). Вычисление идёт выше этой пары.

В вызов `create_pipeline_run` добавить:

```python
        preprocessing_id=run_preprocessing_id,
```

- [ ] **Step 7: То же для requeue и возобновления**

В `backend/requeue_service.py::start_requeue_run` (имя по строке 99), перед
`create_pipeline_run`:

```python
    # A requeue is a NEW run: it reprocesses with the config in force now, so
    # it gets its own id rather than inheriting the original run's. Resuming a
    # stopped run is the exception — it continues with the snapshot taken when
    # the run was stopped (app.py:659), so the id must come from that file.
    preprocessing_id = numbering.current_preprocessing_id(
        preprocessing_snapshot if preprocessing_snapshot is not None else None
    )
```

И в вызов `create_pipeline_run`:

```python
        preprocessing_id=preprocessing_id,
```

- [ ] **Step 8: Прогнать весь backend**

Run: `cd backend && python -m pytest . -q`
Expected: PASS, без новых падений. Ожидается, что `test_requeue_service.py` и
`test_app_start_guard.py` продолжают проходить: сигнатура `scope_for_run` не
менялась, поэтому их заглушки трогать не нужно.

- [ ] **Step 9: Коммит**

```bash
git add backend/database.py backend/numbering.py backend/app.py \
        backend/requeue_service.py backend/test_run_preprocessing_pin.py
git commit -m "fix(kappa): закрепить preprocessing_id при старте прогона (KI-085)"
```

---

## Task 2: Загрузчик читает закреплённое значение

**Files:**
- Modify: `backend/kappa_uploader.py:75-110`
- Modify: `backend/kappa_delivery_worker.py:40-66`
- Modify: `backend/pipeline_monitor.py:140-185`
- Test: `backend/test_uploader_preprocessing_source.py`

**Interfaces:**
- Consumes: `PipelineRun.preprocessing_id` (Task 1).
- Produces: `KappaUploader(..., preprocessing_id: Optional[str] = None)`. При
  переданном значении конструктор **не читает** файл конфига.

- [ ] **Step 1: Написать падающий тест**

Создать `backend/test_uploader_preprocessing_source.py`:

```python
"""KappaUploader берёт preprocessing_id с прогона, а файл — только как фолбэк.

Тест на сам дефект KI-085: прогон стартовал при одном конфиге, выгружается
при другом, а в реестр обязан попасть идентификатор СТАРТА.
"""
import yaml
import pytest

from kappa_uploader import KappaUploader
from preprocessing_version import compute_preprocessing_id


def _config(path, sigma):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "atlas": {"name": "MNI152", "filename": "mni.nii.gz"},
        "modalities": ["t1"],
        "steps": [{"name": "smooth", "enabled": True,
                   "params": {"sigma": sigma}}],
    }), encoding="utf-8")
    return path


def _uploader(tmp_path, **kwargs):
    return KappaUploader(
        run_id="run-1",
        output_path=str(tmp_path / "out"),
        token="t",
        user_id=1,
        user_type_id=3,
        lesion_type="glioblastoma",
        **kwargs,
    )


def test_pinned_id_wins_over_the_live_config(tmp_path):
    at_start = compute_preprocessing_id(
        str(_config(tmp_path / "start" / "c.yaml", sigma=1.0)))
    live = _config(tmp_path / "live" / "c.yaml", sigma=9.0)

    uploader = _uploader(
        tmp_path,
        preprocessing_config_path=str(live),
        preprocessing_id=at_start,
    )
    assert uploader.preprocessing_id == at_start
    assert uploader.preprocessing_id != compute_preprocessing_id(str(live))


def test_pinned_id_means_the_config_file_is_never_read(tmp_path):
    """Жёсткая проверка: путь заведомо несуществующий, конструктор обязан пройти."""
    uploader = _uploader(
        tmp_path,
        preprocessing_config_path=str(tmp_path / "нет-такого.yaml"),
        preprocessing_id="deadbeef",
    )
    assert uploader.preprocessing_id == "deadbeef"


def test_without_a_pinned_id_the_file_is_used(tmp_path):
    """Путь деградации для прогонов, стартовавших до этой правки."""
    live = _config(tmp_path / "legacy" / "c.yaml", sigma=3.0)
    uploader = _uploader(tmp_path, preprocessing_config_path=str(live))
    assert uploader.preprocessing_id == compute_preprocessing_id(str(live))


def test_no_pinned_id_and_no_file_still_fails_loudly(tmp_path):
    """Молча проглотить отсутствие и файла, и метки нельзя."""
    with pytest.raises(OSError):
        _uploader(tmp_path,
                  preprocessing_config_path=str(tmp_path / "нет.yaml"))
```

- [ ] **Step 2: Прогнать и убедиться, что падает**

Run: `cd backend && python -m pytest test_uploader_preprocessing_source.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument
'preprocessing_id'`.

- [ ] **Step 3: Параметр в конструкторе**

В `backend/kappa_uploader.py::KappaUploader.__init__` добавить параметр после
`superseding_sessions`:

```python
        preprocessing_id: Optional[str] = None,
```

и заменить строку 105:

```python
        self.preprocessing_id = compute_preprocessing_id(preprocessing_config_path)
```

на:

```python
        # Pinned at run start (backend/numbering.py) and read off the run row
        # by both build sites. Recomputed from the file ONLY when the run
        # predates that column: the file is the LIVE config, and a deferred
        # delivery can run days or months after the processing it labels
        # (KI-085). Same shape of fallback as dataset_id above.
        self.preprocessing_id = (
            preprocessing_id
            or compute_preprocessing_id(preprocessing_config_path)
        )
```

- [ ] **Step 4: Прогнать тесты задачи**

Run: `cd backend && python -m pytest test_uploader_preprocessing_source.py -q`
Expected: PASS (4 теста).

- [ ] **Step 5: Передать значение из обоих мест сборки**

В `backend/kappa_delivery_worker.py::build_uploader` (строка ~56), в вызов
`KappaUploader`:

```python
        preprocessing_id=getattr(run, "preprocessing_id", None),
```

В `backend/pipeline_monitor.py::_create_kappa_uploader` в том же блоке, где уже
читается строка прогона (строки ~166-172), добавить к чтению:

```python
                pinned_preprocessing_id = (
                    run.preprocessing_id if run else None
                )
```

и в вызов `KappaUploader`:

```python
                preprocessing_id=pinned_preprocessing_id,
```

- [ ] **Step 6: Прогнать весь backend**

Run: `cd backend && python -m pytest . -q`
Expected: PASS.

- [ ] **Step 7: Коммит**

```bash
git add backend/kappa_uploader.py backend/kappa_delivery_worker.py \
        backend/pipeline_monitor.py \
        backend/test_uploader_preprocessing_source.py
git commit -m "fix(kappa): выгрузка берёт preprocessing_id с прогона, а не из живого конфига (KI-085)"
```

---

## Task 3: История версий конфига в SQLite

**Files:**
- Modify: `backend/preprocessing_version.py`
- Modify: `backend/registry_models.py` (новая модель)
- Modify: `backend/app.py` (вызов `register_version` при старте)
- Modify: `backend/requeue_service.py` (то же)
- Test: `backend/test_preprocessing_version_store.py`

**Interfaces:**
- Consumes: `numbering.current_preprocessing_id` (Task 1).
- Produces: модель `PreprocessingVersion(preprocessing_id, created_at, config_json)`.
- Produces: `register_version(config_path) -> Optional[str]` пишет в SQLite;
  `get_version_config(preprocessing_id)` читает сначала SQLite, затем старый JSON.

**Почему не в JSON:** `configs/preprocessing_versions.json` отслеживается гитом,
а `configs/` смонтирован в контейнер на запись. Писать туда из приложения
означает делать рабочую копию грязной при каждой правке конфига — рядом с
тремя файлами, которые и так намеренно грязные.

- [ ] **Step 1: Написать падающий тест**

Создать `backend/test_preprocessing_version_store.py`:

```python
"""По preprocessing_id должны находиться настройки, которыми считали.

KI-085, вторая часть: register_version существовал, но в продакшене не
вызывался нигде, и у трёх из четырёх использованных идентификаторов за ними
не стояло ничего.
"""
import json

import yaml

import preprocessing_version as pv


def _config(path, sigma):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({
        "atlas": {"name": "MNI152", "filename": "mni.nii.gz"},
        "modalities": ["t1", "t1c"],
        "steps": [{"name": "smooth", "enabled": True,
                   "params": {"sigma": sigma}}],
    }), encoding="utf-8")
    return path


def test_register_version_makes_the_id_resolvable(tmp_path):
    path = _config(tmp_path / "c.yaml", sigma=1.0)
    prep_id = pv.register_version(str(path))

    stored = pv.get_version_config(prep_id)
    assert stored is not None
    assert stored["steps"][0]["params"]["sigma"] == 1.0


def test_register_version_is_idempotent(tmp_path):
    path = _config(tmp_path / "c.yaml", sigma=1.0)
    assert pv.register_version(str(path)) == pv.register_version(str(path))


def test_register_version_does_not_touch_the_json_file(tmp_path, monkeypatch):
    """Приложение не имеет права делать рабочую копию грязной."""
    legacy = tmp_path / "preprocessing_versions.json"
    legacy.write_text(json.dumps({"old1234": {"config": {"steps": []}}}),
                      encoding="utf-8")
    before = legacy.read_text(encoding="utf-8")
    monkeypatch.setattr(pv, "VERSIONS_FILE", legacy)

    pv.register_version(str(_config(tmp_path / "c.yaml", sigma=2.0)))

    assert legacy.read_text(encoding="utf-8") == before


def test_legacy_json_is_still_readable(tmp_path, monkeypatch):
    """Единственная унаследованная запись 1e2b93ad должна остаться доступной."""
    legacy = tmp_path / "preprocessing_versions.json"
    legacy.write_text(json.dumps({
        "1e2b93ad": {"created_at": "2026-04-02T08:49:56",
                     "config": {"steps": [{"name": "legacy"}]}},
    }), encoding="utf-8")
    monkeypatch.setattr(pv, "VERSIONS_FILE", legacy)

    assert pv.get_version_config("1e2b93ad")["steps"][0]["name"] == "legacy"


def test_unknown_id_resolves_to_none(tmp_path, monkeypatch):
    monkeypatch.setattr(pv, "VERSIONS_FILE", tmp_path / "нет.json")
    assert pv.get_version_config("ffffffff") is None
```

- [ ] **Step 2: Прогнать и убедиться, что падает**

Run: `cd backend && python -m pytest test_preprocessing_version_store.py -q`
Expected: FAIL — `test_register_version_does_not_touch_the_json_file` падает
(сейчас `register_version` пишет именно в этот файл), а
`test_register_version_makes_the_id_resolvable` проходит по старому пути.
Падение именно третьего теста и есть доказательство дефекта.

- [ ] **Step 3: Модель**

В `backend/registry_models.py` добавить:

```python
class PreprocessingVersion(Base):
    """What a preprocessing_id actually means.

    The id is a hash of the functional part of the preprocessing config, and
    it is written into patient_registry for every uploaded study. Without the
    settings behind it the hash is a number that cannot be checked against
    anything — which is exactly what happened: register_version was never
    called in production, so three of the four ids ever used had nothing
    behind them (KI-085).

    Lives in the DB, not in configs/preprocessing_versions.json: that file is
    tracked by git and configs/ is mounted into the container read-write, so
    writing it from the application dirties the working copy.
    """
    __tablename__ = "preprocessing_versions"

    preprocessing_id = Column(String, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    config_json = Column(Text, nullable=False)
```

- [ ] **Step 4: Переписать запись и чтение**

В `backend/preprocessing_version.py` заменить тела `register_version` и
`get_version_config`:

```python
def register_version(config_path: str) -> Optional[str]:
    """Record what this preprocessing_id means, so it can be resolved later.

    Called at run start, where the config is still the one being used. Never
    raises: a run must start even if the bookkeeping fails.
    """
    try:
        preprocessing_id = compute_preprocessing_id(config_path)
    except Exception as exc:  # noqa: BLE001 — старт важнее метки
        logger.warning("Could not compute preprocessing id from %s: %s",
                       config_path, exc)
        return None

    try:
        from database import SessionLocal
        from registry_models import PreprocessingVersion

        with open(config_path, "r", encoding="utf-8") as fh:
            functional = _extract_functional_config(yaml.safe_load(fh))

        db = SessionLocal()
        try:
            if db.get(PreprocessingVersion, preprocessing_id) is None:
                db.add(PreprocessingVersion(
                    preprocessing_id=preprocessing_id,
                    config_json=json.dumps(functional, ensure_ascii=False,
                                           sort_keys=True),
                ))
                db.commit()
                logger.info("Registered preprocessing version %s",
                            preprocessing_id)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not store preprocessing version %s: %s",
                       preprocessing_id, exc)

    return preprocessing_id


def get_version_config(preprocessing_id: str) -> Optional[Dict[str, Any]]:
    """Settings behind an id: the DB first, then the legacy JSON file.

    The JSON is read-only now. It holds exactly one entry (1e2b93ad, created
    02.04.2026) from before this moved into the DB, and that entry must stay
    resolvable.
    """
    if not preprocessing_id:
        return None
    try:
        from database import SessionLocal
        from registry_models import PreprocessingVersion

        db = SessionLocal()
        try:
            row = db.get(PreprocessingVersion, preprocessing_id)
            if row is not None:
                return json.loads(row.config_json)
        finally:
            db.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read preprocessing version %s: %s",
                       preprocessing_id, exc)

    entry = _load_versions().get(preprocessing_id)
    return entry.get("config") if entry else None
```

`_save_versions` больше не вызывается — удалить её вместе с комментарием о
записи, оставив `_load_versions` для чтения унаследованного файла.

- [ ] **Step 5: Прогнать тесты задачи**

Run: `cd backend && python -m pytest test_preprocessing_version_store.py test_preprocessing_version.py -q`
Expected: PASS. Старый `test_preprocessing_version.py` проверяет
`register_version` + `get_version_config` — он должен проходить без правок,
потому что контракт сохранён; если он опирался на файл, поправить его на
проверку через `get_version_config`.

- [ ] **Step 6: Вызвать при старте прогона**

В `backend/app.py`, рядом с вычислением `run_preprocessing_id` (Task 1, Step 6),
заменить вызов на регистрирующий:

```python
    # register_version both computes the id and records the settings behind
    # it, so the label on the data stays resolvable.
    from preprocessing_version import register_version as _register_version
    run_preprocessing_id = (
        _register_version(str(PREPROCESSING_CONFIG))
        if PREPROCESSING_CONFIG.is_file() else None
    )
```

То же в `backend/requeue_service.py` — но с файлом снимка, когда он передан:

```python
    _prep_source = preprocessing_snapshot or PREPROCESSING_CONFIG
    preprocessing_id = (
        register_version(str(_prep_source)) if Path(_prep_source).is_file()
        else None
    )
```

(`PREPROCESSING_CONFIG` импортировать из `pipeline_monitor`, `Path` — из
`pathlib`, `register_version` — из `preprocessing_version`.)

- [ ] **Step 7: Прогнать весь backend**

Run: `cd backend && python -m pytest . -q`
Expected: PASS.

- [ ] **Step 8: Коммит**

```bash
git add backend/preprocessing_version.py backend/registry_models.py \
        backend/app.py backend/requeue_service.py \
        backend/test_preprocessing_version_store.py
git commit -m "fix(kappa): история версий препроцессинга в SQLite, регистрация при старте (KI-085)"
```

---

## Task 4: Ключ реестра — пара «исследование + датасет»

**Files:**
- Modify: `backend/registry_models.py` (`PatientRegistry.study_hash`, `__table_args__`)
- Modify: `backend/patient_registry.py` (`ensure_tables`, `register_patient`, `find_by_study_hash`)
- Modify: `backend/test_patient_registry.py` (вызовы `find_by_study_hash`)
- Test: `backend/test_registry_dataset_key.py`

**Interfaces:**
- Produces: уникальный индекс `ix_patient_registry_study_dataset` по
  `(study_hash, COALESCE(kappa_dataset_id, 0))`; старый
  `ix_patient_registry_study_hash` снимается.
- Produces: `find_by_study_hash(study_hash, dataset_id=None)` — запись для
  датасета, либо список всех записей исследования при `dataset_id=None`.
- Поведение `register_patient`: точная пара → строка с `NULL` → новая строка.

- [ ] **Step 1: Написать падающий тест**

Создать `backend/test_registry_dataset_key.py`:

```python
"""Одно исследование может лежать в нескольких датасетах Kappa.

KI-084: реестр ключевался одним study_hash, и выгрузка в другой датасет
перезаписывала kappa_dataset_id/kappa_entity_id, стирая адрес сущности в
прежнем датасете. Наблюдалось на догрузке test500 в датасет 375: строки
перетекли 347 205->167, 348 114->79, 375 0->74 при неизменном общем числе.
"""
from database import SessionLocal
from patient_registry import (
    ensure_tables,
    find_by_bids_id,
    find_by_study_hash,
    register_patient,
)
from registry_models import PatientRegistry


def setup_function():
    ensure_tables()
    db = SessionLocal()
    try:
        db.query(PatientRegistry).filter(
            PatientRegistry.study_hash.like("dskey_%")
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


def _register(dataset_id, entity_id, study_hash="dskey_one", bids="sub-007"):
    return register_patient(
        bids_id=f"{bids}_ses-001",
        study_hash=study_hash,
        original_patient_id="P-1",
        kappa_entity_id=entity_id,
        kappa_dataset_id=dataset_id,
        pipeline_run_id="run-x",
        lesion_type="glioblastoma",
    )


def test_same_study_in_two_datasets_keeps_both_addresses():
    _register(347, "entity-347")
    _register(375, "entity-375")

    rows = find_by_study_hash("dskey_one")
    assert len(rows) == 2
    assert {r["kappa_dataset_id"]: r["kappa_entity_id"] for r in rows} == {
        347: "entity-347", 375: "entity-375",
    }


def test_reupload_into_the_same_dataset_updates_one_row():
    _register(347, "entity-a")
    _register(347, "entity-b")

    rows = find_by_study_hash("dskey_one")
    assert len(rows) == 1
    assert rows[0]["kappa_entity_id"] == "entity-b"


def test_undelivered_row_is_claimed_by_the_upload():
    """Передача session_registry -> KappaUploader обязана уцелеть."""
    _register(None, None)            # как пишет session_registry
    _register(375, "entity-375")     # как дополняет выгрузка

    rows = find_by_study_hash("dskey_one")
    assert len(rows) == 1
    assert rows[0]["kappa_dataset_id"] == 375
    assert rows[0]["kappa_entity_id"] == "entity-375"


def test_two_finishes_without_kappa_do_not_duplicate():
    """UNIQUE в SQLite считает NULL различными — отсюда COALESCE в индексе."""
    _register(None, None)
    _register(None, None)

    assert len(find_by_study_hash("dskey_one")) == 1


def test_rows_carry_their_own_dataset():
    _register(347, "entity-347")
    _register(375, "entity-375")

    for row in find_by_study_hash("dskey_one"):
        assert row["kappa_entity_id"] == f"entity-{row['kappa_dataset_id']}"


def test_mask_replacement_lookup_sees_the_right_dataset():
    """Именно этот путь ломался: app.py:3168 ищет сущность по bids_id+датасету."""
    _register(347, "entity-347")
    _register(375, "entity-375")

    records = find_by_bids_id("sub-007_ses-001", {347})
    entity = next((r["kappa_entity_id"] for r in records
                   if r.get("kappa_entity_id")), None)
    assert entity == "entity-347"
```

- [ ] **Step 2: Прогнать и убедиться, что падает**

Run: `cd backend && python -m pytest test_registry_dataset_key.py -q`
Expected: FAIL — первый тест даёт 1 строку вместо 2 (вторая регистрация
перезаписала первую), `find_by_study_hash` не принимает `dataset_id` и
возвращает одну запись, а не список.

- [ ] **Step 3: Модель и индекс**

В `backend/registry_models.py` снять уникальность с `study_hash`:

```python
    study_hash = Column(String, nullable=False, index=True)
```

и добавить в класс `PatientRegistry`:

```python
    # One row per (study, Kappa dataset), not per study. A study legitimately
    # lives in more than one dataset — the same person in two datasets is two
    # different subject numbers — and the old single-row key made the second
    # upload overwrite the first one's address (KI-084).
    #
    # COALESCE, because SQLite treats NULLs as distinct under UNIQUE: a plain
    # UNIQUE(study_hash, kappa_dataset_id) would allow any number of
    # not-yet-delivered duplicates of one study. 0 is not a Kappa dataset id
    # (they are positive), so it is safe as "not delivered yet".
    __table_args__ = (
        Index(
            "ix_patient_registry_study_dataset",
            "study_hash",
            text("COALESCE(kappa_dataset_id, 0)"),
            unique=True,
        ),
    )
```

Добавить в импорты модуля `from sqlalchemy import Index, text`.

- [ ] **Step 4: Миграция индекса**

В `backend/patient_registry.py` добавить и вызвать из `ensure_tables` **до**
`_migrate_from_json`:

```python
def _migrate_study_hash_index() -> None:
    """Replace the unique index on study_hash with one on (study, dataset).

    Idempotent, and refuses rather than merges: these are clinical records,
    and deciding which of two rows is the right one is not a job for
    automation. A conflict leaves the DB untouched and names the pairs.
    """
    import sqlalchemy
    from database import engine

    with engine.connect() as conn:
        names = {row[0] for row in conn.execute(sqlalchemy.text(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='patient_registry'"
        ))}
        if "ix_patient_registry_study_dataset" in names:
            return

        conflicts = list(conn.execute(sqlalchemy.text(
            "SELECT study_hash, COALESCE(kappa_dataset_id, 0) AS ds, "
            "       COUNT(*) AS n "
            "FROM patient_registry GROUP BY 1, 2 HAVING n > 1"
        )))
        if conflicts:
            lines = ", ".join(
                f"{row[0]}@{row[1]} x{row[2]}" for row in conflicts)
            logger.error(
                "Реестр: нельзя перевести на ключ (исследование, датасет) — "
                "в базе уже есть дубли: %s. База не изменена.", lines)
            return

        if "ix_patient_registry_study_hash" in names:
            conn.execute(sqlalchemy.text(
                "DROP INDEX ix_patient_registry_study_hash"))
        conn.execute(sqlalchemy.text(
            "CREATE UNIQUE INDEX ix_patient_registry_study_dataset "
            "ON patient_registry (study_hash, COALESCE(kappa_dataset_id, 0))"
        ))
        conn.commit()
        logger.info("Реестр: ключ переведён на (исследование, датасет)")
```

- [ ] **Step 5: Правило upsert**

В `backend/patient_registry.py::register_patient` заменить поиск (строки
118-120) на:

```python
        # Exact pair first; then the row session_registry left behind with no
        # dataset, which this upload CLAIMS rather than duplicating. That
        # hand-off is what session_registry's docstring relies on, and it has
        # to survive the key change. Only then a new row.
        existing = db.query(PatientRegistry).filter(
            PatientRegistry.study_hash == study_hash,
            PatientRegistry.kappa_dataset_id == kappa_dataset_id,
        ).first()
        if existing is None and kappa_dataset_id is not None:
            existing = db.query(PatientRegistry).filter(
                PatientRegistry.study_hash == study_hash,
                PatientRegistry.kappa_dataset_id.is_(None),
            ).first()
```

> Внимание: при `kappa_dataset_id is None` первая выборка должна сравнивать с
> `NULL` через `is_(None)`, иначе `== None` в SQLAlchemy даст верный SQL, но
> читается хуже. Явно:
> ```python
> if kappa_dataset_id is None:
>     first = PatientRegistry.kappa_dataset_id.is_(None)
> else:
>     first = PatientRegistry.kappa_dataset_id == kappa_dataset_id
> ```

- [ ] **Step 6: `find_by_study_hash`**

Заменить функцию целиком. Возвращает **всегда список**: функция, чей тип
результата зависит от значения аргумента, — плохая сигнатура, и вводить её
не надо. Потребителей в продакшене нет, фильтрация по датасету остаётся на
вызывающем.

```python
def find_by_study_hash(study_hash: str) -> List[Dict[str, Any]]:
    """Все записи этого исследования, по одной на датасет Kappa.

    A study legitimately lives in more than one dataset (KI-084), so this can
    no longer return a single row. Always a list — possibly empty, possibly
    with one entry whose kappa_dataset_id is None (produced locally, not yet
    delivered).
    """
    db = SessionLocal()
    try:
        records = db.query(PatientRegistry).filter(
            PatientRegistry.study_hash == study_hash
        ).all()
        return [_to_dict(r) for r in records]
    finally:
        db.close()
```

Поправить вызов в `backend/test_patient_registry.py:60`: там ожидалась одна
запись, станет `found = find_by_study_hash("test_hash_abc123")[0]`.

- [ ] **Step 7: Прогнать тесты задачи**

Run: `cd backend && python -m pytest test_registry_dataset_key.py test_patient_registry.py test_registry_reconcile.py -q`
Expected: PASS.

- [ ] **Step 8: Пересобрать `schema.sql`**

Файл поддерживается руками как дамп `.schema`. Пересобрать из чистой БД:

```bash
REPO=$(git rev-parse --show-toplevel)
rm -f /tmp/schema_tmp.db
DATABASE_URL="sqlite:////tmp/schema_tmp.db" python -c "
import sys; sys.path.insert(0, '$REPO/backend')
from database import init_db
from patient_registry import ensure_tables
init_db(); ensure_tables()
"
sqlite3 /tmp/schema_tmp.db .schema > "$REPO/backend/schema.sql"
```

Сверить `git diff backend/schema.sql`: ожидаются ровно три изменения —
`pipeline_runs.preprocessing_id`, таблица `preprocessing_versions`, замена
`ix_patient_registry_study_hash` на `ix_patient_registry_study_dataset`.
Ничего больше.

- [ ] **Step 9: Прогнать все наборы**

```bash
cd backend && python -m pytest . -q
cd .. && python -m pytest test_*.py -q
python -m pytest scripts/ -q && python -m pytest utils/ -q
```
Expected: всё зелёное.

- [ ] **Step 10: Коммит**

```bash
git add backend/registry_models.py backend/patient_registry.py \
        backend/test_patient_registry.py backend/schema.sql \
        backend/test_registry_dataset_key.py
git commit -m "fix(registry): ключ (исследование, датасет) вместо одной строки на исследование (KI-084)"
```

---

## Task 5: Разовое восстановление строк, перенаправленных догрузкой

**Files:**
- Create: `scripts/restore_registry_from_dump.py`
- Test: `backend/test_restore_registry_from_dump.py`

**Почему тест в `backend/`, хотя скрипт в `scripts/`.** Тесту нужна настроенная
БД, а это делает `backend/conftest.py` при импорте. Отдельный
`scripts/conftest.py` пришлось бы снабдить `sys.path.insert` на `backend/`, и
тогда `import config` в тестах `scripts/` начал бы разрешаться в
`backend/config.py` вместо `utils/config_loader` — риск сломать 97 уже зелёных
тестов ради одного нового. Скрипт загружается по пути через `importlib`, без
правок `sys.path`.

**Interfaces:**
- Consumes: правило upsert из Task 4 и дамп
  `backend/data/registry-before-375-*.json`, снятый перед догрузкой.

Догрузка `test500` в датасет 375 перенаправила на него строки 319
исследований, указывавших на 347 и 348. Их прежние значения есть в дампе.
После Task 4 строки можно вернуть, не ломая новые: на каждую запись дампа
создаётся строка `(study_hash, прежний dataset_id)`, а строка с 375 остаётся.

Отдельным скриптом, не частью миграции: зависит от конкретного дампа и должен
запускаться осознанно.

- [ ] **Step 1: Написать падающий тест**

Создать `backend/test_restore_registry_from_dump.py`:

```python
"""Восстановление строк реестра из дампа, снятого перед догрузкой."""
import importlib.util
import json
from pathlib import Path

import pytest

from database import SessionLocal
from patient_registry import ensure_tables, register_patient
from registry_models import PatientRegistry

_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "restore_registry_from_dump.py")


def _load_script():
    """Import the one-off script by path, without touching sys.path."""
    spec = importlib.util.spec_from_file_location(
        "restore_registry_from_dump", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


restore = _load_script().restore


def _clean():
    db = SessionLocal()
    try:
        db.query(PatientRegistry).filter(
            PatientRegistry.study_hash.like("restore_%")
        ).delete(synchronize_session=False)
        db.commit()
    finally:
        db.close()


@pytest.fixture
def registry_with_row_on_375():
    """One study already delivered to dataset 375, as the backfill left it."""
    ensure_tables()
    _clean()
    register_patient(
        bids_id="sub-009_ses-001",
        study_hash="restore_one",
        original_patient_id="P-9",
        kappa_entity_id="entity-375",
        kappa_dataset_id=375,
        pipeline_run_id="run-backfill",
        lesion_type="glioblastoma",
    )
    yield
    _clean()


def test_dry_run_changes_nothing(tmp_path, registry_with_row_on_375):
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps([{
        "study_hash": "restore_one", "bids_id": "sub-009_ses-001",
        "original_patient_id": "P-9", "kappa_entity_id": "entity-347",
        "kappa_dataset_id": 347,
    }]), encoding="utf-8")

    result = restore(dump, apply=False)
    assert result["would_restore"] == 1
    assert result["restored"] == 0


def test_apply_recreates_the_old_row_and_keeps_the_new_one(
    tmp_path, registry_with_row_on_375
):
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps([{
        "study_hash": "restore_one", "bids_id": "sub-009_ses-001",
        "original_patient_id": "P-9", "kappa_entity_id": "entity-347",
        "kappa_dataset_id": 347,
    }]), encoding="utf-8")

    restore(dump, apply=True)

    from patient_registry import find_by_study_hash
    rows = {r["kappa_dataset_id"]: r["kappa_entity_id"]
            for r in find_by_study_hash("restore_one")}
    assert rows == {347: "entity-347", 375: "entity-375"}


def test_running_twice_is_idempotent(tmp_path, registry_with_row_on_375):
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps([{
        "study_hash": "restore_one", "bids_id": "sub-009_ses-001",
        "original_patient_id": "P-9", "kappa_entity_id": "entity-347",
        "kappa_dataset_id": 347,
    }]), encoding="utf-8")

    restore(dump, apply=True)
    restore(dump, apply=True)

    from patient_registry import find_by_study_hash
    assert len(find_by_study_hash("restore_one")) == 2
```


- [ ] **Step 2: Прогнать и убедиться, что падает**

Run: `cd backend && python -m pytest test_restore_registry_from_dump.py -q`
Expected: FAIL — `FileNotFoundError` на пути к
`scripts/restore_registry_from_dump.py`: файла ещё нет, и `_load_script`
падает на импорте модуля теста.

- [ ] **Step 3: Скрипт**

Создать `scripts/restore_registry_from_dump.py`:

```python
#!/usr/bin/env python3
"""Restore registry rows a backfill repointed at another dataset.

The 19.11.2026 backfill of run test500 into Kappa dataset 375 repointed the
rows of 319 studies that until then addressed datasets 347 and 348: the
registry kept one row per study_hash and register_patient overwrote its
kappa_dataset_id / kappa_entity_id (KI-084). Their old values were dumped
before the write.

Now that a study can hold one row per dataset, the old rows can come back:
the row for 375 occupies a different pair, so re-registering the dumped
values creates a second row rather than overwriting anything.

One-off, not part of the schema migration: it depends on one particular dump
file and must be run deliberately. Dry run by default.

Usage:
    python scripts/restore_registry_from_dump.py backend/data/registry-before-375-*.json
    python scripts/restore_registry_from_dump.py <dump> --apply
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))


def restore(dump_path: Union[str, Path], apply: bool = False) -> dict:
    """Re-create the dumped rows. Returns a report; never partially commits."""
    from patient_registry import find_by_study_hash, register_patient

    rows = json.loads(Path(dump_path).read_text(encoding="utf-8"))
    report = {"in_dump": len(rows), "would_restore": 0, "restored": 0,
              "skipped": 0, "already_there": 0}

    for row in rows:
        study_hash = row.get("study_hash")
        dataset_id = row.get("kappa_dataset_id")
        if not study_hash or not dataset_id:
            # Nothing to address the row by — restoring it would guess.
            report["skipped"] += 1
            continue

        present = {r.get("kappa_dataset_id")
                   for r in find_by_study_hash(study_hash)}
        if dataset_id in present:
            report["already_there"] += 1
            continue

        report["would_restore"] += 1
        if not apply:
            continue

        register_patient(
            bids_id=row.get("bids_id") or "",
            study_hash=study_hash,
            original_patient_id=row.get("original_patient_id") or "",
            patient_name=row.get("patient_name") or "",
            scan_date=row.get("scan_date") or "",
            study_instance_uid=row.get("study_instance_uid") or "",
            kappa_entity_id=row.get("kappa_entity_id"),
            kappa_dataset_id=dataset_id,
            pipeline_run_id=row.get("pipeline_run_id") or "",
            lesion_type=row.get("lesion_type") or "",
            preprocessing_id=row.get("preprocessing_id") or "",
        )
        report["restored"] += 1

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", help="registry-before-*.json")
    parser.add_argument("--apply", action="store_true",
                        help="Without it the script only reports")
    args = parser.parse_args()

    report = restore(args.dump, apply=args.apply)
    print("сухой прогон:" if not args.apply else "готово:")
    for key, value in report.items():
        print("  {:16s} {}".format(key, value))


if __name__ == "__main__":
    main()
```

Проверить, что `register_patient` принимает `preprocessing_id` — он есть в
сигнатуре (`patient_registry.py:110`). Если в дампе этого поля нет, пустая
строка оставит колонку пустой, что верно: метку происхождения мы не
придумываем.

- [ ] **Step 4: Прогнать тесты задачи**

Run: `cd backend && python -m pytest test_restore_registry_from_dump.py -q`
Expected: PASS (3 теста).

- [ ] **Step 5: Коммит**

```bash
git add scripts/restore_registry_from_dump.py \
        backend/test_restore_registry_from_dump.py
git commit -m "chore(registry): разовое восстановление строк из дампа догрузки (KI-084)"
```

---

## Task 6: Закрыть KI и обновить документацию

**Files:**
- Modify: `KNOWN_ISSUES.md` (KI-084, KI-085 → ✅ ЗАКРЫТО с описанием сделанного)
- Modify: `CLAUDE.md` (раздел Data stores: новая таблица; Gotchas — где живёт история версий)

- [ ] **Step 1: Пометить KI закрытыми**

В каждой записи добавить `✅ ЗАКРЫТО` в заголовок и абзац «Закрыто
19.11.2026» с тем, что именно сделано, и ссылкой на тесты, которые держат
инвариант.

- [ ] **Step 2: Обновить CLAUDE.md**

В Data stores к списку таблиц `brain_lesion.db` добавить
`preprocessing_versions`. В Gotchas — короткий пункт: `preprocessing_id`
закрепляется на прогоне при старте, история версий в БД, а
`configs/preprocessing_versions.json` только читается как унаследованный.

- [ ] **Step 3: Прогнать все наборы и закоммитить**

```bash
cd backend && python -m pytest . -q && cd ..
python -m pytest test_*.py -q
git add KNOWN_ISSUES.md CLAUDE.md
git commit -m "docs: закрыть KI-084 и KI-085"
```

---

## Проверка плана относительно спеки

Прошёл по разделам спеки и сверил покрытие:

| Раздел спеки | Задача |
|---|---|
| §4.1 колонка и миграция | Task 1, Step 3 |
| §4.2 запись при старте | Task 1, Steps 4, 6, 7 |
| §4.3 чтение при выгрузке | Task 2 |
| §4.4 регистрация версии | Task 3, Step 6 |
| §4.1 история версий в SQLite | Task 3, Steps 3–4 |
| §5.1 уникальность с COALESCE | Task 4, Steps 3–4 |
| §5.2 правило upsert | Task 4, Step 5 |
| §5.3 `find_by_study_hash` | Task 4, Step 6 |
| §6 миграция с отказом при конфликте | Task 4, Step 4 |
| §6.1 восстановление 319 строк | Task 5 |
| §7 тесты | по тесту в каждой задаче |

**Отличие от спеки, решённое при планировании.** Спека предполагала, что
`numbering.scope_for_run` начнёт возвращать четвёрку
`(scope, dataset_id, preprocessing_id, note)`. От этого отказался: у функции
один продакшен-вызов, но около двенадцати распаковок в трёх тестовых файлах
(`test_numbering_scope.py`, `test_kappa_offline_mode.py`,
`test_app_start_guard.py`), и менять их все — чистая ломка без выигрыша.
Вместо этого `numbering.current_preprocessing_id()` даёт **одно** определение
того, из какого файла берётся идентификатор, и его зовут оба места, которые
создают прогоны. Сигнатура `scope_for_run` не меняется, тесты не трогаются.

**Порядок задач.** Task 4 независима от Tasks 1–3 и может идти первой, если
по реестру хочется результат раньше. Task 5 требует Task 4. Task 6 — последняя.
