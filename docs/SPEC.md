# Техническое задание: рефакторинг под мульти-агентную систему

**Проект:** Web-сервис AI-сегментации поражений головного мозга по МРТ
**Версия ТЗ:** 0.3 (после завершения Этапа 2)
**Дата:** 18 мая 2026
**Ветка:** `feat/mas` (от `main`)
**Автор:** Kate Roppert (НГУ, AI Research Center)

---

## 1. Контекст и мотивация

Текущая версия сервиса поддерживает одну сегментационную модель — глиобластома (nnUNet v1, `Task115_AllData5foldsMeta`). Модель и сервер сегментации жёстко связаны через хардкодные параметры в `simple_server.py` и `nnUNet_inference.py`.

В ближайшее время в систему добавляются:

1. **Модель сегментации рассеянного склероза** (nnUNet v2 + `nnUNetTrainerCATMIL`, обучена на MSLesSeg, `Dataset333_MSLesSegnnUNetTrainerCATMIL`)
2. **Модель сегментации метастазов** (планируется, nnUNet v2)
3. **Модель определения MGMT-статуса по МРТ** (планируется, классификатор, не сегментация)

Параллельно проект эволюционирует в сторону полноценной мульти-агентной системы (МАС) с координатором, ресурсным менеджером, диагностическим LLM-агентом и переговорными протоколами (BDI, Contract Net).

Существующая монолитная архитектура segmentation-контейнера не масштабируется на эти задачи. Цель данного ТЗ — провести рефакторинг, который:

- разнесёт каждую модель в отдельный изолированный сервис;
- введёт единый контракт взаимодействия (HTTP API + манифест возможностей);
- сохранит существующий функционал глиобластомы без потери качества;
- подготовит инфраструктурный задел для будущих агентов МАС;
- закроет несколько накопившихся багов и технических долгов.

---

## 2. Целевая архитектура

### 2.1 Концептуальное разделение: сервисы и агенты

В терминологии проекта вводится разделение:

- **Сервис (service)** — HTTP-эндпоинт, выполняющий конкретную задачу инференса (сегментация, классификация). Не обладает автономным целеполаганием. Реагирует на запросы.
- **Агент (agent)** — автономная сущность с целями, моделью среды и способностью принимать решения. Может вызывать сервисы. В текущем ТЗ агенты не реализуются — резервируется директория `agents/` и контракт совместимости.

Эта граница позволит в будущем добавить координирующих агентов (LLM-координатор, ресурсный менеджер, диагностические агенты) без архитектурного конфликта с сервисами инференса.

### 2.2 Структура проекта после рефакторинга

```
mri_ai_service/
├── backend/                       # Web orchestrator + pipeline (существующее)
│   ├── orchestrator.py
│   ├── pipeline.py
│   ├── app.py
│   ├── schema.sql                 # NEW: схема БД (вместо .db в git)
│   └── ...
├── frontend/                      # React (если отдельно от backend/)
├── services/                      # NEW: сервисы инференса
│   ├── common/
│   │   ├── service_base.py        # базовый класс сервиса
│   │   ├── contracts.py           # dataclasses для request/response
│   │   ├── gpu_monitor.py         # вынесенный GPUMonitor из simple_server.py
│   │   └── requirements.txt
│   ├── gbm-seg/                   # бывший segmentation/
│   │   ├── Dockerfile
│   │   ├── service_server.py      # бывший simple_server.py
│   │   ├── inference.py           # бывший nnUNet_inference.py (с фиксами)
│   │   ├── manifest.yaml
│   │   └── requirements.txt
│   └── ms-seg/                    # NEW
│       ├── Dockerfile
│       ├── service_server.py
│       ├── inference.py
│       ├── manifest.yaml
│       └── requirements.txt
├── agents/                        # NEW (резерв): будущие МАС-агенты
│   └── .gitkeep
├── configs/
│   ├── services.yaml              # NEW: реестр сервисов
│   ├── kappa_datasets.yaml        # существующий, расширяется
│   └── ...
├── demo_workspace/                # существующая структура, без изменений
│   └── input/{case_id}/
│       ├── bids_organized/
│       ├── nifti/
│       ├── preprocessed/
│       ├── quality_reports/
│       ├── reports/
│       ├── segmentation/
│       │   ├── glioblastoma/      # NEW: подпапки по типу поражения
│       │   │   └── mask.nii.gz
│       │   └── multiple_sclerosis/
│       │       └── mask.nii.gz
│       └── transformations/
├── scripts/
├── docker-compose.yml             # модифицируется
├── web.Dockerfile
└── docs/
    └── SPEC.md                    # этот документ
```

### 2.3 Развёртывание в Docker

Каждый сервис — отдельный контейнер. Связь по HTTP внутри Docker-сети. `demo_workspace/` монтируется как общий volume:

- в `web` — read-write
- в сервисы — read-write (нужен write для записи маски в `segmentation/{lesion_type}/`)

```yaml
services:
  web:
    volumes: [./demo_workspace:/workspace:rw]
  service-gbm-seg:
    ports: ["5000:5000"]
    volumes: [./demo_workspace:/workspace:rw]
  service-ms-seg:
    ports: ["5001:5001"]
    volumes: [./demo_workspace:/workspace:rw]
```

### 2.4 Изоляция зависимостей

Причина отдельных контейнеров для GBM и MS — несовместимые требования:

| Параметр | gbm-seg (nnUNet v1) | ms-seg (nnUNet v2 + CATMIL) |
|---|---|---|
| numpy | 2.3.5 | 1.26.4 (жёстко) |
| nnUNet | fork v1 (Павловского) | fork v2 (Luu, `luumsk/SmallLesionMRI`) |
| trainer | `nnUNetTrainerV2` | `nnUNetTrainerCATMIL` |
| plans | `nnUNetPlansv2.1` | `nnUNetPlans` |
| checkpoint ext | `.model` | `.pth` |
| monai | не требуется | 1.3.0 (зависимость форка) |

Объединение в один контейнер потребует ручного разруливания версий и хрупкое.

---

## 3. Единый контракт сервиса

Все сервисы (gbm-seg, ms-seg, и будущие mets-seg, mgmt-classify) реализуют один и тот же HTTP-контракт.

### 3.1 Эндпоинты

#### `GET /health`

Liveness и готовность модели.

```json
{
  "service_id": "gbm-seg",
  "status": "ready" | "loading",
  "queue_size": 0,
  "available_gpus": [0]
}
```

#### `GET /manifest`

Статический манифест возможностей сервиса.

```yaml
service:
  id: gbm-seg
  type: segmentation                # | classification
  version: "0.2.0"
  description: "Glioblastoma segmentation, nnUNet v1, Task115"

capabilities:
  lesion_types: [glioblastoma]
  modalities_required: [T1, T1c, T2, FLAIR]
  modality_channels:                # порядок каналов модели
    T1: 0
    T1c: 1
    T2: 2
    FLAIR: 3
  output:
    kind: multi_class_mask
    classes:
      0: background
      1: ed                         # edema
      2: net                        # non-enhancing tumor
      3: et                         # enhancing tumor
      4: ncr                        # necrotic core

resources:
  gpu_memory_gb: 4
  estimated_runtime_sec: 90
  needs_gpu: true

model:
  framework: nnunetv1
  task_name: "Task115_AllData5foldsMeta"
  trainer: "nnUNetTrainerV2"
  plans: "nnUNetPlansv2.1"
  configuration: "3d_fullres"
  folds: [0, 1, 2, 3, 4]
  checkpoint: "model_final_checkpoint"
```

#### `POST /predict`

Асинхронный запуск инференса. Возвращает `job_id` сразу, не дожидаясь окончания.

**Request:**

```json
{
  "case_id": "13_03_1850",
  "input_dir": "/workspace/input/13_03_1850/preprocessed",
  "output_dir": "/workspace/input/13_03_1850/segmentation/glioblastoma",
  "lesion_type": "glioblastoma",
  "options": {
    "use_tta": false,
    "folds": [0, 1, 2, 3, 4]
  }
}
```

Контракт по файлам в `input_dir`: orchestrator готовит файлы по конвенции nnUNet (`{case_id}_0000.nii.gz`, `{case_id}_0001.nii.gz`, ...), соответствующей `modality_channels` из манифеста.

**Response (202 Accepted):**

```json
{
  "job_id": "uuid",
  "status": "queued"
}
```

#### `GET /predict/{job_id}/status`

```json
{
  "job_id": "uuid",
  "status": "queued" | "running" | "succeeded" | "failed",
  "progress": 0,
  "message": "Running inference on GPU 0",
  "gpu_id": 0,
  "submitted_at": "2026-05-14T10:00:00Z",
  "started_at": "2026-05-14T10:00:02Z",
  "finished_at": null,
  "error": null
}
```

#### `GET /predict/{job_id}/result`

```json
{
  "job_id": "uuid",
  "mask_path": "/workspace/input/13_03_1850/segmentation/glioblastoma/mask.nii.gz",
  "output_classes": {"0": "background", "1": "ed", "2": "net", "3": "et", "4": "ncr"},
  "lesion_type": "glioblastoma",
  "model": {
    "framework": "nnunetv1",
    "task": "Task115_AllData5foldsMeta",
    "folds_used": [0, 1, 2, 3, 4],
    "tta": false
  },
  "metrics": {
    "inference_time_sec": 87.3,
    "gpu_utilization_avg": 78.5,
    "gpu_memory_max_mb": 3421
  }
}
```

Если `status != succeeded` — возвращается 409 Conflict с текущим статусом.

### 3.2 Принципы контракта

- **Передача данных только по путям.** Никаких multipart upload. Сервис читает из `input_dir`, пишет в `output_dir`. Это требует общего volume — реализуется через монтирование `demo_workspace/` во все контейнеры.
- **Сервис не знает структуру workspace.** Orchestrator формирует `input_dir` и `output_dir` и передаёт готовые пути.
- **Каждый сервис возвращает свою маппу классов.** Frontend перестаёт хардкодить классы — рендерит то, что пришло в `output_classes`.
- **`lesion_type` дублируется в payload и манифесте.** В payload — для логирования и сверки; манифест — источник истины о возможностях сервиса.

### 3.3 Что не входит в контракт сейчас

- Аутентификация / авторизация (внутренняя Docker-сеть, доверенные клиенты).
- Версионирование API (`/v1/`, `/v2/`). Резервируется на будущее.
- Прогресс инференса в процентах (только переключение состояний queued → running → succeeded). Тонкая гранулярность — задача nnUNet, не сервиса.

---

## 4. Реестр сервисов

Файл: `configs/services.yaml`

```yaml
services:
  - id: gbm-seg
    url: http://service-gbm-seg:5000
    enabled: true
    handles:
      lesion_types: [glioblastoma]

  - id: ms-seg
    url: http://service-ms-seg:5001
    enabled: true
    handles:
      lesion_types: [multiple_sclerosis]

  # Закомментировано — резерв на будущее:
  # - id: mets-seg
  #   url: http://service-mets-seg:5002
  #   enabled: false
  #   handles:
  #     lesion_types: [brain_metastasis]
  # - id: mgmt-classify
  #   url: http://service-mgmt-classify:5003
  #   enabled: false
  #   handles:
  #     classification_tasks: [mgmt_methylation_status]
```

Orchestrator при старте:

1. Читает `services.yaml`.
2. Для каждого enabled сервиса вызывает `GET /manifest` — проверка доступности и согласованности.
3. Строит маршрутизационную таблицу: `lesion_type → service_url`.
4. При запуске stage 05 segmentation выбирает сервис по `lesion_type` из запроса пользователя.

---

## 5. План работ по этапам

Все работы выполняются в ветке `feat/mas`. Каждый этап завершается логически связной серией коммитов и проверкой критериев приёмки.

### Этап 0 — Подготовка инфраструктуры

**Цель:** привести репозиторий в порядок до начала рефакторинга кода.

**Задачи:**

- Извлечь схему БД: `sqlite3 backend/brain_lesion.db .schema > backend/schema.sql`
- Расширить `.gitignore` (БД, weights, workspace, секреты)
- Удалить `backend/brain_lesion.db` из отслеживания git (`git rm --cached`)
- Создать `agents/.gitkeep` как резерв
- Создать `docs/SPEC.md` (этот документ)

**Коммиты:**

1. `chore: extend gitignore with db, runtime data, weights`
2. `chore(db): stop tracking local sqlite database`
3. `docs(db): add sqlite schema as source of truth`
4. `docs: add MAS refactoring specification`
5. `chore: reserve agents/ directory for future MAS agents`

**Критерии приёмки:**

- `git status` чист после коммитов; БД не отслеживается, но физически на месте.
- `backend/schema.sql` содержит все CREATE-statements текущей БД.
- `docs/SPEC.md` присутствует в репо.

---

### Этап 1 — Перенос gbm-seg в новую структуру ✅ ЗАВЕРШЁН

**Цель:** переместить существующий segmentation-сервис в `services/gbm-seg/` и адаптировать под единый базовый класс. Функционал глио должен работать идентично до и после этапа.

**Фактически выполненные сабшаги:**

**Сабшаг 1.1+1.2** (объединены) — Перенос и переименование файлов
- `mv segmentation/ → services/gbm-seg/` (как обычный mv; git трекнул как delete+add)
- `simple_server.py` → `service_server.py`
- `nnUNet_inference.py` → `inference.py`
- Обновлены `docker-compose.yml`, `Dockerfile`, `configs/segmentation_config.yaml`
- Коммит: `refactor(services/gbm-seg): migrate from segmentation/ to new structure`

**Сабшаг 1.3** — Вынос GPUMonitor в services/common/
- Создан `services/common/` как Python-пакет
- `GPUMonitor` вынесен в `services/common/gpu_monitor.py` с улучшениями: type hints, logging, graceful pynvml fallback
- Build context Docker расширен с `./services/gbm-seg` до `./services` (Вариант A: расширенный контекст)
- Веса переведены с `COPY` внутрь образа на `volume mount` (read-only) — критическое архитектурное решение, упрощает все будущие сервисы
- Добавлен `services/.dockerignore`
- Добавлен `ENV PYTHONPATH=/app` для обнаружения common/
- Коммит: `refactor(services/common): extract GPUMonitor into reusable module`

**Сабшаг 1.4** — ServiceBase и contracts
- Создан `services/common/contracts.py` — dataclasses `Job`, `JobStatus`
- Создан `services/common/service_base.py` — абстрактный класс с HTTP-сервером (Quart), очередью задач, GPU pool, GPUMonitor lifecycle
- Контракт согласован с SPEC.md §3
- Коммит: `feat(services/common): add abstract ServiceBase and Job contracts`

**Сабшаг 1.5** — Перепись gbm-seg на ServiceBase (включает manifest)
- `GbmSegService(ServiceBase)` реализует `load_model()` и `run_inference()`
- Создан `services/gbm-seg/manifest.yaml` (изначально планировался отдельным сабшагом 1.6)
- Удалено legacy: HTML-форма `/`, sync `/v1/inference`, `/test_task`, `/test_async_simple`, `MODELS` dict, глобальный `is_processing`
- Сохранено как тонкие обёртки для совместимости с текущим web: `/v1/inference_async`, `/v1/info`, `/v1/models`, `/uploads/<file>`, `/get_status` (Опция Y из обсуждения)
- Объём кода: ~900 строк → ~320 строк
- Коммит: `refactor(services/gbm-seg): rewrite server on top of common ServiceBase`

**Что НЕ было сделано (отложено):**
- Исправление бага порядка аргументов в `predict_for_api` — отмечено в коде, не критично для функционирования, фикс перенесён на удобный момент
- Параметризация `predict_for_api` через config — функция остаётся как есть, новый ServiceBase оборачивает её

**Критерии приёмки — выполнены:**

- ✅ `docker compose up service-gbm-seg` поднимает контейнер без ошибок
- ✅ `curl /health` возвращает `{"status": "ready"}`
- ✅ `curl /manifest` возвращает корректный JSON манифест
- ✅ End-to-end pipeline с тестовым кейсом глиобластомы — маска идентична доэтапной
- ✅ Web-контейнер работает без изменений благодаря legacy wrappers

---

### Этап 2 — Контракт по путям + обновление orchestrator ✅ ЗАВЕРШЁН

**Цель:** перейти с multipart upload на передачу файловых путей через shared volume. Обновить pipeline в orchestrator. Удалить legacy wrappers из gbm-seg после миграции.

**Контекст:** в Этапе 1 в gbm-seg оставлены legacy-эндпоинты (`/v1/inference_async`, `/v1/info`, `/v1/models`, `/uploads/<file>`, `/get_status`) как обёртки над новым контрактом. Этап 2 завершает миграцию: переводит pipeline на новый контракт и удаляет обёртки.

**Фактически выполненные сабшаги:**

**Сабшаг 2.1** — Новый клиент в скрипте сегментации
- В `scripts/06_segmentation.py` добавлен метод `AsyncSegmentationClient.segment_by_path_async` — реализует контракт SPEC.md §3 (`POST /predict` с JSON, polling `/predict/{id}/status`, чтение `/predict/{id}/result`)
- Добавлен helper `_wait_for_completion_v2`
- `check_server_availability` расширен — пробует `/health` и `/v1/models` (fallback)
- Существующий старый клиент сохранён нетронутым (миграция через сосуществование)
- Коммит: `feat(scripts/06_segmentation): add path-based async client method`

**Сабшаг 2.3** — Переключение pipeline на новый контракт
- `BIDSScanner._create_session`: output mask path теперь в подпапке `{lesion_type}/`, имя файла `*_segmask.nii.gz` сохранено
- `SegmentationRunner._process_sessions_async`: использует `segment_by_path_async` вместо старого метода; маска переименовывается с `mask.nii.gz` на `*_segmask.nii.gz` для совместимости с backend mask_versions, Kappa upload, Slicer integration
- Stage 07 (`07_inverse_transform.py`) и Stage 08 (`08_lobar_localization.py`) обновлены — все производные файлы (4 native masks, lobar_report) также пишутся в `{lesion_type}/`
- В `scripts/preprocessing_steps/registration.py` функция `inverse_transform_subject_masks` получила optional параметр `lesion_type`
- В `service_server.py`: метод `_resolve_modalities` расширен под BIDS-конвенцию (`T1w`, `T2w`, `ce-gd_T1w`, `FLAIR`); исправлен pre-existing баг с T1 vs T1c глобом
- В `docker-compose.yml`: добавлен mount `/home:/home:rw` для service-gbm-seg (симметрия с web — для host-style путей)
- Введена константа `LESION_TYPE = "glioblastoma"` в трёх скриптах (хардкод до Этапа 4)
- Коммит: `refactor(pipeline): switch segmentation flow to /predict contract`

**Сабшаг 2.5** — Удаление legacy
- Из `service_server.py` удалён метод `_register_legacy_routes` (5 эндпоинтов: `/v1/inference_async`, `/v1/info`, `/v1/models`, `/uploads/<file>`, `/get_status`)
- Удалены глобальные `_INPUT_DIR`, `_OUTPUT_DIR` (использовались только в legacy)
- Удалены неиспользуемые импорты: `send_file`, `send_from_directory`, `secure_filename`, `aiofiles`
- В `06_segmentation.py` удалены: класс `SegmentationClient` (sync), методы `segment_async`, `segment_async_with_status`, `_wait_for_completion_with_status`, `_wait_for_completion`, `_download_result`, `SegmentationInput.prepare_for_server`
- `check_server_availability` упрощён до проверки только `/health`
- Объём изменений: **+14 строк, −503 строки** в одном коммите
- Коммит: `refactor: drop legacy multipart segmentation contract`

**Архитектурные решения, принятые по ходу:**

- Имя файла маски сохранено как `*_segmask.nii.gz` — это критично для совместимости с backend (mask_versions, Slicer integration, Kappa upload). Сервис пишет `mask.nii.gz`, pipeline после получения переименовывает.
- Контракт `/predict` использует абсолютные host-style пути (типа `/home/ubuntu/mri_ai_service/demo_workspace/...`). Это работает потому что и web, и gbm-seg монтируют `/home`. Это **известный технический долг** — в распределённой архитектуре потребуется переход на контейнерные пути или URI-адресацию (см. §6).
- HTTP file transfer mode (multipart upload + download) явно вынесен в §6 как not-in-scope — потребуется для cube/barguzin профилей в будущем.
- Сабшаг 2.2 и 2.4 объединены с 2.3 в один коммит, так как разделение усложняло бы тестирование.

**Критерии приёмки — выполнены:**

- ✅ End-to-end на 3+ тестовых кейсах глио даёт идентичные маски
- ✅ В логах service-gbm-seg виден `/predict` (не `/v1/inference_async`)
- ✅ Маска в `demo_workspace/input/{case}/segmentation/sub-XXX/ses-YYY/anat/glioblastoma/sub-XXX_ses-YYY_T1w_segmask.nii.gz`
- ✅ Native-маски и lobar_report также в подпапке `glioblastoma/`
- ✅ Backend (Slicer integration, Kappa upload, mask versioning) работает без изменений

---

### Этап 3 — Добавление ms-seg сервиса

**Цель:** новый сервис РС с моделью CATMIL, веса с `bigdata.nsu.ru`. Полное прохождение end-to-end на одном тестовом кейсе РС из SibBMS.

**Задачи:**

- Скопировать веса CATMIL с `bigdata.nsu.ru:8833`:
  - Папка: `/media/storage/luu/nnUNet_results/Dataset333_MSLesSegnnUNetTrainerCATMIL__nnUNetPlans__3d_fullres`
  - Целевая локация: `services/ms-seg/weights/` (не коммитим в git — добавляем в `.gitignore`)
- Создать `services/ms-seg/Dockerfile`:
  - base: `nvcr.io/nvidia/pytorch:24.10-py3` или подобный
  - `git clone https://github.com/luumsk/SmallLesionMRI.git /opt/SmallLesionMRI`
  - `pip install -e /opt/SmallLesionMRI/slsseg` (заменит стандартный nnunetv2 на форк)
  - `pip install -r requirements.txt` для самого сервиса
- Создать `services/ms-seg/service_server.py` на основе общего базового класса:
  - принудительный импорт `nnunetv2.training.nnUNetTrainer.nnUNetTrainerCATMIL` при старте
  - использование `nnUNetPredictor` из v2 (а не `predict_from_folder` из v1)
- Создать `services/ms-seg/manifest.yaml`:
  - modalities: T1, T2, FLAIR (3 канала)
  - output: binary mask (0=background, 1=ms_lesion)
- Создать `services/ms-seg/requirements.txt`.
- Добавить `service-ms-seg` в `docker-compose.yml`.
- Расширить `configs/kappa_datasets.yaml`: добавить `multiple_sclerosis` как тип поражения.
- Расширить `configs/services.yaml`: добавить регистрацию ms-seg.

**Коммиты:**

1. `chore(services/ms-seg): add weights to gitignore`
2. `feat(services/ms-seg): scaffold service on common base`
3. `feat(services/ms-seg): integrate CATMIL trainer via slsseg fork`
4. `feat(services/ms-seg): add inference logic via nnUNetPredictor`
5. `feat(services/ms-seg): add manifest`
6. `build(services/ms-seg): Dockerfile with slsseg fork installation`
7. `build(infra): register service-ms-seg in docker-compose`
8. `feat(configs): register ms-seg in services.yaml`
9. `feat(configs/kappa_datasets): add multiple_sclerosis lesion type`

**Критерии приёмки:**

- `docker compose up service-ms-seg` поднимается без ошибок, модель грузится в GPU.
- `curl http://localhost:5001/manifest` возвращает корректный манифест с 3 модальностями.
- Прогон с одним тестовым кейсом из SibBMS (DICOM → весь pipeline) даёт непустую маску в `segmentation/multiple_sclerosis/mask.nii.gz`.
- Визуальная проверка маски в NiiVue — лезии находятся в FLAIR-гиперинтенсивных областях.

---

### Этап 4 — Динамический выбор сервиса в orchestrator

**Цель:** убрать хардкод URL сегментации в orchestrator. Pipeline динамически выбирает сервис по `lesion_type` из запроса.

**Задачи:**

- В `backend/orchestrator.py` (или соответствующем модуле) — читать `configs/services.yaml` при старте.
- Опрашивать `/manifest` каждого enabled сервиса — построить таблицу `lesion_type → service_url`.
- В stage 05: получать `lesion_type` из запроса пользователя (он уже есть в payload, т.к. фронт его передаёт через выпадающий список).
- Передавать запрос в соответствующий сервис.
- Обработка ошибок: если сервис недоступен или не поддерживает `lesion_type` — внятное сообщение в логи и в API ответ.
- Удалить переменную окружения `SEGMENTATION_URL` из docker-compose web-сервиса (заменена на конфиг).

**Коммиты:**

1. `feat(backend/orchestrator): load services.yaml registry on startup`
2. `feat(backend/orchestrator): build lesion_type routing table from manifests`
3. `refactor(backend/pipeline): dynamic service selection by lesion_type`
4. `refactor(infra): remove hardcoded SEGMENTATION_URL env var`

**Критерии приёмки:**

- Прогон кейса глио → попадает в gbm-seg.
- Прогон кейса РС → попадает в ms-seg.
- В логах orchestrator виден выбранный сервис по lesion_type.
- При выключении ms-seg в `services.yaml` (`enabled: false`) — запрос с lesion_type=multiple_sclerosis возвращает понятную ошибку.

---

### Этап 5 — Адаптация фронтенда

**Цель:** убрать из фронта хардкоды классов глиобластомы. Сделать рендеринг универсальным под любую модель.

**Задачи:**

- **NIfTIViewer:** строить colormap из `output_classes`, пришедших с результатом инференса. Бинарная маска (1 класс) и мульти-класс (4 класса глио) — единая логика.
- **ClinicalReportContent:** генерация отчёта по `output_classes`:
  - для глио — текущая разбивка по NCR/ED/NET/ET с объёмами
  - для РС — общий объём лезий, количество, распределение по долям
  - в коде — табличная маршрутизация: lesion_type → шаблон отчёта
- **Slicer plugin** (если в проекте): имена сегментов и цвета — из `output_classes`, а не хардкод.
- Селектор типа поражения — уже работает через `getLesionTypes()` из `kappa_datasets.yaml`. После добавления `multiple_sclerosis` в конфиг — автоматически появится в UI.

**Коммиты:**

1. `refactor(frontend/NIfTIViewer): build colormap from response output_classes`
2. `refactor(frontend/ClinicalReportContent): route report template by lesion_type`
3. `feat(frontend/ClinicalReportContent): add MS report template (volume, count, lobar distribution)`
4. `refactor(slicer): use output_classes for segment names and colors`

**Критерии приёмки:**

- Глио-кейс отрисовывается так же, как до рефакторинга.
- РС-кейс отрисовывается корректно: бинарная маска одним цветом, в отчёте — объём+количество+доли.
- Переключение типа поражения в UI работает.

---

### Этап 6 — Стабилизация и слияние

**Цель:** end-to-end проверки, документация, merge в main.

**Задачи:**

- Прогнать pytest по всем существующим тестам.
- Прогнать end-to-end на двух кейсах: один глио, один РС из SibBMS.
- Обновить README репозитория: новая структура, описание сервисов и контракта.
- Поставить тэг релиза.
- Merge `feat/mas` → `main`.

**Коммиты:**

1. `test: end-to-end smoke test on glioblastoma case`
2. `test: end-to-end smoke test on multiple sclerosis case`
3. `docs: update README with MAS-ready architecture overview`
4. `docs: document service contract and registry`
5. (merge commit) `Merge branch 'feat/mas' into main`
6. `chore: tag v0.2.0-mas-foundation`

**Критерии приёмки:**

- Pytest зелёный.
- Оба end-to-end теста проходят.
- README отражает текущую структуру.
- `main` обновлён, тэг проставлен.

---

## 6. Что НЕ входит в это ТЗ

Чтобы не разрастаться — следующее явно отнесено к будущим итерациям:

- Реализация координирующего LLM-агента (`agents/coordinator/`).
- Реализация ресурсного менеджера (`agents/resource-manager/`).
- BDI-логика, Contract Net Protocol, переговорные механизмы.
- Сервис сегментации метастазов (`services/mets-seg/`).
- Сервис классификации MGMT-статуса (`services/mgmt-classify/`).
- Распределённое развёртывание сервисов на разных машинах (S3/MinIO для shared storage).
- **HTTP file transfer mode** (`POST /predict` multipart + `GET /files/{id}`) для распределённых развёртываний без shared storage. Текущая архитектура предполагает доступ к shared volume; HTTP-передача файлов будет добавлена как опциональный режим при необходимости работы с удалёнными inference-серверами (cube/barguzin на bigdata.nsu.ru). Профили подключения уже описаны в `configs/segmentation_config.yaml`.
- **Унификация путей host vs container.** Текущий pipeline передаёт абсолютные host-style пути (`/home/ubuntu/mri_ai_service/demo_workspace/...`), которые работают только потому, что web и gbm-seg оба монтируют `/home`. В долгосрочной перспективе следует перейти на контейнерные пути (`/workspace/...`) или абстрактные URI. Это связанная задача с HTTP file transfer mode выше — обе нужны для распределённого развёртывания.
- Аутентификация и версионирование API.
- CI/CD pipeline.

Архитектура текущего ТЗ оставляет открытой возможность каждого из этих расширений без переделок.

---

## 7. Риски и митигации

| Риск | Митигация |
|---|---|
| Изменение поведения gbm-seg при рефакторинге | Этап 1 ведёт к идентичной маске; критерий приёмки — попиксельное сравнение. |
| Несовместимость зависимостей форка `slsseg` | Полная изоляция в отдельном контейнере; pip install -e указывает на форк целиком. |
| Конфликт `nnunetv2` имени пакета между форком и стандартным | Форк регистрируется как `nnunetv2` (по setup.py), стандартный не ставится в ms-seg. |
| Проблемы с CATMIL trainer auto-discovery | Принудительный импорт модуля при старте сервера. |
| Потеря данных при `git mv segmentation/ services/gbm-seg/` | Бэкап ветка `main` до начала работ. |
| Большие веса CATMIL долго копируются с `bigdata.nsu.ru` | Однократная операция, выполняется до Этапа 3. |

---

## 8. Глоссарий

- **Сервис (service)** — HTTP-эндпоинт с моделью инференса, выполняющий конкретную задачу. Не имеет автономного целеполагания.
- **Агент (agent)** — автономная сущность с BDI или подобной архитектурой. В текущем ТЗ не реализуется, резервируется.
- **Манифест (manifest)** — статическое YAML-описание возможностей сервиса (модальности, классы, ресурсы).
- **Реестр сервисов (services registry)** — конфиг-файл со списком доступных сервисов и их URL.
- **Lesion type** — тип поражения мозга: `glioblastoma`, `multiple_sclerosis`, `brain_metastasis` и т.д.
- **CATMIL** — Component-Adaptive Tversky + Multiple Instance Learning, составная loss-функция nnUNet v2, разработанная M.S.K. Luu (НГУ).
- **MSLesSeg** — публичный датасет МРТ-снимков пациентов с РС (Università di Catania).
- **SibBMS** — Сибирский датасет МРТ с РС (НГУ + ИМТЦ СО РАН), доступ авторизован.
- **MS3SEG** — публичный датасет МРТ РС с raw DICOM и ground truth (иранские центры, Nature 2026).

---

*Документ живой. По мере прогресса в реализации детали могут уточняться через PR в `docs/SPEC.md`.*