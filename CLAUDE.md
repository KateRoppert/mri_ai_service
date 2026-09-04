# MRI AI Service

ИИ-сервис диагностики поражений головного мозга по МРТ. Пайплайн DICOM → BIDS → NIfTI → QA → предобработка → сегментация → нативная маска → анатомическая локализация. Два типа поражений: глиобластома и рассеянный склероз. Модели — отдельные микросервисы (задел под МАС). Интеграция с Kappa (курация датасетов) и 3D Slicer (ручная правка масок).

Owner: Kate. Milestone: v1.0 / Этап 6 Production Readiness (`ROADMAP.md`). Баги и долг: `KNOWN_ISSUES.md`. README.md устарел (Flask, старое дерево) — этому файлу и `ROADMAP.md` верить больше.

## Commands

```bash
# Полный стек (web :8000 + gbm-seg :5000 + ms-seg :5001, GPU)
docker compose --profile full up --build

# Пайплайн без веба
source venv/bin/activate
python orchestrator.py --config pipeline_config.yaml

# Slicer-агент (экспертное редактирование масок)
cd slicer && python slicer_agent.py

# Тесты (файлы лежат рядом с кодом, не в tests/ — эта папка в .gitignore)
source venv/bin/activate
python -m pytest backend/ -q
python -m pytest test_*.py -q
cd frontend && npm run lint
```

Локальный frontend-dev: `cd frontend && npm run dev` (Vite проксирует `/api` и `/ws` на `:8000`).

## Architecture

```
orchestrator.py          # CLI-запуск этапов из pipeline_config.yaml
backend/                 # FastAPI: запуски, реестр пациентов, Kappa, маски, WS
frontend/                # React 19 + Ant Design + NiiVue, axios в src/services/api.js
scripts/01..08_*.py      # этапы пайплайна (+ preprocessing_steps/, quality_metrics/)
services/gbm-seg         # nnUNet v1, Quart, порт 5000, lesion_type=glioblastoma
services/ms-seg          # nnUNet v2 + CATMIL, порт 5001, lesion_type=multiple_sclerosis
services/common/         # ServiceBase, контракт /predict (JSON с путями, shared volume)
utils/config_loader.py   # YAML, lesion_types, пути этапов
slicer/                  # агент для ручного редактирования сегментации
agents/                  # пустой резерв под будущих МАС-агентов
docs/superpowers/        # specs/ и plans/ на каждую задачу
```

Поток: UI/CLI → `backend/pipeline_manager.py` пишет runtime-конфиг → `orchestrator.py` гоняет этапы → stage 06 POST на сервис из `configs/services.yaml` → результаты в `root_output_dir` (логи в `{output}/logs`) → загрузка в Kappa.

Docker: 3 образа (`web`, `service-gbm-seg`, `service-ms-seg`). `web` ограничен 20g RAM. Контейнеры монтируют `/home:/home` и `./demo_workspace:/workspace` — хостовые пути работают внутри. `pipeline_config.yaml` смонтирован в web, правки подхватываются без rebuild.

## Pipeline stages

| # | Script | Output dir key | Notes |
|---|---------|----------------|-------|
| 01 | `01_reorganize_folders.py` | `bids_organized` | анонимизация, BIDS, completeness по `lesion_types.yaml` |
| 02 | `metadata_extractor.py` | `metadata` | обычно **выключен** (теги уже сняты) |
| 03 | `03_convert_to_nifti.py` | `nifti` | |
| 04 | `04_assess_quality.py` | `quality_reports` | тяжёлый по RAM; workers подобраны под 20g |
| 05 | `05_preprocessing.py` | `preprocessed` | ANTs/FSL; GBM: reference t1c; MS: t1 |
| 06 | `06_segmentation.py` | `segmentation` | диспетчер по `configs/services.yaml` |
| 07 | `07_inverse_transform.py` | `segmentation` | маска в native space |
| 08 | `08_lobar_localization.py` | `segmentation` | GBM: lobar/volume; MS: McDonald zones |

UI в `backend/config.py` нумерует этапы 1–7, пропуская выключенный metadata — не путать с номерами скриптов 01–08.

Запуск можно остановить из UI и затем возобновить. Остановка мгновенная (убийство process-group). При возобновлении настройки предобработки сравниваются со снимком остановленного запуска.

`lesion_type` прокидывается из UI/CLI в `config['general']['lesion_type']`. Канон модальностей: `configs/lesion_types.yaml`. Не хардкодить `{t1,t1c,t2,t2fl}`.

## Key files

- `pipeline_config.yaml` — мастер-конфиг (пути, workers, enable этапов)
- `configs/lesion_types.yaml` — required modalities + reference + reports
- `configs/services.yaml` — URL сегментационных сервисов (docker DNS)
- `configs/kappa_datasets.yaml` — Kappa dataset id по lesion_type
- `backend/schema.sql` — SQLite-схема (не Postgres/Mongo)
- `backend/data/` — volume: `brain_lesion.db` создаётся при первом запуске
- `docs/SPEC.md` — MAS-рефакторинг; актуальный статус — `ROADMAP.md`

## Data stores

- SQLite `backend/data/brain_lesion.db`: pipeline_runs, patient_registry, mask_versions, Kappa sessions, validations
- Kappa: auth `https://kappa.nsu.ru:8061/user-micro-services/v1`, data `.../data-micro-services/v1` (`backend/kappa_auth.py`, `kappa_client.py`, `kappa_uploader.py`)
- Сессии Kappa в БД, `kappa_session_id` на фронте в localStorage

Секреты: `.env` (не коммитить). Логины Kappa не класть в git/логи.

## Workflow

Ветки: `feat/*`, `fix/*`, `chore/*` от `main`. На задачу: spec в `docs/superpowers/specs/`, план в `docs/superpowers/plans/`, согласовать до кода. Реализация по шагам с проверкой. Коммит и push — только по явной просьбе. Merge commit, не squash.

## Testing

Тесты: `backend/test_*.py`, корневые `test_*.py`, часть в `scripts/`. Папка `tests/` в `.gitignore` — новые тесты туда не класть. CI пока нет. Покрытие наращиваем на затронутых путях, не «тесты ради тестов». Frontend: ESLint (`npm run lint`), форматтера Python в репо нет.

## Gotchas

- Не поднимать workers в stage 04/05 без пересчёта RAM: контейнер 20g, иначе OOM хоста. См. комментарии в `pipeline_config.yaml` и KI-042.
- Stage 02 на анонимизированных BIDS ломает смысл — теги уже сняты.
- `skip_existing` + requeue шарят один `output_path`.
- Не коммитить: `data/` (клинические DICOM), веса (`*.pth`, `nnUNet*_data/`, `nnUNet_results/`), `*.db`, `.env`, `demo_workspace/`, `venv/`.
- Новую inference-модель: манифест + наследник `ServiceBase` + запись в `configs/services.yaml` + `lesion_types.yaml`. Контракт — JSON с путями, не multipart.
- Параллелизм этапов разный; простой GPU vs перегруз CPU — смотреть `scripts/performance_monitor.py` и `KNOWN_ISSUES.md`.
