# Known Issues & Technical Debt

Документ фиксирует все известные проблемы и архитектурные долги, выявленные
во время рефакторинга MAS-инфраструктуры (ветка `feat/mas-refactor`).
Все пункты не блокируют merge в main — это работа на будущие ветки.

Список разделён по областям. Каждый пункт помечен приоритетом и
рекомендуемой веткой для исправления.

---

## Pipeline portability (мульти-датасет support)

### KI-001 — Stage 01: `_detect_contrast` слишком жадный
**Приоритет:** низкий (косметика, не влияет на функциональность)
**Branch:** `chore/post-mas-cleanup`

FLAIR-серии в MS-клинических данных размечаются как `contrast=True`.
Скорее всего из-за того, что ContrastBolusAgent поле читается слишком
жадно, или пациенту делали FLAIR после контраста и тэги сохранились.

Модальность распознаётся корректно (`t2fl`), но в логах это создаёт
ложное впечатление. Фикс: уточнить условия `_detect_contrast` — FLAIR
по природе не контрастная, и contrast=True для неё должен либо
игнорироваться, либо понижать confidence.

### KI-002 — Stage 01: `print_summary` хардкодит 4 модальности
**Приоритет:** низкий (косметика)
**Branch:** `chore/post-mas-cleanup`

Для МС-кейса в выводе будет `t1c: 0 series`, что вводит в заблуждение.
Сделать список модальностей зависимым от `lesion_type` (по образцу
`CompletenessChecker`).

### KI-003 — Stage 01: фильтр модальностей дублирован
**Приоритет:** низкий
**Branch:** `chore/post-mas-cleanup`

Константа `{'t1', 't1c', 't2', 't2fl'}` встречается в `process_single_patient`
и `run_sequential`. Кандидат на вынос в module-level constant.

### KI-004 — Stage 04 (QA) не проверен на МС
**Приоритет:** высокий (вероятный блокер для MS-валидации)
**Branch:** `feat/lesion-type-aware-pipeline`

Скрипт `scripts/04_quality_assessment.py` (или похожий) не тестирован на
МС-данных. Скорее всего тоже хардкодит {t1, t1c, t2, t2fl} где-то.
Нужно: пройти по коду, найти аналогичные `REQUIRED_MODALITIES`, сделать
lesion-type-aware.

### KI-005 — Stage 05 (preprocessing) не проверен на МС
**Приоритет:** высокий (вероятный блокер)
**Branch:** `feat/lesion-type-aware-pipeline`

`05_preprocessing.py` использует T1c для регистрации (T1c→T1). Для МС
T1c отсутствует, и эта регистрация либо упадёт, либо нужно вообще не
запускать. Решение: conditional на lesion_type — для МС регистрировать
только T2→T1 и FLAIR→T1.

Оптимизация: для МС skip T1c-обработки экономит ~25% времени stage 05.

---

## Stage 06 (segmentation)

### KI-006 — Дубликат лога `PROCESSING SUMMARY`
**Приоритет:** низкий (косметика)
**Branch:** `chore/post-mas-cleanup`

`_process_sessions_async` и `run()` оба вызывают `stats.log_summary()`,
поэтому строка `PROCESSING SUMMARY` печатается дважды в stage_06.log.

### KI-007 — Validation `_scan_output_structure` не учитывает lesion_type subfolder
**Приоритет:** средний
**Branch:** `chore/post-mas-cleanup`

После Сабшага 2.3 outputs организованы под `{lesion_type}/` подпапкой,
но `_scan_output_structure` ищет на старом уровне. Поэтому в логе всегда:
```
Found 0 subjects in output
Success rate: 0.0%
```

Фикс: учесть подпапку `{lesion_type}/` при сканировании.

---

## MAS-сервисы

### KI-008 — torch.load monkey-patch дублирован
**Приоритет:** низкий (DRY)
**Branch:** `chore/post-mas-cleanup`

Patch для PyTorch 2.6+ (`weights_only=False` + numpy scalar allowlist)
повторяется идентично в `services/gbm-seg/src/service_server.py` и
`services/ms-seg/src/service_server.py`. Кандидат на вынос в
`services/common/torch_compat.py` с функцией
`enable_legacy_checkpoint_loading()`.

### KI-009 — Дрейф версий пакетов между gbm-seg и ms-seg
**Приоритет:** средний
**Branch:** `chore/post-mas-cleanup`

Quart, Hypercorn, Flask и др. указаны в requirements.txt каждого сервиса
отдельно. Любое расхождение вызывает несовместимости (как в KI-XX).
Решение: общий `services/common/requirements-base.txt`, наследуемый
обоими сервисами через `pip install -r ../common/requirements-base.txt`.

Альтернатива (более радикальная): общий base Docker image
`brain-lesion-base:latest`, от которого наследуются конкретные сервисы.

### KI-010 — GPU sharing между gbm-seg и ms-seg
**Приоритет:** низкий (известное ограничение)
**Branch:** Future MAS work (`feat/mas-coordinator`)

Оба сервиса хотят GPU 0. Когда один обрабатывает кейс, второй стоит в
очереди или может OOM-нуться. В клиническом сценарии (один запрос за раз)
это не блокер. При параллельном использовании потребуется GPU-coordinator
агент (часть Этапа MAS Implementation).

---

## Pipeline contract

### KI-011 — Host-style paths работают только при mount `/home`
**Приоритет:** средний (тех. долг, блокер для распределённого развёртывания)
**Branch:** Future (`feat/mas-advanced`, после v1.0)

Pipeline передаёт абсолютные host-пути (`/home/ubuntu/...`) через
`/predict` контракт. Это работает только потому, что и web, и
service-gbm-seg/service-ms-seg монтируют `/home`. Для cube/barguzin
профилей (распределённое развёртывание) потребуется либо унификация
путей (`/workspace/...`), либо HTTP file transfer mode.

### KI-012 — SSH-туннели не работают с новым контрактом
**Приоритет:** низкий (deferred deployment scenario)
**Branch:** Future (`feat/mas-advanced`)

Профили `cube` и `barguzin` в `segmentation_config.yaml` ссылаются на
SSH-туннели для удалённых машин. Новый `/predict` контракт требует
shared volume, которого нет при туннелировании. Сейчас работает только
local профиль.

Решение: HTTP file transfer mode (`POST /predict` multipart +
`GET /files/{id}`) как опциональный режим.

---

## Backend / Frontend

### KI-013 — История запусков не показывает MS-кейсы
**Приоритет:** высокий (визуальный блокер для пользователя)
**Branch:** `chore/post-mas-cleanup`

После прогона MS-кейса (verified: ms-seg отработал, Kappa получила
upload), история запусков в UI не отображает этот run. Скорее всего
где-то в backend SQL-запросе или frontend фильтре есть хардкод
`lesion_type = 'glioblastoma'`.

Диагностика:
```bash
grep -rn "glioblastoma\|lesion_type" backend/ frontend/src/ | grep -v test
```

### KI-014 — 3D Slicer не загружает маску автоматически для МС
**Приоритет:** высокий (важная UX-фича, ломается на МС)
**Branch:** `chore/post-mas-cleanup`

Slicer plugin раньше автоматически подгружал маску после завершения
pipeline. После перехода на `/predict` контракт + `{lesion_type}/`
подпапки эта интеграция перестала работать (хотя бы для МС).

Нужно: проверить какие пути Slicer plugin ожидает, обновить под новую
структуру output.

### KI-015 — Ручной vs автосозданный Kappa-датасет
**Приоритет:** низкий (operational)
**Branch:** `chore/post-mas-cleanup` или manual ops

KappaUploader автоматически создаёт датасет если в YAML нет записи.
Это привело к тому, что вручную созданный MS-датасет остался пустым,
а data попала в новый автоматический.

Решение: после первого прогона МС, удалить пустой ручной датасет
в Каппе. Дальше всё работает корректно. Это не баг кода — это
operational footgun.

Возможно стоит добавить логику синхронизации (при наличии нескольких
кандидатов спрашивать оператора), но это перебор для текущего масштаба.

---

## Архитектурные направления (не баги, future work)

### KI-016 — config-driven lesion_types
**Приоритет:** средний
**Branch:** `feat/lesion-type-aware-pipeline`

Сейчас `LESION_TYPE_MODALITIES` хардкодится в `CompletenessChecker`,
а в stage 05 (и других) ещё нет понимания. Чище — вынести в
`configs/lesion_types.yaml`:
```yaml
glioblastoma:
  required_modalities: [t1, t1c, t2, t2fl]
  reference_modality: t1
  reports: [volume, lobar]
multiple_sclerosis:
  required_modalities: [t1, t2, t2fl]
  reference_modality: t1
  reports: [lesion_count, volume_per_lobe]
```

Читать в каждом stage и не дублировать константы.

### KI-017 — Multi-trainer dispatch в ms-seg
**Приоритет:** низкий (future MAS feature)
**Branch:** `feat/mas-coordinator`

11 trainer'ов доступны в Dataset333_MSLesSeg, используется только
CATMIL. MAS-координатор будет выбирать trainer по кейсу. Сейчас
явно reject (NotImplementedError) если `options.trainer != CATMIL`.

### KI-018 — Cascade/Fallback/Ensemble режимы агентов (твоя идея)
**Приоритет:** future MAS direction
**Branch:** `feat/mas-coordinator`

Идея: каждая модель = отдельный агент, координатор оценивает confidence
и:
- **Cascade:** primary model → если confidence низкая → fallback
- **Fallback:** primary → при ошибке → автоматическое переключение
- **Ensemble:** N моделей параллельно → консенсус

Это **research-level задача** на Этап "MAS Implementation".

### KI-019 — Клинический отчёт адаптировать под МС
**Приоритет:** высокий (часть MS adaptation)
**Branch:** `feat/lesion-type-aware-pipeline`

Сейчас клинический отчёт (что бы это ни было — точно проверить в коде)
заточен под глио (объёмы, доли). Для МС нужны другие метрики:
- Количество очагов
- Распределение по локализации (перивентрикулярные, юкстакортикальные...)
- Изменения между сессиями (longitudinal analysis для МС-данных)

Это требует **отдельной экспертизы** — какие метрики клинически
релевантны для МС.

### KI-020 — Multi-center datasets
**Приоритет:** низкий (deferred to v1.1)
**Branch:** `feat/multi-center-datasets`

Текущий маппинг: `(lesion_type) → dataset_id`. Будущий:
`(lesion_type × center) → dataset_id`. Примеры:
- МТЦ Глиобластомы
- ФЦН Глиобластомы
- МТЦ РС
- ФЦН РС

Требует расширения kappa_datasets.yaml и UI выбора центра.

### KI-021 — Метастазы
**Приоритет:** v1.1
**Branch:** `feat/mets-seg`

Веса метастаз скопированы локально, но отдельный сервис не создан.
По аналогии с ms-seg:
- `services/mets-seg/`
- `Dataset_xxx_BrainMets` weights
- Manifest, registry entry
- Адаптация pipeline (множественные мелкие очаги имеют свои особенности
  по preprocessing и постпроцессингу)

---

## Performance & ops

### KI-022 — `.pyc` cache footgun в Docker
**Приоритет:** средний (commit-time protection done, runtime guard pending)
**Branch:** `chore/post-mas-cleanup`

В ms-seg мы добавили `ENV PYTHONDONTWRITEBYTECODE=1`. В gbm-seg и web
этого нет. После любой правки скриптов рискуем словить устаревший .pyc.

Решение: добавить `PYTHONDONTWRITEBYTECODE=1` во все Dockerfile.

### KI-023 — Benchmark для MAS-метрик
**Приоритет:** низкий (для MAS Implementation)
**Branch:** `feat/mas-coordinator`

`BenchmarkLogger` фиксирует stage-level метрики. Когда добавим
agent-level взаимодействие — нужно расширить под измерения межагентной
коммуникации, contract net handshakes, GPU pool wait times.

### KI-024 — Backup runtime configs и mapping files
**Приоритет:** низкий (operational)
**Branch:** `feat/prod-readiness`

`runtime_configs/` и `kappa_datasets.yaml` хранят критичный state.
В prod нужен backup strategy (rotation, off-site copy). Сейчас всё в
docker volume без backup.

---

## Документация

### KI-025 — Архитектурная документация для разработчиков
**Приоритет:** низкий (prod prep)
**Branch:** `feat/prod-readiness`

SPEC.md описывает MAS-рефакторинг. Нужна:
- README с quickstart
- Architecture overview
- API documentation
- Deployment guide
- Developer onboarding guide

---

## Закрытые в этой ветке

Для справки — что было исправлено до merge в main:

- ✅ Stage 01 `UPENN-GBM-` prefix filter
- ✅ Stage 01 `parse_date_from_series_name` US-only format
- ✅ Stage 01 `REQUIRED_MODALITIES` хардкод (теперь lesion-type-aware)
- ✅ `LESION_TYPE` хардкод в stages 06/07/08 (теперь CLI args)
- ✅ Sequential vs parallel `process_one_mask` в stages 07/08 (lesion_type
  как явный параметр функции)
- ✅ Legacy multipart контракт в gbm-seg (удалён)
- ✅ Quart/Flask version drift между gbm-seg и ms-seg (синхронизированы)
- ✅ numpy/scipy ABI mismatch в ms-seg (force-reinstall numpy 2.3+ scipy 1.13+)

---

*Последнее обновление: финализация ветки `feat/mas-refactor`.*
