# Known Issues & Technical Debt

Документ фиксирует все известные проблемы и архитектурные долги, выявленные
во время рефакторинга MAS-инфраструктуры (ветка `feat/mas-refactor`) и
последующих работ.
Все пункты не блокируют merge в main — это работа на будущие ветки.

Список разделён по областям. Каждый пункт помечен приоритетом и
рекомендуемой веткой для исправления.

---

## Pipeline portability (мульти-датасет support)

### KI-001 — Stage 01: `_detect_contrast` слишком жадный для FLAIR
**Приоритет:** высокий (первый приоритет Этапа 6, см. KI-027)
**Branch:** `feat/prod-readiness` (Этап 6, перенесён из `feat/lesion-type-aware-pipeline`)

FLAIR-серии в MS-клинических данных размечаются как `contrast=True`.
Скорее всего из-за того, что ContrastBolusAgent читается слишком жадно,
или пациенту делали FLAIR после контраста и тэги сохранились.

Модальность распознаётся корректно (`t2fl`), но в логах это создаёт
ложное впечатление.

**Перенос обоснован:** контрастный FLAIR — реальный клинический случай,
не косметический баг. Правильное решение — не игнорировать `contrast=True`
для FLAIR в логе, а ввести скоринг confidence в `ModalityDetector` и
осмысленно учитывать его при выборе серии. Это связано с KI-027 и
делается одной работой в `feat/prod-readiness` (протокол-специфичная
фильтрация на Stage 01).

### KI-002 — Stage 01: `print_summary` хардкодит 4 модальности ✅ ЗАКРЫТО
**Приоритет:** низкий (косметика)
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `43cb89c`

Для МС-кейса в выводе будет `t1c: 0 series`, что вводит в заблуждение.
Сделать список модальностей зависимым от `lesion_type` (по образцу
`CompletenessChecker`). В функции три места хардкода: инициализация
`modality_counts`, инициализация `modality_slices`, цикл вывода.

### KI-003 — Stage 01: фильтр модальностей дублирован ✅ ЗАКРЫТО
**Приоритет:** низкий
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `6fa72dc`

Константа `{'t1', 't1c', 't2', 't2fl'}` встречается в:
- `process_single_patient` (parallel path)
- `run_sequential` (sequential path)
- `bids_suffix_map` в `FileOrganizer.copy_series` (третье место,
  выявлено в ходе анализа Этапа 3)

Кандидат на вынос в единый источник — module-level constant либо
переиспользование `CompletenessChecker.LESION_TYPE_MODALITIES`.

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

### KI-006 — Дубликат лога `PROCESSING SUMMARY` ✅ ЗАКРЫТО
**Приоритет:** низкий (косметика)
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `3efa78a`

`_process_sessions_async` и `run()` оба вызывают `stats.log_summary()`,
поэтому строка `PROCESSING SUMMARY` печатается дважды в stage_06.log.

### KI-007 — Validation `_scan_output_structure` не учитывает lesion_type subfolder ✅ ЗАКРЫТО
**Приоритет:** средний
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `3efa78a`

После Сабшага 2.3 outputs организованы под `{lesion_type}/` подпапкой,
но `_scan_output_structure` ищет на старом уровне. Поэтому в логе всегда:
Found 0 subjects in output
Success rate: 0.0%

Фикс: учесть подпапку `{lesion_type}/` при сканировании.

---

## MAS-сервисы

### KI-008 — torch.load monkey-patch дублирован ✅ ЗАКРЫТО
**Приоритет:** низкий (DRY)
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `29a3984`

Patch для PyTorch 2.6+ (`weights_only=False` + numpy scalar allowlist)
повторяется идентично в `services/gbm-seg/src/service_server.py` и
`services/ms-seg/src/service_server.py`. Кандидат на вынос в
`services/common/torch_compat.py` с функцией
`enable_legacy_checkpoint_loading()`.

### KI-009 — Дрейф версий пакетов между gbm-seg и ms-seg ✅ ЗАКРЫТО
**Приоритет:** средний
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `e754e2e`

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

### KI-013 — Вкладка валидации не показывает MS-датасет ✅ ЗАКРЫТО
**Приоритет:** высокий (визуальный блокер для пользователя)
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `1f5a370` (`chore/post-mas-cleanup`)

**Уточнение по ходу диагностики:** изначально проблема была сформулирована
как «история запусков не показывает MS». Оказалось, что история запусков
работает корректно — баг был во вкладке **Валидация**, которая видела
только глио-датасет.

**Root cause:** `ValidationPanel.jsx::loadLesionTypes` брал первый
`lesion_type` с `dataset_id` (`types.find(t => t.dataset_id)`) и
блокировался на нём. UI-переключателя датасетов не существовало —
`lesionTypes` хранились в state, но нигде не отрисовывались. Backend
`get_lesion_types()` отдавал оба типа корректно.

**Fix:** добавлен Antd `<Select>` в `extra` карточки валидации, привязан
к `selectedDatasetId`. Существующий `useEffect([selectedDatasetId])`
автоматически подтягивает entities при смене значения. Backend не
менялся.

### KI-014 — 3D Slicer не загружает маску для multi-patient/multi-session runs ✅ ЗАКРЫТО
**Приоритет:** высокий (UX-блокер, проявлялся для МС)
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `7f11f2d` (`chore/post-mas-cleanup`)

**Уточнение по ходу диагностики:** изначально проблема была сформулирована
как «Slicer не работает для MS». На самом деле этот баг существовал
давно и для глио, но оставался невидимым на типичном single-session
UPENN-GBM кейсе.

**Root cause:** `/api/slicer/open/{run_id}` делал `record = records[0]`
после `find_by_run_id(run_id)`, всегда выбирая первого пациента
независимо от того, какого пользователь открыл в UI. Для MS-кейса с
двумя сессиями одного пациента (`ses-001` без масок, `ses-002` с
маской) этот баг сразу проявился: клик на `ses-002` загружал данные
`ses-001` → пустую историю масок → Slicer без маски.

**Fix:** эндпоинт принимает опциональный query-параметр `entity_id`.
Если передан — выбирается соответствующая запись из `find_by_run_id`.
Если не передан и в run'е больше одной записи — 400. Single-record
runs сохраняют обратную совместимость (entity_id опционален).
Фронт `openInSlicer` теперь пробрасывает `entityId` из `ValidationActions`.

**Гипотеза про цветовую легенду MS, отвергнутая по ходу:** изначально
предполагалось, что маска MS не отображается из-за несовпадения color
table (4 класса у глио vs 1 у MS). Гипотеза неверна — Slicer не
получал путь к маске вообще, проблема была чисто в выборе записи. Если
проблемы с легендой реально появятся при отображении MS-масок — это
будет уже отдельный пункт для `feat/lesion-type-aware-pipeline`.

### KI-015 — Ручной vs автосозданный Kappa-датасет ✅ ЗАКРЫТО
**Приоритет:** низкий (operational)
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `a81e073`

KappaUploader автоматически создаёт датасет если в YAML нет записи.
Это привело к тому, что вручную созданный MS-датасет остался пустым,
а data попала в новый автоматический.

Решение: после первого прогона МС, удалить пустой ручной датасет
в Каппе. Дальше всё работает корректно. Это не баг кода — это
operational footgun.

Возможно стоит добавить логику синхронизации (при наличии нескольких
кандидатов спрашивать оператора), но это перебор для текущего масштаба.

### KI-026 — Silent Kappa dedup confuses validation UX
**Приоритет:** низкий (UX-improvement)
**Branch:** `feat/prod-readiness`

Pipeline дедуплицирует пациентов по `study_hash` при загрузке в Каппу:
если пациент уже был загружен в Каппе ранее, текущий run корректно
завершается, но новых entity не создаёт. Как следствие, для такого
run'а в UI отсутствует валидационная панель и история масок —
валидировать в контексте этого run'а нечего.

Поведение корректное (валидация привязана к Каппа-entity, дубликаты не
создаются), но молчаливое: пользователь видит «нет панели» и думает,
что что-то сломалось. Это путает даже разработчика (выявлено по ходу
работы над KI-014).

Предложение:
- В карточке завершённого run'а без `patient_registry`-записей с
  `kappa_entity_id` показывать сообщение типа «Этот прогон не содержит
  новых данных для валидации (все пациенты уже загружены в Каппу ранее)».
- Опционально — линк на run, в котором эти entity действительно лежат
  (потребует обратного запроса в `patient_registry` по `study_hash` или
  по паре `original_patient_id + scan_date`).

Это не баг, это UX-полировка. Откладывается в `feat/prod-readiness`.

### KI-035 — Kappa dedup блокирует доукомплектование неполного исследования
**Приоритет:** средний (data-integrity, реальный клинический сценарий)
**Branch:** `feat/prod-readiness` (или своя `feat/kappa-reupload`)

Связан с KI-026, но глубже. Сейчас дедуп по `study_hash` пропускает
повторный прогон того же пациента целиком. Если первый прогон был с
**неполными** данными (например, одна сессия без части модальностей →
без маски), а потом данные дослали и пациента прогнали заново с
**полным** набором — система видит дубликат и в Каппе навсегда остаётся
неполная версия. Полные данные никогда не попадут в Каппу.

**Клиническая реальность:** сценарий не редкий (5–15% кейсов в курации
архивов): пациент не вынес полный протокол → дослали последовательности
позже; motion-артефакт → пересняли в другой день; серии приходят из PACS
батчами; постконтраст залит отдельно. То есть «неполное → позже полное»
— штатная ситуация для дозагружаемых клинических архивов.

**Выбранное решение — content-aware update:**
- Дедуп остаётся по идентичности исследования (`study_hash`), но при
  совпадении сравнивается **сигнатура полноты** (набор модальностей и
  сессий с масками) входящего прогона против сохранённого в Каппе.
- Если входящие данные — надмножество (полнее) → не пропускаем, а
  **обновляем** существующую entity (перезаливаем данные/маски), а не
  создаём дубликат и не пропускаем молча.
- Если входящие данные равны или беднее → текущее поведение (skip).
- Требует update-семантики в KappaUploader/kappa_client (PATCH/replace
  entity вместо create), и расчёта completeness-сигнатуры.

Альтернативы, отклонённые при обсуждении: completeness-gated upload (не
грузить неполные вовсе — теряем валидацию частичных данных);
`--force-reupload` флаг (ручное действие оператора, не решает проблему
системно).

**Операционный воркэраунд до реализации:** перед повторным прогоном
того же пациента удалить его entity из Каппы вручную, либо использовать
свежий датасет. Иначе валидация показывает данные первого прогона.

### KI-036 — `empty.json` заглушки засоряют сущности в Каппе
**Приоритет:** низкий (operational cleanup)
**Branch:** `feat/prod-readiness`

`kappa_client.update_entity_status` отправляет multipart с обязательным
полем `files`, даже когда файла нет — заглушку `("empty.json", b"{}")`.
Каппа сохраняет её как файл сущности, поэтому у каждой провалидированной
сущности накапливаются `empty.json`. Не баг (API так устроен), но мусор.
Решение: проверить, принимает ли API status-update без `files` (или с
пустым списком), либо чистить заглушки после смены статуса.

### KI-037 — Недетерминизм счёта очагов РС между прогонами
**Приоритет:** средний (доверие к отчёту)
**Branch:** `feat/ms-clinical-metrics` (вариант C) или модельная работа

Количество очагов РС меняется от прогона к прогону на одних данных.
Причина — недетерминизм самой сегментации: подтверждено чтением
`services/ms-seg/src/service_server.py` — `cudnn.benchmark = True` и
`use_mirroring=True` (TTA), без фиксированного seed где-либо в
`services/ms-seg/src/`. Маска каждый раз чуть иная (границы мелких
очагов смещаются на 1-2 вокселя) → другое число связных компонент. Это
уровень модели, не алгоритма счёта. Варианты: детерминированный инференс
(фиксированный seed, отключение TTA — ценой качества), или
усреднение/ансамбль с порогом. Связано с KI-018 (cascade/ensemble).
Мин-фильтр размера очага (см. ниже) частично снижает разброс за счёт
удаления шумовых вокселей, но корень — в модели.

**Затрагивает не только счёт, но и cross-session diff** (Plan 3,
`feat/ms-clinical-metrics`): `backend/lesion_diff.py` сопоставляет очаги
между сессиями по перекрытию масок с допуском `dilation_voxels`
(`configs/ms_longitudinal_config.yaml`). При `dilation_voxels=1` то же
самое смещение границ ложно классифицировало стабильные мелкие очаги
как пары new+resolved. Поднято до `dilation_voxels=3` эмпирическим
подбором (3 повторных прогона одних данных) — снимает симптом для diff,
но не лечит причину; тот же недетерминизм продолжит проявляться в
голом счёте очагов (см. выше) и в любом другом потребителе MS-масок.

### KI-038 — Hover-объём срабатывает по клику, а не по наведению
**Приоритет:** низкий (UX-polish)
**Branch:** `feat/prod-readiness`

`niivue.onLocationChange` срабатывает при движении перекрестия (клик/драг),
а не при пассивном наведении мыши. Поэтому tooltip объёма очага появляется
по клику. Для true-hover нужен отдельный `mousemove`-листенер на canvas,
переводящий экранные координаты в воксель и читающий метку маски. Не
критично — объём всё равно доступен, просто по клику.

### KI-039 — Viewer: «Сброс» не работает, 3D открыт увеличенным
**Приоритет:** низкий (viewer polish)
**Branch:** `feat/prod-readiness` (или попутно)

`NIfTIViewer.resetView` зовёт только `setSliceType(multiplanar)` — не
сбрасывает зум/пан. А `setScale(2.0)` при загрузке открывает 3D-объём
увеличенным относительно срезов. Нужно: в `resetView` восстанавливать
масштаб и центрирование; пересмотреть дефолтный `setScale`.

### KI-040 — История запусков не обновляется автоматически
**Приоритет:** низкий
**Branch:** `feat/prod-readiness`

`PipelineHistory` обновляется по `setInterval` только пока есть `running`
прогоны; завершённые/новые не подтягиваются без ручного обновления.
Нужно: периодический refresh или инвалидция после завершения прогона.

### KI-041 — Медленная загрузка снимков в валидации
**Приоритет:** средний (UX, prod)
**Branch:** `feat/prod-readiness`

В валидации NIfTI грузятся через backend-прокси из Каппы — медленно.
Кандидаты: кэширование скачанных файлов, потоковая передача, сжатие,
параллельная загрузка модальностей, либо прямые подписанные URL Каппы.
Требует профилирования (где именно затык — Каппа, прокси, сеть).

### KI-044 — Клинический отчёт: выбор пациента через выпадающий список
**Приоритет:** низкий (UX)
**Branch:** `main`

Сейчас `ClinicalReportContent` при нескольких пациентах в прогоне рендерит их
единым полотном (все пациенты подряд, сгруппированы по пациенту). Для многопа-
циентных прогонов удобнее выпадающий список выбора пациента, показывающий отчёт
по одному. Не критично: отчёт корректен и читаем, вопрос эргономики. Реализация:
Select с `original_patient_id`/`sub-XXX` (маппинг уже есть через
`/api/run/{run_id}/patient-map`), рендерить только выбранного пациента.

---

## Архитектурные направления (не баги, future work)

### KI-016 — config-driven lesion_types ⚠️ ЧАСТИЧНО ЗАКРЫТО
**Приоритет:** средний
**Branch:** `feat/lesion-type-aware-pipeline`
**Partial in:** commit `55bee8f` (`chore/post-mas-cleanup`) — `configs/lesion_types.yaml` создан, код не мигрирован

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

### KI-018 — Cascade/Fallback/Ensemble режимы агентов
**Приоритет:** future MAS direction
**Branch:** `feat/mas-coordinator`

Идея: каждая модель = отдельный агент, координатор оценивает confidence
и:
- **Cascade:** primary model → если confidence низкая → fallback
- **Fallback:** primary → при ошибке → автоматическое переключение
- **Ensemble:** N моделей параллельно → консенсус

Это **research-level задача** на Этап "MAS Implementation".

### KI-019 — Клинический отчёт адаптировать под МС ✅ ЗАКРЫТО
**Приоритет:** высокий (часть MS adaptation)
**Branch:** `feat/ms-clinical-metrics` (Этап 5.6, Вариант C)
**Closed in:** merge commit `c9279c9`

Сейчас клинический отчёт заточен под глио (объёмы, доли). Для МС нужны
другие метрики:
- ~~Количество очагов~~
- ~~Распределение по локализации (перивентрикулярные, юкстакортикальные...)~~
- ~~Изменения между сессиями (longitudinal analysis для МС-данных)~~

Реализовано: McDonald-классификация локализации очагов (Stage 08,
`MSZoneAnalyzer` + API + UI), детекция новых/растущих/разрешившихся
очагов между сессиями (`backend/lesion_diff.py` + API + UI). Открытые
моменты, явно зафиксированные как deferred в design spec, не баги:
spinal cord зона не реализована (нет spine-регистрации в пайплайне);
гадолиний-накапливающие очаги (T1c) отложены до решения по T1c-протоколу
для МС; порог "growing" (`growth_threshold_relative/absolute_cm3`) —
временный дефолт, ждёт подтверждения клиницистов; `dilation_voxels` —
см. KI-037 (недетерминизм сегментации).

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

### KI-022 — `.pyc` cache footgun в Docker ✅ ЗАКРЫТО
**Приоритет:** средний
**Branch:** `chore/post-mas-cleanup`
**Closed in:** commit `8a830e9` (`chore/post-mas-cleanup`)

**Fix:** в `services/gbm-seg/Dockerfile` и `web.Dockerfile` добавлены
`ENV PYTHONDONTWRITEBYTECODE=1` и `ENV PYTHONUNBUFFERED=1` (по образцу
ms-seg, где они уже были). Унифицированно во всех трёх Dockerfile.
Решение по `PYTHONUNBUFFERED=1` принято по ходу — раз уж унифицируем,
делаем по полному образцу для согласованности логов в `docker logs`.

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

### KI-042 — Ресурсоёмкость этапов зависит от разрешения данных, а конфиг статичен
**Приоритет:** высокий (архитектурный — вход для MAS resource allocation)
**Branch:** `main` (выявлено при тестировании на SibBMS, 15.07.2026)

`workers` в `pipeline_config.yaml` — это **статические константы**, не зависящие
от того, какие данные пришли. Но реальное потребление памяти воркером линейно
масштабируется с числом вокселей, и разброс между датасетами — на порядок:

| Датасет | Разрешение | Размерность | Вокселей | Пик/воркер (Stage 04, до фикса) |
|---|---|---|---|---|
| MS_5 (baseline) | ~1 мм | — | ~10–20M | ~0.5 ГБ |
| SibBMS | **0.35 мм** | `[310, 864, 864]` | **231M** | **7.6 ГБ** |

Один и тот же конфиг (`workers: 12`), успешно отработавший на MS_5, на SibBMS
дал 12 × 7.6 ≈ **91 ГБ** запроса памяти на машине с 30 ГБ → global OOM (см.
KI-043). Ошибка не в логике, а в том, что **система не знает цену задачи до
её запуска**.

Дополнительный множитель — количество сессий: «3 пациента» SibBMS = **11 сессий
/ 33 тома** (датасет лонгитюдный, у sub-003 шесть таймпоинтов). Счёт «пациентов»
не отражает объём работы.

**Что из этого следует для MAS распределения ресурсов:**
- Агенту нужен **профиль ресурсов** = f(этап, разрешение, число томов), а не
  константа. Хотя бы линейная оценка: `bytes_per_worker ≈ voxels × dtype_size × k_stage`,
  где `k_stage` — эмпирический множитель (для Stage 04 после фикса k ≈ 2.2).
- Нужна **предполётная оценка** (dry-run по заголовкам NIfTI/DICOM — размерность
  читается без загрузки вокселей, ср. `nib-ls`), чтобы планировать параллелизм
  до старта, а не узнавать об OOM по факту.
- Единица планирования — **сессия/том**, не «пациент».

**Измерено (15.07.2026, `scripts/system_monitor.py`, SibBMS 3 пациента = 11 сессий,
`[310,864,864]` @0.35мм = 231M вокселей, cap контейнера 20 ГБ, фон хоста ~8.3 ГБ):**

| Этап | Воркеров | Пик RAM (система) | Доля контейнера | ГБ/воркер | Итог |
|---|---|---|---|---|---|
| 01 reorganize | 8 | 10.1 ГБ | ~1.8 ГБ | ~0.2 | OK |
| 03 convert | 8 | 13.5 ГБ | ~5.2 ГБ | ~0.7 | OK |
| 04 quality | 6 | 19.5 ГБ | ~11.2 ГБ | **~1.9** | OK (после фикса; было ~7.6) |
| 05 preprocessing | 6 | 25.9 ГБ | ~17.6 ГБ | **~2.9** | ⚠️ прошёл, запас 2.4 ГБ + swap 2.1 ГБ |
| 06 segmentation | 2 (GPU) | 10.9 ГБ | ~2.6 ГБ | ~1.3 | OK (нагрузка на GPU, не RAM) |
| 07 inverse | 5 | 26.2 ГБ | упёрся в cap | **~6.0** | ❌ cgroup OOM → BrokenProcessPool |

Ключевые наблюдения для планировщика:
- Разброс `ГБ/воркер` между этапами — **×30** (0.2 → 6.0). Единый `workers` на все
  этапы принципиально не может быть верным.
- Самые дорогие — этапы с ANTs, держащие нативный reference: 05 и 07. У 07 на
  каждую модальность в памяти reference (0.93 ГБ) + warped output (0.93 ГБ) +
  внутренние буферы ANTs.
- Этап 06 (сегментация) по хостовой RAM дёшев — его бюджет это **VRAM**, отдельное
  измерение (`gpu_monitor.py`), другая размерность ресурса.
- **Логи врут об успехе:** этап 05 отрапортовал `SUCCESS`, хотя занял 17.6 из 20 ГБ
  и лез в swap. Без внешнего монитора такие «прошёл на грани» не видны — планировщику
  нужна телеметрия, а не только код возврата.
- У этапа 07 **уже есть** авто-тюнер ([`07_inverse_transform.py:255-281`]), но он
  считает только по CPU (`effective_cpu // 4`, резерв 2 ядра под ОС) и полностью
  слеп к памяти. Это готовая точка врезки для memory-aware политики.

Прикидка для оценки бюджета: `bytes_per_worker ≈ voxels × k_stage`, где для
231M вокселей: `k_04 ≈ 8`, `k_05 ≈ 12.5`, `k_07 ≈ 26` (байт/воксель).
Проверить на датасете с другим разрешением (MosMed / clinical_dicom) — если
линейность подтвердится, это рабочая модель стоимости для планировщика.

### KI-043 — Нет memory-aware admission control (OOM ронял ОС) ⚠️ ЧАСТИЧНО ЗАКРЫТО
**Приоритет:** высокий (ops/robustness)
**Branch:** `main`
**Partially closed in:** commits `4dcd792`, `6fece76`

**Что произошло (15.07.2026):** запуск на 3 пациентах SibBMS исчерпал 30 ГБ RAM
+ 8 ГБ swap. Kernel OOM killer убил **десктоп** (VSCode → Firefox →
gnome-software → nautilus) — потребовалась перезагрузка. Подтверждено:
`Out of memory: Killed process ... global_oom`, invoked из `cpuset=docker-…`.
Причина: контейнер `web`, в котором крутятся этапы 01–08, **не имел лимита
памяти** и мог съесть всю машину.

**Закрыто:**
- `docker-compose.yml`: `mem_limit: 20g` + `memswap_limit: 20g` на `web`
  (равные значения отключают swap контейнера → нет swap-thrash). Теперь
  перегруженный этап падает **внутри контейнера**, ОС выживает.
- Stage 04: пик воркера **7.6 → 2.1 ГБ** (float32-загрузка + in-place градиент).
- Stage 04: `multiprocessing.Pool.map()` **вис навечно**, когда воркера убивал
  cgroup OOM killer (>1 ч при ~0% CPU, ждал результат от мёртвого процесса).
  Заменён на `ProcessPoolExecutor` → `BrokenProcessPool`, fail-fast.

**Что осталось открытым:**
- **Нет admission control:** система по-прежнему запускает `workers` штук
  воркеров, не проверив, влезут ли они в лимит. Лимит только ловит последствия.
- **Контейнеры `service-gbm-seg` / `service-ms-seg` без `mem_limit`** — там
  только GPU-резервация. Хостовую RAM при инференсе никто не ограничивает.
- **Нет динамической деградации:** при нехватке памяти правильное поведение —
  снизить параллелизм и доработать медленнее, а не упасть. Сейчас только падение.
- `k_stage` для этапов 05–08 не измерены (см. KI-042).

**Для MAS:** это ровно та задача, которую должен решать координатор ресурсов —
приём заявок с оценкой стоимости, бюджетирование по памяти/GPU, backpressure
и graceful degradation вместо OOM. Связано с KI-023 (benchmark MAS-метрик)
и KI-042 (профиль ресурсов).

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

## Stage 01 deep-review findings (выявлены при анализе для Этапа 3)

### KI-027 — `_select_best_series` использует упрощённую эвристику
**Приоритет:** высокий (первый приоритет Этапа 6)
**Branch:** `feat/prod-readiness` (Этап 6, перенесён из `feat/lesion-type-aware-pipeline`)

Сейчас `SeriesDeduplicator._select_best_series` выбирает серию с
максимальным `slice_count`. Это сильное упрощение.

В клинической практике у пациента в одной сессии может быть:
- несколько T1 разного качества (с/без contrast, разная толщина срезов);
- сериа FLAIR с/без motion artifact;
- "пробные" короткие серии и "полноценные" длинные.

Правильный выбор требует скоринга:
- разрешение и толщина срезов;
- наличие ключевых слов в описании (`MPRAGE`, `SPACE`, `optimal`);
- отсутствие маркеров неудачи (`failed`, `repeat`, `motion`);
- для t1c — наличие контрастных маркеров;
- для FLAIR — длинный TI;
- для МС — особые требования (3D FLAIR predпочтительнее 2D).

Связано с KI-001: контрастный FLAIR — реальный случай, его осмысленная
обработка требует confidence-скоринга в `ModalityDetector` и
интеграции скоринга в `_select_best_series`. Делается одной работой —
протокол-специфичная фильтрация на Stage 01 в Этапе 6.

### KI-028 — `validate_copy` может давать ложные результаты при частичном обновлении
**Приоритет:** низкий
**Branch:** `feat/prod-readiness`

`FileOrganizer.validate_copy` сравнивает количества:
- source: `rglob("*.dcm")` (рекурсивно)
- target: `glob("*.dcm")` (плоско)

Это правильно для текущего use case (target всегда плоский). Но если
когда-нибудь добавится режим частичного обновления (заменить только
одну модальность у пациента), в target могут оказаться файлы от
предыдущего прогона, и валидация даст ложный результат.

Не критично сейчас, но стоит зафиксировать на этап подготовки к проду.

### KI-029 — Hardcoded `.dcm` extension по всему скрипту
**Приоритет:** низкий
**Branch:** `feat/prod-readiness`

`scripts/01_reorganize_folders.py` везде использует `*.dcm` для поиска
DICOM-файлов. Реальные клинические данные часто бывают:
- без расширения (`IM-0001-0001`);
- с расширением `.IMA`, `.dicom`, `.DCM` (case-sensitive на ext4!).

Если придёт центр с такой раскладкой, скрипт молча скажет «no DICOM
files» и пропустит пациента. Нужно: либо detection через DICM-магию в
начале файла, либо расширяемый список патернов в конфиге.

### KI-030 — `pydicom force=True` маскирует не-DICOM файлы
**Приоритет:** низкий
**Branch:** `feat/prod-readiness`

В `ModalityDetector.detect_modality` используется
`pydicom.dcmread(..., force=True)`. При этом любой не-DICOM файл
(`README.txt` переименованный в `series.dcm`) читается как пустой
Dataset, серия проходит как «NO MATCH». В логах это неинформативно.

Улучшение: проверять `dcm.SOPClassUID` или хотя бы `len(dcm.dir())`
перед тем как считать файл валидным DICOM-ом.

### KI-031 — `ModalityDetector._cache` бесполезен в parallel-режиме
**Приоритет:** низкий (performance optimization)
**Branch:** `feat/prod-readiness` или `feat/mas-coordinator`

Кэш `self._cache: Dict[Path, ...]` живёт внутри инстанса детектора,
который создаётся в каждом форке заново. Это означает, что повторное
сканирование одной и той же серии (если бы оно случилось) в разных
форках не использует кэш.

Не баг (per-process кэш всё равно работает внутри одного процесса),
но если перейти на shared cache через `mp.Manager` — будет ощутимое
ускорение при больших датасетах. Кандидат на оптимизацию, когда
будем работать над prod-readiness.

### KI-032 — Логи в parallel-режиме перемешиваются построчно
**Приоритет:** низкий (DX)
**Branch:** `feat/prod-readiness`

В `process_single_patient` создаётся отдельный logger
`f'patient_{patient_dir.name}'`, но все они в итоге пишут в один
StreamHandler через root logger. При `workers >= 4` строки разных
пациентов будут перемежаться в `stdout`.

Решение: `logging.handlers.QueueHandler` + `QueueListener` в основном
процессе, или per-process логфайлы. Делается одной правкой, но
требует тестирования.

### KI-033 — Stage 05: N4 bias correction последовательна по модальностям внутри воркера
**Приоритет:** низкий (performance optimization)
**Branch:** `feat/prod-readiness` или отдельная `perf/stage05-intra-session-parallel`

Внутри одной сессии N4 bias correction запускается последовательно: t1c → t1 → t2 → t2fl.
Эти вычисления независимы и могли бы идти параллельно через `ThreadPoolExecutor` внутри воркера.

Ожидаемый выигрыш для 4 модальностей:
- Сейчас:  t1c(280s) + t1(46s) + t2(7s) + t2fl(154s) ≈ 487s
- С intra-session parallel: max(280, 46, 7, 154) ≈ 280s (~1.7× быстрее для этого шага)

Ограничение: нельзя реализовать как work-stealing между воркерами (разные сессии — разные
процессы с изолированными файлами). Только параллелизм внутри одного воркера через потоки.

Зависимость: registration зависит от результата N4 (нужен bias-corrected T1 для регистрации
к атласу). Параллелить можно только шаг N4, не всю цепочку.

---

### KI-034 — Stage 06: `args` используется как module-level global внутри методов класса
**Приоритет:** средний (скрытая ошибка при импорте)
**Branch:** `feat/prod-readiness`

`_create_session` (строка ~603) и `_process_sessions_async` (строки ~1008-1010) обращаются
к переменной `args` напрямую как к глобальной переменной модуля, а не через `self.args`.
Это работает только при запуске как `__main__`, когда `args = parse_arguments()` в глобальном
пространстве имён. При импорте модуля или тестировании `NameError: name 'args' is not defined`.

Правильное решение: заменить все обращения к `args` внутри методов класса `SegmentationRunner`
на `self.args`.

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

Закрытые в `chore/post-mas-cleanup`:

- ✅ KI-013 (commit `1f5a370`) — Validation lesion-type selector
- ✅ KI-014 (commit `7f11f2d`) — Slicer entity_id selection
- ✅ KI-022 (commit `8a830e9`) — PYTHONDONTWRITEBYTECODE in all Dockerfiles
- ✅ KI-002 (commit `43cb89c`) — print_summary lesion-type-aware
- ✅ KI-003 (commit `6fa72dc`) — modality whitelist centralized
- ✅ KI-006 (commit `3efa78a`) — duplicate PROCESSING SUMMARY removed
- ✅ KI-007 (commit `3efa78a`) — _scan_output_structure uses rglob for lesion_type subfolder
- ✅ KI-008 (commit `29a3984`) — torch_compat extracted to services/common/
- ✅ KI-009 (commit `e754e2e`) — requirements-base.txt for shared deps
- ✅ KI-015 (commit `a81e073`) — KappaUploader warns on empty duplicate dataset
- ⚠️ KI-016 (commit `55bee8f`) — lesion_types.yaml created; code migration in feat/lesion-type-aware-pipeline

Дополнительные исправления (не KI, выявлены в ходе stage survey):

- ✅ Stage 03: dcm2niix artifact cleanup, orchestrator max_subjects fix
- ✅ Stage 04: skip logic, parallel skipped counter, UPENN tuple, modality parsing (7 bugs)
- ✅ Stage 05: 6 bugs incl. reference_modality by name, skipped counter, FSL log level
- ✅ Stage 05/07: auto-tune parallelism + OS core reservation (effective_cpu = cpu_count - 2)
- ✅ Stage 06: server availability checks correct service URL per lesion_type
- ✅ Stage 06: GPU retry with backoff on cudaErrorDevicesUnavailable (3 attempts, 30/60s)
- ✅ Stage 07: thread limits in parallel mode, skipped counter in benchmark
- ✅ Stage 08: lesion_type filter, skipped counter, atlas reload in parallel mode
- ✅ Docker: build network:host + proxy args for seg services; pip timeout 120s

---

*Последнее обновление: расширение Этапа 3 (deep review of Stage 01),
добавление KI-027..032, перенос KI-001 в `feat/lesion-type-aware-pipeline`.*