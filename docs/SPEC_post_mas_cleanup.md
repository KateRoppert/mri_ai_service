# Техническое задание: post-MAS cleanup

**Проект:** Web-сервис AI-сегментации поражений головного мозга по МРТ
**Версия ТЗ:** 0.3 (после закрытия Этапов 1 и 2, расширения Этапа 3)
**Дата:** 26 мая 2026
**Ветка:** `chore/post-mas-cleanup` (от `main` после merge `feat/mas-refactor`)
**Автор:** Kate Roppert (НГУ, AI Research Center)

---

## 1. Контекст и мотивация

В ветке `feat/mas-refactor` (9 коммитов, замержена в main) монолитный
сервис под глиобластому был превращён в multi-service инфраструктуру с
двумя сервисами (gbm-seg + ms-seg) и динамической диспетчеризацией по
`lesion_type`. End-to-end проверено на реальных кейсах глио (UPENN-GBM)
и МС (P000915).

В процессе рефакторинга накопился технический долг, не блокирующий merge,
но требующий чистки до следующих больших работ. Все известные пункты
зафиксированы в `KNOWN_ISSUES.md`.

Цель данной ветки — **стабилизация baseline**: закрыть все пункты с
приоритетом «средний» и «высокий», помеченные branch'ем
`chore/post-mas-cleanup`, плюс тщательно пройти по всем этапам pipeline
на предмет багов и косметических артефактов рефакторинга.

**Принципиальное ограничение:** в этой ветке **нет архитектурных
изменений на уровне контракта сервисов**. Контракт `/predict`, схема БД,
layout output — не трогаем. Внутренние рефакторинги pipeline-скриптов
(включая унификацию sequential/parallel логики) разрешены и желательны.

---

## 2. Принципы работы в этой ветке

- **Атомарные коммиты.** Один коммит = одна завершённая мысль. Каждый
  коммит проверяется `python3 -m py_compile` + (где применимо) запуском
  pipeline.
- **Conventional commits.** Формат: `<type>(<scope>): <subject>`,
  основные types: `fix`, `refactor`, `chore`, `docs`, `test`.
- **Smoke-тесты после каждого этапа** (см. §4):
  - Поведенческие изменения → smoke на соответствующем типе кейса.
  - Косметика → `py_compile` + любой прогон pipeline.
  - Infra → `docker compose build` + `/health` обоих сервисов.
- **Обсуждение перед кодом.** Любая неочевидная развилка обсуждается
  до написания кода.
- **Фиксация после каждого сабшага.** KNOWN_ISSUES.md и этот SPEC
  обновляются по мере закрытия пунктов (отметка «закрыто» с указанием
  коммита/этапа).

---

## 3. Этапы ветки

### Этап 1 — Hygiene & defaults ✅ ЗАВЕРШЁН

**Закрывает:** KI-022 (`PYTHONDONTWRITEBYTECODE=1` во все Dockerfile)
**Тип:** Infra
**Smoke-критерий:** `docker compose build` без ошибок, оба `/health` отвечают `ready`

**Цель:** защитить остальную ветку от `.pyc`-фантомов. Должно быть
сделано **в самом начале**, чтобы все последующие правки скриптов
гарантированно подхватывались без кэша.

**Задачи:**

- Добавить `ENV PYTHONDONTWRITEBYTECODE=1` в:
  - `web.Dockerfile`
  - `services/gbm-seg/Dockerfile`
  - (`services/ms-seg/Dockerfile` — уже есть, проверить)
- Rebuild всех контейнеров.
- Проверить `/health` обоих сервисов.

**Коммиты:**

1. `8a830e9` `Add ENV PYTHONDONTWRITEBYTECODE=1`

**Реализация:**

Найдено три Dockerfile: `web.Dockerfile`, `services/gbm-seg/Dockerfile`,
`services/ms-seg/Dockerfile`. В ms-seg обе нужные переменные
(`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`) уже были; в двух
других — отсутствовали.

Решение по ходу: добавить **обе** переменные (а не только
`PYTHONDONTWRITEBYTECODE`, как требовал минимальный scope KI-022),
чтобы все три Dockerfile стали однородными.
`PYTHONUNBUFFERED=1` — стандартная практика для контейнеров: `print()`
и `logging` сразу попадают в `docker logs` без буферизации.

В `web.Dockerfile` переменные помещены сразу после `FROM` (до
`USER root` и любых `RUN`), чтобы они действовали на все слои образа.

Smoke: `docker compose build` + оба `/health` отвечают `ready` + один
прогон pipeline без регрессий.

---

### Этап 2 — UX-блокеры для MS ✅ ЗАВЕРШЁН

**Закрывает:** KI-013 (вкладка валидации не показывает MS-датасет), KI-014 (Slicer не загружает маску для multi-patient/multi-session runs)
**Тип:** Backend/Frontend + Slicer plugin
**Приоритет:** высокий
**Smoke-критерий:** прогон MS-кейса виден в истории UI; Slicer plugin корректно подгружает маску для MS-кейса

**Цель:** MS становится first-class citizen в пользовательских
интерфейсах.

**Задачи:**

- **KI-013** — диагностика и фикс вкладки валидации.
- **KI-014** — Slicer plugin под новый layout output.

**Коммиты:**

1. `1f5a370` `fix(frontend): add lesion-type selector in validation panel` (KI-013)
2. `7f11f2d` `fix(slicer): select correct patient record by entity_id` (KI-014)

**Реализация:**

По ходу диагностики обе формулировки issue были существенно
переопределены — описываю по-новому.

**KI-013.** Изначально предполагалось, что «история запусков не
показывает MS». Уточнение от пользователя: история работает корректно,
проблема в **вкладке Валидация**, которая видит только глио-датасет.
Корень — во фронте: `ValidationPanel.jsx::loadLesionTypes` брал первый
`lesion_type` с `dataset_id` и UI-переключателя датасетов не имел.
Backend `get_lesion_types()` уже возвращал оба типа корректно. Фикс —
только во фронте: добавлен Antd `<Select>` в `extra` карточки. Решение
по `Select` (а не `Radio.Group`) — масштабируется при добавлении
третьего lesion_type (метастазы, Этап 8 ROADMAP).

**KI-014.** Изначально предполагалось, что Slicer не загружает маску
из-за подпапки `{lesion_type}/`. Уточнение по ходу диагностики —
проблема глубже:

- Slicer-агент сам путей не строит, он принимает `mask_path` готовым.
- `/api/slicer/open/{run_id}` строил `mask_path` через
  `find_by_run_id(run_id)` → `records[0]`, всегда выбирая первого
  пациента в run'е независимо от того, кого выбрал пользователь в UI.
- На single-session glio-кейсах (UPENN-GBM) баг был невидим, поскольку
  единственный пациент = единственная запись.
- На MS-кейсе с двумя сессиями одного пациента (ses-001 без масок,
  ses-002 с маской) баг сразу проявился: клик на ses-002 загружал
  данные ses-001 → пустую `mask_history` → Slicer без маски.
- Тот же баг был и на multi-patient glio-run'ах, просто пользователь
  на него не натыкался.

Архитектурное решение, принятое по ходу: эндпоинт получает опциональный
query-параметр `entity_id`. Если передан — выбирается соответствующая
запись. Если не передан и в run'е больше одной записи — 400 Bad Request
(а не fallback на `records[0]`). Single-record runs сохраняют обратную
совместимость. Фронт `openInSlicer` пробрасывает `entityId` из
`ValidationActions`, который уже доступен там как prop.

**Дополнительный пункт, выявленный по ходу:** молчаливая Каппа-дедупликация
делает UX путаным (повторные run'ы не имеют валидационной панели,
поскольку Каппа корректно отказывается принимать дубликаты).
Не баг, а UX-improvement — заведён как KI-026 в `feat/prod-readiness`.

**Гипотеза про цветовую легенду MS, отвергнутая:** изначально
предполагалось, что маска MS не отображается в Slicer из-за
несовпадения color table (4 класса у глио vs 1 у MS). Гипотеза
неверна — root cause был чисто в выборе записи, маска не доходила до
агента вообще. Если проблемы с легендой реально появятся при
полноценной отладке MS-масок — это будет уже отдельный пункт для
`feat/lesion-type-aware-pipeline`.

Smoke: прогон обоих кейсов из `mask_versions` (glio `111acc31`/sub-003,
MS `9ce28980`/ses-002) — Slicer корректно загружает соответствующие
маски.

---

### Этап 3 — Deep review of Stage 01 (`01_reorganize_folders.py`)

**Закрывает:** KI-002, KI-003, KI-006 (частично, в нужном месте), плюс
ряд находок диагностики
**Тип:** Mix — косметика + bug fixes + DRY-рефакторинг
**Приоритет:** средний
**Smoke-критерий:** прогон обоих типов кейсов в **обоих** режимах (sequential, parallel) — `py_compile` + видимо корректный JSON `dataset_mapping.json` + успешная completeness check

**Цель:** систематически пройти по `01_reorganize_folders.py`,
закрыть известные пункты KNOWN_ISSUES и обнаруженные в ходе диагностики
проблемы. Скрипт — точка входа pipeline, его стабильность критична для
всех последующих stages.

**Контекст находок:** в ходе анализа выявлено что sequential и parallel
режимы дублируют логику обработки пациента (~200 строк кода)
и расходятся по составу полей в `mapping_data['patients'][...]`. Это
самостоятельное архитектурное замечание, оформляется как отдельный
коммит-рефакторинг внутри этапа (см. ниже).

**Задачи:**

1. **Мелочи / артефакты мерджа** — двойной `parser.parse_args()`,
   дубль `DatasetScanner(logger)` в `run_sequential`/`run_parallel`,
   тройная инициализация `all_*`-списков перед batch loop.

2. **KI-003** — модальности `{'t1', 't1c', 't2', 't2fl'}`
   дублированы в трёх местах (`process_single_patient`,
   `run_sequential`, `bids_suffix_map` в `copy_series`). Вынести в
   единый источник — module-level constant или брать из
   `CompletenessChecker.LESION_TYPE_MODALITIES`. **Расширено по ходу
   анализа:** ранее KI-003 фиксировал два места, теперь добавляется третье.

3. **KI-002** — `print_summary` хардкодит 4 модальности (инициализация
   `modality_counts`/`modality_slices` и цикл вывода). Сделать
   lesion-type-aware: принимать `lesion_type`, использовать
   `LESION_TYPE_MODALITIES`.

4. **`metadata_saved` в summary** — счётчик `FileOrganizer.metadata_saved`
   инкрементируется, но нигде не выводится. Добавить строку в
   `print_summary`.

5. **Рефакторинг общей логики обработки пациента** — выделить
   `_process_one_patient_core(...)` и звать её из обоих режимов.
   Унифицировать структуру `patient_data` (с полями
   `duplicates_removed`, `files_copied`, `validation_failed` всегда).

6. **Симметрия sequential/parallel** — `monitor.stop()` вызывается в
   parallel, но не в sequential; `id_mapper` через `restore_id_mapper`
   восстанавливается централизованно в `main`, проверить прокидывание
   в оба режима без расхождений.

7. **Failed patients в parallel-режиме** — сейчас `process_single_patient`
   при exception возвращает `None`, и такие пациенты теряются. В
   `ExperimentMetrics` зашит `failed=0`. Исправить: возвращать структуру
   `{"status": ..., ...}`, считать `successful`/`failed`, отражать в
   benchmark-метриках.

**KI-001 не входит в этап.** По решению пользователя
([_detect_contrast](подробности в KNOWN_ISSUES, KI-001)) перенесён в
`feat/lesion-type-aware-pipeline` вместе с KI-027 (скоринг выбора серии).
Контрастный FLAIR — реальный клинический случай, и его адекватная
обработка требует не косметической правки логирования, а полноценного
скоринга confidence — вместе с проработкой логики `_select_best_series`.

**Коммиты:**

1. `chore(stage01): remove duplicated parser and scanner instantiations`
   — задача 1.
2. `refactor(stage01): centralize modality whitelist` — KI-003 (задача 2).
3. `fix(stage01): make print_summary lesion-type-aware` — KI-002 (задача 3).
4. `fix(stage01): surface metadata_saved counter in summary` — задача 4.
5. `refactor(stage01): unify per-patient processing for sequential/parallel`
   — задача 5. Самый большой коммит, обсуждается отдельно перед
   написанием кода (общая сигнатура функции, структура возврата).
6. `fix(stage01): symmetrize monitor and id_mapper between sequential and parallel`
   — задача 6.
7. `fix(stage01): account for failed patients in parallel mode and benchmarks`
   — задача 7.

**Открытые вопросы:**

- Точная сигнатура общей функции `_process_one_patient_core` —
  обсуждаем до Коммита 5.
- Объём отчётности по failed patients — нужны ли все детали ошибок в
  JSON-отчёте, или достаточно счётчиков? Обсуждаем до Коммита 7.

---

### Этап 4 — Validation correctness

**Закрывает:** KI-007 (`_scan_output_structure` не учитывает `{lesion_type}/`)
**Тип:** Bug fix
**Приоритет:** средний
**Smoke-критерий:** прогон обоих типов кейсов (glio + MS) — в логах валидации видны `Found N subjects, Success rate: 100%`

**Цель:** validation корректно сканирует output после изменения layout
в MAS-рефакторинге.

**Задачи:**

- Найти `_scan_output_structure` в pipeline.
- Обновить логику сканирования под `{lesion_type}/` подпапку.
- Smoke на обоих типах кейсов.

**Коммиты:**

1. `fix(validation): account for lesion_type subfolder in output scan` (KI-007)

**Открытые вопросы:**

- Должен ли `_scan_output_structure` искать **все** `{lesion_type}/`
  подпапки (если их несколько) или только одну, переданную через
  параметр? Обсудить до кода.

---

### Этап 5 — Services hygiene (DRY)

**Закрывает:** KI-008 (torch.load patch dup), KI-009 (дрейф версий gbm-seg/ms-seg)
**Тип:** DRY refactor
**Приоритет:** низкий (KI-008) + средний (KI-009)
**Smoke-критерий:** оба сервиса собираются и поднимаются, `/health` отвечает `ready`, прогон одного кейса каждого типа

**Цель:** убрать дубликаты в `services/`, унифицировать версии
зависимостей.

**Задачи:**

- **KI-008:** вынести torch.load monkey-patch в
  `services/common/torch_compat.py` с функцией
  `enable_legacy_checkpoint_loading()`. Заменить в обоих сервисах на
  вызов этой функции.
- **KI-009:** создать `services/common/requirements-base.txt` с общими
  пакетами (Quart, Hypercorn, Flask, etc.). В `requirements.txt`
  обоих сервисов оставить только service-specific dependencies +
  `-r ../common/requirements-base.txt`.

**Коммиты:**

1. `refactor(services/common): extract torch_compat patch to shared module` (KI-008)
2. `refactor(services): introduce requirements-base.txt for common deps` (KI-009)

**Открытые вопросы:**

- KI-009 в KNOWN_ISSUES упоминает альтернативу — общий base Docker image
  `brain-lesion-base:latest`. Обсудить: пин зависимостей через
  `requirements-base.txt` (легче) vs base image (правильнее, но дольше).
  В рамках чистки берём более лёгкий путь, base image — кандидат на
  `feat/prod-readiness`.

---

### Этап 6 — Operational footgun

**Закрывает:** KI-015 (ручной vs автосозданный Kappa-датасет)
**Тип:** Defensive logging
**Приоритет:** низкий (operational)
**Smoke-критерий:** unit-проверка warning в логах при наличии пустого датасета-дубликата

**Цель:** защитить операторов от повторения footgun'а с ручным/автосозданным
Kappa-датасетом.

**Задачи:**

- В `KappaUploader` (или соответствующем модуле) при создании нового
  датасета — `WARNING`-лог, если в Каппе уже есть датасет с тем же
  `lesion_type`, но он пустой. Не блокировать выполнение.

**Коммиты:**

1. `feat(kappa): warn on empty duplicate dataset before auto-create` (KI-015)

---

### Этап 7 — Lesion-types config prep

**Закрывает:** KI-016 (config-driven lesion_types) — **частично**
**Тип:** Infrastructure prep
**Приоритет:** средний
**Smoke-критерий:** `configs/lesion_types.yaml` валидный YAML, документация на формат, **никаких изменений в коде**

**Цель:** подготовить точку входа для следующей ветки
`feat/lesion-type-aware-pipeline` без рисков в текущей.

**Задачи:**

- Создать `configs/lesion_types.yaml` с текущими данными
  (`glioblastoma`, `multiple_sclerosis`).
- Структура — как в KI-016: `required_modalities`, `reference_modality`,
  `reports`.
- **Не менять код**, который сейчас читает `LESION_TYPE_MODALITIES` из
  констант — это работа следующей ветки.
- Документация формата (краткий комментарий в YAML + упоминание в SPEC).

**Коммиты:**

1. `feat(config): introduce lesion_types.yaml schema (not yet wired)` (KI-016 partial)

---

### Этап 8 — Обход остальных pipeline stages

**Закрывает:** «пройтись чисткой по всем этапам» из стартового сообщения,
включая KI-006
**Тип:** Survey + targeted fixes
**Приоритет:** N/A (фоновая задача)
**Smoke-критерий:** документирован проход по stages 02–08, найденные мелкие пункты либо исправлены, либо добавлены в KNOWN_ISSUES.md

**Цель:** аналогично Этапу 3, но для остальных stages. Stage 01 уже
сделан тщательно — здесь проходим по `02_*` ... `08_*` с тем же подходом.

**Задачи:**

- Пройти по каждому `scripts/0X_*.py` (X=2..8) и проверить:
  - Хардкоды модальностей, lesion_type, путей.
  - Дубликаты констант между sequential/parallel реализациями.
  - Логи и summary, которые могут быть неточны для не-глио.
  - Артефакты мерджей (дубли вызовов, мёртвый код).
- Известный пункт: **KI-006** — дубликат `PROCESSING SUMMARY` в stage 06.
- Найденные мелкие баги — фиксим в этом этапе с отдельными коммитами.
- Найденные крупные вопросы — добавляем в KNOWN_ISSUES.md как новые
  KI-NNN, не пытаемся решить в этой ветке.

**Коммиты:**

- `fix(stage06): remove duplicate log_summary() call` (KI-006)
- N коммитов по найденным мелочам (формат: `fix(stageXX): ...`)
- 1 коммит: `docs(known-issues): add findings from stages survey` (если
  найдено что-то крупное)

**Открытые вопросы:**

- Объём этого этапа заранее непредсказуем. Договоримся, что если
  по ходу обхода обнаружим что-то крупное (например, structural bug) —
  обсуждаем отдельно, возможно выносим в отдельный этап.

---

### Этап 9 — Финализация и merge

**Тип:** QA + docs + merge
**Smoke-критерий:** end-to-end на обоих кейсах (glio + MS), оба зелёные

**Цель:** проверка целостности ветки, обновление документации, merge
в main.

**Задачи:**

- End-to-end smoke test на глио-кейсе (UPENN-GBM).
- End-to-end smoke test на MS-кейсе (P000915).
- Обновить KNOWN_ISSUES.md — пометить закрытые пункты, оставить
  открытыми те, что не сделаны.
- Обновить SPEC.md (этот документ) — финальная версия 1.0 с заметками
  по каждому этапу.
- Обновить главный `docs/SPEC.md` (архитектурный) — если требуется
  отразить какие-то изменения.
- Merge `chore/post-mas-cleanup` → `main`.

**Коммиты:**

1. `test: end-to-end smoke on glioblastoma case`
2. `test: end-to-end smoke on multiple sclerosis case`
3. `docs(known-issues): mark cleared items, refresh statuses`
4. `docs(spec): finalize post-MAS cleanup phase notes`
5. (merge commit) `Merge branch 'chore/post-mas-cleanup' into main`

---

## 4. Smoke-тесты: матрица

| Этап | Что меняется | Smoke-минимум |
|------|--------------|---------------|
| 1 ✅ | Dockerfiles | `docker compose build` + оба `/health` |
| 2 ✅ | Backend/frontend/Slicer | прогон MS-кейса + проверка в UI/Slicer |
| 3 | Stage 01 (pipeline entry) | `py_compile` + прогон обоих типов в обоих режимах (sequential, parallel) |
| 4 | Validation logic | прогон обоих типов, проверка логов валидации |
| 5 | Services common code | rebuild + `/health` обоих + по одному кейсу каждого типа |
| 6 | KappaUploader (warning) | unit-проверка warning в логах |
| 7 | Configs only | `python3 -c 'import yaml; yaml.safe_load(open(...))'` |
| 8 | Зависит от находок | определяется по ходу |
| 9 | Финал | полный end-to-end на обоих типах |

---

## 5. Что НЕ входит в эту ветку

- **Архитектурные изменения контракта.** Любые правки `/predict`,
  схемы БД, layout output, и т.д. — не трогаем.
- **KI-001** — перенесён в `feat/lesion-type-aware-pipeline` вместе с
  KI-027 (контрастный FLAIR требует логики скоринга, а не косметики).
- **KI-016 в полной реализации.** Только создание YAML без миграции
  кода. Полная миграция — в `feat/lesion-type-aware-pipeline`.
- **KI-004, KI-005, KI-019** — MS adaptation работы, отложены в
  `feat/lesion-type-aware-pipeline`.
- **KI-010, KI-011, KI-012, KI-017, KI-018, KI-020, KI-021, KI-023,
  KI-024, KI-025, KI-026, KI-027, KI-028, KI-029, KI-030, KI-031,
  KI-032** — future work, отложены в соответствующие ветки
  (см. ROADMAP.md и KNOWN_ISSUES.md).

---

## 6. Принципы фиксации прогресса

После каждого закрытого этапа:

1. KNOWN_ISSUES.md — отметка «закрыто в `chore/post-mas-cleanup`,
   коммит `<sha>`».
2. SPEC.md (этот документ) — секция «Реализация» в конце этапа: что
   сделано, какие архитектурные решения приняты по ходу, какие критерии
   приёмки выполнены.
3. Коммит документации отдельно от коммита кода (не мешаем).

---

## 7. Глоссарий

Терминология та же, что в основном SPEC.md (`feat/mas-refactor`).
Новых сущностей в этой ветке не вводится.

---

*Документ живой. Обновляется после каждого этапа.*