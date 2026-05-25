# Техническое задание: post-MAS cleanup

**Проект:** Web-сервис AI-сегментации поражений головного мозга по МРТ
**Версия ТЗ:** 0.1 (стартовая, перед началом работ)
**Дата:** 25 мая 2026
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
на предмет косметических артефактов рефакторинга.

**Принципиальное ограничение:** в этой ветке **нет архитектурных
изменений**. Только баг-фиксы, DRY-рефакторинги внутри существующих
модулей, документация. Все изменения, требующие пересмотра контракта
или схемы данных, отложены в следующие ветки.

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

### Этап 1 — Hygiene & defaults

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

1. `chore(infra): add PYTHONDONTWRITEBYTECODE=1 to all Dockerfiles`

---

### Этап 2 — UX-блокеры для MS

**Закрывает:** KI-013 (история запусков не показывает MS), KI-014 (Slicer не грузит маску для МС)
**Тип:** Backend/Frontend + Slicer plugin
**Приоритет:** высокий
**Smoke-критерий:** прогон MS-кейса виден в истории UI; Slicer plugin корректно подгружает маску для MS-кейса

**Цель:** MS становится first-class citizen в пользовательских
интерфейсах. Оба пункта — последствия того, что MAS-рефакторинг изменил
структуру output (`segmentation/{lesion_type}/mask.nii.gz`), а UI и
Slicer plugin не были адаптированы.

**Задачи:**

- **KI-013** — диагностика и фикс истории запусков:
  - `grep -rn "glioblastoma\|lesion_type" backend/ frontend/src/ | grep -v test`
  - найти хардкод фильтра (SQL/REST/frontend filter)
  - убрать хардкод, проверить что оба типа кейсов отображаются
- **KI-014** — Slicer plugin под новый layout output:
  - найти ожидаемые пути в Slicer plugin
  - обновить под `{lesion_type}/mask.nii.gz`
  - проверить на MS-кейсе

**Коммиты:**

1. `fix(backend|frontend): show all lesion types in run history` (KI-013)
2. `fix(slicer): adapt mask path resolution to lesion_type subfolder` (KI-014)

**Открытые вопросы для обсуждения до кода:**

- Где именно живёт хардкод KI-013 (backend SQL vs frontend filter) —
  выяснится после grep, влияет на структуру коммита.
- Какой формат пути ожидает Slicer plugin сейчас — может потребовать
  отдельный sub-step «диагностика» перед фиксом.

---

### Этап 3 — Pipeline scripts cleanup (косметика)

**Закрывает:** KI-001 (`_detect_contrast` для FLAIR), KI-002 (`print_summary` хардкод 4 модальностей), KI-003 (дубликат фильтра модальностей), KI-006 (дубликат `PROCESSING SUMMARY`)
**Тип:** Cosmetic
**Приоритет:** низкий
**Smoke-критерий:** `py_compile` всех изменённых файлов + один прогон pipeline (любой тип)

**Цель:** убрать косметические артефакты в pipeline scripts, выявленные
во время прогонов MS-кейса.

**Задачи:**

- **KI-001:** в `scripts/01_*.py` уточнить условия `_detect_contrast` —
  для FLAIR contrast=True должен либо игнорироваться, либо понижать
  confidence.
- **KI-002:** в `scripts/01_*.py` `print_summary` сделать список
  модальностей зависимым от `lesion_type` (по образцу `CompletenessChecker`).
- **KI-003:** в `scripts/01_*.py` вынести константу `{'t1', 't1c', 't2', 't2fl'}`
  в module-level constant (или взять из существующего источника
  `LESION_TYPE_MODALITIES`).
- **KI-006:** в `scripts/06_*.py` убрать дубликат вызова `stats.log_summary()`
  (`_process_sessions_async` + `run()` оба вызывают).

**Коммиты:**

1. `fix(stage01): refine _detect_contrast for FLAIR modality` (KI-001)
2. `fix(stage01): make print_summary lesion-type-aware` (KI-002)
3. `refactor(stage01): extract modality filter to module-level constant` (KI-003)
4. `fix(stage06): remove duplicate log_summary() call` (KI-006)

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
- Smoke на обоих типах кейсов (с MS и без — поведение должно быть
  идентичным).

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

**Открытые вопросы:**

- Если найден непустой дубликат с тем же `lesion_type` — `WARNING`
  тоже? Или это нормальная ситуация (например, multi-center в будущем)?
  Обсудить до кода.

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

**Открытые вопросы:**

- Точный набор полей в YAML. Сейчас в KI-016 предложены
  `required_modalities`, `reference_modality`, `reports`. Возможно
  добавить `display_name`, `description` для UI. Обсудить до создания
  файла.

---

### Этап 8 — Тщательный обход всех pipeline stages

**Закрывает:** «пройтись чисткой по всем этапам» из стартового сообщения
**Тип:** Survey + targeted fixes
**Приоритет:** N/A (фоновая задача)
**Smoke-критерий:** документирован проход по stages 01–08, найденные мелкие пункты либо исправлены, либо добавлены в KNOWN_ISSUES.md

**Цель:** не оставить незамеченных артефактов рефакторинга. Systematic
review stages 01–08 на предмет того, что не покрыто предыдущими
этапами.

**Задачи:**

- Пройти по каждому `scripts/0X_*.py` и проверить:
  - Хардкоды модальностей (помимо уже найденных в KI-001..006).
  - Хардкоды `lesion_type` или путей.
  - Дубликаты констант между sequential/parallel реализациями.
  - Логи и summary, которые могут быть неточны для не-глио.
- Найденные мелкие баги — фиксим в этом этапе с отдельными коммитами.
- Найденные крупные вопросы — добавляем в KNOWN_ISSUES.md как новые
  KI-NNN, не пытаемся решить в этой ветке.

**Коммиты:**

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
  открытыми те, что не сделаны (`feat/lesion-type-aware-pipeline`,
  `feat/mas-coordinator` и т.д.).
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
| 1 | Dockerfiles | `docker compose build` + оба `/health` |
| 2 | Backend/frontend/Slicer | прогон MS-кейса + проверка в UI/Slicer |
| 3 | Pipeline scripts (косметика) | `py_compile` + любой прогон |
| 4 | Validation logic | прогон обоих типов, проверка логов валидации |
| 5 | Services common code | rebuild + `/health` обоих + по одному кейсу каждого типа |
| 6 | KappaUploader (warning) | unit-проверка warning в логах |
| 7 | Configs only | `python3 -c 'import yaml; yaml.safe_load(open(...))'` |
| 8 | Зависит от находок | определяется по ходу |
| 9 | Финал | полный end-to-end на обоих типах |

---

## 5. Что НЕ входит в эту ветку

- **Архитектурные изменения.** Любые правки контракта `/predict`,
  схемы БД, layout output, и т.д.
- **KI-016 в полной реализации.** Только создание YAML без миграции
  кода. Полная миграция — в `feat/lesion-type-aware-pipeline`.
- **KI-004, KI-005, KI-019** — MS adaptation работы, отложены в
  `feat/lesion-type-aware-pipeline`.
- **KI-010, KI-011, KI-012, KI-017, KI-018, KI-020, KI-021, KI-023,
  KI-024, KI-025** — future work, отложены в соответствующие ветки
  (см. ROADMAP.md).

---

## 6. Принципы фиксации прогресса

После каждого закрытого этапа:

1. KNOWN_ISSUES.md — отметка «закрыто в `chore/post-mas-cleanup`,
   коммит `<sha>`».
2. SPEC.md (этот документ) — секция «Реализация» в конце этапа, как в
   feat/mas-refactor SPEC: что сделано, какие архитектурные решения
   приняты по ходу, какие критерии приёмки выполнены.
3. Коммит документации отдельно от коммита кода (не мешаем).

---

## 7. Глоссарий

Терминология та же, что в основном SPEC.md (`feat/mas-refactor`).
Новых сущностей в этой ветке не вводится.

---

*Документ живой. Обновляется после каждого этапа.*