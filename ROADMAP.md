# Project Roadmap

Крупноблочный план развития сервиса. Каждый этап — отдельная ветка
и отдельный чат с ассистентом для сохранения контекста.

Текущее состояние: завершён MAS-рефакторинг (Этапы 1-3),
готовится merge в main.

---

## ✅ Этапы 1–3: MAS-инфраструктура (завершено)

**Ветка:** `feat/mas-refactor`
**Период:** май 2026
**Статус:** ✅ Готово к merge в main

**Этап 1 — `ServiceBase` для gbm-seg.** Унификация контракта между
сервисами. Извлечение общей инфраструктуры (HTTP, queue, GPU pool)
в `services/common/service_base.py`. gbm-seg переписан как наследник
`ServiceBase`.

**Этап 2 — Путь-ориентированный контракт `/predict`.** Замена legacy
multipart upload на JSON-payload с путями файлов через shared volume.
Подпапки `{lesion_type}/` для multi-model coexistence. Удаление legacy
endpoints (−503 строки).

**Этап 3 — ms-seg сервис.** Второй inference-сервис (МС-сегментация
через nnUNet v2 + CATMIL fork). Реестр сервисов (`configs/services.yaml`).
Параметризация `lesion_type` через всю цепочку: frontend → backend →
orchestrator → pipeline stages. End-to-end verified на реальном
MS-клиническом кейсе.

---

## ✅ Этап 4: Post-MAS Cleanup

**Ветка:** `chore/post-mas-cleanup`
**Зависит от:** Этапов 1-3
**Содержание:** см. KNOWN_ISSUES.md (приоритеты низкий/средний)

Чистка накопившегося тех. долга после рефакторинга:

- Дубликаты констант в pipeline scripts
- `_scan_output_structure` под новый layout
- Косметические дубликаты логов (`PROCESSING SUMMARY`)
- `torch.load` patch и `requirements-base.txt` в `services/common/`
- `PYTHONDONTWRITEBYTECODE=1` во все Dockerfile
- Slicer integration после смены контракта (KI-014)
- История запусков в UI для MS-кейсов (KI-013)
- Sync ручного vs автосозданного Kappa-датасета (KI-015)
- Косметика: `print_summary`, `_detect_contrast` для FLAIR

**Цель:** стабилизация baseline перед следующими большими работами.

---

## ✅ Этап 5: MS Adaptation

**Ветка:** `feat/lesion-type-aware-pipeline`
**Зависит от:** Этапа 4

Сейчас MS-pipeline работает "по краям" — мы прогнали кейс end-to-end,
но не валидировали все этапы детально и не оптимизировали под не-глио
семантику. Этот этап делает MS-pipeline first-class citizen.

**Содержание:**

- **Stage 04 (QA):** lesion-type-aware required modalities
  (см. KI-004)
- **Stage 05 (preprocessing):** skip T1c registration для МС
  (экономит ~25% времени, см. KI-005)
- **Клинический отчёт:** адаптировать метрики под МС (количество
  очагов, локализация, longitudinal analysis между сессиями).
  Требует обсуждения с клиницистами какие метрики критичны.
- **configs/lesion_types.yaml:** config-driven описание модальностей
  и параметров для каждого типа поражения (см. KI-016)
- **Frontend:** корректное отображение МС везде (визуализация маски,
  фильтры, выпадающие списки)

**Цель:** MS равноправен с глио во всех точках pipeline и UI.

---

## □ Этап 6: Production Readiness v1

**Ветка:** `feat/prod-readiness`
**Зависит от:** Этапов 4, 5

Подготовка к развёртыванию в реальной клинической среде.

**Содержание:**

- Покрытие тестами (unit + integration для критических путей)
- Документация:
  - Полный README
  - Architecture overview
  - API documentation (OpenAPI/Swagger)
  - Deployment guide
  - Developer onboarding
- CI/CD pipeline (GitHub Actions): автотест + автодеплой
- Логирование и мониторинг (Prometheus + Grafana, structured logs)
- Безопасность: CORS, rate limiting, auth review
- Backup strategy для БД, runtime configs, mapping files
- Performance benchmarking baseline
- Тестовый прогон на больших датасетах (50+ пациентов)
- Slicer integration final test

**Цель:** v1.0 release. Сервис готов к развёртыванию в боевой среде
для двух типов поражений (глио + МС).

---

## □ MILESTONE: v1.0 в продакшен

Глио + МС работают стабильно, документация готова, прод развёрнут.

---

## □ Этап 7: MAS Implementation (research)

**Ветка:** `feat/mas-coordinator`
**Зависит от:** v1.0 в продакшене
**Тип:** Research-level задача

Это **переход от MAS-ready инфраструктуры к настоящему MAS**.

Инфраструктура, заложенная в Этапах 1-3, поддерживает многосервисную
архитектуру с диспетчеризацией по lesion_type. Теперь добавляем
реальную агентскую логику.

**Содержание:**

- **Coordinator-агент:** LLM-based principal, который анализирует кейс
  и решает какие модели использовать. Использует Kappa, manifest'ы
  сервисов, описание кейса.
- **Quality-агент:** оценивает confidence результата сегментации.
- **Resource-агент:** управляет GPU pool, очередью задач.
- **Реализация cascade/fallback/ensemble режимов** (твоя идея):
  - Cascade: primary → confidence низкая → fallback model
  - Fallback: primary → ошибка → автопереключение
  - Ensemble: N моделей параллельно → консенсус
- **Contract Net Protocol** для агент-сервис коммуникации.
- **Multi-trainer dispatch в ms-seg:** использование всех 11 trainer'ов
  через `options.trainer`.
- **BDI-агенты** (опционально, если research-направление это требует).
- **Benchmark для MAS-метрик:** межагентная коммуникация, latency,
  consensus time.

**Связь с PhD-research:**
Этот этап — основа двух research-направлений:
1. **Функциональные диагностические агенты** (cascade/ensemble)
2. **BDI/Contract Net для ресурсов**

**Цель:** v2.0 с настоящей MAS-архитектурой.

---

## □ MILESTONE: v2.0 — настоящий MAS

---

## □ Этап 8: Метастазы

**Ветка:** `feat/mets-seg`
**Зависит от:** v1.0 (можно параллельно с Этапом 7)

Добавление третьего типа поражения — метастазы в мозге.

**Содержание:**

- Создание `services/mets-seg/` по аналогии с ms-seg
- Веса метастаз → отдельный `Dataset_XXX_BrainMets`
- Manifest и registry entry для `brain_metastases`
- Адаптация pipeline под особенности метастаз:
  - Множественные мелкие очаги (отличается от единственного глио)
  - Возможные специфичные модальности
  - Особенности preprocessing

**Цель:** v1.1 с тремя типами поражений.

---

## □ Этап 9: Multi-center datasets

**Ветка:** `feat/multi-center-datasets`
**Зависит от:** v1.1

Расширение маппинга Каппы с `lesion_type → dataset_id` на
`(lesion_type × center) → dataset_id`.

**Примеры датасетов:**
- МТЦ Глиобластомы
- ФЦН Глиобластомы
- МТЦ РС
- ФЦН РС
- МТЦ Метастазы
- ФЦН Метастазы

**Содержание:**

- Расширение `kappa_datasets.yaml` под двумерный ключ
- Frontend: выбор центра при загрузке кейса
- Backend: маршрутизация uploads по центру
- Миграция существующих датасетов

---

## □ Этап 10: Advanced MAS

**Ветка:** `feat/mas-advanced`
**Зависит от:** v2.0

Развитие MAS-архитектуры для production-уровня:

- **Distributed deployment:**
  - HTTP file transfer mode (`POST /predict` multipart +
    `GET /files/{id}`)
  - Контейнерные пути или URI-адресация
  - SSH-туннели заработают с новым контрактом
- **GPU coordinator:** интеллектуальное распределение задач между
  GPU (когда их несколько)
- **Inter-agent communication framework**
- **Production-grade MAS observability**

**Цель:** Cube/Barguzin профили снова рабочие, готовность к
heterogeneous distributed deployment.

---

## Принципы работы

**Одна ветка = один чат.** В начале каждого нового этапа создаём
новый чат с ассистентом и передаём контекст (стартовое сообщение
готовится заранее).

**KNOWN_ISSUES.md** — живой документ. Новые баги, найденные в любой
ветке, фиксируются туда. После закрытия — помечаются как закрытые
или удаляются.

**SPEC.md** — отражает текущую архитектуру (не roadmap). Обновляется
по мере завершения этапов.

**Атомарные коммиты с conventional commits.** Каждый коммит — одна
завершённая мысль, проверяется py_compile + end-to-end test перед
push.

---

## Текущий фокус

✅ **Завершено:** `feat/lesion-type-aware-pipeline` → merge в main (июнь 2026).
MS-pipeline first-class citizen: lesion-type-aware stages 01/04/05/06/08,
МС-отчёт (нагрузка, hover, лонгитюд), RANO-фикс, Kappa-only валидация,
Slicer-фиксы, viewer-полировка.

➡️ **СЛЕДУЮЩИЙ ЭТАП:** `feat/prod-readiness` (Этап 6) — Production Readiness v1.
Параллельно можно рассмотреть `feat/ms-clinical-metrics` (МС локализация
по Макдональду + детекция новых очагов) как self-contained research-ветку.

---

*Последнее обновление: завершение `feat/lesion-type-aware-pipeline`, июнь 2026.*
