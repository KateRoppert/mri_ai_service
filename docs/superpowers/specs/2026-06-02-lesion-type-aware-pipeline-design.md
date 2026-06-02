# Дизайн: lesion-type-aware pipeline

**Ветка:** `feat/lesion-type-aware-pipeline`
**Дата:** 2026-06-02
**Автор:** Kate Roppert
**Закрывает:** KI-004, KI-005, KI-016, KI-019

---

## 1. Контекст

MAS-инфраструктура (Этапы 1–3) и post-MAS cleanup (`chore/post-mas-cleanup`) завершены
и смёрджены в `main`. МС-пайплайн проходит end-to-end без блокеров, но ряд этапов
не знает о `lesion_type` или содержит глио-ориентированные допущения, которые приводят
к неоптимальной работе для МС: лишние итерации по T1c, некорректные фолбэки,
клинический отчёт только под глио.

Эта ветка делает МС полноправным типом поражения на всех этапах pipeline и фронте,
а также закрывает частично открытый KI-016 (lesion_types.yaml не подключён к коду).

**Выходит за рамки ветки (отложено):**
- KI-001 / KI-027 — confidence-скоринг в ModalityDetector и улучшенная эвристика выбора серий
- KI-010 — координация GPU между сервисами
- Всё с тегами `feat/prod-readiness` и `feat/mas-coordinator` в KNOWN_ISSUES.md

---

## 2. Подход

Работаем поэтапно в порядке pipeline (01 → 03 → 04 → 05 → 06–08 → backend → frontend).
Каждый этап тестируется независимо (остальные отключаются в конфиге оркестратора).
После каждого исправления — smoke-тест на МС-кейсе.

---

## 3. Раздел 1 — Миграция конфига (KI-016)

### Проблема

`lesion_types.yaml` создан в `chore/post-mas-cleanup`, но ни один этап pipeline его
не читает. У каждого этапа свои захардкоженные константы:
- `CompletenessChecker.LESION_TYPE_MODALITIES` в Stage 01
- `modalities: [t1c, t1, t2, t2fl]` по умолчанию в Stage 05
- Stage 04 вообще не знает про lesion_type

### Решение

Добавить `load_lesion_type_config(lesion_type: str) -> dict` в `utils/config_loader.py`.
Возвращает полную запись из `lesion_types.yaml` для нужного типа поражения.

```python
def load_lesion_type_config(lesion_type: str) -> dict:
    """
    Читает конфиг типа поражения из configs/lesion_types.yaml.
    Возвращает required_modalities, reference_modality, reports.
    """
```

Все этапы pipeline заменяют свои захардкоженные константы на вызов этой функции.
Единый источник истины — добавление нового типа поражения требует только правки YAML.

---

## 4. Раздел 2 — Адаптация этапов pipeline

### Stage 01 (`01_reorganize_folders.py`) — только подключение KI-016

Уже адаптирован для МС в `chore/post-mas-cleanup`. Единственное изменение: заменить
захардкоженный dict `LESION_TYPE_MODALITIES` в `CompletenessChecker` на
`load_lesion_type_config()`. Поведение не меняется, чистый DRY-рефакторинг.

### Stage 03 (`03_convert_to_nifti.py`) — изменений не нужно

Конвертирует всё, что Stage 01 разложил в BIDS. T1c просто не будет в BIDS-папке
для МС-пациентов — Stage 03 обработает только нужные модальности без каких-либо правок.

### Stage 04 (`04_assess_quality.py`) — KI-004

**Проблема:** нет аргумента `--lesion-type`. Stage 04 обрабатывает NIfTI-файлы которые
реально есть, поэтому на МС не падает, но не имеет явного знания о том, какие
модальности ожидаются для каждой болезни.

**Изменения:**
1. Добавить `--lesion-type` в argparse (по образцу Stage 01, 06, 07, 08).
2. Читать `required_modalities` через `load_lesion_type_config()`.
3. Использовать список модальностей при формировании отчёта о полноте данных.
4. Добавить `--lesion-type` в command builder оркестратора для Stage 04.

### Stage 05 (`05_preprocessing.py`) — KI-005

**Проблема:** два глио-ориентированных допущения:

1. `modalities` по умолчанию `['t1c', 't1', 't2', 't2fl']` (строка 735). Для МС T1c
   не существует. Текущий код тихо пропускает отсутствующие файлы, но T1c остаётся
   в списке обработки и код по нему итерируется зря. Явное исключение T1c экономит
   ~25% времени Stage 05 для МС (только N4 bias correction T1c занимает ~280с).

2. Фолбэк reference modality захардкожен как `'t1c'` в двух местах (строки 303 и 307).
   Это противоречит `preprocessing_config.yaml`, где уже стоит `reference_modality: "t1"`.
   Фолбэк срабатывает только когда шага registration нет в конфиге или у него нет
   параметра `reference_modality` — но если он сработает для МС-кейса без T1c-файлов,
   Stage 05 пропустит субъект с ошибкой `missing_reference_modality_t1c`.

**Изменения:**
1. Добавить `--lesion-type` в argparse.
2. Читать `required_modalities` и `reference_modality` через `load_lesion_type_config()`.
3. Переопределить список `modalities` (из конфига) списком из lesion_types.yaml.
   Для МС: `[t1, t2, t2fl]`; для глио: `[t1c, t1, t2, t2fl]`.
4. Заменить оба фолбэка `'t1c'` на `lt_config['reference_modality']`
   (= `'t1'` для обоих типов согласно `lesion_types.yaml`).
5. Добавить `--lesion-type` в command builder оркестратора для Stage 05.

### Stages 06 / 07 / 08 — только верификация

Уже параметризованы `--lesion-type` из `feat/mas-refactor`. Прогоняем МС-кейс
для подтверждения отсутствия регрессий. Правки только при выявлении конкретной проблемы.

---

## 5. Раздел 3 — Backend для МС-отчёта

### Подсчёт очагов

**Место:** Stage 08 (`08_lobar_localization.py`).

Stage 08 уже читает маску сегментации для лобарного атласа. Добавляем ветку:
если `lesion_type == 'multiple_sclerosis'` — запускаем `scipy.ndimage.label()` на
бинарной маске, считаем связные компоненты (connected components).

Выходной файл: `lesion_stats_report.json` (рядом с существующим `lobar_report.json`):

```json
{
  "patient_id": "sub-P000915",
  "session_id": "ses-002",
  "lesion_count": 14,
  "total_volume_cm3": 3.21,
  "mean_lesion_volume_cm3": 0.23,
  "lesion_volumes_cm3": [0.12, 0.45, 0.08, ...]
}
```

### Новые backend-эндпоинты

`GET /api/reports/lesion-stats/{run_id}` — возвращает содержимое `lesion_stats_report.json`
для каждого пациента/сессии в прогоне. Паттерн аналогичен существующим
`get_volume_reports` / `get_lobar_reports`.

`GET /api/reports/longitudinal/{patient_id}?lesion_type=multiple_sclerosis` — запрос
к `patient_registry` (SQLite): все сессии с данным `patient_id` + `lesion_type`,
сортировка по `scan_date`. Для каждой сессии читается `lesion_stats_report.json` (МС)
или `volume_report.txt` (глио) для получения суммарного объёма. Ответ:

```json
[
  {"session_id": "ses-001", "scan_date": "2022-01-18", "total_volume_cm3": 2.14, "lesion_count": 11},
  {"session_id": "ses-002", "scan_date": "2023-03-25", "total_volume_cm3": 3.21, "lesion_count": 14}
]
```

Показывается в UI только при 2+ сессиях в ответе.

---

## 6. Раздел 4 — Frontend

### `NIfTIViewer.jsx` — динамический colormap

**Проблема:** colormap захардкожен под 4 класса глио (NCR/ED/NET/ET).

**Решение:** принимать prop `lesionType` (уже доступен в `ValidationPanel` из записи
прогона). Строить colormap по типу поражения:
- `multiple_sclerosis` → бинарный: 0=фон, 1=очаг (`#52c41a`)
- `glioblastoma` → существующая 4-классовая палитра (NCR/ED/NET/ET), без изменений

Это проще, чем передавать `outputClasses` из результата инференса (который не хранится
в БД после завершения pipeline), и корректно отражает фиксированную структуру классов
каждой модели.

### `ClinicalReportContent.jsx` — роутинг по lesion_type

**Проблема:** монолитный компонент, полностью заточен под глио (объёмы NCR/ED/NET/ET, CE+/CE−).

**Решение:** добавить prop `lesionType`. Разделить рендер:

```jsx
if (lesionType === 'multiple_sclerosis') {
  return <MsReportSection data={lesionStats} lobarData={lobarReports} />;
}
return <GbmReportSection volumeReports={volumeReports} lobarReports={lobarReports} />;
```

`GbmReportSection` — текущий код, вынесенный в подкомпонент, без изменений поведения.
`MsReportSection` — новый подкомпонент (в том же файле), отображает:
- Суммарный объём очагов (Statistic)
- Количество очагов (Statistic)
- Таблица по долям мозга (существующий lobar_report, уже работает для МС)
- `LongitudinalTimeline` (ниже, только при 2+ сессиях)

### `LongitudinalTimeline` — новый подкомпонент

Рядом с `ClinicalReport.jsx`. При монтировании запрашивает
`GET /api/reports/longitudinal/{patient_id}?lesion_type=...`. Если API вернул менее
2 строк — ничего не отображает. Иначе — простая Ant Design таблица:

| Дата | Сессия | Объём (см³) | Кол-во очагов |
|------|--------|------------|---------------|
| 2022-01-18 | ses-001 | 2.14 | 11 |
| 2023-03-25 | ses-002 | 3.21 | 14 |

При 3+ сессиях добавляется колонка Δ (изменение объёма относительно предыдущей сессии).

---

## 7. Порядок работ

| # | Область | Файлы | Закрывает |
|---|---------|-------|-----------|
| 1 | Config utility | `utils/config_loader.py` | KI-016 |
| 2 | Stage 01 wiring | `scripts/01_reorganize_folders.py` | KI-016 |
| 3 | Stage 04 | `scripts/04_assess_quality.py`, `orchestrator.py` | KI-004 |
| 4 | Stage 05 | `scripts/05_preprocessing.py`, `orchestrator.py` | KI-005 |
| 5 | Stages 06–08 smoke | только проверка | — |
| 6 | Stage 08 lesion count | `scripts/08_lobar_localization.py` | KI-019 (частично) |
| 7 | Backend эндпоинты | `backend/app.py` | KI-019 |
| 8 | NIfTIViewer colormap | `frontend/src/components/NIfTIViewer.jsx` | SPEC §5 |
| 9 | ClinicalReportContent routing | `frontend/src/components/ClinicalReportContent.jsx` | KI-019 |
| 10 | LongitudinalTimeline | `frontend/src/components/LongitudinalTimeline.jsx`, `backend/app.py` | KI-019 |

---

## 8. Тестирование

Каждый этап pipeline тестируется независимо (остальные отключаются в конфиге оркестратора).
Последовательность smoke-теста на каждый этап:
1. `python3 -m py_compile <script>` — проверка синтаксиса
2. Прогон этапа на МС-кейсе (P000915), проверка лога
3. Прогон этапа на глио-кейсе (UPENN-GBM), проверка отсутствия регрессий

Frontend: ручное тестирование в браузере после старта dev-сервера.

---

## 9. Конвенции коммитов

`feat(stage04): ...`, `fix(stage05): ...`, `refactor(config): ...` и т.д.
Каждый коммит — одна завершённая подзадача с пройденным smoke-тестом.

---

*Статус документа: утверждён. Следующий шаг — план реализации.*
