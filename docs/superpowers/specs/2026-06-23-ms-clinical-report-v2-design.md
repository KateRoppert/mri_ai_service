# MS Clinical Report v2 — Design Spec (Этап 5.6, Вариант C)

**Branch:** `feat/ms-clinical-metrics`
**Status:** Approved, ready for implementation plan
**Depends on:** Этап 5 (MS first-class pipeline support, merged to main)

## Контекст

Текущий МС-отчёт ("Вариант A", в проде) даёт количество очагов, суммарный/средний
объём, размерные бэнды и агрегатную динамику объёма между сессиями. Этот спек
("Вариант C") добавляет клинически значимые метрики:

- **(A) McDonald-классификация очагов по зонам** — periventricular / juxtacortical /
  infratentorial. Зона **spinal cord — вне скоупа** этой итерации (нет инфраструктуры
  регистрации спинного мозга).
- **(B) Детекция новых/растущих очагов** между сессиями.
- **(C) Гадолиний-накапливающие очаги (CE+) — отложено** в отдельный будущий
  спек/ветку. Причина: модель `ms-seg` не принимает T1c на вход (только T1/T2/FLAIR),
  и Stage 05 для МС намеренно не регистрирует T1c (см. KI-005, экономия ~25% времени);
  T1c для МС не гарантирован протоколом (в отличие от глио). Включать это сейчас —
  значит частично откатывать существующую оптимизацию без чёткого клинического запроса
  на это (пример отчёта от клинициста описывает только зоны локализации, не CE+).

Источник клинических требований: пример отчёта от клинициста (`MS FOCUS VOLUME REPORT`,
получен через Telegram) — описывает зоны Juxtacortical/cortical, Periventricular,
Infratentorial. Раздел "CLINICAL SUMMARY (RANO)" в этом примере — скорее всего
скопирован из глио-шаблона по ошибке (RANO/CE+/CE− — терминология ответа на терапию
для опухолей, для МС обычно не используется в таком виде); в дизайн не переносим.

## Аудит текущего состояния (где что считается)

| Что | Где | Замечание |
|---|---|---|
| Подсчёт очагов, объёмы, бэнды | `scripts/08_lobar_localization.py::compute_lesion_stats()` | МС-специфично, захардкожено внутри файла, который называется "лобарная локализация" |
| Локализация по долям мозга (кора) | `scripts/lobar_analysis.py` (`LobarAnalyzer`) | Только для GBM по факту: атлас Harvard-Oxford **кортикальный**, нет желудочков/ствола/мозжечка |
| Объёмы по BraTS-классам (NCR/ED/NET/ET) | `scripts/compute_volumes.py` | Жёстко завязан на 4 класса глио, для МС не применяется |
| Агрегатная динамика между сессиями | `backend/app.py::get_longitudinal()` (`/api/longitudinal/{patient_id}`) | Скалярный join по `*_lesion_stats_report.json` через `pipeline_manager`, **не трогает сами маски** |
| Рендер МС-отчёта | `frontend/src/components/ClinicalReportContent.jsx` (МС-ветка, ~L443-527) | Нет секции локализации (атлас уже считается в lobar_report.json, но не используется для МС) |

## Архитектура: разделение Stage 08

`08_lobar_localization.py` переименовывается в **`08_anatomical_analysis.py`**
(номер и позиция в пайплайне не меняются — фиксируется в `pipeline_config.yaml`,
`utils/config_loader.py`, `orchestrator.py`, `test_stage08_fixes.py`).

Вводится общий интерфейс (по аналогии с `SkullStripperBase` из skull-stripping
research):

```python
class AnatomicalAnalyzerBase(ABC):
    @abstractmethod
    def analyze_mask(self, mask_path: Path) -> Optional[Dict]: ...

    @abstractmethod
    def save_report(self, report: Dict, output_path: Path) -> bool: ...
```

- `lobar_analysis.py::LobarAnalyzer(AnatomicalAnalyzerBase)` — без изменений в логике,
  только формальное наследование.
- Новый `ms_localization.py::MSZoneAnalyzer(AnatomicalAnalyzerBase)` — McDonald-зоны.
- Новый `lesion_stats.py` — выносим туда `compute_lesion_stats()` из текущего
  08-скрипта (отдельная задача от классификации по зонам).
- `08_anatomical_analysis.py` становится тонким диспетчером: по `--lesion-type`
  выбирает `LobarAnalyzer` или `MSZoneAnalyzer`, плюс для МС всегда вызывает
  `lesion_stats.compute_lesion_stats()` (как сейчас).

Готовность к Этапу 8 (метастазы): третий анализатор подключается без изменения
диспетчера.

## (A) McDonald-классификация очагов

**Зоны:** periventricular, juxtacortical, infratentorial, + служебная категория
"deep white matter" (для очагов, не попавших ни в одну McDonald-зону — нужна, чтобы
сумма по зонам сходилась с total burden). Spinal cord — не поддерживается, явно
помечается в отчёте/UI как недоступная зона (не пропускается молча).

**Атласы (стандартное MNI152-пространство):**
- Желудочки: `HarvardOxford-sub-maxprob-thr25-1mm` (Left/Right Lateral Ventricle)
- Ствол/мозжечок (infratentorial): тот же `HarvardOxford-sub` (Brain-Stem) +
  `Cerebellum-MNIflirt-maxprob-thr25-1mm`
- Кора (juxtacortical): существующий `HarvardOxford-cort` атлас (реюз)

Источник: извлекаются из Docker-образа `kateroppert/mri-ai-service:latest`
(там уже есть FSL) через одноразовый скрипт `scripts/fetch_ms_atlases.sh`
(`docker run --rm -v <host_dir>:/out kateroppert/mri-ai-service:latest cp ...`).
Регистрируются **только под шаблон MNI152** (МС-пайплайн использует только его,
в отличие от GBM с тремя шаблонами) — без кросс-шаблонной регистрации.
Файлы кладутся в `data/templates/lobar_atlas/` рядом с существующими; конфигурация
зон — новый `configs/ms_zones_config.yaml` (по аналогии с `lobar_atlas_config.yaml`).

**Алгоритм классификации очага (на уровне компоненты связности):**
Иерархия по клинической значимости: если очаг касается желудочка (пересечение с
dilation=1 воксель, допуск на ошибку регистрации) → `periventricular`; иначе если
касается коры → `juxtacortical`; иначе если попадает в infratentorial-зону →
`infratentorial`; иначе → `deep_white_matter`.

**Выход:** `*_mcdonald_report.json` (аналог `lobar_report.json`): per-зона
count + total_volume_cm3, плюс per-lesion zone label (для hover во вьювере,
аналогично `lesion_volumes_by_label`).

## (B) Детекция новых/растущих очагов

**Ключевая упрощающая предпосылка:** обе сессии пациента независимо
регистрируются в один и тот же атлас (MNI152, фиксированный output_resolution)
→ маски разных сессий лежат на идентичной воксельной сетке. **Без дополнительной
inter-session регистрации.**

**Где считается:** не в Stage 08 (он обрабатывает одну сессию за прогон и не знает
о предыдущих сессиях пациента). Логика — в бэкенде, рядом с существующим
`get_longitudinal()` (`backend/app.py:784`), по аналогии с паттерном
`pipeline_manager.get_lesion_stats_reports()`:
- Новый `pipeline_manager.get_segmask_label_path(output_path, patient_id, session_id)`
  — путь к `*_segmask_labels.nii.gz` конкретной сессии.
- Новый модуль `backend/lesion_diff.py` — сравнение двух labeled-масок.

**Алгоритм:** сопоставление компонент сессии N и N-1 по пересечению (overlap,
допуск dilation=1 воксель, как в McDonald-классификации). Классификация очага:
- `new` — нет пересечения с очагами N-1
- `resolved` — был в N-1, нет в N
- `growing` — пересекается, объём вырос ≥ порога
- `stable` — пересекается, рост в пределах порога

**Порог роста** — конфигурируемый параметр (не хардкод в коде), дефолт
`max(20% относительного роста, 0.03 см³ абсолютного)` — выбран как разумное
приближение, **подлежит уточнению клиницистами** (McDonald 2017 формально
описывает только "one or more new T2 lesions", без чёткого порога роста для
существующих очагов). Хранится в `configs/ms_zones_config.yaml`.

**Новый эндпоинт:** `GET /api/longitudinal/{patient_id}/diff?lesion_type=multiple_sclerosis`
— отдаёт классификацию очагов между последней парой сессий (new/growing/stable/resolved
с volume для каждого).

## Фронтенд

**`ClinicalReportContent.jsx` (МС-ветка):**
- Новая секция "Локализация очагов (McDonald)" — аналог секции 3 у GBM
  (`EnvironmentOutlined`), отображает зоны вместо долей мозга. Spinal cord — серый
  тег "не поддерживается".
- `normalizeKappaEntity()` (L56-108) расширяется под `mcdonald_report` — рендер из
  Kappa и из локального API должен остаться единым источником правды.

**`LongitudinalTimeline.jsx`:**
- Добавляется колонка/тег "новые очаги" (count new + count growing) на основе
  нового diff-эндпоинта, в дополнение к существующему агрегатному Δ объёма.

## Тестирование

- Юнит-тесты на `MSZoneAnalyzer` (зональная классификация) и `lesion_diff.py`
  (new/growing/stable/resolved) — синтетические маски с известным расположением
  относительно зон/друг друга.
- End-to-end на `data/MS_5/P000915` (2 реальные сессии одного пациента,
  2022-01-18 и 2023-03-25, T1/T2/FLAIR присутствуют) — единственный доступный
  реальный multi-session МС-кейс в репозитории.
- Обновить `test_stage08_fixes.py` под новое имя файла/модулей.

## Ограничения и открытые вопросы (явно зафиксированы, не скрыты)

- Spinal cord зона не реализована (нет spine-регистрации в пайплайне).
- Гадолиний-накопление (CE+) отложено — отдельный спек после решения по T1c для МС.
- Порог "growing" — временный дефолт, ждёт подтверждения от клиницистов.
- Точность periventricular/infratentorial зависит от качества аффинной регистрации
  в MNI152 (та же оговорка, что уже принята для существующей лобарной локализации
  GBM — не новый риск, а тот же класс допущения).
