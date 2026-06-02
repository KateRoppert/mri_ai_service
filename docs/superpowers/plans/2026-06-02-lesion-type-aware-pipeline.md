# Lesion-Type-Aware Pipeline — план реализации

> **Для агентных воркеров:** ТРЕБУЕМЫЙ СУБСКИЛЛ: использовать `superpowers:subagent-driven-development` (рекомендуется) или `superpowers:executing-plans` для пошагового выполнения этого плана. Шаги используют синтаксис чекбоксов (`- [ ]`) для отслеживания прогресса.

**Цель:** сделать МС-пайплайн полноправным типом поражения на всех этапах (Stage 01–08), добавить МС-отчёт на фронте с подсчётом очагов и лонгитюдным сравнением.

**Архитектура:** единый источник конфига в `lesion_types.yaml`, читается через утилиту `load_lesion_type_config()`. Каждый pipeline-этап получает `--lesion-type` и использует эту утилиту вместо захардкоженных констант. Stage 08 вычисляет `lesion_stats_report.json` для МС. Frontend роутит `ClinicalReportContent` по `lesion_type`.

**Стек:** Python 3.12, FastAPI, SQLite (patient_registry), React + Ant Design, pytest, scipy.ndimage (для подсчёта connected components).

---

## Карта файлов

| Файл | Действие | Ответственность |
|------|----------|-----------------|
| `utils/config_loader.py` | Изменить | Добавить `load_lesion_type_config()` |
| `tests/test_config_loader.py` | Создать | Unit-тесты утилиты |
| `scripts/01_reorganize_folders.py` | Изменить | Заменить `LESION_TYPE_MODALITIES` на утилиту |
| `scripts/04_assess_quality.py` | Изменить | Добавить `--lesion-type`, использовать утилиту |
| `scripts/05_preprocessing.py` | Изменить | Добавить `--lesion-type`, исправить t1c-фолбэки, использовать утилиту |
| `orchestrator.py` | Изменить | Добавить stage_04 и stage_05 в список инъекции `--lesion-type` |
| `scripts/08_lobar_localization.py` | Изменить | Добавить `compute_lesion_stats()` для МС |
| `backend/models.py` | Изменить | Добавить `lesion_type` в `PipelineStatusResponse`, новые Pydantic-модели |
| `backend/pipeline_manager.py` | Изменить | Добавить `get_lesion_stats_reports()` |
| `backend/app.py` | Изменить | Два новых эндпоинта: `lesion-stats`, `longitudinal` |
| `frontend/src/services/api.js` | Изменить | Добавить `getLesionStatsReports()`, `getLongitudinalReport()` |
| `frontend/src/components/NIfTIViewer.jsx` | Изменить | Динамический colormap по `lesionType` |
| `frontend/src/components/ClinicalReport.jsx` | Изменить | Прокидывать `lesionType` |
| `frontend/src/components/ClinicalReportContent.jsx` | Изменить | Роутинг по `lesionType`, МС-секция |
| `frontend/src/components/LongitudinalTimeline.jsx` | Создать | Компонент с таблицей динамики |

---

## Задача 1: Утилита `load_lesion_type_config()`

**Файлы:**
- Изменить: `utils/config_loader.py`
- Создать: `tests/test_config_loader.py`

- [ ] **Шаг 1.1: Написать падающий тест**

```python
# tests/test_config_loader.py
import pytest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config

def test_glioblastoma_modalities():
    cfg = load_lesion_type_config('glioblastoma')
    assert set(cfg['required_modalities']) == {'t1', 't1c', 't2', 't2fl'}
    assert cfg['reference_modality'] == 't1'
    assert 'volume' in cfg['reports']

def test_multiple_sclerosis_modalities():
    cfg = load_lesion_type_config('multiple_sclerosis')
    assert set(cfg['required_modalities']) == {'t1', 't2', 't2fl'}
    assert 't1c' not in cfg['required_modalities']
    assert cfg['reference_modality'] == 't1'
    assert 'lesion_count' in cfg['reports']

def test_unknown_lesion_type_raises():
    with pytest.raises(KeyError):
        load_lesion_type_config('brain_metastasis')
```

- [ ] **Шаг 1.2: Убедиться, что тест падает**

```bash
cd /home/ubuntu/mri_ai_service && python -m pytest tests/test_config_loader.py -v
```
Ожидаем: `ImportError` или `AttributeError` — функция не существует.

- [ ] **Шаг 1.3: Реализовать функцию**

В конец файла `utils/config_loader.py` добавить:

```python
def load_lesion_type_config(lesion_type: str) -> Dict[str, Any]:
    """
    Load per-lesion-type configuration from configs/lesion_types.yaml.

    Returns dict with keys: required_modalities, reference_modality, reports.
    Raises KeyError if lesion_type not found in the YAML.
    """
    config_path = Path(__file__).parent.parent / 'configs' / 'lesion_types.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        all_configs = yaml.safe_load(f)
    if lesion_type not in all_configs:
        raise KeyError(
            f"Unknown lesion_type '{lesion_type}'. "
            f"Available: {list(all_configs.keys())}"
        )
    return all_configs[lesion_type]
```

- [ ] **Шаг 1.4: Убедиться, что тесты проходят**

```bash
cd /home/ubuntu/mri_ai_service && python -m pytest tests/test_config_loader.py -v
```
Ожидаем: `3 passed`.

- [ ] **Шаг 1.5: Проверить синтаксис**

```bash
python -m py_compile utils/config_loader.py
```

- [ ] **Шаг 1.6: Коммит**

```bash
git add utils/config_loader.py tests/test_config_loader.py
git commit -m "feat(config): add load_lesion_type_config() utility (KI-016)"
```

---

## Задача 2: Stage 01 — заменить LESION_TYPE_MODALITIES

**Файлы:**
- Изменить: `scripts/01_reorganize_folders.py:571-590` и строки 886–890

- [ ] **Шаг 2.1: Добавить импорт утилиты в начало файла**

В `scripts/01_reorganize_folders.py`, в секции импортов (после строки `import yaml`), добавить:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config
```

- [ ] **Шаг 2.2: Заменить LESION_TYPE_MODALITIES в CompletenessChecker**

Найти в `scripts/01_reorganize_folders.py` блок (~строки 565–585):

```python
    DEFAULT_REQUIRED_MODALITIES = {'t1', 't1c', 't2', 't2fl'}  # glioma case

    # Per-lesion-type modality requirements (extended over time)
    LESION_TYPE_MODALITIES = {
        'glioblastoma': {'t1', 't1c', 't2', 't2fl'},
        'multiple_sclerosis': {'t1', 't2', 't2fl'},
    }

    def __init__(self, logger: logging.Logger, lesion_type: str = 'glioblastoma'):
        self.logger = logger
        self.required_modalities = self.LESION_TYPE_MODALITIES.get(
            lesion_type, self.DEFAULT_REQUIRED_MODALITIES
        )
```

Заменить на:

```python
    DEFAULT_REQUIRED_MODALITIES = {'t1', 't1c', 't2', 't2fl'}  # fallback for unknown types

    def __init__(self, logger: logging.Logger, lesion_type: str = 'glioblastoma'):
        self.logger = logger
        try:
            cfg = load_lesion_type_config(lesion_type)
            self.required_modalities = set(cfg['required_modalities'])
        except KeyError:
            self.logger.warning(
                f"Unknown lesion_type '{lesion_type}', using default modalities"
            )
            self.required_modalities = self.DEFAULT_REQUIRED_MODALITIES
```

- [ ] **Шаг 2.3: Заменить второе использование LESION_TYPE_MODALITIES (~строки 886–890)**

Найти (~строка 886):
```python
        CompletenessChecker.LESION_TYPE_MODALITIES.get(
```

Заменить блок на вызов утилиты. Найти точный контекст:

```bash
grep -n "LESION_TYPE_MODALITIES" scripts/01_reorganize_folders.py
```

Для каждого вхождения заменить `CompletenessChecker.LESION_TYPE_MODALITIES.get(lesion_type, ...)` на:

```python
        try:
            set(load_lesion_type_config(lesion_type)['required_modalities'])
        except KeyError:
            CompletenessChecker.DEFAULT_REQUIRED_MODALITIES
```

- [ ] **Шаг 2.4: Проверить синтаксис**

```bash
python -m py_compile scripts/01_reorganize_folders.py
```

- [ ] **Шаг 2.5: Smoke-тест Stage 01 на МС**

Отключить в `pipeline_config.yaml` все stages кроме `stage_01_reorganize`, затем:

```bash
cd /home/ubuntu/mri_ai_service && python orchestrator.py --config pipeline_config.yaml
```

Ожидаем: в логе `required modalities: ['t1', 't2', 't2fl']` для МС-кейса, нет T1c.

- [ ] **Шаг 2.6: Коммит**

```bash
git add scripts/01_reorganize_folders.py
git commit -m "refactor(stage01): replace LESION_TYPE_MODALITIES with load_lesion_type_config (KI-016)"
```

---

## Задача 3: Stage 04 — добавить `--lesion-type` (KI-004)

**Файлы:**
- Изменить: `scripts/04_assess_quality.py`
- Изменить: `orchestrator.py`

- [ ] **Шаг 3.1: Добавить импорт утилиты**

В `scripts/04_assess_quality.py`, в секции импортов:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config
```

- [ ] **Шаг 3.2: Добавить `--lesion-type` в argparse**

Найти в `scripts/04_assess_quality.py` функцию `parse_arguments()` (~строка 757). Добавить аргумент (перед финальным `return parser`):

```python
    parser.add_argument(
        '--lesion-type',
        type=str,
        default='glioblastoma',
        choices=['glioblastoma', 'multiple_sclerosis'],
        help='Type of brain lesion being processed'
    )
```

- [ ] **Шаг 3.3: Читать модальности из конфига в main()**

В функции `main()` после `args = parse_arguments()`, найти место где используется конфиг и добавить загрузку lesion_type-специфичных модальностей. Найти строку где строится список ожидаемых модальностей или создаётся ассессор. Добавить сразу после `args = parse_arguments()`:

```python
    try:
        lt_config = load_lesion_type_config(args.lesion_type)
        expected_modalities = set(lt_config['required_modalities'])
    except KeyError:
        logger.warning(f"Unknown lesion_type '{args.lesion_type}', using all modalities")
        expected_modalities = {'t1', 't1c', 't2', 't2fl'}
    
    logger.info(f"Lesion type: {args.lesion_type}, expected modalities: {sorted(expected_modalities)}")
```

- [ ] **Шаг 3.4: Обновить orchestrator.py**

В `orchestrator.py`, найти строку (~строка 63):

```python
    if stage_name in (
        'stage_01_reorganize',
        'stage_06_segmentation',
        'stage_07_inverse_transform',
        'stage_08_lobar_localization',
    ):
```

Заменить на:

```python
    if stage_name in (
        'stage_01_reorganize',
        'stage_04_quality',
        'stage_05_preprocessing',
        'stage_06_segmentation',
        'stage_07_inverse_transform',
        'stage_08_lobar_localization',
    ):
```

- [ ] **Шаг 3.5: Проверить синтаксис обоих файлов**

```bash
python -m py_compile scripts/04_assess_quality.py && python -m py_compile orchestrator.py
```

- [ ] **Шаг 3.6: Smoke-тест Stage 04 на МС и глио**

Включить только `stage_04_quality` в config. Прогнать МС-кейс, проверить лог — должно быть `Lesion type: multiple_sclerosis, expected modalities: ['t1', 't2', 't2fl']`. Прогнать глио — лог должен быть `glioblastoma, ... ['t1', 't1c', 't2', 't2fl']`.

- [ ] **Шаг 3.7: Коммит**

```bash
git add scripts/04_assess_quality.py orchestrator.py
git commit -m "feat(stage04): add --lesion-type, read expected modalities from lesion_types.yaml (KI-004)"
```

---

## Задача 4: Stage 05 — исправить T1c-фолбэки и добавить `--lesion-type` (KI-005)

**Файлы:**
- Изменить: `scripts/05_preprocessing.py`

- [ ] **Шаг 4.1: Добавить импорт утилиты**

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.config_loader import load_lesion_type_config
```

- [ ] **Шаг 4.2: Добавить `--lesion-type` в `parse_arguments()`**

В функции `parse_arguments()` (~строка 593) добавить аргумент:

```python
    parser.add_argument(
        '--lesion-type',
        type=str,
        default='glioblastoma',
        choices=['glioblastoma', 'multiple_sclerosis'],
        help='Type of brain lesion — determines which modalities to process'
    )
```

- [ ] **Шаг 4.3: Переопределить список модальностей в `main()`**

В `main()`, строка 735 где читается список модальностей:

```python
        modalities = config.get('modalities', ['t1c', 't1', 't2', 't2fl'])
```

Заменить на:

```python
        try:
            lt_config = load_lesion_type_config(args.lesion_type)
            modalities = lt_config['required_modalities']
            logger.info(
                f"Lesion type: {args.lesion_type}, processing modalities: {modalities}"
            )
        except KeyError:
            modalities = config.get('modalities', ['t1c', 't1', 't2', 't2fl'])
            logger.warning(f"Unknown lesion_type '{args.lesion_type}', using config modalities")
```

- [ ] **Шаг 4.4: Исправить фолбэк reference_modality (~строки 303, 307)**

Найти функцию `process_subject()` (~строка 265). В ней блок:

```python
    reference_modality = None
    for step in config.get('steps', []):
        if step['name'] == 'registration':
            reference_modality = step.get('params', {}).get('reference_modality', 't1c')
            break

    if reference_modality is None:
        reference_modality = 't1c'  # Default fallback
```

Функция `process_subject()` принимает `config` и `modalities` — добавить также `lesion_type: str = 'glioblastoma'` в её сигнатуру:

```python
def process_subject(
    subject_id: str,
    session_id: str,
    anat_dir: Path,
    output_dir: Path,
    temp_dir: Path,
    config: dict,
    modalities: list,
    threads: int = 4,
    lesion_type: str = 'glioblastoma',
) -> dict:
```

Заменить блок фолбэка на:

```python
    reference_modality = None
    for step in config.get('steps', []):
        if step['name'] == 'registration':
            reference_modality = step.get('params', {}).get('reference_modality')
            break

    if reference_modality is None:
        # Derive from lesion_types.yaml, not hardcoded t1c
        try:
            reference_modality = load_lesion_type_config(lesion_type)['reference_modality']
        except KeyError:
            reference_modality = 't1'
```

- [ ] **Шаг 4.5: Исправить фолбэк в строке ~412**

Найти строку:
```python
                "success": registration_results.get(step_params.get('reference_modality', 't1c'), {}).get('success', False),
```

Заменить на:
```python
                "success": registration_results.get(step_params.get('reference_modality', reference_modality), {}).get('success', False),
```

- [ ] **Шаг 4.6: Прокинуть lesion_type в вызовы process_subject**

Найти все места где вызывается `process_subject(...)` в `main()` (в sequential и parallel режимах). Добавить аргумент `lesion_type=args.lesion_type`.

```bash
grep -n "process_subject(" scripts/05_preprocessing.py
```

Для каждого вызова добавить именованный аргумент:

```python
result = process_subject(
    ...,
    lesion_type=args.lesion_type,
)
```

- [ ] **Шаг 4.7: Проверить синтаксис**

```bash
python -m py_compile scripts/05_preprocessing.py
```

- [ ] **Шаг 4.8: Smoke-тест Stage 05 на МС**

Включить только `stage_05_preprocessing` в config. Прогнать МС-кейс. В логе должно быть `processing modalities: ['t1', 't2', 't2fl']`, T1c не упоминается. Время Stage 05 для МС должно быть ~25% быстрее (нет N4 для T1c).

Smoke-тест на глио — все 4 модальности, нет регрессий.

- [ ] **Шаг 4.9: Коммит**

```bash
git add scripts/05_preprocessing.py
git commit -m "fix(stage05): add --lesion-type, skip t1c for MS, fix reference_modality fallback (KI-005)"
```

---

## Задача 5: Stage 08 — подсчёт очагов МС (KI-019)

**Файлы:**
- Изменить: `scripts/08_lobar_localization.py`

- [ ] **Шаг 5.1: Добавить импорт scipy**

В `scripts/08_lobar_localization.py` добавить в импорты:

```python
from scipy.ndimage import label as ndimage_label
```

- [ ] **Шаг 5.2: Добавить функцию `compute_lesion_stats()`**

После существующих импортов, добавить функцию:

```python
def compute_lesion_stats(mask_path: Path) -> dict:
    """
    Count connected components (individual lesions) in a binary mask.
    Used for MS where each component = one lesion.

    Returns dict with lesion_count, total_volume_cm3, mean_lesion_volume_cm3,
    lesion_volumes_cm3.
    """
    import nibabel as nib
    import numpy as np

    img = nib.load(str(mask_path))
    data = np.asarray(img.dataobj)
    voxel_vol_mm3 = float(np.prod(np.abs(np.diag(img.affine[:3, :3]))))
    voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

    binary = (data > 0).astype(np.uint8)
    labeled, n_components = ndimage_label(binary)

    lesion_volumes = []
    for i in range(1, n_components + 1):
        voxel_count = int(np.sum(labeled == i))
        lesion_volumes.append(round(voxel_count * voxel_vol_cm3, 4))

    total = round(sum(lesion_volumes), 4)
    mean = round(total / n_components, 4) if n_components > 0 else 0.0

    return {
        "lesion_count": n_components,
        "total_volume_cm3": total,
        "mean_lesion_volume_cm3": mean,
        "lesion_volumes_cm3": sorted(lesion_volumes, reverse=True),
    }
```

- [ ] **Шаг 5.3: Вызвать функцию в `process_one_mask()` для МС**

В функции `process_one_mask()` (~строка 134), после строки `analyzer.save_report(report, report_path)`, добавить:

```python
        # For MS: compute and save per-lesion statistics
        if lesion_type == 'multiple_sclerosis':
            stats = compute_lesion_stats(mask_path)
            stats["patient_id"] = subject_id
            stats["session_id"] = session_id
            stats_path = report_path.parent / report_path.name.replace(
                "_lobar_report.json", "_lesion_stats_report.json"
            )
            import json as _json
            with open(stats_path, 'w', encoding='utf-8') as f:
                _json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(
                f"Lesion stats: {stats['lesion_count']} lesions, "
                f"{stats['total_volume_cm3']:.3f} cm³ → {stats_path.name}"
            )
```

- [ ] **Шаг 5.4: Проверить синтаксис**

```bash
python -m py_compile scripts/08_lobar_localization.py
```

- [ ] **Шаг 5.5: Smoke-тест Stage 08 на МС**

Включить только `stage_08_lobar_localization`. Прогнать МС-кейс.

Проверить, что создан файл вида:
```
demo_workspace/input/.../segmentation/.../multiple_sclerosis/..._lesion_stats_report.json
```

Открыть файл и убедиться, что `lesion_count > 0` и `total_volume_cm3 > 0`.

Прогнать глио — убедиться, что `_lesion_stats_report.json` не создаётся (нет ошибок).

- [ ] **Шаг 5.6: Коммит**

```bash
git add scripts/08_lobar_localization.py
git commit -m "feat(stage08): compute lesion count via connected components for MS (KI-019)"
```

---

## Задача 6: Backend — новые эндпоинты и модели

**Файлы:**
- Изменить: `backend/models.py`
- Изменить: `backend/pipeline_manager.py`
- Изменить: `backend/app.py`

- [ ] **Шаг 6.1: Добавить Pydantic-модели в `backend/models.py`**

В конец файла `backend/models.py` добавить:

```python
# ============================================
# МОДЕЛИ ДЛЯ МС-ОТЧЁТА
# ============================================

class LesionStatsReport(BaseModel):
    """Статистика очагов МС для одной сессии"""
    patient_id: str
    session_id: str
    lesion_count: int
    total_volume_cm3: float
    mean_lesion_volume_cm3: float
    lesion_volumes_cm3: List[float]

class LesionStatsListResponse(BaseModel):
    total: int
    reports: List[LesionStatsReport]

class LongitudinalPoint(BaseModel):
    """Одна точка лонгитюдного ряда"""
    session_id: str
    scan_date: Optional[str]
    total_volume_cm3: float
    lesion_count: Optional[int] = None

class LongitudinalResponse(BaseModel):
    patient_id: str
    lesion_type: str
    points: List[LongitudinalPoint]
```

Также добавить `lesion_type: Optional[str] = None` в `PipelineStatusResponse` — найти класс (~строка 158) и добавить поле после `error`:

```python
    lesion_type: Optional[str] = Field(None, description="Тип поражения (glioblastoma / multiple_sclerosis)")
```

- [ ] **Шаг 6.2: Добавить `get_lesion_stats_reports()` в `pipeline_manager.py`**

После метода `get_lobar_reports()` (~строка 422+) добавить:

```python
    def get_lesion_stats_reports(self, output_path: str) -> Optional[List[Dict[str, Any]]]:
        """
        Read lesion_stats_report.json files produced by Stage 08 for MS cases.
        Pattern: segmentation/**/*_lesion_stats_report.json
        """
        seg_dir = Path(output_path) / "segmentation"
        if not seg_dir.exists():
            return None

        report_files = list(seg_dir.rglob("*_lesion_stats_report.json"))
        if not report_files:
            return None

        reports = []
        for report_file in report_files:
            try:
                with open(report_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                reports.append(data)
                logger.info(f"Lesion stats loaded: {report_file.name}")
            except Exception as e:
                logger.error(f"Failed to read {report_file}: {e}")

        return reports or None
```

- [ ] **Шаг 6.3: Добавить эндпоинт `GET /api/lesion-stats/{run_id}`**

В `backend/app.py`, после эндпоинта `get_lobar_reports` (~строка 730), добавить:

```python
@app.get("/api/lesion-stats/{run_id}", response_model=LesionStatsListResponse)
async def get_lesion_stats(
    run_id: str,
    db: Session = Depends(get_db)
):
    """Статистика очагов МС (количество, объёмы) для run_id"""
    run = get_pipeline_run(db, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Pipeline run not found")
    if run.current_stage < 7:
        raise HTTPException(status_code=400, detail="Lobar localization not yet completed")

    reports = pipeline_manager.get_lesion_stats_reports(run.output_path)
    if not reports:
        raise HTTPException(status_code=404, detail="Lesion stats not found (MS only)")

    return LesionStatsListResponse(
        total=len(reports),
        reports=[LesionStatsReport(**r) for r in reports]
    )
```

Добавить `LesionStatsListResponse, LesionStatsReport` в импорт из `models` в начале `app.py`.

- [ ] **Шаг 6.4: Добавить эндпоинт `GET /api/longitudinal/{patient_id}`**

После предыдущего эндпоинта добавить:

```python
@app.get("/api/longitudinal/{patient_id}", response_model=LongitudinalResponse)
async def get_longitudinal(
    patient_id: str,
    lesion_type: str = "multiple_sclerosis",
    db: Session = Depends(get_db)
):
    """
    Лонгитюдный анализ: все сессии пациента по данному lesion_type.

    patient_id — original_patient_id (напр. "P000915").
    Матчинг lesion_stats: bids_id в registry = patient_id в stats файле
    (оба имеют формат "sub-P000915").
    """
    from patient_registry import find_by_patient_id

    all_records = find_by_patient_id(patient_id)
    records = [r for r in all_records if r.get("lesion_type") == lesion_type]

    if not records:
        raise HTTPException(status_code=404, detail="No sessions found for this patient/lesion_type")

    # Collect all stats across unique run_ids, match by bids_id == patient_id-in-stats
    seen_run_ids = set()
    all_stats = []
    for record in records:
        run_id = record.get("run_id")
        if not run_id or run_id in seen_run_ids:
            continue
        seen_run_ids.add(run_id)
        run = get_pipeline_run(db, run_id)
        if run and run.output_path:
            stats_list = pipeline_manager.get_lesion_stats_reports(run.output_path) or []
            bids_id = record.get("bids_id", "")  # "sub-P000915"
            for s in stats_list:
                # stats["patient_id"] is also "sub-P000915" (set by Stage 08)
                if s.get("patient_id") == bids_id:
                    all_stats.append({**s, "_scan_date": record.get("scan_date")})

    # Deduplicate by session_id, sort by scan_date
    seen_sessions = set()
    points = []
    for s in sorted(all_stats, key=lambda x: x.get("_scan_date") or ""):
        sid = s.get("session_id", "")
        if sid in seen_sessions:
            continue
        seen_sessions.add(sid)
        points.append(LongitudinalPoint(
            session_id=sid,
            scan_date=str(s["_scan_date"]) if s.get("_scan_date") else None,
            total_volume_cm3=s.get("total_volume_cm3", 0.0),
            lesion_count=s.get("lesion_count"),
        ))

    if len(points) < 2:
        raise HTTPException(
            status_code=404,
            detail="Not enough sessions for longitudinal analysis (need ≥ 2)"
        )

    return LongitudinalResponse(
        patient_id=patient_id,
        lesion_type=lesion_type,
        points=points,
    )
```

Добавить `LongitudinalResponse, LongitudinalPoint` в импорт из `models`.

- [ ] **Шаг 6.5: Добавить `lesion_type` в ответ статуса pipeline**

В `backend/app.py`, найти функцию `get_pipeline_status()` (~строка 310), найти возврат `PipelineStatusResponse(...)`. Добавить `lesion_type=run.lesion_type`:

```python
    return PipelineStatusResponse(
        run_id=run.run_id,
        ...
        lesion_type=run.lesion_type,
    )
```

- [ ] **Шаг 6.6: Проверить синтаксис**

```bash
python -m py_compile backend/models.py && python -m py_compile backend/pipeline_manager.py && python -m py_compile backend/app.py
```

- [ ] **Шаг 6.7: Проверить эндпоинты вручную**

Запустить веб-сервис (или только `uvicorn backend.app:app`), затем:

```bash
# Подставить реальный run_id из вашей БД
curl -s http://localhost:8000/api/lesion-stats/<run_id> | python3 -m json.tool
curl -s "http://localhost:8000/api/longitudinal/P000915?lesion_type=multiple_sclerosis" | python3 -m json.tool
```

- [ ] **Шаг 6.8: Коммит**

```bash
git add backend/models.py backend/pipeline_manager.py backend/app.py
git commit -m "feat(backend): add lesion-stats and longitudinal endpoints for MS report (KI-019)"
```

---

## Задача 7: Frontend — новые API-функции

**Файлы:**
- Изменить: `frontend/src/services/api.js`

- [ ] **Шаг 7.1: Добавить функции**

В `frontend/src/services/api.js`, после функции `getLobarReports` (~строка 68), добавить:

```js
/**
 * Статистика очагов МС (количество, объёмы)
 */
export const getLesionStatsReports = async (runId) => {
  const response = await apiClient.get(`/lesion-stats/${runId}`);
  return response.data;
};

/**
 * Лонгитюдная динамика пациента по типу поражения
 */
export const getLongitudinalReport = async (patientId, lesionType = 'multiple_sclerosis') => {
  const response = await apiClient.get(`/longitudinal/${patientId}`, {
    params: { lesion_type: lesionType },
  });
  return response.data;
};
```

Добавить `getLesionStatsReports` и `getLongitudinalReport` в экспорт в конце файла (там где перечислены другие экспорты).

- [ ] **Шаг 7.2: Коммит**

```bash
git add frontend/src/services/api.js
git commit -m "feat(frontend/api): add getLesionStatsReports and getLongitudinalReport"
```

---

## Задача 8: NIfTIViewer — динамический colormap

**Файлы:**
- Изменить: `frontend/src/components/NIfTIViewer.jsx`

- [ ] **Шаг 8.1: Добавить `lesionType` prop**

В объявление компонента (~строка 39) добавить `lesionType = 'glioblastoma'`:

```jsx
const NIfTIViewer = ({ runId, visible, onClose, customFiles = null, validationRef = null, lesionType = 'glioblastoma' }) => {
```

- [ ] **Шаг 8.2: Добавить функцию `createMsColormap()`**

После `createSegmentationColormap()` (~строка 37) добавить:

```jsx
// Binary colormap for MS: 0=transparent background, 1=green lesion
const createMsColormap = () => ({
  R: [0, 82],
  G: [0, 196],
  B: [0, 26],
  A: [0, 255],
});
```

- [ ] **Шаг 8.3: Заменить хардкод colormap на выбор по `lesionType`**

Найти все 2 места где вызывается `createSegmentationColormap()` и добавляется colormap (~строки 207 и 375). Для каждого заменить:

```jsx
      const segColormap = createSegmentationColormap();
      nv.addColormap('seg_custom', segColormap);
```

На:

```jsx
      const segColormap = lesionType === 'multiple_sclerosis'
        ? createMsColormap()
        : createSegmentationColormap();
      nv.addColormap('seg_custom', segColormap);
```

- [ ] **Шаг 8.4: Исправить `cal_max` для МС**

Найти места где устанавливается `cal_max: 4` (~строки 225 и 385). Заменить на:

```jsx
        cal_max: lesionType === 'multiple_sclerosis' ? 1 : 4,
```

- [ ] **Шаг 8.5: Прокинуть `lesionType` из `ValidationPanel.jsx`**

В `frontend/src/components/ValidationPanel.jsx`, найти `<NIfTIViewer` (~строка 267):

```jsx
      <NIfTIViewer
        visible={viewerOpen}
        onClose={() => setViewerOpen(false)}
        customFiles={viewerFiles}
        validationRef={viewerEntityRef}
      />
```

Добавить `lesionType`. `lesion_type` уже доступен через `entity.dsEntityInfo?.lesion_type` (строка ~149). Сохранить его в state `viewerLesionType`. Добавить:

```jsx
  const [viewerLesionType, setViewerLesionType] = useState('glioblastoma');
```

При открытии viewer'а (там же где устанавливается `viewerFiles`) добавить `setViewerLesionType(entity.dsEntityInfo?.lesion_type || 'glioblastoma')`.

Прокинуть в компонент:

```jsx
      <NIfTIViewer
        visible={viewerOpen}
        onClose={() => setViewerOpen(false)}
        customFiles={viewerFiles}
        validationRef={viewerEntityRef}
        lesionType={viewerLesionType}
      />
```

- [ ] **Шаг 8.6: Тест в браузере**

Запустить `cd frontend && npm run dev`. Открыть Validation → выбрать МС-кейс → открыть маску. Убедиться, что очаги зелёного цвета (не красного/синего как у глио). Открыть глио-кейс — убедиться, что цвета не изменились.

- [ ] **Шаг 8.7: Коммит**

```bash
git add frontend/src/components/NIfTIViewer.jsx frontend/src/components/ValidationPanel.jsx
git commit -m "feat(frontend/NIfTIViewer): dynamic colormap by lesion_type (MS=binary green)"
```

---

## Задача 9: `LongitudinalTimeline` — новый компонент

**Файлы:**
- Создать: `frontend/src/components/LongitudinalTimeline.jsx`

- [ ] **Шаг 9.1: Создать компонент**

```jsx
// frontend/src/components/LongitudinalTimeline.jsx
import { useEffect, useState } from 'react';
import { Table, Spin, Alert, Tag } from 'antd';
import { getLongitudinalReport } from '../services/api';

const LongitudinalTimeline = ({ patientId, lesionType }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!patientId) return;
    setLoading(true);
    getLongitudinalReport(patientId, lesionType)
      .then(resp => setData(resp.points))
      .catch(err => {
        if (err.response?.status === 404) setData([]);
        else setError('Не удалось загрузить динамику');
      })
      .finally(() => setLoading(false));
  }, [patientId, lesionType]);

  if (loading) return <Spin size="small" />;
  if (error) return <Alert message={error} type="warning" showIcon />;
  if (!data || data.length < 2) return null;

  const columns = [
    { title: 'Дата', dataIndex: 'scan_date', key: 'date', width: 130 },
    { title: 'Сессия', dataIndex: 'session_id', key: 'session', width: 130 },
    {
      title: 'Объём (см³)',
      dataIndex: 'total_volume_cm3',
      key: 'volume',
      align: 'right',
      render: val => val.toFixed(3),
    },
    {
      title: 'Очагов',
      dataIndex: 'lesion_count',
      key: 'count',
      align: 'right',
      render: val => val ?? '—',
    },
    {
      title: 'Δ объём',
      key: 'delta',
      align: 'right',
      render: (_, record, idx) => {
        if (idx === 0) return '—';
        const prev = data[idx - 1].total_volume_cm3;
        const delta = record.total_volume_cm3 - prev;
        const color = delta > 0 ? 'red' : delta < 0 ? 'green' : 'default';
        return <Tag color={color}>{delta > 0 ? '+' : ''}{delta.toFixed(3)}</Tag>;
      },
    },
  ];

  return (
    <Table
      columns={columns}
      dataSource={data.map((p, i) => ({ ...p, key: i }))}
      pagination={false}
      size="small"
      bordered
    />
  );
};

export default LongitudinalTimeline;
```

- [ ] **Шаг 9.2: Коммит**

```bash
git add frontend/src/components/LongitudinalTimeline.jsx
git commit -m "feat(frontend): add LongitudinalTimeline component for MS longitudinal analysis"
```

---

## Задача 10: `ClinicalReportContent` — роутинг по lesion_type

**Файлы:**
- Изменить: `frontend/src/components/ClinicalReportContent.jsx`
- Изменить: `frontend/src/components/ClinicalReport.jsx`

- [ ] **Шаг 10.1: Добавить импорты и новые API-вызовы в `ClinicalReportContent.jsx`**

В начало `ClinicalReportContent.jsx`, к существующим импортам добавить:

```jsx
import { getLesionStatsReports, getLongitudinalReport } from '../services/api';
import LongitudinalTimeline from './LongitudinalTimeline';
```

- [ ] **Шаг 10.2: Добавить `lesionType` prop и состояние МС-данных**

Изменить сигнатуру компонента (~строка 14):

```jsx
const ClinicalReportContent = ({ runId, autoLoad = false, lesionType = 'glioblastoma' }) => {
```

Добавить state для МС-данных рядом с существующими `useState`:

```jsx
  const [lesionStatsReports, setLesionStatsReports] = useState([]);
```

- [ ] **Шаг 10.3: Загружать МС-данные в `fetchAllData()`**

Найти функцию `fetchAllData()`. Изменить `Promise.all` — добавить условный запрос:

```jsx
  const fetchAllData = async () => {
    setLoading(true);
    setError(null);
    try {
      const fetches = [
        getVolumeReports(runId).catch(() => ({ reports: [] })),
        getLobarReports(runId).catch(() => ({ reports: [] })),
      ];
      if (lesionType === 'multiple_sclerosis') {
        fetches.push(getLesionStatsReports(runId).catch(() => ({ reports: [] })));
      }
      const results = await Promise.all(fetches);
      setVolumeReports(results[0].reports || []);
      setLobarReports(results[1].reports || []);
      if (lesionType === 'multiple_sclerosis') {
        setLesionStatsReports(results[2]?.reports || []);
      }
      setLoaded(true);
    } catch (err) {
      console.error('Ошибка загрузки отчёта:', err);
      setError('Не удалось загрузить данные');
    } finally {
      setLoading(false);
    }
  };
```

- [ ] **Шаг 10.4: Добавить МС-рендер в `return`**

В функции рендера, перед финальным `return (<>...</>)`, добавить ветку:

```jsx
  if (lesionType === 'multiple_sclerosis') {
    if (!loaded || lesionStatsReports.length === 0) return null;
    return (
      <>
        {lesionStatsReports.map((stats, idx) => (
          <div key={idx} style={{ marginBottom: 32 }}>
            <div style={{ marginBottom: 16 }}>
              <Tag>{stats.patient_id}</Tag>
              <Tag>{stats.session_id}</Tag>
            </div>

            <Divider orientation="left" style={{ fontSize: 14 }}>
              <Space><MedicineBoxOutlined /> Показатели МС</Space>
            </Divider>

            <Row gutter={32} style={{ marginBottom: 16 }}>
              <Col>
                <Statistic
                  title="Суммарный объём очагов"
                  value={stats.total_volume_cm3}
                  precision={3}
                  suffix="см³"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col>
                <Statistic
                  title="Количество очагов"
                  value={stats.lesion_count}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col>
                <Statistic
                  title="Средний объём очага"
                  value={stats.mean_lesion_volume_cm3}
                  precision={3}
                  suffix="см³"
                />
              </Col>
            </Row>

            {/* Лобарное распределение (из lobar_report) */}
            {(() => {
              const lobar = lobarReports.find(
                lr => lr.patient_id === stats.patient_id && lr.session_id === stats.session_id
              );
              return lobar ? (
                <>
                  <Divider orientation="left" style={{ fontSize: 14 }}>
                    <Space><EnvironmentOutlined /> Анатомическая локализация</Space>
                  </Divider>
                  <Table
                    columns={lobarColumns}
                    dataSource={getLobarTableData(lobar)}
                    pagination={false}
                    size="small"
                    bordered
                    style={{ marginBottom: 24 }}
                  />
                </>
              ) : null;
            })()}

            {/* Лонгитюдная динамика */}
            <Divider orientation="left" style={{ fontSize: 14 }}>
              <Space>📈 Динамика между сессиями</Space>
            </Divider>
            <LongitudinalTimeline
              patientId={stats.patient_id}
              lesionType="multiple_sclerosis"
            />
          </div>
        ))}
      </>
    );
  }
```

Этот блок ставится перед основным `return (<>...</>)` (который рендерит глио).

- [ ] **Шаг 10.5: Обновить `ClinicalReport.jsx` — прокинуть `lesionType`**

`ClinicalReport` получает только `runId`. Нужно также получать `lesionType`. Изменить компонент:

```jsx
const ClinicalReport = ({ runId, visible, onClose, lesionType = 'glioblastoma' }) => {
  return (
    <Modal ...>
      <ClinicalReportContent runId={runId} autoLoad={visible} lesionType={lesionType} />
    </Modal>
  );
};
```

- [ ] **Шаг 10.6: Прокинуть `lesionType` из `App.jsx`**

В `App.jsx`, найти место где вызывается `<ClinicalReport` (~строка 234). Нужно передать `lesionType`. `lesionType` можно получить из run-объекта. В `handleShowHistoryClinicalReport` хранится только `runId`. Изменить:

1. Добавить state `historyClinicalReportLesionType`:
```jsx
  const [historyClinicalReportLesionType, setHistoryClinicalReportLesionType] = useState('glioblastoma');
```

2. В `PipelineHistory` callback — передать `lesion_type` вместе с `run_id`. Изменить `onShowClinicalReport` в `PipelineHistory.jsx`:

```jsx
// В PipelineHistory.jsx, строка ~223:
onClick={() => onShowClinicalReport(record.run_id, record.lesion_type)}
```

3. В `App.jsx` обновить handler:
```jsx
  const handleShowHistoryClinicalReport = (runId, lesionType = 'glioblastoma') => {
    setHistoryClinicalReportRunId(runId);
    setHistoryClinicalReportLesionType(lesionType);
    // ...
  };
```

4. Прокинуть в компонент:
```jsx
<ClinicalReport
  runId={historyClinicalReportRunId}
  visible={!!historyClinicalReportRunId}
  onClose={() => setHistoryClinicalReportRunId(null)}
  lesionType={historyClinicalReportLesionType}
/>
```

- [ ] **Шаг 10.7: Найти `lesion_type` в `PipelineHistory.jsx`**

Убедиться, что `record.lesion_type` доступен в таблице истории. Найти место где формируется `record` (~строка где отображается список прогонов). Если `lesion_type` не включён — добавить в запрос к `GET /api/pipeline/runs` или берётся из ответа статуса.

```bash
grep -n "lesion_type\|run_id" frontend/src/components/PipelineHistory.jsx | head -20
```

Если `record.lesion_type` уже есть — ничего менять не нужно. Если нет — убедиться что endpoint истории возвращает это поле.

- [ ] **Шаг 10.8: Тест в браузере**

Запустить `cd frontend && npm run dev`. Открыть историю → запустить МС-кейс (или использовать существующий) → кликнуть «Клинический отчёт». Убедиться, что:
- Показывается МС-секция (объём, количество, лобарное распределение)
- Лонгитюдный блок появляется для пациента P000915 (2 сессии)
- Для глио-кейса показывается прежний отчёт без изменений

- [ ] **Шаг 10.9: Коммит**

```bash
git add frontend/src/components/ClinicalReportContent.jsx \
         frontend/src/components/ClinicalReport.jsx \
         frontend/src/components/PipelineHistory.jsx \
         frontend/src/App.jsx
git commit -m "feat(frontend): lesion_type-aware ClinicalReport with MS section and longitudinal timeline (KI-019)"
```

---

## Финальная верификация

- [ ] **Проверить KNOWN_ISSUES.md**

Отметить закрытые пункты:
- KI-004 → закрыт в Задаче 3
- KI-005 → закрыт в Задаче 4
- KI-016 → полностью закрыт в Задачах 1–2
- KI-019 → закрыт в Задачах 5–10

```bash
# Пример правки для каждого:
# **Branch:** `feat/lesion-type-aware-pipeline`
# добавить: **Closed in:** `feat/lesion-type-aware-pipeline`
```

- [ ] **End-to-end: МС-кейс**

Запустить полный пайплайн на P000915 с `lesion_type=multiple_sclerosis`. Проверить:
1. Stage 01: модальности t1/t2/t2fl, без t1c
2. Stage 04: в логе `expected modalities: ['t1', 't2', 't2fl']`
3. Stage 05: `processing modalities: ['t1', 't2', 't2fl']`
4. Stage 08: создан `*_lesion_stats_report.json` с `lesion_count > 0`
5. Фронт: МС-отчёт показывает объём и количество очагов
6. Фронт: лонгитюдная таблица показывает 2 сессии

- [ ] **End-to-end: глио-кейс**

Прогнать UPENN-GBM кейс. Убедиться в отсутствии регрессий:
- Stage 05 использует t1c (т.к. lesion_type=glioblastoma)
- Клинический отчёт показывает глио-данные (NCR/ED/NET/ET)
- NIfTIViewer использует 4-классовый colormap

- [ ] **Коммит KNOWN_ISSUES**

```bash
git add KNOWN_ISSUES.md
git commit -m "docs(known-issues): mark KI-004, KI-005, KI-016, KI-019 as closed in feat/lesion-type-aware-pipeline"
```

---

*Документ создан: 2026-06-02. Ветка: `feat/lesion-type-aware-pipeline`.*
