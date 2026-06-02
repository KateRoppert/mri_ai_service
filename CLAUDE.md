# Project: MRI AI Service

## Project Overview

### What this project does
[ИИ-система автоматической диагностики поражений головного мозга по МРТ-снимкам. Пайплайн сейчас состоит из последовательных этапов обработки изображений. Каждый этап сохраняет свои промежуточные результаты в папку. Сейчас есть две модели сегментации: для глиобластом и для рассеянного склероза. Мы развиваем архитектуру сейчас так, чтобы впоследствии развернуть из этой системы МАС. Поэтому модели сегментации развёрнуты как микросервисы. Система работает через докер. Есть 3 образа: веб, глио, рс. Система конфигурируется через файлы yaml]

### Key stakeholders & timeline
- **Owner/Lead**: [Kate]
- **Current deadline/milestone**: [Проходим по всем этапам и чиним баги]
- **Critical users/dependencies**: [Сервис интегрируется с Каппой - фреймворком курации датасетов. При запуске пайплайна данные автоматически загружаются в Каппу. Экспертный режим предполагает, что можно выгрузить данные из Каппы и верифицировать их через интерфейс нашего сервиса.]

## Architecture & Tech Stack

### Core technologies
- **Language**: [Python]
- **Framework**: [ React, FastAPI]
- **Database**: [PostgreSQL, MongoDB]
- **Key libraries**: [FSL, ANTs, другие библиотеки нейровизуализации, полный список в requirements.txt]

### Project structure
```
mri_ai_service/
├── agents
├── backend
│   ├── app.py
│   ├── brain_lesion.db
│   ├── config.py
│   ├── database.py
│   ├── kappa_auth.py
│   ├── kappa_client.py
│   ├── kappa_dataset_mapping.py
│   ├── kappa_uploader.py
│   ├── mask_service.py
│   ├── migrate_masks.py
│   ├── models.py
│   ├── patient_registry.py
│   ├── pipeline_manager.py
│   ├── pipeline_monitor.py
│   ├── preprocessing_version.py
│   ├── __pycache__
│   │   ├── app.cpython-312.pyc
│   │   ├── config.cpython-312.pyc
│   │   ├── database.cpython-312.pyc
│   │   ├── kappa_auth.cpython-312.pyc
│   │   ├── kappa_client.cpython-312.pyc
│   │   ├── kappa_dataset_mapping.cpython-312.pyc
│   │   ├── kappa_uploader.cpython-312.pyc
│   │   ├── mask_service.cpython-312.pyc
│   │   ├── patient_registry.cpython-312.pyc
│   │   ├── pipeline_manager.cpython-312.pyc
│   │   ├── preprocessing_version.cpython-312.pyc
│   │   ├── registry_models.cpython-312.pyc
│   │   └── validation_service.cpython-312.pyc
│   ├── registry_models.py
│   ├── requirements.txt
│   ├── schema.sql
│   ├── test_kappa_client.py
│   ├── test_kappa_uploader.py
│   ├── test_mask_service.py
│   ├── test_patient_registry.py
│   ├── test_preprocessing_version.py
│   ├── validation_service.py
│   └── websocket_manager.py
├── CLAUDE.md
├── configs
│   ├── dicom_tags.yaml
│   ├── kappa_datasets.yaml
│   ├── lobar_atlas_config.yaml
│   ├── patient_registry.json.backup
│   ├── preprocessing_config_ms60.yaml
│   ├── preprocessing_config.yaml
│   ├── preprocessing_versions.json
│   ├── quality_config.yaml
│   ├── segmentation_config.yaml
│   └── services.yaml
├── data
│   ├── MS_5
│   │   └── P000915
│   ├── templates
│   │   ├── lobar_atlas
│   │   ├── MNI152_T1_1mm.nii.gz
│   │   ├── mni_icbm152_t1_tal_nlin_sym_09a.nii
│   │   └── sri24_t1.nii.gz
│   └── UPENN-GBM
│       └── dicom
├── demo_workspace
│   └── input
│       ├── 13_03_1850
│       ├── 13_03_1900
│       └── test_01
├── docker-compose.yml
├── docs
│   ├── SPEC.md
│   └── SPEC_post_mas_cleanup.md
├── frontend
│   ├── eslint.config.js
│   ├── index.html
│   ├── package.json
│   ├── package-lock.json
│   ├── public
│   │   └── vite.svg
│   ├── README.md
│   ├── src
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── assets
│   │   ├── components
│   │   ├── index.css
│   │   ├── main.jsx
│   │   └── services
│   └── vite.config.js
├── KNOWN_ISSUES.md
├── model_weights
├── orchestrator.py
├── pipeline
│   └── run_pipeline.py
├── pipeline_config.yaml
├── __pycache__
│   └── orchestrator.cpython-312.pyc
├── README.md
├── requirements.txt
├── ROADMAP.md
├── scripts
│   ├── 01_reorganize_folders.py
│   ├── 03_convert_to_nifti.py
│   ├── 04_assess_quality.py
│   ├── 05_preprocessing.py
│   ├── 06_segmentation.py
│   ├── 07_inverse_transform.py
│   ├── 08_lobar_localization.py
│   ├── analyze_results.py
│   ├── compute_volumes.py
│   ├── gpu_monitor.py
│   ├── lobar_analysis.py
│   ├── metadata_extractor.py
│   ├── performance_monitor.py
│   ├── pipeline_validator.py
│   ├── preprocessing_steps
│   │   ├── bias_correction.py
│   │   ├── registration.py
│   │   ├── reorient.py
│   │   ├── resampling.py
│   │   └── skull_stripping.py
│   ├── __pycache__
│   │   ├── 01_reorganize_folders.cpython-312.pyc
│   │   ├── 06_segmentation.cpython-312.pyc
│   │   ├── 07_inverse_transform.cpython-312.pyc
│   │   ├── 08_lobar_localization.cpython-312.pyc
│   │   ├── metadata_extractor.cpython-312.pyc
│   │   ├── performance_monitor.cpython-312.pyc
│   │   └── pipeline_validator.cpython-312.pyc
│   ├── quality_metrics
│   │   ├── base.py
│   │   ├── cnr.py
│   │   ├── coefficient_of_variation.py
│   │   ├── efc.py
│   │   ├── fber.py
│   │   ├── gradient_sharpness.py
│   │   ├── __init__.py
│   │   ├── intensity_variance.py
│   │   ├── snr.py
│   │   ├── utils.py
│   │   └── voxel_anisotropy.py
│   ├── simulate_pipeline.py
│   ├── test_bids_scanner.py
│   ├── test_parallel_requests.py
│   ├── test_performance_monitor.py
│   └── test_segmentation_scenarios.sh
├── services
│   ├── common
│   │   ├── contracts.py
│   │   ├── gpu_monitor.py
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── service_base.py
│   ├── gbm-seg
│   │   ├── Dockerfile
│   │   ├── manifest.yaml
│   │   ├── nnUNet
│   │   ├── nnUNetv1_data
│   │   ├── nnUNetv2_data
│   │   ├── README.md
│   │   ├── requirements.txt
│   │   └── src
│   └── ms-seg
│       ├── Dockerfile
│       ├── manifest.yaml
│       ├── nnNNet_results
│       ├── nnUNet_results
│       ├── requirements.txt
│       ├── server_config.yaml
│       ├── SmallLesionMRI
│       ├── SMALLLESIONMRI_COMMIT.txt
│       └── src
├── simulation_results
│   ├── cpu_utilization_comparison.png
│   ├── makespan_comparison.png
│   ├── mas_allocation_timeline.png
│   └── simulation_comparison.csv
├── slicer
│   ├── load_in_slicer.py
│   └── slicer_agent.py
├── test_benchmark_results
│   ├── cpu_baseline_w1.json
│   ├── cpu_parallel_w4.json
│   ├── gpu_baseline_barguzin_c1.json
│   ├── gpu_baseline_cube_c1.json
│   ├── gpu_parallel_barguzin_c8.json
│   └── metrics.csv
├── test_config_loader.py
├── test_orchestrator.py
├── test_real_config.py
├── test_resampling.py
├── utils
│   └── config_loader.py
├── venv
│   ├── bin
│   │   ├── activate
│   │   ├── activate.csh
│   │   ├── activate.fish
│   │   ├── Activate.ps1
│   │   ├── ants-tools.py
│   │   ├── django-admin
│   │   ├── dotenv
│   │   ├── f2py
│   │   ├── fastapi
│   │   ├── flask
│   │   ├── fonttools
│   │   ├── httpx
│   │   ├── hypercorn
│   │   ├── nib-conform
│   │   ├── nib-convert
│   │   ├── nib-dicomfs
│   │   ├── nib-diff
│   │   ├── nib-ls
│   │   ├── nib-nifti-dx
│   │   ├── nib-roi
│   │   ├── nib-stats
│   │   ├── nib-tck2trk
│   │   ├── nib-trk2tck
│   │   ├── normalizer
│   │   ├── numpy-config
│   │   ├── parrec2nii
│   │   ├── pip
│   │   ├── pip3
│   │   ├── pip3.12
│   │   ├── __pycache__
│   │   ├── pydicom
│   │   ├── pyftmerge
│   │   ├── pyftsubset
│   │   ├── python -> python3
│   │   ├── python3 -> /usr/bin/python3
│   │   ├── python3.12 -> python3
│   │   ├── quart
│   │   ├── sqlformat
│   │   ├── ttx
│   │   └── uvicorn
│   ├── include
│   │   ├── python3.12
│   │   └── site
│   ├── lib
│   │   └── python3.12
│   ├── lib64 -> lib
│   ├── pyvenv.cfg
│   └── share
│       └── man
└── web.Dockerfile

```

### Design principles
- [Модульность, Микросервисная архитектура]
- [Система в дальнейшем будет развиваться в МАС]

## Development Setup

### Prerequisites
- Python 3.12 / Node.js v18.19.1
- [Any special tools: CUDA Version: 13.2, Docker version 29.1.3, build f52814d, etc.]

### Quick start
```bash
# Installation
docker compose --profile full up --build

# Running locally
docker compose --profile full up

# если не поднимать веб-сервис, то главный скрипт запуска пайплайна это orchestrator.py, необходимо передать требуемые аргументы
python orchestrator.py

# Running slicer agent для того, чтобы работала интеграция со слайсером для ручного редактирования сегментации экспертом
cd slicer
python slicer_agent.py
```

### Configuration
- **Environment variables**: [yaml config files]
- **Secrets management**: [Where/how secrets are stored]
- **Local overrides**: [Any local.env or similar patterns]

## Workflow & Git

### Branch strategy
- **Main branch**: main
- **Development branch**: [chore/post-mas-cleanup]
- **Branch naming**: [convention, e.g., "feature/*, fix/*, etc."]

### Pull request process
- **Code review**: [сами с тобой проверяем перед коммитами и мерджами]
- **Testing requirement**: [Must pass CI, coverage threshold, etc.]
- **Merge strategy**: [merge commit]

### Deployment
- **Deployment environments**: [dev, staging, prod]
- **Deployment process**: [Manual, automatic, approval gates]
- **Deployment frequency**: [Сейчас готовим к проду первый релиз]
- **Rollback process**: [How to rollback if needed]

## Testing & Quality

### Test structure
- **Test framework**: [pytest, unittest, jest, etc.]
- **Test locations**: [tests/, __tests__, etc.]
- **Test coverage expectation**: [в зависимости от задачи]

### Running tests
сейчас тестов мало, надо будет наверстать

### Types of tests in project
- **Unit tests**: [будем добавлять]
- **Integration tests**: [будем добавлять]
- **E2E/System tests**: [будем добавлять]

### CI/CD Pipeline
- **CI tool**: [GitHub Actions, GitLab CI, etc.]
- **Pipeline file**: [/home/ubuntu/mri_ai_service/orchestrator.py]
- **Key checks**: [What must pass: lint, tests, type-check, etc.]

## Common Tasks

### Adding a new feature
[Создаём новую ветку в гите, прорабатываем план действий, фиксируем свой SPEC.md для каждой задачи, согласовываем план перед реализацией. Реализацию делаем по шагам, проверяем, коммитим, пушим каждый сабшаг.]

### Fixing a bug
[Смотрим логи, определяем причину ошибки, предполагаем гипотезу, реализуем. Если не помогло, ищем другое решение]

### Database migrations
[/home/ubuntu/mri_ai_service/backend/brain_lesion.db - локальная БД, которая хранит историю запусков пайплайна и прочее]

### Configuration changes
[Есть основной конфиг пайплайна. Это /home/ubuntu/mri_ai_service/pipeline_config.yaml. Для каждого этапа пайплайна есть тоже свои конфиги, они лежат в папке configs проекта]

## Known Issues & Gotchas

### Performance bottlenecks
- каждый этап имеет разные точки насыщения при параллелизме. имеет место простой ресурсов и наоборот перегруз. надо оптимизировать эффективность и скорость выполнения пайплайна

### Technical debt
- [в файле KNOWN_ISSUES.md]

## Monitoring & Observability

### Logs
- **Log location/service**: [/home/ubuntu/mri_ai_service/demo_workspace/input/6_04_1443/logs - логи каждого запуска пишутся в папку запуска, вот пример. А также логи смотрим в терминале]
- **Log levels**: [What's typically logged]
- **How to debug**: [Common debugging patterns]

### Metrics
- **Key metrics to watch**: [Request latency, error rate, etc.]
- **Dashboard**: [URL or how to access]

### Alerting
- **Alert service**: [пока не реализовано]
- **On-call rotation**: [пока не реализовано]

## Dependencies & Integrations

### External APIs
- **[Service name]**: [давай заполним в ходе работы автоматически]
- **[Service name]**: [Endpoint, auth method, rate limits]

### Internal services
- **Depends on**: [давай заполним в ходе работы автоматически]
- **Used by**: [Other services that depend on this]

## Security & Compliance

### Security considerations
- [Медицинские данные чувствительны в плане безопасности, поэтому мы на первом шаге пайплайна делаем анонимизацию и извлекаем персаональную инфо из тэгов.]

### Compliance requirements
- [GDPR, HIPAA, SOC2, etc., if applicable]

## Code Style & Conventions

### Style guide
- **Linter**: [eslint, pylint, golangci-lint, etc.]
- **Formatter**: [prettier, black, gofmt, etc.]
- **Type checking**: [mypy, TypeScript, etc., if applicable]

### File organization
- [How to name files, organize modules]
- [Naming conventions: snake_case]

### Patterns to follow
- [пожалуйста, заполни автоматически в ходе работы]
- [Anti-patterns to avoid]

## Documentation

### Where to find docs
- **README**: [mri_ai_service/README.md]
- **API docs**: [URL or location]
- **Architecture docs**: [Location or wiki]
- **Runbooks**: [Location for operational procedures]

### How to keep docs updated
- [в папке docs]

## Useful Commands

```bash
# Development
[dev server start command]

# Testing
[test commands]

# Linting/Formatting
[lint and format commands]

# Deployment
[deploy command or process]

# Debugging
[useful debugging commands or procedures]
```

## Notes & Reminders

[Any project-specific notes, recurring issues, or things to remember when working on this project]
