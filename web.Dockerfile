# syntax=docker/dockerfile:1
# 1. Берем старый образ как фундамент (там уже есть рабочие FSL и ANTs)
FROM kateroppert/mri-ai-service:latest

# Отключаем кэш .pyc и буферизацию stdout/stderr для предсказуемых логов
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# 2. Переключаемся под пользователя root, чтобы иметь права на очистку
USER root

# 3. УДАЛЯЕМ старый код полностью, чтобы он не мешался
RUN rm -rf /app/* && rm -rf /workspace/*

WORKDIR /app

# 4. Устанавливаем Node.js (он нужен для сборки вашего нового React фронтенда)
RUN apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/*


# ПОРЯДОК СЛОЁВ. Python-зависимости идут ДО фронтенда намеренно.
#
# Docker инвалидирует все слои ниже изменившегося. Когда `COPY frontend/`
# стоял выше, любая правка одной строки в React обнуляла и установку torch —
# и сборка заново тянула ~5 ГБ колёс CUDA, включая cuDNN на 658 МБ. Правки
# фронта случаются постоянно, requirements.txt меняется редко, поэтому
# дорогое и стабильное должно лежать выше дешёвого и изменчивого.
#
# Устойчивость загрузок: колёса тут стокилобайтные не бывают, а обрыв на
# середине большого файла ронял всю сборку. pip 25.1+ умеет докачивать.
# Стоит здесь, а не в начале файла: ENV инвалидирует кэш всех слоёв ниже,
# а выше — apt, Node и сборка фронта, которые пересобирать незачем.
ENV PIP_RETRIES=5
ENV PIP_TIMEOUT=60
ENV PIP_RESUME_RETRIES=5

# 6. Устанавливаем ВАШИ Python-зависимости
# В старом образе точно есть питон. Ставим поверх нужные вам библиотеки.
COPY requirements.txt .
# --mount вместо --no-cache-dir: кэш pip живёт ВНЕ образа, в кэше BuildKit.
# Размер образа от этого не растёт, зато прерванная сборка не качает заново
# то, что уже скачала. Именно --no-cache-dir делал каждую повторную попытку
# полноценной пятигигабайтной загрузкой.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --ignore-installed -r requirements.txt

# 6a. Torch, закреплённый на сборку под CUDA 12.8 — СТРОГО ДО hd-bet.
#
# Порядок здесь несущий, а не косметический. hd-bet требует лишь
# torch>=2.0.0, nnunetv2 — torch>=2.1.2,!=2.9.*; обе границы открыты сверху,
# поэтому при отсутствующем torch pip берёт самый свежий, а он собран под
# CUDA 13.0. Это давало две беды сразу:
#   1) рантайм: cu13 не стартует на драйверах старше 13.x ("The NVIDIA
#      driver on your system is too old") — ловили на баргузине;
#   2) сборка: ~3-4 ГБ колёс nvidia-*-cu13 скачивались только ради того,
#      чтобы следующим шагом быть заменёнными на cu128. С --no-cache-dir
#      это повторялось при каждой пересборке, и одного таймаута на
#      nvidia_nccl_cu13 (216 МБ) хватало, чтобы уронить многочасовую сборку.
#
# Когда нужная версия уже стоит, pip помечает требование выполненным и не
# качает ничего: проверено `pip install --dry-run hd-bet==2.0.1` в готовом
# образе — скачивается один argparse (23 КБ).
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# 6b. HD-BET — альтернативный скалстриппер для этапа 05
# (см. configs/preprocessing_config.yaml -> skull_stripping.method).
# Работает на GPU, если он проброшен в контейнер, иначе на CPU — медленно,
# но работает. Веса (~123 МБ) скачиваются при первом запуске. HD-BET 2.x
# игнорирует переменные окружения и жёстко пишет их в ~/hd-bet_params —
# этот каталог смонтирован томом в docker-compose, иначе они качались бы
# заново после каждого пересоздания контейнера.
# Torch выше уже удовлетворяет его требование — CUDA-колёса не скачиваются.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install hd-bet==2.0.1

# pyarrow приходит транзитивно, но его бинарник требует GLIBCXX_3.4.32,
# а в базовом образе максимум 3.4.30 — любой импорт sklearn (через
# nnunetv2 внутри HD-BET) падает с ImportError на libstdc++. sklearn
# импортирует pyarrow опционально, но ловит только ModuleNotFoundError,
# поэтому ImportError пробивается наружу. Ничто в проекте от pyarrow не
# зависит (pip show -> Required-by пусто), поэтому убираем.
RUN pip uninstall -y pyarrow

# 5. Собираем ВАШ НОВЫЙ фронтенд
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# 7. Копируем ВАШ НОВЫЙ код бэкенда и оркестратора
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/templates/ ./data/templates/
COPY utils/ ./utils/
COPY orchestrator.py .
COPY pipeline_config.yaml .

ENV PYTHONPATH=/app

# 8. Открываем порт
EXPOSE 8000

# Сбрасываем старый ENTRYPOINT из базового образа
ENTRYPOINT []

# Указываем явно команду запуска как ENTRYPOINT
ENTRYPOINT ["python3", "backend/app.py"]