FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Отключаем интерактивные диалоги во время установки.
# TZ обязателен: без него tzdata во время apt-get спрашивает регион и
# сборка виснет намертво (воспроизводилось на barguzin). UTC выбран, чтобы
# не менять то, в чём контейнеры уже пишут логи.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Отключаем кэш .pyc и буферизацию stdout/stderr для предсказуемых логов
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Устанавливаем Python 3.11 через репозиторий deadsnakes
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-distutils \
    python3.11-dev \
    python3-pip \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Делаем Python 3.11 версией по умолчанию для команды python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Обновляем pip именно для новой версии Python
RUN python3 -m pip install --upgrade pip

WORKDIR /app
COPY gbm-seg/requirements.txt .
RUN python3 -m pip install --no-cache-dir --ignore-installed -r requirements.txt

# Common shared code (used by all services)
COPY common/ ./common/

# GBM-specific code and resources
COPY gbm-seg/*.py .
COPY gbm-seg/src/ ./src/
COPY gbm-seg/nnUNet/ ./nnUNet/
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN cd nnUNet && pip install --timeout 120 .

# Pin torch to a CUDA 12.8 build, matching the base image above.
# Unpinned (`torch>=2.6.0`), pip installs whatever is newest, which is now
# built against CUDA 13.0 — that refuses to start on any driver older than
# 13.x ("The NVIDIA driver on your system is too old (found version
# 12080)"), as hit on barguzin. A cu128 build runs on newer drivers too,
# so one pin works everywhere. Installed last so neither nnUNet's setup.py
# nor requirements.txt can pull a default-index torch back over it.
RUN pip install --no-cache-dir \
    torch==2.11.0 torchvision \
    --index-url https://download.pytorch.org/whl/cu128

RUN mkdir -p /app/data/input /app/data/tmp /app/data/output
ENV PYTHONPATH=/app:$PYTHONPATH
EXPOSE 5000

# Запускаем сервер
CMD ["python3", "src/service_server.py"]