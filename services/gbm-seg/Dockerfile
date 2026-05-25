FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Отключаем интерактивные диалоги во время установки
ENV DEBIAN_FRONTEND=noninteractive

# Отключаем кэш .pyc и буферизацию stdout/stderr для предсказуемых логов
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Устанавливаем Python 3.11 через репозиторий deadsnakes
RUN apt-get update && apt-get install -y \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
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
RUN cd nnUNet && pip install .

RUN mkdir -p /app/data/input /app/data/tmp /app/data/output
ENV PYTHONPATH=/app:$PYTHONPATH
EXPOSE 5000

# Запускаем сервер
CMD ["python3", "src/service_server.py"]