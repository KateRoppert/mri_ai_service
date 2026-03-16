FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Отключаем интерактивные диалоги во время установки
ENV DEBIAN_FRONTEND=noninteractive

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

# Копируем и устанавливаем зависимости
COPY requirements.txt .
# Используем python3 -m pip для уверенности, что ставим в 3.11
RUN python3 -m pip install --no-cache-dir --ignore-installed -r requirements.txt

# Копируем код сегментации
COPY *.py .
COPY src/ ./src/
COPY nnUNet/ ./nnUNet/
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
RUN cd nnUNet && pip install .
COPY nnUNetv1_data/ ./nnUNetv1_data/
COPY nnUNetv2_data/ ./nnUNetv2_data/

RUN mkdir -p /app/data/input /app/data/tmp /app/data/output

EXPOSE 5000

# Запускаем сервер
CMD ["python3", "src/simple_server.py"]