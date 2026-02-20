# Используем чистую Ubuntu 22.04 как фундамент
FROM ubuntu:22.04

# Настройки для автоматической установки пакетов
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

WORKDIR /app

# --- 1. Установка системных зависимостей, Python 3.11 и инструментов ---
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    zlib1g-dev \
    libgl1-mesa-glx \
    libglu1-mesa \
    libsm6 \
    libxrender1 \
    libxext6 \
    bc \
    dc \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-distutils \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Делаем Python 3.11 основным
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && python3 -m pip install --upgrade pip

# --- 2. Установка FSL через NeuroDebian ---
RUN apt-get update && apt-get install -y gnupg wget && \
    # Добавляем ключи и репозиторий NeuroDebian
    wget -O- http://neuro.debian.net/lists/jammy.us-nh.full | tee /etc/apt/sources.list.d/neurodebian.sources.list && \
    apt-key adv --recv-keys --keyserver hkps://keyserver.ubuntu.com 0xA5D32F012649A5A9 && \
    apt-get update && \
    # Устанавливаем fsl-core (минимальный набор без тяжелых атласов, если они не нужны)
    apt-get install -y fsl-core && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Настройка переменных окружения для FSL (в NeuroDebian пути именно такие)
ENV FSLDIR=/usr/lib/fsl/5.0
ENV PATH=${FSLDIR}:${PATH}
ENV FSLOUTPUTTYPE=NIFTI_GZ
# Важно: NeuroDebian использует скрипт для настройки, добавим его в профиль
ENV POSSUMDIR=$FSLDIR
ENV LD_LIBRARY_PATH=${FSLDIR}:${LD_LIBRARY_PATH}

# --- 3. Установка ANTs (бинарная сборка) ---
# Используем проверенную ссылку на бинарники для Ubuntu 22.04
RUN wget -q https://github.com/ANTsX/ANTs/releases/download/v2.4.3/antX-v2.4.3-Ubuntu22.04.tar.gz \
    && tar -xzf antX-v2.4.3-Ubuntu22.04.tar.gz \
    && mv install/* /usr/local/ \
    && rm antX-v2.4.3-Ubuntu22.04.tar.gz && rm -rf install

ENV ANTSPATH=/usr/local/bin/
ENV PATH=${ANTSPATH}:${PATH}

# --- 4. Установка Node.js для фронтенда ---
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs

# --- 5. Сборка React фронтенда ---
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# --- 6. Установка Python зависимостей ---
COPY requirements.txt .
# Используем флаг --break-system-packages для Ubuntu 22.04
RUN python3 -m pip install --no-cache-dir --break-system-packages -r requirements.txt

# --- 7. Копирование кода сервиса ---
COPY backend/ ./backend/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/templates/ ./data/templates/
COPY utils/ ./utils/
COPY orchestrator.py .
COPY pipeline_config.yaml .

# Порт бэкенда
EXPOSE 8000

# Запуск
CMD ["python3", "backend/app.py"]