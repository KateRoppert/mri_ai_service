FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Устанавливаем Python 3.10
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Копируем код сегментации
COPY *.py .
COPY src/ ./src/

# Открываем порт 5000 (не 8080!)
EXPOSE 5000

# Запускаем Quart сервер
CMD ["python3", "simple_server.py"]