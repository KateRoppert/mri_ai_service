
FROM kateroppert/mri-ai-service:latest

WORKDIR /app

# Устанавливаем Node.js для сборки React фронтенда
RUN apt-get update && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем и собираем фронтенд
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

COPY frontend/ ./frontend/
RUN cd frontend && npm run build
# → Результат: frontend/dist/ с production build

# Копируем бэкенд
COPY backend/ ./backend/
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные компоненты
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY data/templates/ ./data/templates/
COPY utils/ ./utils/
COPY orchestrator.py .
COPY pipeline_config.yaml .

# Открываем только порт 8000 (фронтенд + бэкенд на одном порту)
EXPOSE 8000

# Запускаем бэкенд (который будет отдавать React статику)
CMD ["python", "backend/app.py"]