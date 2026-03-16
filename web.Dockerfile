# 1. Берем старый образ как фундамент (там уже есть рабочие FSL и ANTs)
FROM kateroppert/mri-ai-service:latest

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

# 5. Собираем ВАШ НОВЫЙ фронтенд
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install
COPY frontend/ ./frontend/
RUN cd frontend && npm run build

# 6. Устанавливаем ВАШИ Python-зависимости
# В старом образе точно есть питон. Ставим поверх нужные вам библиотеки.
COPY requirements.txt .
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

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