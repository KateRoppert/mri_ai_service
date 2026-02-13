FROM python:3.10-slim
   WORKDIR /app
   COPY frontend/ ./frontend/
   COPY backend/ ./backend/
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   CMD ["python", "start_servers.py"]