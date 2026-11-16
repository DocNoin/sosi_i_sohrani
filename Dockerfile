# Dockerfile для МТС EnergyTool ИИ-консультанта
# Этап сборки
FROM python:3.11-slim as builder

WORKDIR /app

# Установка системных зависимостей для компиляции
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Этап выполнения
FROM python:3.11-slim

WORKDIR /app

# Установка рантайм зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Копируем установленные пакеты из builder
COPY --from=builder /root/.local /root/.local

# Добавляем локальные пакеты в PATH
ENV PATH=/root/.local/bin:$PATH

# Копируем код приложения
COPY solution.py .
COPY requirements.txt .
COPY rag/ ./rag/
COPY utils/ ./utils/
COPY app/ ./app/
COPY templates/ ./templates/

# Создаем директорию для данных (будет смонтирована)
RUN mkdir -p /app/data

# Переменные окружения
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Порт для веб-интерфейса
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Запуск веб-сервера
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]