# Указываем базовый образ, от которого будем строить наш контейнер
FROM python:3.10-slim

# Устанавливаем рабочую директорию в контейнере
WORKDIR /app

# Копируем файлы приложения из текущей директории в контейнер
COPY app app/
COPY requirements.txt .

# Устанавливаем все зависимости, указанные в requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Устанавливаем переменные окружения
ENV OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxx

# Указываем команду, которая будет запускаться при старте контейнера
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
