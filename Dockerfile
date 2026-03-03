# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код
COPY . .

# Создаём папки для данных
RUN mkdir -p data logs

# Запускаем сканер (НЕ торговлю)
CMD ["python", "-u", "test_scanner.py"]
