FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p /app/data /app/logs
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "main.py"]
