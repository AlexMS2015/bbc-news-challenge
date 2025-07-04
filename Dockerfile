FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY config.yaml .
COPY src/ .

RUN pip install --no-cache-dir -r requirements.txt

# Default command: run your main script
CMD ["python", "-m", "src.pipeline"]