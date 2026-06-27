FROM python:3.11-slim

WORKDIR /app

# System deps: build tools for native wheels, curl for the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY src ./src
COPY data ./data
COPY app.py .
COPY ingest_documents.py .

# API + Streamlit ports.
EXPOSE 8000 8501

# Default: run the API. (docker-compose overrides the command for the UI service.)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
