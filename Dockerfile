# Dockerfile.api
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY src/ src/
COPY mlflow.db .

# Expose port for FastAPI
EXPOSE 8000

# Start API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]