FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and supervisor via apt (more reliable)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY data/ data/
COPY mlflow.db .
COPY mlruns/ mlruns/
# Create logs directory for supervisord
RUN mkdir -p /app/logs
# Add supervisor config
COPY supervisord.conf /etc/supervisord.conf

EXPOSE 8000 8501

CMD ["supervisord", "-c", "/etc/supervisord.conf"]
