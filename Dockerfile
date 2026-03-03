# Dockerfile
# --------------------------
# Use official Python slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ src/
COPY data/ data/

# Expose Streamlit port
EXPOSE 8501

# Start FastAPI in background, then Streamlit
CMD uvicorn src.mlops_tp.api:app --host 0.0.0.0 --port 8000 & \
    streamlit run src/mlops.py --server.port=8501 --server.address=0.0.0.0