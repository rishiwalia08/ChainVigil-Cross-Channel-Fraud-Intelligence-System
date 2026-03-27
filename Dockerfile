FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch first (avoids downloading CUDA ~2GB)
RUN pip install --no-cache-dir \
    torch==2.1.0+cpu \
    torch-geometric==2.4.0 \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port (Render will override via $PORT env var)
EXPOSE 8000

# Health check using the dynamic port
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run with uvicorn, binding to $PORT (Render sets this automatically)
CMD uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 75
