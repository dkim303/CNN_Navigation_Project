FROM python:3.12-slim

# Prevent Python from creating .pyc files and buffer-free logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install Linux dependencies needed by many ML/image packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependencies first so Docker can cache this layer
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Run as a non-root user
RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["python", "-m", "src.train"]