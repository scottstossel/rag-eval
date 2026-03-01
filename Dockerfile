# Stage 1: Build environment
FROM python:3.11-slim as builder

WORKDIR /build

# Install system dependencies for compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first (leverage Docker layer caching)
COPY requirements.txt .

# Install Python dependencies to /install
RUN pip install --no-cache-dir --prefix=/install --break-system-packages \
    -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY config.yaml .
COPY run_eval.py .

# Create non-root user (OpenShift security requirement)
RUN useradd -m -u 1001 evaluator && \
    chown -R evaluator:evaluator /app && \
    mkdir -p /data && \
    chown evaluator:evaluator /data

USER evaluator

# Environment variables (override at runtime)
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY=""

# Entrypoint
ENTRYPOINT ["python", "run_eval.py"]
CMD ["--config", "config.yaml"]

# Labels for metadata (useful in registries)
LABEL org.opencontainers.image.title="RAG Evaluation Pipeline"
LABEL org.opencontainers.image.description="Shift-aware RAG evaluation for OpenShift AI"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.authors="Your Name <your.email@example.com>"