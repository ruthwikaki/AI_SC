FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY ./ml/requirements.txt /app/ml/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r /app/ml/requirements.txt

# Install ONNX Runtime with GPU support
RUN pip3 install onnxruntime-gpu

# Copy config
COPY ./config /app/config

# Copy ML code
COPY ./ml /app/ml

# Set environment variables
ENV PYTHONPATH=/app
ENV MODELS_DIR=/app/models
ENV CONFIG_DIR=/app/config

# Create model directory
RUN mkdir -p /app/models

# Expose port
EXPOSE 8001

# Run the ML server
CMD ["python3", "/app/ml/inference/server.py"]