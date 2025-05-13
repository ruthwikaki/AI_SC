#!/bin/bash
set -e

# Script to set up the ML inference server on AWS EC2
echo "Setting up ML inference server..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    unzip \
    nvidia-driver-525 \
    nvidia-cuda-toolkit \
    cmake \
    htop

# Verify NVIDIA installation
echo "Verifying NVIDIA installation..."
nvidia-smi || (echo "NVIDIA drivers not installed correctly" && exit 1)

# Create directories
mkdir -p /opt/supply-chain-llm/ml/models
mkdir -p /opt/supply-chain-llm/logs
mkdir -p /opt/supply-chain-llm/temp

# Clone the repository
git clone https://github.com/yourusername/supply-chain-llm.git /opt/supply-chain-llm/src

# Install Python dependencies
cd /opt/supply-chain-llm/src/ml
pip3 install -r requirements.txt

# Install ONNX Runtime with GPU support
pip3 install onnxruntime-gpu

# Install TensorRT (if needed)
# This may require additional steps depending on your specific CUDA version

# Set up environment variables
cat > /etc/profile.d/supply-chain-llm.sh << 'EOF'
export MODELS_DIR=/opt/supply-chain-llm/ml/models
export CONFIG_DIR=/opt/supply-chain-llm/src/config
export LOG_DIR=/opt/supply-chain-llm/logs
export TEMP_DIR=/opt/supply-chain-llm/temp
EOF

source /etc/profile.d/supply-chain-llm.sh

# Download models
cd /opt/supply-chain-llm/src
python3 scripts/model_download.py --output-dir $MODELS_DIR

# Set up systemd service for the ML inference server
cat > /etc/systemd/system/ml-inference.service << 'EOF'
[Unit]
Description=ML Inference Server
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/opt/supply-chain-llm/src/ml
ExecStart=/usr/bin/python3 inference/server.py
Restart=always
Environment="MODELS_DIR=/opt/supply-chain-llm/ml/models"
Environment="CONFIG_DIR=/opt/supply-chain-llm/src/config"
Environment="LOG_DIR=/opt/supply-chain-llm/logs"
Environment="TEMP_DIR=/opt/supply-chain-llm/temp"

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable ml-inference
sudo systemctl start ml-inference

echo "ML inference server setup complete!"