#!/bin/bash
set -e

echo "Setting up Supply Chain LLM project..."

# Check for required tools
command -v docker >/dev/null 2>&1 || { echo "Docker is required but not installed. Aborting."; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed. Aborting."; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required but not installed. Aborting."; exit 1; }

# Create directories if they don't exist
mkdir -p ./ml/models/mistral/weights
mkdir -p ./ml/models/llama3/weights
mkdir -p ./ml/models/tokenizers

# Set up Python virtual environment for backend
echo "Setting up backend virtual environment..."
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# Set up Python virtual environment for ML
echo "Setting up ML virtual environment..."
cd ml
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
deactivate
cd ..

# Install frontend dependencies
echo "Setting up frontend dependencies..."
cd frontend
npm install
cd ..

# Download models (optional based on size)
read -p "Do you want to download ML models? This may take a while. (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Downloading models..."
    python3 scripts/model_download.py
fi

# Set up initial configuration
echo "Setting up configuration files..."
cp config/backend.yaml.example config/backend.yaml
cp config/ml.yaml.example config/ml.yaml
cp config/connection.yaml.example config/connection.yaml

echo "Setup complete!"
echo "To start the development environment, run: docker-compose -f deployment/docker/docker-compose.yml up"