#!/bin/bash
set -e

# Script to deploy the Supply Chain LLM application
echo "Deploying Supply Chain LLM application..."

# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    python3-pip \
    nginx \
    supervisor \
    build-essential \
    nodejs \
    npm \
    git \
    curl \
    wget \
    postgresql-client

# Create directories
mkdir -p /opt/supply-chain-llm/app
mkdir -p /opt/supply-chain-llm/logs
mkdir -p /opt/supply-chain-llm/frontend-build

# Clone the repository
git clone https://github.com/yourusername/supply-chain-llm.git /opt/supply-chain-llm/src

# Install backend dependencies
cd /opt/supply-chain-llm/src/backend
pip3 install -r requirements.txt

# Set up environment variables
cat > /etc/profile.d/supply-chain-llm-app.sh << 'EOF'
export APP_DIR=/opt/supply-chain-llm/src/backend
export CONFIG_DIR=/opt/supply-chain-llm/src/config
export LOG_DIR=/opt/supply-chain-llm/logs
export DATABASE_URL=$1
export ML_SERVER_URL=$2
export ENVIRONMENT=production
EOF

source /etc/profile.d/supply-chain-llm-app.sh

# Build frontend
cd /opt/supply-chain-llm/src/frontend
npm install
npm run build
cp -r dist/* /opt/supply-chain-llm/frontend-build/

# Configure Nginx
cat > /etc/nginx/sites-available/supply-chain-llm << 'EOF'
server {
    listen 80;
    server_name _;

    # Frontend static files
    location / {
        root /opt/supply-chain-llm/frontend-build;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable the site
ln -s /etc/nginx/sites-available/supply-chain-llm /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Set up supervisor for the backend
cat > /etc/supervisor/conf.d/supply-chain-llm.conf << 'EOF'
[program:supply-chain-llm]
command=/usr/bin/python3 /opt/supply-chain-llm/src/backend/main.py
directory=/opt/supply-chain-llm/src/backend
user=ubuntu
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/opt/supply-chain-llm/logs/backend.err.log
stdout_logfile=/opt/supply-chain-llm/logs/backend.out.log
environment=
    DATABASE_URL="$1",
    ML_SERVER_URL="$2",
    ENVIRONMENT="production"
EOF

# Reload and restart services
sudo supervisorctl reread
sudo supervisorctl update
sudo systemctl restart nginx

echo "Supply Chain LLM application deployed successfully!"