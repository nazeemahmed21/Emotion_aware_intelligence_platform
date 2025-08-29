9. CI/CD Pipeline Documentation
Overview
This documentation covers the complete Continuous Integration and Continuous Deployment (CI/CD) pipeline for your Emotion-Aware Voice Bot. It includes GitHub Actions workflows, Docker deployment, and cloud platform configurations.

GitHub Actions Workflows
Main CI/CD Workflow
yaml
# .github/workflows/ci-cd.yml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

env:
  DOCKER_IMAGE: emotion-bot
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    name: Run Tests
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:alpine
        ports:
          - 6379:6379
        options: --health-cmd "redis-cli ping" --health-interval 10s --health-timeout 5s --health-retries 5

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
        cache: 'pip'

    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y ffmpeg

    - name: Install Python dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
        pip install pytest pytest-asyncio pytest-cov

    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml

    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --cov=src --cov-append

    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  build-and-push:
    name: Build and Push Docker Image
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to GitHub Container Registry
      uses: docker/login-action@v2
      with:
        registry: ghcr.io
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata for Docker
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ghcr.io/${{ github.repository }}
        tags: |
          type=sha,prefix=
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

  deploy-staging:
    name: Deploy to Staging
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.event_name == 'push' && github.ref == 'refs/heads/develop'

    environment: staging
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Deploy to Staging
      uses: appleboy/ssh-action@v0.1.10
      with:
        host: ${{ secrets.STAGING_HOST }}
        username: ${{ secrets.STAGING_USER }}
        key: ${{ secrets.STAGING_SSH_KEY }}
        script: |
          cd /opt/emotion-bot
          git pull origin develop
          docker-compose -f docker-compose.staging.yml pull
          docker-compose -f docker-compose.staging.yml up -d
          docker system prune -f

  deploy-production:
    name: Deploy to Production
    runs-on: ubuntu-latest
    needs: build-and-push
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    environment: production
    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Deploy to Production
      uses: appleboy/ssh-action@v0.1.10
      with:
        host: ${{ secrets.PRODUCTION_HOST }}
        username: ${{ secrets.PRODUCTION_USER }}
        key: ${{ secrets.PRODUCTION_SSH_KEY }}
        script: |
          cd /opt/emotion-bot
          git pull origin main
          docker-compose -f docker-compose.prod.yml pull
          docker-compose -f docker-compose.prod.yml up -d
          docker system prune -f

  security-scan:
    name: Security Scan
    runs-on: ubuntu-latest
    needs: test

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      with:
        scan-type: 'fs'
        scan-ref: '.'
        format: 'sarif'
        output: 'trivy-results.sarif'

    - name: Upload Trivy scan results to GitHub Security tab
      uses: github/codeql-action/upload-sarif@v2
      with:
        sarif_file: 'trivy-results.sarif'

  performance-test:
    name: Performance Testing
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'

    steps:
    - name: Checkout code
      uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: pip install locust

    - name: Run performance tests
      run: |
        locust -f tests/performance/locustfile.py --headless -u 10 -r 5 -t 1m --host=http://localhost:8000
Docker Compose Configuration
Development Environment
yaml
# docker-compose.dev.yml
version: '3.8'

services:
  emotion-bot-api:
    build:
      context: .
      target: development
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=development
      - WHISPER_MODEL=base
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - .:/app
      - ./models:/app/models
      - ./logs:/app/logs
    depends_on:
      - redis
      - ollama
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  streamlit-app:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - FASTAPI_URL=http://emotion-bot-api:8000
    volumes:
      - .:/app
    depends_on:
      - emotion-bot-api

volumes:
  ollama_data:
  redis_data:
Production Environment
yaml
# docker-compose.prod.yml
version: '3.8'

services:
  emotion-bot-api:
    image: ghcr.io/your-username/emotion-bot:latest
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - WHISPER_MODEL=base
      - OLLAMA_HOST=http://ollama:11434
      - REDIS_URL=redis://redis:6379/0
      - MODEL_CACHE_DIR=/app/models
    volumes:
      - model_cache:/app/models
      - logs:/app/logs
    depends_on:
      - redis
      - ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2'

  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'

  redis:
    image: redis:alpine
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - emotion-bot-api
    restart: unless-stopped

volumes:
  ollama_data:
  redis_data:
  model_cache:
  logs:
Multi-stage Dockerfile
dockerfile
# Dockerfile
# Build stage
FROM python:3.9-slim as builder

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.9-slim as runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]

# Development stage
FROM runtime as development

USER root
RUN apt-get update && apt-get install -y \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

USER appuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
Nginx Configuration
nginx
# nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream emotion_bot {
        server emotion-bot-api:8000;
    }

    server {
        listen 80;
        server_name your-domain.com;
        
        # Redirect HTTP to HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/ssl/certs/your-domain.crt;
        ssl_certificate_key /etc/ssl/certs/your-domain.key;

        # SSL configuration
        ssl_protocols TLSv1.2 TLSv1.3;
        ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
        ssl_prefer_server_ciphers off;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # Proxy settings
        location / {
            proxy_pass http://emotion_bot;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket support
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Static files
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # Health check endpoint
        location /health {
            proxy_pass http://emotion_bot/health;
            access_log off;
        }
    }
}
Deployment Scripts
Automated Deployment Script
bash
#!/bin/bash
# scripts/deploy.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" >&2
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARN: $1${NC}"
}

# Check environment
ENVIRONMENT=${1:-production}
VALID_ENVIRONMENTS=("production" "staging" "development")

if [[ ! " ${VALID_ENVIRONMENTS[@]} " =~ " ${ENVIRONMENT} " ]]; then
    error "Invalid environment: $ENVIRONMENT. Valid options: ${VALID_ENVIRONMENTS[*]}"
    exit 1
fi

log "Starting deployment to $ENVIRONMENT environment..."

# Load environment-specific variables
source .env.${ENVIRONMENT}

# Check required environment variables
required_vars=("DOCKER_REGISTRY" "APP_NAME")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        error "Required environment variable $var is not set"
        exit 1
    fi
done

# Build and push Docker images
log "Building Docker images..."
docker-compose -f docker-compose.${ENVIRONMENT}.yml build

if [ "$ENVIRONMENT" != "development" ]; then
    log "Pushing images to registry..."
    docker-compose -f docker-compose.${ENVIRONMENT}.yml push
fi

# Deployment based on environment
case $ENVIRONMENT in
    production)
        deploy_production
        ;;
    staging)
        deploy_staging
        ;;
    development)
        deploy_development
        ;;
esac

log "Deployment to $ENVIRONMENT completed successfully!"

deploy_production() {
    log "Deploying to production..."
    
    # SSH to production server and deploy
    ssh -i "$PRODUCTION_SSH_KEY" "$PRODUCTION_USER@$PRODUCTION_HOST" << EOF
        cd /opt/emotion-bot
        git pull origin main
        docker-compose -f docker-compose.prod.yml pull
        docker-compose -f docker-compose.prod.yml up -d --force-recreate
        
        # Run migrations if any
        docker-compose -f docker-compose.prod.yml exec emotion-bot-api python manage.py migrate
        
        # Clean up old images
        docker system prune -af
        
        # Verify deployment
        sleep 10
        curl -f http://localhost:8000/health || exit 1
EOF
    
    if [ $? -eq 0 ]; then
        log "Production deployment verified successfully"
    else
        error "Production deployment failed"
        exit 1
    fi
}

deploy_staging() {
    log "Deploying to staging..."
    
    # Similar to production but with staging credentials
    ssh -i "$STAGING_SSH_KEY" "$STAGING_USER@$STAGING_HOST" << EOF
        cd /opt/emotion-bot
        git pull origin develop
        docker-compose -f docker-compose.staging.yml pull
        docker-compose -f docker-compose.staging.yml up -d --force-recreate
        docker system prune -af
        sleep 10
        curl -f http://localhost:8000/health || exit 1
EOF
}

deploy_development() {
    log "Starting development environment..."
    docker-compose -f docker-compose.dev.yml up -d --build
}
Environment Configuration Files
Production Environment
bash
# .env.production
# Docker
DOCKER_REGISTRY=ghcr.io
APP_NAME=emotion-bot

# Deployment
PRODUCTION_HOST=your-production-server.com
PRODUCTION_USER=deploy
PRODUCTION_SSH_KEY=~/.ssh/production_key

# Application
ENVIRONMENT=production
WHISPER_MODEL=base
OLLAMA_HOST=http://ollama:11434
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=your-production-secret-key
JWT_SECRET=your-jwt-secret-key

# Monitoring
SENTRY_DSN=your-sentry-dsn
PROMETHEUS_MULTIPROC_DIR=/tmp
Staging Environment
bash
# .env.staging
# Docker
DOCKER_REGISTRY=ghcr.io
APP_NAME=emotion-bot-staging

# Deployment
STAGING_HOST=your-staging-server.com
STAGING_USER=deploy
STAGING_SSH_KEY=~/.ssh/staging_key

# Application
ENVIRONMENT=staging
WHISPER_MODEL=base
OLLAMA_HOST=http://ollama:11434
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=your-staging-secret-key
JWT_SECRET=your-staging-jwt-secret

# Monitoring
SENTRY_DSN=your-staging-sentry-dsn
Cloud Deployment Configurations
Railway.app Configuration
yaml
# railway.json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS",
    "buildCommand": "pip install -r requirements.txt && python -m pip install --upgrade pip"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  },
  "env": {
    "ENVIRONMENT": "production",
    "WHISPER_MODEL": "base",
    "PYTHON_VERSION": "3.9.0"
  }
}
Render.com Configuration
yaml
# render.yaml
services:
  - type: web
    name: emotion-bot-api
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: ENVIRONMENT
        value: production
      - key: WHISPER_MODEL
        value: base
    healthCheckPath: /health
    autoDeploy: true

  - type: web
    name: emotion-bot-streamlit
    env: python
    plan: free
    buildCommand: pip install -r requirements-streamlit.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    envVars:
      - key: FASTAPI_URL
        value: https://emotion-bot-api.onrender.com
Hugging Face Spaces Configuration
yaml
# huggingface/spaces.yaml
title: Emotion-Aware Voice Bot
sdk: docker
app_port: 8501
models:
  - whisper
  - emotion-detection

# Hardware requirements
hardware:
  cpu: 4
  memory: 16GB
  gpu: true

# Environment variables
env:
  - name: WHISPER_MODEL
    value: base
  - name: ENVIRONMENT
    value: production

# Build instructions
build:
  context: .
  dockerfile: Dockerfile.huggingface
Monitoring and Alerting
Prometheus Configuration
yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'emotion-bot'
    static_configs:
      - targets: ['emotion-bot-api:8000']
    metrics_path: /metrics
    scrape_interval: 5s

  - job_name: 'docker'
    static_configs:
      - targets: ['docker-engine:9323']
    
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

rule_files:
  - alerts.yml
Alert Rules
yaml
# monitoring/alerts.yml
groups:
- name: emotion-bot-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 10% for the last 5 minutes"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is above 2 seconds"

  - alert: ServiceDown
    expr: up == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Service is down"
      description: "{{ $labels.instance }} is not responding"

  - alert: HighMemoryUsage
    expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage"
      description: "Memory usage is above 90%"
Backup and Recovery
Backup Script
bash
#!/bin/bash
# scripts/backup.sh

set -e

# Configuration
BACKUP_DIR="/backups/emotion-bot"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RETENTION_DAYS=7

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Create backup directory
mkdir -p "$BACKUP_DIR/$TIMESTAMP"

log "Starting backup process..."

# Backup conversation data
log "Backing up conversation data..."
docker-compose exec emotion-bot-api python -c "
import json
from main import conversation_history
with open('/tmp/conversations.json', 'w') as f:
    json.dump(conversation_history, f)
"
docker cp emotion-bot-api:/tmp/conversations.json "$BACKUP_DIR/$TIMESTAMP/"

# Backup models
log "Backing up models..."
tar -czf "$BACKUP_DIR/$TIMESTAMP/models.tar.gz" ./models

# Backup Ollama models
log "Backing up Ollama models..."
docker-compose exec ollama ollama list > "$BACKUP_DIR/$TIMESTAMP/ollama_models.txt"

# Backup database (if using)
# docker-compose exec db pg_dump -U postgres emotion_bot > "$BACKUP_DIR/$TIMESTAMP/database.sql"

# Create backup manifest
cat > "$BACKUP_DIR/$TIMESTAMP/manifest.json" << EOF
{
  "timestamp": "$TIMESTAMP",
  "components": [
    "conversations",
    "models",
    "ollama_models"
  ],
  "size": "$(du -sh "$BACKUP_DIR/$TIMESTAMP" | cut -f1)"
}
EOF

log "Backup completed: $BACKUP_DIR/$TIMESTAMP"

# Clean up old backups
log "Cleaning up old backups..."
find "$BACKUP_DIR" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} \;

log "Backup process completed successfully"
Recovery Script
bash
#!/bin/bash
# scripts/restore.sh

set -e

BACKUP_DIR="/backups/emotion-bot"
LATEST_BACKUP=$(ls -t "$BACKUP_DIR" | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "No backups found"
    exit 1
fi

echo "Restoring from backup: $LATEST_BACKUP"

# Restore conversation data
echo "Restoring conversation data..."
docker cp "$BACKUP_DIR/$LATEST_BACKUP/conversations.json" emotion-bot-api:/tmp/
docker-compose exec emotion-bot-api python -c "
import json
from main import conversation_history
with open('/tmp/conversations.json', 'r') as f:
    data = json.load(f)
conversation_history.clear()
conversation_history.extend(data)
"

# Restore models
echo "Restoring models..."
tar -xzf "$BACKUP_DIR/$LATEST_BACKUP/models.tar.gz" -C ./

# Restore Ollama models (would need to be pulled again)
echo "Please manually restore Ollama models from: $BACKUP_DIR/$LATEST_BACKUP/ollama_models.txt"

echo "Restore completed successfully"
This comprehensive CI/CD pipeline documentation provides everything you need to automate testing, building, and deployment of your Emotion-Aware Voice Bot across different environments, with proper monitoring, backup, and recovery procedures.