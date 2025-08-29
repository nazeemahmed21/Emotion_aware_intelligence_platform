7. Deployment and Production Setup
Overview
This section covers deploying your Emotion-Aware Voice Bot to production environments using Docker, cloud platforms, and ensuring optimal performance and reliability.

Docker Deployment
Complete Dockerfile
dockerfile
# Dockerfile
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=on \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
Docker Compose for Full Stack
yaml
# docker-compose.yml
version: '3.8'

services:
  emotion-bot-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - WHISPER_MODEL=base
      - OLLAMA_HOST=http://ollama:11434
      - MODEL_CACHE_DIR=/app/models
    volumes:
      - model_cache:/app/models
      - ./logs:/app/logs
    depends_on:
      - ollama
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  ollama:
    image: ollama/ollama:latest
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    environment:
      - OLLAMA_HOST=0.0.0.0
    restart: unless-stopped

  streamlit-app:
    build:
      context: .
      dockerfile: Dockerfile.streamlit
    ports:
      - "8501:8501"
    environment:
      - FASTAPI_URL=http://emotion-bot-api:8000
    depends_on:
      - emotion-bot-api
    restart: unless-stopped

volumes:
  ollama_data:
  model_cache:
Streamlit-specific Dockerfile
dockerfile
# Dockerfile.streamlit
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-streamlit.txt .
RUN pip install --no-cache-dir -r requirements-streamlit.txt

COPY streamlit_app/ .

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
Cloud Deployment Options
1. Hugging Face Spaces (Free Tier)
yaml
# huggingface/spaces.yaml
title: Emotion-Aware Voice Bot
sdk: docker
app_port: 8501
models:
  - whisper
  - emotion-detection

# requirements.txt for Hugging Face
streamlit==1.28.0
openai-whisper==20231117
librosa==0.10.1
pyAudioAnalysis==0.3.14
requests==2.31.0
numpy==1.24.3
2. Railway.app Deployment
yaml
# railway.json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
3. Render.com Deployment
yaml
# render.yaml
services:
  - type: web
    name: emotion-bot-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: 3.9.0
      - key: WHISPER_MODEL
        value: base
Environment Configuration
Production Environment File
python
# config/production.py
import os
from pathlib import Path

class ProductionConfig:
    # Application
    DEBUG = False
    TESTING = False
    
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    MODEL_DIR = BASE_DIR / "models"
    LOG_DIR = BASE_DIR / "logs"
    
    # Ensure directories exist
    MODEL_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)
    
    # Model paths
    WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
    EMOTION_MODEL_PATH = MODEL_DIR / "emotion_model.pkl"
    
    # API settings
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    REQUEST_TIMEOUT = 30  # seconds
    
    # CORS
    ALLOWED_ORIGINS = [
        "http://localhost:8501",
        "https://your-domain.com",
    ]
    
    # Rate limiting
    RATE_LIMIT = "100/minute"
Environment Management
python
# config/__init__.py
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_config():
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "production":
        from .production import ProductionConfig
        return ProductionConfig()
    elif env == "testing":
        from .testing import TestingConfig
        return TestingConfig()
    else:
        from .development import DevelopmentConfig
        return DevelopmentConfig()

config = get_config()
Performance Optimization
Model Caching and Preloading
python
# utils/model_loader.py
import threading
import time
from functools import lru_cache
import whisper
import joblib

class ModelLoader:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._models = {}
                cls._instance._load_models()
        return cls._instance
    
    def _load_models(self):
        """Preload models on startup"""
        print("Preloading models...")
        
        # Load Whisper model
        try:
            self._models['whisper'] = whisper.load_model("base")
            print("✓ Whisper model loaded")
        except Exception as e:
            print(f"✗ Failed to load Whisper model: {e}")
        
        # Load emotion model
        try:
            emotion_model_path = "models/emotion_model.pkl"
            self._models['emotion'] = joblib.load(emotion_model_path)
            print("✓ Emotion model loaded")
        except Exception as e:
            print(f"✗ Failed to load emotion model: {e}")
    
    @lru_cache(maxsize=1)
    def get_whisper_model(self):
        return self._models.get('whisper')
    
    @lru_cache(maxsize=1)
    def get_emotion_model(self):
        return self._models.get('emotion')

# Singleton instance
model_loader = ModelLoader()
Async Processing with Celery
python
# tasks/celery_worker.py
from celery import Celery
from celery.signals import worker_ready
import whisper
from emotion_detection import detect_emotion
from ollama_response import generate_empathetic_response

# Celery configuration
app = Celery('emotion_bot')
app.conf.update(
    broker_url='redis://localhost:6379/0',
    result_backend='redis://localhost:6379/0',
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    task_routes={
        'tasks.process_audio': {'queue': 'audio_processing'},
        'tasks.generate_response': {'queue': 'response_generation'},
    }
)

@worker_ready.connect
def preload_models(sender, **kwargs):
    """Preload models when worker starts"""
    print("Preloading models in Celery worker...")
    sender.app.whisper_model = whisper.load_model("base")

@app.task
def process_audio(audio_path):
    """Process audio file asynchronously"""
    try:
        # Transcribe audio
        transcription = transcribe_audio(audio_path)
        
        # Extract features and detect emotion
        features = extract_audio_features(audio_path)
        emotion, confidence = detect_emotion(features)
        
        return {
            'transcription': transcription,
            'emotion': emotion,
            'confidence': confidence,
            'features': features
        }
    except Exception as e:
        raise self.retry(exc=e, countdown=60)

@app.task
def generate_response(transcription, emotion, confidence):
    """Generate response asynchronously"""
    try:
        response = generate_empathetic_response(transcription, emotion, confidence)
        return {'response': response}
    except Exception as e:
        raise self.retry(exc=e, countdown=30)
Monitoring and Logging
Structured Logging
python
# utils/logger.py
import logging
import json
from pythonjsonlogger import jsonlogger
from datetime import datetime

class StructuredLogger:
    def __init__(self, name, level=logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(levelname)s %(name)s %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(console_handler)
    
    def info(self, message, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def error(self, message, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def warning(self, message, **kwargs):
        self.logger.warning(message, extra=kwargs)

# Global logger instance
logger = StructuredLogger("emotion-bot")
Performance Monitoring
python
# utils/monitoring.py
import time
from functools import wraps
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
REQUEST_COUNT = Counter('request_count', 'Total API requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('request_latency', 'Request latency in seconds', ['endpoint'])
PROCESSING_TIME = Histogram('processing_time', 'Audio processing time', ['stage'])

def monitor_requests(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            response = await func(*args, **kwargs)
            REQUEST_COUNT.labels(
                method=getattr(args[0], 'method', 'UNKNOWN'),
                endpoint=func.__name__
            ).inc()
            return response
        finally:
            latency = time.time() - start_time
            REQUEST_LATENCY.labels(endpoint=func.__name__).observe(latency)
    return wrapper

def track_processing_time(stage):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                PROCESSING_TIME.labels(stage=stage).observe(duration)
        return wrapper
    return decorator
Security Configuration
API Security Middleware
python
# security/middleware.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta

class JWTBearer(HTTPBearer):
    def __init__(self, auto_error: bool = True):
        super().__init__(auto_error=auto_error)
    
    async def __call__(self, request: Request):
        credentials: HTTPAuthorizationCredentials = await super().__call__(request)
        if credentials:
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(status_code=403, detail="Invalid token")
            return credentials.credentials
        else:
            raise HTTPException(status_code=403, detail="Invalid authorization code")
    
    def verify_jwt(self, token: str) -> bool:
        try:
            payload = jwt.decode(token, "SECRET_KEY", algorithms=["HS256"])
            return payload is not None
        except:
            return False

def create_jwt_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=1)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, "SECRET_KEY", algorithm="HS256")
Rate Limiting
python
# security/rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address)

def setup_rate_limiting(app):
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
Health Checks and Readiness Probes
Comprehensive Health Check
python
# health/check.py
import asyncio
from datetime import datetime
from typing import Dict, List
import whisper
import requests

class HealthChecker:
    def __init__(self):
        self.checks = {
            'api': self.check_api,
            'models': self.check_models,
            'storage': self.check_storage,
            'external_services': self.check_external_services
        }
    
    async def check_api(self) -> Dict:
        return {"status": "healthy", "timestamp": datetime.now()}
    
    async def check_models(self) -> Dict:
        try:
            # Check if models are loaded
            whisper_model = whisper.load_model("base", device="cpu")
            return {
                "status": "healthy",
                "whisper_loaded": True,
                "timestamp": datetime.now()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def check_storage(self) -> Dict:
        try:
            # Check if storage is writable
            with open("/tmp/healthcheck", "w") as f:
                f.write("test")
            return {"status": "healthy", "timestamp": datetime.now()}
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def check_external_services(self) -> Dict:
        try:
            # Check Ollama service
            response = requests.get("http://ollama:11434/api/tags", timeout=5)
            return {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "ollama_status": response.status_code,
                "timestamp": datetime.now()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now()
            }
    
    async def run_all_checks(self) -> Dict:
        results = {}
        for check_name, check_func in self.checks.items():
            results[check_name] = await check_func()
        
        # Overall status
        all_healthy = all(result["status"] == "healthy" for result in results.values())
        results["overall"] = {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.now()
        }
        
        return results

# Global health checker
health_checker = HealthChecker()
Deployment Scripts
Automated Deployment Script
bash
#!/bin/bash
# deploy.sh

set -e

echo "🚀 Starting deployment..."

# Environment check
if [ -z "$ENVIRONMENT" ]; then
    echo "❌ ENVIRONMENT variable not set"
    exit 1
fi

# Build Docker images
echo "📦 Building Docker images..."
docker-compose build

# Run tests
echo "🧪 Running tests..."
docker-compose run --rm emotion-bot-api python -m pytest tests/ -v

# Deploy based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "🚀 Deploying to production..."
    docker-compose -f docker-compose.prod.yml up -d --force-recreate
    
    # Run migrations if any
    docker-compose exec emotion-bot-api python manage.py migrate
    
elif [ "$ENVIRONMENT" = "staging" ]; then
    echo "🔧 Deploying to staging..."
    docker-compose -f docker-compose.staging.yml up -d --force-recreate
    
else
    echo "💻 Starting development environment..."
    docker-compose up -d
fi

echo "✅ Deployment completed successfully!"
Backup and Restore Script
bash
#!/bin/bash
# backup.sh

set -e

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/emotion-bot_$TIMESTAMP"

echo "📦 Creating backup: $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"

# Backup models
echo "💾 Backing up models..."
tar -czf "$BACKUP_DIR/models.tar.gz" ./models

# Backup conversation history
echo "💾 Backing up conversation data..."
docker-compose exec emotion-bot-api python -c "
import json
from main import conversation_history
with open('/tmp/conversations.json', 'w') as f:
    json.dump(conversation_history, f)
"
docker cp emotion-bot-api:/tmp/conversations.json "$BACKUP_DIR/"

# Backup Ollama models
echo "💾 Backing up Ollama models..."
docker-compose exec ollama ollama list > "$BACKUP_DIR/ollama_models.txt"

echo "✅ Backup completed: $BACKUP_DIR"
This comprehensive deployment setup ensures your Emotion-Aware Voice Bot is production-ready with proper monitoring, security, and scalability configurations.

