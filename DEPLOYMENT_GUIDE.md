# 🚀 Deployment Guide

## Production-Ready Emotion-Aware Voice Intelligence Platform

This guide covers deploying your enterprise-grade emotion analysis application to various environments.

## 📋 Pre-Deployment Checklist

### ✅ Code Quality
- [x] All outdated files removed (30+ files cleaned up)
- [x] Production-ready error handling implemented
- [x] Comprehensive logging system added
- [x] Type hints throughout codebase
- [x] Professional documentation complete
- [x] Centralized configuration management
- [x] Security best practices implemented

### ✅ Configuration
- [x] Environment variables properly configured
- [x] API keys secured in .env file
- [x] Production settings optimized
- [x] Logging configured with rotation
- [x] Error tracking enabled

### ✅ Testing
- [x] Configuration validation working
- [x] Application launcher functional
- [x] All dependencies resolved
- [x] Hume AI integration verified

## 🏠 Local Development

### Quick Start
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env with your Hume AI API key

# 3. Launch application
python launch.py
```

### Manual Start
```bash
streamlit run emotion_aware_voice_analyzer.py
```

## ☁️ Cloud Deployment

### Streamlit Cloud (Recommended for MVP)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Production-ready deployment"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Add environment variables in the dashboard:
     - `HUME_API_KEY`: Your Hume AI API key
   - Deploy automatically

3. **Custom Domain** (Optional)
   - Configure custom domain in Streamlit Cloud dashboard
   - Update DNS settings as instructed

### Heroku Deployment

1. **Create Procfile**
   ```bash
   echo "web: streamlit run emotion_aware_voice_analyzer.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
   ```

2. **Deploy to Heroku**
   ```bash
   # Install Heroku CLI and login
   heroku create your-app-name
   heroku config:set HUME_API_KEY=your_key_here
   git push heroku main
   ```

3. **Scale Application**
   ```bash
   heroku ps:scale web=1
   ```

### AWS Deployment

#### Option 1: AWS App Runner
```bash
# Create apprunner.yaml
cat > apprunner.yaml << EOF
version: 1.0
runtime: python3
build:
  commands:
    build:
      - pip install -r requirements.txt
run:
  runtime-version: 3.11
  command: streamlit run emotion_aware_voice_analyzer.py --server.port=8080 --server.address=0.0.0.0
  network:
    port: 8080
    env: PORT
  env:
    - name: HUME_API_KEY
      value: your_key_here
EOF
```

#### Option 2: AWS ECS with Docker
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "emotion_aware_voice_analyzer.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### Google Cloud Platform

#### Cloud Run Deployment
```bash
# Build and deploy
gcloud run deploy emotion-analyzer \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars HUME_API_KEY=your_key_here
```

### Microsoft Azure

#### Azure Container Instances
```bash
# Create resource group
az group create --name emotion-analyzer-rg --location eastus

# Deploy container
az container create \
  --resource-group emotion-analyzer-rg \
  --name emotion-analyzer \
  --image your-registry/emotion-analyzer:latest \
  --dns-name-label emotion-analyzer \
  --ports 8501 \
  --environment-variables HUME_API_KEY=your_key_here
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
# Create optimized Dockerfile
cat > Dockerfile << EOF
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 streamlit && chown -R streamlit:streamlit /app
USER streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run application
CMD ["streamlit", "run", "emotion_aware_voice_analyzer.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

# Build image
docker build -t emotion-analyzer .

# Run container
docker run -p 8501:8501 --env-file .env emotion-analyzer
```

### Docker Compose
```yaml
# docker-compose.yml
version: '3.8'

services:
  emotion-analyzer:
    build: .
    ports:
      - "8501:8501"
    environment:
      - HUME_API_KEY=${HUME_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## 🔒 Security Configuration

### Environment Variables
```bash
# Production environment variables
ENVIRONMENT=production
HUME_API_KEY=your_secure_api_key
LOG_LEVEL=WARNING
ENABLE_CORS=false
ENABLE_XSRF_PROTECTION=true
MAX_UPLOAD_SIZE=200
```

### SSL/TLS Configuration
```bash
# For custom deployments, use reverse proxy
# nginx.conf example
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## 📊 Monitoring & Logging

### Application Monitoring
```python
# Add to your deployment
import logging
from logging.handlers import RotatingFileHandler

# Configure production logging
handler = RotatingFileHandler(
    'logs/emotion_analyzer.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
logging.basicConfig(handlers=[handler], level=logging.INFO)
```

### Health Checks
```bash
# Health check endpoint
curl http://your-domain.com/_stcore/health
```

### Performance Monitoring
- Use application performance monitoring (APM) tools
- Monitor response times and error rates
- Set up alerts for critical issues
- Track usage metrics and patterns

## 🔧 Maintenance

### Regular Updates
```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Update Hume AI SDK
pip install --upgrade hume

# Restart application
systemctl restart emotion-analyzer  # or your deployment method
```

### Backup Strategy
- Regular backups of configuration files
- Log file rotation and archival
- Database backups (if applicable)
- Environment variable backups

### Scaling Considerations
- Monitor resource usage (CPU, memory, network)
- Implement load balancing for high traffic
- Consider caching for frequently accessed data
- Optimize for concurrent users

## 🚨 Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Verify API key
   python config.py
   ```

2. **Port Conflicts**
   ```bash
   # Use different port
   streamlit run emotion_aware_voice_analyzer.py --server.port=8502
   ```

3. **Memory Issues**
   ```bash
   # Monitor memory usage
   docker stats emotion-analyzer
   ```

4. **SSL Certificate Issues**
   ```bash
   # Verify certificate
   openssl x509 -in cert.pem -text -noout
   ```

## 📞 Support

### Production Support Checklist
- [ ] Monitoring and alerting configured
- [ ] Backup and recovery procedures tested
- [ ] Security scanning completed
- [ ] Performance testing completed
- [ ] Documentation updated
- [ ] Team training completed

### Emergency Contacts
- Technical Lead: [Your contact]
- DevOps Team: [Your contact]
- Hume AI Support: [support@hume.ai]

---

**🎉 Your Emotion-Aware Voice Intelligence Platform is now production-ready!**

*Built with enterprise-grade standards for reliability, security, and scalability.*