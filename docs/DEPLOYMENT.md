# Deployment Guide

## Overview

This guide covers deploying the Binary Reasoning ML Pipeline in production environments. We'll cover local deployment, cloud deployment, and best practices for scaling.

## Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- 4GB RAM minimum (8GB recommended)
- 10GB disk space for models and data

## Deployment Options

### Option 1: Direct Python Deployment

#### 1. Environment Setup
```bash
# Clone repository
git clone <repository-url>
cd reasoning-ml-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

#### 2. Environment Configuration
```bash
# Copy example environment
cp .env.example .env

# Edit .env file
DEEPSEEK_API_KEY=your-api-key
MODEL_PATH=./models/
LOG_LEVEL=INFO
```

#### 3. Model Preparation
```bash
# Ensure models are trained
python src/train_binary_reasoning.py --model-type logistic
python src/train_binary_reasoning.py --model-type neural

# Verify models exist
ls models/*.pth models/*.pkl
```

#### 4. Start API Server
```bash
# Development
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

# Production with gunicorn
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Option 2: Docker Deployment

#### 1. Create Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Set environment
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/

# Expose port
EXPOSE 8000

# Run application
CMD ["gunicorn", "src.api:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
```

#### 2. Build and Run
```bash
# Build image
docker build -t reasoning-ml-api .

# Run container
docker run -d \
  --name reasoning-api \
  -p 8000:8000 \
  -e DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
  -v $(pwd)/models:/app/models \
  reasoning-ml-api
```

### Option 3: Docker Compose

#### 1. Create docker-compose.yml
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEEPSEEK_API_KEY=${DEEPSEEK_API_KEY}
      - MODEL_PATH=/app/models
      - LOG_LEVEL=INFO
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - api
    restart: unless-stopped
```

#### 2. Nginx Configuration
```nginx
events {
    worker_connections 1024;
}

http {
    upstream api {
        server api:8000;
    }

    server {
        listen 80;
        client_max_body_size 10M;

        location / {
            proxy_pass http://api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        }
    }
}
```

#### 3. Deploy with Compose
```bash
docker-compose up -d
```

## Cloud Deployment

### AWS EC2 Deployment

#### 1. Launch EC2 Instance
- AMI: Ubuntu 22.04 LTS
- Instance Type: t3.medium (minimum)
- Storage: 20GB EBS
- Security Group: Allow ports 22, 80, 443

#### 2. Instance Setup
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-instance-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose -y

# Clone repository
git clone <repository-url>
cd reasoning-ml-pipeline
```

#### 3. Deploy Application
```bash
# Set environment variables
export DEEPSEEK_API_KEY=your-key

# Build and run
sudo docker-compose up -d
```

### Google Cloud Run Deployment

#### 1. Prepare Container
```bash
# Build and tag image
docker build -t gcr.io/your-project/reasoning-api .

# Push to Container Registry
docker push gcr.io/your-project/reasoning-api
```

#### 2. Deploy to Cloud Run
```bash
gcloud run deploy reasoning-api \
  --image gcr.io/your-project/reasoning-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY
```

### Kubernetes Deployment

#### 1. Create Deployment YAML
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reasoning-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reasoning-api
  template:
    metadata:
      labels:
        app: reasoning-api
    spec:
      containers:
      - name: api
        image: your-registry/reasoning-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DEEPSEEK_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-secrets
              key: deepseek-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: reasoning-api-service
spec:
  selector:
    app: reasoning-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### 2. Deploy to Kubernetes
```bash
# Create secret
kubectl create secret generic api-secrets \
  --from-literal=deepseek-key=$DEEPSEEK_API_KEY

# Apply deployment
kubectl apply -f deployment.yaml

# Check status
kubectl get pods
kubectl get service reasoning-api-service
```

## Model Serving Optimization

### 1. Model Quantization
```python
# Reduce model size for faster loading
import torch
model = torch.load('models/neural_network.pth')
quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized, 'models/neural_network_quantized.pth')
```

### 2. ONNX Export
```python
# Export for optimized inference
import torch.onnx
dummy_input = torch.randn(1, 50, dtype=torch.long)
torch.onnx.export(
    model,
    dummy_input,
    "models/neural_network.onnx",
    export_params=True,
    opset_version=11
)
```

### 3. Model Caching
```python
# Implement in-memory model caching
from functools import lru_cache

@lru_cache(maxsize=4)
def load_model(model_name):
    return torch.load(f"models/{model_name}.pth")
```

## Production Configuration

### 1. Environment Variables
```bash
# Production .env
DEEPSEEK_API_KEY=your-production-key
MODEL_PATH=/app/models/
LOG_LEVEL=WARNING
MAX_WORKERS=4
TIMEOUT_SECONDS=30
CORS_ORIGINS=["https://your-domain.com"]
```

### 2. Logging Configuration
```python
# logging_config.py
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/api.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'default',
        },
    },
    'root': {
        'level': 'INFO',
        'handlers': ['file'],
    },
}
```

### 3. Performance Tuning
```bash
# Gunicorn configuration
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:8000"
keepalive = 120
timeout = 300
max_requests = 1000
max_requests_jitter = 50
```

## Monitoring and Health Checks

### 1. Health Check Endpoint
```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": len(loaded_models),
        "uptime_seconds": time.time() - start_time
    }
```

### 2. Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, generate_latest

prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 3. Application Monitoring
- Use AWS CloudWatch, Google Cloud Monitoring, or Datadog
- Set up alerts for:
  - High error rates (>1%)
  - High latency (>500ms)
  - Low availability (<99.9%)

## Security Best Practices

### 1. API Authentication
```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    if not validate_api_key(token):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return token
```

### 2. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("100/minute")
async def predict(request: Request):
    # Prediction logic
```

### 3. Input Validation
```python
from pydantic import BaseModel, validator

class PredictionRequest(BaseModel):
    prompt: str
    model_id: str = "logistic_regression"
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if len(v) > 1000:
            raise ValueError('Prompt too long')
        return v
```

## Backup and Recovery

### 1. Model Backup
```bash
# Backup models to S3
aws s3 sync models/ s3://your-bucket/model-backups/$(date +%Y%m%d)/
```

### 2. Database Backup (if using PostgreSQL)
```bash
# Backup database
pg_dump -h localhost -U user -d reasoning_ml > backup_$(date +%Y%m%d).sql
```

### 3. Disaster Recovery Plan
1. Keep model backups in multiple regions
2. Document model training procedures
3. Maintain infrastructure as code
4. Test recovery procedures quarterly

## Scaling Considerations

### Horizontal Scaling
- Use load balancer (nginx, HAProxy, or cloud LB)
- Deploy multiple API instances
- Use Redis for shared caching

### Vertical Scaling
- Increase instance resources for complex models
- Use GPU instances for neural models
- Optimize memory usage

### Auto-scaling
```yaml
# Kubernetes HPA
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reasoning-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reasoning-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Troubleshooting Deployment

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common deployment issues and solutions.