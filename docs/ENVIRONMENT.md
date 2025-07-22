# Environment Configuration

This document details all environment variables and configuration options for the Binary Reasoning ML Pipeline.

## Environment Variables

### Required Variables

#### `DEEPSEEK_API_KEY`
- **Description**: API key for DeepSeek reasoning API
- **Required**: Yes (for dataset generation)
- **Format**: String starting with "sk-"
- **Example**: `DEEPSEEK_API_KEY=sk-1234567890abcdef`
- **Usage**: Dataset generation, reasoning trace creation

### Optional Variables

#### API Configuration

##### `DEEPSEEK_API_URL`
- **Description**: Custom DeepSeek API endpoint
- **Default**: `https://api.deepseek.com/v1`
- **Example**: `DEEPSEEK_API_URL=https://custom.deepseek.com/v1`

##### `API_TIMEOUT`
- **Description**: API request timeout in seconds
- **Default**: `60`
- **Example**: `API_TIMEOUT=120`

##### `API_MAX_RETRIES`
- **Description**: Maximum retry attempts for failed API calls
- **Default**: `3`
- **Example**: `API_MAX_RETRIES=5`

#### Model Configuration

##### `MODEL_PATH`
- **Description**: Directory containing trained models
- **Default**: `./models/`
- **Example**: `MODEL_PATH=/opt/ml/models/`

##### `DEFAULT_MODEL`
- **Description**: Default model to use for predictions
- **Default**: `logistic_regression`
- **Options**: `logistic_regression`, `neural_network`, `mc_dropout`, `hierarchical_bayesian`
- **Example**: `DEFAULT_MODEL=mc_dropout`

##### `MODEL_CACHE_SIZE`
- **Description**: Number of models to keep in memory
- **Default**: `4`
- **Example**: `MODEL_CACHE_SIZE=2`

#### Server Configuration

##### `HOST`
- **Description**: API server host address
- **Default**: `0.0.0.0`
- **Example**: `HOST=127.0.0.1`

##### `PORT`
- **Description**: API server port
- **Default**: `8000`
- **Example**: `PORT=8080`

##### `WORKERS`
- **Description**: Number of worker processes
- **Default**: `4`
- **Example**: `WORKERS=8`

##### `RELOAD`
- **Description**: Enable auto-reload in development
- **Default**: `false`
- **Example**: `RELOAD=true`

#### Logging Configuration

##### `LOG_LEVEL`
- **Description**: Logging verbosity
- **Default**: `INFO`
- **Options**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- **Example**: `LOG_LEVEL=DEBUG`

##### `LOG_FILE`
- **Description**: Log file path
- **Default**: `logs/app.log`
- **Example**: `LOG_FILE=/var/log/reasoning-api.log`

##### `LOG_FORMAT`
- **Description**: Log message format
- **Default**: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`
- **Example**: `LOG_FORMAT=%(levelname)s: %(message)s`

##### `LOG_MAX_BYTES`
- **Description**: Maximum log file size before rotation
- **Default**: `10485760` (10MB)
- **Example**: `LOG_MAX_BYTES=52428800` (50MB)

##### `LOG_BACKUP_COUNT`
- **Description**: Number of backup log files to keep
- **Default**: `5`
- **Example**: `LOG_BACKUP_COUNT=10`

#### Database Configuration (Optional)

##### `DATABASE_URL`
- **Description**: PostgreSQL connection string
- **Default**: None (uses file storage)
- **Format**: `postgresql://user:password@host:port/dbname`
- **Example**: `DATABASE_URL=postgresql://ml_user:secret@localhost/reasoning_db`

##### `DB_POOL_SIZE`
- **Description**: Database connection pool size
- **Default**: `5`
- **Example**: `DB_POOL_SIZE=10`

##### `DB_MAX_OVERFLOW`
- **Description**: Maximum overflow connections
- **Default**: `10`
- **Example**: `DB_MAX_OVERFLOW=20`

#### MLflow Configuration (Optional)

##### `MLFLOW_TRACKING_URI`
- **Description**: MLflow tracking server URL
- **Default**: `./mlruns`
- **Example**: `MLFLOW_TRACKING_URI=http://localhost:5000`

##### `MLFLOW_EXPERIMENT_NAME`
- **Description**: Default experiment name
- **Default**: `binary_reasoning`
- **Example**: `MLFLOW_EXPERIMENT_NAME=production_models`

#### Performance Tuning

##### `BATCH_SIZE`
- **Description**: Default batch size for training
- **Default**: `32`
- **Example**: `BATCH_SIZE=64`

##### `MAX_SEQUENCE_LENGTH`
- **Description**: Maximum token sequence length
- **Default**: `50`
- **Example**: `MAX_SEQUENCE_LENGTH=100`

##### `CUDA_VISIBLE_DEVICES`
- **Description**: GPU devices to use
- **Default**: All available GPUs
- **Example**: `CUDA_VISIBLE_DEVICES=0,1` (use first two GPUs)
- **Example**: `CUDA_VISIBLE_DEVICES=""` (force CPU only)

##### `OMP_NUM_THREADS`
- **Description**: OpenMP thread count
- **Default**: System default
- **Example**: `OMP_NUM_THREADS=4`

#### Security Configuration

##### `CORS_ORIGINS`
- **Description**: Allowed CORS origins
- **Default**: `["*"]` (all origins in development)
- **Format**: JSON array
- **Example**: `CORS_ORIGINS=["https://app.example.com","https://www.example.com"]`

##### `API_KEY_HEADER`
- **Description**: Header name for API authentication
- **Default**: `X-API-Key`
- **Example**: `API_KEY_HEADER=Authorization`

##### `RATE_LIMIT`
- **Description**: API rate limit per minute
- **Default**: `100`
- **Example**: `RATE_LIMIT=1000`

##### `SSL_CERT_FILE`
- **Description**: SSL certificate file path
- **Default**: None
- **Example**: `SSL_CERT_FILE=/etc/ssl/certs/server.crt`

##### `SSL_KEY_FILE`
- **Description**: SSL private key file path
- **Default**: None
- **Example**: `SSL_KEY_FILE=/etc/ssl/private/server.key`

## Configuration Files

### `.env` File Example
```bash
# API Keys
DEEPSEEK_API_KEY=sk-your-key-here

# Model Configuration
MODEL_PATH=./models/
DEFAULT_MODEL=logistic_regression

# Server Configuration
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# Database (optional)
DATABASE_URL=postgresql://user:pass@localhost/reasoning_db

# Performance
BATCH_SIZE=32
CUDA_VISIBLE_DEVICES=0

# Security
CORS_ORIGINS=["https://myapp.com"]
RATE_LIMIT=100
```

### `configs/production.yaml`
```yaml
model:
  embedding_dim: 128
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3

training:
  batch_size: 64
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10

data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  max_sequence_length: 50

experiment:
  seed: 42
  save_best_only: true
  
monitoring:
  log_interval: 10
  eval_interval: 100
  checkpoint_interval: 1000
```

## Environment-Specific Settings

### Development
```bash
# .env.development
LOG_LEVEL=DEBUG
RELOAD=true
CORS_ORIGINS=["*"]
DATABASE_URL=sqlite:///dev.db
```

### Staging
```bash
# .env.staging
LOG_LEVEL=INFO
WORKERS=2
CORS_ORIGINS=["https://staging.example.com"]
DATABASE_URL=postgresql://user:pass@staging-db/reasoning
```

### Production
```bash
# .env.production
LOG_LEVEL=WARNING
WORKERS=8
CORS_ORIGINS=["https://app.example.com"]
DATABASE_URL=postgresql://user:pass@prod-db/reasoning
SSL_CERT_FILE=/etc/ssl/certs/server.crt
SSL_KEY_FILE=/etc/ssl/private/server.key
```

## Loading Configuration

### Python Code
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Loads .env by default
# or
load_dotenv('.env.production')  # Load specific env file

# Access variables
api_key = os.getenv('DEEPSEEK_API_KEY')
model_path = os.getenv('MODEL_PATH', './models/')
log_level = os.getenv('LOG_LEVEL', 'INFO')
```

### Docker
```dockerfile
# Pass environment variables
ENV MODEL_PATH=/app/models
ENV LOG_LEVEL=INFO

# Or use env file
docker run --env-file .env.production reasoning-api
```

### Docker Compose
```yaml
services:
  api:
    env_file:
      - .env.production
    environment:
      - MODEL_PATH=/app/models
      - WORKERS=4
```

## Best Practices

### 1. Security
- Never commit `.env` files with secrets
- Use `.env.example` as a template
- Rotate API keys regularly
- Use environment-specific configurations

### 2. Organization
```
.
├── .env                  # Local development (git ignored)
├── .env.example          # Template with all variables
├── .env.test            # Test environment
├── .env.staging         # Staging environment
└── .env.production      # Production environment
```

### 3. Validation
```python
# validate_env.py
import os
import sys

required_vars = ['DEEPSEEK_API_KEY', 'MODEL_PATH']
missing = [var for var in required_vars if not os.getenv(var)]

if missing:
    print(f"Missing required environment variables: {missing}")
    sys.exit(1)
```

### 4. Documentation
- Document all environment variables
- Provide examples and defaults
- Explain the impact of each setting
- Keep `.env.example` updated

## Troubleshooting

### Variable Not Loading
```bash
# Check if variable is set
echo $DEEPSEEK_API_KEY

# Check .env file location
ls -la .env

# Debug in Python
import os
print(os.environ.keys())
```

### Wrong Configuration Loaded
```bash
# Check which env file is loaded
python -c "from dotenv import dotenv_values; print(dotenv_values())"

# Force reload
python -c "from dotenv import load_dotenv; load_dotenv(override=True)"
```

### Performance Issues
- Reduce `WORKERS` if memory constrained
- Increase `BATCH_SIZE` for better GPU utilization
- Set `CUDA_VISIBLE_DEVICES=""` to force CPU
- Tune `OMP_NUM_THREADS` for CPU performance