# Troubleshooting Guide

## Common Issues and Solutions

### Dataset Generation Issues

#### 1. API Timeout Errors
**Error:**
```
TimeoutError: API request timed out after 60 seconds
```

**Solution:**
- Reduce batch size:
  ```bash
  python quick_batch.py 25  # Smaller batches
  ```
- Add delays between requests:
  ```python
  time.sleep(2)  # Add to generation loop
  ```
- Check API status: https://status.deepseek.com

#### 2. Dataset Incomplete
**Error:**
```
Generated only 681/1000 samples before timeout
```

**Solution:**
- Use incremental generation (automatically resumes):
  ```bash
  python incremental_generate.py --samples 1000 --output data/dataset.json
  ```
- Check current progress:
  ```bash
  python -c "import json; d = json.load(open('data/api_dataset_incremental.json')); print(f'Current: {len(d[\"samples\"])}/1000')"
  ```

#### 3. API Key Invalid
**Error:**
```
Error 401: Invalid API key
```

**Solution:**
- Verify API key:
  ```bash
  echo $DEEPSEEK_API_KEY
  ```
- Set correctly:
  ```bash
  export DEEPSEEK_API_KEY="sk-..."
  ```
- Check .env file:
  ```bash
  grep DEEPSEEK .env
  ```

### Model Training Issues

#### 1. CUDA Out of Memory
**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size in config:
  ```yaml
  training:
    batch_size: 16  # Reduced from 32
  ```
- Clear GPU cache:
  ```python
  torch.cuda.empty_cache()
  ```
- Use CPU instead:
  ```bash
  CUDA_VISIBLE_DEVICES="" python src/train_binary_reasoning.py
  ```

#### 2. Module Not Found
**Error:**
```
ModuleNotFoundError: No module named 'pyro'
```

**Solution:**
- Activate correct environment:
  ```bash
  pyenv shell reasonML
  # or
  source venv/bin/activate
  ```
- Install missing packages:
  ```bash
  pip install pyro-ppl
  ```
- Verify installation:
  ```bash
  pip list | grep pyro
  ```

#### 3. Tensor Device Mismatch
**Error:**
```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**Solution:**
- Check model device:
  ```python
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = model.to(device)
  data = data.to(device)
  ```
- Force CPU usage:
  ```python
  device = torch.device('cpu')
  ```

#### 4. Training Divergence
**Symptoms:**
- Loss becomes NaN
- Accuracy drops to 0 or stays at 50%

**Solution:**
- Reduce learning rate:
  ```yaml
  training:
    learning_rate: 0.0001  # Reduced from 0.001
  ```
- Check data quality:
  ```python
  # Verify labels are balanced
  python -c "import json; d = json.load(open('data/dataset.json')); print(f'Yes: {sum(1 for s in d[\"samples\"] if s[\"correct_answer\"].lower() == \"yes\")}')"
  ```
- Use gradient clipping:
  ```python
  torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  ```

### API Server Issues

#### 1. Port Already in Use
**Error:**
```
[Errno 98] Address already in use
```

**Solution:**
- Find process using port:
  ```bash
  lsof -i :8000
  ```
- Kill process:
  ```bash
  kill -9 <PID>
  ```
- Use different port:
  ```bash
  uvicorn src.api:app --port 8001
  ```

#### 2. Model Loading Failure
**Error:**
```
FileNotFoundError: models/logistic_regression.pkl not found
```

**Solution:**
- Train models first:
  ```bash
  python src/train_binary_reasoning.py --model-type logistic
  ```
- Check model directory:
  ```bash
  ls -la models/
  ```
- Update model path in config:
  ```python
  MODEL_PATH = os.getenv('MODEL_PATH', './models/')
  ```

#### 3. Slow Inference
**Symptoms:**
- API responses take >1 second
- Timeouts on predictions

**Solution:**
- Use simpler model:
  ```json
  {"model_id": "logistic_regression"}  // Fastest
  ```
- Enable model caching:
  ```python
  @lru_cache(maxsize=4)
  def load_model(name):
      return joblib.load(f"models/{name}.pkl")
  ```
- Reduce MC samples for Bayesian models:
  ```python
  mc_samples = 20  # Reduced from 50
  ```

### Deployment Issues

#### 1. Docker Build Fails
**Error:**
```
ERROR: failed to solve: rpc error: code = Unknown desc = failed to compute cache key
```

**Solution:**
- Clear Docker cache:
  ```bash
  docker system prune -a
  ```
- Build without cache:
  ```bash
  docker build --no-cache -t reasoning-api .
  ```
- Check Dockerfile paths:
  ```dockerfile
  COPY ./src ./src  # Ensure paths are correct
  ```

#### 2. Container Crashes
**Error:**
```
Container exited with code 137
```

**Solution:**
- Increase memory limits:
  ```bash
  docker run -m 4g reasoning-api
  ```
- Check logs:
  ```bash
  docker logs reasoning-api
  ```
- Add health check:
  ```dockerfile
  HEALTHCHECK --interval=30s --timeout=3s \
    CMD curl -f http://localhost:8000/health || exit 1
  ```

#### 3. Permission Denied
**Error:**
```
PermissionError: [Errno 13] Permission denied: '/app/models'
```

**Solution:**
- Fix permissions in Dockerfile:
  ```dockerfile
  RUN chmod -R 755 /app
  ```
- Use non-root user:
  ```dockerfile
  USER app
  ```
- Mount with correct permissions:
  ```bash
  docker run -v $(pwd)/models:/app/models:rw reasoning-api
  ```

### Performance Issues

#### 1. High Memory Usage
**Symptoms:**
- OOM kills
- Slow responses
- System freezing

**Solution:**
- Limit worker processes:
  ```python
  workers = 2  # Reduced from 4
  ```
- Use model quantization:
  ```python
  model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
  ```
- Implement request queuing:
  ```python
  from asyncio import Queue
  request_queue = Queue(maxsize=100)
  ```

#### 2. Database Connection Issues
**Error:**
```
psycopg2.OperationalError: could not connect to server
```

**Solution:**
- Check connection string:
  ```bash
  echo $DATABASE_URL
  ```
- Test connection:
  ```bash
  psql $DATABASE_URL -c "SELECT 1"
  ```
- Use connection pooling:
  ```python
  from sqlalchemy.pool import QueuePool
  engine = create_engine(url, poolclass=QueuePool, pool_size=5)
  ```

### Data Quality Issues

#### 1. Model Accuracy Lower Than Expected
**Symptoms:**
- Test accuracy <90% for logistic regression
- Large gap between train and test accuracy

**Solution:**
- Check dataset quality:
  ```python
  # Find mislabeled samples
  errors = [s for s in samples if s['model_answer'] != s['correct_answer']]
  print(f"Found {len(errors)} errors in dataset")
  ```
- Increase dataset size:
  ```bash
  python incremental_generate.py --samples 2000
  ```
- Balance categories:
  ```python
  # Check category distribution
  from collections import Counter
  cats = Counter(s['category'] for s in samples)
  print(cats)
  ```

#### 2. Reasoning Traces Missing
**Error:**
```
KeyError: 'reasoning_trace'
```

**Solution:**
- Regenerate affected samples:
  ```python
  incomplete = [s for s in samples if 'reasoning_trace' not in s]
  ```
- Add default trace:
  ```python
  sample['reasoning_trace'] = {'tokens': ['No reasoning available']}
  ```

### Environment Issues

#### 1. Package Version Conflicts
**Error:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution:**
- Use exact versions:
  ```bash
  pip freeze > requirements-exact.txt
  pip install -r requirements-exact.txt
  ```
- Create fresh environment:
  ```bash
  pyenv virtualenv 3.10.11 reasonML-clean
  pyenv shell reasonML-clean
  pip install -r requirements.txt
  ```

#### 2. SSL Certificate Errors
**Error:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solution:**
- Update certificates:
  ```bash
  pip install --upgrade certifi
  ```
- Temporary workaround (not for production):
  ```python
  import ssl
  ssl._create_default_https_context = ssl._create_unverified_context
  ```

## Debug Commands

### Check System Resources
```bash
# Memory usage
free -h

# GPU usage
nvidia-smi

# Disk space
df -h

# CPU usage
top
```

### Verify Installation
```bash
# Python version
python --version

# Check imports
python -c "import torch; print(torch.__version__)"
python -c "import pyro; print(pyro.__version__)"

# List installed packages
pip list
```

### Test Individual Components
```bash
# Test dataset loading
python -c "from src.data import load_dataset; data = load_dataset('data/api_dataset_incremental.json'); print(f'Loaded {len(data)} samples')"

# Test model loading
python -c "import joblib; model = joblib.load('models/binaryReasoningMath_model_sklearn.pkl'); print('Model loaded successfully')"

# Test API endpoint
curl -X GET http://localhost:8000/health
```

## Getting Help

If you encounter issues not covered here:

1. Check the [GitHub Issues](https://github.com/your-repo/issues)
2. Review the [API Documentation](API_DOCUMENTATION.md)
3. Check model-specific issues in [MODEL_ANALYSIS.md](../models/MODEL_ANALYSIS.md)
4. Enable debug logging:
   ```bash
   export LOG_LEVEL=DEBUG
   ```
5. Collect diagnostic information:
   ```bash
   python --version
   pip freeze
   nvidia-smi  # If using GPU
   docker version  # If using Docker
   ```