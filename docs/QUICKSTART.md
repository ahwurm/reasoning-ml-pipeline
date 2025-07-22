# Quick Start Guide

Get the Binary Reasoning ML Pipeline running in 5 minutes!

## Prerequisites

Before starting, ensure you have:
- Python 3.10 or higher
- Git
- 4GB free RAM
- DeepSeek API key (get one at https://platform.deepseek.com)

## 1. Installation (2 minutes)

### Clone and Setup
```bash
# Clone the repository
git clone https://github.com/your-repo/reasoning-ml-pipeline.git
cd reasoning-ml-pipeline

# Run automatic setup
./quickstart.sh
```

The quickstart script will:
- Set up a Python virtual environment
- Install all dependencies
- Create necessary directories
- Verify the installation

### Manual Setup (if quickstart fails)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p models logs data
```

## 2. Configuration (1 minute)

### Set API Key
```bash
# Linux/Mac
export DEEPSEEK_API_KEY="your-api-key-here"

# Windows
set DEEPSEEK_API_KEY=your-api-key-here
```

### Create .env file (optional)
```bash
cp .env.example .env
# Edit .env and add your API key
```

## 3. Quick Test (2 minutes)

### Option A: Use Pre-trained Model (Fastest)
```bash
# Download pre-trained model
curl -L https://example.com/models/logistic_regression.pkl \
  -o models/binaryReasoningMath_model_sklearn.pkl

# Test prediction
python -c "
from src.models.binary_reasoning_math_model import BinaryReasoningMathModel
model = BinaryReasoningMathModel('logistic')
model.load('models/binaryReasoningMath_model_sklearn.pkl')
result = model.predict(['Is 25 a prime number?'])
print(f'Prediction: {result}')
"
```

### Option B: Generate Small Dataset and Train
```bash
# Generate 100 samples for testing
python incremental_generate.py --samples 100 --output data/quickstart.json

# Train simple model
python src/train_binary_reasoning.py \
  --data data/quickstart.json \
  --model-type logistic \
  --output models/quickstart_model.pkl
```

## Common Use Cases

### 1. Making Predictions via API

Start the API server:
```bash
uvicorn src.api:app --reload
```

Make a prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Is 100 > 50?",
    "model_id": "logistic_regression"
  }'
```

### 2. Batch Processing

Process multiple questions:
```python
# batch_predict.py
import requests

questions = [
    "Is 25 a prime number?",
    "Is 144 a perfect square?",
    "Is 17 × 9 = 153?"
]

response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"prompts": questions}
)

for result in response.json()["predictions"]:
    print(f"{result['prompt']} -> {result['prediction']} ({result['confidence']:.2%})")
```

### 3. Using Different Models

```python
# Try different models
models = ["logistic_regression", "neural_network", "mc_dropout"]

for model_id in models:
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "prompt": "Is 31 a prime number?",
            "model_id": model_id,
            "include_uncertainty": True
        }
    )
    result = response.json()
    print(f"{model_id}: {result['prediction']} (conf: {result['confidence']:.2%})")
```

## Quick Recipes

### Generate Full Dataset (10-15 minutes)
```bash
python incremental_generate.py --samples 1000
```

### Train All Models (30 minutes)
```bash
# Standard models
python src/train_binary_reasoning.py --model-type logistic
python src/train_binary_reasoning.py --model-type neural

# Bayesian models
python src/train_bayesian_reasoning.py --model-type mc_dropout
python src/train_bayesian_reasoning.py --model-type hierarchical
```

### Analyze Model Performance
```bash
python src/analyze_models.py
# Check models/model_comparison_analysis.png
```

### Start Production Server
```bash
gunicorn src.api:app -w 4 -k uvicorn.workers.UvicornWorker
```

## Interactive Demo

Try the interactive Jupyter notebook:
```bash
jupyter notebook notebooks/quickstart_demo.ipynb
```

Or use the Python REPL:
```python
# Python interactive demo
from src.models.binary_reasoning_math_model import BinaryReasoningMathModel

# Load model
model = BinaryReasoningMathModel("logistic")
model.load("models/binaryReasoningMath_model_sklearn.pkl")

# Interactive predictions
while True:
    question = input("Enter a yes/no math question (or 'quit'): ")
    if question.lower() == 'quit':
        break
    
    prediction = model.predict([question])[0]
    confidence = model.predict_proba([question])[0].max()
    
    print(f"Answer: {prediction} (Confidence: {confidence:.2%})\n")
```

## Next Steps

1. **Explore the API**: Visit http://localhost:8000/docs
2. **Read the documentation**: 
   - [Architecture](ARCHITECTURE.md)
   - [API Reference](API_DOCUMENTATION.md)
   - [Model Analysis](../models/MODEL_ANALYSIS.md)
3. **Train custom models**: See [Training Guide](DEPLOYMENT.md#model-training)
4. **Deploy to production**: Follow [Deployment Guide](DEPLOYMENT.md)

## Troubleshooting Quick Fixes

### "Module not found" error
```bash
pip install -r requirements.txt
```

### "API key not set" error
```bash
export DEEPSEEK_API_KEY="your-key"
```

### "Model not found" error
```bash
# Train a model first
python src/train_binary_reasoning.py --model-type logistic
```

### Port already in use
```bash
# Use different port
uvicorn src.api:app --port 8001
```

For more issues, see [Troubleshooting Guide](TROUBLESHOOTING.md).

## 🎉 Success!

You now have a working Binary Reasoning ML Pipeline! Try:
- Making predictions via the API
- Training different models
- Exploring uncertainty quantification
- Analyzing model performance

Happy reasoning! 🤖