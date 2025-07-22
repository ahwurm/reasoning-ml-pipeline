# API Documentation

## Overview

The Binary Reasoning ML Pipeline provides a RESTful API for making predictions on binary mathematical reasoning tasks. The API supports multiple model types with uncertainty quantification capabilities.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication for local development. Production deployment should implement API key authentication.

```http
Authorization: Bearer <api_key>
```

## Endpoints

### Health Check

#### `GET /health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-07-15T10:30:00Z",
  "version": "1.0.0"
}
```

### Model Information

#### `GET /models`

List all available models with their characteristics.

**Response:**
```json
{
  "models": [
    {
      "id": "logistic_regression",
      "name": "Logistic Regression",
      "accuracy": 0.98,
      "supports_uncertainty": false,
      "inference_time_ms": 5
    },
    {
      "id": "neural_network",
      "name": "Neural Network",
      "accuracy": 0.787,
      "supports_uncertainty": false,
      "inference_time_ms": 20
    },
    {
      "id": "mc_dropout",
      "name": "MC Dropout Bayesian",
      "accuracy": 0.827,
      "supports_uncertainty": true,
      "inference_time_ms": 100
    },
    {
      "id": "hierarchical_bayesian",
      "name": "Hierarchical Bayesian",
      "accuracy": 0.447,
      "supports_uncertainty": true,
      "inference_time_ms": 150
    }
  ]
}
```

### Single Prediction

#### `POST /predict`

Make a prediction for a single binary reasoning question.

**Request Body:**
```json
{
  "prompt": "Is 153 ÷ 17 = 9?",
  "model_id": "logistic_regression",
  "include_reasoning": true,
  "include_uncertainty": false
}
```

**Parameters:**
- `prompt` (required): The binary question to answer
- `model_id` (optional): Model to use (default: "logistic_regression")
- `include_reasoning` (optional): Include reasoning trace (default: false)
- `include_uncertainty` (optional): Include uncertainty estimates (default: false)

**Response:**
```json
{
  "prediction": "yes",
  "confidence": 0.95,
  "reasoning_trace": [
    "Let me check this calculation",
    "I need to compute 153 ÷ 17",
    "The result is 9",
    "The given answer is 9",
    "These match"
  ],
  "model_used": "logistic_regression",
  "inference_time_ms": 5
}
```

**Response with Uncertainty (Bayesian models):**
```json
{
  "prediction": "yes",
  "confidence": 0.85,
  "uncertainty": {
    "total": 0.015,
    "epistemic": 0.003,
    "aleatoric": 0.012
  },
  "confidence_interval": [0.82, 0.88],
  "model_used": "mc_dropout",
  "inference_time_ms": 100
}
```

### Batch Prediction

#### `POST /predict/batch`

Make predictions for multiple questions.

**Request Body:**
```json
{
  "prompts": [
    "Is 153 ÷ 17 = 9?",
    "Is 25 a prime number?",
    "Is 100 > 99?"
  ],
  "model_id": "logistic_regression",
  "include_uncertainty": false
}
```

**Response:**
```json
{
  "predictions": [
    {
      "prompt": "Is 153 ÷ 17 = 9?",
      "prediction": "yes",
      "confidence": 0.95
    },
    {
      "prompt": "Is 25 a prime number?",
      "prediction": "no",
      "confidence": 0.99
    },
    {
      "prompt": "Is 100 > 99?",
      "prediction": "yes",
      "confidence": 0.999
    }
  ],
  "model_used": "logistic_regression",
  "total_inference_time_ms": 15
}
```

### Category-Specific Prediction

#### `POST /predict/category`

Make a prediction with category information (for hierarchical model).

**Request Body:**
```json
{
  "prompt": "Is 153 ÷ 17 = 9?",
  "category": "math_verification",
  "model_id": "hierarchical_bayesian"
}
```

**Categories:**
- `math_verification`: Arithmetic calculations
- `comparison`: Numerical comparisons
- `number_property`: Prime, divisibility checks
- `pattern_recognition`: Sequence patterns

**Response:**
```json
{
  "prediction": "yes",
  "confidence": 0.75,
  "category_performance": {
    "category": "math_verification",
    "category_accuracy": 0.471,
    "category_uncertainty": 0.693
  },
  "uncertainty": {
    "total": 0.693,
    "epistemic": 0.049,
    "aleatoric": 0.644
  }
}
```

### Model Training

#### `POST /train`

Trigger a new model training job (requires authentication).

**Request Body:**
```json
{
  "model_type": "neural_network",
  "dataset_path": "data/api_dataset_incremental.json",
  "config": {
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

**Response:**
```json
{
  "job_id": "train_12345",
  "status": "started",
  "estimated_time_minutes": 30
}
```

#### `GET /train/{job_id}`

Check training job status.

**Response:**
```json
{
  "job_id": "train_12345",
  "status": "completed",
  "metrics": {
    "final_accuracy": 0.82,
    "best_validation_accuracy": 0.85,
    "training_time_minutes": 25
  },
  "model_path": "models/neural_network_12345.pth"
}
```

## Error Responses

### 400 Bad Request
```json
{
  "error": "Invalid request",
  "message": "The 'prompt' field is required",
  "code": "MISSING_FIELD"
}
```

### 404 Not Found
```json
{
  "error": "Model not found",
  "message": "Model 'unknown_model' does not exist",
  "code": "MODEL_NOT_FOUND"
}
```

### 422 Unprocessable Entity
```json
{
  "error": "Invalid input",
  "message": "Prompt must be a binary yes/no question",
  "code": "INVALID_PROMPT"
}
```

### 500 Internal Server Error
```json
{
  "error": "Internal server error",
  "message": "An unexpected error occurred",
  "code": "INTERNAL_ERROR"
}
```

## Rate Limiting

API implements rate limiting to prevent abuse:
- **Anonymous users**: 100 requests per minute
- **Authenticated users**: 1000 requests per minute

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1628789120
```

## Versioning

The API uses URL versioning. Current version: v1

Future versions:
```
http://localhost:8000/v2/predict
```

## WebSocket Support (Planned)

For real-time predictions:
```javascript
ws://localhost:8000/ws/predict

// Send
{
  "type": "predict",
  "prompt": "Is 100 > 50?",
  "model_id": "logistic_regression"
}

// Receive
{
  "type": "prediction",
  "result": "yes",
  "confidence": 0.999
}
```

## SDK Examples

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={
        "prompt": "Is 25 a prime number?",
        "model_id": "mc_dropout",
        "include_uncertainty": True
    }
)
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Uncertainty: {result['uncertainty']['total']}")
```

### cURL
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Is 100 > 50?",
    "model_id": "logistic_regression"
  }'
```

### JavaScript
```javascript
fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    prompt: 'Is 25 a prime number?',
    model_id: 'neural_network'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Performance Considerations

### Response Times
- Logistic Regression: ~5ms
- Neural Network: ~20ms
- MC Dropout: ~100ms (50 forward passes)
- Hierarchical Bayesian: ~150ms

### Batch Processing
For multiple predictions, use the batch endpoint for better performance:
- Single predictions: O(n) API calls
- Batch predictions: O(1) API call

### Caching
Results are cached for 5 minutes for identical prompts and models.

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
```
http://localhost:8000/openapi.json
```

Interactive documentation:
```
http://localhost:8000/docs
```