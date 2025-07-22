# System Architecture Documentation

## Overview

The Binary Reasoning ML Pipeline is a modular system designed to generate, process, and model binary mathematical reasoning tasks. The architecture supports multiple model types with varying complexity and capabilities.

## High-Level Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Generation   │────▶│  Model Training  │────▶│   Model Serving │
│  (DeepSeek API)     │     │  (PyTorch/sklearn)│     │   (FastAPI)     │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
         │                           │                         │
         ▼                           ▼                         ▼
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Dataset Storage    │     │  Model Registry  │     │   Predictions   │
│  (JSON/PostgreSQL)  │     │  (MLflow/Local)  │     │   (REST API)    │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
```

## Component Architecture

### 1. Data Generation Layer

```
incremental_generate.py
├── API Client (DeepSeek)
├── Prompt Templates
├── Response Parser
├── Quality Validator
└── Incremental Saver
```

**Key Features:**
- Batch processing with resume capability
- Automatic quality validation (99.7% accuracy achieved)
- Category balancing
- Progress tracking

### 2. Data Processing Layer

```
src/data/
├── Dataset Loaders
│   ├── BinaryReasoningDataset (PyTorch)
│   └── SKLearnDataProcessor
├── Feature Extractors
│   ├── TF-IDF Vectorizer
│   └── Token Embeddings
└── Data Validators
```

### 3. Model Architecture

#### 3.1 Standard Models

**Logistic Regression Pipeline:**
```
Input Text → TF-IDF Vectorization → Logistic Regression → Binary Output
```

**Neural Network Architecture:**
```
Input Tokens → Embedding(vocab_size, 128)
            → BiLSTM(128, 256, 2 layers)
            → Attention Mechanism
            → Linear(512, 256)
            → ReLU + Dropout(0.3)
            → Linear(256, 2)
            → Softmax
```

#### 3.2 Bayesian Models

**MC Dropout Architecture:**
```
Standard NN + Persistent Dropout in Inference
→ Multiple Forward Passes (50 samples)
→ Uncertainty Estimation
```

**Hierarchical Bayesian:**
```
Global Hyperpriors
├── Category-Level Parameters
│   ├── Math Verification
│   ├── Comparison
│   ├── Number Property
│   └── Pattern Recognition
└── Instance-Level Predictions
```

### 4. Training Pipeline

```
src/train_*.py
├── Data Loading
├── Model Initialization
├── Training Loop
│   ├── Forward Pass
│   ├── Loss Calculation
│   ├── Backpropagation
│   └── Validation
├── Early Stopping
├── Model Checkpointing
└── Metric Logging
```

### 5. Inference Architecture

```
FastAPI Application
├── Model Registry
│   ├── Load Models
│   └── Version Management
├── Prediction Endpoints
│   ├── Single Prediction
│   ├── Batch Prediction
│   └── Uncertainty Estimation
└── Monitoring
    ├── Latency Tracking
    └── Error Logging
```

## Data Flow

### Training Flow
1. **Dataset Loading**: JSON → Python Dict → Dataset Object
2. **Preprocessing**: Tokenization → Vectorization/Embedding
3. **Model Training**: Batch Processing → Gradient Updates
4. **Validation**: Hold-out Set Evaluation
5. **Model Saving**: Checkpoint → Model Registry

### Inference Flow
1. **Request Reception**: REST API → Request Validation
2. **Preprocessing**: Same as training pipeline
3. **Model Inference**: Forward Pass (+ MC sampling for Bayesian)
4. **Post-processing**: Probability → Binary Decision
5. **Response**: JSON with prediction + confidence

## Model Registry Structure

```
models/
├── Standard Models
│   ├── logistic_regression.pkl
│   ├── neural_network.pth
│   └── vectorizers.pkl
├── Bayesian Models
│   ├── mc_dropout.pth
│   └── hierarchical.pth
├── Metadata
│   ├── model_results.json
│   └── training_configs.yaml
└── Visualizations
    ├── comparison_plots.png
    └── uncertainty_analysis.png
```

## Technology Stack

### Core Technologies
- **Python 3.10+**: Primary language
- **PyTorch 2.7.1**: Neural network framework
- **scikit-learn 1.7.0**: Classical ML algorithms
- **Pyro 1.9.1**: Probabilistic programming

### Supporting Libraries
- **FastAPI**: REST API framework
- **Pydantic**: Data validation
- **NumPy/Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

### Infrastructure
- **MLflow**: Experiment tracking (optional)
- **PostgreSQL**: Database backend (optional)
- **Docker**: Containerization (planned)

## Scalability Considerations

### Horizontal Scaling
- Stateless API design allows multiple instances
- Model serving can be distributed
- Training can use distributed PyTorch

### Vertical Scaling
- GPU acceleration for neural models
- Batch processing for efficiency
- Caching for repeated predictions

### Performance Optimizations
- Model quantization for deployment
- ONNX export for inference optimization
- Request batching in API

## Security Architecture

### API Security
- API key authentication
- Rate limiting
- Input validation
- Sanitization of user inputs

### Data Security
- No PII in datasets
- Secure API key storage
- Encrypted model storage (planned)

## Monitoring and Observability

### Metrics Tracked
- Model accuracy over time
- Prediction latency
- API request volume
- Error rates by category

### Logging Strategy
- Structured logging (JSON)
- Centralized log aggregation
- Error tracking and alerting

## Extension Points

### Adding New Models
1. Implement model class in `src/models/`
2. Add training script in `src/`
3. Register in model factory
4. Update API endpoints

### Adding New Data Sources
1. Implement data loader
2. Add preprocessing pipeline
3. Update dataset generation
4. Validate data quality

### Adding New Features
1. Extend feature extractors
2. Update model architectures
3. Retrain models
4. A/B test performance

## Architecture Decisions

### Why Multiple Model Types?
- **Simple Models**: High accuracy, fast inference
- **Complex Models**: Better generalization potential
- **Bayesian Models**: Uncertainty quantification
- **Hierarchical Models**: Category-specific insights

### Why Incremental Generation?
- Handles API timeouts gracefully
- Allows quality monitoring
- Enables partial dataset usage
- Supports resume on failure

### Why PyTorch + scikit-learn?
- Best of both worlds
- PyTorch for deep learning flexibility
- scikit-learn for robust classical ML
- Shared preprocessing pipeline