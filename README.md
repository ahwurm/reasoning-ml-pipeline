# AI Reasoning Visualization - ML Pipeline

A Python-based machine learning pipeline for training decision drift models using reasoning tokens.

## Features

- Decision drift detection models
- Reasoning token analysis
- Real-time inference API
- MLflow experiment tracking
- Ensemble model support
- GPU-accelerated training

## Technology Stack

- **Language**: Python 3.9+
- **ML Framework**: PyTorch & scikit-learn
- **API**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy
- **Experiment Tracking**: MLflow
- **Testing**: pytest

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 14+
- CUDA-capable GPU (optional, for acceleration)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd reasoning-ml-pipeline
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

5. Configure database and MLflow in `.env`:
```
DATABASE_URL=postgresql://user:password@localhost/reasoning_ml
MLFLOW_TRACKING_URI=http://localhost:5000
```

## Development

### Training Models

Run the training pipeline:
```bash
python src/train.py --config configs/default.yaml
```

### Starting the API Server

Start the FastAPI server:
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### MLflow UI

Start MLflow tracking server:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Access the UI at `http://localhost:5000`

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## Code Quality

Format code:
```bash
black src tests
isort src tests
```

Run linters:
```bash
flake8 src tests
mypy src
pylint src
```

## Project Structure

```
src/
├── models/      # Model architectures
├── training/    # Training pipeline
├── inference/   # Inference API
├── data/        # Data processing utilities
└── config/      # Configuration management

tests/          # Test files
configs/        # Training configurations
notebooks/      # Jupyter notebooks (excluded from git)
```

## Model Training

1. Prepare your data in the required format
2. Configure hyperparameters in `configs/`
3. Run training with MLflow tracking
4. Evaluate model performance
5. Deploy best model via API

## API Endpoints

- `POST /predict` - Make predictions
- `GET /models` - List available models
- `GET /health` - Health check
- `POST /train` - Trigger training job

## Contributing

1. Create a feature branch from `develop`
2. Follow PEP 8 style guide
3. Add type hints to all functions
4. Write tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

[Your License Here]