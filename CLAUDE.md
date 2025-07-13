# AI Reasoning Visualization - ML Pipeline

## Project Overview
Python-based machine learning pipeline for training decision drift models using reasoning tokens.

## Technology Stack
- Python 3.9+ with virtual environment
- PyTorch for neural networks
- scikit-learn for classical ML
- FastAPI for REST API
- PostgreSQL for data storage
- MLflow for experiment tracking

## Key Commands
- `python -m venv venv && source venv/bin/activate` - Setup environment
- `pip install -r requirements.txt` - Install dependencies
- `python src/train.py` - Train models
- `python src/api.py` - Start API server
- `pytest tests/` - Run test suite

## Code Style Guidelines
- Use type hints throughout
- Follow PEP 8 style guide
- Implement proper error handling
- Use logging for debugging
- Separate concerns between training and inference
- Use configuration files for hyperparameters

## Project Structure
- `src/models/` - Model definitions
- `src/training/` - Training pipeline
- `src/inference/` - Inference API
- `src/data/` - Data processing utilities
- `src/config/` - Configuration management
- `tests/` - Test files

## Critical Requirements
- Model versioning with MLflow
- Proper data validation
- Reproducible training runs
- API must be production-ready
- Comprehensive model evaluation
- Memory-efficient processing

## Testing Strategy
- Unit tests for all functions
- Integration tests for API endpoints
- Model performance tests
- Data pipeline validation tests