# Binary Reasoning ML Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive machine learning pipeline for binary reasoning tasks, featuring multiple model architectures from simple logistic regression to hierarchical Bayesian models with uncertainty quantification. Now includes ambiguous debate questions and interactive visualization tools.

## 🚀 Highlights

- **98% Accuracy** with simple logistic regression on mathematical reasoning
- **4 Model Architectures**: From simple to Bayesian with uncertainty
- **Interactive Visualizer**: See reasoning and evidence accumulation in real-time
- **Ambiguous Debates**: "Is a hot dog a sandwich?" - perfect for uncertainty
- **Complete Pipeline**: Dataset generation → Training → API serving → Visualization
- **Production Ready**: Docker support, API documentation, monitoring

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Features](#-features)
- [Documentation](#-documentation)
- [Installation](#-installation)
- [Dataset Generation](#-dataset-generation)
- [Model Training](#-model-training)
- [API Usage](#-api-usage)
- [Development](#-development)
- [Contributing](#-contributing)

## 🎯 Quick Start

Get up and running in 5 minutes:

```bash
# 1. Clone and setup
git clone <repository-url>
cd reasoning-ml-pipeline
./quickstart.sh

# 2. Configure API key
export DEEPSEEK_API_KEY="your-api-key-here"

# 3. Choose your dataset:
# Option A: Mathematical reasoning (98% accuracy)
python incremental_generate.py --samples 1000

# Option B: Ambiguous debates (great for uncertainty)
python src/generate_debates_dataset.py --samples 100

# 4. Train MC Dropout model (best for uncertainty)
python src/train_bayesian_reasoning.py --model-type mc_dropout

# 5. Launch interactive visualizer
streamlit run app/reasoning_visualizer.py
```

Visit http://localhost:8501 to explore reasoning with uncertainty!

Visit http://localhost:8000/docs for interactive API documentation.

For detailed setup instructions, see the [Quick Start Guide](docs/QUICKSTART.md).

## ✨ Features

### Core Capabilities
- **Multi-Model Support**: Logistic Regression, Neural Networks, Bayesian Models
- **Uncertainty Quantification**: Epistemic and aleatoric uncertainty decomposition
- **Category-Specific Analysis**: Performance insights by question type
- **Production API**: FastAPI with OpenAPI documentation
- **Incremental Dataset Generation**: Resume-capable data creation
- **Comprehensive Evaluation**: Accuracy, F1, calibration metrics

### Model Performance
| Model | Accuracy | Uncertainty | Inference Time |
|-------|----------|-------------|----------------|
| Logistic Regression | 98.0% | ❌ | 5ms |
| Neural Network | 78.7% | ❌ | 20ms |
| MC Dropout Bayesian | 82.7% | ✅ | 100ms |
| Hierarchical Bayesian | 44.7% | ✅ | 150ms |

## 📚 Documentation

### Core Documentation
- **[Architecture Overview](docs/ARCHITECTURE.md)** - System design and components
- **[API Documentation](docs/API_DOCUMENTATION.md)** - Endpoints and examples
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment instructions
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

### Guides & Tutorials
- **[Quick Start Guide](docs/QUICKSTART.md)** - Get running in 5 minutes
- **[Visual App Guide](docs/VISUAL_APP_GUIDE.md)** - Interactive reasoning visualizer
- **[Process Documentation](docs/PROCESS_DOCUMENTATION.md)** - Complete development process
- **[Environment Configuration](docs/ENVIRONMENT.md)** - All environment variables

### Dataset Documentation
- **[Dataset Methodology](docs/dataset_methodology.md)** - Why viral debates for reasoning
- **[Dataset Technical Docs](docs/dataset_technical.md)** - Implementation details
- **[Dataset Usage Guide](docs/dataset_usage.md)** - How to use the datasets

### Model Documentation
- **[Model Analysis](models/MODEL_ANALYSIS.md)** - Detailed model comparisons
- **[Model Card](models/MODEL_CARD.md)** - Model specifications and limitations

## 🛠 Technology Stack

- **Language**: Python 3.9+
- **ML Framework**: PyTorch & scikit-learn
- **API**: FastAPI
- **Database**: PostgreSQL with SQLAlchemy
- **Experiment Tracking**: MLflow
- **Testing**: pytest

## 📋 Prerequisites

- Python 3.10 or higher
- 4GB RAM (8GB recommended for neural models)
- DeepSeek API key (for dataset generation)
- CUDA-capable GPU (optional, for faster training)

## 💾 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd reasoning-ml-pipeline
```

2. Run the quickstart script:
```bash
./quickstart.sh
```

This will:
- Set up a Python virtual environment using pyenv
- Install all required dependencies
- Configure the environment for dataset generation

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your DEEPSEEK_API_KEY
```

## 📊 Dataset Generation

### Three Dataset Options

#### 1. Mathematical Reasoning Dataset
Best for high accuracy models and clear reasoning:

```bash
# Generate 1000 math questions
python incremental_generate.py --samples 1000 --output data/api_dataset_incremental.json
```

#### 2. Ambiguous Debates Dataset (NEW!)
Perfect for MC Dropout and uncertainty quantification:

```bash
# Generate debate questions like "Is a hot dog a sandwich?"
python src/generate_debates_dataset.py --samples 100 --output data/debates_dataset.json

# Collect human votes for comparison
python src/collect_human_votes.py --dataset data/debates_dataset.json --interface web
```

#### 3. Viral Debates Dataset with Variations (LATEST!)
1000 questions testing reasoning consistency across phrasings:

```bash
# Already generated: data/debates_1000_final.json
# 100 base questions → 1000 variations (5 phrasings × 2 polarities)
# 80 viral debates + 20 quality control questions
```

See [Dataset Methodology](docs/dataset_methodology.md) for details.

### Dataset Categories

**Math Dataset**: arithmetic, comparisons, number properties, patterns

**Debates Dataset**:
- **Food Classification**: Is a hot dog a sandwich? Is cereal soup?
- **Edge Cases**: Is a tomato a fruit? Are birds dinosaurs?
- **Social/Cultural**: Is Die Hard a Christmas movie? Is golf a sport?
- **Thresholds**: Is 6'0" tall? Is $20 expensive for lunch?

**Viral Debates Dataset** (data/debates_1000_final.json):
- **Taxonomy Wars**: Hot dog/sandwich debates
- **Impossible Counting**: Wheels vs doors
- **Battle Scenarios**: 100 humans vs gorilla
- **Simple but Deep**: Is water wet?
- **Quality Control**: Clear consensus questions for sanity checking

### Interactive Visualization

Launch the reasoning visualizer to explore how models think:

```bash
streamlit run app/reasoning_visualizer.py
```

Features:
- Side-by-side reasoning and evidence charts
- Click points to highlight reasoning steps
- Real-time custom questions
- Uncertainty visualization

### Dataset Structure

Each sample contains:
- **prompt**: The binary yes/no question
- **correct_answer**: Ground truth answer
- **model_answer**: Model's prediction
- **reasoning_trace**: Step-by-step reasoning tokens with evidence scores
- **human_votes**: Real human consensus (debates dataset)
- **evidence_analysis**: Token-level evidence contribution

## 🔧 Development

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

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=src --cov-report=html
```

## 📏 Code Quality

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

### Available Models

We provide four different model architectures with varying trade-offs:

| Model | Accuracy | Key Feature | Best For |
|-------|----------|-------------|----------|
| **Logistic Regression** | 98.0% | Simple, fast | Production deployment |
| **Neural Network** | 78.7% | Complex patterns | Harder reasoning tasks |
| **MC Dropout Bayesian** | 82.7% | Uncertainty estimates | When confidence matters |
| **Hierarchical Bayesian** | 44.7% | Category insights | Research & analysis |

### Training Commands

```bash
# Train standard models (Logistic Regression or Neural Network)
python src/train_binary_reasoning.py --model-type logistic
python src/train_binary_reasoning.py --model-type neural

# Train Bayesian models with uncertainty
python src/train_bayesian_reasoning.py --model-type mc_dropout
python src/train_bayesian_reasoning.py --model-type hierarchical
```

### Model Analysis

For detailed model comparisons and selection guidance, see:
- [Model Analysis Report](models/MODEL_ANALYSIS.md) - Comprehensive analysis of all models
- [Model Comparison Visualizations](models/model_comparison_analysis.png)

### Quick Model Selection Guide

- **Need highest accuracy?** → Use Logistic Regression (98%)
- **Need confidence estimates?** → Use MC Dropout Bayesian (82.7%)
- **Need to understand categories?** → Use Hierarchical Bayesian
- **Have complex reasoning patterns?** → Try Neural Network

## API Endpoints

- `POST /predict` - Make predictions
- `GET /models` - List available models
- `GET /health` - Health check
- `POST /train` - Trigger training job

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick contribution guide:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- DeepSeek for providing the reasoning API
- PyTorch team for the excellent deep learning framework
- Pyro team for probabilistic programming tools

## 📞 Support

- 📧 Email: support@example.com
- 💬 Discussions: [GitHub Discussions](https://github.com/your-repo/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/your-repo/issues)

---

<p align="center">Built with ❤️ by the AI Reasoning Team</p>