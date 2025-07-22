# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0] - 2025-07-15

### Added
- **Ambiguous Debates Dataset** - New dataset type for questions like "Is a hot dog a sandwich?"
  - 4 categories: food classification, edge cases, social/cultural, thresholds
  - Real DeepSeek reasoning traces (no simulation)
  - Human vote collection interface
- **Interactive Reasoning Visualizer** - Streamlit app for exploring model reasoning
  - Side-by-side reasoning and evidence charts
  - Interactive token highlighting
  - Real-time custom question support
  - MC Dropout uncertainty visualization
- **Evidence Extraction System** - Analyzes reasoning traces for evidence patterns
  - Token-level evidence scoring
  - Uncertainty factor identification
  - Evidence flow analysis
- **Real-time Query Support** - Query DeepSeek in real-time
  - Caching for efficiency
  - Streaming response support
  - Automatic category inference

### Enhanced
- README with new dataset options and visualization guide
- Documentation structure with tutorials and guides
- Model comparison with focus on uncertainty

### Fixed
- Device placement issues in hierarchical Bayesian model
- Import errors in training scripts

## [1.0.0] - 2025-07-14

### Added
- Initial release with mathematical reasoning dataset
- 4 model architectures:
  - Logistic Regression (98% accuracy)
  - Neural Network with attention
  - MC Dropout Bayesian
  - Hierarchical Bayesian
- Incremental dataset generation with resume capability
- Comprehensive documentation:
  - Architecture overview
  - API documentation
  - Deployment guide
  - Troubleshooting guide
- Model analysis and comparison tools
- FastAPI-based inference server

### Features
- Binary yes/no mathematical reasoning
- Uncertainty quantification for Bayesian models
- Category-specific performance analysis
- Production-ready API with OpenAPI spec
- Docker support for deployment