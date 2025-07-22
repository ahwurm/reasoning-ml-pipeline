# Binary Reasoning Math Models - Complete Process Documentation

## Overview

This document captures the end-to-end process for creating a machine learning pipeline for binary mathematical reasoning tasks, from dataset generation to model deployment recommendations.

## Phase 1: Dataset Generation

### 1.1 Initial Setup
```bash
# Environment setup
./quickstart.sh
export DEEPSEEK_API_KEY="your-api-key-here"
```

### 1.2 Incremental Dataset Generation
We used incremental generation to create 1000 samples:
```bash
python incremental_generate.py --samples 1000 --output data/api_dataset_incremental.json
```

**Key Parameters:**
- Batch size: 50 samples (to avoid timeouts)
- Categories: math_verification, comparison, number_property, pattern_recognition
- Answer distribution target: 50/50 Yes/No

### 1.3 Dataset Quality Metrics
- **Total samples**: 1000
- **Model accuracy**: 99.7% (only 3 errors)
- **Balance**: 46.4% Yes, 53.6% No
- **Category distribution**: Well-balanced across 4 categories

### 1.4 Error Analysis
Identified 3 errors where model reasoned correctly but output wrong answer:
- All were "should be Yes but said No" errors
- Showed disconnect between reasoning and final answer

## Phase 2: Folder Cleanup

### 2.1 Files Removed
- Test scripts (test_*.py)
- Temporary datasets
- Old generation scripts

### 2.2 Files Preserved
- `incremental_generate.py` - Main generation script
- `data/api_dataset_incremental.json` - Final dataset
- Configuration files
- README and documentation

## Phase 3: Model Development

### 3.1 Standard Models

#### Logistic Regression
```bash
python src/train_binary_reasoning.py --model-type logistic
```
- **Architecture**: TF-IDF features + Logistic Regression
- **Results**: 98% accuracy, 0.980 F1 score
- **Training time**: <1 minute

#### Neural Network
```bash
python src/train_binary_reasoning.py --model-type neural
```
- **Architecture**: Embedding + BiLSTM + Attention
- **Results**: 78.7% accuracy, 0.787 F1 score
- **Key hyperparameters**:
  - Embedding dim: 128
  - Hidden dim: 256
  - Layers: 2
  - Dropout: 0.3

### 3.2 Bayesian Models

#### MC Dropout
```bash
python src/train_bayesian_reasoning.py --model-type mc_dropout
```
- **Architecture**: Neural network with Monte Carlo Dropout
- **Results**: 82.7% accuracy, 0.827 F1 score
- **Uncertainty**: ECE = 0.154, Mean uncertainty = 0.015
- **MC samples**: 50 for inference

#### Hierarchical Bayesian
```bash
python src/train_bayesian_reasoning.py --model-type hierarchical
```
- **Architecture**: Hierarchical priors with category-specific parameters
- **Results**: 44.7% accuracy, 0.447 F1 score
- **Uncertainty decomposition**:
  - Epistemic: 8.2%
  - Aleatoric: 91.8%
- **Category performance**: Varied from 45% to 59.1%

## Phase 4: Model Analysis

### 4.1 Analysis Script Creation
```python
# src/analyze_models.py
- Load all model results
- Create comparison visualizations
- Generate performance metrics
- Analyze uncertainty decomposition
```

### 4.2 Visualizations Generated
1. **Model Comparison (4-panel)**:
   - Accuracy comparison bar chart
   - Uncertainty vs accuracy scatter
   - F1 score comparison
   - Category-specific heatmap

2. **Model Timeline**:
   - Shows progression from simple to complex
   - Illustrates accuracy vs complexity trade-off

### 4.3 Key Insights
1. Simple models outperform complex ones on this dataset
2. High aleatoric uncertainty indicates data ambiguity
3. Category-specific performance varies significantly
4. Clear trade-offs between accuracy and uncertainty quantification

## Phase 5: Documentation

### 5.1 Created Documents
1. **MODEL_ANALYSIS.md**: Comprehensive analysis with:
   - Executive summary
   - Model comparison table
   - Selection guidelines
   - Technical learnings
   - Next steps

2. **Updated README.md**: Added:
   - Model training section
   - Quick selection guide
   - Links to analysis

3. **Process Documentation**: This document

## Technical Stack Used

### Libraries
- **Data Processing**: pandas, numpy
- **ML Frameworks**: scikit-learn, PyTorch
- **Bayesian ML**: Pyro
- **Visualization**: matplotlib, seaborn
- **API Generation**: DeepSeek API

### Key Dependencies
```
torch==2.7.1
scikit-learn==1.7.0
pyro-ppl==1.9.1
numpy==2.2.6
pandas
matplotlib
seaborn
tqdm
```

## Lessons Learned

### 1. Dataset Generation
- Incremental generation with resume capability is essential
- Batch processing prevents timeouts
- Model accuracy during generation is a good quality indicator

### 2. Model Development
- Always start with simple baselines
- High-quality features (reasoning traces) can make simple models excel
- Uncertainty quantification comes at accuracy cost
- Hierarchical models provide insights but may underperform

### 3. Analysis
- Visual comparisons are crucial for understanding trade-offs
- Uncertainty decomposition reveals data vs model challenges
- Category-specific analysis uncovers hidden patterns

## Reproducibility Checklist

- [ ] Environment setup with pyenv
- [ ] API key configuration
- [ ] Dataset generation (1000 samples)
- [ ] Train all 4 model types
- [ ] Run analysis script
- [ ] Generate visualizations
- [ ] Review model outputs

## Next Dataset Considerations

When applying this process to a new dataset:

1. **Adapt generation script** for new task format
2. **Modify model architectures** if needed for new input types
3. **Update evaluation metrics** for task-specific needs
4. **Consider dataset difficulty** - our dataset may have been too easy
5. **Plan for longer training** if dataset is more complex
6. **Add task-specific baselines** for comparison

## Commands Summary

```bash
# Setup
./quickstart.sh
export DEEPSEEK_API_KEY="..."

# Generate dataset
python incremental_generate.py --samples 1000 --output data/dataset.json

# Train models
python src/train_binary_reasoning.py --model-type logistic
python src/train_binary_reasoning.py --model-type neural
python src/train_bayesian_reasoning.py --model-type mc_dropout
python src/train_bayesian_reasoning.py --model-type hierarchical

# Analyze
python src/analyze_models.py

# View results
ls models/*.png
cat models/MODEL_ANALYSIS.md
```