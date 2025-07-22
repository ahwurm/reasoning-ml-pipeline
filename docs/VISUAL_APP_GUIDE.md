# Visual App Guide: Interactive Reasoning Visualizer

## Overview

The Interactive Reasoning Visualizer is a Streamlit-based web application that showcases MC Dropout model predictions alongside DeepSeek reasoning traces with interactive evidence accumulation charts. This guide covers installation, usage, and customization.

## Features

- **Side-by-side visualization** of reasoning and evidence accumulation
- **Interactive token highlighting** - click chart points to highlight reasoning steps
- **Real-time DeepSeek queries** for custom questions
- **Uncertainty visualization** with MC Dropout predictions
- **Human consensus display** from collected votes

## Installation

### Prerequisites
```bash
# Python 3.10+
python --version

# Install required packages
pip install streamlit plotly pandas numpy
pip install openai  # For DeepSeek API
```

### Setup
```bash
# Clone repository
git clone <repository-url>
cd reasoning-ml-pipeline

# Install all dependencies
pip install -r requirements.txt

# Set API key
export DEEPSEEK_API_KEY="your-api-key"
```

## Running the App

### Basic Launch
```bash
# From project root
streamlit run app/reasoning_visualizer.py
```

The app will open at `http://localhost:8501`

### With Custom Port
```bash
streamlit run app/reasoning_visualizer.py --server.port 8080
```

### Production Mode
```bash
streamlit run app/reasoning_visualizer.py \
  --server.headless true \
  --server.address 0.0.0.0 \
  --server.port 80
```

## User Interface Guide

### Layout Overview
```
┌─────────────────────────────────────────────────────┐
│                  Question Display                    │
├─────────────────────┬───────────────────────────────┤
│                     │                               │
│  Reasoning Panel    │   Evidence Chart             │
│  (Left)            │   (Right)                     │
│                     │                               │
├─────────────────────┼───────────────────────────────┤
│  MC Dropout         │   Human Consensus            │
│  Prediction         │   Votes                      │
└─────────────────────┴───────────────────────────────┘
```

### 1. Sidebar Controls

#### Mode Selection
- **Dataset Questions**: Browse pre-generated debate questions
- **Real-time Query**: Enter custom questions

#### Category Filter (Dataset Mode)
- All categories
- Food Classification
- Edge Classifications
- Social/Cultural
- Threshold Questions

#### Question Selector
- Dropdown showing question ID and preview
- Updates main display on selection

### 2. Main Display

#### Question Header
- Full question text
- Category label

#### Reasoning Panel (Left)
- **Numbered reasoning steps** with color coding:
  - 🟢 Green: Supporting evidence
  - 🔴 Red: Opposing evidence
  - 🟡 Yellow: Uncertainty
  - ⚪ Gray: Neutral
- **Interactive tokens**: Click to highlight on evidence chart
- **Final answer** displayed at bottom

#### Evidence Chart (Right)
- **X-axis**: Reasoning steps
- **Y-axis**: Evidence score (-1 to +1)
- **Interactive points**: Click to highlight reasoning token
- **Colored regions**: Show evidence type for each step
- **Zero line**: Separates positive/negative evidence

### 3. Bottom Panels

#### MC Dropout Prediction (Left)
- **Prediction**: Yes/No answer
- **Confidence**: Percentage with uncertainty bounds
- **Uncertainty gauge**: Visual confidence indicator

#### Human Consensus (Right)
- **Pie chart**: Yes/No vote distribution
- **Vote count**: Total human votes
- **Consensus strength**: Agreement level indicator

## Interactive Features

### 1. Token-Chart Interaction

Click on any point in the evidence chart to:
- Highlight corresponding reasoning token
- Show token details in info box
- Update token opacity in reasoning panel

### 2. Hover Information

Hover over chart points to see:
- Step number
- Evidence score
- Token preview (first 50 characters)

### 3. Real-time Mode

1. Select "Real-time Query" in sidebar
2. Enter your question in text area
3. Click "🚀 Analyze" button
4. Wait for DeepSeek processing
5. View results with same interactive features

## Example Walkthrough

### Analyzing "Is a hot dog a sandwich?"

1. **Select Question**
   - Choose from dropdown or enter in real-time mode

2. **Observe Reasoning**
   ```
   1. A sandwich typically consists of ingredients between bread
   2. A hot dog has meat between a split bun
   3. However, the bun is connected on one side
   4. Submarine sandwiches also use connected bread
   5. Cultural usage differs from technical definition
   ```

3. **Track Evidence**
   - Watch evidence accumulate from 0
   - See uncertainty at step 3 ("However...")
   - Final score indicates "No" answer

4. **Check Uncertainty**
   - MC Dropout: 65% confident "No" ± 12%
   - High uncertainty due to ambiguous nature

5. **Compare with Humans**
   - Human votes: 35% Yes, 65% No
   - Model aligns with human consensus

## Customization

### Adding Custom Categories

Edit `src/debate_templates.py`:
```python
DEBATE_CATEGORIES['my_category'] = {
    'description': 'My custom debates',
    'templates': [...],
    'question_formats': [...]
}
```

### Modifying Evidence Extraction

Edit `src/extract_reasoning_evidence.py`:
```python
# Add custom evidence patterns
CUSTOM_PATTERNS = [
    r'\b(my_keyword)\b',
    r'\b(another_pattern)\b'
]
```

### Styling the App

Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#2196F3"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

## API Integration

### Custom Question Analysis
```python
from src.realtime_deepseek_query import RealtimeDeepSeekQuery

querier = RealtimeDeepSeekQuery()
result = querier.analyze_custom_question("Is water wet?")
```

### Batch Processing
```python
questions = [
    {"question": "Is a taco a sandwich?", "category": "food"},
    {"question": "Is chess a sport?", "category": "social"}
]
results = querier.batch_query(questions)
```

## Troubleshooting

### Common Issues

1. **"Dataset not found" error**
   ```bash
   # Generate dataset first
   python src/generate_debates_dataset.py --samples 100
   ```

2. **"MC Dropout model not loaded" warning**
   ```bash
   # Train MC Dropout model
   python src/train_bayesian_reasoning.py --model-type mc_dropout
   ```

3. **API timeout errors**
   - Check internet connection
   - Verify API key is valid
   - Reduce request frequency

4. **Chart not updating**
   - Refresh browser (F5)
   - Clear Streamlit cache: Settings → Clear cache

### Performance Optimization

1. **Enable caching**
   ```python
   @st.cache_data
   def load_dataset():
       return json.load(open('data/debates_dataset.json'))
   ```

2. **Limit dataset size**
   ```bash
   # Use smaller dataset for testing
   python src/generate_debates_dataset.py --samples 50
   ```

3. **Use GPU for MC Dropout**
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

## Advanced Features

### 1. Export Results
```python
# Add to app
if st.button("Export Analysis"):
    df = pd.DataFrame(analysis_results)
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "analysis.csv")
```

### 2. Compare Multiple Models
```python
# Show predictions from different models
models = ['logistic', 'neural', 'mc_dropout']
for model in models:
    pred = get_prediction(question, model)
    st.metric(f"{model}", pred)
```

### 3. Reasoning Replay
```python
# Animate evidence accumulation
for i in range(len(evidence_scores)):
    chart.update(evidence_scores[:i+1])
    time.sleep(0.5)
```

## Deployment

### Local Network
```bash
# Share on local network
streamlit run app/reasoning_visualizer.py \
  --server.address 0.0.0.0
```

### Streamlit Cloud
1. Push to GitHub
2. Connect repo to Streamlit Cloud
3. Set secrets:
   ```toml
   DEEPSEEK_API_KEY = "your-key"
   ```

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "app/reasoning_visualizer.py"]
```

## Next Steps

1. **Collect more human votes**
   ```bash
   streamlit run src/collect_human_votes.py
   ```

2. **Train better models**
   - Fine-tune MC Dropout on debate dataset
   - Experiment with ensemble methods

3. **Enhance visualization**
   - Add confidence bands to evidence chart
   - Show token-level attention weights
   - Create reasoning flow diagram

4. **Add features**
   - Save/load analysis sessions
   - Compare reasoning across similar questions
   - Generate explanation reports

## Support

- **Documentation**: See `/docs` folder
- **Issues**: GitHub Issues page
- **Updates**: Check CHANGELOG.md

Happy reasoning visualization! 🧠✨