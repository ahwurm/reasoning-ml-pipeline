# TODO: Ambiguous Debates Dataset Pipeline

## ✅ Completed Setup Steps
- [x] Created dataset generator (`src/generate_debates_dataset.py`)
- [x] Created debate templates (`src/debate_templates.py`)
- [x] Created human vote collection interface (`src/collect_human_votes.py`)
- [x] Created evidence extraction (`src/extract_reasoning_evidence.py`)
- [x] Built interactive visual app (`app/reasoning_visualizer.py`)
- [x] Created visual app guide (`docs/VISUAL_APP_GUIDE.md`)
- [x] Added real-time query support (`src/realtime_deepseek_query.py`)

## 🚀 Remaining Steps

### Step 1: Generate Ambiguous Debates Dataset (IN PROGRESS)
```bash
# Set API key first
export DEEPSEEK_API_KEY='your-key-here'

# Generate initial dataset (50 samples for testing)
python src/generate_debates_dataset.py --samples 50 --output data/debates_dataset.json

# Full dataset (1000 samples)
python src/generate_debates_dataset.py --samples 1000 --output data/debates_dataset_full.json
```

### Step 2: Collect Human Votes
```bash
# CLI interface for quick collection
python src/collect_human_votes.py --dataset data/debates_dataset.json --output data/debates_with_votes.json

# Or use Streamlit interface for better UX
streamlit run src/collect_human_votes.py -- --dataset data/debates_dataset.json
```

### Step 3: Train MC Dropout Model
```bash
# Train the model optimized for uncertainty on ambiguous questions
python src/train_mc_dropout_debates.py \
  --dataset data/debates_with_votes.json \
  --epochs 100 \
  --dropout 0.3 \
  --output models/mc_dropout_debates.pth
```

### Step 4: Extract Evidence Patterns
```bash
# Analyze reasoning traces for evidence accumulation
python src/extract_reasoning_evidence.py \
  --dataset data/debates_with_votes.json \
  --output data/debates_with_evidence.json
```

### Step 5: Launch Interactive Visualizer
```bash
# Start the Streamlit app
streamlit run app/reasoning_visualizer.py

# Access at http://localhost:8501
```

### Step 6: Test Real-time Queries
```bash
# Test the real-time query interface
python src/realtime_deepseek_query.py --test

# Or use the API endpoint in the visualizer
```

### Step 7: Analyze Model Performance
```bash
# Generate performance analysis
python src/analyze_debates_performance.py \
  --model models/mc_dropout_debates.pth \
  --dataset data/debates_with_votes.json \
  --output analysis/debates_analysis.md
```

### Step 8: Create Demo and Documentation
- [ ] Record demo video showing:
  - Dataset generation process
  - Human vote collection
  - Model training with uncertainty
  - Interactive visualization
  - Real-time query capabilities
- [ ] Create presentation slides
- [ ] Write blog post about ambiguous reasoning

## 📊 Success Metrics
- Dataset diversity: 4 categories, 250+ samples each
- Human agreement rate: Track split decisions
- Model calibration: Uncertainty correlates with human disagreement
- Visualization clarity: Evidence accumulation visible
- Query latency: <2s for real-time queries

## 🔧 Development Notes
- Use batch processing for API calls to avoid rate limits
- Cache reasoning traces to reduce API costs
- Implement retry logic for failed API calls
- Monitor token usage and costs
- Add progress bars for long operations

## 🐛 Known Issues
- DeepSeek API may rate limit on large batches
- Some debate questions may not generate good reasoning
- Evidence extraction patterns may need tuning
- Streamlit app may need optimization for large datasets

## 📅 Timeline
- Step 1: 2-3 hours (API calls are slow)
- Step 2: 1-2 hours (depends on human availability)
- Step 3: 30 minutes (model training)
- Step 4: 30 minutes (analysis)
- Step 5-8: 1-2 hours (testing and documentation)

Total estimated time: 5-8 hours

## 🎯 Next Actions
1. Set DEEPSEEK_API_KEY environment variable
2. Run Step 1 with small batch (50 samples) to test
3. Monitor API usage and adjust batch size if needed
4. Proceed through steps sequentially