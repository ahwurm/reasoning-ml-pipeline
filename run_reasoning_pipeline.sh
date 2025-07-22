#!/bin/bash
# Run the complete reasoning pipeline

echo "=== Viral Debates Reasoning Pipeline ==="
echo

# Check if DeepSeek API key is set
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo "Error: DEEPSEEK_API_KEY environment variable not set"
    echo "Please run: export DEEPSEEK_API_KEY='your-key-here'"
    exit 1
fi

# Step 1: Generate reasoning traces
echo "Step 1: Generating reasoning traces..."
echo "This will take 2-3 hours for all 1000 questions"
echo

# Check if reasoning already exists
if [ -f "data/debates_reasoning.json" ]; then
    echo "Reasoning file already exists. Checking progress..."
    python -c "import json; d=json.load(open('data/debates_reasoning.json')); print(f'Progress: {len(d[\"questions\"])}/1000 questions')"
    echo
    read -p "Continue from existing progress? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. To start fresh, delete data/debates_reasoning.json"
        exit 1
    fi
fi

# Generate reasoning (with limit for testing)
if [ "$1" == "--test" ]; then
    echo "TEST MODE: Processing only 50 questions"
    python generate_reasoning_traces.py --limit 50
else
    python generate_reasoning_traces.py
fi

# Check if reasoning generation succeeded
if [ $? -ne 0 ]; then
    echo "Error: Reasoning generation failed"
    exit 1
fi

echo
echo "Step 2: Training MC Dropout model..."
echo

# Create models directory if it doesn't exist
mkdir -p models

# Train the model
python train_mc_dropout_debates.py \
    --dataset data/debates_reasoning.json \
    --epochs 50 \
    --dropout 0.25 \
    --output models/mc_dropout_debates.pth

# Check if training succeeded
if [ $? -ne 0 ]; then
    echo "Error: Model training failed"
    exit 1
fi

echo
echo "=== Pipeline Complete! ==="
echo
echo "Results:"
echo "- Reasoning traces: data/debates_reasoning.json"
echo "- Trained model: models/mc_dropout_debates.pth"
echo "- Training plots: models/mc_dropout_training.png"
echo
echo "Next steps:"
echo "1. Analyze model performance on QC vs viral questions"
echo "2. Visualize uncertainty distributions"
echo "3. Test consistency across variations"