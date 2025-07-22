#!/bin/bash
# Quick start script for ambiguous debates dataset and visualization

echo "🎯 Binary Reasoning ML Pipeline - Debates Dataset Quick Start"
echo "============================================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || { echo "Failed to activate venv"; exit 1; }

# Install requirements
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet streamlit plotly pandas numpy openai python-dotenv tqdm

# Check for API key
if [ -z "$DEEPSEEK_API_KEY" ]; then
    echo ""
    echo "⚠️  DEEPSEEK_API_KEY not set!"
    echo "Please set it with: export DEEPSEEK_API_KEY='your-key-here'"
    echo ""
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data models logs cache app

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Set your API key: export DEEPSEEK_API_KEY='your-key'"
echo "2. Generate debate dataset: python src/generate_debates_dataset.py --samples 50"
echo "3. Launch visualizer: streamlit run app/reasoning_visualizer.py"
echo ""
echo "For a demo without generating data:"
echo "python src/debate_templates.py  # See example questions"
echo ""