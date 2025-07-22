#!/bin/bash
# Quick start script for R1 extraction tools

echo "R1 Extraction Tools - Quick Start"
echo "================================="
echo

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements-r1.txt

# Check if .env exists
if [ ! -f ".env" ]; then
    echo
    echo "Creating .env file..."
    cp .env.example .env
    echo "Please edit .env and add your DeepSeek API key"
fi

# Run setup test
echo
echo "Running setup test..."
python test_r1_setup.py

echo
echo "Setup complete!"
echo
echo "To get started:"
echo "1. Edit .env and add your DeepSeek API key"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the example: python src/r1_extractor.py"
echo "4. Or collect data: python src/collect_r1_data.py data/sample_prompts.txt"