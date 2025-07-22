#!/bin/bash
# Quick start script for local R1 model setup

echo "R1 Local Model Setup - Quick Start"
echo "=================================="
echo

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$python_version < 3.8" | bc) -eq 1 ]]; then
    echo "Error: Python 3.8+ required (found $python_version)"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate

# Check for CUDA
echo
echo "Checking for GPU support..."
python3 -c "import sys; sys.exit(0)" 2>/dev/null
if python3 -c "import torch 2>/dev/null && torch.cuda.is_available()" 2>/dev/null; then
    echo "✓ CUDA GPU detected"
    gpu_available=true
else
    echo "ℹ No CUDA GPU detected - will use CPU (slower)"
    gpu_available=false
fi

# Install requirements
echo
echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements-local.txt

# Check disk space
echo
echo "Checking disk space..."
available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$available_space" -lt 20 ]; then
    echo "Warning: Low disk space (${available_space}GB available)"
    echo "Model download requires ~15GB. Continue? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
else
    echo "✓ Sufficient disk space (${available_space}GB available)"
fi

# Run setup test
echo
echo "Running setup test..."
python test_r1_setup.py

echo
echo "Setup complete!"
echo
echo "Next steps:"
echo "1. Download the model (one-time, ~15GB):"
echo "   python src/download_model.py"
echo
echo "2. Test local extraction:"
if [ "$gpu_available" = true ]; then
    echo "   python src/r1_local_extractor.py"
    echo "   # Or with 8-bit quantization for lower memory:"
    echo "   python src/r1_local_extractor.py --8bit"
else
    echo "   python src/r1_local_extractor.py --cpu"
fi
echo
echo "3. Process your prompts:"
echo "   python src/collect_r1_local.py data/sample_prompts.txt"