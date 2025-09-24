#!/bin/bash
# Script to run Jupyter notebook in qlib environment

echo "🔧 Setting up qlib environment for Jupyter..."

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate qlib

echo "✅ Environment activated: $(echo $CONDA_DEFAULT_ENV)"

# Check if Jupyter is installed
if ! command -v jupyter &> /dev/null; then
    echo "📦 Installing Jupyter in qlib environment..."
    pip install jupyter
fi

echo "🚀 Starting Jupyter notebook server..."
echo "📍 Access at: http://localhost:8888"
echo "🛑 Press Ctrl+C to stop the server"

# Start Jupyter with proper configuration
jupyter notebook \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --notebook-dir="/root/mycode/qlibjianbo"