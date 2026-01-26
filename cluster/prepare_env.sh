#!/bin/bash
# cluster/prepare_env.sh
# Run this script once on the cluster to set up your environment

set -e  # Exit on error

ENV_NAME="vmax_gnn"

echo "=== Setting up Environment '$ENV_NAME' ==="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please load the anaconda/miniconda module or install it."
    exit 1
fi

# Create environment if it doesn't exist
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating environment '$ENV_NAME'..."
    conda create -n $ENV_NAME python=3.9 -y
fi

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "=== Installing Dependencies ==="

# Install PyTorch (adjust cuda version if needed, e.g., pytorch-cuda=12.1)
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install PyG
echo "Installing PyTorch Geometric..."
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install other requirements
echo "Installing other requirements..."
pip install -r ../requirements_gnn.txt

# Install the project in editable mode
echo "Installing project in editable mode..."
cd ..
pip install -e .

echo ""
echo "=== Setup Complete! ==="
echo "To activate: conda activate $ENV_NAME"
