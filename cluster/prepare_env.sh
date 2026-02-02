#!/bin/bash
# cluster/prepare_env.sh
# Robust environment setup for ENSIA HPC Kubeflow notebooks / SLURM nodes

set -euo pipefail

ENV_NAME="vmax_gnn"
PY_VER="3.10"

echo "=== Setting up conda environment: ${ENV_NAME} (python=${PY_VER}) ==="

# Load conda in non-interactive shells
if ! command -v conda &> /dev/null; then
  echo "Error: conda not found. Please load anaconda/miniconda or ensure /opt/conda is available."
  exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env if missing
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Environment '${ENV_NAME}' already exists."
else
  echo "Creating environment '${ENV_NAME}'..."
  conda create -y -n "${ENV_NAME}" python="${PY_VER}"
fi

conda activate "${ENV_NAME}"

echo "=== Installing GPU PyTorch (pip, cu121) ==="
# Remove any existing torch installed from conda/pip to avoid CPU-only builds
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
conda remove -y pytorch torchvision torchaudio libtorch pytorch-mutex 2>/dev/null || true

# Install CUDA-enabled PyTorch wheels
pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "=== Verifying CUDA availability ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available to PyTorch. Ensure your notebook/job has a GPU allocated.")
PY

echo "=== Installing PyTorch Geometric (PyG) matching torch 2.5.x + cu121 ==="
pip install --no-cache-dir torch_geometric
pip install --no-cache-dir pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

echo "=== Installing remaining requirements (excluding torch & PyG compiled deps) ==="
# Avoid re-installing torch/pyg deps from requirements_gnn.txt
REQ_IN="../requirements_gnn.txt"
REQ_OUT="/tmp/requirements_no_torch.txt"

if [[ -f "${REQ_IN}" ]]; then
  grep -v -E '^(torch($|[<>=])|torch-geometric|torch_scatter|torch-scatter|torch_sparse|torch-sparse|torch_cluster|torch-cluster|torch_spline_conv|torch-spline-conv|pyg_lib)' \
    "${REQ_IN}" > "${REQ_OUT}" || true
  pip install --no-cache-dir -r "${REQ_OUT}"
else
  echo "Warning: ${REQ_IN} not found, skipping."
fi

echo "=== Installing project in editable mode ==="
# Run from repo root so editable install works
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
pip install --no-cache-dir -e .

echo ""
echo "=== Setup Complete! ==="
echo "Activate with: conda activate ${ENV_NAME}"
