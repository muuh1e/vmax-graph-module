#!/bin/bash
#SBATCH --job-name=vmax_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --chdir=/home/jovyan/pfe/vmax-graph-module/cluster

set -euo pipefail

# -------- Paths (robust) --------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_TFRECORD="/home/jovyan/pfe/vmax-graph-module/training_subset.tfrecord"
RESULTS_DIR="${PROJECT_ROOT}/results/cluster_run_$(date +%Y%m%d_%H%M%S)"
PROCESSED_DIR="${PROJECT_ROOT}/processed_graphs_cluster"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${PROCESSED_DIR}"

echo "===== SLURM INFO ====="
echo "Job ID: ${SLURM_JOB_ID:-N/A}"
echo "Node: ${SLURMD_NODENAME:-N/A}"
echo "Working dir: $(pwd)"
echo "Script dir: ${SCRIPT_DIR}"
echo "Project root: ${PROJECT_ROOT}"
echo "TFRecord: ${DATA_TFRECORD}"
echo "Results: ${RESULTS_DIR}"
echo "======================"

# -------- Load conda & activate env --------
# Prefer a fixed conda path when available (Kubeflow images typically use /opt/conda)
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  # fallback
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda activate vmax_gnn

echo "===== ENV CHECK ====="
which python
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cuda_available', torch.cuda.is_available())"
nvidia-smi || true
echo "====================="

# Fail early if dataset is missing
if [[ ! -f "${DATA_TFRECORD}" ]]; then
  echo "ERROR: TFRecord not found at: ${DATA_TFRECORD}"
  exit 2
fi

# -------- Run Training --------
cd "${PROJECT_ROOT}/gnn_pipeline/train"

python run.py \
  --model simple_gnn \
  --graph_preset baseline \
  --tfrecord "${DATA_TFRECORD}" \
  --processed_dir "${PROCESSED_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --checkpoint_dir "${RESULTS_DIR}/checkpoints" \
  --epochs 100 \
  --batch_size 32 \
  --hidden 256 \
  --num_layers 4 \
  --lr 5e-4

echo "Training finished!"
