#!/bin/bash
#SBATCH --job-name=vmax_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --cpus-per-task=8   # Request 8 CPUs
#SBATCH --mem=32G           # Request 32GB RAM
#SBATCH --partition=gpu     # Change this to your cluster's GPU partition name

# cluster/train_job.sh

# 1. Load Modules (Adjust based on your cluster)
# module load anaconda3
# module load cuda/11.8

# 2. Setup Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vmax_gnn

# 3. Define Paths
PROJECT_ROOT=$(pwd)/..  # Assuming running from cluster/ dir or submitted from there
DATA_DIR="/path/to/cluster/data/waymo_converted"  # <-- UPDATE THIS
RESULTS_DIR="$PROJECT_ROOT/results/cluster_run_$(date +%Y%m%d_%H%M%S)"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Project Root: $PROJECT_ROOT"
echo "Data Dir: $DATA_DIR"
echo "Results Dir: $RESULTS_DIR"

# 4. Run Training
# Example: Hierarchical GNN with Smart A2A
cd $PROJECT_ROOT/gnn_pipeline/train

python run.py \
    --model hierarchical_gnn \
    --graph_preset smart_a2a_k5 \
    --tfrecord "$DATA_DIR/training.tfrecord" \
    --processed_dir "$PROJECT_ROOT/processed_graphs_cluster" \
    --results_dir "$RESULTS_DIR" \
    --checkpoint_dir "$RESULTS_DIR/checkpoints" \
    --epochs 100 \
    --batch_size 32 \
    --hidden 256 \
    --num_layers 4 \
    --lr 5e-4 

echo "Training finished!"
