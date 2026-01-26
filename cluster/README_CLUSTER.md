# Cluster Deployment Guide

This folder contains scripts to help you deploy and train the V-Max GNN pipeline on a compute cluster (SLURM-based).

## 1. Preparing for Upload

Before uploading to the cluster:

1.  **Clean up**: You don't need to upload `processed_graphs/` or `checkpoints/` if they are large.
2.  **Upload**: Copy the entire `V-Max` directory to your cluster.
    ```bash
    scp -r /path/to/V-Max user@cluster:/path/to/remote/
    ```

## 2. Setting up Variable Environment (On Cluster)

1.  Log in to the cluster.
2.  Navigate to the project folder: `cd V-Max/cluster`
3.  Run the setup script (requires internet access or pre-loaded modules):
    ```bash
    bash prepare_env.sh
    ```
    This creates a conda environment `vmax_gnn` and installs PyTorch, PyG, and the project package.

## 3. Running Training Jobs

1.  **Edit `cluster/train_job.sh`**:
    *   Update `#SBATCH` directives (partition, GPU type, time limit).
    *   **CRITICAL**: Update `DATA_DIR` to point to where the Waymo TFRecords are located on the cluster.

2.  **Submit the job**:
    ```bash
    sbatch train_job.sh
    ```

3.  **Monitor**:
    *   Check status: `squeue -u $USER`
    *   Check logs: `tail -f logs/train_*.out`

## Troubleshooting

- **Import Errors**: Ensure you installed the package with `pip install -e .` (done by `prepare_env.sh`).
- **CUDA/GPU Issues**: Ensure the PyTorch version matches the cluster's CUDA drivers (check `nvidia-smi`).
- **Data Loading**: Using local SSDs (e.g., `/scratch`) is faster than network storage for TFRecords.

## Project Structure for Cluster

- `vmax_gnn/`: Core package (installed via pip)
- `gnn_pipeline/`: Training scripts
- `requirements_gnn.txt`: Dependencies list
- `setup.py`: Packaging configuration
