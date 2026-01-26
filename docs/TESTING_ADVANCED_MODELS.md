# Testing Advanced GNN Models

This guide explains how to test the **TypedHeteroGNN** and **HierarchicalGNN** models that are more advanced than the baseline SimpleHeteroGNN.

## Models Overview

### 1. SimpleHeteroGNN (Baseline)
- Basic heterogeneous GNN with uniform message passing
- Simple MLP encoders + HeteroConv layers
- Single-mode trajectory prediction
- **Use case**: Baseline model for quick experiments

### 2. TypedHeteroGNN (LaneGCN-inspired)
- Semantically-aware convolutions per edge type (successor, predecessor, neighbor)
- Dilated lane graph for multi-hop reasoning
- Agent-Lane fusion with bidirectional cross-attention
- Typed A2A convolutions for different interaction types (following, leading, adjacent, crossing)
- **Use case**: When you want explicit modeling of different relationship types

### 3. HierarchicalGNN (HiVT-inspired)
- Local interaction via k-nearest neighbor attention
- Global interaction via sparse attention
- Ego-centric aggregation with dedicated ego processing
- Multi-modal prediction (predicts K possible futures with confidences)
- **Use case**: When you want multi-modal predictions and hierarchical reasoning

---

## Quick Start Commands

### Test TypedHeteroGNN

```bash
# Basic training with typed GNN (default settings)
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --epochs 50 \
    --batch_size 16

# With lane-to-lane edges (recommended for typed_gnn)
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset with_l2l \
    --epochs 50

# With optimized graph config
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset optimized \
    --epochs 50 \
    --hidden 256

# With edge attributes (uses edge features in message passing)
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset with_l2l \
    --use_edge_attr \
    --epochs 50
```

### Test HierarchicalGNN

```bash
# Basic training with hierarchical GNN
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --epochs 50 \
    --batch_size 16

# With larger model (recommended for hierarchical_gnn)
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --hidden 256 \
    --epochs 50

# With optimized graph config
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --graph_preset smart_a2a_k5 \
    --epochs 50 \
    --hidden 256

# Fewer samples for quick test
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --max_records 100 \
    --epochs 20 \
    --batch_size 8
```

---

## Side-by-Side Comparison

Run all three models with the same settings to compare:

```bash
# 1. Baseline (SimpleHeteroGNN)
python gnn_pipeline/train/run.py \
    --model simple_gnn \
    --graph_preset baseline \
    --epochs 30 \
    --checkpoint_dir checkpoints/simple_gnn

# 2. TypedHeteroGNN
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset with_l2l \
    --epochs 30 \
    --checkpoint_dir checkpoints/typed_gnn

# 3. HierarchicalGNN
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --graph_preset smart_a2a_k5 \
    --epochs 30 \
    --checkpoint_dir checkpoints/hierarchical_gnn
```

Then compare the validation metrics (ADE/FDE) in the final output.

---

## Quick Test (5 minutes)

Test all models quickly with a small subset of data:

```bash
# Quick test with 50 records, 10 epochs
for model in simple_gnn typed_gnn hierarchical_gnn; do
    echo "Testing $model..."
    python gnn_pipeline/train/run.py \
        --model $model \
        --max_records 50 \
        --epochs 10 \
        --batch_size 8 \
        --checkpoint_dir checkpoints/$model \
        --results_dir results/$model
done
```

---

## Understanding the Output

During training, you'll see:
```
Epoch 1/50
----------------------------------------
Train Loss: 15.2341 | ADE: 4.53m | FDE: 12.31m
Val   Loss: 14.8912 | ADE: 4.21m | FDE: 11.89m
  -> Saved best model (ADE: 4.21m)
```

**Key Metrics:**
- **Loss**: MSE loss value (lower is better)
- **ADE** (Average Displacement Error): Average error across all timesteps (in meters)
- **FDE** (Final Displacement Error): Error at the final timestep (in meters)

**Expected Performance (after 50 epochs):**
- **SimpleHeteroGNN**: ADE ~4.3-4.5m, FDE ~12-13m
- **TypedHeteroGNN**: ADE ~4.0-4.3m, FDE ~11-12m (better with L2L edges)
- **HierarchicalGNN**: ADE ~3.8-4.2m, FDE ~10-11m (with larger hidden size)

---

## Model-Specific Options

### TypedHeteroGNN Options

```bash
# Use dilated lane convolutions (multi-hop reasoning)
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset typed_l2l \
    --epochs 50

# Increase number of interaction types
# (Edit typed_gnn.py: num_interaction_types parameter)

# Use typed L2L edges
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --l2l_mode typed \
    --epochs 50
```

### HierarchicalGNN Options

```bash
# Multi-modal prediction is enabled by default
# Adjust k-nearest neighbors (edit hierarchical_gnn.py):
# - local_k_agents: how many nearby agents to attend to (default: 10)
# - local_k_lanes: how many nearby lanes to attend to (default: 5)
# - global_k: sparse attention top-k (default: 20)

# More layers for deeper hierarchy
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --num_layers 4 \
    --epochs 50
```

---

## Recommended Configurations

### For TypedHeteroGNN
```bash
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset typed_l2l \
    --hidden 256 \
    --num_layers 4 \
    --conv_type gat \
    --use_edge_attr \
    --epochs 50 \
    --batch_size 12
```

### For HierarchicalGNN
```bash
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --graph_preset smart_a2a_k5 \
    --hidden 256 \
    --num_layers 3 \
    --epochs 50 \
    --batch_size 12
```

---

## Checking Results

After training, results are saved to:
- **Checkpoints**: `checkpoints/best_model.pt` (best model weights)
- **Configs**: `results/graph_config.json`, `results/model_config.json`
- **Training history**: Stored in trainer.history (printed at end)

To load and evaluate a trained model:

```python
import torch
from gnn_pipeline import get_model, GraphConfig, ModelConfig

# Load configs
graph_config = GraphConfig.load('results/graph_config.json')
model_config = ModelConfig.load('results/model_config.json')

# Create model
model = get_model(
    model_config=model_config,
    graph_config=graph_config,
    agent_in_channels=66,
    lane_in_channels=40,
)

# Load weights
checkpoint = torch.load('checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

print(f"Best Val ADE: {checkpoint['val_ade']:.2f}m")
print(f"Best Val FDE: {checkpoint['val_fde']:.2f}m")
```

---

## Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
--batch_size 8

# Reduce hidden dimension
--hidden 128

# Reduce number of records
--max_records 500

# Use smaller graph
--graph_preset baseline
```

### Slow Training
```bash
# Use fewer GNN layers
--num_layers 2

# Reduce number of records
--max_records 1000

# Use CPU if GPU is slow
--device cpu
```

### Poor Performance
```bash
# Train longer
--epochs 100

# Use larger model
--hidden 256 --num_layers 4

# Use better graph config
--graph_preset optimized

# Use edge attributes
--use_edge_attr
```

---

## Next Steps

1. **Start with quick test**: Run the 5-minute test to verify everything works
2. **Full comparison**: Run all three models with 30-50 epochs
3. **Tune hyperparameters**: Adjust hidden size, num_layers, learning rate
4. **Experiment with graph configs**: Try different presets and edge types
5. **Visualize predictions**: Use the visualization tools in `viz_hetero_graph.py`

For more details, see:
- `docs/PIPELINE_GUIDE.md` - General pipeline documentation
- `gnn_pipeline/models/typed_gnn.py` - TypedHeteroGNN implementation
- `gnn_pipeline/models/hierarchical_gnn.py` - HierarchicalGNN implementation
