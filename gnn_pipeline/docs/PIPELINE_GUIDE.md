# GNN Pipeline Guide

A modular architecture for heterogeneous graph neural networks for motion prediction.

## Overview

The `gnn_pipeline` package provides a config-driven approach to building and training GNN models for trajectory prediction. Key features:

- **Config-driven**: Use `GraphConfig` and `ModelConfig` to control graph structure and model architecture
- **Modular**: Easy to add new models, edge types, or graph features
- **Reproducible**: Configs are saved with checkpoints for experiment tracking

## Quick Start

The training script is located at `gnn_pipeline/train/run.py`. You can run it from the project root:

```bash
# Basic training with default settings
python gnn_pipeline/train/run.py --epochs 50

# Training with specific graph configuration
python gnn_pipeline/train/run.py --graph_preset with_l2l --epochs 50

# Full configuration with all features
python gnn_pipeline/train/run.py --graph_preset full --conv_type gat --num_layers 5
```

## CLI Usage Examples

### Basic Training

```bash
# Baseline training (agents + lanes, ego->others A2A edges)
python gnn_pipeline/train/run.py --epochs 50 --batch_size 16

# With more hidden dimensions
python gnn_pipeline/train/run.py --hidden 256 --num_layers 4
```

### Graph Configuration Presets

#### Basic Presets

```bash
# Baseline: agents + lanes only (BEST ADE SO FAR)
python gnn_pipeline/train/run.py --graph_preset baseline

# With lane-to-lane edges
python gnn_pipeline/train/run.py --graph_preset with_l2l

# With traffic lights
python gnn_pipeline/train/run.py --graph_preset with_tl

# Full agent-to-agent connectivity (all pairs, not just ego->others)
python gnn_pipeline/train/run.py --graph_preset with_full_a2a

# Everything enabled
python gnn_pipeline/train/run.py --graph_preset full
```

#### Advanced Presets (NEW)

```bash
# k-nearest A2A (k=5) - balanced agent interaction
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5

# Typed L2L edges (LaneGCN-style: successor, predecessor, left_of, right_of)
python gnn_pipeline/train/run.py --graph_preset typed_l2l

# Filtered traffic lights (only controlling TLs)
python gnn_pipeline/train/run.py --graph_preset filtered_tl

# Optimized: all smart features combined
python gnn_pipeline/train/run.py --graph_preset optimized
```

### Override Presets with Flags

#### Basic Overrides

```bash
# Start with baseline and add L2L edges
python gnn_pipeline/train/run.py --include_l2l

# Start with baseline and add traffic lights
python gnn_pipeline/train/run.py --include_tl

# Combine multiple overrides
python gnn_pipeline/train/run.py --include_l2l --full_a2a

# Custom combination
python gnn_pipeline/train/run.py --include_l2l --include_tl --hidden 256
```

#### Advanced Overrides (NEW)

```bash
# A2A mode: k-nearest with custom k
python gnn_pipeline/train/run.py --a2a_mode k_nearest --a2a_k 10

# L2L mode: typed edges
python gnn_pipeline/train/run.py --include_l2l --l2l_mode typed

# TL filtering: only controlling traffic lights
python gnn_pipeline/train/run.py --include_tl --tl_mode controlling --tl_connect_to lanes

# Lane filtering: ego-relevant lanes only
python gnn_pipeline/train/run.py --lane_mode ego_relevant --lane_max 40

# Complex custom combination
python gnn_pipeline/train/run.py \
  --a2a_mode k_nearest --a2a_k 7 \
  --include_l2l --l2l_mode typed \
  --lane_mode ego_relevant --lane_max 50
```

### Model Architecture Options

```bash
# Use GAT instead of SAGE
python gnn_pipeline/train/run.py --conv_type gat

# Deeper model
python gnn_pipeline/train/run.py --num_layers 5

# Larger hidden dimension
python gnn_pipeline/train/run.py --hidden 256

# Use edge attributes (experimental)
python gnn_pipeline/train/run.py --use_edge_attr
```

## Graph Configuration Options

### Basic Options

| Flag | Effect | Default |
|------|--------|---------|
| `--graph_preset` | Configuration preset | `baseline` |
| `--include_l2l` | Add lane-to-lane edges | `False` |
| `--include_tl` | Add traffic light nodes and edges | `False` |
| `--full_a2a` | All agent pairs (not just ego→others) | `False` |

### Advanced Options (NEW)

| Flag | Options | Default | Description |
|------|---------|---------|-------------|
| `--a2a_mode` | `ego_only`, `k_nearest`, `same_lane`, `directional` | `ego_only` | A2A edge building mode |
| `--a2a_k` | integer | `5` | k for k-nearest mode |
| `--a2a_max_dist` | float | `50.0` | Max distance for A2A edges (meters) |
| `--l2l_mode` | `all`, `typed`, `successor_only` | `all` | L2L edge mode |
| `--tl_mode` | `all`, `relevant`, `controlling` | `all` | TL filtering mode |
| `--tl_connect_to` | `agents`, `lanes`, `both` | `both` | What TLs connect to |
| `--lane_mode` | `all`, `drivable`, `ego_relevant`, `ego_path` | `all` | Lane filtering mode |
| `--lane_max` | integer | `80` | Max number of lanes to keep |

### Presets Explained

| Preset | L2L | TL | A2A Mode | Description |
|--------|-----|-----|----------|-------------|
| `baseline` | ❌ | ❌ | ego_only | Basic agent+lane graph (BEST ADE) |
| `with_l2l` | ✅ | ❌ | ego_only | + lane connectivity |
| `with_tl` | ❌ | ✅ | ego_only | + traffic lights |
| `with_full_a2a` | ❌ | ❌ | all_pairs | + all agent pairs |
| `full` | ✅ | ✅ | all_pairs | Everything enabled |
| `smart_a2a_k5` | ❌ | ❌ | k_nearest (k=5) | + balanced agent interaction |
| `typed_l2l` | ✅ (typed) | ❌ | ego_only | + semantic lane edges |
| `filtered_tl` | ❌ | ✅ (filtered) | ego_only | + controlling TLs only |
| `optimized` | ✅ (typed) | ✅ (filtered) | k_nearest (k=5) | All smart features |

## Model Configuration Options

| Flag | Default | Options | Description |
|------|---------|---------|-------------|
| `--model` | `simple_gnn` | `simple_gnn`, `typed_gnn`, `hierarchical_gnn` | Model architecture |
| `--hidden` | `128` | - | Hidden dimension |
| `--num_layers` | `3` | - | Number of GNN layers |
| `--dropout` | `0.1` | - | Dropout rate |
| `--conv_type` | `sage` | `sage`, `gat` | Convolution type |
| `--use_edge_attr` | `False` | - | Use edge attributes |

### Model Types Explained

- **`simple_gnn`** (SimpleHeteroGNN): Basic heterogeneous GNN, uniform message passing. Best for baseline.
- **`typed_gnn`** (TypedHeteroGNN): LaneGCN-inspired with typed convolutions. Best with `--graph_preset typed_l2l`.
- **`hierarchical_gnn`** (HierarchicalGNN): HiVT-inspired hierarchical encoder with multi-modal decoding. State-of-the-art (more compute).

## Adding a New Model

1. Create a new file in `gnn_pipeline/models/`:

```python
# gnn_pipeline/models/my_model.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from . import register_model
from .base_model import BaseMotionPredictor


@register_model("my_model")
class MyModel(BaseMotionPredictor):
    """My custom motion prediction model."""

    def __init__(
        self,
        agent_in_channels: int,
        lane_in_channels: int,
        tl_in_channels: int = 12,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_future_steps: int = 80,
        dropout: float = 0.1,
        conv_type: str = "sage",
        edge_types: Optional[List[Tuple[str, str, str]]] = None,
        use_edge_attr: bool = False,
    ):
        super().__init__(num_future_steps=num_future_steps)
        
        # Your model architecture here
        self.encoder = nn.Linear(agent_in_channels, hidden_channels)
        self.decoder = nn.Linear(hidden_channels, num_future_steps * 2)

    def forward(
        self,
        data: HeteroData,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: HeteroData batch
            ego_indices: Optional tensor of ego agent indices
            
        Returns:
            Predicted trajectories [batch_size, 80, 2]
        """
        # Get ego indices
        if ego_indices is None:
            ego_indices = data.ego_indices_global
        
        # Your forward logic here
        ego_features = data['agent'].x[ego_indices]
        h = self.encoder(ego_features)
        pred = self.decoder(h)
        
        return pred.view(-1, self.num_future_steps, 2)
```

2. Import in `gnn_pipeline/models/__init__.py`:

```python
from .my_model import MyModel
```

3. Use the model:

```bash
python train.py --model my_model
```

## Project Structure

```
gnn_pipeline/
├── __init__.py              # Package exports
├── configs/
│   ├── __init__.py
│   ├── graph_config.py      # GraphConfig dataclass
│   └── model_config.py      # ModelConfig dataclass
├── graphs/
│   ├── __init__.py
│   ├── hetero_graph.py      # Graph building
│   └── graph_dataset.py     # PyTorch Geometric dataset
├── models/
│   ├── __init__.py          # Model registry
│   ├── base_model.py        # Abstract base class
│   └── simple_gnn.py        # SimpleHeteroGNN implementation
├── train/
│   ├── __init__.py
│   ├── metrics.py           # ADE/FDE computation
│   └── trainer.py           # Training loop
└── docs/
    └── PIPELINE_GUIDE.md    # This file
```

## Results Tracking

Training results are saved to the `results/` directory:
- `graph_config.json`: Graph configuration used
- `model_config.json`: Model configuration used

Checkpoints are saved to `checkpoints/`:
- `best_model.pt`: Best model (by validation ADE)

Training plots are saved to `plots/`:
- `{model_name}_training_curves.png`: ADE/FDE curves over epochs

**Tip**: Use descriptive model names in plots by varying `--graph_preset` and `--model` flags. The plot filename will include these for easy comparison.

## Expected Results

With the default baseline configuration:
- **Val ADE**: ~4.3-4.5m (BEST SO FAR)
- **Val FDE**: ~12-13m

Results may vary with different graph configurations and hyperparameters.

**Note**: Baseline config currently achieves the best ADE. Your goal is to beat this by experimenting with different models and configs!

## Programmatic Usage

```python
from gnn_pipeline import (
    GraphConfig,
    ModelConfig,
    get_model,
    WaymoGraphDataset,
    Trainer,
)

# Create configs
graph_config = GraphConfig.typed_l2l()  # Use any preset
model_config = ModelConfig(
    model_type="typed_gnn",  # Match with graph config
    hidden_channels=256,
    num_layers=4,
    conv_type="gat",
)
model_config.sync_with_graph_config(graph_config)

# Load dataset
dataset = WaymoGraphDataset(
    root="./processed_graphs",
    tfrecord_path="path/to/data.tfrecord",
    config=graph_config,
)

# Create model
model = get_model(
    model_config=model_config,
    graph_config=graph_config,
    agent_in_channels=66,
    lane_in_channels=40,
    tl_in_channels=12,
)

# Create data loaders and trainer...
```

---

## Experimentation Workflow

### Goal: Beat Baseline ADE

The baseline config achieves the best ADE so far. Here's a systematic approach to find better configs:

#### Step 1: Test Model Architectures

Start by testing different models with baseline graph config:

```bash
# 1. Baseline (already tested)
python gnn_pipeline/train/run.py --graph_preset baseline --model simple_gnn --epochs 50

# 2. TypedGNN (semantic convolutions)
python gnn_pipeline/train/run.py --graph_preset baseline --model typed_gnn --epochs 50

# 3. HierarchicalGNN (state-of-the-art)
python gnn_pipeline/train/run.py --graph_preset baseline --model hierarchical_gnn --epochs 50
```

#### Step 2: Test Graph Presets

Using the best model from Step 1, test different graph configurations:

```bash
# Assuming simple_gnn was best:

# 1. k-NN A2A (balanced interaction)
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --model simple_gnn --epochs 50

# 2. Typed L2L (semantic lanes)
python gnn_pipeline/train/run.py --graph_preset typed_l2l --model simple_gnn --epochs 50

# 3. Filtered TL (efficient traffic lights)
python gnn_pipeline/train/run.py --graph_preset filtered_tl --model simple_gnn --epochs 50

# 4. Optimized (all smart features)
python gnn_pipeline/train/run.py --graph_preset optimized --model simple_gnn --epochs 50
```

**Note**: If using `typed_l2l` preset, consider switching to `typed_gnn` model to fully exploit typed edges.

#### Step 3: Tune Hyperparameters

With the best config from Step 2, tune model size:

```bash
# Assuming smart_a2a_k5 + simple_gnn was best:

# Wider network
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --hidden 256 --epochs 50

# Deeper network
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --num_layers 5 --epochs 50

# GAT instead of SAGE
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --conv_type gat --epochs 50

# Combination
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --hidden 256 --num_layers 4 --conv_type gat --epochs 50
```

#### Step 4: Fine-tune Config Parameters

Tune specific parameters of your best configuration:

```bash
# Example: Tune k for k-NN A2A
python gnn_pipeline/train/run.py --graph_preset baseline --a2a_mode k_nearest --a2a_k 3 --epochs 50
python gnn_pipeline/train/run.py --graph_preset baseline --a2a_mode k_nearest --a2a_k 7 --epochs 50
python gnn_pipeline/train/run.py --graph_preset baseline --a2a_mode k_nearest --a2a_k 10 --epochs 50

# Example: Tune lane filtering
python gnn_pipeline/train/run.py --graph_preset optimized --lane_max 20 --epochs 50
python gnn_pipeline/train/run.py --graph_preset optimized --lane_max 60 --epochs 50
```

### Tracking Your Experiments

Create a simple spreadsheet or log to track:

| Experiment | Graph Preset | Model | Hidden | Layers | Conv | Val ADE | Val FDE | Notes |
|------------|--------------|-------|--------|--------|------|---------|---------|-------|
| baseline | baseline | simple_gnn | 128 | 3 | sage | 4.35 | 12.5 | Best so far |
| exp_1 | smart_a2a_k5 | simple_gnn | 128 | 3 | sage | ? | ? | Test k-NN |
| exp_2 | typed_l2l | typed_gnn | 128 | 3 | sage | ? | ? | Semantic lanes |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

### Quick Tips

- **Start simple**: Don't jump to `optimized` or `hierarchical_gnn` first.
- **One change at a time**: Easier to understand what helps.
- **Check graph stats**: Look at logs for node/edge counts to verify filtering.
- **Watch for overfitting**: Compare train vs val ADE.
- **Training time matters**: Balance accuracy vs speed for iteration.

### Common Combinations Worth Trying

```bash
# 1. Typed lane graph with matching model
python gnn_pipeline/train/run.py --graph_preset typed_l2l --model typed_gnn --hidden 256

# 2. k-NN agents with wider network
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --hidden 256 --num_layers 4

# 3. Filtered scene for efficiency
python gnn_pipeline/train/run.py --lane_mode ego_relevant --lane_max 40 --a2a_mode k_nearest --a2a_k 5

# 4. Hierarchical with baseline (test model alone)
python gnn_pipeline/train/run.py --graph_preset baseline --model hierarchical_gnn
```

---

## Additional Resources

- **CONFIG_REFERENCE.md**: Comprehensive guide to all configuration options
- **ARCHITECTURE.md**: Internal design and data flow details
- **Results tracking**: Check `results/`, `checkpoints/`, and `plots/` directories after training

For questions or issues, check the main project README or documentation.
```
