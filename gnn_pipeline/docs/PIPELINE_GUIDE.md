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

```bash
# Baseline: agents + lanes only
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

### Override Presets with Flags

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

| Flag | Effect | Default |
|------|--------|---------|
| `--graph_preset` | Configuration preset | `baseline` |
| `--include_l2l` | Add lane-to-lane edges | `False` |
| `--include_tl` | Add traffic light nodes and edges | `False` |
| `--full_a2a` | All agent pairs (not just ego→others) | `False` |

### Presets Explained

| Preset | L2L | TL | Full A2A | Description |
|--------|-----|-----|---------|-------------|
| `baseline` | ❌ | ❌ | ❌ | Basic agent+lane graph |
| `with_l2l` | ✅ | ❌ | ❌ | + lane connectivity |
| `with_tl` | ❌ | ✅ | ❌ | + traffic lights |
| `with_full_a2a` | ❌ | ❌ | ✅ | + all agent pairs |
| `full` | ✅ | ✅ | ✅ | Everything enabled |

## Model Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `simple_gnn` | Model architecture |
| `--hidden` | `128` | Hidden dimension |
| `--num_layers` | `3` | Number of GNN layers |
| `--dropout` | `0.1` | Dropout rate |
| `--conv_type` | `sage` | Convolution type (`sage` or `gat`) |
| `--use_edge_attr` | `False` | Use edge attributes |

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

## Expected Results

With the default baseline configuration:
- **Val ADE**: ~4.3-4.5m
- **Val FDE**: ~12-13m

Results may vary with different graph configurations and hyperparameters.

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
graph_config = GraphConfig.with_l2l()
model_config = ModelConfig(
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
)

# Create data loaders and trainer...
```
