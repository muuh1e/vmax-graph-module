"""
GNN Pipeline for Motion Prediction

A modular architecture for heterogeneous graph neural networks
with config-driven model and graph selection.
"""

__version__ = "0.1.0"

from .configs import GraphConfig, ModelConfig
from .models import get_model, list_models, register_model
from .graphs import build_hetero_graph, WaymoGraphDataset
from .train import Trainer, compute_ade, compute_fde, compute_metrics

__all__ = [
    # Configs
    "GraphConfig",
    "ModelConfig",
    # Models
    "get_model",
    "list_models",
    "register_model",
    # Graphs
    "build_hetero_graph",
    "WaymoGraphDataset",
    # Training
    "Trainer",
    "compute_ade",
    "compute_fde",
    "compute_metrics",
]
