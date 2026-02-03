"""Model registry and factory functions."""

from typing import Type, Dict, Optional, List

import torch.nn as nn

from .base_model import BaseMotionPredictor

# Global model registry
MODEL_REGISTRY: Dict[str, Type[BaseMotionPredictor]] = {}


def register_model(name: str):
    """
    Decorator to register a model class in the registry.
    
    Usage:
        @register_model("my_model")
        class MyModel(BaseMotionPredictor):
            ...
    
    Args:
        name: Name to register the model under
        
    Returns:
        Decorator function
    """
    def decorator(cls: Type[BaseMotionPredictor]) -> Type[BaseMotionPredictor]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        if not issubclass(cls, BaseMotionPredictor):
            raise TypeError(f"Model must be a subclass of BaseMotionPredictor")
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(
    model_config,
    graph_config,
    agent_in_channels: int,
    lane_in_channels: int,
    tl_in_channels: int = 12,
    goal_in_channels: int = 10,
) -> BaseMotionPredictor:
    """
    Factory function to create a model from configs.
    
    Args:
        model_config: ModelConfig instance
        graph_config: GraphConfig instance
        agent_in_channels: Number of agent input features
        lane_in_channels: Number of lane input features
        tl_in_channels: Number of traffic light input features
        goal_in_channels: Number of goal input features
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not registered
    """
    model_type = model_config.model_type
    
    if model_type not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available models: {available}"
        )
    
    model_cls = MODEL_REGISTRY[model_type]
    
    # Get edge types based on configs
    edge_types = model_config.get_edge_types(graph_config)
    
    # Create model with appropriate arguments
    model_kwargs = dict(
        agent_in_channels=agent_in_channels,
        lane_in_channels=lane_in_channels,
        tl_in_channels=tl_in_channels,
        goal_in_channels=goal_in_channels,
        hidden_channels=model_config.hidden_channels,
        num_layers=model_config.num_layers,
        num_future_steps=model_config.num_future_steps,
        dropout=model_config.dropout,
        conv_type=model_config.conv_type,
        edge_types=edge_types,
        use_edge_attr=model_config.use_edge_attr,
    )

    # Add temporal encoder params if supported (Phase 2A)
    if hasattr(model_config, 'use_temporal_encoder'):
        model_kwargs.update(
            use_temporal_encoder=model_config.use_temporal_encoder,
            temporal_encoder_type=model_config.temporal_encoder_type,
            temporal_hidden_dim=model_config.temporal_hidden_dim,
            temporal_num_layers=model_config.temporal_num_layers,
            temporal_num_heads=model_config.temporal_num_heads,
            temporal_dropout=model_config.temporal_dropout,
        )

    # Add polyline encoder params if supported (Phase 2C)
    if hasattr(model_config, 'use_polyline_encoder'):
        model_kwargs.update(
            use_polyline_encoder=model_config.use_polyline_encoder,
            polyline_encoder_type=model_config.polyline_encoder_type,
            polyline_hidden_dim=model_config.polyline_hidden_dim,
            polyline_num_layers=model_config.polyline_num_layers,
        )

    model = model_cls(**model_kwargs)

    return model


def list_models() -> List[str]:
    """
    List all registered model names.
    
    Returns:
        List of registered model names
    """
    return list(MODEL_REGISTRY.keys())


# Import models to trigger registration
from .simple_gnn import SimpleHeteroGNN
from .typed_gnn import TypedHeteroGNN
from .hierarchical_gnn import HierarchicalGNN

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    "list_models",
    "BaseMotionPredictor",
    "SimpleHeteroGNN",
    "TypedHeteroGNN",
    "HierarchicalGNN",
]
