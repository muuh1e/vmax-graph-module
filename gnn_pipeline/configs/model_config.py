"""Model configuration for GNN architectures."""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .graph_config import GraphConfig


@dataclass
class ModelConfig:
    """
    Configuration for GNN model architecture.
    
    Attributes:
        model_type: Model architecture type (e.g., "simple_gnn")
        hidden_channels: Hidden dimension size
        num_layers: Number of GNN layers
        dropout: Dropout rate
        conv_type: Convolution type ("sage" or "gat")
        use_edge_attr: Whether to use edge attributes in message passing
        num_future_steps: Number of future timesteps to predict
        
        # Edge type usage flags
        use_a2a: Use agent-to-agent edges
        use_a2l: Use agent-to-lane edges
        use_l2a: Use lane-to-agent (reverse) edges
        use_l2l: Use lane-to-lane edges
        use_a2tl: Use agent-to-traffic-light edges
        use_l2tl: Use lane-to-traffic-light edges
    """
    
    # Model architecture
    model_type: str = "simple_gnn"
    hidden_channels: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    conv_type: str = "sage"  # "sage" or "gat"
    use_edge_attr: bool = False
    num_future_steps: int = 80
    
    # Edge type usage flags
    use_a2a: bool = True
    use_a2l: bool = True
    use_l2a: bool = True  # Reverse of a2l
    use_l2l: bool = False
    use_a2tl: bool = False
    use_l2tl: bool = False
    
    # -------------------------
    # Serialization methods
    # -------------------------
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        """Create config from dictionary."""
        return cls(**d)
    
    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    # -------------------------
    # Edge type methods
    # -------------------------
    
    def get_edge_types(self, graph_config: GraphConfig) -> list[tuple[str, str, str]]:
        """
        Get list of edge types to use based on model config and graph config.
        
        Returns edge types that are both:
        1. Enabled in this model config (use_* flags)
        2. Available in the graph config
        
        Args:
            graph_config: Graph configuration to check availability
            
        Returns:
            List of (src_type, edge_name, dst_type) tuples
        """
        edge_types = []
        
        # Agent-to-agent (always available)
        if self.use_a2a:
            edge_types.append(("agent", "to", "agent"))
        
        # Agent-to-lane (always available)
        if self.use_a2l:
            edge_types.append(("agent", "to", "lane"))
        
        # Lane-to-agent (reverse, always available)
        if self.use_l2a:
            edge_types.append(("lane", "rev_to", "agent"))
        
        # Lane-to-lane (only if graph has it)
        if self.use_l2l and graph_config.include_l2l:
            edge_types.append(("lane", "to", "lane"))
        
        # Agent-to-traffic-light (only if graph has TLs)
        if self.use_a2tl and (graph_config.include_tl or graph_config.include_a2tl):
            edge_types.append(("agent", "to", "tl"))
        
        # Lane-to-traffic-light (only if graph has TLs)
        if self.use_l2tl and (graph_config.include_tl or graph_config.include_l2tl):
            edge_types.append(("lane", "to", "tl"))
        
        return edge_types
    
    def sync_with_graph_config(self, graph_config: GraphConfig) -> None:
        """
        Update edge type flags to match graph config.
        
        Enables edge types that are available in the graph.
        """
        self.use_l2l = graph_config.include_l2l
        self.use_a2tl = graph_config.include_tl or graph_config.include_a2tl
        self.use_l2tl = graph_config.include_tl or graph_config.include_l2tl
