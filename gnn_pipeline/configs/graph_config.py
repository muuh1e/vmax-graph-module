"""Graph configuration for heterogeneous graph construction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GraphConfig:
    """
    Configuration for heterogeneous graph construction.
    
    Controls which node types and edge types are included in the graph,
    as well as parameters for edge construction.
    
    Attributes:
        include_tl: Include traffic light nodes
        full_a2a: Full agent-to-agent connectivity (vs ego-only)
        include_l2l: Include lane-to-lane edges
        include_a2tl: Include agent-to-traffic-light edges
        include_l2tl: Include lane-to-traffic-light edges
        a2a_max_dist: Maximum distance for A2A edges (meters)
        a2l_k: Number of nearest lanes per agent
        l2l_max_dist: Maximum distance for L2L edges (meters)
        k_past: Number of past timesteps to include in features
        
        # A2A filtering (NEW)
        a2a_mode: Mode for A2A edge construction ("ego_only", "k_nearest", "same_lane", "directional")
        a2a_k_nearest: Number of nearest agents per agent for k_nearest mode
        a2a_same_lane_only: Only connect agents sharing a lane
        a2a_directional: Only connect agents ahead (in field of view)
        a2a_fov_angle: Field of view angle for directional mode (degrees)
        
        # L2L typing (NEW)
        l2l_mode: Mode for L2L edges ("all", "typed", "successor_only")
        l2l_typed_convs: Use separate convolutions per L2L edge type
        l2l_max_successor_dist: Max gap for successor edges (meters)
        l2l_max_neighbor_dist: Max lateral distance for neighbor edges (meters)
        
        # TL filtering (NEW)
        tl_filter_mode: TL filtering mode ("all", "relevant", "controlling")
        tl_connect_to: What TLs connect to ("agents", "lanes", "both")
        tl_max_relevant: Max number of TLs to keep per scenario
        tl_max_distance: Max distance from ego for TLs (meters)
        tl_ahead_only: Only TLs ahead of ego
        
        # Lane filtering (NEW)
        lane_filter_mode: Lane filtering ("all", "drivable", "ego_relevant", "ego_path")
        lane_max_count: Max number of lanes to keep
        lane_max_distance: Max distance from ego for lanes (meters)
        lane_drivable_only: Exclude non-drivable lanes
        
        # Enhanced edge features (Phase 1)
        use_enhanced_a2a_features: Enable trajectory + interaction features (20→32 dims)
        use_frenet_a2l_features: Enable Frenet coordinates for A2L (9→15 dims)
        a2a_collision_threshold: Distance threshold for collision prediction (meters)
        a2a_prediction_horizon: Time horizon for trajectory prediction (seconds)
    """
    
    # Node/edge inclusion flags (existing)
    include_tl: bool = False
    full_a2a: bool = False
    include_l2l: bool = False
    include_a2tl: bool = False
    include_l2tl: bool = False
    
    # Edge construction parameters (existing)
    a2a_max_dist: float = 50.0
    a2l_k: int = 3
    l2l_max_dist: float = 5.0
    k_past: int = 10
    
    # A2A filtering options (NEW)
    a2a_mode: str = "ego_only"  # "ego_only" | "k_nearest" | "same_lane" | "directional"
    a2a_k_nearest: int = 5
    a2a_same_lane_only: bool = False
    a2a_directional: bool = False
    a2a_fov_angle: float = 120.0  # degrees
    
    # L2L typing options (NEW)
    l2l_mode: str = "all"  # "all" | "typed" | "successor_only"
    l2l_typed_convs: bool = False
    l2l_max_successor_dist: float = 3.0
    l2l_max_neighbor_dist: float = 5.0
    
    # TL filtering options (NEW)
    tl_filter_mode: str = "all"  # "all" | "relevant" | "controlling"
    tl_connect_to: str = "both"  # "agents" | "lanes" | "both"
    tl_max_relevant: int = 4
    tl_max_distance: float = 100.0
    tl_ahead_only: bool = True
    
    # Lane filtering options (NEW)
    lane_filter_mode: str = "all"  # "all" | "drivable" | "ego_relevant" | "ego_path"
    lane_max_count: int = 80
    lane_max_distance: float = 50.0
    lane_drivable_only: bool = False
    
    # Enhanced edge features (Phase 1)
    use_enhanced_a2a_features: bool = True   # Enable trajectory + interaction features (20→32 dims)
    use_frenet_a2l_features: bool = True     # Enable Frenet coordinates (9→15 dims)
    a2a_collision_threshold: float = 3.0     # meters for collision prediction
    a2a_prediction_horizon: float = 2.0      # seconds for trajectory prediction
    
    # -------------------------
    # Preset class methods
    # -------------------------
    
    @classmethod
    def baseline(cls) -> GraphConfig:
        """Baseline configuration: agents + lanes only, minimal edges."""
        return cls(
            include_tl=False,
            full_a2a=False,
            include_l2l=False,
            include_a2tl=False,
            include_l2tl=False,
        )
    
    @classmethod
    def with_l2l(cls) -> GraphConfig:
        """Baseline + lane-to-lane edges."""
        return cls(
            include_tl=False,
            full_a2a=False,
            include_l2l=True,
            include_a2tl=False,
            include_l2tl=False,
        )
    
    @classmethod
    def with_tl(cls) -> GraphConfig:
        """Baseline + traffic light nodes and edges."""
        return cls(
            include_tl=True,
            full_a2a=False,
            include_l2l=False,
            include_a2tl=True,
            include_l2tl=True,
        )
    
    @classmethod
    def with_full_a2a(cls) -> GraphConfig:
        """Baseline + full agent-to-agent connectivity."""
        return cls(
            include_tl=False,
            full_a2a=True,
            include_l2l=False,
            include_a2tl=False,
            include_l2tl=False,
        )
    
    @classmethod
    def full(cls) -> GraphConfig:
        """Full configuration: all features enabled."""
        return cls(
            include_tl=True,
            full_a2a=True,
            include_l2l=True,
            include_a2tl=True,
            include_l2tl=True,
        )
    
    # --- NEW PRESETS ---
    
    @classmethod
    def smart_a2a_k5(cls) -> GraphConfig:
        """Baseline with k-nearest A2A (k=5)."""
        return cls(
            a2a_mode="k_nearest",
            a2a_k_nearest=5,
        )
    
    @classmethod
    def typed_l2l(cls) -> GraphConfig:
        """Baseline with typed L2L edges (LaneGCN style)."""
        return cls(
            include_l2l=True,
            l2l_mode="typed",
            l2l_typed_convs=True,
        )
    
    @classmethod
    def filtered_tl(cls) -> GraphConfig:
        """Baseline with filtered traffic lights (only controlling TLs via lanes)."""
        return cls(
            include_tl=True,
            tl_filter_mode="controlling",
            tl_connect_to="lanes",
            tl_max_relevant=2,
            include_l2tl=True,
            include_a2tl=False,
        )
    
    @classmethod
    def filtered_lanes(cls, max_count: int = 40) -> GraphConfig:
        """Baseline with filtered lanes (ego-relevant only)."""
        return cls(
            lane_filter_mode="ego_relevant",
            lane_max_count=max_count,
            lane_drivable_only=True,
        )
    
    @classmethod
    def optimized(cls) -> GraphConfig:
        """Optimized configuration: all smart features combined."""
        return cls(
            # Smart A2A
            a2a_mode="k_nearest",
            a2a_k_nearest=5,
            # Typed L2L
            include_l2l=True,
            l2l_mode="typed",
            l2l_typed_convs=True,
            # Filtered TL
            include_tl=True,
            tl_filter_mode="controlling",
            tl_connect_to="lanes",
            tl_max_relevant=2,
            include_l2tl=True,
            include_a2tl=False,
            # Filtered lanes
            lane_filter_mode="ego_relevant",
            lane_max_count=40,
            lane_drivable_only=True,
        )
    
    # -------------------------
    # Serialization methods
    # -------------------------
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> GraphConfig:
        """Create config from dictionary."""
        return cls(**d)
    
    def save(self, path: str | Path) -> None:
        """Save config to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str | Path) -> GraphConfig:
        """Load config from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))
    
    # -------------------------
    # Utility methods
    # -------------------------
    
    def describe(self) -> str:
        """
        Return a short string describing the configuration.
        
        Returns:
            String like "baseline", "baseline+l2l", "baseline+tl+l2l", etc.
        """
        parts = ["baseline"]
        
        if self.include_l2l:
            parts.append("l2l")
        if self.include_tl:
            parts.append("tl")
        if self.full_a2a:
            parts.append("full_a2a")
        if self.include_a2tl and not self.include_tl:
            parts.append("a2tl")
        if self.include_l2tl and not self.include_tl:
            parts.append("l2tl")
        
        # Check if it's actually baseline
        if len(parts) == 1:
            return "baseline"
        
        return "+".join(parts)
    
    def get_hash(self) -> str:
        """
        Return a short hash of the configuration for cache keys.
        
        Returns:
            8-character hex string
        """
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_edge_types(self) -> list[tuple[str, str, str]]:
        """
        Return list of edge types that will be present in the graph.
        
        Returns:
            List of (src_type, edge_name, dst_type) tuples
        """
        edge_types = [
            ("agent", "to", "agent"),
            ("agent", "to", "lane"),
        ]
        
        # L2L edges (typed or single)
        if self.include_l2l:
            if self.l2l_mode == "typed":
                # Typed L2L edges (LaneGCN style)
                edge_types.extend([
                    ("lane", "successor", "lane"),
                    ("lane", "predecessor", "lane"),
                    ("lane", "left_of", "lane"),
                    ("lane", "right_of", "lane"),
                ])
            else:
                # Single L2L edge type
                edge_types.append(("lane", "to", "lane"))
        
        # TL edges
        if self.include_a2tl or (self.include_tl and self.tl_connect_to in ["agents", "both"]):
            edge_types.append(("agent", "to", "tl"))
        
        if self.include_l2tl or (self.include_tl and self.tl_connect_to in ["lanes", "both"]):
            edge_types.append(("tl", "controls", "lane"))
        
        return edge_types
