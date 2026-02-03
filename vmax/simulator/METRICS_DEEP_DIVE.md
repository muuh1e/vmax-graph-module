# VMAX Metrics System: Deep Dive for Graph Building & Search

This document provides a comprehensive analysis of the VMAX metrics system and how you can leverage it for:
1. **Building scenario graphs**
2. **Studying Wayformer attention patterns offline**
3. **Building a metrics-based scenario search engine**

---

## Table of Contents

1. [Metrics Architecture Overview](#metrics-architecture-overview)
2. [When & How Metrics Are Calculated](#when--how-metrics-are-calculated)
3. [Available Metrics Catalog](#available-metrics-catalog)
4. [Data Flow & Dependencies](#data-flow--dependencies)
5. [Using Metrics for Graph Building](#using-metrics-for-graph-building)
6. [Studying Wayformer Attention with Metrics](#studying-wayformer-attention-with-metrics)
7. [Building a Search Engine](#building-a-search-engine)
8. [Practical Code Examples](#practical-code-examples)
9. [Key Insights & Recommendations](#key-insights--recommendations)

---

## Metrics Architecture Overview

### File Structure

```
vmax/simulator/metrics/
├── __init__.py              # Registry & exports
├── collector.py             # Episode aggregation pipeline
├── aggregators.py           # Aggregation functions (nuPlan/VMAX scores)
├── utils.py                 # Shared utilities (geometry, filtering)
├── at_fault_collision.py    # Collision attribution
├── comfort.py               # Ride comfort (acceleration/jerk)
├── driving_direction_compliance.py  # Wrong-way detection
├── on_multiple_lanes.py     # Lane straddling
├── progress_ratio.py        # Route progress vs expert
├── red_light.py             # Traffic light violations
├── route.py                 # Off-route detection
├── speed_limit.py           # Speed limit violations
└── ttc.py                   # Time-to-collision
```

### Core Design Pattern

All metrics inherit from Waymax's `abstract_metric.AbstractMetric`:

```python
class SomeMetric(abstract_metric.AbstractMetric):
    def compute(self, simulator_state: datatypes.SimulatorState) -> MetricResult:
        # Pure function: state -> metric value
        return MetricResult.create_and_validate(value, valid)
```

**Key Properties:**
- **Stateless**: Each computation is independent
- **JAX-based**: GPU-accelerated, differentiable where applicable
- **Timestep-granular**: Computed at each simulation step

---

## When & How Metrics Are Calculated

### Computation Timeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SCENARIO LIFECYCLE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Load TFRecord ─────► Initialize Env ─────► Simulation Loop         │
│       │                     │                     │                  │
│       │                     │              ┌──────┴──────┐           │
│       ▼                     ▼              ▼             ▼           │
│  ScenarioMax           Register        Step t=0      Step t=90      │
│  DataLoader            Metrics         metrics       metrics        │
│                                           │             │            │
│                                           └──────┬──────┘            │
│                                                  ▼                   │
│                                        Collect & Aggregate           │
│                                           (per episode)              │
│                                                  │                   │
│                                                  ▼                   │
│                                         Final Scores                 │
│                                    (nuPlan score, VMAX score)        │
└─────────────────────────────────────────────────────────────────────┘
```

### Two Computation Modes

#### 1. **Online (During Simulation)**
Metrics are computed **per-timestep** as the simulation runs:

```python
# In simulation loop (simplified from env.step())
for t in range(91):
    state = env.step(action)
    for metric_name, metric in metrics_registry.items():
        result = metric.compute(state)  # <-- Per-timestep computation
        metrics_buffer[metric_name].append(result.value)
```

**Location:** `vmax/simulator/wrappers/reward.py` and Waymax's `env.PlanningAgentEnvironment`

#### 2. **Offline (Post-hoc on Logged Data)**
You can compute metrics on **any SimulatorState** without running simulation:

```python
from vmax.simulator import metrics

# Load a scenario
scenario = load_scenario_from_tfrecord(...)

# Create SimulatorState from logged trajectory
state = create_simulator_state_from_scenario(scenario)

# Compute any metric
ttc_metric = metrics.get_metrics("ttc")
result = ttc_metric.compute(state)
```

**This is your key opportunity for offline analysis!**

---

## Available Metrics Catalog

### Safety & Collision Metrics

| Metric | Key Question It Answers | Inputs Used | Output |
|--------|------------------------|-------------|--------|
| `at_fault_collision` | "Is ego responsible for this collision?" | Trajectories, overlaps, VRU types | Count (0-N) |
| `ttc` | "How many seconds until collision?" | Current positions, velocities | Time (0-5 sec) |
| `overlap` (Waymax) | "Is ego overlapping with any object?" | Bounding boxes | Binary (0/1) |

### Navigation & Route Metrics

| Metric | Key Question It Answers | Inputs Used | Output |
|--------|------------------------|-------------|--------|
| `progress_ratio_nuplan` | "How much progress vs expert trajectory?" | Ego vs logged trajectory | Ratio (0-1+) |
| `sdc_off_route` | "How far from planned route?" | Position, SDC paths | Distance (m) |
| `driving_direction_compliance` | "Is ego driving into oncoming traffic?" | Position, heading, lane directions | Distance (m) |

### Traffic Rule Compliance

| Metric | Key Question It Answers | Inputs Used | Output |
|--------|------------------------|-------------|--------|
| `run_red_light` | "Did ego cross a red light?" | Position change, lane IDs, traffic states | Binary (0/1) |
| `speed_limit` | "How much over the speed limit?" | Ego speed, lane type | Excess (m/s) |

### Comfort & Driving Quality

| Metric | Key Question It Answers | Inputs Used | Output |
|--------|------------------------|-------------|--------|
| `comfort` | "Are accelerations within human comfort?" | Last 10 timesteps trajectory | Binary or [0-1] |
| `on_multiple_lanes` | "Is ego straddling multiple lanes?" | Corner positions, lane centers | Distance (m) |

---

## Data Flow & Dependencies

### Input: SimulatorState

Every metric receives a `SimulatorState` object containing:

```python
SimulatorState:
├── sim_trajectory        # Simulated trajectory for ALL objects
│   ├── x, y, z           # Position
│   ├── vel_x, vel_y      # Velocity
│   ├── yaw               # Heading
│   ├── length, width     # Dimensions
│   ├── bbox_corners      # 4 corner points (8 values)
│   └── valid             # Validity mask
│
├── log_trajectory        # Expert (ground truth) trajectory
│
├── object_metadata
│   ├── is_sdc            # Boolean mask for ego vehicle
│   └── object_types      # 1=vehicle, 2=pedestrian, 3=cyclist
│
├── roadgraph_points      # Road network
│   ├── xyz               # Point positions
│   ├── types             # 1=freeway, 2=surface_street
│   ├── dir_xy            # Direction vectors
│   ├── ids               # Lane IDs
│   └── valid             # Validity mask
│
├── sdc_paths             # Planned routes (10 paths x 300 points)
│
├── log_traffic_light     # Traffic light states over time
│
└── timestep              # Current timestep index (0-90)
```

### Output: MetricResult

```python
MetricResult:
├── value    # Scalar or array (the metric value)
└── valid    # Boolean validity flag
```

---

## Using Metrics for Graph Building

### Why Metrics Are Valuable for Graphs

Metrics encode **semantic relationships** that are perfect for graph edges:

| Metric | Graph Representation |
|--------|---------------------|
| `ttc` | Edge weight between ego and potential collision partners |
| `at_fault_collision` | Directed edge "ego → caused collision with → agent" |
| `on_multiple_lanes` | Edge "ego → occupies → lane1, lane2" |
| `speed_limit` | Node attribute on ego "speeding_by=X m/s" |
| `driving_direction_compliance` | Edge "ego → wrong_way_on → lane" |

### Architecture: Metrics as Graph Features

```
┌──────────────────────────────────────────────────────────────────┐
│                    SCENARIO GRAPH STRUCTURE                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│   NODES                          EDGES                            │
│   ─────                          ─────                            │
│   • Ego vehicle                  • ttc_edge(ego, agent)          │
│     - progress_ratio             • collision_edge(ego, agent)    │
│     - comfort_score              • lane_occupancy(ego, lane)     │
│     - speed_violation            • route_edge(ego, path)         │
│     - is_off_route                                                │
│                                                                   │
│   • Other agents                 • spatial_proximity(agent_i,    │
│     - relative_ttc                                 agent_j)      │
│     - is_vru                                                      │
│                                                                   │
│   • Lanes                        • traffic_light_state(lane,     │
│     - speed_limit                                   light)       │
│     - direction                                                   │
│                                                                   │
│   • Traffic lights                                                │
│     - current_state                                               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Extracting Metric Features for Graphs

```python
import jax.numpy as jnp
from vmax.simulator import metrics

def compute_scenario_graph_features(simulator_state):
    """Extract metric-based features for graph construction."""

    features = {}

    # 1. TTC - creates edges to all agents with collision potential
    ttc_metric = metrics.get_metrics("ttc")
    ttc_result = ttc_metric.compute(simulator_state)
    features['ego_min_ttc'] = ttc_result.value

    # 2. Access internal TTC per-agent (modify metric or extract from state)
    # This gives you ttc[agent_idx] for each agent

    # 3. At-fault collision potential
    collision_metric = metrics.get_metrics("at_fault_collision")
    collision_result = collision_metric.compute(simulator_state)
    features['at_fault_count'] = collision_result.value

    # 4. Route progress (useful for scenario complexity)
    progress_metric = metrics.get_metrics("progress_ratio_nuplan")
    progress_result = progress_metric.compute(simulator_state)
    features['progress_ratio'] = progress_result.value

    # 5. Driving direction compliance (wrong-way indicator)
    direction_metric = metrics.get_metrics("driving_direction_compliance")
    direction_result = direction_metric.compute(simulator_state)
    features['wrong_way_distance'] = direction_result.value

    # 6. Speed compliance
    speed_metric = metrics.get_metrics("speed_limit")
    speed_result = speed_metric.compute(simulator_state)
    features['overspeed'] = speed_result.value

    # 7. Comfort (driving quality)
    comfort_metric = metrics.get_metrics("comfort")
    comfort_result = comfort_metric.compute(simulator_state)
    features['comfort_score'] = comfort_result.value

    return features
```

### Direct Use vs. Reusing Logic

#### Option A: Use Metrics Directly (Recommended for Most Cases)

```python
# Pro: Maintains consistency with simulation evaluation
# Con: Designed for SimulatorState, not raw scenarios

from vmax.simulator import metrics
metric = metrics.get_metrics("ttc")
result = metric.compute(simulator_state)
```

#### Option B: Extract & Reuse Metric Logic

The utility functions in `metrics/utils.py` are **highly reusable**:

```python
from vmax.simulator.metrics.utils import (
    get_distance_to_lane_centers,      # For lane proximity
    get_closest_lane_center_idx,       # For lane assignment
    is_agent_ahead,                    # For spatial reasoning
    is_agent_behind,                   # For rear collision analysis
    get_agent_relative_angle,          # For heading analysis
)
```

These functions work with raw position/heading data and can be used in your graph builder without needing a full SimulatorState.

---

## Studying Wayformer Attention with Metrics

### Key Insight

Metrics quantify **what matters** in a scenario. By correlating Wayformer attention patterns with metric values, you can answer:

- "Does the model attend more to agents with low TTC?"
- "When ego is speeding, what does the model attend to?"
- "Does attention to VRUs change when they're in the collision zone?"

### Methodology

```
┌─────────────────────────────────────────────────────────────────────┐
│              WAYFORMER ATTENTION ANALYSIS PIPELINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Pre-compute Metrics for Dataset                                  │
│     ─────────────────────────────────                                │
│     For each scenario:                                               │
│       - Compute all 9 metrics at each timestep                       │
│       - Store as scenario metadata                                   │
│                                                                      │
│  2. Extract Wayformer Attention                                      │
│     ────────────────────────────                                     │
│     For each scenario:                                               │
│       - Run Wayformer forward pass                                   │
│       - Extract attention weights from each head/layer               │
│       - attention_matrix[query_agent, key_agent]                     │
│                                                                      │
│  3. Correlation Analysis                                             │
│     ────────────────────                                             │
│     For each (metric, attention_head) pair:                          │
│       - Compute correlation across scenarios                         │
│       - Statistical significance testing                             │
│                                                                      │
│  4. Visualization                                                    │
│     ─────────────                                                    │
│     - Attention heatmaps colored by metric values                    │
│     - Scatter plots: metric value vs attention weight                │
│     - Per-scenario: attention overlay on bird's eye view             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Example: TTC-Attention Correlation

```python
import numpy as np
from vmax.simulator import metrics

def compute_ttc_attention_correlation(scenarios, wayformer_model):
    """Correlate TTC with Wayformer's attention to agents."""

    ttc_metric = metrics.get_metrics("ttc")
    correlations = []

    for scenario in scenarios:
        state = scenario_to_simulator_state(scenario)

        # 1. Compute per-agent TTC
        # Note: The metric returns min TTC, but internally computes per-agent
        # You may need to modify the metric to return full ttc array
        ttc_result = ttc_metric.compute(state)

        # 2. Get Wayformer attention (you'll need to expose this)
        attention_weights = wayformer_model.get_attention_weights(scenario)
        # Shape: [num_heads, num_agents, num_agents]

        # 3. For each agent, correlate their TTC with attention received
        ego_attention_to_others = attention_weights[:, ego_idx, :]  # What ego attends to

        # Higher attention to low-TTC agents?
        # Invert TTC so low TTC = high value
        ttc_importance = 1.0 / (ttc_per_agent + 1e-6)

        corr = np.corrcoef(ttc_importance, ego_attention_to_others.mean(axis=0))[0, 1]
        correlations.append(corr)

    return np.mean(correlations), np.std(correlations)
```

### Metrics Most Relevant for Attention Analysis

| Metric | Attention Question |
|--------|-------------------|
| `ttc` | Does model prioritize collision-risk agents? |
| `at_fault_collision` | Does model differentiate VRUs from vehicles? |
| `speed_limit` | Does attention change when ego is speeding? |
| `run_red_light` | Does model attend to traffic lights when approaching? |
| `driving_direction_compliance` | Does model notice oncoming traffic? |

---

## Building a Search Engine

### Architecture: Graph + Metrics Search Index

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SCENARIO SEARCH ENGINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│   │ TFRecords   │────►│  Metrics    │────►│  Search     │           │
│   │ (Waymo)     │     │  Extractor  │     │  Index      │           │
│   └─────────────┘     └─────────────┘     └─────────────┘           │
│                              │                   │                   │
│                              ▼                   ▼                   │
│                       ┌─────────────┐     ┌─────────────┐           │
│                       │  Graph      │────►│  Query      │           │
│                       │  Builder    │     │  Engine     │           │
│                       └─────────────┘     └─────────────┘           │
│                                                 │                    │
│                                                 ▼                    │
│                                          ┌─────────────┐            │
│                                          │  Results    │            │
│                                          │  Ranker     │            │
│                                          └─────────────┘            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Search Index Schema

```python
# Per-scenario index entry
scenario_index = {
    "scenario_id": str,
    "source": "waymo" | "nuplan",

    # Aggregate metrics (computed once)
    "metrics": {
        "min_ttc": float,              # Minimum TTC across scenario
        "has_collision": bool,         # Any collision occurred
        "has_at_fault": bool,          # Ego at fault for collision
        "progress_ratio": float,       # Route completion
        "max_overspeed_kmh": float,    # Speed violation magnitude
        "has_red_light_run": bool,     # Red light violation
        "comfort_score": float,        # Driving comfort
        "wrong_way_distance": float,   # Oncoming traffic distance
    },

    # Graph features
    "graph": {
        "num_agents": int,
        "num_vrus": int,               # Pedestrians + cyclists
        "num_lanes": int,
        "has_intersection": bool,
        "has_traffic_lights": bool,
        "scenario_complexity": float,  # Derived score
    },

    # Temporal features
    "temporal": {
        "scenario_length_sec": float,
        "timestep_collision": int | null,
        "timestep_min_ttc": int,
    }
}
```

### Query Language Examples

```python
# Natural language queries -> structured filters

"scenarios with near-collisions"
    -> { "min_ttc": {"$lt": 2.0}, "has_collision": False }

"red light violations on surface streets"
    -> { "has_red_light_run": True, "lane_type": "surface_street" }

"high-speed scenarios with pedestrians"
    -> { "max_overspeed_kmh": {"$gt": 10}, "num_vrus": {"$gt": 0} }

"complex intersections with multiple vehicles"
    -> { "has_intersection": True, "num_agents": {"$gt": 5} }

"scenarios where expert was uncomfortable"
    -> { "comfort_score": {"$lt": 0.5} }
```

### Implementation: Metric-Based Indexer

```python
from vmax.simulator import metrics
import json

class ScenarioIndexer:
    """Build a searchable index of Waymo scenarios using metrics."""

    def __init__(self):
        self.metrics_registry = {
            "ttc": metrics.get_metrics("ttc"),
            "at_fault_collision": metrics.get_metrics("at_fault_collision"),
            "progress_ratio_nuplan": metrics.get_metrics("progress_ratio_nuplan"),
            "speed_limit": metrics.get_metrics("speed_limit"),
            "run_red_light": metrics.get_metrics("run_red_light"),
            "comfort": metrics.get_metrics("comfort"),
            "driving_direction_compliance": metrics.get_metrics("driving_direction_compliance"),
            "on_multiple_lanes": metrics.get_metrics("on_multiple_lanes"),
        }
        self.index = []

    def index_scenario(self, scenario_id: str, states: list):
        """Index a single scenario by computing metrics at all timesteps."""

        timestep_metrics = {name: [] for name in self.metrics_registry}

        for state in states:
            for name, metric in self.metrics_registry.items():
                result = metric.compute(state)
                timestep_metrics[name].append(float(result.value))

        # Aggregate using collector logic
        entry = {
            "scenario_id": scenario_id,
            "metrics": {
                "min_ttc": min(timestep_metrics["ttc"]),
                "has_collision": max(timestep_metrics["at_fault_collision"]) > 0,
                "has_at_fault": max(timestep_metrics["at_fault_collision"]) > 0,
                "progress_ratio": timestep_metrics["progress_ratio_nuplan"][-1],
                "max_overspeed_kmh": 3.6 * max(timestep_metrics["speed_limit"]),
                "has_red_light_run": max(timestep_metrics["run_red_light"]) > 0,
                "comfort_score": min(timestep_metrics["comfort"]),
                "wrong_way_distance": sum(timestep_metrics["driving_direction_compliance"]),
            },
            "temporal": {
                "scenario_length_sec": len(states) * 0.1,
                "timestep_min_ttc": timestep_metrics["ttc"].index(min(timestep_metrics["ttc"])),
            }
        }

        self.index.append(entry)
        return entry

    def search(self, query: dict):
        """Search index with MongoDB-style queries."""
        results = []
        for entry in self.index:
            if self._matches(entry, query):
                results.append(entry)
        return results

    def _matches(self, entry: dict, query: dict) -> bool:
        # Simple query matching (expand as needed)
        for key, condition in query.items():
            value = entry["metrics"].get(key) or entry["temporal"].get(key)
            if isinstance(condition, dict):
                if "$lt" in condition and value >= condition["$lt"]:
                    return False
                if "$gt" in condition and value <= condition["$gt"]:
                    return False
            elif value != condition:
                return False
        return True
```

---

## Practical Code Examples

### Example 1: Batch Metrics Computation

```python
"""Compute metrics for all scenarios in a dataset."""

from waymax import dataloader
from vmax.simulator import metrics, sim_factory
import pandas as pd

def batch_compute_metrics(tfrecord_path: str, max_scenarios: int = 1000):
    """Compute metrics for scenarios and return as DataFrame."""

    # Load scenarios
    data_config = dataloader.WaymaxDataConfig(
        path=tfrecord_path,
        max_num_objects=64,
    )
    dataset = dataloader.create_dataloader(data_config)

    results = []

    for i, scenario in enumerate(dataset):
        if i >= max_scenarios:
            break

        # Create environment to get SimulatorState
        env = sim_factory.make_env(scenario)
        state, _ = env.reset()

        # Compute metrics at each timestep
        scenario_metrics = {
            "scenario_id": f"scenario_{i}",
            "ttc_values": [],
            "collision_values": [],
            "comfort_values": [],
        }

        for t in range(91):  # Waymo scenarios are 9.1 seconds
            state, _, done, _, _ = env.step(None)  # Replay logged trajectory

            scenario_metrics["ttc_values"].append(
                float(metrics.get_metrics("ttc").compute(state).value)
            )
            scenario_metrics["collision_values"].append(
                float(metrics.get_metrics("at_fault_collision").compute(state).value)
            )
            scenario_metrics["comfort_values"].append(
                float(metrics.get_metrics("comfort").compute(state).value)
            )

            if done:
                break

        # Aggregate
        scenario_metrics["min_ttc"] = min(scenario_metrics["ttc_values"])
        scenario_metrics["has_collision"] = max(scenario_metrics["collision_values"]) > 0
        scenario_metrics["avg_comfort"] = sum(scenario_metrics["comfort_values"]) / len(scenario_metrics["comfort_values"])

        results.append(scenario_metrics)

    return pd.DataFrame(results)
```

### Example 2: Extracting Per-Agent TTC for Graph Edges

```python
"""Modified TTC computation that returns per-agent values."""

import jax.numpy as jnp
from vmax.simulator import constants, operations
from vmax.simulator.metrics.utils import is_agent_ahead
from waymax import datatypes
from waymax.utils import geometry

def compute_per_agent_ttc(simulator_state, time_horizon=5.0):
    """Compute TTC from ego to each agent (for graph edge weights)."""

    dt = constants.TIME_DELTA

    current_traj = datatypes.dynamic_slice(
        simulator_state.sim_trajectory,
        simulator_state.timestep,
        1,
        -1,
    )

    sdc_index = operations.get_index(simulator_state.object_metadata.is_sdc)

    # Get positions and velocities
    traj_5dof = current_traj.stack_fields(["x", "y", "length", "width", "yaw"])
    velocities = current_traj.vel_xy
    timesteps = jnp.arange(0, time_horizon, dt)

    initial_positions = traj_5dof[:, :, :2]
    other_fields = traj_5dof[:, :, 2:]

    # Predict future positions (constant velocity model)
    future_positions = initial_positions + velocities * timesteps[None, :, None]

    other_fields = jnp.broadcast_to(
        other_fields,
        (future_positions.shape[0], future_positions.shape[1], other_fields.shape[-1]),
    )

    future_traj_5dof = jnp.concatenate([future_positions, other_fields], axis=-1)

    # Check collisions at each timestep
    import jax
    check_overlap = jax.vmap(geometry.has_overlap, (0, None), -1)
    check_overlap = jax.vmap(check_overlap, (None, 0), -1)
    check_overlap = jax.vmap(check_overlap, 1)

    collision_matrix = check_overlap(future_traj_5dof, future_traj_5dof)

    # Get ego's collisions with each agent over time
    sdc_collisions = collision_matrix[:, sdc_index].T  # [num_agents, num_timesteps]

    # Find first collision timestep for each agent
    is_gonna_collide = jnp.any(sdc_collisions, axis=1)
    first_collision = jnp.argmax(sdc_collisions, axis=1)

    per_agent_ttc = dt * first_collision.astype(jnp.float32)
    per_agent_ttc = jnp.where(is_gonna_collide, per_agent_ttc, time_horizon)

    # Filter to only ahead agents
    agents_xy = current_traj.xy.squeeze()
    ego_yaw = current_traj.yaw[sdc_index].squeeze()
    ego_xy = agents_xy[sdc_index]

    is_ahead = is_agent_ahead(ego_xy, ego_yaw, agents_xy)
    per_agent_ttc = jnp.where(is_ahead, per_agent_ttc, time_horizon)

    return per_agent_ttc, current_traj.valid.squeeze()
```

### Example 3: Scenario Complexity Score

```python
"""Compute a complexity score combining multiple metrics."""

from vmax.simulator import metrics
import numpy as np

def compute_scenario_complexity(states: list) -> dict:
    """
    Compute scenario complexity based on metrics.
    Higher score = more challenging/interesting scenario.
    """

    # Collect per-timestep metrics
    ttc_values = []
    collision_values = []
    speed_violations = []
    lane_violations = []

    for state in states:
        ttc_values.append(float(metrics.get_metrics("ttc").compute(state).value))
        collision_values.append(float(metrics.get_metrics("at_fault_collision").compute(state).value))
        speed_violations.append(float(metrics.get_metrics("speed_limit").compute(state).value))
        lane_violations.append(float(metrics.get_metrics("on_multiple_lanes").compute(state).value))

    # Complexity factors
    factors = {
        # Near-miss intensity (inverse of min TTC, capped)
        "collision_risk": min(1.0, 1.0 / (min(ttc_values) + 0.1)),

        # Actual collision
        "has_collision": float(max(collision_values) > 0),

        # Traffic rule challenges
        "speed_challenge": min(1.0, max(speed_violations) / 10.0),  # Normalize by 10 m/s

        # Lane navigation challenges
        "lane_challenge": min(1.0, sum(lane_violations) / 50.0),  # Normalize by 50m total

        # Object count at any timestep
        "agent_density": len([s for s in states if s.object_metadata.is_sdc.sum() > 0]) / len(states),
    }

    # Weighted complexity score
    weights = {
        "collision_risk": 3.0,
        "has_collision": 2.0,
        "speed_challenge": 1.0,
        "lane_challenge": 1.0,
        "agent_density": 0.5,
    }

    complexity_score = sum(factors[k] * weights[k] for k in factors) / sum(weights.values())

    return {
        "complexity_score": complexity_score,
        "factors": factors,
    }
```

---

## Key Insights & Recommendations

### What to Use Directly

| Component | Use Directly? | Why |
|-----------|---------------|-----|
| `MetricResult.value` | Yes | Clean API, consistent with evaluation |
| `metrics/utils.py` functions | Yes | Pure geometry, works anywhere |
| `aggregators.py` functions | Yes | Standardized aggregation logic |
| `collector.collect()` | Maybe | Good for episode aggregation, but tied to simulation |

### What to Reimplement

| Component | Reimplement? | Why |
|-----------|--------------|-----|
| Per-agent TTC | Yes | Current metric only returns min; you need full array |
| Scenario-level features | Yes | Metrics are timestep-level; you need aggregates |
| Graph connectivity | Yes | Metrics don't model relationships, just values |

### Recommended Architecture for Your Goals

```
                      ┌─────────────────────────────────┐
                      │      Your Pipeline              │
                      └─────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │  Graph Builder  │    │ Attention Study │    │ Search Engine   │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                      │                      │
              ▼                      ▼                      ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ Per-agent TTC   │    │ Pre-computed    │    │ Metrics Index   │
    │ Collision edges │    │ metric features │    │ + Graph Index   │
    │ Lane occupancy  │    │ + attention     │    │ + Full-text     │
    └─────────────────┘    └─────────────────┘    └─────────────────┘
              │                      │                      │
              └──────────────────────┼──────────────────────┘
                                     ▼
                      ┌─────────────────────────────────┐
                      │  vmax/simulator/metrics/        │
                      │  (Reuse compute logic)          │
                      └─────────────────────────────────┘
```

### Summary: Your Three Use Cases

1. **Graph Building**
   - Use `utils.py` functions for geometry
   - Modify TTC metric to return per-agent values for edge weights
   - Use metric values as node/edge attributes

2. **Wayformer Attention Study**
   - Pre-compute all metrics for your dataset
   - Store as scenario-level features
   - Correlate with attention weights post-hoc

3. **Search Engine**
   - Build index using aggregated metrics
   - Store both raw timestep values (for temporal queries) and aggregates
   - Combine with graph features for rich queries

---

## Appendix: Metric Thresholds Reference

| Metric | Threshold | Meaning |
|--------|-----------|---------|
| TTC | < 0.95s | Dangerous |
| Progress | > 0.2 | Making progress |
| Speed | 70 mph (freeway), 45 mph (surface) | Speed limits |
| Comfort - Lateral Accel | ≤ 2.0 m/s² | Comfortable |
| Comfort - Long Accel | -4.05 to 2.40 m/s² | Comfortable |
| Comfort - Yaw Rate | ≤ 0.95 rad/s | Comfortable |
| Lane Width | 3.7m | Standard lane |
| Lane Margin | 0.2m | Acceptable offset |

---

*Document generated for VMAX metrics analysis - January 2026*
