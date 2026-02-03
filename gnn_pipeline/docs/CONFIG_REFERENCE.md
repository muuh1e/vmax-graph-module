# Configuration Reference Guide

Complete reference for all configuration options in the GNN pipeline. Use this guide to understand what each config does and how to combine them for optimal performance.

## Table of Contents

- [Quick Start](#quick-start)
- [GraphConfig Reference](#graphconfig-reference)
- [ModelConfig Reference](#modelconfig-reference)
- [Graph Presets](#graph-presets)
- [Combining Configurations](#combining-configurations)
- [Experimentation Guide](#experimentation-guide)

---

## Quick Start

**Current Best Performance**: Baseline config achieves the best ADE so far.

```bash
# Baseline (best so far)
python gnn_pipeline/train/run.py --graph_preset baseline --epochs 50

# To experiment with different configs:
python gnn_pipeline/train/run.py --graph_preset typed_l2l --model typed_gnn
python gnn_pipeline/train/run.py --graph_preset optimized --hidden 256
```

---

## GraphConfig Reference

The `GraphConfig` controls what nodes and edges are included in the heterogeneous graph, and how they're constructed.

### Node/Edge Inclusion Flags

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `include_tl` | `bool` | `False` | Include traffic light nodes in the graph |
| `full_a2a` | `bool` | `False` | Full agent-to-agent connectivity (all pairs) vs ego-only |
| `include_l2l` | `bool` | `False` | Include lane-to-lane edges |
| `include_a2tl` | `bool` | `False` | Include agent-to-traffic-light edges |
| `include_l2tl` | `bool` | `False` | Include lane-to-traffic-light edges |

### Basic Edge Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `a2a_max_dist` | `float` | `50.0` | Maximum distance for A2A edges (meters) |
| `a2l_k` | `int` | `3` | Number of nearest lanes per agent |
| `l2l_max_dist` | `float` | `5.0` | Maximum distance for L2L edges (meters) |
| `k_past` | `int` | `10` | Number of past timesteps to include in features |

### Advanced A2A Filtering (NEW)

Control how agent-to-agent edges are constructed:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `a2a_mode` | `str` | `"ego_only"` | `"ego_only"`, `"k_nearest"`, `"same_lane"`, `"directional"` | A2A edge construction mode |
| `a2a_k_nearest` | `int` | `5` | - | Number of nearest agents per agent (for `k_nearest` mode) |
| `a2a_same_lane_only` | `bool` | `False` | - | Only connect agents sharing a lane |
| `a2a_directional` | `bool` | `False` | - | Only connect agents ahead (in field of view) |
| `a2a_fov_angle` | `float` | `120.0` | - | Field of view angle for directional mode (degrees) |

**A2A Modes Explained:**
- **`ego_only`**: Only ego agent connects to others (baseline, most sparse)
- **`k_nearest`**: Each agent connects to k nearest neighbors (more interaction)
- **`same_lane`**: Only connect agents on the same lane (lane-following)
- **`directional`**: Only connect to agents in front (forward attention)

### Advanced L2L Typing (NEW)

Control how lane-to-lane edges are structured:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `l2l_mode` | `str` | `"all"` | `"all"`, `"typed"`, `"successor_only"` | L2L edge construction mode |
| `l2l_typed_convs` | `bool` | `False` | - | Use separate convolutions per L2L edge type |
| `l2l_max_successor_dist` | `float` | `3.0` | - | Max gap for successor edges (meters) |
| `l2l_max_neighbor_dist` | `float` | `5.0` | - | Max lateral distance for neighbor edges (meters) |

**L2L Modes Explained:**
- **`all`**: Single edge type for all lane-lane connections
- **`typed`**: LaneGCN-style with 4 types: successor, predecessor, left_of, right_of
- **`successor_only`**: Only forward-direction lane connections

### Advanced TL Filtering (NEW)

Control which traffic lights are included and how they connect:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `tl_filter_mode` | `str` | `"all"` | `"all"`, `"relevant"`, `"controlling"` | TL filtering mode |
| `tl_connect_to` | `str` | `"both"` | `"agents"`, `"lanes"`, `"both"` | What TLs connect to |
| `tl_max_relevant` | `int` | `4` | - | Max number of TLs to keep per scenario |
| `tl_max_distance` | `float` | `100.0` | - | Max distance from ego for TLs (meters) |
| `tl_ahead_only` | `bool` | `True` | - | Only TLs ahead of ego |

**TL Filter Modes Explained:**
- **`all`**: Include all traffic lights in the scene
- **`relevant`**: Only TLs near ego's path (within distance/ahead)
- **`controlling`**: Only TLs that control lanes ego might use

### Advanced Lane Filtering (NEW)

Control which lanes are included:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `lane_filter_mode` | `str` | `"all"` | `"all"`, `"drivable"`, `"ego_relevant"`, `"ego_path"` | Lane filtering mode |
| `lane_max_count` | `int` | `80` | - | Max number of lanes to keep |
| `lane_max_distance` | `float` | `50.0` | - | Max distance from ego for lanes (meters) |
| `lane_drivable_only` | `bool` | `False` | - | Exclude non-drivable lanes |

**Lane Filter Modes Explained:**
- **`all`**: Include all lanes in the scene
- **`drivable`**: Only drivable lanes (exclude sidewalks, bike lanes)
- **`ego_relevant`**: Lanes within distance of ego
- **`ego_path`**: Lanes on or connected to ego's likely path

---

## ModelConfig Reference

The `ModelConfig` controls the neural network architecture.

### Architecture Parameters

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `model_type` | `str` | `"simple_gnn"` | `"simple_gnn"`, `"typed_gnn"`, `"hierarchical_gnn"` | Model architecture type |
| `hidden_channels` | `int` | `128` | - | Hidden dimension size |
| `num_layers` | `int` | `3` | - | Number of GNN layers |
| `dropout` | `float` | `0.1` | - | Dropout rate |
| `conv_type` | `str` | `"sage"` | `"sage"`, `"gat"` | Convolution type |
| `use_edge_attr` | `bool` | `False` | - | Whether to use edge attributes |
| `num_future_steps` | `int` | `80` | - | Number of future timesteps to predict |

### Model Type Descriptions

#### `simple_gnn` (SimpleHeteroGNN)
- **Description**: Basic heterogeneous GNN with uniform message passing
- **Best for**: Baseline experiments, quick iterations
- **Characteristics**: Single convolution type for all edge types
- **When to use**: Starting point for any experiment

#### `typed_gnn` (TypedHeteroGNN) - NEW
- **Description**: LaneGCN-inspired with semantic edge types
- **Best for**: Experiments with typed L2L edges
- **Characteristics**:
  - Separate convolutions per L2L relationship (successor, predecessor, left_of, right_of)
  - Typed A2A convolutions for different interaction types
  - Agent-Lane fusion with bidirectional cross-attention
- **When to use**: When using `l2l_mode="typed"` preset

#### `hierarchical_gnn` (HierarchicalGNN) - NEW
- **Description**: HiVT-inspired hierarchical encoder
- **Best for**: State-of-the-art performance experiments
- **Characteristics**:
  - Temporal transformers for agent history
  - Polyline encoders for lanes
  - Local k-NN attention + global sparse attention
  - Multi-modal decoder (K possible futures)
- **When to use**: Pushing for best performance (more compute intensive)

### Edge Type Usage Flags

Control which edge types the model should use:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_a2a` | `bool` | `True` | Use agent-to-agent edges |
| `use_a2l` | `bool` | `True` | Use agent-to-lane edges |
| `use_l2a` | `bool` | `True` | Use lane-to-agent (reverse) edges |
| `use_l2l` | `bool` | `False` | Use lane-to-lane edges |
| `use_a2tl` | `bool` | `False` | Use agent-to-traffic-light edges |
| `use_l2tl` | `bool` | `False` | Use lane-to-traffic-light edges |

**Note**: These are automatically synced with GraphConfig via `sync_with_graph_config()`.

---

## Graph Presets

Pre-configured combinations for common use cases.

### Basic Presets

#### `baseline` (Default)
```python
GraphConfig.baseline()
```
- **Nodes**: Agents + Lanes
- **Edges**: Ego→Agents (A2A), Agents↔Lanes (A2L)
- **Use for**: Starting point, fastest to process
- **Performance**: **Best ADE so far** ✓

#### `with_l2l`
```python
GraphConfig.with_l2l()
```
- **Baseline +** Lane-to-lane edges
- **Use for**: Testing lane graph reasoning
- **Trade-off**: More edges, slower processing

#### `with_tl`
```python
GraphConfig.with_tl()
```
- **Baseline +** Traffic light nodes + edges
- **Use for**: Scenarios with traffic signals
- **Trade-off**: More nodes, may help at intersections

#### `with_full_a2a`
```python
GraphConfig.with_full_a2a()
```
- **Baseline +** Full agent-agent connectivity (all pairs)
- **Use for**: Dense interaction modeling
- **Trade-off**: O(N²) edges, much slower

#### `full`
```python
GraphConfig.full()
```
- **Everything enabled**: L2L, TL, full A2A
- **Use for**: Maximum information (probably overkill)
- **Trade-off**: Very slow, may overfit

### Advanced Presets (NEW)

#### `smart_a2a_k5`
```python
GraphConfig.smart_a2a_k5()
```
- **Baseline +** k-nearest A2A (k=5)
- **Use for**: More agent interaction without full O(N²)
- **Trade-off**: More balanced than `with_full_a2a`

#### `typed_l2l`
```python
GraphConfig.typed_l2l()
```
- **Baseline +** Typed L2L edges (LaneGCN-style)
- **Edge types**: successor, predecessor, left_of, right_of
- **Use for**: Semantic lane relationships
- **Best with**: `--model typed_gnn`

#### `filtered_tl`
```python
GraphConfig.filtered_tl()
```
- **Baseline +** Filtered traffic lights
- **Only controlling TLs**: Via lanes only, max 2 relevant
- **Use for**: Efficient TL modeling without noise
- **Trade-off**: Much cleaner than `with_tl`

#### `optimized`
```python
GraphConfig.optimized()
```
- **Smart combination** of all advanced features:
  - k-nearest A2A (k=5)
  - Typed L2L edges
  - Filtered TLs (controlling only)
  - Filtered lanes (ego-relevant, max 40)
- **Use for**: Exploring best-case scenario
- **Trade-off**: Complex, needs careful tuning

---

## Combining Configurations

### Using Presets with CLI

```bash
# Start with a preset
python gnn_pipeline/train/run.py --graph_preset typed_l2l

# Override specific parameters
python gnn_pipeline/train/run.py --graph_preset baseline --include_l2l

# Combine multiple overrides
python gnn_pipeline/train/run.py --graph_preset baseline --include_l2l --a2a_mode k_nearest --a2a_k 10
```

### Common Combinations

#### Experiment 1: Typed Lane Graph
```bash
python gnn_pipeline/train/run.py \
  --graph_preset typed_l2l \
  --model typed_gnn \
  --hidden 256 \
  --num_layers 4
```

#### Experiment 2: k-NN Agent Interaction
```bash
python gnn_pipeline/train/run.py \
  --graph_preset smart_a2a_k5 \
  --a2a_k 10 \
  --hidden 256
```

#### Experiment 3: Filtered Scene
```bash
python gnn_pipeline/train/run.py \
  --graph_preset baseline \
  --lane_mode ego_relevant \
  --lane_max 40 \
  --a2a_mode k_nearest \
  --a2a_k 5
```

#### Experiment 4: Hierarchical Model
```bash
python gnn_pipeline/train/run.py \
  --graph_preset baseline \
  --model hierarchical_gnn \
  --hidden 128 \
  --num_layers 3
```

### Programmatic Configuration

```python
from gnn_pipeline import GraphConfig, ModelConfig

# Start with preset and customize
graph_config = GraphConfig.baseline()
graph_config.a2a_mode = "k_nearest"
graph_config.a2a_k_nearest = 10
graph_config.include_l2l = True
graph_config.l2l_mode = "typed"

# Create model config
model_config = ModelConfig(
    model_type="typed_gnn",
    hidden_channels=256,
    num_layers=4,
    conv_type="gat",
)

# Sync edge types
model_config.sync_with_graph_config(graph_config)
```

---

## Experimentation Guide

### Goal: Find Best Config + Model Combination

**Baseline Performance (to beat):**
- Config: `baseline`
- Model: `simple_gnn`
- Metric: Best ADE

### Recommended Experiment Strategy

#### Phase 1: Model Architecture
Test different models with baseline config:

```bash
# 1. Baseline
python gnn_pipeline/train/run.py --graph_preset baseline --model simple_gnn

# 2. Typed GNN
python gnn_pipeline/train/run.py --graph_preset baseline --model typed_gnn

# 3. Hierarchical GNN
python gnn_pipeline/train/run.py --graph_preset baseline --model hierarchical_gnn
```

#### Phase 2: Graph Connectivity
Test different graph structures with best model from Phase 1:

```bash
# Assuming simple_gnn was best:

# 1. Baseline (already tested)
# 2. k-NN A2A
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --model simple_gnn

# 3. Typed L2L
python gnn_pipeline/train/run.py --graph_preset typed_l2l --model simple_gnn

# 4. Filtered TL
python gnn_pipeline/train/run.py --graph_preset filtered_tl --model simple_gnn

# 5. Combined
python gnn_pipeline/train/run.py --graph_preset optimized --model simple_gnn
```

#### Phase 3: Hyperparameters
Test different sizes with best config from Phase 2:

```bash
# Assuming smart_a2a_k5 was best:

# 1. Wider
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --hidden 256

# 2. Deeper
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --num_layers 5

# 3. Both
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --hidden 256 --num_layers 5

# 4. GAT instead of SAGE
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --conv_type gat
```

#### Phase 4: Fine-tuning
Tune specific parameters of best combination:

```bash
# Example: Tune k value for k-NN A2A
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --a2a_k 3
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --a2a_k 7
python gnn_pipeline/train/run.py --graph_preset smart_a2a_k5 --a2a_k 10

# Example: Tune lane count for filtered lanes
python gnn_pipeline/train/run.py --graph_preset optimized --lane_max 20
python gnn_pipeline/train/run.py --graph_preset optimized --lane_max 60
```

### Tracking Experiments

Results are saved to `results/` and `checkpoints/` directories:

```bash
# Compare metrics
ls results/
# graph_config.json  # Graph configuration used
# model_config.json  # Model configuration used

# Best models
ls checkpoints/
# best_model.pt      # Best checkpoint by val ADE

# Training curves
ls plots/
# {model_name}_training_curves.png
```

### What to Look For

1. **ADE (Average Displacement Error)**: Primary metric, lower is better
2. **FDE (Final Displacement Error)**: Secondary metric
3. **Training time**: Balance accuracy vs speed
4. **Overfitting**: Val ADE vs Train ADE gap
5. **Graph size**: Number of nodes/edges (check logs)

### Common Pitfalls

❌ **Don't**: Enable everything at once (`--graph_preset full`)
- Too many edges, slow training, may overfit
- Start simple, add complexity

❌ **Don't**: Use typed L2L without typed model
- `--graph_preset typed_l2l` needs `--model typed_gnn` to shine
- Simple GNN will treat all edge types the same

✅ **Do**: Ablate one change at a time
- Change config OR model OR hyperparams
- Easier to understand what helps

✅ **Do**: Check graph statistics in logs
- Number of nodes/edges per graph
- Ensure filtering is working as expected

---

## Quick Reference Table

### All Presets at a Glance

| Preset | A2A Mode | L2L | TL | Lanes | Best For |
|--------|----------|-----|----|----|----------|
| `baseline` | ego_only | ❌ | ❌ | all | **Current best**, starting point |
| `with_l2l` | ego_only | ✅ (all) | ❌ | all | Testing lane reasoning |
| `with_tl` | ego_only | ❌ | ✅ (all) | all | Traffic light scenarios |
| `with_full_a2a` | all pairs | ❌ | ❌ | all | Dense agent interaction |
| `full` | all pairs | ✅ | ✅ | all | Maximum info (slow) |
| `smart_a2a_k5` | k_nearest (k=5) | ❌ | ❌ | all | Balanced agent interaction |
| `typed_l2l` | ego_only | ✅ (typed) | ❌ | all | Semantic lane graph |
| `filtered_tl` | ego_only | ❌ | ✅ (filtered) | all | Efficient TL modeling |
| `optimized` | k_nearest (k=5) | ✅ (typed) | ✅ (filtered) | filtered (40) | All smart features |

### CLI Quick Reference

```bash
# Graph structure
--graph_preset {baseline|with_l2l|with_tl|with_full_a2a|full|smart_a2a_k5|typed_l2l|filtered_tl|optimized}
--include_l2l              # Add L2L edges
--include_tl               # Add traffic lights
--full_a2a                 # Full agent connectivity

# A2A filtering
--a2a_mode {ego_only|k_nearest|same_lane|directional}
--a2a_k N                  # k for k-nearest mode

# L2L typing
--l2l_mode {all|typed|successor_only}

# TL filtering
--tl_mode {all|relevant|controlling}
--tl_connect_to {agents|lanes|both}

# Lane filtering
--lane_mode {all|drivable|ego_relevant|ego_path}
--lane_max N               # Max lane count

# Model
--model {simple_gnn|typed_gnn|hierarchical_gnn}
--hidden N                 # Hidden dimension
--num_layers N             # Number of GNN layers
--conv_type {sage|gat}     # Convolution type

# Training
--epochs N
--batch_size N
--lr FLOAT
```

---

## Next Steps

1. **Run baseline** to establish performance floor
2. **Test one model variant** (typed_gnn or hierarchical_gnn)
3. **Try one graph preset** (smart_a2a_k5 or typed_l2l)
4. **Iterate** based on results

Good luck with your experiments! 🚀
