# GNN Pipeline Documentation

Complete documentation for the GNN motion prediction pipeline.

## Documentation Index

### 🚀 [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)
**Start here if you want to run experiments.**

User-facing guide covering:
- Quick start and CLI usage
- Graph configuration presets
- Model architecture options
- Training workflow
- Systematic experimentation approach
- Results tracking

**Best for**: Running training, testing different configs, understanding CLI flags.

---

### 📖 [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md)
**Complete reference for all configuration options.**

Comprehensive guide covering:
- All `GraphConfig` parameters and their effects
- All `ModelConfig` parameters and their effects
- Detailed explanations of all presets
- How to combine configurations
- Experimentation strategies
- Quick reference tables

**Best for**: Understanding what each config parameter does, finding the right combination, systematic ablation studies.

---

### 🏗️ [ARCHITECTURE.md](ARCHITECTURE.md)
**Internal design and implementation details.**

Technical documentation covering:
- System architecture and data flow
- Core component descriptions
- Graph building logic
- Model registry and available models
- Advanced features (filtering, typing, caching)
- How to extend the pipeline

**Best for**: Understanding the codebase, adding new models/features, debugging.

---

## Quick Navigation

### I want to...

**Run my first experiment:**
→ Start with [PIPELINE_GUIDE.md - Quick Start](PIPELINE_GUIDE.md#quick-start)

**Understand all available configs:**
→ Read [CONFIG_REFERENCE.md - GraphConfig Reference](CONFIG_REFERENCE.md#graphconfig-reference)

**Find the best model/config combination:**
→ Follow [CONFIG_REFERENCE.md - Experimentation Guide](CONFIG_REFERENCE.md#experimentation-guide)

**Add a new model architecture:**
→ See [ARCHITECTURE.md - Extending the Pipeline](ARCHITECTURE.md#5-extending-the-pipeline)

**Understand how graphs are built:**
→ Read [ARCHITECTURE.md - Graph Building](ARCHITECTURE.md#22-graph-building-graphs)

**See all available presets:**
→ Check [CONFIG_REFERENCE.md - Graph Presets](CONFIG_REFERENCE.md#graph-presets)

**Track my experiment results:**
→ See [PIPELINE_GUIDE.md - Results Tracking](PIPELINE_GUIDE.md#results-tracking)

---

## Current Best Performance

**Baseline to Beat:**
- **Config**: `baseline` (agents + lanes, ego→others A2A)
- **Model**: `simple_gnn` (128 hidden, 3 layers, SAGE)
- **Val ADE**: ~4.3-4.5m ✓
- **Val FDE**: ~12-13m

Your goal: Find a better config/model combination!

---

## Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| `simple_gnn` | Basic heterogeneous GNN | Baseline, fast iterations |
| `typed_gnn` | LaneGCN-inspired with semantic edges | Typed L2L experiments |
| `hierarchical_gnn` | HiVT-inspired hierarchical encoder | State-of-the-art performance |

---

## Available Graph Presets

### Basic Presets
- `baseline`: Agents + lanes, ego→others (BEST SO FAR)
- `with_l2l`: + lane-to-lane edges
- `with_tl`: + traffic lights
- `with_full_a2a`: + full agent-agent connectivity
- `full`: Everything enabled

### Advanced Presets
- `smart_a2a_k5`: k-nearest A2A (k=5)
- `typed_l2l`: Typed L2L edges (LaneGCN-style)
- `filtered_tl`: Filtered traffic lights (controlling only)
- `optimized`: All smart features combined

See [CONFIG_REFERENCE.md](CONFIG_REFERENCE.md) for complete details.

---

## Example Commands

```bash
# Baseline (current best)
python gnn_pipeline/train/run.py --graph_preset baseline --epochs 50

# Test typed lane graph
python gnn_pipeline/train/run.py --graph_preset typed_l2l --model typed_gnn --epochs 50

# Test hierarchical model
python gnn_pipeline/train/run.py --graph_preset baseline --model hierarchical_gnn --epochs 50

# Custom combination
python gnn_pipeline/train/run.py \
  --a2a_mode k_nearest --a2a_k 10 \
  --include_l2l --l2l_mode typed \
  --hidden 256 --num_layers 4
```

---

## File Organization

```
gnn_pipeline/
├── docs/
│   ├── README.md              # This file (documentation index)
│   ├── PIPELINE_GUIDE.md      # User guide for experiments
│   ├── CONFIG_REFERENCE.md    # Complete config reference
│   └── ARCHITECTURE.md        # Internal design docs
├── configs/
│   ├── graph_config.py        # GraphConfig dataclass
│   └── model_config.py        # ModelConfig dataclass
├── graphs/
│   ├── hetero_graph.py        # Graph building logic
│   ├── edge_builders.py       # Edge construction functions
│   └── graph_dataset.py       # PyTorch Geometric dataset
├── models/
│   ├── simple_gnn.py          # SimpleHeteroGNN
│   ├── typed_gnn.py           # TypedHeteroGNN
│   ├── hierarchical_gnn.py    # HierarchicalGNN
│   └── layers/                # Reusable model components
└── train/
    ├── run.py                 # Training script
    ├── trainer.py             # Training loop
    └── metrics.py             # ADE/FDE metrics
```

---

## Getting Help

1. Check the relevant documentation file above
2. Look at example commands in PIPELINE_GUIDE.md
3. Refer to CONFIG_REFERENCE.md for parameter details
4. Check ARCHITECTURE.md for implementation details

Happy experimenting! 🚀
