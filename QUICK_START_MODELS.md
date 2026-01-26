# Quick Start: Testing Advanced Models

You have 3 GNN models available:
1. **simple_gnn** - Your baseline model
2. **typed_gnn** - LaneGCN-inspired with typed convolutions
3. **hierarchical_gnn** - HiVT-inspired with multi-modal prediction

## Fastest Way to Test

### Option 1: Use the test script (RECOMMENDED)

```bash
# Quick test all models (5-10 minutes)
./test_models.sh quick

# Full training for typed_gnn
./test_models.sh full typed_gnn with_l2l

# Full training for hierarchical_gnn
./test_models.sh full hierarchical_gnn smart_a2a_k5

# Compare all models
./test_models.sh compare
```

### Option 2: Direct CLI commands

```bash
# Test TypedHeteroGNN (quick)
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset with_l2l \
    --max_records 100 \
    --epochs 20 \
    --batch_size 8

# Test HierarchicalGNN (quick)
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --graph_preset smart_a2a_k5 \
    --max_records 100 \
    --epochs 20 \
    --batch_size 8 \
    --hidden 256

# Full training TypedHeteroGNN
python gnn_pipeline/train/run.py \
    --model typed_gnn \
    --graph_preset with_l2l \
    --epochs 50 \
    --batch_size 16 \
    --hidden 256

# Full training HierarchicalGNN
python gnn_pipeline/train/run.py \
    --model hierarchical_gnn \
    --graph_preset smart_a2a_k5 \
    --epochs 50 \
    --batch_size 16 \
    --hidden 256
```

## Available Models

| Model | Description | Best Graph Preset | Expected ADE |
|-------|-------------|-------------------|--------------|
| `simple_gnn` | Baseline heterogeneous GNN | `baseline` | 4.3-4.5m |
| `typed_gnn` | LaneGCN-style typed convs | `with_l2l` or `typed_l2l` | 4.0-4.3m |
| `hierarchical_gnn` | HiVT-style hierarchical | `smart_a2a_k5` | 3.8-4.2m |

## Key Command Line Options

```bash
--model <name>              # Model type: simple_gnn, typed_gnn, hierarchical_gnn
--graph_preset <preset>     # baseline, with_l2l, smart_a2a_k5, optimized, full
--epochs <n>                # Number of training epochs (default: 50)
--batch_size <n>            # Batch size (default: 16, use 8 if OOM)
--hidden <n>                # Hidden dimension (default: 128, try 256)
--num_layers <n>            # GNN layers (default: 3)
--max_records <n>           # Limit dataset size for quick tests
--checkpoint_dir <path>     # Where to save models
--device <cuda|cpu|auto>    # Training device
```

## Recommended First Steps

1. **Quick sanity check** (5 min):
   ```bash
   ./test_models.sh quick
   ```

2. **Train typed_gnn properly** (30-60 min):
   ```bash
   ./test_models.sh full typed_gnn with_l2l
   ```

3. **Train hierarchical_gnn properly** (30-60 min):
   ```bash
   ./test_models.sh full hierarchical_gnn smart_a2a_k5
   ```

4. **Compare all models** (1-2 hours):
   ```bash
   ./test_models.sh compare
   ```

## Output

You'll see training progress like this:
```
Epoch 10/50
----------------------------------------
Train Loss: 12.3456 | ADE: 4.21m | FDE: 11.34m
Val   Loss: 11.9876 | ADE: 4.08m | FDE: 10.98m
  -> Saved best model (ADE: 4.08m)
```

Lower ADE/FDE is better!

## Where to Find Results

- **Best model**: `checkpoints/best_model.pt`
- **Configs**: `results/graph_config.json`, `results/model_config.json`
- **Training logs**: Printed to console

## For More Details

See `docs/TESTING_ADVANCED_MODELS.md` for comprehensive documentation.
