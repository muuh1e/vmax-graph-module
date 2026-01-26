# Context: Graph-Based Motion Prediction Pipeline

This file documents the purpose and flow of the graph-based pipeline built from:
- `hetero_graph.py`
- `graph_dataset.py`
- `simple_gnn.py`
- `train.py`

It explains how each file contributes to building heterogeneous graphs from Waymo
TFRecords, batching them, training a GNN, and saving checkpoints.

---

## 1) `hetero_graph.py` — Build a `HeteroData` graph from a TFRecord

**Responsibility**
- Reads a single TFRecord example.
- Extracts agent and lane features in an ego-centric coordinate frame.
- Builds agent-to-agent (A2A) and agent-to-lane (A2L) edges with attributes.
- Outputs a `torch_geometric.data.HeteroData` object with labels for future trajectories.

**Key flow (high level)**
1. Read one TFRecord example.
2. Extract ego state.
3. Extract agent features (current + past).
4. Extract future trajectories for supervision.
5. Extract lane features from roadgraph samples.
6. Build A2A and A2L edges.
7. Store everything in `HeteroData`.

**Core entry point**
```python
# hetero_graph.py

def build_hetero_graph(tfrecord_path, record_index, K_past=10, lane_config=None, a2l_k=3, a2a_max_dist=50.0):
    example = read_example_from_tfrecord(tfrecord_path, record_index)
    features = example.features.feature

    ego = get_ego_state(features)

    agent_features, valid_indices, ego_local_idx = extract_agent_features(features, ego, K_past=K_past)
    future_xy, future_valid = extract_future_trajectory(features, ego, valid_indices)

    rg = get_roadgraph_samples(features)
    map_features = build_map_features_from_roadgraph(rg, ego, cfg=lane_config)
    lane_features, lane_polylines = extract_lane_features(map_features, ego)

    a2a_edge_index, a2a_edge_attr, a2a_relations = build_a2a_edges(agent_features, ego_local_idx, max_dist=a2a_max_dist)
    a2l_edge_index, a2l_edge_attr = build_a2l_edges(agent_features, lane_features, lane_polylines, k=a2l_k)

    data = HeteroData()
    data['agent'].x = torch.from_numpy(agent_features)
    data['lane'].x = torch.from_numpy(lane_features)
    data['agent', 'to', 'agent'].edge_index = torch.from_numpy(a2a_edge_index)
    data['agent', 'to', 'agent'].edge_attr = torch.from_numpy(a2a_edge_attr)
    data['agent', 'to', 'lane'].edge_index = torch.from_numpy(a2l_edge_index)
    data['agent', 'to', 'lane'].edge_attr = torch.from_numpy(a2l_edge_attr)

    data.ego_idx = ego_local_idx
    data.y = torch.from_numpy(future_xy)                # [N_agents, 80, 2]
    data.future_valid = torch.from_numpy(future_valid)  # [N_agents, 80]
    data.ego_future = data.y[ego_local_idx]             # [80, 2]

    return data
```

### Agent features
Agents are transformed to the **ego frame** (ego at origin, yaw=0). The feature vector includes
current state, type one-hot, ego flag, and a history window.

```python
# hetero_graph.py (extract_agent_features)
# Features: [x, y, vx, vy, yaw, speed, cos(yaw), sin(yaw), length, width,
#            type_onehot(5), is_ego(1), past_x(K), past_y(K), past_vx(K), past_vy(K), past_valid(K)]
# Total: 16 + 5*K

agent_features[:, 0] = x_ego
agent_features[:, 1] = y_ego
agent_features[:, 2] = vx_ego
agent_features[:, 3] = vy_ego
agent_features[:, 4] = yaw_ego
agent_features[:, 5] = speed
agent_features[:, 6] = np.cos(yaw_ego)
agent_features[:, 7] = np.sin(yaw_ego)
...
agent_features[ego_local_idx, 15] = 1.0  # is_ego
```

### Future trajectory labels
Future positions are used as labels for motion prediction, also in ego frame.

```python
# hetero_graph.py (extract_future_trajectory)
future_xy = np.zeros((N_valid, T_future, 2), dtype=np.float32)
future_valid_out = np.zeros((N_valid, T_future), dtype=np.float32)

fx, fy = transform_to_ego_frame(fut_x[idx], fut_y[idx], ego_x, ego_y, ego_yaw)
future_xy[i, :T, 0] = fx
future_xy[i, :T, 1] = fy
future_valid_out[i, :T] = fut_valid[idx]
```

### Lane features
Lane features summarize geometry (centroid, bbox), speed limit, heading alignment, and a fixed
number of sampled polyline points (also transformed to ego frame).

```python
# hetero_graph.py (extract_lane_features)
# Features: centroid, closest point, bbox, speed limit, heading, num_points,
#           type one-hot, sampled polyline points

lane_features[i, 0] = cx[0]  # centroid x (ego)
lane_features[i, 1] = cy[0]  # centroid y (ego)
...
pts_ego = world_to_ego_points_xy(pts_world, np.array([ego_x, ego_y]), ego_yaw)
lane_features[i, offset:offset+n_pts] = pts_ego[:, 0]
```

### Edge construction
A2A edges connect **ego -> other agents** within a distance threshold and include semantic
edge attributes (relative position, velocity, TTC, distance buckets, etc.). A2L edges connect
agents to their nearest lanes with geometric relationship features.

```python
# hetero_graph.py (build_a2a_edges)
if dist > max_dist:
    continue

edge_attr = [
    dx, dy, dist,
    rel_vx, rel_vy,
    closing_speed,
    ttc, has_ttc,
    is_ahead, is_behind,
    is_left, is_right,
    dist_very_close, dist_close, dist_medium, dist_far,
    is_approaching, is_moving_away,
    is_leading, is_following,
]
```

```python
# hetero_graph.py (build_a2l_edges)
edge_attr = [
    dx, dy, dist,
    lateral_offset,
    heading_alignment,
    is_on_lane,
    is_approaching_lane,
    progress,
    angle_to_lane,
]
```

---

## 2) `graph_dataset.py` — Dataset that builds and stores graphs

**Responsibility**
- Provides a PyTorch Geometric `InMemoryDataset` wrapper.
- Converts all (or a subset of) TFRecord examples into a list of `HeteroData` graphs.
- Saves processed graphs to disk so they can be reused without rebuilding.

**Processing loop**
```python
# graph_dataset.py
for i in tqdm(range(num_records), desc="Building graphs"):
    graph = build_hetero_graph(self.tfrecord_path, i)

    ego_future = graph.y[graph.ego_idx].clone()
    ego_future_valid = graph.future_valid[graph.ego_idx].clone()

    graph.ego_future_target = ego_future
    graph.ego_future_valid = ego_future_valid
    graph.ego_idx_tensor = torch.tensor([graph.ego_idx], dtype=torch.long)

    del graph.a2a_relations  # remove non-tensor attributes for batching
    data_list.append(graph)
```

**Custom collation**
`Batch.from_data_list` handles most of it, but ego indices need to be offset by the
cumulative number of agent nodes.

```python
# graph_dataset.py
ptr = batch['agent'].ptr  # [batch_size + 1]
ego_indices = batch.ego_idx_tensor.squeeze(-1)
ego_indices_global = ego_indices + ptr[:-1]

batch.ego_indices_global = ego_indices_global
```

---

## 3) `simple_gnn.py` — Models for ego trajectory prediction

**Responsibility**
- Defines GNN models that take the hetero graphs and predict ego future trajectory.

There are two models:
1. **`SimpleHeteroGNN`** — uses `HeteroConv` with `SAGEConv` layers.
2. **`SimpleHeteroGNNWithEdgeAttr`** — custom message passing using edge attributes.

### 3.1 SimpleHeteroGNN
A minimal heterogeneous GNN:
- Project agent/lane features into a shared hidden space.
- Run multiple `HeteroConv` layers (agent-agent, agent-lane, lane-agent).
- Extract the ego node embedding and decode into a trajectory.

```python
# simple_gnn.py
self.agent_proj = Linear(agent_in_channels, hidden_channels)
self.lane_proj = Linear(lane_in_channels, hidden_channels)

self.convs = nn.ModuleList([
    HeteroConv({
        ('agent', 'to', 'agent'): SAGEConv(hidden_channels, hidden_channels, aggr='mean'),
        ('agent', 'to', 'lane'):  SAGEConv(hidden_channels, hidden_channels, aggr='mean'),
        ('lane', 'rev_to', 'agent'): SAGEConv(hidden_channels, hidden_channels, aggr='mean'),
    }, aggr='sum')
    for _ in range(num_layers)
])
```

```python
# simple_gnn.py (forward)
edge_index_dict = {
    ('agent', 'to', 'agent'): data['agent', 'to', 'agent'].edge_index,
    ('agent', 'to', 'lane'): data['agent', 'to', 'lane'].edge_index,
}

# Reverse edges for lane->agent
l2a_edge_index = torch.stack([a2l_edge_index[1], a2l_edge_index[0]], dim=0)
edge_index_dict[('lane', 'rev_to', 'agent')] = l2a_edge_index

# Extract ego embedding
ego_embeddings = x_dict['agent'][ego_indices]

# Decode to trajectory
pred = self.decoder(ego_embeddings).view(-1, self.num_future_steps, 2)
```

### 3.2 SimpleHeteroGNNWithEdgeAttr
Uses edge attributes in message aggregation.

```python
# simple_gnn.py
self.a2a_edge_proj = Linear(a2a_edge_channels, hidden_channels // 4)
self.a2l_edge_proj = Linear(a2l_edge_channels, hidden_channels // 4)
```

```python
# simple_gnn.py (aggregate_messages)
src_feats = src_features[src_idx]
edge_feats = edge_proj(edge_attr)
messages = torch.cat([src_feats, edge_feats], dim=-1)

from torch_scatter import scatter_mean
aggregated = scatter_mean(messages, dst_idx, dim=0, dim_size=num_dst_nodes)
```

---

## 4) `train.py` — Training loop and evaluation

**Responsibility**
- Loads the dataset.
- Splits into train/val sets.
- Trains the GNN with masked MSE.
- Reports ADE/FDE metrics.
- Saves best checkpoint to `checkpoints/best_model.pt`.

### Training loop
```python
# train.py
pred = model(batch)  # [batch, 80, 2]

# Targets
target = batch.ego_future_target.view(-1, 80, 2)
valid_mask = batch.ego_future_valid.view(-1, 80)

loss = compute_loss(pred, target, valid_mask)
```

### Metrics
```python
# train.py (compute_metrics)
# ADE: mean L2 over valid steps
masked_l2 = l2_dist * valid_mask
ade = (masked_l2.sum() / num_valid).item() if num_valid > 0 else 0.0

# FDE: L2 at the last valid timestep for each sample
last_valid = valid_steps[-1].item()
fde_values.append(l2_dist[i, last_valid].item())
```

### Saving best model
```python
# train.py
if val_metrics['ade'] < best_val_ade:
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_ade': val_metrics['ade'],
        'val_fde': val_metrics['fde'],
        'val_loss': val_loss,
    }, checkpoint_path)
```

---

## End-to-end data flow summary

1. **TFRecord -> Hetero graph**
   - `hetero_graph.py` converts one TFRecord example to a `HeteroData` graph.

2. **Graphs -> Dataset**
   - `graph_dataset.py` wraps graph construction and stores a list of graphs.

3. **Dataset -> Model**
   - `simple_gnn.py` defines GNNs that predict ego trajectories.

4. **Training**
   - `train.py` trains the GNN, evaluates ADE/FDE, and saves the best checkpoint.

---

## Where to look next
- If features or edge types need changes, start in `hetero_graph.py`.
- If batching or dataset caching is off, inspect `graph_dataset.py`.
- If model capacity is limited, adjust `simple_gnn.py`.
- For training settings, metrics, or checkpoints, edit `train.py`.
