import os
import sys
import argparse
from pathlib import Path

# Add project root to path so we can import gnn_pipeline and spatial_graph
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import numpy as np
from torch_geometric.loader import DataLoader

from gnn_pipeline import (
    GraphConfig,
    ModelConfig,
    get_model,
    Trainer,
    WaymoGraphDataset,
)
from gnn_pipeline.train.trainer import custom_collate


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GNN motion prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data arguments
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--tfrecord",
        type=str,
        default=os.path.expanduser("~/vmax/data/waymo_converted/training.tfrecord"),
        help="Path to TFRecord file"
    )
    data_group.add_argument(
        "--processed_dir",
        type=str,
        default=str(PROJECT_ROOT / "processed_graphs"),
        help="Directory for processed graph data"
    )
    data_group.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records to use (None for all)"
    )

    # Graph config arguments
    graph_group = parser.add_argument_group("Graph Configuration")
    graph_group.add_argument(
        "--graph_preset",
        type=str,
        default="baseline",
        choices=["baseline", "with_l2l", "with_tl", "with_full_a2a", "full",
                 "smart_a2a_k5", "typed_l2l", "filtered_tl", "optimized"],
        help="Graph configuration preset"
    )
    graph_group.add_argument(
        "--include_l2l",
        action="store_true",
        help="Include lane-to-lane edges (overrides preset)"
    )
    graph_group.add_argument(
        "--include_tl",
        action="store_true",
        help="Include traffic light nodes and edges (overrides preset)"
    )
    graph_group.add_argument(
        "--full_a2a",
        action="store_true",
        help="Use full agent-to-agent connectivity (overrides preset)"
    )
    
    # A2A filtering arguments (NEW)
    graph_group.add_argument(
        "--a2a_mode",
        type=str,
        default=None,
        choices=["ego_only", "k_nearest", "same_lane", "directional"],
        help="A2A edge building mode (overrides preset)"
    )
    graph_group.add_argument(
        "--a2a_k",
        type=int,
        default=5,
        help="Number of nearest agents for k_nearest mode"
    )
    graph_group.add_argument(
        "--a2a_max_dist",
        type=float,
        default=50.0,
        help="Maximum distance for A2A edges (meters)"
    )
    
    # L2L typing arguments (NEW)
    graph_group.add_argument(
        "--l2l_mode",
        type=str,
        default=None,
        choices=["all", "typed", "successor_only"],
        help="L2L edge mode (overrides preset)"
    )
    
    # TL filtering arguments (NEW)
    graph_group.add_argument(
        "--tl_mode",
        type=str,
        default=None,
        choices=["all", "relevant", "controlling"],
        help="Traffic light filtering mode (overrides preset)"
    )
    graph_group.add_argument(
        "--tl_connect_to",
        type=str,
        default=None,
        choices=["agents", "lanes", "both"],
        help="What TLs connect to (overrides preset)"
    )
    
    # Lane filtering arguments (NEW)
    graph_group.add_argument(
        "--lane_mode",
        type=str,
        default=None,
        choices=["all", "drivable", "ego_relevant", "ego_path"],
        help="Lane filtering mode (overrides preset)"
    )
    graph_group.add_argument(
        "--lane_max",
        type=int,
        default=None,
        help="Maximum number of lanes to keep (overrides preset)"
    )

    # Model config arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="simple_gnn",
        help="Model type"
    )
    model_group.add_argument(
        "--hidden",
        type=int,
        default=128,
        help="Hidden dimension size"
    )
    model_group.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of GNN layers"
    )
    model_group.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )
    model_group.add_argument(
        "--conv_type",
        type=str,
        default="sage",
        choices=["sage", "gat"],
        help="Convolution type"
    )
    model_group.add_argument(
        "--use_edge_attr",
        action="store_true",
        help="Use edge attributes in message passing"
    )
    
    # Temporal encoder arguments (Phase 2A)
    temporal_group = parser.add_argument_group("Temporal Encoder")
    temporal_group.add_argument(
        "--use_temporal_encoder",
        action="store_true",
        help="Use temporal encoder for agent history (Phase 2A)"
    )
    temporal_group.add_argument(
        "--temporal_encoder_type",
        type=str,
        default="transformer",
        choices=["transformer", "conv1d", "gru"],
        help="Type of temporal encoder"
    )
    temporal_group.add_argument(
        "--temporal_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for temporal encoder"
    )
    temporal_group.add_argument(
        "--temporal_num_layers",
        type=int,
        default=2,
        help="Number of layers in temporal encoder"
    )
    temporal_group.add_argument(
        "--temporal_num_heads",
        type=int,
        default=4,
        help="Number of attention heads for transformer temporal encoder"
    )

    # Polyline encoder arguments (Phase 2C)
    polyline_group = parser.add_argument_group("Polyline Encoder")
    polyline_group.add_argument(
        "--use_polyline_encoder",
        action="store_true",
        help="Use polyline encoder for lane geometry (Phase 2C)"
    )
    polyline_group.add_argument(
        "--polyline_encoder_type",
        type=str,
        default="pointnet",
        choices=["pointnet", "transformer", "conv1d"],
        help="Type of polyline encoder"
    )
    polyline_group.add_argument(
        "--polyline_max_points",
        type=int,
        default=20,
        help="Max points per lane polyline"
    )
    polyline_group.add_argument(
        "--polyline_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for polyline encoder"
    )
    polyline_group.add_argument(
        "--polyline_num_layers",
        type=int,
        default=3,
        help="Number of layers in polyline encoder"
    )

    # Goal/SDC path arguments (Phase 2B)
    goal_group = parser.add_argument_group("Goal Nodes")
    goal_group.add_argument(
        "--include_goal",
        action="store_true",
        help="Include goal/SDC path nodes (Phase 2B)"
    )
    goal_group.add_argument(
        "--goal_mode",
        type=str,
        default="waypoints",
        choices=["endpoint", "waypoints"],
        help="Goal representation mode"
    )
    goal_group.add_argument(
        "--goal_num_waypoints",
        type=int,
        default=10,
        help="Number of goal waypoints (for waypoints mode)"
    )
    goal_group.add_argument(
        "--goal_waypoint_spacing",
        type=float,
        default=5.0,
        help="Spacing between goal waypoints in meters"
    )
    goal_group.add_argument(
        "--goal_max_distance",
        type=float,
        default=80.0,
        help="Max distance from ego for goal waypoints"
    )

    # Training arguments
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs"
    )
    train_group.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    train_group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    train_group.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio"
    )
    train_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to train on"
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Output arguments
    output_group = parser.add_argument_group("Output")
    output_group.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(PROJECT_ROOT / "checkpoints"),
        help="Directory to save checkpoints"
    )
    output_group.add_argument(
        "--results_dir",
        type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save results"
    )
    output_group.add_argument(
        "--plots_dir",
        type=str,
        default=str(PROJECT_ROOT / "plots"),
        help="Directory to save training plots"
    )

    return parser.parse_args()


def build_graph_config(args) -> GraphConfig:
    """Build GraphConfig from arguments."""
    # Start with preset
    preset_map = {
        "baseline": GraphConfig.baseline,
        "with_l2l": GraphConfig.with_l2l,
        "with_tl": GraphConfig.with_tl,
        "with_full_a2a": GraphConfig.with_full_a2a,
        "full": GraphConfig.full,
        # New presets
        "smart_a2a_k5": GraphConfig.smart_a2a_k5,
        "typed_l2l": GraphConfig.typed_l2l,
        "filtered_tl": GraphConfig.filtered_tl,
        "optimized": GraphConfig.optimized,
    }
    config = preset_map[args.graph_preset]()

    # Apply legacy overrides
    if args.include_l2l:
        config.include_l2l = True
    if args.include_tl:
        config.include_tl = True
        config.include_a2tl = True
        config.include_l2tl = True
    if args.full_a2a:
        config.full_a2a = True

    # Apply new A2A overrides
    if args.a2a_mode is not None:
        config.a2a_mode = args.a2a_mode
    if args.a2a_k is not None:
        config.a2a_k_nearest = args.a2a_k
    if args.a2a_max_dist is not None:
        config.a2a_max_dist = args.a2a_max_dist

    # Apply L2L overrides
    if args.l2l_mode is not None:
        config.l2l_mode = args.l2l_mode
        config.include_l2l = True
        if args.l2l_mode == "typed":
            config.l2l_typed_convs = True

    # Apply TL overrides
    if args.tl_mode is not None:
        config.tl_filter_mode = args.tl_mode
        config.include_tl = True
    if args.tl_connect_to is not None:
        config.tl_connect_to = args.tl_connect_to

    # Apply Lane overrides
    if args.lane_mode is not None:
        config.lane_filter_mode = args.lane_mode
    if args.lane_max is not None:
        config.lane_max_count = args.lane_max

    # Apply Polyline encoder overrides (Phase 2C)
    if args.use_polyline_encoder:
        config.use_polyline_encoder = True
        config.polyline_max_points = args.polyline_max_points

    # Apply Goal overrides (Phase 2B)
    if args.include_goal:
        config.include_goal = True
        config.goal_mode = args.goal_mode
        config.goal_num_waypoints = args.goal_num_waypoints
        config.goal_waypoint_spacing = args.goal_waypoint_spacing
        config.goal_max_distance = args.goal_max_distance

    return config


def build_model_config(args, graph_config: GraphConfig) -> ModelConfig:
    """Build ModelConfig from arguments."""
    config = ModelConfig(
        model_type=args.model,
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
        use_edge_attr=args.use_edge_attr,
        # Temporal encoder (Phase 2A)
        use_temporal_encoder=args.use_temporal_encoder,
        temporal_encoder_type=args.temporal_encoder_type,
        temporal_hidden_dim=args.temporal_hidden_dim,
        temporal_num_layers=args.temporal_num_layers,
        temporal_num_heads=args.temporal_num_heads,
        # Polyline encoder (Phase 2C)
        use_polyline_encoder=args.use_polyline_encoder,
        polyline_encoder_type=args.polyline_encoder_type,
        polyline_hidden_dim=args.polyline_hidden_dim,
        polyline_num_layers=args.polyline_num_layers,
    )

    # Sync edge type flags with graph config
    config.sync_with_graph_config(graph_config)

    return config


def main():
    args = parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Build configs
    graph_config = build_graph_config(args)
    model_config = build_model_config(args, graph_config)

    print(f"\n=== Configuration ===")
    print(f"Graph config: {graph_config.describe()}")
    print(f"Model type: {model_config.model_type}")
    print(f"Hidden channels: {model_config.hidden_channels}")
    print(f"Num layers: {model_config.num_layers}")
    print(f"Conv type: {model_config.conv_type}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Load dataset
    print(f"\n=== Loading Dataset ===")
    print(f"TFRecord: {args.tfrecord}")
    print(f"Processed dir: {args.processed_dir}")

    dataset = WaymoGraphDataset(
        root=args.processed_dir,
        tfrecord_path=args.tfrecord,
        config=graph_config,
        max_records=args.max_records,
    )
    print(f"Dataset size: {len(dataset)} graphs")

    if len(dataset) == 0:
        print("Error: No graphs in dataset!")
        return

    # Get feature dimensions from first sample
    sample = dataset[0]
    agent_in_channels = sample['agent'].x.shape[1]
    lane_in_channels = sample['lane'].x.shape[1]
    tl_in_channels = sample['tl'].x.shape[1] if sample['tl'].x.shape[0] > 0 else 12
    goal_in_channels = sample['goal'].x.shape[1] if 'goal' in sample.node_types and sample['goal'].x.shape[0] > 0 else 10

    print(f"Agent features: {agent_in_channels}")
    print(f"Lane features: {lane_in_channels}")
    print(f"TL features: {tl_in_channels}")
    print(f"Goal features: {goal_in_channels}")

    # Train/val split
    num_val = int(len(dataset) * args.val_split)
    num_train = len(dataset) - num_val

    indices = torch.randperm(len(dataset)).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_dataset = [dataset[i] for i in train_indices]
    val_dataset = [dataset[i] for i in val_indices]

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
        num_workers=0,
    )

    # Create model
    print(f"\n=== Creating Model ===")
    model = get_model(
        model_config=model_config,
        graph_config=graph_config,
        agent_in_channels=agent_in_channels,
        lane_in_channels=lane_in_channels,
        tl_in_channels=tl_in_channels,
        goal_in_channels=goal_in_channels,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Create trainer and run
    # Build a descriptive model name for plots
    model_display_name = f"{args.model}_{args.graph_preset}_h{args.hidden}_L{args.num_layers}"

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        plots_dir=args.plots_dir,
        model_name=model_display_name,
    )

    results = trainer.train(epochs=args.epochs)

    # Save configs and results
    config_path = os.path.join(args.results_dir, 'graph_config.json')
    graph_config.save(config_path)

    model_config_path = os.path.join(args.results_dir, 'model_config.json')
    model_config.save(model_config_path)

    print(f"\n" + "=" * 60)
    print("=== Output Locations ===")
    print("=" * 60)
    print(f"Configs:     {args.results_dir}")
    print(f"Checkpoints: {args.checkpoint_dir}")
    print(f"Plots:       {args.plots_dir}")
    print(f"Best model:  {os.path.join(args.checkpoint_dir, 'best_model.pt')}")


if __name__ == "__main__":
    main()
