#!/usr/bin/env python3
"""
graph_dataset.py

PyTorch Geometric InMemoryDataset for Waymo motion prediction.
Loads TFRecords and builds heterogeneous graphs using hetero_graph.py.
"""

import os
from typing import Optional, Callable
import torch
from torch_geometric.data import InMemoryDataset, HeteroData
from tqdm import tqdm

from hetero_graph import build_hetero_graph


def count_tfrecord_examples(tfrecord_path: str) -> int:
    """Count number of examples in a TFRecord file."""
    import tensorflow as tf
    count = 0
    for _ in tf.data.TFRecordDataset(tfrecord_path):
        count += 1
    return count


class WaymoGraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for Waymo motion prediction.

    Loads TFRecords and builds heterogeneous graphs with:
    - Agent nodes with trajectory features
    - Lane nodes with road geometry features
    - Agent-to-agent and agent-to-lane edges

    Args:
        root: Root directory where processed dataset will be saved
        tfrecord_path: Path to the TFRecord file
        max_records: Maximum number of records to load (None for all)
        transform: Optional transform to apply to each sample
        pre_transform: Optional pre-transform to apply during processing
    """

    def __init__(
        self,
        root: str,
        tfrecord_path: str,
        max_records: Optional[int] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.tfrecord_path = tfrecord_path
        self.max_records = max_records
        self._graph_list = None
        super().__init__(root, transform, pre_transform)

        # Load graphs from file
        self._graph_list = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # We use TFRecord directly, no raw files needed
        return []

    @property
    def processed_file_names(self):
        # Include max_records in filename for different subsets
        suffix = f"_{self.max_records}" if self.max_records else "_all"
        return [f'data{suffix}.pt']

    def download(self):
        # No download needed, TFRecord is provided
        pass

    def process(self):
        """Process TFRecords and build graphs."""
        print(f"Processing TFRecord: {self.tfrecord_path}")

        # Count total records
        total_records = count_tfrecord_examples(self.tfrecord_path)
        print(f"Total records in TFRecord: {total_records}")

        # Limit if specified
        num_records = min(total_records, self.max_records) if self.max_records else total_records
        print(f"Processing {num_records} records...")

        data_list = []
        failed = 0

        for i in tqdm(range(num_records), desc="Building graphs"):
            try:
                graph = build_hetero_graph(self.tfrecord_path, i)

                # Store ego future target as a separate tensor for easier batching
                # This will be automatically batched to [batch_size, 80, 2]
                ego_future = graph.y[graph.ego_idx].clone()  # [80, 2]
                ego_future_valid = graph.future_valid[graph.ego_idx].clone()  # [80]

                # Store as graph attributes (not nested in node stores)
                graph.ego_future_target = ego_future
                graph.ego_future_valid = ego_future_valid
                graph.ego_idx_tensor = torch.tensor([graph.ego_idx], dtype=torch.long)

                # Remove non-tensor attributes that can't be batched
                del graph.a2a_relations

                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)

                data_list.append(graph)

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"\nWarning: Failed to process record {i}: {e}")
                continue

        print(f"\nProcessed {len(data_list)} graphs successfully ({failed} failed)")

        # Save as a simple list (no collation)
        torch.save(data_list, self.processed_paths[0])
        print("Done!")

    def len(self) -> int:
        return len(self._graph_list) if self._graph_list is not None else 0

    def get(self, idx: int) -> HeteroData:
        return self._graph_list[idx]


def collate_fn(data_list):
    """Custom collate function for HeteroData batching.

    PyTorch Geometric's Batch.from_data_list handles most batching,
    but we need to track ego indices properly across the batch.
    """
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(data_list)

    # The ego_idx_tensor is already batched as [batch_size, 1]
    # We need to offset them by the cumulative number of agent nodes

    # Get cumulative agent node counts
    ptr = batch['agent'].ptr  # [batch_size + 1]

    # Offset ego indices
    ego_indices = batch.ego_idx_tensor.squeeze(-1)  # [batch_size]
    ego_indices_global = ego_indices + ptr[:-1]  # Add offset for each graph

    batch.ego_indices_global = ego_indices_global

    return batch


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build Waymo Graph Dataset")
    parser.add_argument(
        "--tfrecord",
        type=str,
        default=os.path.expanduser("~/vmax/data/waymo_converted/training.tfrecord"),
        help="Path to TFRecord file"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="./processed_graphs",
        help="Root directory for processed data"
    )
    parser.add_argument(
        "--max_records",
        type=int,
        default=None,
        help="Maximum number of records to process"
    )
    args = parser.parse_args()

    print(f"Building dataset from {args.tfrecord}")
    print(f"Saving to {args.root}")

    dataset = WaymoGraphDataset(
        root=args.root,
        tfrecord_path=args.tfrecord,
        max_records=args.max_records,
    )

    print(f"\n=== Dataset Statistics ===")
    print(f"Number of graphs: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\n=== Sample Graph ===")
        print(f"Agent nodes: {sample['agent'].x.shape}")
        print(f"Lane nodes: {sample['lane'].x.shape}")
        print(f"A2A edges: {sample['agent', 'to', 'agent'].edge_index.shape}")
        print(f"A2L edges: {sample['agent', 'to', 'lane'].edge_index.shape}")
        print(f"Ego future target: {sample.ego_future_target.shape}")
        print(f"Ego future valid: {sample.ego_future_valid.shape}")

        # Test batching
        from torch_geometric.loader import DataLoader
        loader = DataLoader(dataset[:4], batch_size=2, follow_batch=['agent', 'lane'])

        for batch in loader:
            print(f"\n=== Batch Test ===")
            print(f"Batch agent nodes: {batch['agent'].x.shape}")
            print(f"Batch ego_future_target: {batch.ego_future_target.shape}")
            print(f"Batch agent ptr: {batch['agent'].ptr}")
            break

    print("\nDone!")
