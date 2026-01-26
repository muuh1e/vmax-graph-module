
import torch
from torch_geometric.data import HeteroData, Batch

def create_dummy_graph(idx):
    data = HeteroData()
    data['agent'].x = torch.randn(10, 66)
    data.ego_idx_tensor = torch.tensor([idx])
    return data

def test():
    # Create batch
    graphs = [create_dummy_graph(i) for i in range(4)]
    batch = Batch.from_data_list(graphs)
    
    # Custom collate logic (monkey-patching)
    batch.ego_indices_global = torch.arange(4)
    
    print("Before .to(device):")
    print(f"  Has ego_idx_tensor: {hasattr(batch, 'ego_idx_tensor')}")
    print(f"  Has ego_indices_global: {hasattr(batch, 'ego_indices_global')}")
    
    # Move to device (CPU for test, but .to() should trigger the behavior)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nMoving to {device}...")
    batch_gpu = batch.to(device)
    
    print("After .to(device):")
    print(f"  Has ego_idx_tensor: {hasattr(batch_gpu, 'ego_idx_tensor')}")
    print(f"  Has ego_indices_global: {hasattr(batch_gpu, 'ego_indices_global')}")

if __name__ == "__main__":
    test()
