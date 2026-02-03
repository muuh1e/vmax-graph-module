"""
Test script for Phase 2C Polyline Encoders.

Tests the PointNetEncoder, TransformerPolylineEncoder, and Conv1DPolylineEncoder
classes and the get_polyline_encoder factory function.

Run with: python -m gnn_pipeline.tests.test_polyline_encoder
"""

import torch
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from gnn_pipeline.models.layers.polyline_encoder import (
    PointNetEncoder,
    TransformerPolylineEncoder,
    Conv1DPolylineEncoder,
    get_polyline_encoder,
)


def test_pointnet_encoder():
    """Test PointNetEncoder."""
    print("Testing PointNetEncoder...")

    encoder = PointNetEncoder(
        input_dim=2,
        hidden_dim=64,
        output_dim=128,
        num_layers=3,
        use_position_encoding=True,
    )

    # Test with batch of polylines
    batch_size = 16
    max_points = 20
    points = torch.randn(batch_size, max_points, 2)
    mask = torch.ones(batch_size, max_points, dtype=torch.bool)
    mask[:, 15:] = False  # Some points invalid

    out = encoder(points, mask)
    assert out.shape == (batch_size, 128), f"Expected (16, 128), got {out.shape}"
    print(f"  Input: {points.shape} -> Output: {out.shape}")

    # Test without mask
    out_no_mask = encoder(points, None)
    assert out_no_mask.shape == (batch_size, 128)
    print("  No mask test passed")

    # Test with all masked
    all_masked = torch.zeros(batch_size, max_points, dtype=torch.bool)
    out_all_masked = encoder(points, all_masked)
    assert out_all_masked.shape == (batch_size, 128)
    assert not torch.isnan(out_all_masked).any(), "NaN in all-masked output"
    print("  All-masked test passed")

    print("  PointNetEncoder tests PASSED\n")


def test_transformer_encoder():
    """Test TransformerPolylineEncoder."""
    print("Testing TransformerPolylineEncoder...")

    encoder = TransformerPolylineEncoder(
        input_dim=2,
        hidden_dim=64,
        output_dim=128,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
    )

    batch_size = 16
    max_points = 20
    points = torch.randn(batch_size, max_points, 2)
    mask = torch.ones(batch_size, max_points, dtype=torch.bool)
    mask[:, 15:] = False

    out = encoder(points, mask)
    assert out.shape == (batch_size, 128), f"Expected (16, 128), got {out.shape}"
    print(f"  Input: {points.shape} -> Output: {out.shape}")

    # Test without mask
    out_no_mask = encoder(points, None)
    assert out_no_mask.shape == (batch_size, 128)
    print("  No mask test passed")

    print("  TransformerPolylineEncoder tests PASSED\n")


def test_conv1d_encoder():
    """Test Conv1DPolylineEncoder."""
    print("Testing Conv1DPolylineEncoder...")

    encoder = Conv1DPolylineEncoder(
        input_dim=2,
        hidden_dim=64,
        output_dim=128,
        num_layers=3,
        kernel_size=3,
    )

    batch_size = 16
    max_points = 20
    points = torch.randn(batch_size, max_points, 2)
    mask = torch.ones(batch_size, max_points, dtype=torch.bool)
    mask[:, 15:] = False

    out = encoder(points, mask)
    assert out.shape == (batch_size, 128), f"Expected (16, 128), got {out.shape}"
    print(f"  Input: {points.shape} -> Output: {out.shape}")

    # Test without mask
    out_no_mask = encoder(points, None)
    assert out_no_mask.shape == (batch_size, 128)
    print("  No mask test passed")

    print("  Conv1DPolylineEncoder tests PASSED\n")


def test_factory_function():
    """Test get_polyline_encoder factory function."""
    print("Testing get_polyline_encoder factory...")

    for enc_type in ['pointnet', 'transformer', 'conv1d']:
        encoder = get_polyline_encoder(
            encoder_type=enc_type,
            input_dim=2,
            hidden_dim=64,
            output_dim=128,
        )

        # Simulate batch of lane polylines
        points = torch.randn(16, 20, 2)  # [batch, max_points, xy]
        mask = torch.ones(16, 20, dtype=torch.bool)
        mask[:, 15:] = False  # Some points invalid

        out = encoder(points, mask)
        print(f"  {enc_type}: input {points.shape} -> output {out.shape}")
        assert out.shape == (16, 128), f"Expected (16, 128), got {out.shape}"

    # Test invalid encoder type
    try:
        get_polyline_encoder("invalid_type", 2, 64, 128)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Invalid type correctly raises: {e}")

    print("  Factory function tests PASSED\n")


def test_gradient_flow():
    """Test that gradients flow properly through encoders."""
    print("Testing gradient flow...")

    for enc_type in ['pointnet', 'transformer', 'conv1d']:
        encoder = get_polyline_encoder(
            encoder_type=enc_type,
            input_dim=2,
            hidden_dim=64,
            output_dim=128,
        )

        points = torch.randn(8, 20, 2, requires_grad=True)
        mask = torch.ones(8, 20, dtype=torch.bool)

        out = encoder(points, mask)
        loss = out.sum()
        loss.backward()

        assert points.grad is not None, f"{enc_type}: No gradient computed"
        assert not torch.isnan(points.grad).any(), f"{enc_type}: NaN in gradients"
        print(f"  {enc_type}: gradients computed successfully")

    print("  Gradient flow tests PASSED\n")


def test_different_point_counts():
    """Test encoders with different numbers of points."""
    print("Testing different point counts...")

    for n_points in [5, 10, 20, 40]:
        encoder = get_polyline_encoder(
            encoder_type='pointnet',
            input_dim=2,
            hidden_dim=64,
            output_dim=128,
        )

        points = torch.randn(4, n_points, 2)
        mask = torch.ones(4, n_points, dtype=torch.bool)

        out = encoder(points, mask)
        assert out.shape == (4, 128), f"Failed for n_points={n_points}"
        print(f"  n_points={n_points}: OK")

    print("  Different point counts tests PASSED\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Phase 2C: Polyline Encoder Tests")
    print("=" * 60 + "\n")

    test_pointnet_encoder()
    test_transformer_encoder()
    test_conv1d_encoder()
    test_factory_function()
    test_gradient_flow()
    test_different_point_counts()

    print("=" * 60)
    print("All polyline encoder tests PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
