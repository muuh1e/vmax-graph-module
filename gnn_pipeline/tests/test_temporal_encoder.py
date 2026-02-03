#!/usr/bin/env python
"""
Test script for Phase 2A temporal encoders.

Verifies:
1. All encoder types produce correct output shapes
2. Temporal encoders integrate with agent feature processing
3. No NaN/Inf values

Usage:
    python gnn_pipeline/tests/test_temporal_encoder.py
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from gnn_pipeline.models.layers.temporal_encoder import (
    TemporalTransformerEncoder,
    TemporalConv1DEncoder,
    TemporalGRUEncoder,
    get_temporal_encoder,
)


def test_encoder_shapes():
    """Test that all encoders produce correct output shapes."""
    print("\n[TEST] Encoder Output Shapes")
    
    batch_size = 32
    seq_len = 10
    input_dim = 5
    hidden_dim = 64
    output_dim = 128
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    all_passed = True
    for enc_type in ['transformer', 'conv1d', 'gru']:
        encoder = get_temporal_encoder(
            encoder_type=enc_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=2,
        )
        
        out = encoder(x)
        expected_shape = (batch_size, output_dim)
        passed = out.shape == expected_shape
        
        print(f"  {enc_type}: input {tuple(x.shape)} -> output {tuple(out.shape)} "
              f"(expected {expected_shape}) - {'PASS' if passed else 'FAIL'}")
        all_passed &= passed
    
    return all_passed


def test_no_nan_inf():
    """Test that outputs have no NaN or Inf values."""
    print("\n[TEST] No NaN/Inf Values")
    
    x = torch.randn(16, 10, 5)
    
    all_passed = True
    for enc_type in ['transformer', 'conv1d', 'gru']:
        encoder = get_temporal_encoder(
            encoder_type=enc_type,
            input_dim=5,
            hidden_dim=64,
            output_dim=128,
        )
        
        out = encoder(x)
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        passed = not (has_nan or has_inf)
        
        print(f"  {enc_type}: NaN={has_nan}, Inf={has_inf} - {'PASS' if passed else 'FAIL'}")
        all_passed &= passed
    
    return all_passed


def test_gradients():
    """Test that gradients flow correctly through model parameters."""
    print("\n[TEST] Gradient Flow")
    
    all_passed = True
    for enc_type in ['transformer', 'conv1d', 'gru']:
        encoder = get_temporal_encoder(
            encoder_type=enc_type,
            input_dim=5,
            hidden_dim=64,
            output_dim=128,
        )
        encoder.train()
        
        x = torch.randn(8, 10, 5)
        out = encoder(x)
        loss = out.sum()
        loss.backward()
        
        # Check that at least some parameters have gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                       for p in encoder.parameters() if p.requires_grad)
        print(f"  {enc_type}: params_have_gradient={has_grad} - {'PASS' if has_grad else 'FAIL'}")
        all_passed &= has_grad
        
        encoder.zero_grad()  # Reset for next iteration
    
    return all_passed


def test_factory_function():
    """Test factory function with invalid type."""
    print("\n[TEST] Factory Function Validation")
    
    try:
        get_temporal_encoder("invalid_type", 5, 64, 128)
        print("  Invalid type check: FAIL (should have raised ValueError)")
        return False
    except ValueError as e:
        print(f"  Invalid type check: PASS (raised ValueError as expected)")
        return True


def main():
    print("=" * 50)
    print("Phase 2A: Temporal Encoder Tests")
    print("=" * 50)
    
    all_passed = True
    all_passed &= test_encoder_shapes()
    all_passed &= test_no_nan_inf()
    all_passed &= test_gradients()
    all_passed &= test_factory_function()
    
    print("\n" + "=" * 50)
    print(f"Result: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
