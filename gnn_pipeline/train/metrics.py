"""
Metrics for motion prediction evaluation.

Provides ADE (Average Displacement Error) and FDE (Final Displacement Error)
computation with support for validity masks.
"""

from typing import Dict

import torch
import numpy as np


def compute_ade(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Compute Average Displacement Error.

    ADE is the mean L2 distance between predicted and ground truth
    positions across all valid timesteps.

    Args:
        pred: [batch, T, 2] predicted positions
        target: [batch, T, 2] ground truth positions
        valid_mask: [batch, T] validity mask (1 = valid, 0 = invalid)

    Returns:
        ADE in meters (scalar)
    """
    # L2 distance at each timestep
    l2_dist = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # [batch, T]

    # Masked mean
    masked_l2 = l2_dist * valid_mask
    num_valid = valid_mask.sum()

    if num_valid > 0:
        ade = (masked_l2.sum() / num_valid).item()
    else:
        ade = 0.0

    return ade


def compute_fde(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> float:
    """
    Compute Final Displacement Error.

    FDE is the L2 distance at the final valid timestep for each sample,
    averaged across the batch.

    Args:
        pred: [batch, T, 2] predicted positions
        target: [batch, T, 2] ground truth positions
        valid_mask: [batch, T] validity mask (1 = valid, 0 = invalid)

    Returns:
        FDE in meters (scalar)
    """
    # L2 distance at each timestep
    l2_dist = torch.sqrt(((pred - target) ** 2).sum(dim=-1))  # [batch, T]

    batch_size = pred.shape[0]
    fde_values = []

    for i in range(batch_size):
        valid_steps = valid_mask[i].nonzero(as_tuple=True)[0]
        if len(valid_steps) > 0:
            last_valid = valid_steps[-1].item()
            fde_values.append(l2_dist[i, last_valid].item())

    fde = np.mean(fde_values) if fde_values else 0.0
    return fde


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all motion prediction metrics.

    Args:
        pred: [batch, T, 2] predicted positions
        target: [batch, T, 2] ground truth positions
        valid_mask: [batch, T] validity mask

    Returns:
        Dictionary with 'ade' and 'fde' in meters
    """
    return {
        'ade': compute_ade(pred, target, valid_mask),
        'fde': compute_fde(pred, target, valid_mask),
    }


def compute_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute masked MSE loss.

    Args:
        pred: [batch, T, 2] predicted positions
        target: [batch, T, 2] ground truth positions
        valid_mask: [batch, T] validity mask (1 = valid, 0 = invalid)

    Returns:
        Scalar MSE loss over valid positions
    """
    # Expand mask to match prediction shape
    mask = valid_mask.unsqueeze(-1).expand_as(pred)  # [batch, T, 2]

    # Compute squared error
    sq_error = (pred - target) ** 2  # [batch, T, 2]

    # Mask and average
    masked_sq_error = sq_error * mask
    num_valid = mask.sum()

    if num_valid > 0:
        loss = masked_sq_error.sum() / num_valid
    else:
        loss = torch.tensor(0.0, device=pred.device)

    return loss
