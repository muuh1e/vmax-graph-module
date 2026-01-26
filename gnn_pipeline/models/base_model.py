"""Base model class for motion prediction."""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData


class BaseMotionPredictor(nn.Module, ABC):
    """
    Abstract base class for motion prediction models.
    
    All motion prediction models should inherit from this class
    and implement the forward method.
    
    Attributes:
        num_future_steps: Number of future timesteps to predict
    """
    
    def __init__(self, num_future_steps: int = 80):
        """
        Initialize the base predictor.
        
        Args:
            num_future_steps: Number of future timesteps to predict
        """
        super().__init__()
        self.num_future_steps = num_future_steps
    
    @abstractmethod
    def forward(
        self,
        data: HeteroData,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for motion prediction.
        
        Args:
            data: HeteroData batch with node features and edge indices
            ego_indices: Optional tensor of ego agent indices in the batch.
                        If None, will be inferred from data.ego_indices_global
        
        Returns:
            Predicted future trajectories of shape [batch_size, num_future_steps, 2]
        """
        pass
    
    @torch.no_grad()
    def predict(
        self,
        data: HeteroData,
        ego_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Inference wrapper that sets model to eval mode.
        
        Args:
            data: HeteroData batch with node features and edge indices
            ego_indices: Optional tensor of ego agent indices
            
        Returns:
            Predicted future trajectories of shape [batch_size, num_future_steps, 2]
        """
        was_training = self.training
        self.eval()
        pred = self.forward(data, ego_indices)
        if was_training:
            self.train()
        return pred
    
    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
