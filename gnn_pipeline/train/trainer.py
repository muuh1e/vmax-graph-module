"""
Trainer class for GNN motion prediction.

Provides a clean interface for training, validation, and checkpointing.
"""

import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots

from .metrics import compute_loss, compute_metrics


def custom_collate(data_list):
    """Custom collate function that computes ego_indices_global."""
    batch = Batch.from_data_list(data_list)

    # Get cumulative agent node counts
    ptr = batch['agent'].ptr  # [batch_size + 1]

    # Offset ego indices by cumulative node count
    ego_indices = batch.ego_idx_tensor.squeeze(-1)  # [batch_size]
    ego_indices_global = ego_indices + ptr[:-1]

    batch.ego_indices_global = ego_indices_global

    return batch


class Trainer:
    """
    Trainer for GNN motion prediction models.

    Handles training loop, validation, checkpointing, and metrics tracking.

    Args:
        model: The model to train (should inherit from BaseMotionPredictor)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to train on
        lr: Learning rate
        checkpoint_dir: Directory to save checkpoints
        plots_dir: Directory to save training plots
        model_name: Name of the model (for plot titles and filenames)
        scheduler_patience: Patience for ReduceLROnPlateau scheduler
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 1e-3,
        checkpoint_dir: str = "./checkpoints",
        plots_dir: str = "./plots",
        model_name: str = "model",
        scheduler_patience: int = 5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.plots_dir = plots_dir
        self.model_name = model_name
        self.initial_lr = lr

        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Optimizer and scheduler
        self.optimizer = Adam(model.parameters(), lr=lr)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=scheduler_patience
        )

        # Tracking
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_ade': [], 'val_ade': [],
            'train_fde': [], 'val_fde': [],
            'learning_rate': [],
        }
        self.best_val_ade = float('inf')
        self.best_val_fde = float('inf')
        self.best_epoch = 0

        # Timestamp for unique filenames
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def train_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc="Train", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred = self.model(batch)  # [batch, 80, 2]

            # Get targets
            target = batch.ego_future_target.view(-1, 80, 2)  # [batch, 80, 2]
            valid_mask = batch.ego_future_valid.view(-1, 80)  # [batch, 80]

            # Compute loss
            loss = compute_loss(pred, target, valid_mask)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(pred, target, valid_mask)

            total_loss += loss.item()
            total_ade += metrics['ade']
            total_fde += metrics['fde']
            num_batches += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ade': f"{metrics['ade']:.2f}",
                'fde': f"{metrics['fde']:.2f}"
            })

        avg_loss = total_loss / num_batches
        avg_metrics = {
            'ade': total_ade / num_batches,
            'fde': total_fde / num_batches,
        }

        return avg_loss, avg_metrics

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()

        total_loss = 0.0
        total_ade = 0.0
        total_fde = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc="Val", leave=False)
        for batch in pbar:
            batch = batch.to(self.device)

            # Forward pass
            pred = self.model(batch)

            # Get targets
            target = batch.ego_future_target.view(-1, 80, 2)
            valid_mask = batch.ego_future_valid.view(-1, 80)

            # Compute loss and metrics
            loss = compute_loss(pred, target, valid_mask)
            metrics = compute_metrics(pred, target, valid_mask)

            total_loss += loss.item()
            total_ade += metrics['ade']
            total_fde += metrics['fde']
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_metrics = {
            'ade': total_ade / num_batches,
            'fde': total_fde / num_batches,
        }

        return avg_loss, avg_metrics

    def train(self, epochs: int, verbose: bool = True) -> Dict:
        """
        Run full training loop.

        Args:
            epochs: Number of epochs to train
            verbose: Whether to print progress

        Returns:
            Dictionary with training history and best metrics
        """
        if verbose:
            print("\n" + "=" * 60)
            print("Starting training...")
            print("=" * 60)

        for epoch in range(1, epochs + 1):
            if verbose:
                print(f"\nEpoch {epoch}/{epochs}")
                print("-" * 40)

            # Train
            train_loss, train_metrics = self.train_epoch()

            # Validate
            val_loss, val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_ade'].append(train_metrics['ade'])
            self.history['val_ade'].append(val_metrics['ade'])
            self.history['train_fde'].append(train_metrics['fde'])
            self.history['val_fde'].append(val_metrics['fde'])
            self.history['learning_rate'].append(current_lr)

            if verbose:
                print(f"Train Loss: {train_loss:.4f} | ADE: {train_metrics['ade']:.2f}m | FDE: {train_metrics['fde']:.2f}m")
                print(f"Val   Loss: {val_loss:.4f} | ADE: {val_metrics['ade']:.2f}m | FDE: {val_metrics['fde']:.2f}m")

            # Save best model (based on ADE)
            if val_metrics['ade'] < self.best_val_ade:
                self.best_val_ade = val_metrics['ade']
                self.best_val_fde = val_metrics['fde']
                self.best_epoch = epoch

                self._save_checkpoint(epoch, val_metrics, val_loss, is_best=True)
                if verbose:
                    print(f"  -> Saved best model (ADE: {self.best_val_ade:.2f}m)")

        # Final summary
        if verbose:
            print("\n" + "=" * 60)
            print("=== Training Complete ===")
            print("=" * 60)
            print(f"Best Epoch: {self.best_epoch}")
            print(f"Best Val ADE: {self.best_val_ade:.2f}m")
            print(f"Best Val FDE: {self.best_val_fde:.2f}m")

        # Generate and save plots
        self.plot_training_curves()
        self.save_training_history()

        return {
            'history': self.history,
            'best_epoch': self.best_epoch,
            'best_val_ade': self.best_val_ade,
            'best_val_fde': self.best_val_fde,
        }

    def plot_training_curves(self) -> None:
        """Generate and save training visualization plots."""
        epochs = range(1, len(self.history['train_loss']) + 1)

        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f'Training Curves: {self.model_name}\n'
            f'Best Val ADE: {self.best_val_ade:.2f}m @ Epoch {self.best_epoch}',
            fontsize=14, fontweight='bold'
        )

        # Plot 1: Loss curves
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch})')
        ax1.set_xlabel('Epoch', fontsize=11)
        ax1.set_ylabel('Loss (MSE)', fontsize=11)
        ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1, len(epochs))

        # Plot 2: ADE curves
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_ade'], 'b-', label='Train ADE', linewidth=2)
        ax2.plot(epochs, self.history['val_ade'], 'r-', label='Val ADE', linewidth=2)
        ax2.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch})')
        ax2.axhline(y=self.best_val_ade, color='orange', linestyle=':', alpha=0.7, label=f'Best Val ADE ({self.best_val_ade:.2f}m)')
        ax2.set_xlabel('Epoch', fontsize=11)
        ax2.set_ylabel('ADE (meters)', fontsize=11)
        ax2.set_title('Average Displacement Error', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(1, len(epochs))

        # Plot 3: FDE curves
        ax3 = axes[1, 0]
        ax3.plot(epochs, self.history['train_fde'], 'b-', label='Train FDE', linewidth=2)
        ax3.plot(epochs, self.history['val_fde'], 'r-', label='Val FDE', linewidth=2)
        ax3.axvline(x=self.best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best Epoch ({self.best_epoch})')
        ax3.axhline(y=self.best_val_fde, color='orange', linestyle=':', alpha=0.7, label=f'Best Val FDE ({self.best_val_fde:.2f}m)')
        ax3.set_xlabel('Epoch', fontsize=11)
        ax3.set_ylabel('FDE (meters)', fontsize=11)
        ax3.set_title('Final Displacement Error', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(1, len(epochs))

        # Plot 4: Learning Rate
        ax4 = axes[1, 1]
        ax4.plot(epochs, self.history['learning_rate'], 'purple', linewidth=2, marker='o', markersize=3)
        ax4.set_xlabel('Epoch', fontsize=11)
        ax4.set_ylabel('Learning Rate', fontsize=11)
        ax4.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(1, len(epochs))

        plt.tight_layout()

        # Save plot with descriptive filename
        plot_filename = f"{self.model_name}_{self.timestamp}_training_curves.png"
        plot_path = os.path.join(self.plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\nTraining curves saved to: {plot_path}")

        # Also create individual metric plots for detailed analysis
        self._plot_individual_metrics()

    def _plot_individual_metrics(self) -> None:
        """Create individual plots for each metric."""
        epochs = range(1, len(self.history['train_loss']) + 1)

        # ADE comparison plot (larger, more detailed)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, self.history['train_ade'], 'b-', label='Train ADE', linewidth=2, alpha=0.8)
        ax.plot(epochs, self.history['val_ade'], 'r-', label='Val ADE', linewidth=2)
        ax.fill_between(epochs, self.history['train_ade'], self.history['val_ade'], alpha=0.2, color='gray')
        ax.axvline(x=self.best_epoch, color='g', linestyle='--', linewidth=2, label=f'Best Epoch ({self.best_epoch})')
        ax.axhline(y=self.best_val_ade, color='orange', linestyle=':', linewidth=2, label=f'Best: {self.best_val_ade:.2f}m')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Average Displacement Error (meters)', fontsize=12)
        ax.set_title(f'{self.model_name} - ADE Over Training', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(epochs))

        # Add text annotation for final values
        final_train = self.history['train_ade'][-1]
        final_val = self.history['val_ade'][-1]
        textstr = f'Final Train: {final_train:.2f}m\nFinal Val: {final_val:.2f}m\nBest Val: {self.best_val_ade:.2f}m'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f"{self.model_name}_{self.timestamp}_ade.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Train vs Val gap plot (to detect overfitting)
        fig, ax = plt.subplots(figsize=(10, 6))
        ade_gap = [t - v for t, v in zip(self.history['train_ade'], self.history['val_ade'])]
        fde_gap = [t - v for t, v in zip(self.history['train_fde'], self.history['val_fde'])]

        ax.plot(epochs, ade_gap, 'b-', label='ADE Gap (Train - Val)', linewidth=2)
        ax.plot(epochs, fde_gap, 'r-', label='FDE Gap (Train - Val)', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.fill_between(epochs, 0, ade_gap, alpha=0.3, color='blue')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Gap (meters)', fontsize=12)
        ax.set_title(f'{self.model_name} - Generalization Gap (negative = overfitting)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, len(epochs))

        plt.tight_layout()
        plot_path = os.path.join(self.plots_dir, f"{self.model_name}_{self.timestamp}_gap.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

    def save_training_history(self) -> None:
        """Save training history to JSON file."""
        history_data = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'best_epoch': self.best_epoch,
            'best_val_ade': self.best_val_ade,
            'best_val_fde': self.best_val_fde,
            'initial_lr': self.initial_lr,
            'total_epochs': len(self.history['train_loss']),
            'history': self.history,
        }

        history_filename = f"{self.model_name}_{self.timestamp}_history.json"
        history_path = os.path.join(self.plots_dir, history_filename)

        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        print(f"Training history saved to: {history_path}")

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: Dict[str, float],
        val_loss: float,
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_ade': metrics['ade'],
            'val_fde': metrics['fde'],
            'val_loss': val_loss,
        }

        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')

        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load a checkpoint."""
        checkpoint = torch.load(path, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint

    def load_best_model(self) -> Dict:
        """Load the best model checkpoint."""
        path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        return self.load_checkpoint(path)
