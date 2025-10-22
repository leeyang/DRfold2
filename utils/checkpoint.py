"""
Checkpoint utilities for DRfold2 training
Handles saving and loading model checkpoints
"""

import os
import torch
import json
from pathlib import Path
from typing import Dict, Optional, Any
import glob


class CheckpointManager:
    """
    Manages model checkpoints during training.
    Handles saving, loading, and maintaining checkpoint history.
    """

    def __init__(self, checkpoint_dir: str, keep_last_n: int = 5):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n

        self.best_metric = float('inf')
        self.best_checkpoint_path = None

    def save_checkpoint(self,
                       epoch: int,
                       step: int,
                       model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any] = None,
                       metrics: Optional[Dict] = None,
                       is_best: bool = False,
                       config: Optional[Dict] = None) -> str:
        """
        Save a training checkpoint.

        Args:
            epoch: Current epoch
            step: Current training step
            model: Model to save
            optimizer: Optimizer to save
            scheduler: Learning rate scheduler (optional)
            metrics: Dictionary of metrics (optional)
            is_best: Whether this is the best model so far
            config: Configuration dictionary (optional)

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {},
        }

        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()

        if config is not None:
            checkpoint['config'] = config

        # Regular checkpoint filename
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch{epoch}_step{step}.pt'
        torch.save(checkpoint, checkpoint_path)

        print(f"Saved checkpoint: {checkpoint_path}")

        # Save as best model if applicable
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path
            print(f"Saved best model: {best_path}")

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'
        torch.save(checkpoint, latest_path)

        # Clean up old checkpoints
        self._cleanup_checkpoints()

        return str(checkpoint_path)

    def load_checkpoint(self,
                       checkpoint_path: str,
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       load_optimizer: bool = True,
                       load_scheduler: bool = True,
                       device: str = 'cuda') -> Dict:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Model to load weights into
            optimizer: Optimizer to load state into (optional)
            scheduler: Scheduler to load state into (optional)
            load_optimizer: Whether to load optimizer state
            load_scheduler: Whether to load scheduler state
            device: Device to load checkpoint on

        Returns:
            Dictionary containing checkpoint info
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model state")

        # Load optimizer state if requested
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state")

        # Load scheduler state if requested
        if load_scheduler and scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state")

        return {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', None)
        }

    def load_latest_checkpoint(self,
                              model: torch.nn.Module,
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              scheduler: Optional[Any] = None,
                              load_optimizer: bool = True,
                              load_scheduler: bool = True,
                              device: str = 'cuda') -> Optional[Dict]:
        """
        Load the latest checkpoint if it exists.

        Returns:
            Checkpoint info dict if checkpoint exists, None otherwise
        """
        latest_path = self.checkpoint_dir / 'latest_checkpoint.pt'

        if not latest_path.exists():
            print("No latest checkpoint found")
            return None

        return self.load_checkpoint(
            str(latest_path),
            model,
            optimizer,
            scheduler,
            load_optimizer,
            load_scheduler,
            device
        )

    def load_best_checkpoint(self,
                            model: torch.nn.Module,
                            device: str = 'cuda') -> Optional[Dict]:
        """
        Load the best checkpoint if it exists.

        Returns:
            Checkpoint info dict if checkpoint exists, None otherwise
        """
        best_path = self.checkpoint_dir / 'best_model.pt'

        if not best_path.exists():
            print("No best checkpoint found")
            return None

        return self.load_checkpoint(
            str(best_path),
            model,
            optimizer=None,
            scheduler=None,
            load_optimizer=False,
            load_scheduler=False,
            device=device
        )

    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the last N"""
        if self.keep_last_n <= 0:
            return

        # Get all checkpoint files (excluding best and latest)
        checkpoint_files = list(self.checkpoint_dir.glob('checkpoint_epoch*_step*.pt'))

        # Sort by modification time
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Remove old checkpoints
        for checkpoint_file in checkpoint_files[self.keep_last_n:]:
            try:
                checkpoint_file.unlink()
                print(f"Removed old checkpoint: {checkpoint_file}")
            except Exception as e:
                print(f"Failed to remove checkpoint {checkpoint_file}: {e}")

    def update_best_metric(self, metric_value: float, lower_is_better: bool = True) -> bool:
        """
        Check if current metric is better than best so far.

        Args:
            metric_value: Current metric value
            lower_is_better: Whether lower values are better

        Returns:
            True if this is the best metric so far
        """
        if lower_is_better:
            is_best = metric_value < self.best_metric
        else:
            is_best = metric_value > self.best_metric

        if is_best:
            self.best_metric = metric_value

        return is_best


class MetricsLogger:
    """
    Logger for training metrics.
    Saves metrics to JSON file for later analysis.
    """

    def __init__(self, log_dir: str, filename: str = 'metrics.json'):
        """
        Args:
            log_dir: Directory to save logs
            filename: Name of log file
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename

        self.metrics = []

        # Load existing metrics if file exists
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.metrics = json.load(f)

    def log(self, epoch: int, step: int, metrics: Dict, phase: str = 'train'):
        """
        Log metrics for a training step.

        Args:
            epoch: Current epoch
            step: Current step
            metrics: Dictionary of metrics
            phase: 'train' or 'val'
        """
        log_entry = {
            'epoch': epoch,
            'step': step,
            'phase': phase,
            **metrics
        }

        self.metrics.append(log_entry)

        # Save to file
        with open(self.log_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def get_metrics(self, phase: Optional[str] = None) -> list:
        """
        Get all logged metrics, optionally filtered by phase.

        Args:
            phase: Optional phase filter ('train' or 'val')

        Returns:
            List of metric dictionaries
        """
        if phase is None:
            return self.metrics

        return [m for m in self.metrics if m.get('phase') == phase]

    def get_best_metric(self, metric_name: str, phase: str = 'val',
                       lower_is_better: bool = True) -> Optional[Dict]:
        """
        Get the entry with the best value for a specific metric.

        Args:
            metric_name: Name of metric to optimize
            phase: Phase to filter by
            lower_is_better: Whether lower is better

        Returns:
            Best metric entry or None
        """
        phase_metrics = self.get_metrics(phase)

        if not phase_metrics:
            return None

        # Filter entries that have this metric
        valid_entries = [m for m in phase_metrics if metric_name in m]

        if not valid_entries:
            return None

        # Find best
        if lower_is_better:
            return min(valid_entries, key=lambda x: x[metric_name])
        else:
            return max(valid_entries, key=lambda x: x[metric_name])


def save_config(config: Dict, save_path: str):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {save_path}")


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded config from {config_path}")
    return config
