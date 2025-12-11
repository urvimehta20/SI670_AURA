"""
MLOps Core: Experiment tracking, versioning, and workflow orchestration.

This module provides:
- Experiment run tracking with unique IDs
- Configuration versioning and hashing
- Metrics logging and persistence
- Checkpoint management with full state saving
- Resume capability for interrupted training
- Data versioning and lineage tracking
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import polars as pl
import torch

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Complete configuration for a training run."""
    # Experiment metadata
    run_id: str
    experiment_name: str
    description: Optional[str] = None
    tags: List[str] = None
    
    # Data config
    data_csv: str = ""
    train_split: float = 0.8
    val_split: float = 0.2
    test_split: float = 0.0
    random_seed: int = 42
    
    # Video config
    num_frames: int = 1000
    fixed_size: Optional[int] = 256  # Match scaled video dimensions
    augmentation_config: Optional[Dict[str, Any]] = None
    temporal_augmentation_config: Optional[Dict[str, Any]] = None
    num_augmentations_per_video: int = 3
    
    # Training config
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    gradient_accumulation_steps: int = 1
    early_stopping_patience: int = 5
    
    # System config
    device: str = "cuda"
    num_workers: int = 4
    use_amp: bool = True
    
    # Model config
    model_type: str = "pretrained_inception"  # Model type identifier
    model_specific_config: Dict[str, Any] = None  # Model-specific hyperparameters
    
    # Paths
    project_root: str = ""
    output_dir: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.model_specific_config is None:
            self.model_specific_config = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RunConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RunConfig':
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def compute_hash(self) -> str:
        """Compute deterministic hash of configuration (excluding run_id and paths)."""
        # Create a copy without run_id and paths for hashing
        config_dict = self.to_dict()
        config_dict.pop('run_id', None)
        config_dict.pop('project_root', None)
        config_dict.pop('output_dir', None)
        
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


class ExperimentTracker:
    """Tracks experiments, metrics, and artifacts."""
    
    def __init__(self, run_dir: str, run_id: Optional[str] = None):
        """
        Initialize experiment tracker.
        
        Args:
            run_dir: Directory for this run's artifacts
            run_id: Unique run ID (generated if None)
        """
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        if run_id is None:
            run_id = self._generate_run_id()
        self.run_id = run_id
        
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.config_file = self.run_dir / "config.json"
        self.metadata_file = self.run_dir / "metadata.json"
        
        # Initialize metrics file
        if not self.metrics_file.exists():
            self.metrics_file.touch()
    
    @staticmethod
    def _generate_run_id() -> str:
        """Generate unique run ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(4).hex()
        return f"run_{timestamp}_{random_suffix}"
    
    def log_config(self, config: RunConfig) -> None:
        """Save run configuration."""
        config_dict = config.to_dict()
        config_dict['run_id'] = self.run_id
        
        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("Saved run config to %s", self.config_file)
    
    def log_metadata(self, metadata: Dict[str, Any]) -> None:
        """Log additional metadata (system info, git commit, etc.)."""
        metadata['run_id'] = self.run_id
        metadata['timestamp'] = datetime.now().isoformat()
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def log_metric(self, step: int, metric_name: str, value: float, 
                   epoch: Optional[int] = None, phase: str = "train") -> None:
        """
        Log a metric value.
        
        Args:
            step: Training step (batch number)
            metric_name: Name of metric (e.g., "loss", "accuracy")
            value: Metric value
            epoch: Epoch number (optional)
            phase: Phase (train/val/test)
        """
        metric_entry = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "phase": phase,
            "metric": metric_name,
            "value": float(value),
        }
        
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metric_entry) + '\n')
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = "train") -> None:
        """Log multiple metrics for an epoch."""
        for metric_name, value in metrics.items():
            self.log_metric(step=epoch, metric_name=metric_name, value=value, 
                          epoch=epoch, phase=phase)
    
    def get_metrics(self) -> pl.DataFrame:
        """Load all metrics as a Polars DataFrame."""
        if not self.metrics_file.exists():
            return pl.DataFrame()
        
        try:
            # Read JSONL file
            metrics_list = []
            with open(self.metrics_file, 'r') as f:
                for line in f:
                    if line.strip():
                        metrics_list.append(json.loads(line))
            
            if not metrics_list:
                return pl.DataFrame()
            
            return pl.DataFrame(metrics_list)
        except Exception as e:
            logger.error("Failed to load metrics: %s", e)
            return pl.DataFrame()
    
    def get_best_metric(self, metric_name: str, phase: str = "val", 
                       maximize: bool = True) -> Optional[Dict[str, Any]]:
        """Get best value for a metric."""
        df = self.get_metrics()
        if df.height == 0:
            return None
        
        filtered = df.filter(
            (pl.col("metric") == metric_name) & (pl.col("phase") == phase)
        )
        
        if filtered.height == 0:
            return None
        
        if maximize:
            best = filtered.sort("value", descending=True).row(0, named=True)
        else:
            best = filtered.sort("value", descending=False).row(0, named=True)
        
        return best


class CheckpointManager:
    """Enhanced checkpoint manager with full state saving and resume capability."""
    
    def __init__(self, checkpoint_dir: str, run_id: str):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            run_id: Run ID for this experiment
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        
        self.best_metric: Optional[float] = None
        self.best_epoch: int = 0
        self.stage_checkpoints: Dict[str, str] = {}  # Track stage checkpoints
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[Any], epoch: int, metrics: Dict[str, float],
                       is_best: bool = False, prefix: str = "checkpoint") -> str:
        """
        Save full training state checkpoint.
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'run_id': self.run_id,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"{prefix}_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.best_metric = metrics.get('val_accuracy', metrics.get('val_loss', 0.0))
            self.best_epoch = epoch
            logger.info("Saved best model checkpoint (epoch %d, metric=%.4f)", epoch, self.best_metric)
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """
        Load checkpoint and restore training state.
        
        Returns:
            Dictionary with checkpoint info (epoch, metrics, etc.)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and checkpoint.get('optimizer_state_dict'):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and checkpoint.get('scheduler_state_dict'):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info("Loaded checkpoint from epoch %d (run_id: %s)", 
                   checkpoint['epoch'], checkpoint.get('run_id', 'unknown'))
        
        return checkpoint
    
    def resume_from_latest(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                          scheduler: Optional[Any] = None) -> Optional[int]:
        """
        Resume training from latest checkpoint.
        
        Returns:
            Starting epoch number (None if no checkpoint found)
        """
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        
        if not latest_path.exists():
            return None
        
        checkpoint = self.load_checkpoint(str(latest_path), model, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        
        logger.info("Resuming training from epoch %d", start_epoch)
        return start_epoch
    
    def save_stage_checkpoint(self, stage_name: str, data: Dict[str, Any]) -> str:
        """
        Save a checkpoint for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            data: Data to checkpoint (must be serializable)
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"stage_{stage_name}.pt"
        
        checkpoint = {
            'stage_name': stage_name,
            'run_id': self.run_id,
            'data': data,
            'timestamp': datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.stage_checkpoints[stage_name] = str(checkpoint_path)
        logger.info("Saved stage checkpoint: %s -> %s", stage_name, checkpoint_path)
        
        return str(checkpoint_path)
    
    def load_stage_checkpoint(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
        
        Returns:
            Checkpoint data (None if not found)
        """
        checkpoint_path = self.checkpoint_dir / f"stage_{stage_name}.pt"
        
        if not checkpoint_path.exists():
            return None
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info("Loaded stage checkpoint: %s", stage_name)
        
        return checkpoint.get('data')


class DataVersionManager:
    """Manages data versioning and lineage tracking."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.version_file = self.data_dir / "data_versions.json"
        self.versions: Dict[str, Any] = {}
        
        if self.version_file.exists():
            with open(self.version_file, 'r') as f:
                self.versions = json.load(f)
    
    def register_split(self, split_name: str, config_hash: str, 
                      file_path: str, metadata: Dict[str, Any]) -> None:
        """Register a data split version."""
        if split_name not in self.versions:
            self.versions[split_name] = []
        
        version_entry = {
            "config_hash": config_hash,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata,
        }
        
        self.versions[split_name].append(version_entry)
        self._save()
    
    def register_augmentation(self, config_hash: str, output_dir: str, 
                            metadata: Dict[str, Any]) -> None:
        """Register augmented data version."""
        self.register_split("augmented", config_hash, output_dir, metadata)
    
    def _save(self) -> None:
        """Save version registry."""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2)


def create_run_directory(base_dir: str, experiment_name: str, 
                         run_id: Optional[str] = None) -> tuple[str, str]:
    """
    Create a new run directory with proper structure.
    
    Returns:
        (run_dir, run_id) tuple
    """
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_suffix = os.urandom(4).hex()
        run_id = f"run_{timestamp}_{random_suffix}"
    
    run_dir = Path(base_dir) / "runs" / experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    
    return str(run_dir), run_id


__all__ = [
    "RunConfig",
    "ExperimentTracker",
    "CheckpointManager",
    "DataVersionManager",
    "create_run_directory",
]

