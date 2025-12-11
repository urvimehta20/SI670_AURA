"""
Ensemble model that combines predictions from multiple trained models.
Trains a meta-learner on top of individual model predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib

from lib.data import stratified_kfold
# Lazy import to avoid circular dependency issues
# VideoConfig and VideoDataset will be imported when needed
from lib.training.model_factory import create_model, is_pytorch_model, get_model_config, list_available_models
from lib.training.trainer import OptimConfig, TrainConfig, fit, evaluate
from lib.training._linear import LogisticRegressionBaseline
from lib.training._svm import SVMBaseline
from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


class EnsembleMetaLearner(nn.Module):
    """
    Simple MLP meta-learner that learns to combine predictions from multiple models.
    """
    
    def __init__(self, num_models: int, hidden_dim: int = 64, dropout: float = 0.3):
        """
        Initialize meta-learner.
        
        Args:
            num_models: Number of base models
            hidden_dim: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Input: probabilities from each model (num_models * 2 for binary classification)
        self.fc1 = nn.Linear(num_models * 2, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_dim // 2, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x


class PredictionDataset(Dataset):
    """Dataset that loads predictions from multiple models."""
    
    def __init__(self, predictions: Dict[str, np.ndarray], labels: np.ndarray):
        """
        Args:
            predictions: Dict mapping model_name -> (n_samples, 2) probability array
            labels: (n_samples,) label array
        """
        self.labels = torch.from_numpy(labels).long()
        
        # Stack all predictions: (n_samples, num_models * 2)
        pred_list = []
        for model_name in sorted(predictions.keys()):
            pred_list.append(torch.from_numpy(predictions[model_name]).float())
        
        self.predictions = torch.cat(pred_list, dim=1)  # (n_samples, num_models * 2)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.predictions[idx], self.labels[idx]


def load_trained_model(
    model_type: str,
    fold_dir: Path,
    project_root: str,
    model_config: Dict
) -> Optional[object]:
    """
    Load a trained model from disk.
    
    Args:
        model_type: Model type identifier
        fold_dir: Directory containing saved model
        project_root: Project root directory
        model_config: Model configuration
    
    Returns:
        Loaded model or None if not found
    """
    try:
        if is_pytorch_model(model_type):
            # PyTorch model
            model = create_model(model_type, model_config)
            
            # Look for checkpoint
            checkpoint_path = fold_dir / "checkpoint.pt"
            if not checkpoint_path.exists():
                checkpoint_path = fold_dir / "model.pt"
            
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                logger.info(f"Loaded {model_type} from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found for {model_type} in {fold_dir}")
                return None
            
            model.eval()
            return model
        else:
            # Sklearn model
            if model_type == "logistic_regression":
                model = LogisticRegressionBaseline()
            elif model_type == "svm":
                model = SVMBaseline()
            else:
                return None
            
            model.load(str(fold_dir))
            return model
    except Exception as e:
        logger.error(f"Error loading {model_type} from {fold_dir}: {e}")
        return None


def get_predictions_from_model(
    model: object,
    model_type: str,
    dataset,  # VideoDataset - imported lazily to avoid circular dependency
    device: str,
    project_root: str
) -> np.ndarray:
    """
    Get predictions from a loaded model.
    
    Args:
        model: Loaded model
        model_type: Model type identifier
        dataset: Video dataset
        device: Device to run on
        project_root: Project root directory
    
    Returns:
        Predictions as probabilities (n_samples, 2)
    """
    if is_pytorch_model(model_type):
        # PyTorch model
        model.eval()
        model = model.to(device)
        
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        
        all_probs = []
        with torch.no_grad():
            for clips, _ in loader:
                clips = clips.to(device)
                
                if device.startswith("cuda"):
                    try:
                        with torch.amp.autocast('cuda'):
                            logits = model(clips)
                    except (AttributeError, TypeError):
                        with torch.cuda.amp.autocast():
                            logits = model(clips)
                else:
                    logits = model(clips)
                
                # Convert to probabilities
                if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                    if logits.ndim == 2:
                        logits = logits.squeeze(-1)
                    probs_positive = torch.sigmoid(logits).cpu().numpy()
                    probs = np.column_stack([1 - probs_positive, probs_positive])
                else:
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                
                all_probs.append(probs)
        
        return np.vstack(all_probs)
    else:
        # Sklearn model
        df = pl.DataFrame({
            "video_path": [dataset.video_paths[i] for i in range(len(dataset))]
        })
        probs = model.predict(df, project_root)
        return probs


def train_ensemble_model(
    project_root: str,
    scaled_metadata_path: str,
    base_model_types: List[str],
    base_models_dir: str = "data/training_results",
    n_splits: int = 5,
    num_frames: int = 1000,
    output_dir: str = "data/training_results",
    ensemble_method: str = "meta_learner",  # "meta_learner" or "weighted_average"
    hidden_dim: int = 64
) -> Dict:
    """
    Train an ensemble model using predictions from trained base models.
    
    This function:
    1. Loads all trained base models for each fold
    2. Gets predictions from each model on train/val sets
    3. Trains a meta-learner (or uses weighted averaging) to combine predictions
    4. Saves the ensemble model
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled metadata
        base_model_types: List of base model types to ensemble
        base_models_dir: Directory containing trained base models
        n_splits: Number of k-fold splits (must match base model training)
        num_frames: Number of frames per video
        output_dir: Directory to save ensemble results
        ensemble_method: "meta_learner" or "weighted_average"
        hidden_dim: Hidden dimension for meta-learner
    
    Returns:
        Dictionary of ensemble results
    """
    project_root = Path(project_root)
    base_models_dir = project_root / base_models_dir
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    logger.info("Loading metadata for ensemble training...")
    from lib.utils.paths import load_metadata_flexible
    
    scaled_df = load_metadata_flexible(scaled_metadata_path)
    if scaled_df is None:
        raise FileNotFoundError(f"Scaled metadata not found: {scaled_metadata_path}")
    logger.info(f"Found {scaled_df.height} videos")
    
    # Create video config
    # Lazy import to avoid circular dependency
    from lib.models import VideoConfig
    video_config = VideoConfig(num_frames=num_frames, fixed_size=256)
    
    # Get all folds
    all_folds = stratified_kfold(scaled_df, n_splits=n_splits, random_state=42)
    
    ensemble_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fold_idx in range(n_splits):
        logger.info(f"\n{'='*80}")
        logger.info(f"Ensemble Training - Fold {fold_idx + 1}/{n_splits}")
        logger.info(f"{'='*80}")
        
        train_df, val_df = all_folds[fold_idx]
        
        # Create datasets
        # Lazy import to avoid circular dependency
        from lib.models import VideoDataset
        train_dataset = VideoDataset(train_df, project_root=str(project_root), config=video_config)
        val_dataset = VideoDataset(val_df, project_root=str(project_root), config=video_config)
        
        # Get labels
        train_labels = train_df["label"].to_numpy()
        val_labels = val_df["label"].to_numpy()
        
        # Collect predictions from all base models
        train_predictions = {}
        val_predictions = {}
        
        for model_type in base_model_types:
            model_dir = base_models_dir / model_type / f"fold_{fold_idx + 1}"
            
            if not model_dir.exists():
                logger.warning(f"Model {model_type} fold {fold_idx + 1} not found, skipping")
                continue
            
            model_config = get_model_config(model_type)
            model_config["num_frames"] = num_frames
            
            # Load model
            model = load_trained_model(model_type, model_dir, str(project_root), model_config)
            if model is None:
                continue
            
            logger.info(f"Getting predictions from {model_type}...")
            
            # Get predictions
            try:
                train_preds = get_predictions_from_model(
                    model, model_type, train_dataset, str(device), str(project_root)
                )
                val_preds = get_predictions_from_model(
                    model, model_type, val_dataset, str(device), str(project_root)
                )
                
                train_predictions[model_type] = train_preds
                val_predictions[model_type] = val_preds
                
                logger.info(f"✓ {model_type}: train={train_preds.shape}, val={val_preds.shape}")
            except Exception as e:
                logger.error(f"Error getting predictions from {model_type}: {e}", exc_info=True)
                continue
            
            # Clean up
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            aggressive_gc(clear_cuda=False)
        
        if len(train_predictions) == 0:
            logger.error(f"No predictions collected for fold {fold_idx + 1}")
            continue
        
        logger.info(f"Collected predictions from {len(train_predictions)} models")
        
        # Train ensemble
        if ensemble_method == "meta_learner":
            # Create meta-learner
            num_models = len(train_predictions)
            meta_model = EnsembleMetaLearner(num_models=num_models, hidden_dim=hidden_dim).to(device)
            
            # Create datasets
            train_pred_dataset = PredictionDataset(train_predictions, train_labels)
            val_pred_dataset = PredictionDataset(val_predictions, val_labels)
            
            train_loader = DataLoader(train_pred_dataset, batch_size=32, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_pred_dataset, batch_size=32, shuffle=False, num_workers=0)
            
            # Train
            optim_cfg = OptimConfig(lr=1e-3, weight_decay=1e-4)
            train_cfg = TrainConfig(
                num_epochs=50,
                device=str(device),
                use_amp=True,
                gradient_accumulation_steps=1,
                early_stopping_patience=10
            )
            
            meta_model = fit(meta_model, train_loader, val_loader, optim_cfg, train_cfg)
            
            # Evaluate
            val_loss, val_acc = evaluate(meta_model, val_loader, device=str(device))
            
            # Save ensemble model
            ensemble_dir = output_dir / "ensemble" / f"fold_{fold_idx + 1}"
            ensemble_dir.mkdir(parents=True, exist_ok=True)
            torch.save(meta_model.state_dict(), ensemble_dir / "meta_learner.pt")
            
            # Save which models were used
            with open(ensemble_dir / "base_models.txt", "w") as f:
                f.write("\n".join(sorted(train_predictions.keys())))
            
        elif ensemble_method == "weighted_average":
            # Simple weighted average (learn weights on validation set)
            # This is a simpler approach - just average probabilities
            train_avg = np.mean([train_predictions[m] for m in sorted(train_predictions.keys())], axis=0)
            val_avg = np.mean([val_predictions[m] for m in sorted(val_predictions.keys())], axis=0)
            
            # Evaluate
            val_preds = np.argmax(val_avg, axis=1)
            val_acc = float(np.mean(val_preds == val_labels))
            val_loss = float('nan')  # No loss for simple averaging
            
            # Save ensemble info
            ensemble_dir = output_dir / "ensemble" / f"fold_{fold_idx + 1}"
            ensemble_dir.mkdir(parents=True, exist_ok=True)
            with open(ensemble_dir / "base_models.txt", "w") as f:
                f.write("\n".join(sorted(train_predictions.keys())))
            np.save(ensemble_dir / "weights.npy", np.ones(len(train_predictions)) / len(train_predictions))
        
        ensemble_results.append({
            "fold": fold_idx + 1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "num_base_models": len(train_predictions),
            "base_models": ",".join(sorted(train_predictions.keys()))
        })
        
        logger.info(f"Fold {fold_idx + 1} - Val Acc: {val_acc:.4f}")
        
        # Clean up
        if ensemble_method == "meta_learner":
            del meta_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        aggressive_gc(clear_cuda=False)
    
    # Save results
    if ensemble_results:
        results_df = pl.DataFrame(ensemble_results)
        results_path = output_dir / "ensemble" / "fold_results.csv"
        results_df.write_csv(results_path)
        
        avg_acc = float(results_df["val_acc"].mean())
        logger.info(f"\nEnsemble - Avg Val Acc: {avg_acc:.4f}")
    
    return {"fold_results": ensemble_results}


__all__ = ["train_ensemble_model", "EnsembleMetaLearner"]

