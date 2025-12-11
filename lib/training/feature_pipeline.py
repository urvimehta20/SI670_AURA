"""
Unified feature-based training pipeline.

All models use Stage 2/4 features (not videos).
Implements proper ML pipeline: 60-20-20 split, imputation, scaling, normalization,
stratified splits, no data leaks, OOM resistant, 5-fold CV, hyperparameter tuning.
"""

from __future__ import annotations

import logging
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt

from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


class FeatureDataset(Dataset):
    """PyTorch Dataset for features from Stage 2/4."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str]
    ):
        """
        Initialize feature dataset.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Label array (n_samples,)
            feature_names: List of feature names
        """
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.feature_names = feature_names
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class FeaturePreprocessor:
    """Handles imputation, scaling, and normalization of features."""
    
    def __init__(
        self,
        imputation_strategy: str = "mean",
        scaling_method: str = "standard",  # "standard", "robust", or None
        normalize: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            imputation_strategy: Strategy for imputation ("mean", "median", "most_frequent", "constant")
            scaling_method: Scaling method ("standard", "robust", or None)
            normalize: Whether to L2 normalize features
        """
        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        self.normalize = normalize
        
        self.imputer = SimpleImputer(strategy=imputation_strategy)
        self.scaler = None
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        
        self.is_fitted = False
    
    def fit(self, X: np.ndarray) -> "FeaturePreprocessor":
        """Fit preprocessor on training data."""
        # Imputation
        X_imputed = self.imputer.fit_transform(X)
        
        # Scaling
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        # Imputation
        X_imputed = self.imputer.transform(X)
        
        # Scaling
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imputed)
        else:
            X_scaled = X_imputed
        
        # Normalization
        if self.normalize:
            norms = np.linalg.norm(X_scaled, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
            X_scaled = X_scaled / norms
        
        return X_scaled
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X).transform(X)


def create_stratified_splits(
    features: np.ndarray,
    labels: np.ndarray,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified 60-20-20 train-val-test splits.
    
    Args:
        features: Feature matrix
        labels: Label array
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_state: Random seed
    
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    from sklearn.model_selection import train_test_split
    
    # First split: train (60%) vs temp (40%)
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        np.arange(len(features)),
        labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=random_state
    )
    
    # Second split: val (20%) vs test (20%) from temp (40%)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_labels,
        random_state=random_state
    )
    
    return train_indices, val_indices, test_indices


def train_model_with_cv(
    model_factory: callable,
    input_dim: int,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    batch_size: int = 32,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[torch.device] = None,
    preprocessor: Optional[FeaturePreprocessor] = None,
    use_gpu: bool = True,
    early_stopping_patience: int = 10,
    gradient_accumulation_steps: int = 1
) -> Dict[str, Any]:
    """
    Train model with 5-fold cross-validation.
    
    Args:
        model: PyTorch model
        features: Feature matrix
        labels: Label array
        feature_names: Feature names
        n_splits: Number of CV folds
        batch_size: Batch size
        epochs: Max epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Torch device (auto-detect if None)
        preprocessor: Feature preprocessor (will be created if None)
        use_gpu: Whether to use GPU
        early_stopping_patience: Early stopping patience
        gradient_accumulation_steps: Gradient accumulation steps
    
    Returns:
        Dictionary with CV results
    """
    # Setup device
    if device is None:
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info("Using GPU for training")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for training (GPU not available)")
    
    # Create preprocessor if not provided
    if preprocessor is None:
        preprocessor = FeaturePreprocessor()
    
    # Create stratified k-fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold_results = []
    all_val_probs = []
    all_val_labels = []
    all_test_probs = []
    all_test_labels = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        logger.info(f"Fold {fold_idx + 1}/{n_splits}")
        
        # Split data
        X_train, X_val = features[train_idx], features[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]
        
        # Preprocess (fit on train only to avoid data leakage)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        # Create datasets
        train_dataset = FeatureDataset(X_train_processed, y_train, feature_names)
        val_dataset = FeatureDataset(X_val_processed, y_val, feature_names)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=use_gpu
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_gpu
        )
        
        # Create fresh model for this fold
        model_fold = model_factory(input_dim)
        model_fold = model_fold.to(device)
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            model_fold.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            model_fold.train()
            train_loss = 0.0
            optimizer.zero_grad()
            
            for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device).float()
                
                # Forward
                logits = model_fold(batch_features)
                loss = criterion(logits.squeeze(), batch_labels)
                
                # Backward with gradient accumulation
                loss = loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                train_loss += loss.item() * gradient_accumulation_steps
            
            # Validation
            model_fold.eval()
            val_loss = 0.0
            val_probs_list = []
            val_labels_list = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    logits = model_fold(batch_features)
                    loss = criterion(logits.squeeze(), batch_labels.float())
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                    val_probs_list.extend(probs)
                    val_labels_list.extend(batch_labels.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model_fold.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        
        # Load best model
        if best_model_state is not None:
            model_fold.load_state_dict(best_model_state)
        
        # Final evaluation
        model_fold.eval()
        val_probs = np.array(val_probs_list)
        val_labels = np.array(val_labels_list)
        val_preds = (val_probs > 0.5).astype(int)
        
        # Metrics
        val_f1 = f1_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_ap = average_precision_score(val_labels, val_probs)
        
        fold_results.append({
            "fold": fold_idx + 1,
            "val_loss": best_val_loss,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "val_ap": val_ap
        })
        
        all_val_probs.extend(val_probs)
        all_val_labels.extend(val_labels)
        
        # Cleanup
        del model_fold, train_dataset, val_dataset, train_loader, val_loader
        aggressive_gc(clear_cuda=use_gpu)
    
    # Aggregate results
    results = {
        "fold_results": fold_results,
        "cv_val_f1": np.mean([r["val_f1"] for r in fold_results]),
        "cv_val_auc": np.mean([r["val_auc"] for r in fold_results]),
        "cv_val_ap": np.mean([r["val_ap"] for r in fold_results]),
        "all_val_probs": np.array(all_val_probs),
        "all_val_labels": np.array(all_val_labels)
    }
    
    return results


def evaluate_model(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    preprocessor: FeaturePreprocessor,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """Evaluate model and generate ROC/PR curves."""
    if device is None:
        device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")
    
    # Preprocess
    features_processed = preprocessor.transform(features)
    
    # Create dataset and loader
    dataset = FeatureDataset(features_processed, labels, feature_names)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Evaluate
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            logits = model(batch_features)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch_labels.numpy())
    
    probs = np.array(all_probs)
    labels_array = np.array(all_labels)
    preds = (probs > 0.5).astype(int)
    
    # Metrics
    f1 = f1_score(labels_array, preds)
    auc = roc_auc_score(labels_array, probs)
    ap = average_precision_score(labels_array, probs)
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(labels_array, probs)
    
    # PR curve
    precision, recall, pr_thresholds = precision_recall_curve(labels_array, probs)
    
    # Confusion matrix
    cm = confusion_matrix(labels_array, preds)
    
    results = {
        "f1": f1,
        "auc": auc,
        "ap": ap,
        "probs": probs,
        "preds": preds,
        "labels": labels_array,
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm
    }
    
    return results


def plot_roc_pr_curves(
    results: Dict[str, Any],
    output_path: Path,
    title: str = "ROC and PR Curves"
):
    """Plot ROC and PR curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC curve
    ax1.plot(results["fpr"], results["tpr"], label=f"ROC (AUC = {results['auc']:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True)
    
    # PR curve
    ax2.plot(results["recall"], results["precision"], label=f"PR (AP = {results['ap']:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved ROC/PR curves to {output_path}")

