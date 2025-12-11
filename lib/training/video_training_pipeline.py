"""
Video-based training pipeline for models that process video frames.

All models except baseline (logistic_regression, svm) use video frames as input.
"""

from __future__ import annotations

import logging
import gc
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
import matplotlib.pyplot as plt

from lib.models import VideoDataset, VideoConfig, variable_ar_collate
from lib.training.model_factory import create_model, get_model_config
from lib.training.grid_search import get_hyperparameter_grid
from lib.utils.paths import load_metadata_flexible
from lib.utils.memory import aggressive_gc, log_memory_stats, check_oom_error, handle_oom_error, safe_execute

logger = logging.getLogger(__name__)


# Models that use FEATURES (baseline models)
FEATURE_BASED_MODELS = {
    "logistic_regression",
    "logistic_regression_stage2",
    "logistic_regression_stage2_stage4",
    "svm",
    "svm_stage2",
    "svm_stage2_stage4",
}

# Models that use VIDEO FRAMES (all others)
VIDEO_BASED_MODELS = {
    "naive_cnn",
    "variable_ar_cnn",
    "vit_gru",
    "vit_transformer",
    "slowfast",
    "x3d",
    "pretrained_inception",
    "i3d",
    "r2plus1d",
    "timesformer",
    "vivit",
    "two_stream",
    "slowfast_attention",
    "slowfast_multiscale",
    "xgboost_i3d",
    "xgboost_r2plus1d",
    "xgboost_vit_gru",
    "xgboost_vit_transformer",
    "xgboost_pretrained_inception",
}


def is_feature_based(model_type: str) -> bool:
    """Check if model uses features (True) or video frames (False)."""
    return model_type in FEATURE_BASED_MODELS


def is_video_based(model_type: str) -> bool:
    """Check if model uses video frames."""
    return model_type in VIDEO_BASED_MODELS


def train_video_model(
    model_type: str,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    project_root: str,
    output_dir: Path,
    n_splits: int = 5,
    num_frames: int = 1000,
    batch_size: Optional[int] = None,  # If None, uses model config
    epochs: int = 100,
    use_gpu: bool = True,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Train a video-based model with hyperparameter tuning and CV.
    
    Args:
        model_type: Model type name
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        project_root: Project root directory
        output_dir: Output directory
        n_splits: Number of CV folds
        num_frames: Number of frames per video
        batch_size: Batch size
        epochs: Max epochs
        use_gpu: Use GPU if available
        device: Torch device
    
    Returns:
        Training results dictionary
    """
    # Setup device
    if device is None:
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU for {model_type}")
        else:
            device = torch.device("cpu")
            logger.info(f"Using CPU for {model_type}")
    
    logger.info(f"Training {model_type} with scaled videos: sampling {num_frames} frames per video (uniformly across video duration)")
    
    # Create video config
    # For variable_ar_cnn, use max_size instead of fixed_size
    if model_type == "variable_ar_cnn":
        video_config = VideoConfig(
            num_frames=num_frames,
            max_size=256,  # Variable aspect ratio
            img_size=None
        )
        use_variable_ar = True
    else:
        video_config = VideoConfig(
            num_frames=num_frames,
            fixed_size=256,  # Fixed size for most models
            img_size=None
        )
        use_variable_ar = False
    
    # Get model memory config for batch size and gradient accumulation
    model_mem_config = get_model_config(model_type)
    gradient_accumulation_steps = model_mem_config.get("gradient_accumulation_steps", 1)
    num_workers = model_mem_config.get("num_workers", 0)
    
    # Use model config batch_size if not explicitly provided
    if batch_size is None:
        effective_batch_size = model_mem_config.get("batch_size", 1)
    else:
        effective_batch_size = batch_size
    
    logger.info(f"Memory config: batch_size={effective_batch_size}, gradient_accumulation={gradient_accumulation_steps}, num_workers={num_workers}")
    logger.info(f"Effective batch size: {effective_batch_size * gradient_accumulation_steps}")
    
    # Log initial memory stats
    log_memory_stats("before hyperparameter search", detailed=True)
    
    # Get hyperparameter grid
    param_grid = get_hyperparameter_grid(model_type)
    logger.info(f"Hyperparameter search: {len(list(ParameterGrid(param_grid)))} combinations")
    
    # Combine train and val for hyperparameter search
    trainval_df = pl.concat([train_df, val_df])
    
    # Hyperparameter search
    best_score = -1
    best_params = None
    best_model_state = None
    grid_results = []
    
    for param_idx, params in enumerate(ParameterGrid(param_grid)):
        logger.info(f"Grid search {param_idx + 1}/{len(list(ParameterGrid(param_grid)))}: {params}")
        
        # Aggressive GC before each grid search iteration
        aggressive_gc(clear_cuda=True)
        log_memory_stats(f"grid_search_{param_idx}", detailed=False)
        
        try:
            # Create model with these hyperparameters
            from lib.mlops.config import RunConfig
            model_config = RunConfig(
                run_id=f"{model_type}_grid_{param_idx}",
                experiment_name=model_type,
                model_type=model_type,
                num_frames=num_frames,
                model_specific_config=params
            )
            
            model = create_model(model_type, model_config)
            model = model.to(device)
            
            # Create datasets
            train_dataset = VideoDataset(train_df, project_root, video_config, train=True)
            val_dataset = VideoDataset(val_df, project_root, video_config, train=False)
            
            # Create data loaders with memory-optimized settings
            collate_fn = variable_ar_collate if use_variable_ar else None
            train_loader = DataLoader(
                train_dataset,
                batch_size=effective_batch_size,
                shuffle=True,
                num_workers=num_workers,  # 0 to avoid multiprocessing memory overhead
                pin_memory=use_gpu,
                collate_fn=collate_fn,
                persistent_workers=False  # Don't keep workers alive
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=effective_batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_gpu,
                collate_fn=collate_fn,
                persistent_workers=False
            )
            
            # Setup optimizer
            learning_rate = params.get("learning_rate", 1e-3)
            weight_decay = params.get("weight_decay", 1e-5)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss()
            
            # Training loop with early stopping, gradient accumulation, and OOM handling
            best_val_f1 = -1
            patience_counter = 0
            patience = 10
            
            for epoch in range(epochs):
                # Aggressive GC at start of each epoch
                aggressive_gc(clear_cuda=True)
                
                # Train with gradient accumulation
                model.train()
                train_loss = 0.0
                optimizer.zero_grad()  # Zero gradients at start
                
                for batch_idx, (batch_videos, batch_labels) in enumerate(train_loader):
                    try:
                        batch_videos = batch_videos.to(device, non_blocking=True)
                        batch_labels = batch_labels.to(device, non_blocking=True).float()
                        
                        # Reshape if needed: (B, T, C, H, W) -> (B, C, T, H, W) for 3D CNNs
                        if batch_videos.dim() == 5:
                            batch_videos = batch_videos.permute(0, 2, 1, 3, 4)
                        
                        logits = model(batch_videos)
                        if logits.dim() > 1:
                            logits = logits.squeeze()
                        loss = criterion(logits, batch_labels)
                        
                        # Scale loss by gradient accumulation steps
                        loss = loss / gradient_accumulation_steps
                        loss.backward()
                        
                        train_loss += loss.item() * gradient_accumulation_steps
                        
                        # Update weights every gradient_accumulation_steps
                        if (batch_idx + 1) % gradient_accumulation_steps == 0:
                            # Gradient clipping for stability
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # Aggressive GC after each update
                            if (batch_idx + 1) % (gradient_accumulation_steps * 4) == 0:
                                aggressive_gc(clear_cuda=True)
                    
                    except RuntimeError as e:
                        if check_oom_error(e):
                            handle_oom_error(e, f"training epoch {epoch} batch {batch_idx}")
                            # Clear gradients and try to continue
                            optimizer.zero_grad()
                            aggressive_gc(clear_cuda=True)
                            logger.warning(f"Skipping batch due to OOM, continuing training...")
                            continue
                        else:
                            raise
                
                # Final gradient update if there are remaining gradients
                if (batch_idx + 1) % gradient_accumulation_steps != 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    aggressive_gc(clear_cuda=True)
            
                # Validate with OOM handling
                model.eval()
                val_probs = []
                val_labels_list = []
                with torch.no_grad():
                    for val_batch_idx, (batch_videos, batch_labels) in enumerate(val_loader):
                        try:
                            batch_videos = batch_videos.to(device, non_blocking=True)
                            if batch_videos.dim() == 5:
                                batch_videos = batch_videos.permute(0, 2, 1, 3, 4)
                            
                            logits = model(batch_videos)
                            if logits.dim() > 1:
                                logits = logits.squeeze()
                            probs = torch.sigmoid(logits).cpu().numpy()
                            val_probs.extend(probs)
                            val_labels_list.extend(batch_labels.numpy())
                            
                            # Periodic GC during validation
                            if (val_batch_idx + 1) % 10 == 0:
                                aggressive_gc(clear_cuda=True)
                        
                        except RuntimeError as e:
                            if check_oom_error(e):
                                handle_oom_error(e, f"validation epoch {epoch} batch {val_batch_idx}")
                                aggressive_gc(clear_cuda=True)
                                logger.warning(f"Skipping validation batch due to OOM, continuing...")
                                continue
                            else:
                                raise
            
                val_probs = np.array(val_probs)
                val_labels_array = np.array(val_labels_list)
                val_preds = (val_probs > 0.5).astype(int)
                val_f1 = f1_score(val_labels_array, val_preds)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_f1={val_f1:.4f}")
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
                
                # Aggressive GC after validation
                aggressive_gc(clear_cuda=True)
            
            grid_results.append({
                "params": params,
                "val_f1": best_val_f1
            })
            
            # Cleanup after each grid search iteration
            del model, train_dataset, val_dataset, train_loader, val_loader, optimizer
            aggressive_gc(clear_cuda=True)
            log_memory_stats(f"after grid_search_{param_idx}", detailed=False)
        
        except RuntimeError as e:
            if check_oom_error(e):
                handle_oom_error(e, f"grid_search_{param_idx}")
                logger.error(f"OOM during grid search iteration {param_idx}, skipping this configuration")
                grid_results.append({
                    "params": params,
                    "val_f1": -1.0,  # Mark as failed
                    "error": "OOM"
                })
                aggressive_gc(clear_cuda=True)
                continue
            else:
                raise
        
        if best_val_f1 > best_score:
            best_score = best_val_f1
            best_params = params.copy()
            best_model_state = model.state_dict().copy()
        
        # Cleanup
        del model, train_dataset, val_dataset, train_loader, val_loader
        aggressive_gc(clear_cuda=use_gpu)
    
    logger.info(f"Best hyperparameters: {best_params} (val_f1: {best_score:.4f})")
    
    # Train final model with best params using 5-fold CV on train+val
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    labels = trainval_df["label"].to_list()
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_trainval = np.array([label_map[label] for label in labels])
    
    fold_results = []
    all_val_probs = []
    all_val_labels = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(trainval_df, y_trainval)):
        logger.info(f"CV Fold {fold_idx + 1}/{n_splits}")
        
        fold_train_df = trainval_df[train_idx]
        fold_val_df = trainval_df[val_idx]
        
        # Create model
        from lib.mlops.config import RunConfig
        model_config = RunConfig(
            run_id=f"{model_type}_cv_fold_{fold_idx}",
            experiment_name=model_type,
            model_type=model_type,
            num_frames=num_frames,
            model_specific_config=best_params
        )
        model = create_model(model_type, model_config)
        model = model.to(device)
        
        # Create datasets
        fold_train_dataset = VideoDataset(fold_train_df, project_root, video_config, train=True)
        fold_val_dataset = VideoDataset(fold_val_df, project_root, video_config, train=False)
        
        # Create loaders
        collate_fn = variable_ar_collate if use_variable_ar else None
        fold_train_loader = DataLoader(
            fold_train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_gpu,
            collate_fn=collate_fn
        )
        fold_val_loader = DataLoader(
            fold_val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=use_gpu,
            collate_fn=collate_fn
        )
        
        # Train
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=best_params.get("learning_rate", 1e-3),
            weight_decay=best_params.get("weight_decay", 1e-5)
        )
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            for batch_videos, batch_labels in fold_train_loader:
                batch_videos = batch_videos.to(device)
                batch_labels = batch_labels.to(device).float()
                if batch_videos.dim() == 5:
                    batch_videos = batch_videos.permute(0, 2, 1, 3, 4)
                
                optimizer.zero_grad()
                logits = model(batch_videos)
                if logits.dim() > 1:
                    logits = logits.squeeze()
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            val_probs_list = []
            val_labels_list = []
            val_loss = 0.0
            with torch.no_grad():
                for batch_videos, batch_labels in fold_val_loader:
                    batch_videos = batch_videos.to(device)
                    batch_labels = batch_labels.to(device).float()
                    if batch_videos.dim() == 5:
                        batch_videos = batch_videos.permute(0, 2, 1, 3, 4)
                    
                    logits = model(batch_videos)
                    if logits.dim() > 1:
                        logits = logits.squeeze()
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(logits).cpu().numpy()
                    val_probs_list.extend(probs)
                    val_labels_list.extend(batch_labels.cpu().numpy())
            
            avg_val_loss = val_loss / len(fold_val_loader)
            val_probs_array = np.array(val_probs_list)
            val_labels_array = np.array(val_labels_list)
            val_preds = (val_probs_array > 0.5).astype(int)
            val_f1 = f1_score(val_labels_array, val_preds)
            val_auc = roc_auc_score(val_labels_array, val_probs_array)
            val_ap = average_precision_score(val_labels_array, val_probs_array)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        fold_results.append({
            "fold": fold_idx + 1,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "val_ap": val_ap
        })
        
        all_val_probs.extend(val_probs_list)
        all_val_labels.extend(val_labels_list)
        
        # Cleanup
        del model, fold_train_dataset, fold_val_dataset, fold_train_loader, fold_val_loader
        aggressive_gc(clear_cuda=use_gpu)
    
    # Final evaluation on test set
    from lib.mlops.config import RunConfig
    final_model_config = RunConfig(
        run_id=f"{model_type}_final",
        experiment_name=model_type,
        model_type=model_type,
        num_frames=num_frames,
        model_specific_config=best_params
    )
    final_model = create_model(model_type, final_model_config)
    final_model.load_state_dict(best_model_state)
    final_model = final_model.to(device)
    
    test_dataset = VideoDataset(test_df, project_root, video_config, train=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_gpu,
        collate_fn=collate_fn
    )
    
    final_model.eval()
    test_probs = []
    test_labels_list = []
    with torch.no_grad():
        for batch_videos, batch_labels in test_loader:
            batch_videos = batch_videos.to(device)
            if batch_videos.dim() == 5:
                batch_videos = batch_videos.permute(0, 2, 1, 3, 4)
            
            logits = final_model(batch_videos)
            if logits.dim() > 1:
                logits = logits.squeeze()
            probs = torch.sigmoid(logits).cpu().numpy()
            test_probs.extend(probs)
            test_labels_list.extend(batch_labels.numpy())
    
    test_probs = np.array(test_probs)
    test_labels_array = np.array(test_labels_list)
    test_preds = (test_probs > 0.5).astype(int)
    
    test_f1 = f1_score(test_labels_array, test_preds)
    test_auc = roc_auc_score(test_labels_array, test_probs)
    test_ap = average_precision_score(test_labels_array, test_probs)
    
    # ROC and PR curves
    fpr, tpr, _ = roc_curve(test_labels_array, test_probs)
    precision, recall, _ = precision_recall_curve(test_labels_array, test_probs)
    
    # Save results
    results = {
        "model_type": model_type,
        "best_params": best_params,
        "best_val_f1": best_score,
        "cv_results": {
            "fold_results": fold_results,
            "cv_val_f1": np.mean([r["val_f1"] for r in fold_results]),
            "cv_val_auc": np.mean([r["val_auc"] for r in fold_results]),
            "cv_val_ap": np.mean([r["val_ap"] for r in fold_results]),
        },
        "test_results": {
            "f1": test_f1,
            "auc": test_auc,
            "ap": test_ap,
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "precision": precision.tolist(),
            "recall": recall.tolist(),
        },
        "grid_results": grid_results
    }
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_model_state,
        "model_type": model_type,
        "model_params": best_params,
        "num_frames": num_frames,
    }, output_dir / "model.pt")
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "best_params": best_params,
            "best_val_f1": float(best_score),
            "cv_val_f1": float(results["cv_results"]["cv_val_f1"]),
            "test_f1": float(test_f1),
            "test_auc": float(test_auc),
            "test_ap": float(test_ap),
        }, f, indent=2)
    
    # Plot ROC/PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, label=f"ROC (AUC = {test_auc:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(recall, precision, label=f"PR (AP = {test_ap:.3f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(f"{model_type} - ROC and PR Curves")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_pr_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training complete for {model_type}")
    logger.info(f"  CV F1: {results['cv_results']['cv_val_f1']:.4f}")
    logger.info(f"  Test F1: {test_f1:.4f}")
    logger.info(f"  Test AUC: {test_auc:.4f}")
    
    return results

