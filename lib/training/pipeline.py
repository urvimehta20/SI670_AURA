"""
Model training pipeline.

Trains models using scaled videos and extracted features.
Supports multiple model types and k-fold cross-validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional
import polars as pl
import torch
from torch.utils.data import DataLoader

from lib.data import stratified_kfold, load_metadata
from lib.models import VideoConfig, VideoDataset
from lib.mlops.config import ExperimentTracker, CheckpointManager
from lib.mlops.mlflow_tracker import create_mlflow_tracker, MLFLOW_AVAILABLE
from lib.training.trainer import OptimConfig, TrainConfig, fit
from lib.training.model_factory import create_model, is_pytorch_model, is_xgboost_model, get_model_config
from lib.training.feature_preprocessing import remove_collinear_features, load_and_combine_features
from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


def stage5_train_models(
    project_root: str,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: str,
    model_types: List[str],
    n_splits: int = 5,
    num_frames: int = 8,
    output_dir: str = "data/training_results",
    use_tracking: bool = True,
    use_mlflow: bool = True,
    train_ensemble: bool = False,
    ensemble_method: str = "meta_learner",
    delete_existing: bool = False
) -> Dict:
    """
    Stage 5: Train models using scaled videos and features.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled metadata (from Stage 3)
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        model_types: List of model types to train
        n_splits: Number of k-fold splits
        num_frames: Number of frames per video
        output_dir: Directory to save training results
        use_tracking: Whether to use experiment tracking
        train_ensemble: Whether to train ensemble model after individual models (default: False)
        ensemble_method: Ensemble method - "meta_learner" or "weighted_average" (default: "meta_learner")
        delete_existing: If True, delete existing model checkpoints/results before regenerating (clean mode)
    
    Returns:
        Dictionary of training results
    """
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metadata (support both CSV and Arrow/Parquet)
    logger.info("Stage 5: Loading metadata...")
    
    from lib.utils.paths import load_metadata_flexible
    
    scaled_df = load_metadata_flexible(scaled_metadata_path)
    if scaled_df is None:
        raise FileNotFoundError(f"Scaled metadata not found: {scaled_metadata_path}")
    
    features2_df = load_metadata_flexible(features_stage2_path)
    features4_df = load_metadata_flexible(features_stage4_path)
    
    logger.info(f"Stage 5: Found {scaled_df.height} scaled videos")
    
    # Create video config
    # Use fixed_size=256 to match Stage 3 output (videos scaled to max(width, height) = 256)
    video_config = VideoConfig(
        num_frames=num_frames,
        fixed_size=256,
    )
    
    results = {}
    
    # Delete existing model results if clean mode
    if delete_existing:
        logger.info("Stage 5: Deleting existing model results (clean mode)...")
        deleted_count = 0
        for model_type in model_types:
            model_output_dir = output_dir / model_type
            if model_output_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(model_output_dir)
                    deleted_count += 1
                    logger.info(f"Deleted existing results for {model_type}")
                except Exception as e:
                    logger.warning(f"Could not delete {model_output_dir}: {e}")
        logger.info(f"Stage 5: Deleted {deleted_count} existing model directories")
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 5: Training model: {model_type}")
        logger.info(f"{'='*80}")
        
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model training is already complete (resume mode)
        if not delete_existing:
            checkpoint_dir = model_output_dir / "checkpoints"
            completion_file = model_output_dir / "training_complete.pt"
            if completion_file.exists():
                logger.info(f"Model {model_type} training already complete. Skipping.")
                logger.info(f"To retrain, use --delete-existing flag")
                continue
            elif checkpoint_dir.exists() and any(checkpoint_dir.glob("*.pt")):
                logger.info(f"Found existing checkpoints for {model_type}. Will resume from latest checkpoint.")
        
        # Get model config
        model_config = get_model_config(model_type)
        
        # K-fold cross-validation
        fold_results = []
        
        # CRITICAL FIX: Get all folds at once (stratified_kfold returns list of all folds)
        all_folds = stratified_kfold(
            scaled_df,
            n_splits=n_splits,
            random_state=42
        )
        
        if len(all_folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds, got {len(all_folds)}")
        
        for fold_idx in range(n_splits):
            logger.info(f"\nTraining {model_type} - Fold {fold_idx + 1}/{n_splits}")
            
            # Get the specific fold
            train_df, val_df = all_folds[fold_idx]
            
            # Validate no data leakage (check dup_group if present)
            if "dup_group" in scaled_df.columns:
                train_groups = set(train_df["dup_group"].unique().to_list())
                val_groups = set(val_df["dup_group"].unique().to_list())
                overlap = train_groups & val_groups
                if overlap:
                    logger.error(
                        f"CRITICAL: Data leakage detected in fold {fold_idx + 1}! "
                        f"{len(overlap)} duplicate groups appear in both train and val: {list(overlap)[:5]}"
                    )
                    raise ValueError(f"Data leakage: duplicate groups in both train and val sets")
                logger.info(f"Fold {fold_idx + 1}: No data leakage (checked {len(train_groups)} train groups, {len(val_groups)} val groups)")
            
            # Create datasets
            train_dataset = VideoDataset(
                train_df,
                project_root=str(project_root),
                config=video_config,
            )
            val_dataset = VideoDataset(
                val_df,
                project_root=str(project_root),
                config=video_config,
            )
            
            # Create data loaders
            # GPU-optimized DataLoader settings
            use_cuda = torch.cuda.is_available()
            num_workers = model_config.get("num_workers", 0)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=model_config.get("batch_size", 8),
                shuffle=True,
                num_workers=num_workers,
                pin_memory=use_cuda,  # Faster GPU transfer
                persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=model_config.get("batch_size", 8),
                shuffle=False,
                num_workers=num_workers,
                pin_memory=use_cuda,
                persistent_workers=num_workers > 0,
                prefetch_factor=2 if num_workers > 0 else None,
            )
            
            # Train model
            if is_pytorch_model(model_type):
                # PyTorch model training
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = create_model(model_type, model_config)
                model = model.to(device)
                
                # Create optimizer and scheduler with ML best practices
                optim_cfg = OptimConfig(
                    lr=model_config.get("learning_rate", 1e-4),
                    weight_decay=model_config.get("weight_decay", 1e-4),
                    max_grad_norm=model_config.get("max_grad_norm", 1.0),  # Gradient clipping
                    # Use differential LR for pretrained models
                    backbone_lr=model_config.get("backbone_lr", None),
                    head_lr=model_config.get("head_lr", None),
                )
                train_cfg = TrainConfig(
                    num_epochs=model_config.get("num_epochs", 20),
                    device=str(device),
                    use_amp=model_config.get("use_amp", True),
                    gradient_accumulation_steps=model_config.get("gradient_accumulation_steps", 1),
                    early_stopping_patience=model_config.get("early_stopping_patience", 5),
                    scheduler_type=model_config.get("scheduler_type", "cosine"),  # Better than StepLR
                    warmup_epochs=model_config.get("warmup_epochs", 2),  # LR warmup
                    warmup_factor=model_config.get("warmup_factor", 0.1),
                    log_grad_norm=model_config.get("log_grad_norm", False),  # Debug gradient norms
                )
                
                # Determine if we should use differential LR (for pretrained models)
                use_differential_lr = model_type in [
                    "i3d", "r2plus1d", "slowfast", "x3d", "pretrained_inception",
                    "vit_gru", "vit_transformer"
                ]
                
                # Create tracker and checkpoint manager
                fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)
                
                if use_tracking:
                    tracker = ExperimentTracker(str(fold_output_dir))
                    ckpt_manager = CheckpointManager(str(fold_output_dir))
                    
                    # Create MLflow tracker if available
                    mlflow_tracker = None
                    if use_mlflow and MLFLOW_AVAILABLE:
                        try:
                            mlflow_tracker = create_mlflow_tracker(
                                experiment_name=f"{model_type}",
                                use_mlflow=True
                            )
                            if mlflow_tracker:
                                # Log model config (can be dict or RunConfig)
                                mlflow_tracker.log_config(model_config)
                                mlflow_tracker.set_tag("fold", str(fold_idx + 1))
                                mlflow_tracker.set_tag("model_type", model_type)
                        except Exception as e:
                            logger.warning(f"Failed to create MLflow tracker: {e}")
                else:
                    tracker = None
                    ckpt_manager = None
                    mlflow_tracker = None
                
                logger.info(f"Training PyTorch model {model_type} on fold {fold_idx + 1}...")
                
                # Train
                try:
                    model = fit(
                        model,
                        train_loader,
                        val_loader,
                        optim_cfg,
                        train_cfg,
                        use_differential_lr=use_differential_lr,  # Use differential LR for pretrained models
                    )
                    
                    # Evaluate final model
                    from lib.training.trainer import evaluate
                    val_loss, val_acc = evaluate(model, val_loader, device=str(device))
                    
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    
                    logger.info(f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Log to MLflow if available
                    if 'mlflow_tracker' in locals() and mlflow_tracker is not None:
                        try:
                            mlflow_tracker.log_metrics({
                                "val_loss": val_loss,
                                "val_acc": val_acc,
                            }, step=fold_idx + 1)
                            # Log model artifact (after model is saved)
                            if 'model_path' in locals():
                                model_path_str = str(model_path)
                                mlflow_tracker.log_artifact(
                                    model_path_str, artifact_path="models"
                                )
                        except Exception as e:
                            logger.warning(f"Failed to log to MLflow: {e}")
                    
                    # Save model for ensemble training
                    model.eval()
                    model_path = fold_output_dir / "model.pt"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved model to {model_path}")
                    
                except Exception as e:
                    logger.error(f"Error training fold {fold_idx + 1}: {e}", exc_info=True)
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": float('nan'),
                        "val_acc": float('nan'),
                    })
                
                # End MLflow run if active
                if 'mlflow_tracker' in locals() and mlflow_tracker is not None:
                    try:
                        mlflow_tracker.end_run()
                    except Exception as e:
                        logger.debug(f"Error ending MLflow run: {e}")
                
                # Clear model and aggressively free memory
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                aggressive_gc(clear_cuda=False)
            elif is_xgboost_model(model_type):
                # XGBoost model training (uses pretrained models for feature extraction)
                logger.info(f"Training XGBoost model {model_type} on fold {fold_idx + 1}...")
                
                fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Create XGBoost model
                    model = create_model(model_type, model_config)
                    
                    # Train XGBoost (handles feature extraction internally)
                    model.fit(train_df, project_root=str(project_root))
                    
                    # Evaluate on validation set
                    val_probs = model.predict(val_df, project_root=str(project_root))
                    val_preds = np.argmax(val_probs, axis=1)
                    val_labels = val_df["label"].to_list()
                    label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
                    val_y = np.array([label_map[label] for label in val_labels])
                    
                    val_acc = (val_preds == val_y).mean()
                    val_loss = -np.mean(np.log(val_probs[np.arange(len(val_y)), val_y] + 1e-10))  # Cross-entropy
                    
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    
                    logger.info(f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Save model
                    model.save(str(fold_output_dir))
                    logger.info(f"Saved XGBoost model to {fold_output_dir}")
                    
                except Exception as e:
                    logger.error(f"Error training XGBoost fold {fold_idx + 1}: {e}", exc_info=True)
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": float('nan'),
                        "val_acc": float('nan'),
                    })
                
                # Clear model and aggressively free memory
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                aggressive_gc(clear_cuda=True)
            else:
                # Baseline model training (sklearn)
                logger.info(f"Training baseline model {model_type} on fold {fold_idx + 1}...")
                
                fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                fold_output_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    # Create baseline model
                    model = create_model(model_type, model_config)
                    
                    # Train baseline (handles feature extraction internally)
                    model.fit(train_df, project_root=str(project_root))
                    
                    # Evaluate on validation set
                    val_probs = model.predict(val_df, project_root=str(project_root))
                    val_preds = np.argmax(val_probs, axis=1)
                    val_labels = val_df["label"].to_list()
                    label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
                    val_y = np.array([label_map[label] for label in val_labels])
                    
                    val_acc = (val_preds == val_y).mean()
                    val_loss = -np.mean(np.log(val_probs[np.arange(len(val_y)), val_y] + 1e-10))  # Cross-entropy
                    
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    })
                    
                    logger.info(f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
                    
                    # Save model
                    model.save(str(fold_output_dir))
                    logger.info(f"Saved baseline model to {fold_output_dir}")
                    
                except Exception as e:
                    logger.error(
                        f"Error training baseline fold {fold_idx + 1}: {e}",
                        exc_info=True
                    )
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": float('nan'),
                        "val_acc": float('nan'),
                    })
                
                # Clear model and aggressively free memory
                if 'model' in locals():
                    del model
                aggressive_gc(clear_cuda=False)
        
        # Aggregate results (filter out NaN values)
        if fold_results:
            valid_losses = [
                r["val_loss"] for r in fold_results
                if isinstance(r["val_loss"], (int, float))
                and not (isinstance(r["val_loss"], float)
                         and r["val_loss"] != r["val_loss"])
            ]
            valid_accs = [
                r["val_acc"] for r in fold_results
                if isinstance(r["val_acc"], (int, float))
                and not (isinstance(r["val_acc"], float)
                         and r["val_acc"] != r["val_acc"])
            ]
            
            avg_val_loss = (
                sum(valid_losses) / len(valid_losses)
                if valid_losses else float('nan')
            )
            avg_val_acc = (
                sum(valid_accs) / len(valid_accs)
                if valid_accs else float('nan')
            )
            
            results[model_type] = {
                "fold_results": fold_results,
                "avg_val_loss": avg_val_loss,
                "avg_val_acc": avg_val_acc,
            }
            
            logger.info(
                "\n%s - Avg Val Loss: %.4f, Avg Val Acc: %.4f",
                model_type, avg_val_loss, avg_val_acc
            )
        
        # Aggressive GC after all folds for this model type
        aggressive_gc(clear_cuda=False)
    
    # Train ensemble if requested
    if train_ensemble:
        logger.info("\n" + "="*80)
        logger.info("Training Ensemble Model")
        logger.info("="*80)
        
        try:
            from .ensemble import train_ensemble_model
            
            ensemble_results = train_ensemble_model(
                project_root=str(project_root),
                scaled_metadata_path=scaled_metadata_path,
                base_model_types=model_types,
                base_models_dir=str(output_dir),
                n_splits=n_splits,
                num_frames=num_frames,
                output_dir=str(output_dir),
                ensemble_method=ensemble_method
            )
            
            results["ensemble"] = ensemble_results
            logger.info("✓ Ensemble training completed")
        except Exception as e:
            logger.error(f"Error training ensemble: {e}", exc_info=True)
            logger.warning("Continuing without ensemble results")
    
    return results

