"""
Multi-Model Pipeline: Train multiple models sequentially with shared data pipeline.

This pipeline:
- Loads data once (shared)
- Creates k-fold splits once (shared)
- Generates augmentations once (shared)
- Trains each model sequentially on all folds
- Saves per-model checkpoints and metrics
- Supports resume from last incomplete model
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import polars as pl
import torch
import numpy as np

from .mlops_core import (
    RunConfig, ExperimentTracker, CheckpointManager, 
    DataVersionManager, create_run_directory
)
from .mlops_utils import (
    aggressive_gc, check_oom_error, handle_oom_error, 
    safe_execute, log_memory_stats
)
from .mlops_pipeline import (
    PipelineStage, MLOpsPipeline, fit_with_tracking
)
from .video_data import (
    load_metadata,
    filter_existing_videos,
    stratified_kfold,
    make_balanced_batch_sampler,
    maybe_limit_to_small_test_subset,
)
from .video_modeling import VideoConfig, VideoDataset, variable_ar_collate

from .video_augmentation_pipeline import pregenerate_augmented_dataset
from .video_training import OptimConfig, TrainConfig
from .video_metrics import collect_logits_and_labels, basic_classification_metrics
from .model_factory import create_model, get_model_config, is_pytorch_model, download_pretrained_models

from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def _check_model_complete(model_type: str, output_dir: str, n_splits: int) -> bool:
    """Check if a model has been fully trained (all folds complete)."""
    model_dir = os.path.join(output_dir, "models", model_type)
    if not os.path.exists(model_dir):
        return False
    
    # Check for completion marker or all fold results
    completion_file = os.path.join(model_dir, "training_complete.pt")
    if os.path.exists(completion_file):
        return True
    
    # Check if all folds have results
    results_file = os.path.join(model_dir, "fold_results.csv")
    if os.path.exists(results_file):
        try:
            results_df = pl.read_csv(results_file)
            if results_df.height >= n_splits:
                return True
        except Exception:
            pass
    
    return False


def _train_baseline_model(
    model: Any,
    model_type: str,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    project_root: str,
    output_dir: str,
    fold_idx: int,
    tracker: ExperimentTracker
) -> Dict[str, float]:
    """
    Train a baseline model (sklearn-style).
    
    Args:
        model: Baseline model instance (LogisticRegressionBaseline or SVMBaseline)
        model_type: Model type identifier
        train_df: Training DataFrame
        val_df: Validation DataFrame
        project_root: Project root directory
        output_dir: Output directory
        fold_idx: Fold index
        tracker: Experiment tracker
    
    Returns:
        Dictionary of metrics
    """
    logger.info("Training %s baseline model (fold %d)...", model_type, fold_idx + 1)

    # Guard against degenerate folds in tiny test-mode runs:
    # if the training split has only one class, sklearn classifiers will fail.
    # In that case, we log a warning and return NaN metrics for this fold
    # instead of crashing the whole pipeline.
    train_labels = train_df["label"].to_list()
    unique_labels = sorted(set(train_labels))
    if len(unique_labels) < 2:
        logger.warning(
            "Fold %d for model %s has only one class in the training data (%s). "
            "Skipping baseline training for this fold and returning NaN metrics.",
            fold_idx + 1,
            model_type,
            unique_labels[0] if unique_labels else "N/A",
        )
        metrics = {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
        # Still log to tracker so the fold is visible in experiment logs
        for metric_name, value in metrics.items():
            tracker.log_metric(
                step=fold_idx + 1,
                metric_name=metric_name,
                value=value,
                epoch=fold_idx + 1,
                phase="val",
            )
        logger.info("Fold %d results (degenerate single-class fold): %s", fold_idx + 1, metrics)
        return metrics

    # Train model
    model.fit(train_df, project_root)
    
    # If validation set is empty (can happen in tiny test-mode runs),
    # skip metric computation and return NaNs instead of calling sklearn
    # with an empty feature array.
    if val_df.height == 0:
        logger.warning(
            "Fold %d for model %s has an empty validation set. "
            "Skipping metric computation and returning NaN metrics.",
            fold_idx + 1,
            model_type,
        )
        metrics = {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }
        model_dir = os.path.join(output_dir, "models", model_type, f"fold_{fold_idx+1}")
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
        for metric_name, value in metrics.items():
            tracker.log_metric(
                step=fold_idx + 1,
                metric_name=metric_name,
                value=value,
                epoch=fold_idx + 1,
                phase="val",
            )
        logger.info("Fold %d results (empty validation set): %s", fold_idx + 1, metrics)
        return metrics
    
    # Predict on validation set
    val_probs = model.predict(val_df, project_root)
    
    # Get labels
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    val_labels = np.array([label_map[label] for label in val_df["label"].to_list()])
    
    # Get predictions
    val_preds = np.argmax(val_probs, axis=1)
    
    # Compute metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = float(accuracy_score(val_labels, val_preds))
    precision = float(precision_score(val_labels, val_preds, average='binary', zero_division=0))
    recall = float(recall_score(val_labels, val_preds, average='binary', zero_division=0))
    f1 = float(f1_score(val_labels, val_preds, average='binary', zero_division=0))
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    
    # Save model
    model_dir = os.path.join(output_dir, "models", model_type, f"fold_{fold_idx+1}")
    os.makedirs(model_dir, exist_ok=True)
    model.save(model_dir)
    
    # Log metrics
    for metric_name, value in metrics.items():
        tracker.log_metric(
            step=fold_idx + 1,
            metric_name=metric_name,
            value=value,
            epoch=fold_idx + 1,
            phase="val"
        )
    
    logger.info("Fold %d results: %s", fold_idx + 1, metrics)
    
    return metrics


def build_multimodel_pipeline(
    config: RunConfig,
    model_types: List[str],
    tracker: ExperimentTracker,
    n_splits: int = 5,
    ckpt_manager: Optional[CheckpointManager] = None
) -> MLOpsPipeline:
    """
    Build multi-model pipeline that trains models sequentially.
    
    Args:
        config: Base run configuration
        model_types: List of model types to train
        tracker: Experiment tracker
        n_splits: Number of k-fold splits
        ckpt_manager: Checkpoint manager
    
    Returns:
        MLOpsPipeline instance
    """
    pipeline = MLOpsPipeline(config, tracker)
    
    # Stage 1: Load data (shared)
    def load_data():
        meta_df = load_metadata(config.data_csv)
        logger.info("Loaded %d video records", meta_df.height)
        
        filtered_df = filter_existing_videos(meta_df, config.project_root, check_frames=False)
        logger.info("After filtering: %d valid videos", filtered_df.height)

        # Optional: limit to a tiny balanced subset for SLURM test runs.
        # Use 5 per class by default so 5-fold CV and multi-model runs
        # have enough data per class while remaining small.
        filtered_df = maybe_limit_to_small_test_subset(filtered_df, max_per_class=5)

        # Ultra aggressive GC after data loading
        aggressive_gc(clear_cuda=True)
        return {"metadata": filtered_df}
    
    pipeline.register_stage(PipelineStage("load_data", load_data))
    
    # Stage 2: Download and verify pretrained models (prerequisite)
    def download_pretrained_models_prerequisite():
        """Download and verify all required pretrained models upfront."""
        logger.info("=" * 80)
        logger.info("PREREQUISITE: Downloading and verifying pretrained models...")
        logger.info("=" * 80)
        
        results = download_pretrained_models(model_types)
        
        # Check if all required models succeeded
        failed_models = [m for m, success in results.items() if not success]
        if failed_models:
            error_msg = (
                f"Failed to download/verify pretrained weights for models: {failed_models}. "
                "Please check your network connection and try again."
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        logger.info("=" * 80)
        logger.info("✓ All pretrained models downloaded and verified successfully")
        logger.info("=" * 80)
        
        aggressive_gc(clear_cuda=True)
        return {"pretrained_models_verified": True}
    
    pipeline.register_stage(PipelineStage("download_pretrained_models", download_pretrained_models_prerequisite,
                                         dependencies=["load_data"]))
    
    # Stage 3: Create k-fold splits (shared)
    def create_kfold_splits():
        metadata = pipeline.artifacts["load_data"]["metadata"]
        
        logger.info("Creating %d-fold stratified cross-validation splits...", n_splits)
        folds = stratified_kfold(metadata, n_splits=n_splits, random_state=config.random_seed)
        
        logger.info("Created %d folds", len(folds))
        for i, (train_df, val_df) in enumerate(folds):
            logger.info("Fold %d: Train=%d, Val=%d", i+1, train_df.height, val_df.height)
        
        aggressive_gc(clear_cuda=False)
        return {"folds": folds}
    
    pipeline.register_stage(PipelineStage("create_kfold_splits", create_kfold_splits,
                                         dependencies=["load_data"]))
    
    # Stage 4: Generate shared augmentations (shared)
    def generate_shared_augmentations():
        folds = pipeline.artifacts["create_kfold_splits"]["folds"]
        
        video_cfg = VideoConfig(
            num_frames=config.num_frames,
            fixed_size=config.fixed_size,
            augmentation_config=config.augmentation_config,
            temporal_augmentation_config=config.temporal_augmentation_config,
        )
        
        # Global shared augmentations across models AND runs.
        # Cache path: project_root/intermediate_data/augmented_clips/shared/<config_hash>/
        project_root = config.project_root or os.getcwd()
        config_hash = config.compute_hash()
        global_aug_root = Path(project_root) / "intermediate_data" / "augmented_clips" / "shared"
        shared_aug_dir = global_aug_root / config_hash
        shared_aug_metadata_path = shared_aug_dir / "augmented_metadata.csv"
        os.makedirs(shared_aug_dir, exist_ok=True)
        
        # Get all unique videos from all folds
        all_train_videos = []
        for train_df, _ in folds:
            all_train_videos.append(train_df)
        all_train_df = pl.concat(all_train_videos).unique(subset=["video_path"])
        
        logger.info("Total unique videos: %d", all_train_df.height)
        
        def _generate_and_persist_shared_aug() -> pl.DataFrame:
            logger.info("Generating shared augmentations...")
            df = safe_execute(
                lambda: pregenerate_augmented_dataset(
                    all_train_df,
                    config.project_root,
                    video_cfg,
                    str(shared_aug_dir),
                    config.num_augmentations_per_video,
                ),
                context="shared augmentation generation",
                oom_retry=True,
                max_retries=1,
            )
            # Guard against accidental empty outputs
            if df is None or getattr(df, "height", 0) == 0:
                logger.warning(
                    "Shared augmentation generation returned empty DataFrame. "
                    "Falling back to using original videos without precomputed augmentations."
                )
                # Fallback: use original training videos as a "no-op" augmentation dataset
                # This still creates a valid metadata CSV so future runs can load it.
                fallback_df = all_train_df.select(["video_path", "label"]).with_columns(
                    [
                        pl.col("video_path").alias("original_video"),
                        pl.lit(0).alias("augmentation_idx"),
                    ]
                )
                if fallback_df.height == 0:
                    logger.error(
                        "Fallback augmentation dataset is also empty. "
                        "No training videos available after preprocessing."
                    )
                    raise RuntimeError(
                        "No training videos available to build shared augmentation dataset"
                    )
                fallback_df.write_csv(str(shared_aug_metadata_path))
                logger.info(
                    "✓ Created fallback shared augmentation metadata with %d entries (no precomputed clips).",
                    fallback_df.height,
                )
                return fallback_df

            df.write_csv(str(shared_aug_metadata_path))
            logger.info("✓ Generated %d shared augmented clips into %s", df.height, shared_aug_dir)
            return df

        if shared_aug_metadata_path.exists():
            logger.info("✓ Shared augmentations already exist. Loading from: %s", shared_aug_metadata_path)
            try:
                shared_aug_df = pl.read_csv(str(shared_aug_metadata_path))
                # Handle case where file exists but is empty or has no rows
                if shared_aug_df is None or shared_aug_df.height == 0:
                    logger.warning(
                        "Shared augmentations metadata at %s is empty. "
                        "Regenerating shared augmentations.",
                        shared_aug_metadata_path,
                    )
                    shared_aug_df = _generate_and_persist_shared_aug()
            except Exception as e:
                logger.error(
                    "Failed to read shared augmentations metadata from %s: %s. "
                    "Regenerating shared augmentations.",
                    shared_aug_metadata_path,
                    e,
                )
                shared_aug_df = _generate_and_persist_shared_aug()
        else:
            shared_aug_df = _generate_and_persist_shared_aug()
        
        aggressive_gc(clear_cuda=True)
        return {"shared_aug_df": shared_aug_df, "video_cfg": video_cfg}
    
    pipeline.register_stage(PipelineStage("generate_shared_augmentations", generate_shared_augmentations,
                                         dependencies=["create_kfold_splits"]))
    
    # Stage 5: Train all models sequentially
    def train_all_models():
        folds = pipeline.artifacts["create_kfold_splits"]["folds"]
        shared_aug_df = pipeline.artifacts["generate_shared_augmentations"]["shared_aug_df"]
        video_cfg = pipeline.artifacts["generate_shared_augmentations"]["video_cfg"]
        
        all_model_results = {}
        
        for model_idx, model_type in enumerate(model_types):
            logger.info("=" * 80)
            logger.info("MODEL %d/%d: %s", model_idx + 1, len(model_types), model_type)
            logger.info("=" * 80)
            
            # Check if model is already complete
            if _check_model_complete(model_type, config.output_dir, n_splits):
                logger.info("✓ Model %s already complete. Skipping.", model_type)
                continue
            
            # Get model-specific memory config
            model_mem_config = get_model_config(model_type)
            
            # Create model-specific config by copying base config and updating
            config_dict = config.to_dict()
            config_dict["model_type"] = model_type
            config_dict["batch_size"] = model_mem_config["batch_size"]
            config_dict["num_workers"] = model_mem_config["num_workers"]
            config_dict["num_frames"] = model_mem_config["num_frames"]
            config_dict["gradient_accumulation_steps"] = model_mem_config["gradient_accumulation_steps"]

            # Shared on-disk cache for handcrafted features across ALL models,
            # folds, and future runs. This prevents duplicated compute like:
            # "Extracting handcrafted features for N videos..." per model.
            feature_cache_dir = str(
                (Path(config.project_root or os.getcwd())
                 / "intermediate_data"
                 / "handcrafted_features")
            )
            model_specific_cfg = config_dict.get("model_specific_config") or {}
            if not isinstance(model_specific_cfg, dict):
                model_specific_cfg = {}
            model_specific_cfg.setdefault("feature_cache_dir", feature_cache_dir)
            config_dict["model_specific_config"] = model_specific_cfg
            
            # Create new RunConfig with model-specific settings
            model_config = RunConfig.from_dict(config_dict)
            
            # Check if it's a PyTorch model or baseline
            if is_pytorch_model(model_type):
                # PyTorch model training
                all_fold_results = []
                
                for fold_idx, (train_df, val_df) in enumerate(folds):
                    logger.info("=" * 80)
                    logger.info("FOLD %d/%d", fold_idx + 1, n_splits)
                    logger.info("=" * 80)
                    
                    # Filter shared augmentations for this fold
                    train_video_paths = set(train_df["video_path"].to_list())
                    aug_df = shared_aug_df.filter(
                        pl.col("original_video").is_in(list(train_video_paths))
                    )
                    
                    # Create datasets
                    train_ds = VideoDataset(aug_df, config.project_root, config=video_cfg, train=False)
                    val_ds = VideoDataset(val_df, config.project_root, config=video_cfg, train=False)
                    
                    # Create a fresh model instance for this fold
                    model = create_model(model_type, model_config)
                    
                    # Ultra aggressive GC after model creation
                    aggressive_gc(clear_cuda=True)
                    
                    # For CPU-only runs or when memory is constrained, use num_workers=0
                    # to avoid multiprocessing overhead and OOM from worker processes
                    effective_num_workers = model_config.num_workers
                    if not torch.cuda.is_available() or os.environ.get("FVC_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "y"):
                        effective_num_workers = 0
                        logger.info("Using num_workers=0 (CPU-only or test mode to avoid OOM)")
                    
                    # Create loaders
                    try:
                        balanced_sampler = make_balanced_batch_sampler(
                            aug_df,
                            batch_size=model_config.batch_size,
                            samples_per_class=model_config.batch_size // 2,
                            shuffle=True,
                            random_state=config.random_seed + fold_idx,
                        )
                        train_loader = DataLoader(
                            train_ds,
                            batch_sampler=balanced_sampler,
                            num_workers=effective_num_workers,
                            pin_memory=torch.cuda.is_available(),
                            collate_fn=variable_ar_collate,
                            persistent_workers=effective_num_workers > 0,
                            prefetch_factor=2 if effective_num_workers > 0 else None,
                        )
                    except Exception as e:
                        logger.warning("Balanced sampling failed: %s. Using regular sampling.", e)
                        train_loader = DataLoader(
                            train_ds,
                            batch_size=model_config.batch_size,
                            shuffle=True,
                            num_workers=effective_num_workers,
                            pin_memory=torch.cuda.is_available(),
                            collate_fn=variable_ar_collate,
                        )
                    
                    val_loader = DataLoader(
                        val_ds,
                        batch_size=model_config.batch_size,
                        shuffle=False,
                        num_workers=effective_num_workers,
                        pin_memory=torch.cuda.is_available(),
                        collate_fn=variable_ar_collate,
                    )
                    
                    # Move model to device
                    model = model.to(model_config.device)
                    
                    # Create optimizer and scheduler
                    optim_cfg = OptimConfig(
                        lr=model_config.learning_rate,
                        weight_decay=model_config.weight_decay,
                    )
                    
                    checkpoint_dir = os.path.join(config.output_dir, "checkpoints", model_type, f"fold_{fold_idx+1}")
                    train_cfg = TrainConfig(
                        num_epochs=model_config.num_epochs,
                        device=model_config.device,
                        log_interval=10,
                        use_amp=model_config.use_amp,
                        checkpoint_dir=checkpoint_dir,
                        early_stopping_patience=model_config.early_stopping_patience,
                        gradient_accumulation_steps=model_config.gradient_accumulation_steps,
                    )
                    
                    # Create fold-specific tracker
                    fold_tracker = ExperimentTracker(
                        os.path.join(config.output_dir, "models", model_type, f"fold_{fold_idx+1}"),
                        f"{config.run_id}_{model_type}_fold_{fold_idx+1}"
                    )
                    
                    fold_ckpt_manager = CheckpointManager(checkpoint_dir, f"{config.run_id}_{model_type}_fold_{fold_idx+1}")
                    
                    # Train model
                    # Wrap fold training in a retry loop with automatic batch-size reduction on OOM
                    current_batch_size = model_config.batch_size
                    while True:
                        try:
                            logger.info(
                                "Training %s fold %d with batch_size=%d",
                                model_type, fold_idx + 1, current_batch_size,
                            )
                            model = safe_execute(
                                fit_with_tracking,
                                model, train_loader, val_loader, optim_cfg, train_cfg,
                                fold_tracker, fold_ckpt_manager,
                                context=f"training {model_type} fold {fold_idx+1}",
                                oom_retry=True,
                                max_retries=1,
                            )
                            
                            # Evaluate
                            logits, labels = safe_execute(
                                collect_logits_and_labels,
                                model, val_loader, device=model_config.device,
                                context=f"evaluation {model_type} fold {fold_idx+1}",
                                oom_retry=True,
                                max_retries=1,
                            )
                            
                            # Ultra aggressive GC after evaluation to free memory
                            aggressive_gc(clear_cuda=True)
                            
                            metrics = basic_classification_metrics(logits, labels)
                            
                            # Clean up evaluation tensors immediately
                            del logits, labels
                            aggressive_gc(clear_cuda=True)
                            
                            fold_result = {"fold": fold_idx + 1, **metrics}
                            all_fold_results.append(fold_result)
                            
                            # Log to main tracker
                            for metric_name, value in metrics.items():
                                tracker.log_metric(
                                    step=len(all_fold_results),
                                    metric_name=f"{model_type}_{metric_name}",
                                    value=value,
                                    epoch=fold_idx + 1,
                                    phase="val"
                                )
                            
                            logger.info("Fold %d results: %s", fold_idx + 1, metrics)
                            break  # Successful run for this fold
                        
                        except Exception as e:
                            error_str = str(e).lower()
                            # Check for DataLoader worker crash (often OOM-related)
                            is_dataloader_worker_crash = (
                                "dataloader worker" in error_str and
                                ("exited unexpectedly" in error_str or "killed" in error_str)
                            )
                            
                            if is_dataloader_worker_crash and effective_num_workers > 0:
                                logger.warning(
                                    "DataLoader worker crashed (likely OOM). "
                                    "Retrying with num_workers=0 for %s fold %d.",
                                    model_type, fold_idx + 1,
                                )
                                effective_num_workers = 0
                                
                                # Rebuild loaders with num_workers=0
                                try:
                                    balanced_sampler = make_balanced_batch_sampler(
                                        aug_df,
                                        batch_size=model_config.batch_size,
                                        samples_per_class=model_config.batch_size // 2,
                                        shuffle=True,
                                        random_state=config.random_seed + fold_idx,
                                    )
                                    train_loader = DataLoader(
                                        train_ds,
                                        batch_sampler=balanced_sampler,
                                        num_workers=0,
                                        pin_memory=False,
                                        collate_fn=variable_ar_collate,
                                    )
                                except Exception as e_loader:
                                    logger.warning(
                                        "Balanced sampling rebuild failed: %s. "
                                        "Falling back to regular sampling.",
                                        e_loader,
                                    )
                                    train_loader = DataLoader(
                                        train_ds,
                                        batch_size=model_config.batch_size,
                                        shuffle=True,
                                        num_workers=0,
                                        pin_memory=False,
                                        collate_fn=variable_ar_collate,
                                    )
                                
                                val_loader = DataLoader(
                                    val_ds,
                                    batch_size=model_config.batch_size,
                                    shuffle=False,
                                    num_workers=0,
                                    pin_memory=False,
                                    collate_fn=variable_ar_collate,
                                )
                                
                                aggressive_gc(clear_cuda=True)
                                continue  # Retry with num_workers=0
                            
                            if check_oom_error(e):
                                handle_oom_error(e, f"{model_type} fold {fold_idx+1}")
                                # Reduce batch size and retry, down to minimum of 1
                                new_batch_size = max(1, current_batch_size // 2)
                                if new_batch_size == current_batch_size:
                                    logger.error(
                                        "OOM persists even at batch_size=%d for %s fold %d. Aborting.",
                                        current_batch_size, model_type, fold_idx + 1,
                                    )
                                    raise
                                logger.warning(
                                    "Reducing batch_size from %d to %d for %s fold %d due to OOM.",
                                    current_batch_size, new_batch_size, model_type, fold_idx + 1,
                                )
                                current_batch_size = new_batch_size
                                model_config.batch_size = new_batch_size
                                
                                # Rebuild loaders with smaller batch size
                                try:
                                    balanced_sampler = make_balanced_batch_sampler(
                                        aug_df,
                                        batch_size=model_config.batch_size,
                                        samples_per_class=model_config.batch_size // 2,
                                        shuffle=True,
                                        random_state=config.random_seed + fold_idx,
                                    )
                                    train_loader = DataLoader(
                                        train_ds,
                                        batch_sampler=balanced_sampler,
                                        num_workers=effective_num_workers,
                                        pin_memory=torch.cuda.is_available(),
                                        collate_fn=variable_ar_collate,
                                        persistent_workers=effective_num_workers > 0,
                                        prefetch_factor=2 if effective_num_workers > 0 else None,
                                    )
                                except Exception as e_loader:
                                    logger.warning(
                                        "Balanced sampling rebuild failed after OOM: %s. "
                                        "Falling back to regular sampling with batch_size=%d.",
                                        e_loader, model_config.batch_size,
                                    )
                                    train_loader = DataLoader(
                                        train_ds,
                                        batch_size=model_config.batch_size,
                                        shuffle=True,
                                        num_workers=effective_num_workers,
                                        pin_memory=torch.cuda.is_available(),
                                        collate_fn=variable_ar_collate,
                                    )
                                
                                val_loader = DataLoader(
                                    val_ds,
                                    batch_size=model_config.batch_size,
                                    shuffle=False,
                                    num_workers=effective_num_workers,
                                    pin_memory=torch.cuda.is_available(),
                                    collate_fn=variable_ar_collate,
                                )
                                
                                aggressive_gc(clear_cuda=True)
                                continue  # Retry with smaller batch size
                            
                            logger.error("Fold %d failed: %s", fold_idx + 1, str(e))
                            raise
                    
                    # Ultra aggressive cleanup after each fold
                    # Move model to CPU before deletion to be extra safe
                    try:
                        if isinstance(model, torch.nn.Module):
                            model.to("cpu")
                            # Clear any cached activations or buffers
                            model.eval()  # Set to eval mode to disable dropout/batchnorm
                            for param in model.parameters():
                                param.grad = None  # Clear gradients
                    except Exception:
                        pass
                    
                    # Delete all references
                    del model, train_loader, val_loader, train_ds, val_ds
                    
                    # Multiple aggressive GC passes to ensure complete cleanup
                    for _ in range(2):
                        aggressive_gc(clear_cuda=True)
                
                # Save fold results
                if all_fold_results:
                    results_df = pl.DataFrame(all_fold_results)
                    results_path = os.path.join(config.output_dir, "models", model_type, "fold_results.csv")
                    os.makedirs(os.path.dirname(results_path), exist_ok=True)
                    results_df.write_csv(results_path)
                    
                    # Compute average metrics
                    avg_metrics = {}
                    for col in results_df.columns:
                        if col != "fold":
                            avg_metrics[f"avg_{col}"] = float(results_df[col].mean())
                            avg_metrics[f"std_{col}"] = float(results_df[col].std())
                    
                    all_model_results[model_type] = avg_metrics
                    
                    # Mark as complete
                    completion_file = os.path.join(config.output_dir, "models", model_type, "training_complete.pt")
                    torch.save({"complete": True}, completion_file)
                
            else:
                # Baseline model training (sklearn-style)
                all_fold_results = []
                
                for fold_idx, (train_df, val_df) in enumerate(folds):
                    logger.info("Fold %d/%d", fold_idx + 1, n_splits)
                    
                    # Create fresh model instance for this fold
                    model = create_model(model_type, model_config)
                    
                    # Train baseline
                    metrics = _train_baseline_model(
                        model, model_type, train_df, val_df,
                        config.project_root, config.output_dir,
                        fold_idx, tracker
                    )
                    
                    all_fold_results.append({"fold": fold_idx + 1, **metrics})
                    
                    # Cleanup
                    del model
                    aggressive_gc(clear_cuda=False)
                
                # Save fold results
                if all_fold_results:
                    results_df = pl.DataFrame(all_fold_results)
                    results_path = os.path.join(config.output_dir, "models", model_type, "fold_results.csv")
                    os.makedirs(os.path.dirname(results_path), exist_ok=True)
                    results_df.write_csv(results_path)
                    
                    # Compute average metrics
                    avg_metrics = {}
                    for col in results_df.columns:
                        if col != "fold":
                            avg_metrics[f"avg_{col}"] = float(results_df[col].mean())
                            avg_metrics[f"std_{col}"] = float(results_df[col].std())
                    
                    all_model_results[model_type] = avg_metrics
                    
                    # Mark as complete
                    completion_file = os.path.join(config.output_dir, "models", model_type, "training_complete.pt")
                    torch.save({"complete": True}, completion_file)
            
            # Aggressive cleanup after each model
            aggressive_gc(clear_cuda=True)
            log_memory_stats(f"after {model_type}")
        
        return {"model_results": all_model_results}
    
    pipeline.register_stage(PipelineStage("train_all_models", train_all_models,
                                         dependencies=["generate_shared_augmentations"]))
    
    return pipeline


__all__ = ["build_multimodel_pipeline"]

