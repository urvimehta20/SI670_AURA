"""
MLOps Pipeline with K-Fold Cross-Validation: Prevents overfitting/underfitting.

This module extends the MLOps pipeline with stratified K-fold cross-validation.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional
import polars as pl
import torch

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
    maybe_limit_to_small_test_subset,
)
from .video_modeling import VideoConfig, VideoDataset, variable_ar_collate

from .video_augmentation_pipeline import pregenerate_augmented_dataset
from .video_training import OptimConfig, TrainConfig
from .video_modeling import PretrainedInceptionVideoModel
from .video_metrics import collect_logits_and_labels, basic_classification_metrics
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def build_kfold_pipeline(config: RunConfig, tracker: ExperimentTracker, 
                        n_splits: int = 5, ckpt_manager: Optional[CheckpointManager] = None) -> MLOpsPipeline:
    """
    Build MLOps pipeline with K-fold cross-validation.
    
    Args:
        config: Run configuration
        tracker: Experiment tracker
        n_splits: Number of folds for cross-validation
    """
    pipeline = MLOpsPipeline(config, tracker)
    
    # Stage 1: Load and validate data
    def load_data():
        meta_df = load_metadata(config.data_csv)
        logger.info("Loaded %d video records", meta_df.height)
        
        # Filter existing videos
        filtered_df = filter_existing_videos(meta_df, config.project_root, check_frames=False)
        logger.info("After filtering: %d valid videos", filtered_df.height)

        # Optional: limit to a tiny balanced subset for SLURM test runs.
        # Use 5 per class by default so 5-fold CV has enough data
        # in each training/validation split while still being tiny.
        filtered_df = maybe_limit_to_small_test_subset(filtered_df, max_per_class=5)

        aggressive_gc(clear_cuda=False)  # GC after data loading
        
        return {"metadata": filtered_df}
    
    pipeline.register_stage(PipelineStage("load_data", load_data))
    
    # Stage 2: Create K-fold splits
    def create_kfold_splits():
        metadata = pipeline.artifacts["load_data"]["metadata"]
        
        logger.info("Creating %d-fold stratified cross-validation splits...", n_splits)
        folds = stratified_kfold(metadata, n_splits=n_splits, random_state=config.random_seed)
        
        logger.info("Created %d folds", len(folds))
        for i, (train_df, val_df) in enumerate(folds):
            logger.info("Fold %d: Train=%d, Val=%d", i+1, train_df.height, val_df.height)
        
        # Register data version
        version_manager = DataVersionManager(config.output_dir)
        config_hash = config.compute_hash()
        version_manager.register_split("kfold_splits", config_hash, 
                                      str(Path(config.output_dir) / "kfold_splits"),
                                      {"n_splits": n_splits, "folds": len(folds)})
        
        aggressive_gc(clear_cuda=False)
        
        return {"folds": folds}
    
    pipeline.register_stage(PipelineStage("create_kfold_splits", create_kfold_splits,
                                         dependencies=["load_data"]))
    
    # Stage 3: Train K-fold models
    def train_kfold_models():
        folds = pipeline.artifacts["create_kfold_splits"]["folds"]
        
        video_cfg = VideoConfig(
            num_frames=config.num_frames,
            fixed_size=config.fixed_size,
            augmentation_config=config.augmentation_config,
            temporal_augmentation_config=config.temporal_augmentation_config,
        )
        
        all_fold_results = []
        
        # Generate augmentations ONCE for all videos (shared across folds AND runs)
        # Global cache location (per augmentation config) under project_root:
        #   intermediate_data/augmented_clips/shared/<config_hash>/
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
        
        logger.info("=" * 80)
        logger.info("GENERATING SHARED AUGMENTATIONS (used across all folds)")
        logger.info("=" * 80)
        logger.info("Total unique videos: %d", all_train_df.height)
        
        def _generate_and_persist_shared_aug() -> pl.DataFrame:
            logger.info("Generating shared augmentations (this may take a while)...")
            # Generate augmentations with OOM handling
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
            if df is None or getattr(df, "height", 0) == 0:
                logger.warning(
                    "Shared augmentation generation for K-fold returned empty DataFrame. "
                    "Falling back to using original videos without precomputed augmentations."
                )
                # Fallback: use original training videos as a "no-op" augmentation dataset
                fallback_df = all_train_df.select(["video_path", "label"]).with_columns(
                    [
                        pl.col("video_path").alias("original_video"),
                        pl.lit(0).alias("augmentation_idx"),
                    ]
                )
                if fallback_df.height == 0:
                    logger.error(
                        "Fallback augmentation dataset for K-fold is also empty. "
                        "No training videos available after preprocessing."
                    )
                    raise RuntimeError(
                        "No training videos available to build shared augmentation dataset"
                    )
                fallback_df.write_csv(str(shared_aug_metadata_path))
                logger.info(
                    "✓ Created fallback shared augmentation metadata for K-fold with %d entries "
                    "(no precomputed clips).",
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
                if shared_aug_df is None or shared_aug_df.height == 0:
                    logger.warning(
                        "Shared augmentations metadata at %s is empty for K-fold. "
                        "Regenerating shared augmentations.",
                        shared_aug_metadata_path,
                    )
                    shared_aug_df = _generate_and_persist_shared_aug()
            except Exception as e:
                logger.error(
                    "Failed to read shared augmentations metadata from %s for K-fold: %s. "
                    "Regenerating shared augmentations.",
                    shared_aug_metadata_path,
                    e,
                )
                shared_aug_df = _generate_and_persist_shared_aug()
        else:
            shared_aug_df = _generate_and_persist_shared_aug()
        
        aggressive_gc(clear_cuda=True)
        
        for fold_idx, (train_df, val_df) in enumerate(folds):
            logger.info("=" * 80)
            logger.info("FOLD %d/%d", fold_idx + 1, n_splits)
            logger.info("=" * 80)
            
            fold_tracker = ExperimentTracker(
                os.path.join(config.output_dir, f"fold_{fold_idx+1}"),
                f"{config.run_id}_fold_{fold_idx+1}"
            )
            
            try:
                # Filter shared augmentations to only include videos in this fold's training set
                # Check which column name is used for original video path
                original_col = "original_video" if "original_video" in shared_aug_df.columns else "original_video_path"
                train_video_paths = set(train_df["video_path"].to_list())
                aug_df = shared_aug_df.filter(
                    pl.col(original_col).is_in(list(train_video_paths))
                )
                
                logger.info("Fold %d: Using %d augmented clips from %d training videos", 
                           fold_idx + 1, aug_df.height, train_df.height)
                
                aggressive_gc(clear_cuda=True)
                
                # Create datasets
                train_ds = VideoDataset(aug_df, config.project_root, config=video_cfg, train=False)
                val_ds = VideoDataset(val_df, config.project_root, config=video_cfg, train=False)
                
                # Create loaders
                from .video_data import make_balanced_batch_sampler
                
                # For CPU-only runs or when memory is constrained, use num_workers=0
                # to avoid multiprocessing overhead and OOM from worker processes
                effective_num_workers = config.num_workers
                if not torch.cuda.is_available() or os.environ.get("FVC_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "y"):
                    effective_num_workers = 0
                    logger.info("Using num_workers=0 (CPU-only or test mode to avoid OOM)")
                
                try:
                    balanced_sampler = make_balanced_batch_sampler(
                        aug_df,
                        batch_size=config.batch_size,
                        samples_per_class=config.batch_size // 2,
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
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=effective_num_workers,
                        pin_memory=torch.cuda.is_available(),
                        collate_fn=variable_ar_collate,
                    )
                
                val_loader = DataLoader(
                    val_ds,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=effective_num_workers,
                    pin_memory=torch.cuda.is_available(),
                    collate_fn=variable_ar_collate,
                )
                
                aggressive_gc(clear_cuda=True)
                
                # Initialize model
                model = PretrainedInceptionVideoModel(freeze_backbone=False)
                model.to(config.device)
                
                # Training config
                optim_cfg = OptimConfig(
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay,
                )
                
                checkpoint_dir = os.path.join(config.output_dir, "checkpoints", f"fold_{fold_idx+1}")
                train_cfg = TrainConfig(
                    num_epochs=config.num_epochs,
                    device=config.device,
                    log_interval=10,
                    use_amp=config.use_amp,
                    checkpoint_dir=checkpoint_dir,
                    early_stopping_patience=config.early_stopping_patience,
                    gradient_accumulation_steps=config.gradient_accumulation_steps,
                )
                
                # Checkpoint manager for this fold
                fold_ckpt_manager = CheckpointManager(checkpoint_dir, f"{config.run_id}_fold_{fold_idx+1}")
                
                # Train model (with OOM handling)
                model = safe_execute(
                    lambda: fit_with_tracking(
                        model, train_loader, val_loader, optim_cfg, train_cfg,
                        fold_tracker, fold_ckpt_manager
                    ),
                    context=f"training fold {fold_idx+1}",
                    oom_retry=True,
                    max_retries=1,
                )
                
                # Evaluate (with OOM handling)
                logits, labels = safe_execute(
                    collect_logits_and_labels,
                    model, val_loader, device=config.device,
                    context=f"evaluation fold {fold_idx+1}",
                    oom_retry=True,
                    max_retries=1,
                )
                metrics = basic_classification_metrics(logits, labels)
                
                fold_result = {
                    "fold": fold_idx + 1,
                    **metrics
                }
                all_fold_results.append(fold_result)
                
                # Log fold metrics to main tracker
                for metric_name, value in metrics.items():
                    tracker.log_metric(
                        step=fold_idx + 1,
                        metric_name=metric_name,
                        value=value,
                        epoch=fold_idx + 1,
                        phase="val"
                    )
                
                logger.info("Fold %d results: %s", fold_idx + 1, metrics)
                
                # Aggressive cleanup after each fold
                del model, train_loader, val_loader, train_ds, val_ds
                aggressive_gc(clear_cuda=True)
                
            except Exception as e:
                if check_oom_error(e):
                    handle_oom_error(e, f"fold {fold_idx+1}")
                logger.error("Fold %d failed: %s", fold_idx + 1, str(e))
                aggressive_gc(clear_cuda=True)
                raise
        
        # Compute average metrics across folds
        avg_metrics = {}
        if all_fold_results:
            results_df = pl.DataFrame(all_fold_results)
            for col in results_df.columns:
                if col != "fold":
                    avg_metrics[f"avg_{col}"] = float(results_df[col].mean())
                    avg_metrics[f"std_{col}"] = float(results_df[col].std())
            
            logger.info("=" * 80)
            logger.info("K-FOLD CROSS-VALIDATION RESULTS")
            logger.info("=" * 80)
            for metric, value in avg_metrics.items():
                logger.info("%s: %.4f", metric, value)
            logger.info("=" * 80)
            
            # Save results
            results_path = os.path.join(config.output_dir, "kfold_results.feather")
            results_df.write_ipc(results_path)
            logger.info("Saved K-fold results to %s", results_path)
        
        return {"fold_results": all_fold_results, "avg_metrics": avg_metrics}
    
    pipeline.register_stage(PipelineStage("train_kfold_models", train_kfold_models,
                                         dependencies=["create_kfold_splits"]))
    
    # Store checkpoint manager for pipeline execution
    if ckpt_manager:
        pipeline.ckpt_manager = ckpt_manager
    
    return pipeline


__all__ = ["build_kfold_pipeline"]

