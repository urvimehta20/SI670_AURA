"""
MLOps Pipeline: Orchestrated workflow with dependency management and validation.

This module provides:
- Pipeline orchestration with clear stages
- Dependency checking and validation
- Data validation
- Resume capability
- Error handling and recovery
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Tuple
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
from .video_data import (
    load_metadata,
    filter_existing_videos,
    train_val_test_split,
    SplitConfig,
    stratified_kfold,
    maybe_limit_to_small_test_subset,
)
from .video_modeling import VideoConfig, VideoDataset, variable_ar_collate
from pathlib import Path

from .video_augmentation_pipeline import pregenerate_augmented_dataset
from .video_training import OptimConfig, TrainConfig, fit
from .video_modeling import PretrainedInceptionVideoModel
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PipelineStage:
    """Represents a pipeline stage with dependencies and validation."""
    
    def __init__(self, name: str, func: Callable, dependencies: List[str] = None,
                 validate: Optional[Callable] = None, checkpoint: bool = True):
        self.name = name
        self.func = func
        self.dependencies = dependencies or []
        self.validate = validate
        self.checkpoint = checkpoint
        self.completed = False
        self.result: Any = None
    
    def can_run(self, completed_stages: set) -> bool:
        """Check if all dependencies are satisfied."""
        return all(dep in completed_stages for dep in self.dependencies)
    
    def run(self, *args, ckpt_manager: Optional[CheckpointManager] = None, **kwargs) -> Any:
        """Execute the stage function with checkpointing and OOM handling."""
        logger.info("=" * 80)
        logger.info("STAGE: %s", self.name)
        logger.info("=" * 80)
        
        # Log memory before stage
        log_memory_stats(f"before {self.name}")
        
        # Try to resume from checkpoint
        if ckpt_manager and self.checkpoint:
            checkpoint_data = ckpt_manager.load_stage_checkpoint(self.name)
            if checkpoint_data:
                logger.info("Resuming stage '%s' from checkpoint", self.name)
                self.completed = True
                self.result = checkpoint_data
                return checkpoint_data
        
        try:
            # Execute with OOM handling
            result = safe_execute(
                self.func,
                *args,
                context=f"stage {self.name}",
                oom_retry=True,
                max_retries=1,
                **kwargs
            )
            
            self.completed = True
            self.result = result
            
            # Validate result
            if self.validate:
                self.validate(result)
            
            # Save checkpoint
            if ckpt_manager and self.checkpoint:
                # Only save serializable data
                checkpoint_data = {
                    'result': result,
                    'completed': True,
                }
                ckpt_manager.save_stage_checkpoint(self.name, checkpoint_data)
            
            # Aggressive GC after stage
            aggressive_gc(clear_cuda=True)
            
            # Log memory after stage
            log_memory_stats(f"after {self.name}")
            
            logger.info("✓ Stage '%s' completed successfully", self.name)
            return result
        
        except Exception as e:
            # Handle OOM specifically
            if check_oom_error(e):
                handle_oom_error(e, f"stage {self.name}")
            
            logger.error("✗ Stage '%s' failed: %s", self.name, str(e))
            
            # Aggressive cleanup on failure
            aggressive_gc(clear_cuda=True)
            
            raise


class MLOpsPipeline:
    """Orchestrated ML pipeline with dependency management."""
    
    def __init__(self, config: RunConfig, tracker: ExperimentTracker):
        self.config = config
        self.tracker = tracker
        self.stages: Dict[str, PipelineStage] = {}
        self.completed_stages: set = set()
        self.artifacts: Dict[str, Any] = {}
        self.ckpt_manager: Optional[CheckpointManager] = None
    
    def register_stage(self, stage: PipelineStage) -> None:
        """Register a pipeline stage."""
        self.stages[stage.name] = stage
    
    def run_pipeline(self, ckpt_manager: Optional[CheckpointManager] = None) -> Dict[str, Any]:
        """Execute pipeline stages in dependency order with checkpointing."""
        logger.info("Starting MLOps pipeline (run_id: %s)", self.config.run_id)
        
        # Log configuration
        self.tracker.log_config(self.config)
        
        # Use stored checkpoint manager or provided one
        if ckpt_manager is None:
            ckpt_manager = self.ckpt_manager
        
        # Execute stages
        while len(self.completed_stages) < len(self.stages):
            progress_made = False
            
            for stage_name, stage in self.stages.items():
                if stage_name in self.completed_stages:
                    continue
                
                if stage.can_run(self.completed_stages):
                    try:
                        result = stage.run(ckpt_manager=ckpt_manager)
                        self.completed_stages.add(stage_name)
                        self.artifacts[stage_name] = result
                        progress_made = True
                        
                        # Aggressive GC between stages
                        aggressive_gc(clear_cuda=True)
                    except Exception as e:
                        logger.error("Pipeline failed at stage '%s': %s", stage_name, e)
                        
                        # Final cleanup on failure
                        aggressive_gc(clear_cuda=True)
                        raise
            
            if not progress_made:
                # Circular dependency or missing stage
                remaining = set(self.stages.keys()) - self.completed_stages
                raise RuntimeError(
                    f"Cannot make progress. Remaining stages: {remaining}. "
                    f"Check dependencies."
                )
        
        logger.info("Pipeline completed successfully")
        return self.artifacts


def build_mlops_pipeline(config: RunConfig, tracker: ExperimentTracker, 
                         use_kfold: bool = True, n_splits: int = 5) -> MLOpsPipeline:
    """
    Build the complete MLOps pipeline with all stages.
    
    Args:
        config: Run configuration
        tracker: Experiment tracker
        use_kfold: If True, use K-fold cross-validation instead of single split
        n_splits: Number of folds for K-fold cross-validation
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
        # Use 5 per class by default so 5-fold CV and multi-model runs
        # have enough data while still being small.
        filtered_df = maybe_limit_to_small_test_subset(filtered_df, max_per_class=5)
        
        return {"metadata": filtered_df}
    
    pipeline.register_stage(PipelineStage("load_data", load_data))
    
    # Stage 2: Create data splits
    def create_splits():
        metadata = pipeline.artifacts["load_data"]["metadata"]
        
        splits = train_val_test_split(
            metadata,
            SplitConfig(
                train_size=config.train_split,
                val_size=config.val_split,
                test_size=config.test_split
            ),
            save_dir=os.path.join(config.output_dir, "splits"),
        )
        
        train_df = splits["train"]
        val_df = splits["val"]
        
        logger.info("Train: %d, Val: %d", train_df.height, val_df.height)
        
        # Register data version
        version_manager = DataVersionManager(config.output_dir)
        config_hash = config.compute_hash()
        version_manager.register_split("train", config_hash, 
                                      str(Path(config.output_dir) / "splits" / "train.feather"),
                                      {"count": train_df.height})
        version_manager.register_split("val", config_hash,
                                      str(Path(config.output_dir) / "splits" / "val.feather"),
                                      {"count": val_df.height})
        
        return {"train": train_df, "val": val_df}
    
    pipeline.register_stage(PipelineStage("create_splits", create_splits, 
                                         dependencies=["load_data"]))
    
    # Stage 3: Generate augmentations
    def generate_augmentations():
        train_df = pipeline.artifacts["create_splits"]["train"]
        
        video_cfg = VideoConfig(
            num_frames=config.num_frames,
            fixed_size=config.fixed_size,
            augmentation_config=config.augmentation_config,
            temporal_augmentation_config=config.temporal_augmentation_config,
        )
        
        # Global augmentation cache across runs:
        # project_root/intermediate_data/augmented_clips/shared/<config_hash>/
        project_root = config.project_root or os.getcwd()
        config_hash = config.compute_hash()
        global_aug_root = Path(project_root) / "intermediate_data" / "augmented_clips" / "shared"
        aug_dir = global_aug_root / config_hash
        os.makedirs(aug_dir, exist_ok=True)
        
        # Check if augmentations exist in the global cache
        metadata_path = aug_dir / "augmented_train_metadata.csv"
        if metadata_path.exists() and len(list(aug_dir.glob("*.pt"))) > 0:
            logger.info("Augmentations already exist, loading metadata from %s...", metadata_path)
            aug_df = pl.read_csv(str(metadata_path))
        else:
            logger.info("Generating augmentations...")
            aug_df = pregenerate_augmented_dataset(
                train_df,
                config.project_root,
                video_cfg,
                output_dir=str(aug_dir),
                num_augmentations_per_video=config.num_augmentations_per_video,
            )
            aug_df.write_csv(str(metadata_path))
        
        # Register augmentation version
        version_manager = DataVersionManager(config.output_dir)
        config_hash = config.compute_hash()
        version_manager.register_augmentation(
            config_hash, str(aug_dir),
            {"count": aug_df.height, "augmentations_per_video": config.num_augmentations_per_video}
        )
        
        return {"augmented_train": aug_df}
    
    pipeline.register_stage(PipelineStage("generate_augmentations", generate_augmentations,
                                         dependencies=["create_splits"]))
    
    # Stage 4: Create datasets
    def create_datasets():
        aug_train_df = pipeline.artifacts["generate_augmentations"]["augmented_train"]
        val_df = pipeline.artifacts["create_splits"]["val"]
        
        video_cfg = VideoConfig(
            num_frames=config.num_frames,
            fixed_size=config.fixed_size,
        )
        
        train_ds = VideoDataset(aug_train_df, config.project_root, config=video_cfg, train=False)
        val_ds = VideoDataset(val_df, config.project_root, config=video_cfg, train=False)
        
        logger.info("Train dataset: %d samples", len(train_ds))
        logger.info("Val dataset: %d samples", len(val_ds))
        
        return {"train_dataset": train_ds, "val_dataset": val_ds}
    
    pipeline.register_stage(PipelineStage("create_datasets", create_datasets,
                                         dependencies=["generate_augmentations"]))
    
    # Stage 5: Create data loaders
    def create_loaders():
        train_ds = pipeline.artifacts["create_datasets"]["train_dataset"]
        val_ds = pipeline.artifacts["create_datasets"]["val_dataset"]
        
        from .video_data import make_balanced_batch_sampler
        
        # For CPU-only runs or when memory is constrained, use num_workers=0
        # to avoid multiprocessing overhead and OOM from worker processes
        effective_num_workers = config.num_workers
        if not torch.cuda.is_available() or os.environ.get("FVC_TEST_MODE", "").strip().lower() in ("1", "true", "yes", "y"):
            effective_num_workers = 0
            logger.info("Using num_workers=0 (CPU-only or test mode to avoid OOM)")
        
        # Try balanced sampling
        try:
            balanced_sampler = make_balanced_batch_sampler(
                pipeline.artifacts["generate_augmentations"]["augmented_train"],
                batch_size=config.batch_size,
                samples_per_class=config.batch_size // 2,
                shuffle=True,
                random_state=config.random_seed,
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
        
        return {"train_loader": train_loader, "val_loader": val_loader}
    
    pipeline.register_stage(PipelineStage("create_loaders", create_loaders,
                                         dependencies=["create_datasets"]))
    
    # Stage 6: Initialize model
    def initialize_model():
        model = PretrainedInceptionVideoModel(freeze_backbone=False)
        model.to(config.device)
        
        # Log model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("Model: %d total params, %d trainable", total_params, trainable_params)
        
        tracker.log_metadata({
            "model_total_params": total_params,
            "model_trainable_params": trainable_params,
        })
        
        return {"model": model}
    
    pipeline.register_stage(PipelineStage("initialize_model", initialize_model))
    
    # Stage 7: Train model
    def train_model():
        model = pipeline.artifacts["initialize_model"]["model"]
        train_loader = pipeline.artifacts["create_loaders"]["train_loader"]
        val_loader = pipeline.artifacts["create_loaders"]["val_loader"]
        
        optim_cfg = OptimConfig(
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
        train_cfg = TrainConfig(
            num_epochs=config.num_epochs,
            device=config.device,
            log_interval=10,
            use_amp=config.use_amp,
            checkpoint_dir=checkpoint_dir,
            early_stopping_patience=config.early_stopping_patience,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
        )
        
        # Enhanced checkpoint manager
        ckpt_manager = CheckpointManager(checkpoint_dir, config.run_id)
        
        # Delegate optimizer/scheduler creation and resume logic to fit_with_tracking
        model = fit_with_tracking(
            model,
            train_loader,
            val_loader,
            optim_cfg,
            train_cfg,
            tracker,
            ckpt_manager,
        )
        
        return {"trained_model": model}
    
    pipeline.register_stage(PipelineStage("train_model", train_model,
                                         dependencies=["initialize_model", "create_loaders"]))
    
    return pipeline


def fit_with_tracking(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optim_cfg: OptimConfig,
    train_cfg: TrainConfig,
    tracker: ExperimentTracker,
    ckpt_manager: CheckpointManager,
) -> torch.nn.Module:
    """Enhanced fit function with experiment tracking and aggressive GC."""
    from .video_training import train_one_epoch, evaluate, build_optimizer, build_scheduler
    
    device = train_cfg.device
    model.to(device)
    model.train()
    
    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer)
    
    # Try to resume from checkpoint
    start_epoch = ckpt_manager.resume_from_latest(model, optimizer, scheduler)
    if start_epoch:
        logger.info("Resuming from epoch %d", start_epoch)
    
    best_val_acc = 0.0
    first_epoch = start_epoch or 1
    
    for epoch in range(first_epoch, train_cfg.num_epochs + 1):
        # Aggressive GC at start of each epoch
        aggressive_gc(clear_cuda=True)
        log_memory_stats(f"epoch {epoch} start")
        
        try:
            # Train
            train_loss = safe_execute(
                train_one_epoch,
                model, train_loader, optimizer, device=device,
                use_class_weights=train_cfg.use_class_weights,
                epoch=epoch,
                log_interval=train_cfg.log_interval,
                gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
                context=f"training epoch {epoch}",
                oom_retry=True,
                max_retries=1,
            )
            
            tracker.log_epoch_metrics(epoch, {"loss": train_loss}, phase="train")
            
            # Step scheduler at the end of the epoch (AFTER optimizer.step() calls in train_one_epoch)
            # This ensures scheduler.step() is called after optimizer.step() in the same epoch
            if train_loss is not None:
                scheduler.step()
            
            # Aggressive GC after training
            aggressive_gc(clear_cuda=True)
            
            # Validate
            if val_loader is not None:
                # evaluate returns a tuple, so we need to handle it properly
                val_result = safe_execute(
                    lambda: evaluate(model, val_loader, device=device),
                    context=f"validation epoch {epoch}",
                    oom_retry=True,
                    max_retries=1,
                )
                val_loss, val_acc = val_result
                
                tracker.log_epoch_metrics(epoch, {"loss": val_loss, "accuracy": val_acc}, phase="val")
                
                is_best = val_acc > best_val_acc
                if is_best:
                    best_val_acc = val_acc
                
                # Save checkpoint
                ckpt_manager.save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc},
                    is_best=is_best
                )
            
            # Ultra aggressive GC after epoch
            aggressive_gc(clear_cuda=True)
            log_memory_stats(f"epoch {epoch} end")
            
            # Additional GC pass after logging to ensure maximum cleanup
            aggressive_gc(clear_cuda=True)
        
        except Exception as e:
            if check_oom_error(e):
                handle_oom_error(e, f"epoch {epoch}")
            aggressive_gc(clear_cuda=True)
            raise
    
    return model


__all__ = [
    "PipelineStage",
    "MLOpsPipeline",
    "build_mlops_pipeline",
    "fit_with_tracking",
]

