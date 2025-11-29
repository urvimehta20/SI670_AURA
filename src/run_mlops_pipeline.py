#!/usr/bin/env python3
"""
MLOps Pipeline Runner: Execute the optimized MLOps workflow.

This script demonstrates the new MLOps pipeline with:
- Experiment tracking
- Configuration versioning
- Checkpoint management with resume capability
- Data versioning
- Structured metrics logging
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mlops_runner")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.mlops_core import RunConfig, ExperimentTracker, create_run_directory
from lib.mlops_pipeline import build_mlops_pipeline
from lib.mlops_pipeline_kfold import build_kfold_pipeline
from lib.mlops_pipeline_multimodel import build_multimodel_pipeline
from lib.cleanup_utils import cleanup_runs_and_logs
from lib.model_factory import list_available_models


def main():
    """Run the MLOps pipeline."""
    # Detect project root
    if "SLURM_SUBMIT_DIR" in os.environ:
        project_root = os.environ["SLURM_SUBMIT_DIR"]
    else:
        project_root = str(Path(__file__).parent.parent)
    
    project_root = os.path.abspath(project_root)
    data_csv = os.path.join(project_root, "data", "video_index_input.csv")
    
    # Clean up previous runs, logs, models, and intermediate_data for fresh run
    logger.info("Cleaning up previous runs, logs, models, and intermediate_data for fresh start...")
    cleanup_runs_and_logs(project_root, keep_models=False, keep_intermediate_data=False)
    
    # Create run directory
    output_base = os.path.join(project_root, "runs")
    run_dir, run_id = create_run_directory(output_base, "fvc_binary_classifier")
    
    # Set up per-run file logging so we always have a pipeline log, even if SLURM output is missing
    run_log_dir = Path(run_dir) / "logs"
    run_log_dir.mkdir(parents=True, exist_ok=True)
    run_log_path = run_log_dir / "pipeline.log"
    file_handler = logging.FileHandler(run_log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    logger.info("=" * 80)
    logger.info("MLOps Pipeline Run")
    logger.info("=" * 80)
    logger.info("Run ID: %s", run_id)
    logger.info("Run Directory: %s", run_dir)
    logger.info("Project Root: %s", project_root)
    
    # Create experiment tracker
    tracker = ExperimentTracker(run_dir, run_id)
    
    # Create run configuration
    config = RunConfig(
        run_id=run_id,
        experiment_name="fvc_binary_classifier",
        description="FVC binary video classification with comprehensive augmentations",
        tags=["video_classification", "binary", "augmentations"],
        
        # Data config
        data_csv=data_csv,
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        random_seed=42,
        
        # Video config (reduced for memory efficiency)
        num_frames=8,  # Reduced from 16 to 8 to prevent OOM
        fixed_size=224,
        augmentation_config={
            'rotation_degrees': 15.0,
            'rotation_p': 0.5,
            'affine_p': 0.3,
            'gaussian_noise_std': 0.1,
            'gaussian_noise_p': 0.3,
            'gaussian_blur_p': 0.3,
            'cutout_p': 0.5,
            'cutout_max_size': 32,
            'elastic_transform_p': 0.2,
            'color_jitter_brightness': 0.3,
            'color_jitter_contrast': 0.3,
            'color_jitter_saturation': 0.3,
            'color_jitter_hue': 0.1,
        },
        temporal_augmentation_config={
            'frame_drop_prob': 0.1,
            'frame_dup_prob': 0.1,
            'reverse_prob': 0.1,
        },
        num_augmentations_per_video=3,
        
        # Training config (reduced for memory efficiency)
        batch_size=8,  # Reduced from 32 to 8 to prevent OOM
        num_epochs=20,
        learning_rate=1e-4,
        weight_decay=1e-4,
        gradient_accumulation_steps=2,  # Increased to maintain effective batch size
        early_stopping_patience=5,
        
        # System config (reduced for memory efficiency)
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        num_workers=2,  # Reduced from 4 to 2 to reduce memory usage
        use_amp=True,
        
        # Paths
        project_root=project_root,
        output_dir=run_dir,
    )
    
    # Log system metadata
    import torch
    import platform
    tracker.log_metadata({
        "python_version": sys.version,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    })
    
    # Build and run pipeline
    try:
        # Configuration: Use multi-model pipeline or single model
        use_multimodel = os.environ.get("USE_MULTIMODEL", "true").lower() == "true"
        use_kfold = True
        n_splits = 5
        
        # Create checkpoint manager for pipeline-level checkpointing
        from lib.mlops_core import CheckpointManager
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        ckpt_manager = CheckpointManager(checkpoint_dir, run_id)
        
        if use_multimodel:
            # Multi-model pipeline: train all models sequentially
            logger.info("=" * 80)
            logger.info("MULTI-MODEL TRAINING MODE")
            logger.info("=" * 80)
            
            # Define models to train (all models from proposal)
            all_available = list_available_models()
            models_to_train = [
                "logistic_regression",
                "svm",
                "naive_cnn",
                "vit_gru",
                "vit_transformer",
                "slowfast",
                "x3d",
            ]
            
            # Filter to only available models
            models_to_train = [m for m in models_to_train if m in all_available]
            
            # Option 1: SKIP_MODELS - exclude specific models (comma-separated)
            # Example: SKIP_MODELS="naive_cnn,slowfast"
            if "SKIP_MODELS" in os.environ:
                skip_list = [m.strip() for m in os.environ["SKIP_MODELS"].split(",")]
                models_to_train = [m for m in models_to_train if m not in skip_list]
                logger.info("Skipping models: %s", skip_list)
                logger.info("Remaining models to train: %s", models_to_train)
            
            # Option 2: MODELS_TO_TRAIN - explicitly specify which models (takes precedence over SKIP_MODELS)
            # Example: MODELS_TO_TRAIN="logistic_regression,svm,vit_gru"
            if "MODELS_TO_TRAIN" in os.environ:
                env_models = os.environ["MODELS_TO_TRAIN"].split(",")
                models_to_train = [m.strip() for m in env_models if m.strip() in all_available]
                logger.info("Using models from MODELS_TO_TRAIN: %s", models_to_train)
            
            logger.info("Models to train: %s", models_to_train)
            logger.info("Using %d-fold stratified cross-validation", n_splits)
            logger.info("Models will be trained sequentially with shared data pipeline")
            logger.info("Each model has its own checkpoint directory and can be resumed independently")
            
            # Build multi-model pipeline
            pipeline = build_multimodel_pipeline(
                config, models_to_train, tracker, n_splits=n_splits, ckpt_manager=ckpt_manager
            )
            
        elif use_kfold:
            # Single model with k-fold
            logger.info("Using %d-fold stratified cross-validation (single model: %s)", 
                       n_splits, config.model_type)
            pipeline = build_kfold_pipeline(config, tracker, n_splits=n_splits, ckpt_manager=ckpt_manager)
        else:
            # Single model, single split
            logger.info("Using single train/val split (model: %s)", config.model_type)
            pipeline = build_mlops_pipeline(config, tracker)
            pipeline.ckpt_manager = ckpt_manager
        
        artifacts = pipeline.run_pipeline(ckpt_manager=ckpt_manager)
        
        logger.info("=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("Run ID: %s", run_id)
        logger.info("Results saved to: %s", run_dir)
        logger.info("=" * 80)
        
        # Print summary
        metrics_df = tracker.get_metrics()
        if metrics_df.height > 0:
            best_val = tracker.get_best_metric("accuracy", phase="val", maximize=True)
            if best_val:
                logger.info("Best validation accuracy: %.4f (epoch %d)", 
                           best_val['value'], best_val['epoch'])
        
        return 0
    
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e), exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

