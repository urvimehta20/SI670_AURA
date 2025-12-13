"""
Model training pipeline.

Trains models using scaled videos and extracted features.
Supports multiple model types and k-fold cross-validation.
"""

from __future__ import annotations

import logging
import os
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from lib.data import stratified_kfold
# Lazy import to avoid circular dependency issues
# VideoConfig and VideoDataset will be imported when needed
from lib.mlops.config import ExperimentTracker, CheckpointManager
from lib.mlops.mlflow_tracker import create_mlflow_tracker, MLFLOW_AVAILABLE
from lib.training.trainer import OptimConfig, TrainConfig, fit
from lib.training.model_factory import create_model, is_pytorch_model, is_xgboost_model, get_model_config
from lib.training.metrics_utils import compute_classification_metrics
from lib.training.cleanup_utils import cleanup_model_and_memory
from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


def _flush_logs():
    """Flush all logging handlers and stdout/stderr to ensure immediate output."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        # Flush all logging handlers
        for handler in logging.root.handlers:
            if hasattr(handler, 'stream') and hasattr(handler.stream, 'flush'):
                handler.stream.flush()
            elif hasattr(handler, 'flush'):
                handler.flush()
    except Exception:
        # Ignore errors during flush (non-critical)
        pass


# Constants for model type classification
BASELINE_MODELS = {
    "logistic_regression",
    "logistic_regression_stage2",
    "logistic_regression_stage2_stage4",
    "svm",
    "svm_stage2",
    "svm_stage2_stage4"
}

STAGE2_MODELS = {
    "logistic_regression",
    "logistic_regression_stage2",
    "logistic_regression_stage2_stage4",
    "svm",
    "svm_stage2",
    "svm_stage2_stage4"
}

STAGE4_MODELS = {
    "logistic_regression_stage2_stage4",
    "svm_stage2_stage4"
}

# Model file extensions to copy
MODEL_FILE_EXTENSIONS = ["*.pt", "*.joblib", "*.json"]


def _copy_model_files(source_dir: Path, dest_dir: Path, model_name: str = "") -> None:
    """
    Copy model files from source directory to destination directory.
    
    Args:
        source_dir: Source directory containing model files
        dest_dir: Destination directory to copy files to
        model_name: Optional model name for logging
    
    Raises:
        OSError: If copying fails
    """
    import shutil
    
    if not source_dir.exists():
        logger.warning(f"Source directory does not exist: {source_dir}")
        return
    
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create destination directory {dest_dir}: {e}")
        raise OSError(f"Cannot create destination directory: {dest_dir}") from e
    
    copied_count = 0
    for ext in MODEL_FILE_EXTENSIONS:
        for model_file in source_dir.glob(ext):
            try:
                shutil.copy2(model_file, dest_dir / model_file.name)
                copied_count += 1
            except (OSError, IOError, PermissionError) as e:
                logger.warning(f"Failed to copy {model_file} to {dest_dir}: {e}")
    
    if copied_count > 0:
        log_msg = f"Copied {copied_count} model file(s) from {source_dir.name} to {dest_dir.name}"
        if model_name:
            log_msg = f"Saved best model from {model_name}: {log_msg}"
        logger.info(log_msg)
    else:
        logger.warning(f"No model files found in {source_dir} to copy")


def _ensure_lib_models_exists(project_root_path: Path) -> None:
    """
    Ensure lib/models directory exists with minimal stub files.
    
    Creates the directory and essential files if they don't exist,
    allowing imports to succeed even if the full lib/models wasn't synced.
    """
    models_dir = project_root_path / 'lib' / 'models'
    models_init = models_dir / '__init__.py'
    video_py = models_dir / 'video.py'
    
    # If directory already exists with both files, don't overwrite
    if models_dir.exists() and models_init.exists() and video_py.exists():
        return
    
    # Create directory
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal __init__.py if missing
    if not models_init.exists():
        try:
            models_init.write_text('''"""
Video models and datasets module (minimal stub).

This is a minimal stub created automatically.
For full functionality, ensure lib/models is properly synced to the server.
"""

from .video import VideoConfig, VideoDataset

__all__ = ["VideoConfig", "VideoDataset"]
''')
            logger.info(f"Created minimal lib/models/__init__.py at {models_init}")
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to create minimal __init__.py at {models_init}: {e}")
            raise
    
    # Create minimal video.py if missing
    if not video_py.exists():
        try:
            video_py.write_text('''"""
Video configuration and dataset (minimal stub).

This is a minimal stub created automatically.
For full functionality, ensure lib/models/video.py is properly synced to the server.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union
import torch
from torch.utils.data import Dataset
import polars as pl


@dataclass
class VideoConfig:
    """Configuration for video sampling and preprocessing (minimal stub)."""
    num_frames: int = 16
    fixed_size: Optional[int] = None
    max_size: Optional[int] = None
    img_size: Optional[int] = None
    rolling_window: bool = False
    window_size: Optional[int] = None
    window_stride: Optional[int] = None
    augmentation_config: Optional[dict] = None
    temporal_augmentation_config: Optional[dict] = None
    use_scaled_videos: bool = False


class VideoDataset(Dataset):
    """Dataset over videos (minimal stub - will fail at runtime if used without full implementation)."""
    
    def __init__(
        self,
        df: Union[pl.DataFrame, Any],
        project_root: str,
        config: VideoConfig,
        train: bool = True,
        max_videos: Optional[int] = None,
    ) -> None:
        raise RuntimeError(
            "VideoDataset stub cannot be used. "
            "Please ensure lib/models/video.py is properly synced to the server. "
            f"Expected at: {project_root}/lib/models/video.py"
        )
    
    def __len__(self) -> int:
        raise RuntimeError("VideoDataset stub cannot be used")
    
    def __getitem__(self, idx: int):
        raise RuntimeError("VideoDataset stub cannot be used")
''')
            logger.info(f"Created minimal lib/models/video.py at {video_py}")
            logger.warning(
                "Created minimal lib/models stub. "
                "For PyTorch models to work, ensure the full lib/models directory is synced to the server."
            )
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to create minimal video.py at {video_py}: {e}")
            raise


def _validate_stage5_prerequisites(
    project_root: Path,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: str,
    model_types: List[str]
) -> Dict[str, Any]:
    """
    Validate that all required data exists before starting Stage 5 training.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to Stage 3 scaled metadata
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        model_types: List of model types to train
    
    Returns:
        Dictionary with validation results:
        - stage3_available: bool
        - stage2_available: bool
        - stage4_available: bool
        - stage3_count: int (number of videos)
        - stage2_count: int (number of feature rows)
        - stage4_count: int (number of feature rows)
        - runnable_models: List[str] (models that can be run)
        - missing_models: Dict[str, List[str]] (models that cannot run and why)
    """
    from lib.utils.paths import load_metadata_flexible
    
    results = {
        "stage3_available": False,
        "stage2_available": False,
        "stage4_available": False,
        "stage3_count": 0,
        "stage2_count": 0,
        "stage4_count": 0,
        "runnable_models": [],
        "missing_models": {}  # model_type -> [list of missing requirements]
    }
    
    logger.info("=" * 80)
    logger.info("STAGE 5 PREREQUISITE VALIDATION")
    logger.info("=" * 80)
    
    # Check Stage 3 (REQUIRED for all models)
    logger.info("\n[1/3] Checking Stage 3 (scaled videos) - REQUIRED for all models...")
    scaled_df = load_metadata_flexible(scaled_metadata_path)
    if scaled_df is None or scaled_df.height == 0:
        logger.error(f"✗ Stage 3 metadata not found or empty: {scaled_metadata_path}")
        logger.error("  Stage 3 is REQUIRED for all models. Please run Stage 3 first.")
        results["stage3_available"] = False
    else:
        results["stage3_available"] = True
        results["stage3_count"] = scaled_df.height
        logger.info(f"✓ Stage 3 metadata found: {scaled_df.height} scaled videos")
        logger.info(f"  Path: {scaled_metadata_path}")
    
    # Check Stage 2 (REQUIRED for stage2 models)
    logger.info("\n[2/3] Checking Stage 2 (features) - REQUIRED for *_stage2 models...")
    features2_df = load_metadata_flexible(features_stage2_path)
    if features2_df is None or features2_df.height == 0:
        logger.warning(f"✗ Stage 2 metadata not found or empty: {features_stage2_path}")
        results["stage2_available"] = False
    else:
        results["stage2_available"] = True
        results["stage2_count"] = features2_df.height
        logger.info(f"✓ Stage 2 metadata found: {features2_df.height} feature rows")
        logger.info(f"  Path: {features_stage2_path}")
    
    # Check Stage 4 (REQUIRED for stage2_stage4 models)
    logger.info("\n[3/3] Checking Stage 4 (scaled features) - REQUIRED for *_stage2_stage4 models...")
    features4_df = load_metadata_flexible(features_stage4_path)
    if features4_df is None or features4_df.height == 0:
        logger.warning(f"✗ Stage 4 metadata not found or empty: {features_stage4_path}")
        results["stage4_available"] = False
    else:
        results["stage4_available"] = True
        results["stage4_count"] = features4_df.height
        logger.info(f"✓ Stage 4 metadata found: {features4_df.height} feature rows")
        logger.info(f"  Path: {features_stage4_path}")
    
    # Determine which models can be run
    logger.info("\n" + "=" * 80)
    logger.info("MODEL REQUIREMENTS CHECK")
    logger.info("=" * 80)
    
    # All baseline models (svm, logistic_regression and their variants) require Stage 2 features
    # Stage 5 only trains - features must be pre-extracted in Stage 2/4
    for model_type in model_types:
        missing = []
        
        # All models require Stage 3
        if not results["stage3_available"]:
            missing.append("Stage 3 (scaled videos)")
        
        # Stage 2 models require Stage 2
        if model_type in STAGE2_MODELS and not results["stage2_available"]:
            missing.append("Stage 2 (features)")
        
        # Stage 2+4 models require Stage 4
        if model_type in STAGE4_MODELS and not results["stage4_available"]:
            missing.append("Stage 4 (scaled features)")
        
        if missing:
            results["missing_models"][model_type] = missing
            logger.error(f"✗ {model_type}: CANNOT RUN - Missing: {', '.join(missing)}")
        else:
            results["runnable_models"].append(model_type)
            logger.info(f"✓ {model_type}: CAN RUN")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Stage 3 (scaled videos): {'✓ Available' if results['stage3_available'] else '✗ MISSING'}")
    logger.info(f"  Count: {results['stage3_count']} videos")
    logger.info(f"Stage 2 (features): {'✓ Available' if results['stage2_available'] else '✗ MISSING'}")
    logger.info(f"  Count: {results['stage2_count']} feature rows")
    logger.info(f"Stage 4 (scaled features): {'✓ Available' if results['stage4_available'] else '✗ MISSING'}")
    logger.info(f"  Count: {results['stage4_count']} feature rows")
    logger.info(f"\nRunnable models: {len(results['runnable_models'])}/{len(model_types)}")
    logger.info(f"  {results['runnable_models']}")
    
    if results["missing_models"]:
        logger.error(f"\nCannot run models: {len(results['missing_models'])}/{len(model_types)}")
        for model_type, missing in results["missing_models"].items():
            logger.error(f"  {model_type}: Missing {', '.join(missing)}")
    
    logger.info("=" * 80)
    
    # Write Stage 2/4 failure report to file if needed
    failure_report_path = project_root / "logs" / "stage5_prerequisite_failures.txt"
    failure_report_path.parent.mkdir(parents=True, exist_ok=True)
    
    failures_detected = False
    failure_lines = []
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    failure_lines.append("=" * 80)
    failure_lines.append("STAGE 5 PREREQUISITE FAILURE REPORT")
    failure_lines.append(f"Generated: {timestamp}")
    failure_lines.append("=" * 80)
    failure_lines.append("")
    
    # Check if Stage 2 was expected but missing
    # All baseline models require Stage 2 features
    stage2_required_models = [m for m in model_types if m in STAGE2_MODELS]
    if stage2_required_models and not results["stage2_available"]:
        failures_detected = True
        failure_lines.append("STAGE 2 FAILURE DETECTED")
        failure_lines.append("-" * 80)
        failure_lines.append(f"Expected: Stage 2 features metadata at: {features_stage2_path}")
        failure_lines.append(f"Status: NOT FOUND or EMPTY")
        failure_lines.append(f"Required for ALL baseline models: {', '.join(stage2_required_models)}")
        failure_lines.append("Note: All baseline models (svm, logistic_regression and variants) require Stage 2 features.")
        failure_lines.append("")
        failure_lines.append("ACTION REQUIRED:")
        failure_lines.append("  - Run Stage 2 feature extraction:")
        failure_lines.append("    sbatch src/scripts/slurm_stage2_features.sh")
        failure_lines.append("  - Or check if Stage 2 output exists at a different location")
        failure_lines.append("")
    
    # Check if Stage 4 was expected but missing
    stage4_required_models = [m for m in model_types if m in STAGE4_MODELS]
    if stage4_required_models and not results["stage4_available"]:
        failures_detected = True
        failure_lines.append("STAGE 4 FAILURE DETECTED")
        failure_lines.append("-" * 80)
        failure_lines.append(f"Expected: Stage 4 features metadata at: {features_stage4_path}")
        failure_lines.append(f"Status: NOT FOUND or EMPTY")
        failure_lines.append(f"Required for models: {', '.join(stage4_required_models)}")
        failure_lines.append("")
        failure_lines.append("ACTION REQUIRED:")
        failure_lines.append("  - Run Stage 4 scaled feature extraction:")
        failure_lines.append("    sbatch src/scripts/slurm_stage4_scaled_features.sh")
        failure_lines.append("  - Or check if Stage 4 output exists at a different location")
        failure_lines.append("")
    
    # Check if Stage 3 was expected but missing (critical for all models)
    if not results["stage3_available"]:
        failures_detected = True
        failure_lines.append("STAGE 3 FAILURE DETECTED (CRITICAL)")
        failure_lines.append("-" * 80)
        failure_lines.append(f"Expected: Stage 3 scaled videos metadata at: {scaled_metadata_path}")
        failure_lines.append(f"Status: NOT FOUND or EMPTY")
        failure_lines.append(f"Required for ALL models: {', '.join(model_types)}")
        failure_lines.append("")
        failure_lines.append("ACTION REQUIRED:")
        failure_lines.append("  - Run Stage 3 video scaling:")
        failure_lines.append("    sbatch src/scripts/slurm_stage3_scaling.sh")
        failure_lines.append("  - Or check if Stage 3 output exists at a different location")
        failure_lines.append("")
    
    # Summary
    if failures_detected:
        failure_lines.append("=" * 80)
        failure_lines.append("SUMMARY")
        failure_lines.append("=" * 80)
        failure_lines.append(f"Total models requested: {len(model_types)}")
        failure_lines.append(f"Runnable models: {len(results['runnable_models'])}")
        failure_lines.append(f"Cannot run models: {len(results['missing_models'])}")
        failure_lines.append("")
        if results["runnable_models"]:
            failure_lines.append("Models that CAN run:")
            for model in results["runnable_models"]:
                failure_lines.append(f"  ✓ {model}")
            failure_lines.append("")
        if results["missing_models"]:
            failure_lines.append("Models that CANNOT run:")
            for model, missing in results["missing_models"].items():
                failure_lines.append(f"  ✗ {model}: Missing {', '.join(missing)}")
        failure_lines.append("")
        failure_lines.append("=" * 80)
        
        # Write to file
        try:
            with open(failure_report_path, 'w') as f:
                f.write('\n'.join(failure_lines))
            logger.info(f"\n⚠ Failure report written to: {failure_report_path}")
        except Exception as e:
            logger.warning(f"Failed to write failure report: {e}")
    
    return results


def stage5_train_models(
    project_root: str,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: str,
    model_types: List[str],
    n_splits: int = 5,
    num_frames: int = 1000,
    output_dir: str = "data/stage5",
    use_tracking: bool = True,
    use_mlflow: bool = True,
    train_ensemble: bool = False,
    ensemble_method: str = "meta_learner",
    delete_existing: bool = False,
    resume: bool = True
) -> Dict[str, Any]:
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
        resume: If True, skip training folds that already have saved models (resume mode, default: True)
    
    Returns:
        Dictionary of training results
    """
    # CRITICAL: Set PyTorch memory optimizations at the very start (before any model operations)
    import os
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True at pipeline start")
    
    # Immediate logging to show function has started
    logger.info("=" * 80)
    logger.info("Stage 5: Model Training Pipeline Started")
    logger.info("=" * 80)
    logger.info("Model types: %s", model_types)
    logger.info("K-fold splits: %d", n_splits)
    logger.info("Frames per video: %d", num_frames)
    logger.info("Output directory: %s", output_dir)
    logger.info("Initializing pipeline...")
    _flush_logs()  # Ensure immediate output
    
    # Input validation
    if not project_root or not isinstance(project_root, str):
        raise ValueError(f"project_root must be a non-empty string, got: {type(project_root)}")
    if not scaled_metadata_path or not isinstance(scaled_metadata_path, str):
        raise ValueError(f"scaled_metadata_path must be a non-empty string, got: {type(scaled_metadata_path)}")
    if not features_stage2_path or not isinstance(features_stage2_path, str):
        raise ValueError(f"features_stage2_path must be a non-empty string, got: {type(features_stage2_path)}")
    if not features_stage4_path or not isinstance(features_stage4_path, str):
        raise ValueError(f"features_stage4_path must be a non-empty string, got: {type(features_stage4_path)}")
    if not model_types or not isinstance(model_types, list) or len(model_types) == 0:
        raise ValueError(f"model_types must be a non-empty list, got: {type(model_types)}")
    if n_splits <= 0 or not isinstance(n_splits, int):
        raise ValueError(f"n_splits must be a positive integer, got: {n_splits}")
    if num_frames <= 0 or not isinstance(num_frames, int):
        raise ValueError(f"num_frames must be a positive integer, got: {num_frames}")
    if not isinstance(output_dir, str):
        raise ValueError(f"output_dir must be a string, got: {type(output_dir)}")
    
    # Convert project_root to Path and resolve it once (avoid variable shadowing)
    try:
        project_root_path = Path(project_root).resolve()
        if not project_root_path.exists():
            raise FileNotFoundError(f"Project root directory does not exist: {project_root_path}")
        if not project_root_path.is_dir():
            raise NotADirectoryError(f"Project root is not a directory: {project_root_path}")
    except (OSError, ValueError) as e:
        logger.error(f"Invalid project_root path: {project_root} - {e}")
        raise ValueError(f"Invalid project_root path: {project_root}") from e
    
    project_root_str = str(project_root_path)
    # Keep original string for backward compatibility in function calls
    project_root_str_orig = project_root_str
    
    # Ensure lib/models directory exists (create minimal stub if missing)
    try:
        _ensure_lib_models_exists(project_root_path)
    except Exception as e:
        logger.error(f"Failed to ensure lib/models exists: {e}")
        raise
    
    try:
        output_dir_path = project_root_path / output_dir
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise ValueError(f"Cannot create output directory: {output_dir}") from e
    
    output_dir = output_dir_path
    
    # CRITICAL: Validate all prerequisites before starting any training
    validation_results = _validate_stage5_prerequisites(
        project_root_path,
        scaled_metadata_path,
        features_stage2_path,
        features_stage4_path,
        model_types
    )
    
    # Check if any models can be run
    if not validation_results["runnable_models"]:
        error_msg = (
            "✗ ERROR: None of the requested models can be run with the available data.\n"
            "Please check the validation summary above and re-run the required stages.\n\n"
            "Required stages:\n"
        )
        if not validation_results["stage3_available"]:
            error_msg += "  - Stage 3 (scaled videos): REQUIRED for all models\n"
        if any(m in STAGE2_MODELS for m in model_types) and not validation_results["stage2_available"]:
            error_msg += "  - Stage 2 (features): REQUIRED for all baseline models (svm, logistic_regression and variants)\n"
        if any("stage2_stage4" in m for m in model_types) and not validation_results["stage4_available"]:
            error_msg += "  - Stage 4 (scaled features): REQUIRED for *_stage2_stage4 models\n"
        raise FileNotFoundError(error_msg)
    
    # Filter to only runnable models
    if len(validation_results["runnable_models"]) < len(model_types):
        skipped = set(model_types) - set(validation_results["runnable_models"])
        logger.warning(
            f"\n⚠ WARNING: Skipping {len(skipped)} model(s) due to missing prerequisites: {skipped}\n"
            f"Will train {len(validation_results['runnable_models'])} model(s): {validation_results['runnable_models']}"
        )
        model_types = validation_results["runnable_models"]
    
    # Load metadata (support both CSV and Arrow/Parquet)
    logger.info("\nStage 5: Loading metadata...")
    
    from lib.utils.paths import load_metadata_flexible
    from lib.utils.data_integrity import DataIntegrityChecker
    from lib.utils.guardrails import ResourceMonitor, HealthCheckStatus, ResourceExhaustedError, DataIntegrityError
    
    # CRITICAL: Validate metadata file integrity before loading
    metadata_path_obj = Path(scaled_metadata_path)
    is_valid, error_msg, scaled_df = DataIntegrityChecker.validate_metadata_file(
        metadata_path_obj,
        required_columns={'video_path', 'label'},
        min_rows=3000,
        allow_empty=False
    )
    
    if not is_valid:
        raise DataIntegrityError(f"Metadata validation failed: {error_msg}")
    
    if scaled_df is None:
        # Fallback: try loading manually
        scaled_df = load_metadata_flexible(scaled_metadata_path)
        if scaled_df is None:
            raise FileNotFoundError(f"Scaled metadata not found: {scaled_metadata_path}")
    
    # CRITICAL: Verify dataframe has more than 3000 rows (double-check)
    num_rows = scaled_df.height
    logger.info(f"Loaded metadata: {num_rows} rows")
    if num_rows <= 3000:
        error_msg = (
            f"✗ ERROR: Insufficient data for training. "
            f"Expected more than 3000 rows, but got {num_rows} rows.\n"
            f"Please ensure Stage 3 completed successfully and generated enough scaled videos.\n"
            f"Metadata path: {scaled_metadata_path}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    logger.info(f"✓ Data validation passed: {num_rows} rows (> 3000 required)")
    _flush_logs()
    
    # CRITICAL: Resource health check before proceeding
    monitor = ResourceMonitor()
    health = monitor.full_health_check(project_root_path)
    if health.status.value >= HealthCheckStatus.UNHEALTHY.value:
        logger.warning(f"System health check: {health.status.value} - {health.message}")
        if health.status == HealthCheckStatus.CRITICAL:
            raise ResourceExhaustedError(f"Critical system state: {health.message}")
    
    # Lazy loading: Only load features/videos if needed by any model
    from lib.training.video_training_pipeline import is_feature_based, is_video_based
    
    # Determine what to load based on model types
    needs_features = any(is_feature_based(m) or "xgboost" in m for m in model_types)
    needs_videos = any(is_video_based(m) for m in model_types)
    
    # Lazy load features only if any model requires them
    features2_df = None
    features4_df = None
    if needs_features:
        logger.info("Loading Stage 2 and Stage 4 features (required for feature-based models)...")
        features2_df = load_metadata_flexible(features_stage2_path)
        features4_df = load_metadata_flexible(features_stage4_path)
        if features2_df is None:
            logger.debug("Stage 2 features metadata not found (optional for some models)")
        if features4_df is None:
            logger.debug("Stage 4 features metadata not found (optional for some models)")
    else:
        logger.debug("Skipping feature loading - no feature-based models in training list")
    
    logger.info(f"Stage 5: Found {scaled_df.height} scaled videos")
    _flush_logs()
    
    # CRITICAL: Filter out corrupted videos and videos with no frames before training
    # This prevents runtime errors like "moov atom not found" and "Video has no frames"
    logger.info("=" * 80)
    logger.info("STAGE 5: VALIDATING VIDEOS (checking for corruption and empty videos)")
    logger.info("=" * 80)
    from lib.data import filter_existing_videos
    try:
        # Filter corrupted videos (moov atom errors, etc.) and videos with no frames
        # check_corruption=True: Check for corrupted videos (default, prevents moov atom errors)
        # check_frames=True: Also check that videos have frames (prevents "Video has no frames" errors)
        scaled_df = filter_existing_videos(
            scaled_df, 
            project_root=project_root_str,
            check_frames=True,  # Check for videos with no frames
            check_corruption=True  # Check for corrupted videos (moov atom errors, etc.)
        )
        logger.info(f"✓ Video validation complete: {scaled_df.height} valid videos ready for training")
        logger.info("=" * 80)
        _flush_logs()
    except ValueError as e:
        error_msg = (
            f"✗ ERROR: Video validation failed. "
            f"Please check that scaled videos exist and are not corrupted.\n"
            f"Error: {str(e)}\n\n"
            f"Common issues:\n"
            f"  - Corrupted videos (moov atom not found): Re-run Stage 3 scaling\n"
            f"  - Videos with no frames: Check Stage 3 output\n"
            f"  - Missing video files: Verify data/scaled_videos/ directory"
        )
        logger.error(error_msg)
        raise ValueError(error_msg) from e
    
    # Import VideoConfig - fail fast if not available (required for PyTorch models)
    # lib.models should always be available in Stage 5
    try:
        from lib.models import VideoConfig
    except ImportError as e:
        raise ImportError(
            f"CRITICAL: Cannot import VideoConfig from lib.models. "
            f"lib/models must be available for Stage 5. Error: {e}"
        ) from e
    
    # Create video config (will be used only for PyTorch models)
    # Always use scaled videos - augmentation done in Stage 1, scaling in Stage 3
    # Handle both old and new VideoConfig versions (some servers may not have use_scaled_videos yet)
    # For memory-intensive models (5c+), use chunked frame loading with adaptive sizing to prevent OOM
    # Adaptive chunk size starts at lower values for very memory-intensive models, reduces on OOM (multiplicative decrease), increases on success (additive increase)
    # Memory-intensive models that need chunked frame loading (5c+ scripts)
    # Models 5c-5l need very small initial chunk size (10) due to persistent OOM - consistent with forward pass chunk size
    MEMORY_INTENSIVE_MODELS_SMALL_CHUNK = [
        "naive_cnn",           # 5c - processes 1000 frames at full resolution, very memory intensive
        "pretrained_inception", # 5d - large pretrained model
        "variable_ar_cnn",     # 5e - variable-length videos with many frames
        "vit_gru",             # 5k - Vision Transformer with GRU
        "vit_transformer",     # 5l - Vision Transformer
    ]
    
    # XGBoost models that use pretrained models for feature extraction
    # These need reduced num_frames to prevent OOM during feature extraction
    XGBOOST_PRETRAINED_MODELS = [
        "xgboost_pretrained_inception",  # 5f
        "xgboost_i3d",                    # 5g
        "xgboost_r2plus1d",               # 5h
        "xgboost_vit_gru",                # 5i
        "xgboost_vit_transformer",        # 5j
    ]
    
    MEMORY_INTENSIVE_MODELS = [
        "naive_cnn",           # 5c - processes 1000 frames at full resolution
        "pretrained_inception", # 5d - large pretrained model
        "variable_ar_cnn",     # 5e - variable-length videos with many frames
        "i3d",                 # 5o - 3D CNN
        "r2plus1d",            # 5p - 3D CNN
        "x3d",                 # 5q - very memory intensive
        "slowfast",            # 5r - dual-pathway architecture
        "vit_gru",             # 5k - Vision Transformer with GRU
        "vit_transformer",     # 5l - Vision Transformer
    ]
    use_chunked_loading = False
    chunk_size = None
    for model_type in model_types:
        if model_type in MEMORY_INTENSIVE_MODELS:
            use_chunked_loading = True
            # Use very small chunk size (10) for models 5c-5l that have persistent OOM issues
            if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK:
                chunk_size = 10  # Initial chunk size for very memory-intensive models (5c-5l) - consistent with forward pass, capped at 28
            else:
                chunk_size = 30  # Initial chunk size for other memory-intensive models (5o-5r)
            logger.info(
                f"Enabling adaptive chunked frame loading for {model_type}: "
                f"initial_chunk_size={chunk_size}, num_frames={num_frames}. "
                f"Chunk size will adapt automatically based on OOM events (AIMD algorithm)."
            )
            break
    
    # Frame caching is enabled by default (can be disabled via FVC_USE_FRAME_CACHE=0)
    # CRITICAL: Frame caching is DISK-based (not RAM) - stores frames on disk to avoid repeated video decoding
    # This can speed up training 3-5x by avoiding CPU-intensive video decoding every epoch
    # Default: enabled (FVC_USE_FRAME_CACHE=1), can be disabled by setting FVC_USE_FRAME_CACHE=0
    use_frame_cache = os.environ.get("FVC_USE_FRAME_CACHE", "1") == "1"
    frame_cache_dir = os.environ.get("FVC_FRAME_CACHE_DIR", "data/.frame_cache")
    
    if use_frame_cache:
        logger.info(
            f"Frame caching enabled (default): cache_dir={frame_cache_dir}. "
            f"This will cache processed frames to disk to speed up training. "
            f"First epoch will be slower (building cache), subsequent epochs will be faster. "
            f"To disable, set FVC_USE_FRAME_CACHE=0"
        )
    else:
        logger.info(
            "Frame caching disabled (FVC_USE_FRAME_CACHE=0). "
            "Training will decode videos every epoch (slower but uses less disk space)."
        )
    
    try:
        # Try with use_scaled_videos, chunk_size, and frame_cache options (newer version)
        if use_chunked_loading and chunk_size is not None:
            video_config = VideoConfig(
                num_frames=num_frames,
                use_scaled_videos=True,  # Stage 5 only trains - all preprocessing done in earlier stages
                chunk_size=chunk_size,  # Chunked loading for OOM prevention
                use_frame_cache=use_frame_cache,  # Enable disk-based frame caching
                frame_cache_dir=frame_cache_dir if use_frame_cache else None  # Cache directory
            )
        else:
            video_config = VideoConfig(
                num_frames=num_frames,
                use_scaled_videos=True,  # Stage 5 only trains - all preprocessing done in earlier stages
                use_frame_cache=use_frame_cache,  # Enable disk-based frame caching
                frame_cache_dir=frame_cache_dir if use_frame_cache else None  # Cache directory
            )
    except TypeError:
        # Fallback: server version doesn't support these parameters
        logger.warning(
            "VideoConfig on server doesn't support 'use_scaled_videos', 'chunk_size', or frame_cache parameters. "
            "Using default VideoConfig and setting use_scaled_videos=True and frame_cache manually (videos should already be scaled from Stage 3)."
        )
        video_config = VideoConfig(num_frames=num_frames)
        # CRITICAL: Set use_scaled_videos=True even if constructor doesn't support it
        # Stage 5 ALWAYS uses scaled videos from Stage 3
        video_config.use_scaled_videos = True
        logger.info("Manually set use_scaled_videos=True on VideoConfig (server version fallback)")
        # CRITICAL: Set frame_cache parameters even if constructor doesn't support them
        # Frame caching is enabled by default to speed up training
        if use_frame_cache:
            video_config.use_frame_cache = True
            video_config.frame_cache_dir = frame_cache_dir
            logger.info(f"Manually set use_frame_cache=True, frame_cache_dir={frame_cache_dir} on VideoConfig (server version fallback)")
    
    # CRITICAL: Verify use_scaled_videos is True (Stage 5 ALWAYS uses scaled videos from Stage 3)
    # This ensures it's set correctly even if the constructor supports it but something went wrong
    if not getattr(video_config, 'use_scaled_videos', False):
        logger.warning(
            "CRITICAL: use_scaled_videos is False in VideoConfig for Stage 5! "
            "This should NEVER happen - Stage 5 always uses scaled videos from Stage 3. "
            "Forcing use_scaled_videos=True."
        )
        video_config.use_scaled_videos = True
        logger.info("Forced use_scaled_videos=True on VideoConfig (Stage 5 requirement)")
    
    # CRITICAL: Override num_frames to 500 for small-chunk models (5c-5l) to prevent OOM
    # These models process many frames at full resolution and need to limit to 500 frames max
    # Also override for XGBoost models that use pretrained feature extractors (5f, 5g, 5h, 5i, 5j)
    # Feature extraction with 1000 frames is too memory-intensive
    for model_type in model_types:
        if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK or model_type in XGBOOST_PRETRAINED_MODELS:
            # ARCHITECTURAL IMPROVEMENT: Use more frames for XGBoost models with enhanced feature extraction
            # Enhanced feature extraction (multi-layer + temporal pooling) is more memory-efficient
            # Can use 400 frames instead of 250 for better temporal coverage while staying within memory limits
            target_frames = 500 if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK else 400  # Increased from 250 to 400 for better features
            video_config.num_frames = target_frames
            logger.info(
                f"Overriding num_frames to {target_frames} for {model_type} "
                f"({'small-chunk model' if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK else 'XGBoost pretrained model with enhanced features'}) "
                f"to balance performance and memory. Original num_frames was {num_frames}."
            )
            break  # Only need to set once since video_config is shared
    
    results = {}
    
    # Helper function to check if a fold is already trained
    def _is_fold_complete(fold_dir: Path, model_type: str) -> bool:
        """Check if a fold directory contains a complete trained model."""
        if not fold_dir.exists():
            return False
        
        # PyTorch models: check for model.pt
        if is_pytorch_model(model_type):
            model_file = fold_dir / "model.pt"
            if model_file.exists() and model_file.stat().st_size > 0:
                return True
        
        # XGBoost/Baseline models: check for model files (varies by model type)
        # Common patterns: model.pkl, model.json, model.bst, etc.
        model_files = list(fold_dir.glob("model.*"))
        if model_files:
            # Check if any model file is non-empty
            for model_file in model_files:
                if model_file.stat().st_size > 0:
                    return True
        
        return False
    
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
                except (OSError, PermissionError, FileNotFoundError) as e:
                    logger.warning(f"Could not delete {model_output_dir}: {e}")
        logger.info(f"Stage 5: Deleted {deleted_count} existing model directories")
        _flush_logs()
    
    # Train each model type
    for model_type in model_types:
        logger.info(f"\n{'='*80}")
        logger.info(f"Stage 5: Training model: {model_type}")
        logger.info(f"{'='*80}")
        _flush_logs()
        
        model_output_dir = output_dir / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get model config
        model_config = get_model_config(model_type)
        
        # CRITICAL: Override num_frames in model_config for small-chunk models (5c-5l) and XGBoost models (5f-5j)
        # This ensures the model is created with reduced num_frames instead of 1000
        if model_type in MEMORY_INTENSIVE_MODELS_SMALL_CHUNK:
            model_config["num_frames"] = 500
            logger.info(
                f"Overriding model_config num_frames to 500 for {model_type} "
                f"(small-chunk model) to match video_config."
            )
        elif model_type in XGBOOST_PRETRAINED_MODELS:
            # ARCHITECTURAL IMPROVEMENT: Use 400 frames for enhanced feature extraction
            # Enhanced multi-layer + temporal pooling allows more frames while staying within memory
            model_config["num_frames"] = 400  # Increased from 250 to 400 for better temporal coverage
            logger.info(
                f"Overriding model_config num_frames to 400 for {model_type} "
                f"(XGBoost pretrained model with enhanced feature extraction) to improve feature quality."
            )
        
        # K-fold cross-validation
        fold_results = []
        
        # CRITICAL: Enforce 5-fold stratified cross-validation
        if n_splits != 5:
            logger.warning(f"n_splits={n_splits} specified, but enforcing 5-fold CV as required")
            n_splits = 5
        
        # Get hyperparameter grid for grid search
        from .grid_search import get_hyperparameter_grid, generate_parameter_combinations, select_best_hyperparameters
        from .visualization import generate_all_plots
        
        param_grid = get_hyperparameter_grid(model_type)
        param_combinations = generate_parameter_combinations(param_grid)
        
        logger.info(f"Grid search: {len(param_combinations)} hyperparameter combinations to try")
        _flush_logs()
        
        # OPTIMIZATION: Use smaller stratified sample for hyperparameter search (faster)
        # Final training will use full dataset for robustness
        # Can be controlled via FVC_GRID_SEARCH_SAMPLE_SIZE environment variable (default: 0.1 = 10%)
        from sklearn.model_selection import StratifiedShuffleSplit
        import polars as pl
        
        # Get grid search sample size from environment (default: 10% for fastest results)
        # Can be set to 0.2 for 20% (more robust but slower) or 0.05 for 5% (fastest but less robust)
        grid_search_sample_size = float(os.environ.get("FVC_GRID_SEARCH_SAMPLE_SIZE", "0.1"))
        grid_search_sample_size = max(0.05, min(0.5, grid_search_sample_size))  # Clamp between 5% and 50%
        
        logger.info("=" * 80)
        logger.info(f"HYPERPARAMETER SEARCH: Using {grid_search_sample_size*100:.1f}% stratified sample for efficiency")
        logger.info("=" * 80)
        
        # Sample data for hyperparameter search
        labels = scaled_df["label"].to_list()
        test_size = 1.0 - grid_search_sample_size
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        sample_indices, _ = next(sss.split(scaled_df, labels))
        
        # Create sample DataFrame
        sample_df = scaled_df[sample_indices]
        logger.info(f"Hyperparameter search sample: {sample_df.height} rows ({100.0 * sample_df.height / scaled_df.height:.1f}% of {scaled_df.height} total)")
        logger.info(f"To change sample size, set FVC_GRID_SEARCH_SAMPLE_SIZE environment variable (current: {grid_search_sample_size})")
        _flush_logs()
        
        # Create folds from sample for hyperparameter search
        grid_search_folds = stratified_kfold(
            sample_df,
            n_splits=n_splits,
            random_state=42
        )
        
        if len(grid_search_folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds for grid search, got {len(grid_search_folds)}")
        
        logger.info(f"Using {n_splits}-fold stratified cross-validation on {grid_search_sample_size*100:.1f}% sample for hyperparameter search")
        _flush_logs()
        
        # Store results for all hyperparameter combinations
        all_grid_results = []
        
        # Grid search: try each hyperparameter combination on sample
        if not param_combinations:
            # No grid search, use default config
            param_combinations = [{}]
            logger.info("No hyperparameter grid defined, using default configuration")
        
        for param_idx, params in enumerate(param_combinations):
            logger.info(f"\n{'='*80}")
            logger.info(f"Grid Search: Hyperparameter combination {param_idx + 1}/{len(param_combinations)}")
            logger.info(f"Parameters: {params}")
            logger.info(f"{'='*80}")
            _flush_logs()
            
            # Update model_config with current hyperparameters
            current_config = model_config.copy()
            # Filter batch_size from params for memory-intensive models (will be capped later)
            # Define memory-intensive models and their batch size limits (shared constant)
            MEMORY_INTENSIVE_MODELS_BATCH_LIMITS = {
                "x3d": 1,  # Very memory intensive
                "naive_cnn": 1,  # Processes 1000 frames at full resolution - must use batch_size=1
                "variable_ar_cnn": 2,  # Processes variable-length videos with many frames
                "pretrained_inception": 2,  # Large pretrained model processing many frames
            }
            params_to_apply = params.copy()
            if model_type in MEMORY_INTENSIVE_MODELS_BATCH_LIMITS and "batch_size" in params_to_apply:
                max_batch = MEMORY_INTENSIVE_MODELS_BATCH_LIMITS[model_type]
                if params_to_apply["batch_size"] > max_batch:
                    logger.debug(
                        f"Removing batch_size={params_to_apply['batch_size']} from grid search params for {model_type} "
                        f"(will be capped at {max_batch})"
                    )
                    del params_to_apply["batch_size"]
            current_config.update(params_to_apply)
            
            # Store fold results for this parameter combination
            param_fold_results = []
            
            # Train all folds with this hyperparameter combination (using sample)
            for fold_idx in range(n_splits):
                logger.info(f"\nHyperparameter Search - {model_type} - Fold {fold_idx + 1}/{n_splits} ({grid_search_sample_size*100:.1f}% sample)")
                _flush_logs()
                
                # Delete existing fold if delete_existing is True (do this BEFORE checking if complete)
                fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                if delete_existing and fold_output_dir.exists():
                    try:
                        import shutil
                        shutil.rmtree(fold_output_dir)
                        logger.info(f"Deleted existing hyperparameter search fold {fold_idx + 1} directory (clean mode)")
                        _flush_logs()
                    except (OSError, PermissionError, FileNotFoundError) as e:
                        logger.warning(f"Could not delete {fold_output_dir}: {e}")
                        _flush_logs()
                
                # Check if fold is already complete (resume mode) - only if not deleting
                if resume and not delete_existing and _is_fold_complete(fold_output_dir, model_type):
                    logger.info(f"Fold {fold_idx + 1} already trained (found existing model). Skipping.")
                    logger.info(f"To retrain this fold, use --delete-existing flag")
                    # Load existing results if available, otherwise create placeholder
                    # Note: We can't easily reconstruct metrics from saved models, so we'll skip this fold
                    # in grid search results but continue with other folds
                    continue
                
                # Get the specific fold from sample
                train_df, val_df = grid_search_folds[fold_idx]
            
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
                
                # Train model
                try:
                    if is_pytorch_model(model_type):
                        # Create datasets for PyTorch models
                        # Lazy import to avoid circular dependency
                        # Ensure project root is in Python path for imports
                        # Note: sys and importlib.util are already imported at module level
                        # project_root_path is already resolved at function start
                        
                        # Import VideoDataset and collate function - fail fast if not available (required for video-based models)
                        try:
                            from lib.models import VideoDataset
                            from lib.models.video import variable_ar_collate
                        except ImportError as e:
                            raise ImportError(
                            f"Cannot import VideoDataset from lib.models. "
                            f"Required for video-based models. Error: {e}"
                        ) from e
                        train_dataset = VideoDataset(
                        train_df,
                        project_root=project_root_str_orig,
                        config=video_config,
                        )
                        val_dataset = VideoDataset(
                        val_df,
                        project_root=project_root_str_orig,
                        config=video_config,
                        )
                    
                        # Create data loaders
                        # GPU-optimized DataLoader settings
                        use_cuda = torch.cuda.is_available()
                        num_workers = current_config.get("num_workers", model_config.get("num_workers", 0))
                    
                        # Get batch size and gradient accumulation for logging
                        batch_size = current_config.get("batch_size", model_config.get("batch_size", 8))
                        gradient_accumulation_steps = current_config.get("gradient_accumulation_steps", model_config.get("gradient_accumulation_steps", 1))
                        
                        # CRITICAL: Force smaller batch sizes for memory-intensive models to prevent OOM
                        # These models process many frames (1000) at high resolution, requiring conservative batch sizes
                        # Use same limits as defined above for consistency
                        MEMORY_INTENSIVE_MODELS_BATCH_LIMITS = {
                            "x3d": 1,  # Very memory intensive
                            "naive_cnn": 1,  # Processes 1000 frames at full resolution - must use batch_size=1
                            "variable_ar_cnn": 2,  # Processes variable-length videos with many frames
                            "pretrained_inception": 2,  # Large pretrained model processing many frames
                        }
                        
                        if model_type in MEMORY_INTENSIVE_MODELS_BATCH_LIMITS:
                            max_batch_size = MEMORY_INTENSIVE_MODELS_BATCH_LIMITS[model_type]
                            if batch_size > max_batch_size:
                                # Calculate effective batch size before reduction
                                effective_batch_size = batch_size * gradient_accumulation_steps
                                logger.warning(
                                    f"{model_type} model requires batch_size<={max_batch_size} to prevent OOM. "
                                    f"Overriding batch_size from {batch_size} to {max_batch_size}. "
                                    f"Adjusting gradient_accumulation_steps to maintain effective batch size of {effective_batch_size}."
                                )
                                # Increase gradient accumulation to maintain effective batch size
                                gradient_accumulation_steps = (effective_batch_size + max_batch_size - 1) // max_batch_size
                                batch_size = max_batch_size
                        
                        effective_batch_size = batch_size * gradient_accumulation_steps
                        
                        logger.info(
                            f"Training configuration - Batch size: {batch_size}, "
                            f"Gradient accumulation steps: {gradient_accumulation_steps}, "
                            f"Effective batch size: {effective_batch_size}"
                        )
                        
                        train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=use_cuda,  # Faster GPU transfer
                        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
                        collate_fn=variable_ar_collate,  # Convert (N, T, C, H, W) to (N, C, T, H, W) for 3D CNNs
                        )
                        val_loader = DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=use_cuda,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                        collate_fn=variable_ar_collate,  # Convert (N, T, C, H, W) to (N, C, T, H, W) for 3D CNNs
                        )
                        # PyTorch model training
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        
                        # CRITICAL: Apply PyTorch memory optimizations to prevent GPU memory from being hogged
                        if device.type == "cuda":
                            # Set CUDA memory allocator to use expandable segments (reduces fragmentation)
                            import os
                            if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                            
                            # Disable cuDNN benchmark to reduce memory usage (trade-off: slightly slower)
                            torch.backends.cudnn.benchmark = False
                            
                            # Enable deterministic mode to reduce memory (optional, but helps with memory)
                            torch.backends.cudnn.deterministic = False  # Keep False for performance, but benchmark=False helps memory
                            
                            # Clear cache before model creation
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                            aggressive_gc(clear_cuda=True)
                            logger.info("Applied PyTorch memory optimizations: expandable_segments, cudnn.benchmark=False")
                        
                        try:
                            model = create_model(model_type, model_config)
                        except TypeError as e:
                            error_msg = str(e)
                            if "unexpected keyword argument" in error_msg or "got an unexpected keyword argument" in error_msg:
                                logger.error(
                                    f"CRITICAL: Function signature mismatch in create_model for {model_type}: {e}. "
                                    f"Config type: {type(model_config)}, Config keys: {list(model_config.keys()) if isinstance(model_config, dict) else 'N/A'}. "
                                    f"Please check lib/training/model_factory.py::create_model() for correct signature."
                                )
                            raise
                        except ValueError as e:
                            logger.error(
                                f"Value error creating model {model_type}: {e}. "
                                f"Config: {model_config if isinstance(model_config, dict) else 'RunConfig object'}"
                            )
                            raise
                        model = model.to(device)
                    
                        # Create optimizer and scheduler with ML best practices
                        # Use hyperparameters from grid search if available
                        optim_cfg = OptimConfig(
                        lr=current_config.get("learning_rate", model_config.get("learning_rate", 1e-4)),
                        weight_decay=current_config.get("weight_decay", model_config.get("weight_decay", 1e-4)),
                        max_grad_norm=current_config.get("max_grad_norm", model_config.get("max_grad_norm", 1.0)),  # Gradient clipping
                        # Use differential LR for pretrained models
                        backbone_lr=current_config.get("backbone_lr", model_config.get("backbone_lr", None)),
                        head_lr=current_config.get("head_lr", model_config.get("head_lr", None)),
                        )
                        train_cfg = TrainConfig(
                        num_epochs=current_config.get("num_epochs", model_config.get("num_epochs", 20)),
                        device=str(device),
                        log_interval=model_config.get("log_interval", 10),
                        use_class_weights=model_config.get("use_class_weights", True),
                        use_amp=model_config.get("use_amp", True),
                        gradient_accumulation_steps=gradient_accumulation_steps,  # Use updated value after batch_size override
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
                        # Note: fold_output_dir already checked above for resume mode
                        fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                        fold_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Note: Fold deletion already handled above before the resume check
                    
                        if use_tracking:
                            tracker = ExperimentTracker(str(fold_output_dir))
                        # Generate unique run_id for this fold and hyperparameter combination
                        run_id = f"{model_type}_fold{fold_idx + 1}_param{param_idx + 1}"
                        ckpt_manager = CheckpointManager(str(fold_output_dir), run_id=run_id)
                        
                        # Create MLflow tracker if available
                        mlflow_tracker = None
                        if use_mlflow and MLFLOW_AVAILABLE:
                            try:
                                # End any existing MLflow run before creating a new one
                                try:
                                    import mlflow
                                    if mlflow.active_run() is not None:
                                        mlflow.end_run()
                                        logger.debug("Ended existing MLflow run before creating new one")
                                except Exception:
                                    pass  # Ignore errors when ending runs
                                
                                mlflow_tracker = create_mlflow_tracker(
                                    experiment_name=f"{model_type}",
                                    use_mlflow=True
                                )
                                if mlflow_tracker:
                                    # Log model config (can be dict or RunConfig)
                                    mlflow_tracker.log_config(model_config)
                                    mlflow_tracker.set_tag("fold", str(fold_idx + 1))
                                    mlflow_tracker.set_tag("model_type", model_type)
                                    mlflow_tracker.set_tag("param_combination", str(param_idx + 1))
                            except Exception as e:
                                logger.warning(f"Failed to create MLflow tracker: {e}")
                        else:
                            tracker = None
                            ckpt_manager = None
                    
                        logger.info(f"Training PyTorch model {model_type} on fold {fold_idx + 1}...")
                        _flush_logs()
                    
                        # Validate datasets before training
                        if len(train_dataset) == 0:
                            raise ValueError(f"Training dataset is empty for fold {fold_idx + 1}")
                        if len(val_dataset) == 0:
                            raise ValueError(f"Validation dataset is empty for fold {fold_idx + 1}")
                    
                        # Validate model initialization with OOM-resistant forward pass test
                        try:
                            # Clear CUDA cache before test to maximize available memory
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            
                            model.eval()
                            with torch.no_grad():
                                # Forward pass test uses batch_size=1 to minimize memory
                                # NOTE: This is ONLY for testing model initialization.
                                # Actual training uses batch_size={batch_size} with gradient_accumulation_steps={gradient_accumulation_steps}
                                # (effective batch size = {effective_batch_size})
                                # Create a minimal test loader with batch_size=1
                                test_loader = DataLoader(
                                    train_dataset,
                                    batch_size=1,  # Single sample to minimize memory (ONLY for test, not training)
                                    shuffle=False,
                                    num_workers=0,  # No workers to avoid memory overhead
                                    pin_memory=False,  # Disable pinning for test
                                    collate_fn=variable_ar_collate,  # Use same collate function as training
                                )
                                
                                # Get a single sample
                                sample_batch = next(iter(test_loader))
                                sample_clips, sample_labels = sample_batch
                                
                                # CRITICAL: Validate input dimensions for X3D and SlowFast models
                                # These models require minimum spatial dimensions (typically 32x32 or larger)
                                # X3D uses pooling kernels of size 7x7, so needs at least 7x7, but ideally 32x32+
                                # SlowFast also requires reasonable spatial dimensions
                                if model_type in ["x3d", "slowfast"]:
                                    # Check input shape: (N, C, T, H, W) or (N, T, C, H, W)
                                    if sample_clips.dim() == 5:
                                        if sample_clips.shape[1] == 3:  # (N, C, T, H, W)
                                            N, C, T, H, W = sample_clips.shape
                                        else:  # (N, T, C, H, W)
                                            N, T, C, H, W = sample_clips.shape
                                        
                                        # X3D requires minimum 32x32 spatial dimensions (pooling kernel is 7x7, but needs buffer)
                                        # SlowFast also requires reasonable spatial dimensions
                                        min_spatial_size = 32
                                        if H < min_spatial_size or W < min_spatial_size:
                                            logger.warning(
                                                f"Skipping forward pass test for {model_type}: input spatial dimensions "
                                                f"({H}x{W}) are too small (minimum {min_spatial_size}x{min_spatial_size} required). "
                                                f"Temporal dimension: {T}. "
                                                f"This video may be filtered during training. Continuing with training..."
                                            )
                                            _flush_logs()
                                            # Skip the forward pass test - training will handle small videos via error handling
                                            del sample_batch, sample_clips, sample_labels
                                            del test_loader
                                            if device.type == "cuda":
                                                torch.cuda.empty_cache()
                                                torch.cuda.synchronize()
                                            # Continue to training - the DataLoader will handle small videos
                                            break
                                
                                # Move to device
                                sample_clips = sample_clips.to(device, non_blocking=False)
                                
                                # Test forward pass
                                try:
                                    sample_output = model(sample_clips)
                                    logger.info(f"Model forward pass test successful. Output shape: {sample_output.shape}")
                                    _flush_logs()
                                except RuntimeError as oom_error:
                                    error_msg = str(oom_error)
                                    if "out of memory" in error_msg.lower():
                                        logger.warning(
                                            f"OOM during forward pass test. "
                                            f"Model: {model_type}, Batch size: 1. "
                                            f"This may indicate the model is too large for available GPU memory."
                                        )
                                        # Try to continue anyway - sometimes training with gradient accumulation works
                                        logger.warning("Attempting to continue with training (may fail if model is too large)...")
                                        # Don't raise - let training attempt proceed
                                    elif "smaller than kernel size" in error_msg.lower() or "input image" in error_msg.lower():
                                        # Handle dimension mismatch errors gracefully for X3D/SlowFast
                                        logger.warning(
                                            f"Input dimension mismatch during forward pass test for {model_type}: {oom_error}. "
                                            f"Input shape: {sample_clips.shape}. "
                                            f"This may indicate some videos have very small spatial dimensions. "
                                            f"Training will handle this via error handling. Continuing..."
                                        )
                                        # Don't raise - let training attempt proceed
                                    else:
                                        raise
                                
                                # Cleanup
                                del sample_batch, sample_clips, sample_labels
                                if 'sample_output' in locals():
                                    del sample_output
                                del test_loader
                                
                                # Aggressive cache clearing
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                    
                        except RuntimeError as e:
                            error_msg = str(e)
                            if "out of memory" in error_msg.lower():
                                logger.error(
                                    f"CUDA OOM during model forward pass test: {e}. "
                                    f"Model: {model_type}, Batch size: 1. "
                                    f"GPU memory may be insufficient for this model."
                                )
                                # Clear cache and try to continue
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()
                                # Don't raise - let training attempt proceed with warning
                                logger.warning("Continuing with training despite OOM in forward pass test...")
                            else:
                                logger.error(f"Model forward pass test failed: {e}", exc_info=True)
                                raise ValueError(f"Model initialization failed: {e}") from e
                        except Exception as e:
                            logger.error(f"Model forward pass test failed: {e}", exc_info=True)
                            raise ValueError(f"Model initialization failed: {e}") from e
                    
                        # Train with comprehensive error handling and OOM recovery
                        max_oom_retries = 3
                        oom_retry_count = 0
                        training_successful = False
                        
                        while oom_retry_count <= max_oom_retries and not training_successful:
                            try:
                                # If we've had OOM errors, reduce batch size and recreate loaders
                                if oom_retry_count > 0:
                                    # Reduce batch size by half (minimum 1)
                                    new_batch_size = max(1, batch_size // (2 ** oom_retry_count))
                                    if new_batch_size < batch_size:
                                        logger.warning(
                                            f"OOM retry {oom_retry_count}: Reducing batch size from {batch_size} to {new_batch_size}"
                                        )
                                        batch_size = new_batch_size
                                        # Increase gradient accumulation to maintain effective batch size
                                        gradient_accumulation_steps = effective_batch_size // batch_size
                                        if gradient_accumulation_steps < 1:
                                            gradient_accumulation_steps = 1
                                        
                                        # Update train_cfg with new gradient accumulation
                                        train_cfg.gradient_accumulation_steps = gradient_accumulation_steps
                                        
                                        # Recreate data loaders with new batch size
                                        train_loader = DataLoader(
                                            train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=use_cuda,
                                            persistent_workers=num_workers > 0,
                                            prefetch_factor=2 if num_workers > 0 else None,
                                            collate_fn=variable_ar_collate,
                                        )
                                        val_loader = DataLoader(
                                            val_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_workers,
                                            pin_memory=use_cuda,
                                            persistent_workers=num_workers > 0,
                                            prefetch_factor=2 if num_workers > 0 else None,
                                            collate_fn=variable_ar_collate,
                                        )
                                        
                                        logger.info(
                                            f"Retrying with batch_size={batch_size}, "
                                            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
                                            f"effective_batch_size={batch_size * gradient_accumulation_steps}"
                                        )
                                        
                                        # Clear cache before retry
                                        if device.type == "cuda":
                                            torch.cuda.empty_cache()
                                            torch.cuda.synchronize()
                                
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
                                
                                # Clear cache before evaluation
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                
                                val_metrics = evaluate(model, val_loader, device=str(device))
                                
                                val_loss = val_metrics["loss"]
                                val_acc = val_metrics["accuracy"]
                                val_f1 = val_metrics["f1"]
                                val_precision = val_metrics["precision"]
                                val_recall = val_metrics["recall"]
                                per_class = val_metrics["per_class"]
                                
                                training_successful = True
                                
                                # Aggressive GC after successful training and evaluation
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                    aggressive_gc(clear_cuda=True)
                                
                            except RuntimeError as e:
                                # Catch CUDA OOM, invalid tensor operations, etc.
                                error_msg = str(e)
                                if ("out of memory" in error_msg.lower() or "cuda" in error_msg.lower()) and oom_retry_count < max_oom_retries:
                                    logger.warning(
                                        f"CUDA OOM during training (attempt {oom_retry_count + 1}/{max_oom_retries + 1}): {e}. "
                                        f"Model: {model_type}, Fold: {fold_idx + 1}, "
                                        f"Current batch size: {batch_size}"
                                    )
                                    # Clean up GPU memory aggressively
                                    if device.type == "cuda":
                                        # Multiple passes of cache clearing
                                        for _ in range(3):
                                            torch.cuda.empty_cache()
                                            torch.cuda.synchronize()
                                        aggressive_gc(clear_cuda=True)
                                    # Increment retry count and try again with smaller batch size
                                    oom_retry_count += 1
                                    continue
                                else:
                                    # Not OOM or max retries reached
                                    if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                                        logger.error(
                                            f"CUDA OOM or runtime error during training (max retries reached): {e}. "
                                            f"Model: {model_type}, Fold: {fold_idx + 1}, "
                                            f"Final batch size: {batch_size}"
                                        )
                                    else:
                                        logger.error(
                                            f"Runtime error during training: {e}. "
                                            f"Model: {model_type}, Fold: {fold_idx + 1}"
                                        )
                                    # Clean up GPU memory
                                    if device.type == "cuda":
                                        torch.cuda.empty_cache()
                                    raise
                            except ValueError as e:
                                logger.error(
                                    f"Value error during training (likely input shape issue): {e}. "
                                    f"Model: {model_type}, Fold: {fold_idx + 1}"
                                )
                                raise
                            except Exception as e:
                                logger.error(
                                    f"Unexpected error during training: {e}. "
                                    f"Model: {model_type}, Fold: {fold_idx + 1}",
                                    exc_info=True
                                )
                                # Clean up GPU memory
                                if device.type == "cuda":
                                    torch.cuda.empty_cache()
                                raise
                    
                        # Store results with hyperparameters (only reached on success)
                        result = {
                        "fold": fold_idx + 1,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "val_precision": val_precision,
                        "val_recall": val_recall,
                        "val_f1_class0": per_class.get("0", {}).get("f1", 0.0),
                        "val_precision_class0": per_class.get("0", {}).get("precision", 0.0),
                        "val_recall_class0": per_class.get("0", {}).get("recall", 0.0),
                        "val_f1_class1": per_class.get("1", {}).get("f1", 0.0),
                        "val_precision_class1": per_class.get("1", {}).get("precision", 0.0),
                        "val_recall_class1": per_class.get("1", {}).get("recall", 0.0),
                        }
                        # Add hyperparameters to result
                        result.update(params)
                        param_fold_results.append(result)
                        fold_results.append(result)
                    
                        logger.info(
                        f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                        f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
                        )
                        _flush_logs()
                        if per_class:
                            for class_idx, metrics in per_class.items():
                                logger.info(
                                    f"  Class {class_idx} - Precision: {metrics['precision']:.4f}, "
                                    f"Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}"
                                )
                    
                        # Log to MLflow if available
                        if 'mlflow_tracker' in locals() and mlflow_tracker is not None:
                            try:
                                mlflow_metrics = {
                                    "val_loss": val_loss,
                                    "val_acc": val_acc,
                                    "val_f1": val_f1,
                                    "val_precision": val_precision,
                                    "val_recall": val_recall,
                                }
                                # Add per-class metrics
                                for class_idx, metrics in per_class.items():
                                    mlflow_metrics[f"val_precision_class{class_idx}"] = metrics["precision"]
                                    mlflow_metrics[f"val_recall_class{class_idx}"] = metrics["recall"]
                                    mlflow_metrics[f"val_f1_class{class_idx}"] = metrics["f1"]
                                
                                mlflow_tracker.log_metrics(mlflow_metrics, step=fold_idx + 1)
                                # Log model artifact (after model is saved)
                                if 'model_path' in locals():
                                    model_path_str = str(model_path)
                                    mlflow_tracker.log_artifact(
                                        model_path_str, artifact_path="models"
                                    )
                            except Exception as e:
                                logger.warning(f"Failed to log to MLflow: {e}")
                    
                        # Save model for ensemble training
                        try:
                            model.eval()
                            model_path = fold_output_dir / "model.pt"
                            torch.save(model.state_dict(), model_path)
                            logger.info(f"Saved model to {model_path}")
                        except (OSError, IOError, PermissionError) as e:
                            logger.error(f"Failed to save model to {model_path}: {e}")
                            raise IOError(f"Cannot save model to {model_path}") from e
                    
                    elif is_xgboost_model(model_type):
                        # XGBoost model training (uses pretrained models for feature extraction)
                        logger.info(f"Training XGBoost model {model_type} on fold {fold_idx + 1}...")
                        _flush_logs()
                        
                        fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                        fold_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Note: Fold deletion already handled above before the resume check
                        
                        try:
                            # Create XGBoost model with hyperparameters
                            xgb_config = model_config.copy()
                            xgb_config.update(params)  # Apply grid search hyperparameters
                            model = create_model(model_type, xgb_config)
                            
                            # Train XGBoost (handles feature extraction internally)
                            model.fit(train_df, project_root=project_root_str_orig)
                            
                            # Evaluate on validation set
                            val_probs = model.predict(val_df, project_root=project_root_str_orig)
                            val_preds = np.argmax(val_probs, axis=1)
                            val_labels = val_df["label"].to_list()
                            label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
                            val_y = np.array([label_map[label] for label in val_labels])
                            
                            # Compute comprehensive metrics using shared utility
                            metrics = compute_classification_metrics(
                                y_true=val_y,
                                y_pred=val_preds,
                                y_probs=val_probs
                            )
                            
                            # Store results with hyperparameters
                            result = {
                                "fold": fold_idx + 1,
                                "val_loss": metrics["val_loss"],
                                "val_acc": metrics["val_acc"],
                                "val_f1": metrics["val_f1"],
                                "val_precision": metrics["val_precision"],
                                "val_recall": metrics["val_recall"],
                                "val_f1_class0": metrics["val_f1_class0"],
                                "val_precision_class0": metrics["val_precision_class0"],
                                "val_recall_class0": metrics["val_recall_class0"],
                                "val_f1_class1": metrics["val_f1_class1"],
                                "val_precision_class1": metrics["val_precision_class1"],
                                "val_recall_class1": metrics["val_recall_class1"],
                            }
                            result.update(params)  # Add hyperparameters
                            param_fold_results.append(result)
                            fold_results.append(result)
                            
                            logger.info(
                                f"Fold {fold_idx + 1} - Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}, "
                                f"Val F1: {metrics['val_f1']:.4f}, Val Precision: {metrics['val_precision']:.4f}, Val Recall: {metrics['val_recall']:.4f}"
                            )
                            logger.info(
                                f"  Class 0 - Precision: {metrics['val_precision_class0']:.4f}, "
                                f"Recall: {metrics['val_recall_class0']:.4f}, F1: {metrics['val_f1_class0']:.4f}"
                            )
                            logger.info(
                                f"  Class 1 - Precision: {metrics['val_precision_class1']:.4f}, "
                                f"Recall: {metrics['val_recall_class1']:.4f}, F1: {metrics['val_f1_class1']:.4f}"
                            )
                            
                            # Save model
                            model.save(str(fold_output_dir))
                            logger.info(f"Saved XGBoost model to {fold_output_dir}")
                        
                        except Exception as e:
                            logger.error(f"Error training XGBoost fold {fold_idx + 1}: {e}", exc_info=True)
                            result = {
                                "fold": fold_idx + 1,
                                "val_loss": float('nan'),
                                "val_acc": float('nan'),
                                "val_f1": float('nan'),
                                "val_precision": float('nan'),
                                "val_recall": float('nan'),
                                "val_f1_class0": float('nan'),
                                "val_precision_class0": float('nan'),
                                "val_recall_class0": float('nan'),
                                "val_f1_class1": float('nan'),
                                "val_precision_class1": float('nan'),
                                "val_recall_class1": float('nan'),
                            }
                            result.update(params)
                            param_fold_results.append(result)
                            fold_results.append(result)
                        
                        finally:
                            # Always cleanup resources, even on error
                            cleanup_model_and_memory(model=model if 'model' in locals() else None, clear_cuda=True)
                            aggressive_gc(clear_cuda=True)
                    
                    else:
                        # Baseline model training (sklearn)
                        logger.info(f"Training baseline model {model_type} on fold {fold_idx + 1}...")
                        _flush_logs()
                        
                        fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                        fold_output_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Note: Fold deletion already handled above before the resume check
                        
                        try:
                            # Create baseline model with hyperparameters
                            baseline_config = model_config.copy()
                            baseline_config.update(params)  # Apply grid search hyperparameters
                            
                            # Add feature paths for baseline models
                            # All baseline models (svm, logistic_regression and their variants) require Stage 2 features
                            # Stage 5 only trains - features must be pre-extracted in Stage 2/4
                            from lib.utils.paths import load_metadata_flexible
                            
                            # All baseline models require Stage 2 features
                            if model_type in BASELINE_MODELS:
                                # Check if Stage 2 metadata exists and is not empty
                                stage2_df = load_metadata_flexible(features_stage2_path)
                                if stage2_df is not None and stage2_df.height > 0:
                                    # CRITICAL: Set features_stage2_path in model_specific_config, not top level
                                    # create_model() looks for it in model_specific_config dict
                                    if "model_specific_config" not in baseline_config:
                                        baseline_config["model_specific_config"] = {}
                                    baseline_config["model_specific_config"]["features_stage2_path"] = features_stage2_path
                                    # Also set at top level for backward compatibility
                                    baseline_config["features_stage2_path"] = features_stage2_path
                                    logger.debug(f"Passing Stage 2 features path to {model_type}: {features_stage2_path}")
                                else:
                                    # Stage 2 is required for all baseline models - fail early with clear error
                                    raise ValueError(
                                        f"Stage 2 features are REQUIRED for {model_type}. "
                                        f"Features must be pre-extracted in Stage 2. "
                                        f"Stage 2 metadata not found or empty at: {features_stage2_path}. "
                                        f"Please run Stage 2 feature extraction first."
                                    )
                                
                                # For models that use Stage 4, check if it exists
                                if model_type in STAGE4_MODELS:
                                    stage4_df = load_metadata_flexible(features_stage4_path)
                                    if stage4_df is not None and stage4_df.height > 0:
                                        # CRITICAL: Set features_stage4_path in model_specific_config
                                        if "model_specific_config" not in baseline_config:
                                            baseline_config["model_specific_config"] = {}
                                        baseline_config["model_specific_config"]["features_stage4_path"] = features_stage4_path
                                        # Also set at top level for backward compatibility
                                        baseline_config["features_stage4_path"] = features_stage4_path
                                        logger.debug(f"Passing Stage 4 features path to {model_type}: {features_stage4_path}")
                                    else:
                                        # Stage 4 is required for these models
                                        raise ValueError(
                                            f"Stage 4 features are REQUIRED for {model_type}. "
                                            f"Features must be pre-extracted in Stage 4. "
                                            f"Stage 4 metadata not found or empty at: {features_stage4_path}. "
                                            f"Please run Stage 4 scaled feature extraction first."
                                        )
                                else:
                                    # For stage2_only models, explicitly set stage4_path to None
                                    if "model_specific_config" not in baseline_config:
                                        baseline_config["model_specific_config"] = {}
                                    baseline_config["model_specific_config"]["features_stage4_path"] = None
                                    baseline_config["features_stage4_path"] = None
                            
                            model = create_model(model_type, baseline_config)
                            
                            # Train baseline (handles feature extraction internally)
                            # Wrap in try-except to catch potential crashes
                            logger.info(f"Starting model.fit() for {model_type} fold {fold_idx + 1}...")
                            logger.info(f"Training data: {train_df.height} rows")
                            _flush_logs()
                            
                            try:
                                model.fit(train_df, project_root=project_root_str_orig)
                                logger.info(f"Model.fit() completed successfully for fold {fold_idx + 1}")
                                _flush_logs()
                            except MemoryError as e:
                                logger.error(f"Memory error during model.fit() for fold {fold_idx + 1}: {e}")
                                raise
                            except Exception as e:
                                error_msg = str(e)
                                if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                                    logger.critical(f"CRITICAL: Possible crash during model.fit() for fold {fold_idx + 1}: {e}")
                                    logger.critical("This may indicate a memory issue, corrupted data, or library incompatibility")
                                    logger.critical("Check log file for more details")
                                raise
                            
                            # Evaluate on validation set
                            logger.info(f"Starting model.predict() for {model_type} fold {fold_idx + 1}...")
                            logger.info(f"Validation data: {val_df.height} rows")
                            _flush_logs()
                            
                            try:
                                val_probs = model.predict(val_df, project_root=project_root_str_orig)
                                logger.info(f"Model.predict() completed successfully for fold {fold_idx + 1}")
                                _flush_logs()
                            except MemoryError as e:
                                logger.error(f"Memory error during model.predict() for fold {fold_idx + 1}: {e}")
                                raise
                            except Exception as e:
                                error_msg = str(e)
                                if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                                    logger.critical(f"CRITICAL: Possible crash during model.predict() for fold {fold_idx + 1}: {e}")
                                    logger.critical("This may indicate a memory issue, corrupted data, or library incompatibility")
                                    logger.critical("Check log file for more details")
                                raise
                            val_preds = np.argmax(val_probs, axis=1)
                            val_labels = val_df["label"].to_list()
                            label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
                            val_y = np.array([label_map[label] for label in val_labels])
                            
                            # Compute comprehensive metrics using shared utility
                            metrics = compute_classification_metrics(
                                y_true=val_y,
                                y_pred=val_preds,
                                y_probs=val_probs
                            )
                            
                            # Store results with hyperparameters
                            result = {
                                "fold": fold_idx + 1,
                                "val_loss": metrics["val_loss"],
                                "val_acc": metrics["val_acc"],
                                "val_f1": metrics["val_f1"],
                                "val_precision": metrics["val_precision"],
                                "val_recall": metrics["val_recall"],
                                "val_f1_class0": metrics["val_f1_class0"],
                                "val_precision_class0": metrics["val_precision_class0"],
                                "val_recall_class0": metrics["val_recall_class0"],
                                "val_f1_class1": metrics["val_f1_class1"],
                                "val_precision_class1": metrics["val_precision_class1"],
                                "val_recall_class1": metrics["val_recall_class1"],
                            }
                            result.update(params)  # Add hyperparameters
                            param_fold_results.append(result)
                            fold_results.append(result)
                            
                            logger.info(
                                f"Fold {fold_idx + 1} - Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}, "
                                f"Val F1: {metrics['val_f1']:.4f}, Val Precision: {metrics['val_precision']:.4f}, Val Recall: {metrics['val_recall']:.4f}"
                            )
                            logger.info(
                                f"  Class 0 - Precision: {metrics['val_precision_class0']:.4f}, "
                                f"Recall: {metrics['val_recall_class0']:.4f}, F1: {metrics['val_f1_class0']:.4f}"
                            )
                            logger.info(
                                f"  Class 1 - Precision: {metrics['val_precision_class1']:.4f}, "
                                f"Recall: {metrics['val_recall_class1']:.4f}, F1: {metrics['val_f1_class1']:.4f}"
                            )
                            
                            # Save model
                            model.save(str(fold_output_dir))
                            logger.info(f"Saved baseline model to {fold_output_dir}")
                        
                        except Exception as e:
                            logger.error(
                                f"Error training baseline fold {fold_idx + 1}: {e}",
                                exc_info=True
                            )
                            result = {
                                "fold": fold_idx + 1,
                                "val_loss": float('nan'),
                                "val_acc": float('nan'),
                                "val_f1": float('nan'),
                                "val_precision": float('nan'),
                                "val_recall": float('nan'),
                                "val_f1_class0": float('nan'),
                                "val_precision_class0": float('nan'),
                                "val_recall_class0": float('nan'),
                                "val_f1_class1": float('nan'),
                                "val_precision_class1": float('nan'),
                                "val_recall_class1": float('nan'),
                            }
                            result.update(params)
                            param_fold_results.append(result)
                            fold_results.append(result)
                        finally:
                            # Always clear model and aggressively free memory, even on error
                            cleanup_model_and_memory(model=model if 'model' in locals() else None, clear_cuda=False)
                            aggressive_gc(clear_cuda=False)
                    
                except Exception as e:
                    logger.error(f"Error training fold {fold_idx + 1}: {e}", exc_info=True)
                    fold_results.append({
                        "fold": fold_idx + 1,
                        "val_loss": float('nan'),
                        "val_acc": float('nan'),
                        "val_f1": float('nan'),
                        "val_precision": float('nan'),
                        "val_recall": float('nan'),
                        "val_f1_class0": float('nan'),
                        "val_precision_class0": float('nan'),
                        "val_recall_class0": float('nan'),
                        "val_f1_class1": float('nan'),
                        "val_precision_class1": float('nan'),
                        "val_recall_class1": float('nan'),
                    })
                finally:
                    # Always cleanup resources, even on error
                    # End MLflow run if active
                    if 'mlflow_tracker' in locals() and mlflow_tracker is not None:
                        try:
                            mlflow_tracker.end_run()
                        except Exception as e:
                            logger.debug(f"Error ending MLflow run: {e}")
                    
                    # Clear model and aggressively free memory
                    device_obj = device if 'device' in locals() else None
                    cleanup_model_and_memory(
                        model=model if 'model' in locals() else None,
                        device=device_obj,
                        clear_cuda=device_obj.type == "cuda" if device_obj and device_obj.type == "cuda" else False
                    )
                    aggressive_gc(clear_cuda=device_obj.type == "cuda" if device_obj and device_obj.type == "cuda" else False)
            
            # After all folds for this parameter combination, aggregate results
            if param_fold_results:
                mean_f1 = np.mean([r.get("val_f1", 0) for r in param_fold_results if not np.isnan(r.get("val_f1", 0))])
                mean_acc = np.mean([r.get("val_acc", 0) for r in param_fold_results if not np.isnan(r.get("val_acc", 0))])
                grid_result = {
                    "mean_f1": mean_f1,
                    "mean_acc": mean_acc,
                    "fold_results": param_fold_results
                }
                grid_result.update(params)  # Include hyperparameters
                all_grid_results.append(grid_result)
                logger.info(f"Parameter combination {param_idx + 1} - Mean F1: {mean_f1:.4f}, Mean Acc: {mean_acc:.4f}")
                _flush_logs()
        
        # Select best hyperparameters from grid search (on sample)
        best_params = None
        if param_combinations and all_grid_results and len(all_grid_results) > 1:
            best_params = select_best_hyperparameters(model_type, all_grid_results)
            logger.info(f"Best hyperparameters selected from {grid_search_sample_size*100:.1f}% sample: {best_params}")
            _flush_logs()
        elif param_combinations and len(param_combinations) == 1:
            # Single parameter combination - use it
            best_params = param_combinations[0]
            logger.info(f"Using single parameter combination: {best_params}")
        
        # FINAL TRAINING: Train on full dataset with best hyperparameters
        logger.info("=" * 80)
        logger.info("FINAL TRAINING: Using full dataset with best hyperparameters")
        logger.info("=" * 80)
        _flush_logs()
        
        # Create folds from full dataset for final training
        all_folds = stratified_kfold(
            scaled_df,
            n_splits=n_splits,
            random_state=42
        )
        
        if len(all_folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds for final training, got {len(all_folds)}")
        
        logger.info(f"Final training: Using {n_splits}-fold stratified cross-validation on full dataset ({scaled_df.height} rows)")
        _flush_logs()
        
        # Train on full dataset with best hyperparameters
        fold_results = []
        final_config = model_config.copy()
        if best_params:
            final_config.update(best_params)
            logger.info(f"Final training using best hyperparameters: {best_params}")
        else:
            logger.info("Final training using default hyperparameters (no grid search)")
        
        # Train all folds on full dataset with best hyperparameters
        for fold_idx in range(n_splits):
            logger.info(f"\nFinal Training - {model_type} - Fold {fold_idx + 1}/{n_splits} (full dataset)")
            _flush_logs()
            
            # Delete existing fold if delete_existing is True (do this BEFORE checking if complete)
            fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
            if delete_existing and fold_output_dir.exists():
                try:
                    import shutil
                    shutil.rmtree(fold_output_dir)
                    logger.info(f"Deleted existing final training fold {fold_idx + 1} directory (clean mode)")
                    _flush_logs()
                except (OSError, PermissionError, FileNotFoundError) as e:
                    logger.warning(f"Could not delete {fold_output_dir}: {e}")
                    _flush_logs()
            
            # Check if fold is already complete (resume mode) - only if not deleting
            if resume and not delete_existing and _is_fold_complete(fold_output_dir, model_type):
                logger.info(f"Final training fold {fold_idx + 1} already trained (found existing model). Skipping.")
                logger.info(f"To retrain this fold, use --delete-existing flag")
                # Try to load existing results if available
                # For now, we'll skip and continue - the fold won't be in results
                continue
            
            # Get the specific fold from full dataset
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
            
            # Train model with best hyperparameters (reuse same training code as grid search)
            try:
                if is_pytorch_model(model_type):
                    # PyTorch model training with best hyperparameters
                    from lib.models import VideoDataset
                    from lib.models.video import variable_ar_collate
                    
                    train_dataset = VideoDataset(train_df, project_root=project_root_str_orig, config=video_config)
                    val_dataset = VideoDataset(val_df, project_root=project_root_str_orig, config=video_config)
                    
                    use_cuda = torch.cuda.is_available()
                    num_workers = final_config.get("num_workers", model_config.get("num_workers", 0))
                    batch_size = final_config.get("batch_size", model_config.get("batch_size", 8))
                    gradient_accumulation_steps = final_config.get("gradient_accumulation_steps", model_config.get("gradient_accumulation_steps", 1))
                    
                    # CRITICAL: Force smaller batch sizes for memory-intensive models to prevent OOM
                    # These models process many frames (1000) at high resolution, requiring conservative batch sizes
                    # Use same limits as defined above for consistency
                    MEMORY_INTENSIVE_MODELS_BATCH_LIMITS = {
                        "x3d": 1,  # Very memory intensive
                        "naive_cnn": 1,  # Processes 1000 frames at full resolution - must use batch_size=1
                        "variable_ar_cnn": 2,  # Processes variable-length videos with many frames
                        "pretrained_inception": 2,  # Large pretrained model processing many frames
                    }
                    
                    if model_type in MEMORY_INTENSIVE_MODELS_BATCH_LIMITS:
                        max_batch_size = MEMORY_INTENSIVE_MODELS_BATCH_LIMITS[model_type]
                        if batch_size > max_batch_size:
                            # Calculate effective batch size before reduction
                            effective_batch_size = batch_size * gradient_accumulation_steps
                            logger.warning(
                                f"{model_type} model requires batch_size<={max_batch_size} to prevent OOM. "
                                f"Overriding batch_size from {batch_size} to {max_batch_size}. "
                                f"Adjusting gradient_accumulation_steps to maintain effective batch size of {effective_batch_size}."
                            )
                            # Increase gradient accumulation to maintain effective batch size
                            gradient_accumulation_steps = (effective_batch_size + max_batch_size - 1) // max_batch_size
                            batch_size = max_batch_size
                    
                    train_loader = DataLoader(
                        train_dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=use_cuda,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                        collate_fn=variable_ar_collate
                    )
                    val_loader = DataLoader(
                        val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=use_cuda,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                        collate_fn=variable_ar_collate
                    )
                    
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    
                    # CRITICAL: Apply PyTorch memory optimizations to prevent GPU memory from being hogged
                    if device.type == "cuda":
                        # Set CUDA memory allocator to use expandable segments (reduces fragmentation)
                        import os
                        if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
                            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                        
                        # Disable cuDNN benchmark to reduce memory usage (trade-off: slightly slower)
                        torch.backends.cudnn.benchmark = False
                        
                        # Clear cache before model creation
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        aggressive_gc(clear_cuda=True)
                        logger.info("Applied PyTorch memory optimizations: expandable_segments, cudnn.benchmark=False")
                    
                    model = create_model(model_type, final_config)
                    model = model.to(device)
                    
                    optim_cfg = OptimConfig(
                        lr=final_config.get("learning_rate", model_config.get("learning_rate", 1e-4)),
                        weight_decay=final_config.get("weight_decay", model_config.get("weight_decay", 1e-4)),
                        max_grad_norm=final_config.get("max_grad_norm", model_config.get("max_grad_norm", 1.0)),
                        backbone_lr=final_config.get("backbone_lr", model_config.get("backbone_lr", None)),
                        head_lr=final_config.get("head_lr", model_config.get("head_lr", None)),
                    )
                    train_cfg = TrainConfig(
                        num_epochs=final_config.get("num_epochs", model_config.get("num_epochs", 20)),
                        device=str(device),
                        log_interval=model_config.get("log_interval", 10),
                        use_class_weights=model_config.get("use_class_weights", True),
                        use_amp=model_config.get("use_amp", True),
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        early_stopping_patience=model_config.get("early_stopping_patience", 5),
                        scheduler_type=model_config.get("scheduler_type", "cosine"),
                        warmup_epochs=model_config.get("warmup_epochs", 2),
                        warmup_factor=model_config.get("warmup_factor", 0.1),
                        log_grad_norm=model_config.get("log_grad_norm", False),
                    )
                    
                    use_differential_lr = model_type in [
                        "i3d", "r2plus1d", "slowfast", "x3d", "pretrained_inception",
                        "vit_gru", "vit_transformer"
                    ]
                    
                    fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                    fold_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Note: Fold deletion already handled above before the resume check
                    
                    if use_tracking:
                        tracker = ExperimentTracker(str(fold_output_dir))
                    run_id = f"{model_type}_fold{fold_idx + 1}_final"
                    ckpt_manager = CheckpointManager(str(fold_output_dir), run_id=run_id)
                    
                    mlflow_tracker = None
                    if use_mlflow and MLFLOW_AVAILABLE:
                        try:
                            # End any existing MLflow run before creating a new one
                            try:
                                import mlflow
                                if mlflow.active_run() is not None:
                                    mlflow.end_run()
                                    logger.debug("Ended existing MLflow run before creating new one")
                            except Exception:
                                pass  # Ignore errors when ending runs
                            
                            mlflow_tracker = create_mlflow_tracker(
                                experiment_name=f"{model_type}",
                                use_mlflow=True
                            )
                            if mlflow_tracker:
                                mlflow_tracker.log_config(final_config)
                                mlflow_tracker.set_tag("fold", str(fold_idx + 1))
                                mlflow_tracker.set_tag("model_type", model_type)
                                mlflow_tracker.set_tag("phase", "final_training")
                        except Exception as e:
                            logger.warning(f"Failed to create MLflow tracker: {e}")
                    
                    logger.info(f"Training PyTorch model {model_type} on fold {fold_idx + 1} (full dataset)...")
                    
                    # Clear cache before training
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    model = fit(model, train_loader, val_loader, optim_cfg, train_cfg, use_differential_lr=use_differential_lr)
                    
                    # Clear cache before evaluation
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    
                    from lib.training.trainer import evaluate
                    val_metrics = evaluate(model, val_loader, device=str(device))
                    
                    val_loss = val_metrics["loss"]
                    val_acc = val_metrics["accuracy"]
                    val_f1 = val_metrics["f1"]
                    val_precision = val_metrics["precision"]
                    val_recall = val_metrics["recall"]
                    per_class = val_metrics["per_class"]
                    
                    result = {
                        "fold": fold_idx + 1,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_f1": val_f1,
                        "val_precision": val_precision,
                        "val_recall": val_recall,
                        "val_f1_class0": per_class.get("0", {}).get("f1", 0.0),
                        "val_precision_class0": per_class.get("0", {}).get("precision", 0.0),
                        "val_recall_class0": per_class.get("0", {}).get("recall", 0.0),
                        "val_f1_class1": per_class.get("1", {}).get("f1", 0.0),
                        "val_precision_class1": per_class.get("1", {}).get("precision", 0.0),
                        "val_recall_class1": per_class.get("1", {}).get("recall", 0.0),
                    }
                    if best_params:
                        result.update(best_params)
                    fold_results.append(result)
                    
                    logger.info(
                        f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                        f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
                    )
                    
                    model.eval()
                    model_path = fold_output_dir / "model.pt"
                    torch.save(model.state_dict(), model_path)
                    logger.info(f"Saved model to {model_path}")
                    
                    # Aggressive GC after saving model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        aggressive_gc(clear_cuda=True)
                    
                    if mlflow_tracker:
                        try:
                            mlflow_tracker.log_metrics({
                                "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
                                "val_precision": val_precision, "val_recall": val_recall
                            }, step=fold_idx + 1)
                        except Exception as e:
                            logger.warning(f"Failed to log to MLflow: {e}")
                    
                    cleanup_model_and_memory(model=model, device=device, clear_cuda=device.type == "cuda")
                    aggressive_gc(clear_cuda=device.type == "cuda")
                    
                elif is_xgboost_model(model_type):
                    # XGBoost model training with best hyperparameters
                    fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                    fold_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Note: Fold deletion already handled above before the resume check
                    
                    xgb_config = model_config.copy()
                    if best_params:
                        xgb_config.update(best_params)
                    model = create_model(model_type, xgb_config)
                    
                    model.fit(train_df, project_root=project_root_str_orig)
                    val_probs = model.predict(val_df, project_root=project_root_str_orig)
                    val_preds = np.argmax(val_probs, axis=1)
                    val_labels = val_df["label"].to_list()
                    label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
                    val_y = np.array([label_map[label] for label in val_labels])
                    
                    # Compute comprehensive metrics using shared utility
                    metrics = compute_classification_metrics(
                        y_true=val_y,
                        y_pred=val_preds,
                        y_probs=val_probs
                    )
                    
                    result = {
                        "fold": fold_idx + 1,
                        "val_loss": metrics["val_loss"],
                        "val_acc": metrics["val_acc"],
                        "val_f1": metrics["val_f1"],
                        "val_precision": metrics["val_precision"],
                        "val_recall": metrics["val_recall"],
                        "val_f1_class0": metrics["val_f1_class0"],
                        "val_precision_class0": metrics["val_precision_class0"],
                        "val_recall_class0": metrics["val_recall_class0"],
                        "val_f1_class1": metrics["val_f1_class1"],
                        "val_precision_class1": metrics["val_precision_class1"],
                        "val_recall_class1": metrics["val_recall_class1"],
                    }
                    if best_params:
                        result.update(best_params)
                    fold_results.append(result)
                    
                    logger.info(
                        f"Fold {fold_idx + 1} - Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}, "
                        f"Val F1: {metrics['val_f1']:.4f}, Val Precision: {metrics['val_precision']:.4f}, Val Recall: {metrics['val_recall']:.4f}"
                    )
                    
                    model.save(str(fold_output_dir))
                    cleanup_model_and_memory(model=model, clear_cuda=False)
                    aggressive_gc(clear_cuda=False)
                    
                else:
                    # Baseline model training with best hyperparameters
                    fold_output_dir = model_output_dir / f"fold_{fold_idx + 1}"
                    fold_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Note: Fold deletion already handled above before the resume check
                    
                    baseline_config = model_config.copy()
                    if best_params:
                        baseline_config.update(best_params)
                    
                    from lib.utils.paths import load_metadata_flexible
                    if model_type in BASELINE_MODELS:
                        stage2_df = load_metadata_flexible(features_stage2_path)
                        if stage2_df is not None and stage2_df.height > 0:
                            if "model_specific_config" not in baseline_config:
                                baseline_config["model_specific_config"] = {}
                            baseline_config["model_specific_config"]["features_stage2_path"] = features_stage2_path
                            baseline_config["features_stage2_path"] = features_stage2_path
                        else:
                            raise ValueError(f"Stage 2 features required for {model_type}")
                        
                        if model_type in STAGE4_MODELS:
                            stage4_df = load_metadata_flexible(features_stage4_path)
                            if stage4_df is not None and stage4_df.height > 0:
                                if "model_specific_config" not in baseline_config:
                                    baseline_config["model_specific_config"] = {}
                                baseline_config["model_specific_config"]["features_stage4_path"] = features_stage4_path
                                baseline_config["features_stage4_path"] = features_stage4_path
                            else:
                                raise ValueError(f"Stage 4 features required for {model_type}")
                        else:
                            if "model_specific_config" not in baseline_config:
                                baseline_config["model_specific_config"] = {}
                            baseline_config["model_specific_config"]["features_stage4_path"] = None
                            baseline_config["features_stage4_path"] = None
                    
                    model = create_model(model_type, baseline_config)
                    model.fit(train_df, project_root=project_root_str_orig)
                    val_probs = model.predict(val_df, project_root=project_root_str_orig)
                    val_preds = np.argmax(val_probs, axis=1)
                    val_labels = val_df["label"].to_list()
                    label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
                    val_y = np.array([label_map[label] for label in val_labels])
                    
                    # Compute comprehensive metrics using shared utility
                    metrics = compute_classification_metrics(
                        y_true=val_y,
                        y_pred=val_preds,
                        y_probs=val_probs
                    )
                    
                    result = {
                        "fold": fold_idx + 1,
                        "val_loss": metrics["val_loss"],
                        "val_acc": metrics["val_acc"],
                        "val_f1": metrics["val_f1"],
                        "val_precision": metrics["val_precision"],
                        "val_recall": metrics["val_recall"],
                        "val_f1_class0": metrics["val_f1_class0"],
                        "val_precision_class0": metrics["val_precision_class0"],
                        "val_recall_class0": metrics["val_recall_class0"],
                        "val_f1_class1": metrics["val_f1_class1"],
                        "val_precision_class1": metrics["val_precision_class1"],
                        "val_recall_class1": metrics["val_recall_class1"],
                    }
                    if best_params:
                        result.update(best_params)
                    fold_results.append(result)
                    
                    logger.info(
                        f"Fold {fold_idx + 1} - Val Loss: {metrics['val_loss']:.4f}, Val Acc: {metrics['val_acc']:.4f}, "
                        f"Val F1: {metrics['val_f1']:.4f}, Val Precision: {metrics['val_precision']:.4f}, Val Recall: {metrics['val_recall']:.4f}"
                    )
                    
                    model.save(str(fold_output_dir))
                    cleanup_model_and_memory(model=model, clear_cuda=False)
                    aggressive_gc(clear_cuda=False)
                    
            except Exception as e:
                logger.error(f"Error training final fold {fold_idx + 1}: {e}", exc_info=True)
                fold_results.append({
                    "fold": fold_idx + 1,
                    "val_loss": float('nan'),
                    "val_acc": float('nan'),
                    "val_f1": float('nan'),
                    "val_precision": float('nan'),
                    "val_recall": float('nan'),
                    "val_f1_class0": float('nan'),
                    "val_precision_class0": float('nan'),
                    "val_recall_class0": float('nan'),
                    "val_f1_class1": float('nan'),
                    "val_precision_class1": float('nan'),
                    "val_recall_class1": float('nan'),
                })
        
        # Save best model from final training
        if fold_results:
            best_fold = max(fold_results, key=lambda x: x.get("val_f1", 0) if isinstance(x.get("val_f1"), (int, float)) and not np.isnan(x.get("val_f1", 0)) else -1)
            best_fold_idx = best_fold.get("fold", 1)
            
            best_model_dir = model_output_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            
            best_fold_dir = model_output_dir / f"fold_{best_fold_idx}"
            if best_fold_dir.exists():
                try:
                    _copy_model_files(best_fold_dir, best_model_dir, f"fold {best_fold_idx}")
                except Exception as e:
                    logger.error(f"Failed to copy best model files: {e}")
        
        # Aggregate results (filter out NaN values) - use fold_results from final training
        if fold_results:
            valid_losses = [
                r.get("val_loss") for r in fold_results
                if "val_loss" in r and isinstance(r.get("val_loss"), (int, float))
                and not (isinstance(r.get("val_loss"), float)
                         and r.get("val_loss") != r.get("val_loss"))
            ]
            valid_accs = [
                r.get("val_acc") for r in fold_results
                if "val_acc" in r and isinstance(r.get("val_acc"), (int, float))
                and not (isinstance(r.get("val_acc"), float)
                         and r.get("val_acc") != r.get("val_acc"))
            ]
            valid_f1s = [
                r.get("val_f1") for r in fold_results
                if "val_f1" in r and isinstance(r.get("val_f1"), (int, float))
                and not (isinstance(r.get("val_f1"), float)
                         and r.get("val_f1") != r.get("val_f1"))
            ]
            
            avg_val_loss = (
                sum(valid_losses) / len(valid_losses)
                if valid_losses else float('nan')
            )
            avg_val_acc = (
                sum(valid_accs) / len(valid_accs)
                if valid_accs else float('nan')
            )
            avg_val_f1 = (
                sum(valid_f1s) / len(valid_f1s)
                if valid_f1s else float('nan')
            )
            
            results[model_type] = {
                "fold_results": fold_results,
                "avg_val_loss": avg_val_loss,
                "avg_val_acc": avg_val_acc,
                "avg_val_f1": avg_val_f1,
                "best_hyperparameters": best_params,
            }
            
            logger.info(
                "\n%s - Avg Val Loss: %.4f, Avg Val Acc: %.4f, Avg Val F1: %.4f",
                model_type, avg_val_loss, avg_val_acc, avg_val_f1
            )
            _flush_logs()
            
            # Generate visualization plots
            try:
                plots_dir = model_output_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate CV fold comparison
                from .visualization import plot_cv_fold_comparison, plot_hyperparameter_search
                
                plot_cv_fold_comparison(
                    fold_results,
                    plots_dir / "cv_fold_comparison.png",
                    title=f"{model_type} - Cross-Validation Results (Full Dataset)"
                )
                
                # Plot hyperparameter search results if grid search was performed
                if all_grid_results and len(all_grid_results) > 1:
                    plot_hyperparameter_search(
                        all_grid_results,
                        plots_dir / "hyperparameter_search.png",
                        title=f"{model_type} - Hyperparameter Search Results"
                    )
                
                logger.info(f"Generated plots for {model_type} in {plots_dir}")
                _flush_logs()
            except Exception as e:
                logger.warning(f"Failed to generate plots for {model_type}: {e}", exc_info=True)
        
        # Aggressive GC after all folds for this model type
        aggressive_gc(clear_cuda=False)
    
    # Train ensemble if requested
    if train_ensemble:
        logger.info("\n" + "="*80)
        logger.info("Training Ensemble Model")
        logger.info("="*80)
        _flush_logs()
        
        try:
            from .ensemble import train_ensemble_model
            
            ensemble_results = train_ensemble_model(
                project_root=project_root_str_orig,
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
            _flush_logs()
        except Exception as e:
            logger.error(f"Error training ensemble: {e}", exc_info=True)
            logger.warning("Continuing without ensemble results")
            _flush_logs()
    
    logger.info("=" * 80)
    logger.info("Stage 5: Model Training Pipeline Completed")
    logger.info("=" * 80)
    _flush_logs()
    
    return results

