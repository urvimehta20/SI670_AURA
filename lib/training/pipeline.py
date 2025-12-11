"""
Model training pipeline.

Trains models using scaled videos and extracted features.
Supports multiple model types and k-fold cross-validation.
"""

from __future__ import annotations

import logging
import sys
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader

from lib.data import stratified_kfold, load_metadata
# Lazy import to avoid circular dependency issues
# VideoConfig and VideoDataset will be imported when needed
from lib.mlops.config import ExperimentTracker, CheckpointManager
from lib.mlops.mlflow_tracker import create_mlflow_tracker, MLFLOW_AVAILABLE
from lib.training.trainer import OptimConfig, TrainConfig, fit
from lib.training.model_factory import create_model, is_pytorch_model, is_xgboost_model, get_model_config
from lib.training.feature_preprocessing import remove_collinear_features, load_and_combine_features
from lib.utils.memory import aggressive_gc

logger = logging.getLogger(__name__)


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
        models_init.write_text('''"""
Video models and datasets module (minimal stub).

This is a minimal stub created automatically.
For full functionality, ensure lib/models is properly synced to the server.
"""

from .video import VideoConfig, VideoDataset

__all__ = ["VideoConfig", "VideoDataset"]
''')
        logger.info(f"Created minimal lib/models/__init__.py at {models_init}")
    
    # Create minimal video.py if missing
    if not video_py.exists():
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
    
    # Models that require Stage 2 features (not the old logistic_regression/svm)
    stage2_models = {
        "logistic_regression_stage2",
        "logistic_regression_stage2_stage4",
        "svm_stage2",
        "svm_stage2_stage4"
    }
    
    # Models that require Stage 4 features
    stage4_models = {
        "logistic_regression_stage2_stage4",
        "svm_stage2_stage4"
    }
    
    # Old models (logistic_regression, svm without suffix) extract features from videos
    # They don't require Stage 2/4
    
    for model_type in model_types:
        missing = []
        
        # All models require Stage 3
        if not results["stage3_available"]:
            missing.append("Stage 3 (scaled videos)")
        
        # Stage 2 models require Stage 2
        if model_type in stage2_models and not results["stage2_available"]:
            missing.append("Stage 2 (features)")
        
        # Stage 2+4 models require Stage 4
        if model_type in stage4_models and not results["stage4_available"]:
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
    stage2_required_models = [m for m in model_types if m in stage2_models]
    if stage2_required_models and not results["stage2_available"]:
        failures_detected = True
        failure_lines.append("STAGE 2 FAILURE DETECTED")
        failure_lines.append("-" * 80)
        failure_lines.append(f"Expected: Stage 2 features metadata at: {features_stage2_path}")
        failure_lines.append(f"Status: NOT FOUND or EMPTY")
        failure_lines.append(f"Required for models: {', '.join(stage2_required_models)}")
        failure_lines.append("")
        failure_lines.append("ACTION REQUIRED:")
        failure_lines.append("  - Run Stage 2 feature extraction:")
        failure_lines.append("    sbatch src/scripts/slurm_stage2_features.sh")
        failure_lines.append("  - Or check if Stage 2 output exists at a different location")
        failure_lines.append("")
    
    # Check if Stage 4 was expected but missing
    stage4_required_models = [m for m in model_types if m in stage4_models]
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
    # Convert project_root to Path and resolve it once (avoid variable shadowing)
    project_root_path = Path(project_root).resolve()
    project_root_str = str(project_root_path)
    # Keep original string for backward compatibility in function calls
    project_root_str_orig = project_root_str
    
    # Ensure lib/models directory exists (create minimal stub if missing)
    _ensure_lib_models_exists(project_root_path)
    
    output_dir = project_root_path / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
        if any("stage2" in m for m in model_types) and not validation_results["stage2_available"]:
            error_msg += "  - Stage 2 (features): REQUIRED for *_stage2 models\n"
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
    
    # CRITICAL: Resource health check before proceeding
    monitor = ResourceMonitor()
    health = monitor.full_health_check(project_root_path)
    if health.status.value >= HealthCheckStatus.UNHEALTHY.value:
        logger.warning(f"System health check: {health.status.value} - {health.message}")
        if health.status == HealthCheckStatus.CRITICAL:
            raise ResourceExhaustedError(f"Critical system state: {health.message}")
    
    # Load Stage 2 and Stage 4 features (optional - only used for feature combination models)
    # These are loaded but may not be used for all model types
    features2_df = load_metadata_flexible(features_stage2_path)
    features4_df = load_metadata_flexible(features_stage4_path)
    
    # Note: features2_df and features4_df may be None if files don't exist
    # This is OK for models that don't use them (e.g., PyTorch models that use video data)
    if features2_df is None:
        logger.debug("Stage 2 features metadata not found (optional for some models)")
    if features4_df is None:
        logger.debug("Stage 4 features metadata not found (optional for some models)")
    
    logger.info(f"Stage 5: Found {scaled_df.height} scaled videos")
    
    # Create video config (only needed for PyTorch models)
    # Use fixed_size=256 to match Stage 3 output (videos scaled to max(width, height) = 256)
    # For non-PyTorch models (logistic_regression, svm), VideoConfig is not needed
    # So we'll use a fallback if import fails - only fail if we actually need it for PyTorch models
    
    # Define a minimal VideoConfig fallback class
    @dataclass
    class MinimalVideoConfig:
        """Minimal VideoConfig fallback when lib.models is not available."""
        num_frames: int = 16
        fixed_size: Optional[int] = None
        max_size: Optional[int] = None
        img_size: Optional[int] = None
        rolling_window: bool = False
        window_size: Optional[int] = None
        window_stride: Optional[int] = None
        augmentation_config: Optional[dict] = None
        temporal_augmentation_config: Optional[dict] = None
    
    # Try to import VideoConfig, but use fallback if it doesn't exist
    VideoConfig = None
    
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)
    
    # Try to import VideoConfig (non-critical for non-PyTorch models)
    try:
        from lib.models import VideoConfig
        logger.debug("Successfully imported VideoConfig from lib.models")
    except ImportError:
        try:
            # Try importlib as fallback
            models_init = project_root_path / 'lib' / 'models' / '__init__.py'
            if models_init.exists():
                spec = importlib.util.spec_from_file_location("lib.models", models_init)
                if spec and spec.loader:
                    models_module = importlib.util.module_from_spec(spec)
                    sys.modules['lib.models'] = models_module
                    spec.loader.exec_module(models_module)
                    VideoConfig = models_module.VideoConfig
                    logger.debug("Successfully imported VideoConfig using importlib")
        except Exception:
            pass
        
        if VideoConfig is None:
            try:
                video_py = project_root_path / 'lib' / 'models' / 'video.py'
                if video_py.exists():
                    spec = importlib.util.spec_from_file_location("lib.models.video", video_py)
                    if spec and spec.loader:
                        video_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(video_module)
                        VideoConfig = video_module.VideoConfig
                        logger.debug("Successfully imported VideoConfig from video.py")
            except Exception:
                pass
    
    # Use fallback if import failed
    if VideoConfig is None:
        logger.warning(
            "Could not import VideoConfig from lib.models. Using minimal fallback. "
            "This is OK for non-PyTorch models (logistic_regression, svm), "
            "but PyTorch models will fail if lib/models is missing. "
            "Please ensure lib/models directory exists on the server."
        )
        VideoConfig = MinimalVideoConfig
    
    # Create video config (will be used only for PyTorch models)
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
        
        # CRITICAL: Enforce 5-fold stratified cross-validation
        if n_splits != 5:
            logger.warning(f"n_splits={n_splits} specified, but enforcing 5-fold CV as required")
            n_splits = 5
        
        # CRITICAL FIX: Get all folds at once (stratified_kfold returns list of all folds)
        all_folds = stratified_kfold(
            scaled_df,
            n_splits=n_splits,
            random_state=42
        )
        
        if len(all_folds) != n_splits:
            raise ValueError(f"Expected {n_splits} folds, got {len(all_folds)}")
        
        logger.info(f"Using {n_splits}-fold stratified cross-validation")
        
        # Get hyperparameter grid for grid search
        from .grid_search import get_hyperparameter_grid, generate_parameter_combinations, select_best_hyperparameters
        from .visualization import generate_all_plots
        
        param_grid = get_hyperparameter_grid(model_type)
        param_combinations = generate_parameter_combinations(param_grid)
        
        logger.info(f"Grid search: {len(param_combinations)} hyperparameter combinations to try")
        
        # Store results for all hyperparameter combinations
        all_grid_results = []
        
        # Grid search: try each hyperparameter combination
        if not param_combinations:
            # No grid search, use default config
            param_combinations = [{}]
            logger.info("No hyperparameter grid defined, using default configuration")
        
        for param_idx, params in enumerate(param_combinations):
            logger.info(f"\n{'='*80}")
            logger.info(f"Grid Search: Hyperparameter combination {param_idx + 1}/{len(param_combinations)}")
            logger.info(f"Parameters: {params}")
            logger.info(f"{'='*80}")
            
            # Update model_config with current hyperparameters
            current_config = model_config.copy()
            current_config.update(params)
            
            # Store fold results for this parameter combination
            param_fold_results = []
            
            # Train all folds with this hyperparameter combination
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
                
                # Train model
                if is_pytorch_model(model_type):
                    # Create datasets for PyTorch models
                    # Lazy import to avoid circular dependency
                    # Ensure project root is in Python path for imports
                    # Note: sys and importlib.util are already imported at module level
                    # project_root_path is already resolved at function start
                    
                    if project_root_str not in sys.path:
                        sys.path.insert(0, project_root_str)
                    
                    # Try multiple import strategies with comprehensive error handling
                    VideoDataset = None
                    import_error = None
                    
                    # Strategy 1: Direct import (preferred)
                    try:
                        from lib.models import VideoDataset
                        logger.debug("Successfully imported VideoDataset using direct import")
                    except ImportError as e:
                        import_error = e
                        logger.debug(f"Direct import failed: {e}, trying alternative methods...")
                        
                        # Strategy 2: Import using importlib with explicit path
                        try:
                            models_init = project_root_path / 'lib' / 'models' / '__init__.py'
                            logger.debug(f"Attempting importlib import from: {models_init}")
                            if models_init.exists():
                                spec = importlib.util.spec_from_file_location("lib.models", models_init)
                                if spec and spec.loader:
                                    if 'lib.models' not in sys.modules:
                                        models_module = importlib.util.module_from_spec(spec)
                                        sys.modules['lib.models'] = models_module
                                        spec.loader.exec_module(models_module)
                                    VideoDataset = sys.modules['lib.models'].VideoDataset
                                    logger.debug("Successfully imported VideoDataset using importlib")
                            else:
                                logger.warning(f"lib/models/__init__.py not found at: {models_init}")
                        except Exception as e:
                            logger.debug(f"Importlib import failed: {e}")
                        
                        # Strategy 3: Try importing from video.py directly
                        if VideoDataset is None:
                            try:
                                video_py = project_root_path / 'lib' / 'models' / 'video.py'
                                logger.debug(f"Attempting direct import from video.py: {video_py}")
                                if video_py.exists():
                                    spec = importlib.util.spec_from_file_location("lib.models.video", video_py)
                                    if spec and spec.loader:
                                        video_module = importlib.util.module_from_spec(spec)
                                        spec.loader.exec_module(video_module)
                                        VideoDataset = video_module.VideoDataset
                                        logger.debug("Successfully imported VideoDataset from video.py")
                                else:
                                    logger.warning(f"lib/models/video.py not found at: {video_py}")
                            except Exception as e:
                                logger.debug(f"Direct video.py import failed: {e}")
                    
                    # Final check and error reporting
                    if VideoDataset is None:
                        # Comprehensive diagnostic information
                        models_dir = project_root_path / 'lib' / 'models'
                        models_init = models_dir / '__init__.py'
                        video_py = models_dir / 'video.py'
                        
                        diagnostics = {
                            "project_root": str(project_root_path),
                            "project_root_exists": project_root_path.exists(),
                            "lib_dir_exists": (project_root_path / 'lib').exists(),
                            "lib_models_dir_exists": models_dir.exists(),
                            "lib_models_init_exists": models_init.exists() if models_dir.exists() else False,
                            "lib_models_video_exists": video_py.exists() if models_dir.exists() else False,
                            "python_path_first_3": sys.path[:3],
                            "original_import_error": str(import_error) if import_error else None,
                        }
                        
                        error_msg = (
                            f"CRITICAL: Failed to import VideoDataset from lib.models after trying all strategies.\n"
                            f"Diagnostics: {diagnostics}\n"
                            f"Please ensure lib/models directory exists and contains __init__.py and video.py"
                        )
                        logger.error(error_msg)
                        raise ImportError(error_msg) from import_error
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
                    
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=current_config.get("batch_size", model_config.get("batch_size", 8)),
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=use_cuda,  # Faster GPU transfer
                        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
                        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch batches
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=current_config.get("batch_size", model_config.get("batch_size", 8)),
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=use_cuda,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                    )
                    # PyTorch model training
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = create_model(model_type, model_config)
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
                    
                    # Validate datasets before training
                    if len(train_dataset) == 0:
                        raise ValueError(f"Training dataset is empty for fold {fold_idx + 1}")
                    if len(val_dataset) == 0:
                        raise ValueError(f"Validation dataset is empty for fold {fold_idx + 1}")
                    
                    # Validate model initialization
                    try:
                        model.eval()
                        with torch.no_grad():
                            # Test forward pass with a sample batch
                            sample_batch = next(iter(train_loader))
                            sample_clips, sample_labels = sample_batch
                            sample_clips = sample_clips.to(device)
                            sample_output = model(sample_clips)
                            logger.info(f"Model forward pass test successful. Output shape: {sample_output.shape}")
                            del sample_batch, sample_clips, sample_labels, sample_output
                            if device.type == "cuda":
                                torch.cuda.empty_cache()
                    except Exception as e:
                        logger.error(f"Model forward pass test failed: {e}", exc_info=True)
                        raise ValueError(f"Model initialization failed: {e}") from e
                    
                    # Train with comprehensive error handling
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
                        val_metrics = evaluate(model, val_loader, device=str(device))
                        
                        val_loss = val_metrics["loss"]
                        val_acc = val_metrics["accuracy"]
                        val_f1 = val_metrics["f1"]
                        val_precision = val_metrics["precision"]
                        val_recall = val_metrics["recall"]
                        per_class = val_metrics["per_class"]
                    except RuntimeError as e:
                        # Catch CUDA OOM, invalid tensor operations, etc.
                        error_msg = str(e)
                        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower():
                            logger.error(
                                f"CUDA OOM or runtime error during training: {e}. "
                                f"Model: {model_type}, Fold: {fold_idx + 1}, "
                                f"Batch size: {current_config.get('batch_size', model_config.get('batch_size', 8))}"
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
                        
                        # Store results with hyperparameters
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
                        
                        val_acc = (val_preds == val_y).mean()
                        val_loss = -np.mean(np.log(val_probs[np.arange(len(val_y)), val_y] + 1e-10))  # Cross-entropy
                        
                        # Compute comprehensive metrics
                        from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
                        
                        val_precision = float(precision_score(val_y, val_preds, average='binary', zero_division=0))
                        val_recall = float(recall_score(val_y, val_preds, average='binary', zero_division=0))
                        val_f1 = float(f1_score(val_y, val_preds, average='binary', zero_division=0))
                        
                        # Per-class metrics
                        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
                            val_y, val_preds, average=None, zero_division=0
                        )
                        
                        # Store results with hyperparameters
                        result = {
                            "fold": fold_idx + 1,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_f1": val_f1,
                            "val_precision": val_precision,
                            "val_recall": val_recall,
                            "val_f1_class0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
                            "val_precision_class0": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
                            "val_recall_class0": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
                            "val_f1_class1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
                            "val_precision_class1": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
                            "val_recall_class1": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
                        }
                        result.update(params)  # Add hyperparameters
                        param_fold_results.append(result)
                        fold_results.append(result)
                        
                        logger.info(
                            f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                            f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
                        )
                        if len(f1_per_class) > 0:
                            for class_idx in range(len(f1_per_class)):
                                logger.info(
                                    f"  Class {class_idx} - Precision: {precision_per_class[class_idx]:.4f}, "
                                    f"Recall: {recall_per_class[class_idx]:.4f}, F1: {f1_per_class[class_idx]:.4f}"
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
                        }
                        result.update(params)
                        param_fold_results.append(result)
                        fold_results.append(result)
                    
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
                        # Create baseline model with hyperparameters
                        baseline_config = model_config.copy()
                        baseline_config.update(params)  # Apply grid search hyperparameters
                        
                        # Add feature paths for baseline models (all logistic_regression and svm variants)
                        # Check if metadata files exist and are not empty before passing paths
                        from lib.utils.paths import load_metadata_flexible
                        
                        if model_type in ["logistic_regression", "logistic_regression_stage2", "logistic_regression_stage2_stage4",
                                         "svm", "svm_stage2", "svm_stage2_stage4"]:
                            # Check if Stage 2 metadata exists and is not empty
                            stage2_df = load_metadata_flexible(features_stage2_path)
                            if stage2_df is not None and stage2_df.height > 0:
                                baseline_config["features_stage2_path"] = features_stage2_path
                                logger.debug(f"Passing Stage 2 features path to {model_type}: {features_stage2_path}")
                            else:
                                baseline_config["features_stage2_path"] = None
                                logger.warning(f"Stage 2 metadata not available for {model_type}, will extract features from videos")
                            
                            # For models that use Stage 4, check if it exists
                            if model_type in ["logistic_regression_stage2_stage4", "svm_stage2_stage4"]:
                                stage4_df = load_metadata_flexible(features_stage4_path)
                                if stage4_df is not None and stage4_df.height > 0:
                                    baseline_config["features_stage4_path"] = features_stage4_path
                                    logger.debug(f"Passing Stage 4 features path to {model_type}: {features_stage4_path}")
                                else:
                                    baseline_config["features_stage4_path"] = None
                                    logger.warning(f"Stage 4 metadata not available for {model_type}")
                            else:
                                # For stage2_only models, explicitly set stage4_path to None
                                baseline_config["features_stage4_path"] = None
                        
                        model = create_model(model_type, baseline_config)
                        
                        # Train baseline (handles feature extraction internally)
                        model.fit(train_df, project_root=project_root_str_orig)
                        
                        # Evaluate on validation set
                        val_probs = model.predict(val_df, project_root=project_root_str_orig)
                        val_preds = np.argmax(val_probs, axis=1)
                        val_labels = val_df["label"].to_list()
                        label_map = {label: idx for idx, label in enumerate(sorted(set(val_labels)))}
                        val_y = np.array([label_map[label] for label in val_labels])
                        
                        val_acc = (val_preds == val_y).mean()
                        val_loss = -np.mean(np.log(val_probs[np.arange(len(val_y)), val_y] + 1e-10))  # Cross-entropy
                        
                        # Compute comprehensive metrics
                        from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
                        
                        val_precision = float(precision_score(val_y, val_preds, average='binary', zero_division=0))
                        val_recall = float(recall_score(val_y, val_preds, average='binary', zero_division=0))
                        val_f1 = float(f1_score(val_y, val_preds, average='binary', zero_division=0))
                        
                        # Per-class metrics
                        precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
                            val_y, val_preds, average=None, zero_division=0
                        )
                        
                        # Store results with hyperparameters
                        result = {
                            "fold": fold_idx + 1,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "val_f1": val_f1,
                            "val_precision": val_precision,
                            "val_recall": val_recall,
                            "val_f1_class0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
                            "val_precision_class0": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
                            "val_recall_class0": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
                            "val_f1_class1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
                            "val_precision_class1": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
                            "val_recall_class1": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
                        }
                        result.update(params)  # Add hyperparameters
                        param_fold_results.append(result)
                        fold_results.append(result)
                        
                        logger.info(
                            f"Fold {fold_idx + 1} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                            f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
                        )
                        if len(f1_per_class) > 0:
                            for class_idx in range(len(f1_per_class)):
                                logger.info(
                                    f"  Class {class_idx} - Precision: {precision_per_class[class_idx]:.4f}, "
                                    f"Recall: {recall_per_class[class_idx]:.4f}, F1: {f1_per_class[class_idx]:.4f}"
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
                        }
                        result.update(params)
                        param_fold_results.append(result)
                        fold_results.append(result)
                    
                    # Clear model and aggressively free memory
                    if 'model' in locals():
                        del model
                    aggressive_gc(clear_cuda=False)
            
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
        
        # Select best hyperparameters from grid search
        best_params = None
        best_fold_results = fold_results  # Default to all fold results
        if param_combinations and all_grid_results and len(all_grid_results) > 1:
            best_params = select_best_hyperparameters(model_type, all_grid_results)
            logger.info(f"Best hyperparameters selected: {best_params}")
            
            # Find the best parameter combination's fold results
            best_grid_result = max(all_grid_results, key=lambda x: x.get("mean_f1", 0))
            best_fold_results = best_grid_result.get("fold_results", fold_results)
            
            # Save best model - find the best fold from best hyperparameter combination
            if best_fold_results:
                best_fold = max(best_fold_results, key=lambda x: x.get("val_f1", 0))
                best_fold_idx = best_fold.get("fold", 1)
                
                # Copy best model to best_model directory
                best_model_dir = model_output_dir / "best_model"
                best_model_dir.mkdir(parents=True, exist_ok=True)
                
                best_fold_dir = model_output_dir / f"fold_{best_fold_idx}"
                if best_fold_dir.exists():
                    import shutil
                    # Copy model files
                    for model_file in best_fold_dir.glob("*.pt"):
                        shutil.copy2(model_file, best_model_dir / model_file.name)
                    for model_file in best_fold_dir.glob("*.joblib"):
                        shutil.copy2(model_file, best_model_dir / model_file.name)
                    for model_file in best_fold_dir.glob("*.json"):
                        shutil.copy2(model_file, best_model_dir / model_file.name)
                    logger.info(f"Saved best model from fold {best_fold_idx} to {best_model_dir}")
        elif fold_results:
            # No grid search or only one combination - use best fold from all results
            best_fold = max(fold_results, key=lambda x: x.get("val_f1", 0) if isinstance(x.get("val_f1"), (int, float)) and not np.isnan(x.get("val_f1", 0)) else -1)
            best_fold_idx = best_fold.get("fold", 1)
            
            # Copy best model to best_model directory
            best_model_dir = model_output_dir / "best_model"
            best_model_dir.mkdir(parents=True, exist_ok=True)
            
            best_fold_dir = model_output_dir / f"fold_{best_fold_idx}"
            if best_fold_dir.exists():
                import shutil
                # Copy model files
                for model_file in best_fold_dir.glob("*.pt"):
                    shutil.copy2(model_file, best_model_dir / model_file.name)
                for model_file in best_fold_dir.glob("*.joblib"):
                    shutil.copy2(model_file, best_model_dir / model_file.name)
                for model_file in best_fold_dir.glob("*.json"):
                    shutil.copy2(model_file, best_model_dir / model_file.name)
                logger.info(f"Saved best model from fold {best_fold_idx} to {best_model_dir}")
        
        # Aggregate results (filter out NaN values) - use best fold results if available
        if best_fold_results:
            valid_losses = [
                r["val_loss"] for r in best_fold_results
                if isinstance(r["val_loss"], (int, float))
                and not (isinstance(r["val_loss"], float)
                         and r["val_loss"] != r["val_loss"])
            ]
            valid_accs = [
                r["val_acc"] for r in best_fold_results
                if isinstance(r["val_acc"], (int, float))
                and not (isinstance(r["val_acc"], float)
                         and r["val_acc"] != r["val_acc"])
            ]
            valid_f1s = [
                r["val_f1"] for r in best_fold_results
                if isinstance(r["val_f1"], (int, float))
                and not (isinstance(r["val_f1"], float)
                         and r["val_f1"] != r["val_f1"])
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
                "fold_results": best_fold_results,
                "avg_val_loss": avg_val_loss,
                "avg_val_acc": avg_val_acc,
                "avg_val_f1": avg_val_f1,
                "best_hyperparameters": best_params,
            }
            
            logger.info(
                "\n%s - Avg Val Loss: %.4f, Avg Val Acc: %.4f, Avg Val F1: %.4f",
                model_type, avg_val_loss, avg_val_acc, avg_val_f1
            )
            
            # Generate visualization plots
            try:
                plots_dir = model_output_dir / "plots"
                plots_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate CV fold comparison
                from .visualization import plot_cv_fold_comparison, plot_hyperparameter_search
                
                plot_cv_fold_comparison(
                    best_fold_results,
                    plots_dir / "cv_fold_comparison.png",
                    title=f"{model_type} - Cross-Validation Results"
                )
                
                # Plot hyperparameter search results if grid search was performed
                if all_grid_results and len(all_grid_results) > 1:
                    plot_hyperparameter_search(
                        all_grid_results,
                        plots_dir / "hyperparameter_search.png",
                        title=f"{model_type} - Hyperparameter Search Results"
                    )
                
                logger.info(f"Generated plots for {model_type} in {plots_dir}")
            except Exception as e:
                logger.warning(f"Failed to generate plots for {model_type}: {e}", exc_info=True)
        
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
        except Exception as e:
            logger.error(f"Error training ensemble: {e}", exc_info=True)
            logger.warning("Continuing without ensemble results")
    
    return results

