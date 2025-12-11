"""
Stage 5: Unified training pipeline.

Routes models to appropriate pipeline:
- Baseline models (logistic_regression, svm) -> Feature-based pipeline
- All other models -> Video-based pipeline (process video frames)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from lib.training.feature_training_pipeline import train_feature_model, load_features_for_training
from lib.training.video_training_pipeline import (
    train_video_model, is_feature_based, is_video_based
)
from lib.utils.paths import load_metadata_flexible
from lib.data import stratified_kfold
from lib.utils.video_validation import filter_valid_videos

logger = logging.getLogger(__name__)


def stage5_train_all_models(
    project_root: str,
    scaled_metadata_path: str,
    features_stage2_path: Optional[str],
    features_stage4_path: Optional[str],
    model_types: List[str],
    n_splits: int = 5,
    output_dir: str = "data/stage5",
    use_gpu: bool = True,
    batch_size: int = 32,
    epochs: int = 100,
    num_frames: int = 1000
) -> Dict[str, Dict]:
    """
    Train all models using Stage 2/4 features.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled video metadata (for labels)
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        model_types: List of model types to train
        n_splits: Number of CV folds
        output_dir: Output directory
        use_gpu: Use GPU if available
        batch_size: Batch size
        epochs: Max epochs
    
    Returns:
        Dictionary mapping model_type -> results
    """
    project_root_path = Path(project_root)
    output_dir_path = project_root_path / output_dir
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Load scaled metadata for labels
    logger.info("Loading scaled metadata...")
    scaled_df = load_metadata_flexible(scaled_metadata_path)
    if scaled_df is None or scaled_df.height == 0:
        raise ValueError(f"Scaled metadata not found or empty: {scaled_metadata_path}")
    
    # Validate we have enough data
    if scaled_df.height <= 3000:
        raise ValueError(f"Insufficient data: {scaled_df.height} rows (need > 3000)")
    
    logger.info(f"Loaded {scaled_df.height} videos")
    
    # Get video paths and labels
    video_paths = scaled_df["video_path"].to_list()
    labels = scaled_df["label"].to_list()
    
    # Convert labels to binary (0/1)
    unique_labels = sorted(set(labels))
    if len(unique_labels) != 2:
        raise ValueError(f"Expected binary classification, got {len(unique_labels)} classes: {unique_labels}")
    
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels_array = np.array([label_map[label] for label in labels])
    
    logger.info(f"Labels: {unique_labels} -> {label_map}")
    
    # Load features (collinearity removal happens here, BEFORE any splits)
    logger.info("Loading features from Stage 2/4 (collinearity removal will be done once before splits)...")
    if not features_stage2_path:
        raise ValueError(
            "features_stage2_path is required for feature-based models. "
            "Features should already be extracted in Stage 2. "
            "Please provide the path to Stage 2 features metadata."
        )
    
    try:
        features, feature_names, valid_indices = load_features_for_training(
            features_stage2_path,
            features_stage4_path,
            video_paths,
            project_root
        )
        logger.info(f"✓ Loaded {len(feature_names)} features for {len(features)} videos")
        logger.info(f"  Features already cleaned (collinearity removed before splits)")
    except Exception as e:
        logger.error(f"✗ Failed to load features: {e}", exc_info=True)
        logger.error(
            "Make sure Stage 2 and Stage 4 features are already extracted. "
            "Do NOT re-extract features during training."
        )
        raise
    
    # Filter to valid indices (videos that have features)
    if valid_indices is not None:
        original_count = len(features)
        features = features[valid_indices]
        labels_array = labels_array[valid_indices]
        video_paths = [video_paths[i] for i in valid_indices]
        logger.info(f"After filtering to videos with valid features: {len(features)}/{original_count} videos")
    
    # CRITICAL: Ensure we have at least 3000 videos with valid features
    if len(features) <= 3000:
        error_msg = (
            f"Insufficient valid videos for training. "
            f"Found {len(features)} videos with valid features, but need more than 3000. "
            f"Original dataset had {scaled_df.height} videos. "
            f"Some videos may be missing features or corrupted."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"✓ Sufficient videos with valid features: {len(features)} > 3000")
    
    # Separate models by type
    feature_models = [m for m in model_types if is_feature_based(m)]
    video_models = [m for m in model_types if is_video_based(m)]
    unknown_models = [m for m in model_types if not (is_feature_based(m) or is_video_based(m))]
    
    if unknown_models:
        logger.warning(f"Unknown model types: {unknown_models}, skipping")
    
    logger.info(f"Feature-based models: {feature_models}")
    logger.info(f"Video-based models: {video_models}")
    
    all_results = {}
    
    # Train feature-based models
    if feature_models:
        logger.info("=" * 80)
        logger.info("TRAINING FEATURE-BASED MODELS")
        logger.info("=" * 80)
        
        for model_type in feature_models:
            logger.info("=" * 80)
            logger.info(f"Training {model_type} (feature-based)")
            logger.info("=" * 80)
            
            try:
                model_output_dir = output_dir_path / model_type
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                results = train_feature_model(
                    model_type=model_type,
                    features=features,
                    labels=labels_array,
                    feature_names=feature_names,
                    output_dir=model_output_dir,
                    n_splits=n_splits,
                    batch_size=batch_size,
                    epochs=epochs,
                    use_gpu=use_gpu
                )
                
                all_results[model_type] = results
                logger.info(f"✓ Completed {model_type}")
                
            except Exception as e:
                logger.error(f"✗ Failed to train {model_type}: {e}", exc_info=True)
                all_results[model_type] = {"error": str(e)}
    
    # Train video-based models
    if video_models:
        logger.info("=" * 80)
        logger.info("TRAINING VIDEO-BASED MODELS")
        logger.info("=" * 80)
        
        # CRITICAL: Validate videos and filter corrupted ones BEFORE training
        logger.info("Validating videos and checking for corrupted files...")
        logger.info("This may take a few minutes...")
        
        try:
            # Filter to only valid videos (checks for corruption, missing files, etc.)
            # This ensures we have at least 3000 valid videos
            scaled_df_valid, invalid_videos = filter_valid_videos(
                scaled_df,
                project_root=project_root,
                min_valid_videos=3000,
                check_all=False,  # Use sampling for speed, then check all if needed
                sample_rate=0.1  # Check 10% first to estimate validity rate
            )
            
            logger.info(f"✓ Video validation complete: {scaled_df_valid.height} valid videos")
            
            if invalid_videos:
                # Save invalid videos report
                invalid_report_path = output_dir_path / "invalid_videos_report.txt"
                with open(invalid_report_path, "w") as f:
                    f.write("Invalid/Corrupted Videos Report\n")
                    f.write("=" * 80 + "\n\n")
                    for invalid in invalid_videos:
                        f.write(f"{invalid}\n")
                logger.info(f"Invalid videos report saved to: {invalid_report_path}")
            
            # Update scaled_df to only valid videos
            scaled_df = scaled_df_valid
            
            # Re-extract labels for valid videos only
            video_paths = scaled_df["video_path"].to_list()
            labels = scaled_df["label"].to_list()
            label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
            labels_array = np.array([label_map[label] for label in labels])
            
            logger.info(f"After video validation: {len(scaled_df)} valid videos")
            
        except ValueError as e:
            # Not enough valid videos
            logger.error(f"✗ Video validation failed: {e}")
            logger.error("Training cannot proceed. Please fix corrupted videos or add more valid videos.")
            raise
        
        # Create 60-20-20 split for video models
        from lib.training.feature_pipeline import create_stratified_splits
        indices = np.arange(len(scaled_df))
        train_idx, val_idx, test_idx = create_stratified_splits(
            indices,
            labels_array,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        train_df = scaled_df[train_idx]
        val_df = scaled_df[val_idx]
        test_df = scaled_df[test_idx]
        
        logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        for model_type in video_models:
            logger.info("=" * 80)
            logger.info(f"Training {model_type} (video-based)")
            logger.info("=" * 80)
            
            try:
                model_output_dir = output_dir_path / model_type
                model_output_dir.mkdir(parents=True, exist_ok=True)
                
                results = train_video_model(
                    model_type=model_type,
                    train_df=train_df,
                    val_df=val_df,
                    test_df=test_df,
                    project_root=project_root,
                    output_dir=model_output_dir,
                    n_splits=n_splits,
                    num_frames=num_frames,
                    batch_size=batch_size,
                    epochs=epochs,
                    use_gpu=use_gpu
                )
                
                all_results[model_type] = results
                logger.info(f"✓ Completed {model_type}")
                
            except Exception as e:
                logger.error(f"✗ Failed to train {model_type}: {e}", exc_info=True)
                all_results[model_type] = {"error": str(e)}
    
    logger.info("=" * 80)
    logger.info("STAGE 5 TRAINING COMPLETE")
    logger.info("=" * 80)
    
    # Summary
    successful = [m for m, r in all_results.items() if "error" not in r]
    failed = [m for m, r in all_results.items() if "error" in r]
    
    logger.info(f"Successful: {len(successful)}/{len(model_types)}")
    logger.info(f"Failed: {len(failed)}/{len(model_types)}")
    
    if successful:
        logger.info("Successful models:")
        for model_type in successful:
            results = all_results[model_type]
            if "test_results" in results:
                test_f1 = results["test_results"]["f1"]
                test_auc = results["test_results"]["auc"]
                logger.info(f"  {model_type}: F1={test_f1:.4f}, AUC={test_auc:.4f}")
    
    if failed:
        logger.warning("Failed models:")
        for model_type in failed:
            logger.warning(f"  {model_type}: {all_results[model_type]['error']}")
    
    return all_results

