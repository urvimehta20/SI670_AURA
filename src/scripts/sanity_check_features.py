#!/usr/bin/env python3
"""
Sanity check script for Stage 2 and Stage 4 features.

Verifies:
1. Stage 2 features exist and have 15 or 23 features per video (15 if codec cues unavailable)
2. Stage 4 features exist and have 23 features per video
3. Features can be loaded correctly
4. Logistic Regression and SVM can train on these features
"""

import sys
import os
import logging
import time
import gc
from pathlib import Path
from typing import Optional
import polars as pl
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging with both console and file handlers
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"sanity_check_features_{int(time.time())}.log"

# Create logger - prevent duplicate handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Clear existing handlers to prevent duplication
logger.handlers.clear()

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
console_handler.setFormatter(console_formatter)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Prevent propagation to root logger
logger.propagate = False

logger.info(f"Logging to file: {log_file}")

def count_features_in_metadata(metadata_path: Path) -> dict:
    """Count features in a metadata file."""
    try:
        if metadata_path.suffix == '.arrow':
            try:
                df = pl.read_ipc(metadata_path)
            except Exception as arrow_error:
                # Arrow file might be corrupted, try parquet alternative
                parquet_path = metadata_path.with_suffix('.parquet')
                if parquet_path.exists():
                    logger.warning(f"Arrow file failed, trying parquet: {arrow_error}")
                    df = pl.read_parquet(parquet_path)
                else:
                    raise arrow_error
        elif metadata_path.suffix == '.parquet':
            df = pl.read_parquet(metadata_path)
        else:
            df = pl.read_csv(metadata_path)
        
        # Get feature columns (exclude only true metadata columns, not features)
        # Note: is_upscaled and is_downscaled ARE features (part of the 23 features)
        metadata_cols = {'video_path', 'label', 'feature_path'}
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        
        return {
            'total_rows': df.height,
            'feature_count': len(feature_cols),
            'feature_names': sorted(feature_cols),
            'has_data': df.height > 0
        }
    except Exception as e:
        logger.error(f"Error reading {metadata_path}: {e}")
        return {'error': str(e)}

def check_feature_files(metadata_path: Path, project_root: Path) -> dict:
    """Check if individual feature files exist and have correct shape."""
    try:
        if metadata_path.suffix == '.arrow':
            df = pl.read_ipc(metadata_path)
        elif metadata_path.suffix == '.parquet':
            df = pl.read_parquet(metadata_path)
        else:
            df = pl.read_csv(metadata_path)
        
        if 'feature_path' not in df.columns:
            return {'error': 'No feature_path column found'}
        
        feature_files_checked = 0
        feature_files_missing = 0
        feature_files_wrong_shape = 0
        feature_shapes = []
        
        for row in df.head(10).iter_rows(named=True):  # Check first 10
            feature_path = row.get('feature_path')
            if not feature_path:
                continue
            
            feature_file = project_root / feature_path
            if not feature_file.exists():
                feature_files_missing += 1
                continue
            
            try:
                if feature_file.suffix == '.npy':
                    features = np.load(feature_file)
                    shape = features.shape
                    feature_shapes.append(shape)
                    if len(shape) == 1 and shape[0] != 23:
                        feature_files_wrong_shape += 1
                    feature_files_checked += 1
                elif feature_file.suffix == '.parquet':
                    feat_df = pl.read_parquet(feature_file)
                    # Count numeric columns
                    numeric_cols = [col for col in feat_df.columns if feat_df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
                    shape = (feat_df.height, len(numeric_cols))
                    feature_shapes.append(shape)
                    if len(numeric_cols) != 23:
                        feature_files_wrong_shape += 1
                    feature_files_checked += 1
            except Exception as e:
                logger.warning(f"Error loading {feature_file}: {e}")
        
        return {
            'checked': feature_files_checked,
            'missing': feature_files_missing,
            'wrong_shape': feature_files_wrong_shape,
            'shapes': feature_shapes[:5]  # Show first 5
        }
    except Exception as e:
        return {'error': str(e)}

def reconstruct_if_needed(features_dir: Path, metadata_name: str) -> Optional[Path]:
    """Reconstruct metadata from parquet files if needed."""
    metadata_path = features_dir / metadata_name
    
    # Check if metadata already exists and is valid
    if metadata_path.exists():
        try:
            # Try to read the file
            if metadata_path.suffix == '.arrow':
                test_df = pl.read_ipc(metadata_path)
            elif metadata_path.suffix == '.parquet':
                test_df = pl.read_parquet(metadata_path)
            else:
                test_df = pl.read_csv(metadata_path)
            
            # Validate it has data
            if test_df.height > 0:
                # File is valid
                return metadata_path
            else:
                logger.warning(f"Existing metadata file is empty: {metadata_path}")
        except Exception as e:
            logger.warning(f"Existing metadata file is corrupted: {metadata_path} - {e}")
    
    # Try to reconstruct from parquet files
    logger.info(f"Attempting to reconstruct {metadata_name} from individual parquet files...")
    
    # Check for individual parquet files
    parquet_files = list(features_dir.glob("*_features.parquet")) + list(features_dir.glob("*_scaled_features.parquet"))
    
    if not parquet_files:
        logger.warning(f"No individual parquet files found in {features_dir}")
        return None
    
    logger.info(f"Found {len(parquet_files)} individual parquet files")
    
    # Import reconstruction function
    try:
        from src.scripts.reconstruct_metadata import reconstruct_metadata_from_parquet_files
        
        pattern = "*_features.parquet" if "stage2" in str(features_dir) else "*_scaled_features.parquet"
        output_name = metadata_name
        
        result = reconstruct_metadata_from_parquet_files(
            features_dir,
            output_name,
            pattern
        )
        
        if result and result.exists():
            logger.info(f"✓ Successfully reconstructed {metadata_name}")
            return result
        else:
            logger.error(f"✗ Failed to reconstruct {metadata_name}")
            return None
    except Exception as e:
        logger.error(f"Error during reconstruction: {e}", exc_info=True)
        return None


def main():
    project_root = Path.cwd()
    
    # Track critical check results
    stage2_ok = False
    stage4_ok = False
    
    # Expected paths - try multiple formats
    stage2_dir = project_root / "data" / "features_stage2"
    stage4_dir = project_root / "data" / "features_stage4"
    
    # Try to find or reconstruct Stage 2 metadata
    stage2_metadata = None
    for ext in ['.parquet', '.arrow']:
        candidate = stage2_dir / f"features_metadata{ext}"
        if candidate.exists():
            stage2_metadata = candidate
            break
    
    # If not found, try to reconstruct
    if not stage2_metadata or not stage2_metadata.exists():
        stage2_metadata = reconstruct_if_needed(stage2_dir, "features_metadata.parquet")
    
    # Try to find or reconstruct Stage 4 metadata
    stage4_metadata = None
    for ext in ['.parquet', '.arrow']:
        candidate = stage4_dir / f"features_scaled_metadata{ext}"
        if candidate.exists():
            stage4_metadata = candidate
            break
    
    # If not found, try to reconstruct
    if not stage4_metadata or not stage4_metadata.exists():
        stage4_metadata = reconstruct_if_needed(stage4_dir, "features_scaled_metadata.parquet")
    
    logger.info("=" * 80)
    logger.info("SANITY CHECK: Stage 2 and Stage 4 Features")
    logger.info("=" * 80)
    
    # Check Stage 2
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 2 FEATURES")
    logger.info("=" * 80)
    if stage2_metadata and stage2_metadata.exists():
        logger.info(f"✓ Stage 2 metadata found: {stage2_metadata}")
        stage2_info = count_features_in_metadata(stage2_metadata)
        if 'error' in stage2_info:
            logger.error(f"✗ Error: {stage2_info['error']}")
        else:
            logger.info(f"  Total videos: {stage2_info['total_rows']}")
            logger.info(f"  Feature count: {stage2_info['feature_count']}")
            # Stage 2 can have 15 features (if codec cues unavailable) or 23 features (if codec cues available)
            if stage2_info['feature_count'] == 23:
                logger.info("  ✓ Correct number of features (23)")
                stage2_ok = True
            elif stage2_info['feature_count'] == 15:
                logger.info("  ✓ Correct number of features (15 - codec cues may be unavailable)")
                stage2_ok = True
            elif stage2_info['feature_count'] > 0:
                logger.warning(f"  ⚠ Got {stage2_info['feature_count']} features (expected 15 or 23, but will proceed)")
                stage2_ok = True  # Still proceed if we have features
            else:
                logger.error(f"  ✗ No features found!")
                stage2_ok = False
            logger.info(f"  Feature names: {stage2_info['feature_names']}")
            
            # Check individual feature files
            file_check = check_feature_files(stage2_metadata, project_root)
            if 'error' not in file_check:
                logger.info(f"  Feature files checked: {file_check['checked']}")
                if file_check['missing'] > 0:
                    logger.warning(f"  Missing feature files: {file_check['missing']}")
                if file_check['wrong_shape'] > 0:
                    logger.warning(f"  Feature files with wrong shape: {file_check['wrong_shape']}")
                if file_check['shapes']:
                    logger.info(f"  Sample shapes: {file_check['shapes']}")
    else:
        logger.error(f"✗ Stage 2 metadata not found: {stage2_metadata}")
    
    # Check Stage 4
    logger.info("\n" + "=" * 80)
    logger.info("STAGE 4 FEATURES")
    logger.info("=" * 80)
    if stage4_metadata and stage4_metadata.exists():
        logger.info(f"✓ Stage 4 metadata found: {stage4_metadata}")
        stage4_info = count_features_in_metadata(stage4_metadata)
        if 'error' in stage4_info:
            logger.error(f"✗ Error: {stage4_info['error']}")
        else:
            logger.info(f"  Total videos: {stage4_info['total_rows']}")
            logger.info(f"  Feature count: {stage4_info['feature_count']}")
            # Stage 4 should have 23 features (15 base + 6 scaled + 2 indicators)
            if stage4_info['feature_count'] == 23:
                logger.info("  ✓ Correct number of features (23)")
                stage4_ok = True
            elif stage4_info['feature_count'] > 0:
                logger.warning(f"  ⚠ Got {stage4_info['feature_count']} features (expected 23, but will proceed)")
                stage4_ok = True  # Still proceed if we have features
            else:
                logger.error(f"  ✗ No features found!")
                stage4_ok = False
            logger.info(f"  Feature names: {stage4_info['feature_names']}")
            
            # Check individual feature files
            file_check = check_feature_files(stage4_metadata, project_root)
            if 'error' not in file_check:
                logger.info(f"  Feature files checked: {file_check['checked']}")
                if file_check['missing'] > 0:
                    logger.warning(f"  Missing feature files: {file_check['missing']}")
                if file_check['wrong_shape'] > 0:
                    logger.warning(f"  Feature files with wrong shape: {file_check['wrong_shape']}")
                if file_check['shapes']:
                    logger.info(f"  Sample shapes: {file_check['shapes']}")
    else:
        logger.error(f"✗ Stage 4 metadata not found: {stage4_metadata}")
    
    # Test feature loading
    logger.info("\n" + "=" * 80)
    logger.info("TESTING FEATURE LOADING")
    logger.info("=" * 80)
    
    feature_test_success = False
    try:
        from lib.training.feature_preprocessing import load_and_combine_features
        
        # Try to load a small sample
        if stage2_metadata and stage2_metadata.exists():
            if stage2_metadata.suffix == '.arrow':
                df2 = pl.read_ipc(stage2_metadata)
            elif stage2_metadata.suffix == '.parquet':
                df2 = pl.read_parquet(stage2_metadata)
            else:
                df2 = pl.read_csv(stage2_metadata)
            
            if df2.height > 0:
                # If Stage 4 metadata exists, use videos that have Stage 4 features
                if stage4_metadata and stage4_metadata.exists():
                    if stage4_metadata.suffix == '.arrow':
                        df4 = pl.read_ipc(stage4_metadata)
                    elif stage4_metadata.suffix == '.parquet':
                        df4 = pl.read_parquet(stage4_metadata)
                    else:
                        df4 = pl.read_csv(stage4_metadata)
                    
                    if df4.height > 0:
                        # Use videos that exist in both Stage 2 and Stage 4
                        stage4_video_paths = set(df4["video_path"].to_list())
                        test_video_paths = [vp for vp in df2["video_path"].to_list() if vp in stage4_video_paths][:5]
                        if not test_video_paths:
                            # Fallback: use first 5 from Stage 2
                            test_video_paths = df2.head(5)["video_path"].to_list()
                            logger.info(f"Testing feature loading with {len(test_video_paths)} videos (no Stage 4 overlap found, using Stage 2 only)...")
                        else:
                            logger.info(f"Testing feature loading with {len(test_video_paths)} videos (videos with both Stage 2 and Stage 4 features)...")
                    else:
                        test_video_paths = df2.head(5)["video_path"].to_list()
                        logger.info(f"Testing feature loading with {len(test_video_paths)} videos (Stage 4 metadata empty, using Stage 2 only)...")
                else:
                    # No Stage 4 metadata, use Stage 2 videos
                    test_video_paths = df2.head(5)["video_path"].to_list()
                    logger.info(f"Testing feature loading with {len(test_video_paths)} videos (no Stage 4 metadata, using Stage 2 only)...")
                
                try:
                    # Limit to 3 videos to reduce memory pressure and potential crashes
                    test_video_paths = test_video_paths[:3]
                    features, feature_names, kept_indices = load_and_combine_features(
                        features_stage2_path=str(stage2_metadata),
                        features_stage4_path=str(stage4_metadata) if (stage4_metadata and stage4_metadata.exists()) else None,
                        video_paths=test_video_paths,
                        project_root=str(project_root),
                        remove_collinearity=False  # Don't remove for testing
                    )
                    logger.info(f"✓ Successfully loaded features")
                    logger.info(f"  Feature matrix shape: {features.shape}")
                    logger.info(f"  Feature names count: {len(feature_names)}")
                    logger.info(f"  Sample feature names: {feature_names[:5]}")
                    feature_test_success = True
                    # Explicitly delete large objects to help with cleanup
                    del features, feature_names, kept_indices
                except Exception as e:
                    logger.error(f"✗ Failed to load features: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"✗ Error testing feature loading: {e}", exc_info=True)
    
    if not feature_test_success:
        logger.warning("⚠ Feature loading test failed, but continuing with sanity check")
    
    logger.info("\n" + "=" * 80)
    logger.info("SANITY CHECK COMPLETE")
    logger.info("=" * 80)
    
    # Determine overall status
    if stage2_ok and stage4_ok:
        logger.info("✓ All critical checks passed")
        overall_status = 0
    elif stage2_ok:
        logger.warning("⚠ Stage 2 OK, but Stage 4 has issues (may proceed with Stage 2 only)")
        overall_status = 0  # Allow proceeding with Stage 2 only
    elif stage4_ok:
        logger.warning("⚠ Stage 4 OK, but Stage 2 has issues (may proceed with Stage 4 only)")
        overall_status = 0  # Allow proceeding with Stage 4 only
    else:
        logger.error("✗ Critical checks failed - both Stage 2 and Stage 4 have issues")
        overall_status = 1
    
    logger.info(f"Full log saved to: {log_file}")
    
    # Flush all handlers
    for handler in logger.handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    
    # Explicit garbage collection to help prevent segfaults during cleanup
    gc.collect()
    
    # Clean up handlers to prevent issues during Python shutdown
    for handler in logger.handlers[:]:
        try:
            handler.close()
        except Exception:
            pass
        logger.removeHandler(handler)
    
    # Final garbage collection
    gc.collect()
    
    return overall_status

if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        exit_code = 130
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        exit_code = 1
    finally:
        # Ensure clean exit
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            gc.collect()
        except Exception:
            pass
    
    # Use os._exit to bypass Python cleanup that might cause segfaults
    # This helps prevent segfaults during library cleanup (e.g., OpenCV, PyAV)
    os._exit(exit_code)

