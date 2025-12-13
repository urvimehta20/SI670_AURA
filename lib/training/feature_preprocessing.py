"""
Feature preprocessing utilities for training.

Provides functions to:
- Remove collinear features
- Combine features from multiple stages
- Normalize and scale features
"""

from __future__ import annotations

import sys
import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
import polars as pl
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import sklearn for VIF calculation
try:
    from sklearn.feature_selection import VarianceThreshold
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    SKLEARN_AVAILABLE = True
    STATSMODELS_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    STATSMODELS_AVAILABLE = False
    VarianceThreshold = None
    variance_inflation_factor = None
    add_constant = None


def remove_collinear_features(
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    correlation_threshold: float = 0.95,
    vif_threshold: float = 10.0,
    method: str = "correlation"
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Remove collinear features using correlation analysis or VIF.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        feature_names: Optional list of feature names
        correlation_threshold: Maximum correlation allowed between features (default: 0.95)
        vif_threshold: Maximum VIF allowed (default: 10.0, only used if method="vif")
        method: Method to use ("correlation" or "vif" or "both")
    
    Returns:
        Tuple of:
        - Filtered feature matrix (n_samples, n_features_filtered)
        - Indices of kept features
        - Names of kept features (or empty list if feature_names not provided)
    """
    if features.shape[1] == 0:
        return features, [], []
    
    n_samples, n_features = features.shape
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    if len(feature_names) != n_features:
        logger.warning(f"Feature names length ({len(feature_names)}) doesn't match features ({n_features})")
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Remove features with zero variance
    if SKLEARN_AVAILABLE:
        try:
            variance_selector = VarianceThreshold(threshold=0.0)
            features_var_filtered = variance_selector.fit_transform(features)
            kept_indices = variance_selector.get_support(indices=True).tolist()
            logger.info(f"Removed {n_features - len(kept_indices)} features with zero variance")
        except ValueError as e:
            # All features have zero variance - use manual filtering instead
            logger.warning(f"VarianceThreshold failed (all features may have zero variance): {e}")
            logger.warning("Falling back to manual variance filtering")
            variances = np.var(features, axis=0)
            kept_indices = np.where(variances > 1e-8)[0].tolist()
            if len(kept_indices) == 0:
                # All features have zero variance - keep all features as fallback
                logger.warning("All features have zero variance! Keeping all features as fallback.")
                kept_indices = list(range(n_features))
            features_var_filtered = features[:, kept_indices]
            logger.info(f"Removed {n_features - len(kept_indices)} features with zero variance (manual)")
    else:
        # Manual variance filtering
        variances = np.var(features, axis=0)
        kept_indices = np.where(variances > 1e-8)[0].tolist()
        if len(kept_indices) == 0:
            # All features have zero variance - keep all features as fallback
            logger.warning("All features have zero variance! Keeping all features as fallback.")
            kept_indices = list(range(n_features))
        features_var_filtered = features[:, kept_indices]
        logger.info(f"Removed {n_features - len(kept_indices)} features with zero variance")
    
    if len(kept_indices) == 0:
        logger.warning("All features removed due to zero variance! Using original features as fallback.")
        return features, list(range(n_features)), feature_names
    
    # Update feature names
    kept_feature_names = [feature_names[i] for i in kept_indices]
    
    # Remove collinear features based on correlation
    if method in ["correlation", "both"]:
        # Compute correlation matrix
        corr_matrix = np.corrcoef(features_var_filtered.T)
        
        # Find highly correlated feature pairs
        # NOTE: range(len()) is necessary here because we need indices for matrix access
        to_remove = set()
        for i in range(len(kept_indices)):
            if i in to_remove:
                continue
            for j in range(i + 1, len(kept_indices)):
                if j in to_remove:
                    continue
                if abs(corr_matrix[i, j]) >= correlation_threshold:
                    # Remove the feature with lower variance (less informative)
                    var_i = np.var(features_var_filtered[:, i])
                    var_j = np.var(features_var_filtered[:, j])
                    if var_i < var_j:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)
                    logger.debug(
                        f"Removing collinear feature: {kept_feature_names[j if var_i < var_j else i]} "
                        f"(correlation={corr_matrix[i, j]:.3f})"
                    )
        
        # Filter features
        # NOTE: range(len()) is necessary here for index-based filtering
        final_kept_indices = [kept_indices[i] for i in range(len(kept_indices)) if i not in to_remove]
        features_filtered = features_var_filtered[:, [i for i in range(len(kept_indices)) if i not in to_remove]]
        final_feature_names = [kept_feature_names[i] for i in range(len(kept_indices)) if i not in to_remove]
        
        logger.info(
            f"Removed {len(kept_indices) - len(final_kept_indices)} collinear features "
            f"(correlation >= {correlation_threshold})"
        )
        
        if method == "both":
            # Also apply VIF filtering
            features_filtered, final_kept_indices, final_feature_names = _remove_vif_collinear(
                features_filtered, final_kept_indices, final_feature_names, vif_threshold
            )
    elif method == "vif":
        # Use VIF only
        features_filtered, final_kept_indices, final_feature_names = _remove_vif_collinear(
            features_var_filtered, kept_indices, kept_feature_names, vif_threshold
        )
    else:
        logger.warning(f"Unknown method '{method}', using correlation method")
        return remove_collinear_features(
            features, feature_names, correlation_threshold, vif_threshold, method="correlation"
        )
    
    logger.info(
        f"Final feature count: {len(final_kept_indices)}/{n_features} "
        f"({100 * len(final_kept_indices) / n_features:.1f}% retained)"
    )
    
    return features_filtered, final_kept_indices, final_feature_names


def _remove_vif_collinear(
    features: np.ndarray,
    kept_indices: List[int],
    feature_names: List[str],
    vif_threshold: float
) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Remove features with high Variance Inflation Factor (VIF).
    
    Args:
        features: Feature matrix (n_samples, n_features)
        kept_indices: Current list of kept feature indices
        feature_names: Current list of kept feature names
        vif_threshold: Maximum VIF allowed
    
    Returns:
        Tuple of (filtered_features, final_kept_indices, final_feature_names)
    """
    if not STATSMODELS_AVAILABLE:
        logger.warning("statsmodels not available, skipping VIF filtering")
        return features, kept_indices, feature_names
    
    if features.shape[1] == 0:
        return features, kept_indices, feature_names
    
    # Calculate VIF for each feature
    try:
        # Add constant for VIF calculation
        features_with_const = add_constant(features, has_constant='skip')
        
        vif_scores = []
        for i in range(1, features_with_const.shape[1]):  # Skip constant column
            try:
                vif = variance_inflation_factor(features_with_const.values, i)
                vif_scores.append(vif if not np.isnan(vif) and np.isfinite(vif) else np.inf)
            except Exception as e:
                logger.debug(f"Error calculating VIF for feature {i}: {e}")
                vif_scores.append(np.inf)
        
        # Remove features with VIF > threshold
        to_keep = [i for i, vif in enumerate(vif_scores) if vif <= vif_threshold]
        
        if len(to_keep) < len(vif_scores):
            removed_count = len(vif_scores) - len(to_keep)
            logger.info(
                f"Removed {removed_count} features with VIF > {vif_threshold} "
                f"(max VIF: {max(vif_scores):.2f})"
            )
            
            final_features = features[:, to_keep]
            final_indices = [kept_indices[i] for i in to_keep]
            final_names = [feature_names[i] for i in to_keep]
            
            return final_features, final_indices, final_names
        else:
            logger.debug(f"All features have VIF <= {vif_threshold}")
            return features, kept_indices, feature_names
            
    except Exception as e:
        logger.warning(f"Error in VIF calculation: {e}, skipping VIF filtering")
        return features, kept_indices, feature_names


def _normalize_video_path(path: str) -> str:
    """Normalize video path for matching by extracting video ID."""
    path_str = str(path)
    # Extract filename without extension
    filename = Path(path_str).stem
    # Remove common prefixes/suffixes
    # Handle scaled videos: FX5aeuJFQ64_aug1_scaled_aug1 -> FX5aeuJFQ64_aug1
    if '_scaled' in filename:
        filename = filename.split('_scaled')[0]
    # Handle augmentation: FX5aeuJFQ64_aug1 -> FX5aeuJFQ64
    if '_aug' in filename:
        parts = filename.split('_aug')
        if len(parts) > 1:
            try:
                int(parts[1])  # Check if it's a number
                filename = parts[0] + '_aug' + parts[1]  # Keep aug index
            except ValueError:
                filename = parts[0]
    return filename.lower()


def _match_video_path(target_path: str, candidate_paths: List[str]) -> Optional[str]:
    """Match a target video path to a candidate path using normalized matching with O(1) dict lookup."""
    target_normalized = _normalize_video_path(target_path)
    
    # Build normalized lookup dict once for O(1) lookups instead of O(n) iteration
    normalized_lookup = {_normalize_video_path(p): p for p in candidate_paths}
    
    # O(1) exact match lookup
    if target_normalized in normalized_lookup:
        return normalized_lookup[target_normalized]
    
    # Fallback: substring matching (only if exact match fails)
    for norm_path, orig_path in normalized_lookup.items():
        if target_normalized in norm_path or norm_path in target_normalized:
            return orig_path
    
    # Final fallback: match by video ID (filename stem)
    target_id = Path(target_path).stem.lower()
    for orig_path in candidate_paths:
        candidate_id = Path(orig_path).stem.lower()
        if target_id in candidate_id or candidate_id in target_id:
            return orig_path
    
    return None


def load_and_combine_features(
    features_stage2_path: Optional[str],
    features_stage4_path: Optional[str],
    video_paths: List[str],
    project_root: str,
    remove_collinearity: bool = True,
    correlation_threshold: float = 0.95,
    vif_threshold: float = 10.0,
    collinearity_method: str = "correlation"
) -> Tuple[np.ndarray, List[str], Optional[List[int]], Optional[np.ndarray]]:
    """
    Load and combine features from Stage 2 and Stage 4 metadata files.
    Optionally removes collinear features after combining.
    
    Args:
        features_stage2_path: Path to Stage 2 features metadata
        features_stage4_path: Path to Stage 4 features metadata
        video_paths: List of video paths to match features
        project_root: Project root directory
        remove_collinearity: Whether to remove collinear features (default: True)
        correlation_threshold: Maximum correlation allowed (default: 0.95)
        vif_threshold: Maximum VIF allowed (default: 10.0)
        collinearity_method: Method for collinearity removal ("correlation", "vif", or "both")
    
    Returns:
        Tuple of (combined_features, feature_names, kept_feature_indices, valid_video_indices)
        - combined_features: Feature matrix (n_samples, n_features)
        - feature_names: List of feature names
        - kept_feature_indices: Indices of kept features (None if remove_collinearity=False)
        - valid_video_indices: Indices of videos that have valid (non-zero) features (None if all valid)
    """
    all_features = []
    all_feature_names = []
    # Track which videos have features from each stage separately
    has_stage2 = np.zeros(len(video_paths), dtype=bool)
    has_stage4 = np.zeros(len(video_paths), dtype=bool)
    
    # Load Stage 2 features
    stage2_loaded = False
    if features_stage2_path and Path(features_stage2_path).exists():
        logger.info("Loading Stage 2 features...")
        logger.info(f"Stage 2 path: {features_stage2_path}")
        sys.stdout.flush()
        try:
            path_obj = Path(features_stage2_path)
            if path_obj.suffix == '.arrow':
                df2 = pl.read_ipc(path_obj)
            elif path_obj.suffix == '.parquet':
                df2 = pl.read_parquet(path_obj)
            else:
                df2 = pl.read_csv(features_stage2_path)
            logger.info(f"✓ Loaded Stage 2 metadata: {df2.height} rows, {len(df2.columns)} columns")
            
            # Get feature columns (exclude metadata columns)
            metadata_cols = {'video_path', 'label', 'feature_path'}
            feature_cols = [col for col in df2.columns if col not in metadata_cols]
            
            if feature_cols:
                # Build features dictionary with all candidate paths (optimized for O(1) lookup)
                features_dict = {}
                candidate_paths = []
                for row in df2.iter_rows(named=True):
                    video_path_key = row['video_path']
                    features_dict[video_path_key] = [row[col] for col in feature_cols]
                    candidate_paths.append(video_path_key)
                
                stage2_features = []
                stage2_valid_count = 0
                unmatched_samples = []  # Track some unmatched samples for debugging
                for idx, vpath in enumerate(video_paths):
                    # Try to match video path using optimized O(1) matching
                    matched_key = _match_video_path(vpath, candidate_paths)
                    
                    if matched_key and matched_key in features_dict:
                        matched = features_dict[matched_key]
                        stage2_features.append(matched)
                        stage2_valid_count += 1
                        has_stage2[idx] = True  # Mark as having Stage 2 features
                    else:
                        if len(unmatched_samples) < 5:  # Log first 5 unmatched
                            unmatched_samples.append(vpath)
                        matched = [0.0] * len(feature_cols)
                        stage2_features.append(matched)
                
                if unmatched_samples:
                    logger.debug(f"Sample unmatched Stage 2 videos (showing first {len(unmatched_samples)}): {unmatched_samples[:3]}")
                    logger.debug(f"Sample Stage 2 feature paths (showing first 3): {candidate_paths[:3] if candidate_paths else 'None'}")
                
                # Convert to numpy array with defensive error handling
                logger.info(f"Converting Stage 2 features to numpy array...")
                sys.stdout.flush()
                try:
                    stage2_array = np.array(stage2_features)
                    logger.info(f"Stage 2 array shape: {stage2_array.shape}, dtype: {stage2_array.dtype}")
                    all_features.append(stage2_array)
                    all_feature_names.extend([f"stage2_{col}" for col in feature_cols])
                    logger.info(f"Loaded {len(feature_cols)} Stage 2 features ({stage2_valid_count}/{len(video_paths)} videos matched)")
                    stage2_loaded = True
                except MemoryError as e:
                    logger.critical(f"Memory error converting Stage 2 features to numpy: {e}")
                    logger.critical(f"Feature list length: {len(stage2_features)}, Feature count per video: {len(feature_cols)}")
                    raise
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error converting Stage 2 features to numpy: {e}", exc_info=True)
                    if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                        logger.critical("CRITICAL: Possible crash during numpy array conversion")
                    raise
        except MemoryError as e:
            logger.critical(f"Memory error loading Stage 2 features from {features_stage2_path}: {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading Stage 2 features: {e}", exc_info=True)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical("CRITICAL: Possible crash during Stage 2 feature file reading")
                logger.critical(f"File: {features_stage2_path}")
                logger.critical("This may indicate corrupted feature file or polars library issue")
            raise
    
    # Load Stage 4 features
    stage4_loaded = False
    if features_stage4_path and Path(features_stage4_path).exists():
        logger.info("Loading Stage 4 features...")
        logger.info(f"Stage 4 path: {features_stage4_path}")
        sys.stdout.flush()
        try:
            path_obj = Path(features_stage4_path)
            if path_obj.suffix == '.arrow':
                df4 = pl.read_ipc(path_obj)
            elif path_obj.suffix == '.parquet':
                df4 = pl.read_parquet(path_obj)
            else:
                df4 = pl.read_csv(features_stage4_path)
            logger.info(f"✓ Loaded Stage 4 metadata: {df4.height} rows, {len(df4.columns)} columns")
            
            # Get feature columns
            metadata_cols = {'video_path', 'label', 'feature_path'}
            feature_cols = [col for col in df4.columns if col not in metadata_cols]
            
            if feature_cols:
                # Build features dictionary (optimized for O(1) lookup)
                features_dict = {}
                candidate_paths = []
                for row in df4.iter_rows(named=True):
                    video_path_key = row['video_path']
                    features_dict[video_path_key] = [row[col] for col in feature_cols]
                    candidate_paths.append(video_path_key)
                
                stage4_features = []
                stage4_valid_count = 0
                unmatched_samples = []  # Track some unmatched samples for debugging
                for idx, vpath in enumerate(video_paths):
                    # Use optimized O(1) path matching
                    matched_key = _match_video_path(vpath, candidate_paths)
                    
                    if matched_key and matched_key in features_dict:
                        matched = features_dict[matched_key]
                        stage4_features.append(matched)
                        stage4_valid_count += 1
                        has_stage4[idx] = True  # Mark as having Stage 4 features
                    else:
                        if len(unmatched_samples) < 5:  # Log first 5 unmatched
                            unmatched_samples.append(vpath)
                        matched = [0.0] * len(feature_cols)
                        stage4_features.append(matched)
                
                if unmatched_samples:
                    logger.debug(f"Sample unmatched Stage 4 videos (showing first {len(unmatched_samples)}): {unmatched_samples[:3]}")
                    logger.debug(f"Sample Stage 4 feature paths (showing first 3): {candidate_paths[:3] if candidate_paths else 'None'}")
                
                # Convert to numpy array with defensive error handling
                logger.info(f"Converting Stage 4 features to numpy array...")
                sys.stdout.flush()
                try:
                    stage4_array = np.array(stage4_features)
                    logger.info(f"Stage 4 array shape: {stage4_array.shape}, dtype: {stage4_array.dtype}")
                    all_features.append(stage4_array)
                    all_feature_names.extend([f"stage4_{col}" for col in feature_cols])
                    logger.info(f"Loaded {len(feature_cols)} Stage 4 features ({stage4_valid_count}/{len(video_paths)} videos matched)")
                    stage4_loaded = True
                except MemoryError as e:
                    logger.critical(f"Memory error converting Stage 4 features to numpy: {e}")
                    logger.critical(f"Feature list length: {len(stage4_features)}, Feature count per video: {len(feature_cols)}")
                    raise
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error converting Stage 4 features to numpy: {e}", exc_info=True)
                    if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                        logger.critical("CRITICAL: Possible crash during numpy array conversion")
                    raise
        except MemoryError as e:
            logger.critical(f"Memory error loading Stage 4 features from {features_stage4_path}: {e}")
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error loading Stage 4 features: {e}", exc_info=True)
            if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                logger.critical("CRITICAL: Possible crash during Stage 4 feature file reading")
                logger.critical(f"File: {features_stage4_path}")
                logger.critical("This may indicate corrupted feature file or polars library issue")
            raise
    
    if not all_features:
        logger.warning("No features loaded!")
        return np.array([]).reshape(len(video_paths), 0), [], None, None
    
    # Combine features with defensive error handling
    logger.info(f"Combining {len(all_features)} feature arrays...")
    logger.info(f"Feature array shapes: {[arr.shape for arr in all_features]}")
    sys.stdout.flush()
    
    try:
        combined_features = np.hstack(all_features)
        logger.info(f"Combined {len(all_feature_names)} features from {len(all_features)} stages")
        logger.info(f"Combined feature matrix shape: {combined_features.shape}, dtype: {combined_features.dtype}")
        logger.info(f"Combined feature matrix size: {combined_features.nbytes / 1024**2:.2f} MB")
    except MemoryError as e:
        logger.critical(f"Memory error during feature combination: {e}")
        sizes_mb = [f"{arr.nbytes / (1024**2):.2f}" for arr in all_features]
        logger.critical(f"Feature array sizes: {sizes_mb} MB")
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to combine features: {e}", exc_info=True)
        if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
            logger.critical("CRITICAL: Possible crash during numpy.hstack()")
            logger.critical("This may indicate corrupted feature arrays or memory issue")
        raise
    
    # Determine valid video indices based on which stages are loaded
    # If both Stage 2 and Stage 4 are loaded, keep only videos that have BOTH (intersection)
    # If only one stage is loaded, keep videos with that stage's features
    if stage2_loaded and stage4_loaded:
        # Require BOTH Stage 2 AND Stage 4 features (intersection)
        # This ensures we use the smaller set (x-n rows) as the user requested
        valid_video_indices = np.where(has_stage2 & has_stage4)[0]
        logger.info(f"Stage 2: {np.sum(has_stage2)} videos, Stage 4: {np.sum(has_stage4)} videos")
        logger.info(f"Intersection (videos with BOTH Stage 2 AND Stage 4): {len(valid_video_indices)} videos")
    elif stage2_loaded:
        # Only Stage 2 loaded
        valid_video_indices = np.where(has_stage2)[0]
        logger.info(f"Stage 2 only: {len(valid_video_indices)} videos with features")
    elif stage4_loaded:
        # Only Stage 4 loaded
        valid_video_indices = np.where(has_stage4)[0]
        logger.info(f"Stage 4 only: {len(valid_video_indices)} videos with features")
    else:
        # No features loaded
        logger.warning("No features loaded from either stage!")
        return np.array([]).reshape(len(video_paths), 0), [], None, None
    
    logger.info(f"✓ Found {len(valid_video_indices)} videos with valid features")
    
    # Return None if all videos are valid (for backward compatibility)
    if len(valid_video_indices) == len(video_paths):
        valid_video_indices = None
        logger.info(f"All {len(video_paths)} videos have valid features")
    
    # Remove collinear features if requested
    kept_feature_indices = None
    if remove_collinearity and combined_features.shape[1] > 0:
        logger.info("Removing collinear features from combined features...")
        combined_features, kept_feature_indices, all_feature_names = remove_collinear_features(
            combined_features,
            feature_names=all_feature_names,
            correlation_threshold=correlation_threshold,
            vif_threshold=vif_threshold,
            method=collinearity_method
        )
        logger.info(f"Final feature count after collinearity removal: {len(all_feature_names)}")
    
    return combined_features, all_feature_names, kept_feature_indices, valid_video_indices


__all__ = [
    "remove_collinear_features",
    "load_and_combine_features",
    "_normalize_video_path",
    "_match_video_path",
]

