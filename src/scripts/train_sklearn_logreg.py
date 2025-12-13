#!/usr/bin/env python3
"""
Train sklearn LogisticRegression with L1/L2/ElasticNet regularization.

Uses Stage 2/4 features with proper regularization (not MLP).
"""

from __future__ import annotations

import os
import sys
import signal
import traceback
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, confusion_matrix, accuracy_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup signal handlers to catch crashes and log diagnostics
def setup_crash_handlers(logger):
    """Setup signal handlers to catch crashes and log diagnostics."""
    def crash_handler(signum, frame):
        """Handle crash signals and log diagnostics before exit."""
        logger.critical("=" * 80)
        logger.critical("CRITICAL: Process received signal %d (likely crash/segfault)", signum)
        logger.critical("=" * 80)
        logger.critical("Signal name: %s", signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum))
        logger.critical("Attempting to log diagnostics before crash...")
        
        try:
            import psutil
            process = psutil.Process()
            logger.critical("Memory usage: RSS=%.2f GB, VMS=%.2f GB", 
                          process.memory_info().rss / 1024**3,
                          process.memory_info().vms / 1024**3)
        except Exception:
            pass
        
        try:
            logger.critical("Stack trace at crash:")
            for line in traceback.format_stack(frame):
                logger.critical(line.rstrip())
        except Exception:
            pass
        
        logger.critical("=" * 80)
        sys.stdout.flush()
        sys.stderr.flush()
        # Re-raise to get core dump
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)
    
    # Register handlers for common crash signals
    if hasattr(signal, 'SIGSEGV'):
        signal.signal(signal.SIGSEGV, crash_handler)
    if hasattr(signal, 'SIGABRT'):
        signal.signal(signal.SIGABRT, crash_handler)
    if hasattr(signal, 'SIGBUS'):
        signal.signal(signal.SIGBUS, crash_handler)
    if hasattr(signal, 'SIGFPE'):
        signal.signal(signal.SIGFPE, crash_handler)

from lib.training.feature_training_pipeline import load_features_for_training
from lib.training.feature_pipeline import create_stratified_splits
from lib.utils.paths import load_metadata_flexible

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] %(message)s',
    force=True  # Force reconfiguration in case logging was already configured
)
logger = logging.getLogger(__name__)

# Immediately log startup to ensure something is written to log file
logger.info("=" * 80)
logger.info("STAGE 5ALPHA: sklearn LogisticRegression Training Script Starting")
logger.info("=" * 80)
sys.stdout.flush()
sys.stderr.flush()

# Setup crash handlers after logger is created
setup_crash_handlers(logger)


def train_sklearn_logreg(
    project_root: str,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: Optional[str],
    output_dir: str,
    n_splits: int = 5,
    delete_existing: bool = False
) -> Dict[str, Any]:
    """
    Train sklearn LogisticRegression with L1/L2/ElasticNet regularization.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled metadata
        features_stage2_path: Path to Stage 2 features
        features_stage4_path: Path to Stage 4 features (optional)
        output_dir: Output directory
        n_splits: Number of CV folds
    
    Returns:
        Training results dictionary
    """
    # Input validation
    if not project_root or not isinstance(project_root, str):
        raise ValueError(f"project_root must be a non-empty string, got: {type(project_root)}")
    if not scaled_metadata_path or not isinstance(scaled_metadata_path, str):
        raise ValueError(f"scaled_metadata_path must be a non-empty string, got: {type(scaled_metadata_path)}")
    if not features_stage2_path or not isinstance(features_stage2_path, str):
        raise ValueError(f"features_stage2_path must be a non-empty string, got: {type(features_stage2_path)}")
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError(f"output_dir must be a non-empty string, got: {type(output_dir)}")
    if n_splits <= 0 or not isinstance(n_splits, int):
        raise ValueError(f"n_splits must be a positive integer, got: {n_splits}")
    
    try:
        project_root_path = Path(project_root).resolve()
        if not project_root_path.exists():
            raise FileNotFoundError(f"Project root directory does not exist: {project_root_path}")
        if not project_root_path.is_dir():
            raise NotADirectoryError(f"Project root is not a directory: {project_root_path}")
    except (OSError, ValueError) as e:
        logger.error(f"Invalid project_root path: {project_root} - {e}")
        raise ValueError(f"Invalid project_root path: {project_root}") from e
    
    # Handle relative output_dir paths
    try:
        if Path(output_dir).is_absolute():
            output_dir_path = Path(output_dir)
        else:
            output_dir_path = project_root_path / output_dir
        
        # Delete existing output directory if delete_existing is True
        if delete_existing and output_dir_path.exists():
            try:
                import shutil
                shutil.rmtree(output_dir_path)
                logger.info(f"Deleted existing output directory (clean mode): {output_dir_path}")
            except (OSError, PermissionError, FileNotFoundError) as e:
                logger.warning(f"Could not delete {output_dir_path}: {e}")
        
        output_dir_path.mkdir(parents=True, exist_ok=True)
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to create output directory {output_dir}: {e}")
        raise ValueError(f"Cannot create output directory: {output_dir}") from e
    
    # Load metadata
    logger.info("Loading scaled metadata...")
    try:
        scaled_df = load_metadata_flexible(scaled_metadata_path)
        if scaled_df is None or scaled_df.height == 0:
            raise ValueError(f"Scaled metadata not found or empty: {scaled_metadata_path}")
        
        if scaled_df.height <= 3000:
            logger.error(f"Insufficient data for training: {scaled_df.height} rows (need > 3000)")
            raise ValueError(f"Insufficient data: {scaled_df.height} rows (need > 3000)")
        
        # Log validation success message (matches SLURM script expectation)
        # Use a format that's easy to grep: no special characters that might cause issues
        logger.info(f"Data validation passed: {scaled_df.height} rows (> 3000 required)")
        # Also log with checkmark for human readability
        logger.info(f"✓ Data validation check successful")
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception as e:
        logger.error(f"Failed to load scaled metadata from {scaled_metadata_path}: {e}")
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    
    video_paths = scaled_df["video_path"].to_list()
    labels = scaled_df["label"].to_list()
    
    # Explicitly clear DataFrame to avoid cleanup issues
    del scaled_df
    import gc
    gc.collect()
    
    # Convert labels to binary
    unique_labels = sorted(set(labels))
    if len(unique_labels) != 2:
        raise ValueError(f"Expected binary classification, got {len(unique_labels)} classes")
    
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    labels_array = np.array([label_map[label] for label in labels])
    
    logger.info(f"Labels: {unique_labels} -> {label_map}")
    logger.info(f"Loaded {len(video_paths)} videos")
    
    # Load features
    logger.info("Loading features from Stage 2/4...")
    try:
        features, feature_names, valid_video_indices = load_features_for_training(
            features_stage2_path,
            features_stage4_path,
            video_paths,
            project_root
        )
    except Exception as e:
        logger.error(f"Failed to load features: {e}", exc_info=True)
        raise ValueError(f"Feature loading failed: {e}") from e
    
    # Filter to valid video indices if needed
    if valid_video_indices is not None:
        if len(valid_video_indices) > 0:
            # Filter to only videos with valid features
            features = features[valid_video_indices]
            labels_array = labels_array[valid_video_indices]
            video_paths = [video_paths[i] for i in valid_video_indices]
            logger.info(f"Filtered to {len(features)} videos with valid features")
        else:
            # Empty array means no valid videos
            raise ValueError("No videos have valid features (valid_video_indices is empty)")
    
    logger.info(f"✓ Loaded {len(feature_names)} features for {len(features)} videos")
    logger.info(f"  Input dimension: {features.shape[1]}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    if len(features) < 3000:
        logger.error(f"Insufficient data for training: {len(features)} valid videos (need > 3000)")
        sys.stdout.flush()
        sys.stderr.flush()
        raise ValueError(f"Insufficient valid videos: {len(features)} < 3000")
    
    # Create 60-20-20 split
    logger.info("Creating 60-20-20 stratified train-val-test split...")
    train_idx, val_idx, test_idx = create_stratified_splits(
        features, labels_array, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    X_train, X_val, X_test = features[train_idx], features[val_idx], features[test_idx]
    y_train, y_val, y_test = labels_array[train_idx], labels_array[val_idx], labels_array[test_idx]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    sys.stdout.flush()
    sys.stderr.flush()
    
    # OPTIMIZATION: Use 20% stratified sample for hyperparameter search (faster)
    # Final training will use full dataset for robustness
    from sklearn.model_selection import StratifiedShuffleSplit
    
    logger.info("=" * 80)
    logger.info("HYPERPARAMETER SEARCH: Using 20% stratified sample for efficiency")
    logger.info("=" * 80)
    
    # Sample 20% of train+val for hyperparameter search
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=42)
    sample_indices, _ = next(sss.split(X_trainval, y_trainval))
    
    X_trainval_sample = X_trainval[sample_indices]
    y_trainval_sample = y_trainval[sample_indices]
    
    # Split sample into train/val for hyperparameter search
    from sklearn.model_selection import train_test_split
    X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
        X_trainval_sample, y_trainval_sample, test_size=0.2, random_state=42, stratify=y_trainval_sample
    )
    
    logger.info(f"Hyperparameter search sample: {len(X_trainval_sample)} rows ({100.0 * len(X_trainval_sample) / len(X_trainval):.1f}% of {len(X_trainval)} total)")
    logger.info(f"  Sample train: {len(X_train_sample)}, Sample val: {len(X_val_sample)}")
    
    # Hyperparameter grid for sklearn LogisticRegression
    # Note: elasticnet requires l1_ratio parameter and saga solver
    # Max 50 combinations: 4*3*2*1*3 = 36 raw, but after filtering ~44 combinations
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],  # 4 values (reduced from 5)
        "penalty": ["l1", "l2", "elasticnet"],  # 3 values
        "solver": ["liblinear", "saga"],  # saga supports elasticnet (2 values)
        "max_iter": [1000],  # 1 value (reduced from 2)
        "l1_ratio": [0.1, 0.5, 0.9]  # 3 values (only used for elasticnet)
    }
    
    # Filter: elasticnet only works with saga solver and requires l1_ratio
    grid = list(ParameterGrid(param_grid))
    # Remove elasticnet with non-saga solver
    grid = [p for p in grid if not (p["penalty"] == "elasticnet" and p["solver"] != "saga")]
    # For non-elasticnet penalties, remove l1_ratio from params dict (not needed by sklearn)
    filtered_grid = []
    for p in grid:
        if p["penalty"] != "elasticnet":
            # Remove l1_ratio for non-elasticnet penalties
            p_clean = {k: v for k, v in p.items() if k != "l1_ratio"}
            filtered_grid.append(p_clean)
        else:
            # Keep l1_ratio for elasticnet (required)
            filtered_grid.append(p)
    grid = filtered_grid
    
    logger.info(f"Hyperparameter search: {len(grid)} combinations")
    
    if len(grid) == 0:
        raise ValueError(
            "No valid parameter combinations after filtering. "
            "This may indicate an issue with the parameter grid configuration."
        )
    
    # Scale features for hyperparameter search (using 20% sample)
    logger.info("Scaling features for hyperparameter search (20% sample)...")
    logger.info(f"Sample train shape: {X_train_sample.shape}, dtype: {X_train_sample.dtype}")
    sys.stdout.flush()
    
    # Check for corrupted features before scaling
    if np.any(np.isnan(X_train_sample)):
        nan_count = np.isnan(X_train_sample).sum()
        logger.warning(f"Found {nan_count} NaN values in sample training features, replacing with 0")
        X_train_sample = np.nan_to_num(X_train_sample, nan=0.0)
    if np.any(np.isinf(X_train_sample)):
        inf_count = np.isinf(X_train_sample).sum()
        logger.warning(f"Found {inf_count} Inf values in sample training features, replacing with 0")
        X_train_sample = np.nan_to_num(X_train_sample, posinf=0.0, neginf=0.0)
    
    try:
        scaler_sample = StandardScaler()
        X_train_sample_scaled = scaler_sample.fit_transform(X_train_sample)
        X_val_sample_scaled = scaler_sample.transform(X_val_sample)
        logger.info(f"Feature scaling completed for hyperparameter search. Scaled shape: {X_train_sample_scaled.shape}")
    except MemoryError as e:
        logger.critical(f"Memory error during feature scaling: {e}")
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to scale features: {e}", exc_info=True)
        if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
            logger.critical("CRITICAL: Possible crash during sklearn StandardScaler.fit_transform()")
        raise
    
    # Grid search on 20% sample
    best_score = -1
    best_params = None
    best_model = None
    grid_results = []
    
    for param_idx, params in enumerate(grid):
        logger.info(f"Grid search {param_idx + 1}/{len(grid)}: {params}")
        
        try:
            # Params already have l1_ratio removed for non-elasticnet penalties
            logger.info(f"Training LogisticRegression with params: {params} (20% sample)")
            logger.info(f"Training data: {X_train_sample_scaled.shape[0]} samples, {X_train_sample_scaled.shape[1]} features")
            sys.stdout.flush()
            
            model = LogisticRegression(
                **params,
                random_state=42,
                n_jobs=1
            )
            
            try:
                model.fit(X_train_sample_scaled, y_train_sample)
                logger.info(f"✓ Model.fit() completed successfully")
            except MemoryError as e:
                logger.critical(f"Memory error during LogisticRegression.fit(): {e}")
                raise
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to train LogisticRegression: {e}", exc_info=True)
                if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                    logger.critical("CRITICAL: Possible crash during sklearn LogisticRegression.fit()")
                raise
            
            # Validate with defensive error handling
            logger.info("Running model.predict_proba() on validation set (20% sample)...")
            sys.stdout.flush()
            
            try:
                val_probs_full = model.predict_proba(X_val_sample_scaled)
                logger.info(f"✓ predict_proba() completed (shape: {val_probs_full.shape})")
            except MemoryError as e:
                logger.critical(f"Memory error during LogisticRegression.predict_proba(): {e}")
                raise
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Failed to predict probabilities: {e}", exc_info=True)
                if "core dump" in error_msg.lower() or "segmentation fault" in error_msg.lower() or "aborted" in error_msg.lower():
                    logger.critical("CRITICAL: Possible crash during sklearn LogisticRegression.predict_proba()")
                raise
            
            if val_probs_full.shape[1] != 2:
                raise ValueError(f"Expected binary classification, got {val_probs_full.shape[1]} classes")
            val_probs = val_probs_full[:, 1]
            val_preds = (val_probs > 0.5).astype(int)
            val_f1 = f1_score(y_val_sample, val_preds)
            
            # Check for NaN or invalid values
            if np.any(np.isnan(val_probs)) or np.any(np.isinf(val_probs)):
                logger.warning(f"Invalid probabilities detected for params {params}, skipping")
                continue
            
            grid_results.append({
                "params": params,
                "val_f1": float(val_f1)
            })
            
            if val_f1 > best_score:
                best_score = val_f1
                best_params = params.copy()
                best_model = model
                
        except Exception as e:
            logger.warning(f"Failed with params {params}: {e}")
            continue
    
    if best_model is None:
        error_msg = (
            f"No valid model found during grid search. "
            f"Tried {len(grid)} parameter combinations. "
            f"This may indicate: (1) all parameter combinations failed to converge, "
            f"(2) data issues, or (3) solver/penalty incompatibilities."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Best hyperparameters from 20% sample: {best_params} (val_f1: {best_score:.4f})")
    
    # FINAL TRAINING: Train on full dataset with best hyperparameters
    logger.info("=" * 80)
    logger.info("FINAL TRAINING: Using full dataset with best hyperparameters")
    logger.info("=" * 80)
    
    # Scale full dataset
    logger.info("Scaling full dataset for final training...")
    if np.any(np.isnan(X_train)):
        X_train = np.nan_to_num(X_train, nan=0.0)
    if np.any(np.isinf(X_train)):
        X_train = np.nan_to_num(X_train, posinf=0.0, neginf=0.0)
    
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train)
    X_val_scaled = scaler_final.transform(X_val)
    X_test_scaled = scaler_final.transform(X_test)
    logger.info(f"Full dataset scaled. Train: {X_train_scaled.shape[0]}, Val: {X_val_scaled.shape[0]}, Test: {X_test_scaled.shape[0]}")
    
    # 5-fold CV on full train+val
    X_trainval = np.vstack([X_train_scaled, X_val_scaled])
    y_trainval = np.concatenate([y_train, y_val])
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    cv_aucs = []
    
    for fold_idx, (train_idx_cv, val_idx_cv) in enumerate(cv.split(X_trainval, y_trainval)):
        logger.info(f"CV fold {fold_idx + 1}/{n_splits}")
        
        X_train_cv = X_trainval[train_idx_cv]
        X_val_cv = X_trainval[val_idx_cv]
        y_train_cv = y_trainval[train_idx_cv]
        y_val_cv = y_trainval[val_idx_cv]
        
        # Data is already scaled, no need to scale again
        X_train_cv_scaled = X_train_cv
        X_val_cv_scaled = X_val_cv
        
        # best_params already has l1_ratio removed for non-elasticnet
        model_cv = LogisticRegression(**best_params, random_state=42, n_jobs=1)
        model_cv.fit(X_train_cv_scaled, y_train_cv)
        
        val_probs_cv_full = model_cv.predict_proba(X_val_cv_scaled)
        if val_probs_cv_full.shape[1] != 2:
            raise ValueError(f"Expected binary classification in CV, got {val_probs_cv_full.shape[1]} classes")
        val_probs_cv = val_probs_cv_full[:, 1]
        val_preds_cv = (val_probs_cv > 0.5).astype(int)
        
        cv_f1 = f1_score(y_val_cv, val_preds_cv)
        cv_auc = roc_auc_score(y_val_cv, val_probs_cv)
        
        cv_scores.append(cv_f1)
        cv_aucs.append(cv_auc)
    
    cv_mean_f1 = np.mean(cv_scores)
    cv_std_f1 = np.std(cv_scores)
    cv_mean_auc = np.mean(cv_aucs)
    
    logger.info(f"CV F1: {cv_mean_f1:.4f} ± {cv_std_f1:.4f}")
    logger.info(f"CV AUC: {cv_mean_auc:.4f}")
    
    # Train final model on train+val (already scaled)
    logger.info("Training final model on full training+validation set...")
    X_trainval_scaled = X_trainval  # Already scaled
    
    # best_params already has l1_ratio removed for non-elasticnet
    final_model = LogisticRegression(**best_params, random_state=42, n_jobs=1)
    final_model.fit(X_trainval_scaled, y_trainval)
    
    # Evaluate on test set
    X_test_scaled = scaler_final.transform(X_test)
    test_probs_full = final_model.predict_proba(X_test_scaled)
    if test_probs_full.shape[1] != 2:
        raise ValueError(f"Expected binary classification, got {test_probs_full.shape[1]} classes")
    test_probs = test_probs_full[:, 1]
    test_preds = (test_probs > 0.5).astype(int)
    
    test_f1 = f1_score(y_test, test_preds)
    test_auc = roc_auc_score(y_test, test_probs)
    test_ap = average_precision_score(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_preds)
    
    logger.info(f"Test F1: {test_f1:.4f}")
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test AP: {test_ap:.4f}")
    logger.info(f"Test Acc: {test_acc:.4f}")
    
    # Save model and results
    try:
        joblib.dump(final_model, output_dir_path / "model.joblib")
        joblib.dump(scaler_final, output_dir_path / "scaler.joblib")  # Use scaler_final from full dataset
    except Exception as e:
        logger.error(f"Failed to save model/scaler: {e}")
        raise IOError(f"Cannot save model files to {output_dir_path}") from e
    
    # Ensure JSON serializability (convert numpy types to native Python types)
    def make_json_serializable(obj):
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        return obj
    
    results = {
        "best_params": make_json_serializable(best_params),
        "best_val_f1": float(best_score),
        "cv_val_f1": float(cv_mean_f1),
        "cv_val_f1_std": float(cv_std_f1),
        "cv_val_auc": float(cv_mean_auc),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
        "test_ap": float(test_ap),
        "test_acc": float(test_acc),
        "grid_results": make_json_serializable(grid_results)
    }
    
    try:
        with open(output_dir_path / "results.json", "w") as f:
            json.dump(results, f, indent=2)
    except (OSError, IOError, PermissionError) as e:
        logger.error(f"Failed to save results.json: {e}")
        raise IOError(f"Cannot write results.json to {output_dir_path}") from e
    
    # Also save feature names for reference
    try:
        with open(output_dir_path / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
    except (OSError, IOError, PermissionError) as e:
        logger.warning(f"Failed to save feature_names.json: {e}")
        # Non-critical, continue
    
    # Plot ROC/PR curves
    fpr, tpr, _ = roc_curve(y_test, test_probs)
    precision, recall, _ = precision_recall_curve(y_test, test_probs)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(fpr, tpr, label=f'ROC (AUC={test_auc:.4f})')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(recall, precision, label=f'PR (AP={test_ap:.4f})')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    try:
        plt.savefig(output_dir_path / "roc_pr_curves.png", dpi=150)
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to save ROC/PR curves plot: {e}")
        plt.close()  # Ensure plot is closed even on error
    
    logger.info(f"Training complete. Results saved to: {output_dir_path}")
    logger.info("=" * 80)
    logger.info("STAGE 5ALPHA TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    sys.stdout.flush()
    sys.stderr.flush()
    
    return results


def main():
    # Log immediately to ensure output is written
    logger.info("Parsing command line arguments...")
    sys.stdout.flush()
    sys.stderr.flush()
    
    parser = argparse.ArgumentParser(description="Train sklearn LogisticRegression with L1/L2/ElasticNet")
    parser.add_argument("--project-root", type=str, required=True)
    parser.add_argument("--scaled-metadata", type=str, required=True)
    parser.add_argument("--features-stage2", type=str, required=True)
    parser.add_argument("--features-stage4", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument(
        "--delete-existing",
        action="store_true",
        help="Delete existing output directory before training (clean mode, default: False)"
    )
    
    try:
        args = parser.parse_args()
        logger.info(f"Arguments parsed successfully:")
        logger.info(f"  project-root: {args.project_root}")
        logger.info(f"  scaled-metadata: {args.scaled_metadata}")
        logger.info(f"  features-stage2: {args.features_stage2}")
        logger.info(f"  features-stage4: {args.features_stage4}")
        logger.info(f"  output-dir: {args.output_dir}")
        logger.info(f"  n-splits: {args.n_splits}")
        logger.info(f"  delete-existing: {args.delete_existing}")
        sys.stdout.flush()
        sys.stderr.flush()
    except SystemExit as e:
        logger.error(f"Argument parsing failed: {e}")
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    
    try:
        train_sklearn_logreg(
            project_root=args.project_root,
            scaled_metadata_path=args.scaled_metadata,
            features_stage2_path=args.features_stage2,
            features_stage4_path=args.features_stage4,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            delete_existing=args.delete_existing
        )
    except Exception as e:
        logger.critical(f"Training failed with exception: {type(e).__name__}: {e}", exc_info=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    try:
        exit_code = 0
        try:
            main()
            # Ensure all output is flushed before exit
            sys.stdout.flush()
            sys.stderr.flush()
        except SystemExit as e:
            # Capture exit code from SystemExit
            exit_code = e.code if e.code is not None else 0
            sys.stdout.flush()
            sys.stderr.flush()
        except KeyboardInterrupt:
            logger.critical("Process interrupted by user")
            sys.stdout.flush()
            sys.stderr.flush()
            exit_code = 130
        except Exception as e:
            # Catch any unhandled exceptions that might lead to crashes
            logger.critical("=" * 80)
            logger.critical("UNHANDLED EXCEPTION - This may cause a crash")
            logger.critical("=" * 80)
            logger.critical(f"Exception type: {type(e).__name__}")
            logger.critical(f"Exception message: {str(e)}")
            logger.critical("Full traceback:", exc_info=True)
            logger.critical("=" * 80)
            sys.stdout.flush()
            sys.stderr.flush()
            exit_code = 1
        
        # Explicit cleanup before exit
        import gc
        gc.collect()
        
        # Use os._exit to bypass Python cleanup that might cause crashes
        os._exit(exit_code)
    except SystemExit:
        # Re-raise system exits (normal termination)
        raise
    except Exception:
        # Last resort: force exit
        import os
        os._exit(1)
