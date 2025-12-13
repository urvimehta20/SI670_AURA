#!/usr/bin/env python3
"""
Train XGBoost, LightGBM, and CatBoost on Stage 2/4 features.
"""

from __future__ import annotations

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from sklearn.metrics import (
    f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, accuracy_score
)
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import joblib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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
logger.info("STAGE 5BETA: Gradient Boosting Training Script Starting")
logger.info("=" * 80)
sys.stdout.flush()
sys.stderr.flush()

# Try importing gradient boosting libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not available")


def train_gradient_boosting(
    project_root: str,
    scaled_metadata_path: str,
    features_stage2_path: str,
    features_stage4_path: Optional[str],
    output_dir: str,
    n_splits: int = 5,
    models: List[str] = ["xgboost", "lightgbm", "catboost"],
    delete_existing: bool = False
) -> Dict[str, Any]:
    """
    Train XGBoost, LightGBM, and CatBoost on features.
    
    Args:
        project_root: Project root directory
        scaled_metadata_path: Path to scaled metadata
        features_stage2_path: Path to Stage 2 features
        features_stage4_path: Path to Stage 4 features (optional)
        output_dir: Output directory
        n_splits: Number of CV folds
        models: List of models to train
    
    Returns:
        Training results dictionary
    """
    project_root_path = Path(project_root).resolve()
    # Handle relative output_dir paths
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
    
    # Load metadata
    logger.info("Loading scaled metadata...")
    scaled_df = load_metadata_flexible(scaled_metadata_path)
    if scaled_df is None or scaled_df.height == 0:
        raise ValueError(f"Scaled metadata not found: {scaled_metadata_path}")
    
    if scaled_df.height <= 3000:
        logger.error(f"Insufficient data for training: {scaled_df.height} rows (need > 3000)")
        sys.stdout.flush()
        sys.stderr.flush()
        raise ValueError(f"Insufficient data: {scaled_df.height} rows (need > 3000)")
    
    # Log validation success message (matches SLURM script expectation)
    logger.info(f"Data validation passed: {scaled_df.height} rows (> 3000 required)")
    logger.info(f"✓ Data validation check successful")
    sys.stdout.flush()
    sys.stderr.flush()
    
    video_paths = scaled_df["video_path"].to_list()
    labels = scaled_df["label"].to_list()
    
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
    features, feature_names, valid_video_indices = load_features_for_training(
        features_stage2_path,
        features_stage4_path,
        video_paths,
        project_root
    )
    
    # Filter to valid video indices if needed
    if valid_video_indices is not None:
        if len(valid_video_indices) > 0:
            features = features[valid_video_indices]
            labels_array = labels_array[valid_video_indices]
            video_paths = [video_paths[i] for i in valid_video_indices]
            logger.info(f"Filtered to {len(features)} videos with valid features")
        else:
            # Empty array means no valid videos
            raise ValueError("No videos have valid features (valid_video_indices is empty)")
    
    logger.info(f"✓ Loaded {len(feature_names)} features for {len(features)} videos")
    logger.info(f"  Input dimension: {features.shape[1]}")
    
    if len(features) < 3000:
        raise ValueError(f"Insufficient valid videos: {len(features)} < 3000")
    
    # Create 60-20-20 split
    logger.info("Creating 60-20-20 stratified train-val-test split...")
    train_idx, val_idx, test_idx = create_stratified_splits(
        features, labels_array, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2
    )
    
    X_train, X_val, X_test = features[train_idx], features[val_idx], features[test_idx]
    y_train, y_val, y_test = labels_array[train_idx], labels_array[val_idx], labels_array[test_idx]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
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
    
    all_results = {}
    
    # Hyperparameter grids
    param_grids = {}
    
    if "xgboost" in models and XGBOOST_AVAILABLE:
        param_grids["xgboost"] = {
            "max_depth": [3, 5],  # 2 values (2*2*2*2*2*2*1 = 64 combinations, within 80 limit)
            "learning_rate": [0.01, 0.1],  # 2 values
            "n_estimators": [100, 200],  # 2 values
            "subsample": [0.8, 1.0],  # 2 values
            "colsample_bytree": [0.8, 1.0],  # 2 values
            "reg_alpha": [0, 0.1],  # 2 values
            "reg_lambda": [1.0]  # 1 value (fixed)
        }
    
    if "lightgbm" in models and LIGHTGBM_AVAILABLE:
        param_grids["lightgbm"] = {
            "max_depth": [3, 5],  # 2 values (2*2*2*2*2*2*1 = 64 combinations, within 80 limit)
            "learning_rate": [0.01, 0.1],  # 2 values
            "n_estimators": [100, 200],  # 2 values
            "subsample": [0.8, 1.0],  # 2 values
            "colsample_bytree": [0.8, 1.0],  # 2 values
            "reg_alpha": [0, 0.1],  # 2 values
            "reg_lambda": [1.0]  # 1 value (fixed)
        }
    
    if "catboost" in models and CATBOOST_AVAILABLE:
        param_grids["catboost"] = {
            "depth": [3, 5, 7],  # 3 values (3*2*2*2*2 = 48 combinations, can expand to 80)
            "learning_rate": [0.01, 0.1],  # 2 values
            "iterations": [100, 200],  # 2 values
            "l2_leaf_reg": [1, 3, 5],  # 3 values (3*2*2*3*2 = 72 combinations)
            "border_count": [32, 64]  # 2 values
        }
    
    # Train each model
    for model_name in models:
        if model_name not in param_grids:
            logger.warning(f"Skipping {model_name} (not available or not in param_grids)")
            continue
        
        logger.info("=" * 80)
        logger.info(f"Training {model_name.upper()}")
        logger.info("=" * 80)
        
        model_output_dir = output_dir_path / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Grid search
        grid = list(ParameterGrid(param_grids[model_name]))
        logger.info(f"Hyperparameter search: {len(grid)} combinations")
        
        if len(grid) == 0:
            logger.error(f"No valid parameter combinations for {model_name}")
            all_results[model_name] = {"error": "No valid parameter combinations"}
            continue
        
        best_score = -1
        best_params = None
        best_model = None
        grid_results = []
        
        for param_idx, params in enumerate(grid):
            logger.info(f"Grid search {param_idx + 1}/{len(grid)}: {params}")
            
            try:
                if model_name == "xgboost":
                    model = xgb.XGBClassifier(
                        **params,
                        random_state=42,
                        eval_metric="logloss",
                        use_label_encoder=False
                    )
                elif model_name == "lightgbm":
                    model = lgb.LGBMClassifier(
                        **params,
                        random_state=42,
                        verbose=-1
                    )
                elif model_name == "catboost":
                    model = cb.CatBoostClassifier(
                        **params,
                        random_seed=42,
                        verbose=False
                    )
                else:
                    continue
                
                # Train on 20% sample for hyperparameter search
                model.fit(X_train_sample, y_train_sample)
                
                # Validate on 20% sample
                val_probs_full = model.predict_proba(X_val_sample)
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
            logger.error(f"No valid {model_name} model found during grid search")
            all_results[model_name] = {"error": "No valid model found"}
            continue
        
        logger.info(f"Best hyperparameters from 20% sample: {best_params} (val_f1: {best_score:.4f})")
        
        # FINAL TRAINING: Train on full dataset with best hyperparameters
        logger.info("=" * 80)
        logger.info(f"FINAL TRAINING ({model_name.upper()}): Using full dataset with best hyperparameters")
        logger.info("=" * 80)
        
        # 5-fold CV on full train+val
        X_trainval_full = np.vstack([X_train, X_val])
        y_trainval_full = np.concatenate([y_train, y_val])
        
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        cv_aucs = []
        
        for fold_idx, (train_idx_cv, val_idx_cv) in enumerate(cv.split(X_trainval_full, y_trainval_full)):
            logger.info(f"CV fold {fold_idx + 1}/{n_splits}")
            
            X_train_cv = X_trainval_full[train_idx_cv]
            X_val_cv = X_trainval_full[val_idx_cv]
            y_train_cv = y_trainval_full[train_idx_cv]
            y_val_cv = y_trainval_full[val_idx_cv]
            
            if model_name == "xgboost":
                model_cv = xgb.XGBClassifier(
                    **best_params,
                    random_state=42,
                    eval_metric="logloss",
                    use_label_encoder=False
                )
            elif model_name == "lightgbm":
                model_cv = lgb.LGBMClassifier(
                    **best_params,
                    random_state=42,
                    verbose=-1
                )
            elif model_name == "catboost":
                model_cv = cb.CatBoostClassifier(
                    **best_params,
                    random_seed=42,
                    verbose=False
                )
            
            model_cv.fit(X_train_cv, y_train_cv)
            
            val_probs_cv_full = model_cv.predict_proba(X_val_cv)
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
        
        # Train final model on full train+val
        logger.info("Training final model on full training+validation set...")
        
        if model_name == "xgboost":
            final_model = xgb.XGBClassifier(
                **best_params,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            )
        elif model_name == "lightgbm":
            final_model = lgb.LGBMClassifier(
                **best_params,
                random_state=42,
                verbose=-1
            )
        elif model_name == "catboost":
            final_model = cb.CatBoostClassifier(
                **best_params,
                random_seed=42,
                verbose=False
            )
        
        final_model.fit(X_trainval_full, y_trainval_full)
        
        # Evaluate on test set
        test_probs_full = final_model.predict_proba(X_test)
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
        
        # Save model
        if model_name == "xgboost":
            final_model.save_model(str(model_output_dir / "model.json"))
        elif model_name == "lightgbm":
            joblib.dump(final_model, model_output_dir / "model.joblib")
        elif model_name == "catboost":
            final_model.save_model(str(model_output_dir / "model.cbm"))
        
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
        
        with open(model_output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Also save feature names for reference
        with open(model_output_dir / "feature_names.json", "w") as f:
            json.dump(feature_names, f, indent=2)
        
        # Plot ROC/PR curves
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        precision, recall, _ = precision_recall_curve(y_test, test_probs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(fpr, tpr, label=f'ROC (AUC={test_auc:.4f})')
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'{model_name.upper()} ROC Curve')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(recall, precision, label=f'PR (AP={test_ap:.4f})')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'{model_name.upper()} Precision-Recall Curve')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(model_output_dir / "roc_pr_curves.png", dpi=150)
        plt.close()
        
        all_results[model_name] = results
        logger.info(f"✓ {model_name.upper()} training complete")
    
    logger.info("=" * 80)
    logger.info("GRADIENT BOOSTING TRAINING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Training complete. Results saved to: {output_dir_path}")
    logger.info("=" * 80)
    logger.info("STAGE 5BETA TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    sys.stdout.flush()
    sys.stderr.flush()
    
    return all_results


def main():
    # Log immediately to ensure output is written
    logger.info("Parsing command line arguments...")
    sys.stdout.flush()
    sys.stderr.flush()
    
    parser = argparse.ArgumentParser(description="Train XGBoost, LightGBM, and CatBoost")
    parser.add_argument("--project-root", type=str, required=True)
    parser.add_argument("--scaled-metadata", type=str, required=True)
    parser.add_argument("--features-stage2", type=str, required=True)
    parser.add_argument("--features-stage4", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--models", type=str, nargs="+", default=["xgboost", "lightgbm", "catboost"],
                       help="Models to train: xgboost, lightgbm, catboost")
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
        logger.info(f"  models: {args.models}")
        logger.info(f"  delete-existing: {args.delete_existing}")
        sys.stdout.flush()
        sys.stderr.flush()
    except SystemExit as e:
        logger.error(f"Argument parsing failed: {e}")
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    
    try:
        train_gradient_boosting(
            project_root=args.project_root,
            scaled_metadata_path=args.scaled_metadata,
            features_stage2_path=args.features_stage2,
            features_stage4_path=args.features_stage4,
            output_dir=args.output_dir,
            n_splits=args.n_splits,
            models=args.models,
            delete_existing=args.delete_existing
        )
    except Exception as e:
        logger.critical(f"Training failed with exception: {type(e).__name__}: {e}", exc_info=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
