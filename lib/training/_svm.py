"""
Linear SVM baseline models using Stage 2 and Stage 4 features.

Two versions:
- svm_stage2: Uses only Stage 2 features
- svm_stage2_stage4: Uses Stage 2 + Stage 4 features combined
"""

from __future__ import annotations

import os
import logging
from typing import Optional
import numpy as np
import polars as pl
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib

from lib.training.feature_preprocessing import load_and_combine_features, remove_collinear_features

logger = logging.getLogger(__name__)


class SVMBaseline:
    """
    Linear SVM baseline using Stage 2 and/or Stage 4 features.
    
    Supports two modes:
    - stage2_only: Use only Stage 2 features
    - stage2_stage4: Use Stage 2 + Stage 4 features combined
    """
    
    def __init__(
        self,
        features_stage2_path: Optional[str] = None,
        features_stage4_path: Optional[str] = None,
        use_stage2_only: bool = False,
        cache_dir: Optional[str] = None,
        num_frames: int = 1000
    ):
        """
        Initialize baseline model.
        
        Args:
            features_stage2_path: Path to Stage 2 features metadata (None = extract from videos)
            features_stage4_path: Path to Stage 4 features metadata
            use_stage2_only: If True, use only Stage 2 features; if False, combine Stage 2 + Stage 4
            cache_dir: Directory to cache extracted features (unused, kept for compatibility)
            num_frames: Number of frames to sample (used when extracting from videos)
        """
        self.num_frames = num_frames
        """
        Initialize baseline model.
        
        Args:
            features_stage2_path: Path to Stage 2 features metadata
            features_stage4_path: Path to Stage 4 features metadata
            use_stage2_only: If True, use only Stage 2 features; if False, combine Stage 2 + Stage 4
            cache_dir: Directory to cache extracted features (unused, kept for compatibility)
            num_frames: Number of frames to sample (unused, kept for compatibility)
        """
        self.features_stage2_path = features_stage2_path
        self.features_stage4_path = features_stage4_path
        self.use_stage2_only = use_stage2_only
        self.scaler = StandardScaler()
        self.model = LinearSVC(max_iter=1000, random_state=42)
        self.is_fitted = False
        self.feature_indices = None  # Indices of kept features after collinearity removal
        self.feature_names = None  # Names of kept features
        self.project_root = None  # Store project root for prediction
    
    def fit(self, df: pl.DataFrame, project_root: str) -> None:
        """
        Train the model.
        
        Args:
            df: DataFrame with video_path and label columns
            project_root: Project root directory
        """
        self.project_root = project_root
        
        video_paths = df["video_path"].to_list()
        labels = df["label"].to_list()
        
        # Determine which features to load
        stage2_path = self.features_stage2_path
        stage4_path = None if self.use_stage2_only else self.features_stage4_path
        
        # Validate Stage 2 path if provided (check file exists and is not empty)
        if stage2_path:
            from pathlib import Path
            from lib.utils.paths import load_metadata_flexible
            stage2_path_obj = Path(stage2_path)
            if not stage2_path_obj.exists():
                logger.warning(f"Stage 2 metadata file does not exist: {stage2_path}, falling back to extraction")
                stage2_path = None
            else:
                # Check if file is not empty
                test_df = load_metadata_flexible(stage2_path)
                if test_df is None or test_df.height == 0:
                    logger.warning(f"Stage 2 metadata file is empty: {stage2_path}, falling back to extraction")
                    stage2_path = None
                else:
                    logger.info(f"Using Stage 2 features from: {stage2_path} ({test_df.height} rows)")
        
        # Validate Stage 4 path if provided
        if stage4_path:
            from pathlib import Path
            from lib.utils.paths import load_metadata_flexible
            stage4_path_obj = Path(stage4_path)
            if not stage4_path_obj.exists():
                logger.warning(f"Stage 4 metadata file does not exist: {stage4_path}, using Stage 2 only")
                stage4_path = None
            else:
                # Check if file is not empty
                test_df = load_metadata_flexible(stage4_path)
                if test_df is None or test_df.height == 0:
                    logger.warning(f"Stage 4 metadata file is empty: {stage4_path}, using Stage 2 only")
                    stage4_path = None
        
        # CRITICAL: Never extract features during training - they must already be extracted in Stage 2/4
        if not stage2_path:
            raise ValueError(
                f"Stage 2 features path is required for {self.__class__.__name__}. "
                f"Features should already be extracted in Stage 2. "
                f"Do NOT re-extract features during training. "
                f"Please provide features_stage2_path in model configuration."
            )
        else:
            logger.info(
                f"Loading features for {len(video_paths)} videos "
                f"(Stage 2: {stage2_path}, Stage 4: {stage4_path if stage4_path else 'not used'})..."
            )
            
            # Load and combine features
            # NOTE: Collinearity removal should already be done before splits in the main pipeline
            # We load without removing collinearity here to avoid doing it multiple times
            try:
                features, feature_names, kept_indices = load_and_combine_features(
                    features_stage2_path=stage2_path,
                    features_stage4_path=stage4_path,
                    video_paths=video_paths,
                    project_root=project_root,
                    remove_collinearity=False,  # Already done before splits in main pipeline
                    correlation_threshold=0.95,
                    collinearity_method="correlation"
                )
                logger.info(f"✓ Loaded {len(feature_names)} features (collinearity already removed)")
            except Exception as e:
                logger.error(f"Failed to load features: {e}", exc_info=True)
                logger.error(
                    "Make sure Stage 2/4 features are already extracted. "
                    "Do NOT re-extract features during training."
                )
                raise
        
        # Validate features
        if features is None or features.size == 0:
            raise ValueError("No features loaded from Stage 2/4 metadata")
        if len(features.shape) != 2:
            raise ValueError(f"Invalid feature shape: {features.shape}, expected 2D array")
        if features.shape[0] != len(video_paths):
            raise ValueError(
                f"Feature count mismatch: {features.shape[0]} features for {len(video_paths)} videos"
            )
        
        # Store feature information
        self.feature_indices = kept_indices
        self.feature_names = feature_names
        
        logger.info(f"Using {len(feature_names)} features after collinearity removal")
        
        # Convert labels to binary (0/1)
        label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        if len(label_map) < 2:
            raise ValueError(f"Need at least 2 classes, found {len(label_map)}: {label_map}")
        y = np.array([label_map[label] for label in labels])
        
        # Validate labels
        if len(y) != features.shape[0]:
            raise ValueError(f"Label count mismatch: {len(y)} labels for {features.shape[0]} samples")
        
        # Scale features
        logger.info("Scaling features...")
        try:
            features_scaled = self.scaler.fit_transform(features)
        except Exception as e:
            logger.error(f"Failed to scale features: {e}", exc_info=True)
            raise
        
        # Validate scaled features
        if np.any(np.isnan(features_scaled)) or np.any(np.isinf(features_scaled)):
            logger.warning("NaN or Inf values in scaled features, replacing with 0")
            features_scaled = np.nan_to_num(features_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Train model
        logger.info("Training Linear SVM...")
        try:
            self.model.fit(features_scaled, y)
        except Exception as e:
            logger.error(f"Failed to train Linear SVM: {e}", exc_info=True)
            raise
        
        self.is_fitted = True
        logger.info("✓ Linear SVM trained")
    
    def predict(self, df: pl.DataFrame, project_root: Optional[str] = None) -> np.ndarray:
        """
        Predict labels for videos.
        
        Args:
            df: DataFrame with video_path column
            project_root: Project root directory (uses stored value if None)
        
        Returns:
            Predicted probabilities (n_samples, 2)
            Note: SVM doesn't provide probabilities by default, so we use decision function
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if project_root is None:
            project_root = self.project_root
        if project_root is None:
            raise ValueError("project_root must be provided either in fit() or predict()")
        
        video_paths = df["video_path"].to_list()
        
        # Determine which features to load
        stage2_path = self.features_stage2_path
        stage4_path = None if self.use_stage2_only else self.features_stage4_path
        
        # If no Stage 2 path provided, extract features from videos directly
        if not stage2_path:
            from lib.features.handcrafted import HandcraftedFeatureExtractor
            from lib.utils.memory import aggressive_gc
            
            feature_extractor = HandcraftedFeatureExtractor(
                cache_dir=None,
                num_frames=self.num_frames if hasattr(self, 'num_frames') else 1000,
                include_codec=True
            )
            
            features = feature_extractor.extract_batch(
                video_paths,
                project_root,
                batch_size=1
            )
            
            # Apply same feature filtering as during training
            if self.feature_indices is not None:
                features = features[:, self.feature_indices]
                logger.debug(f"Applied feature filtering: {len(self.feature_indices)} features")
            
            aggressive_gc(clear_cuda=False)
        else:
            # Load and combine features
            features, _, _ = load_and_combine_features(
                features_stage2_path=stage2_path,
                features_stage4_path=stage4_path,
                video_paths=video_paths,
                project_root=project_root,
                remove_collinearity=False  # Don't remove collinearity again, use same features as training
            )
            
            # Apply same feature filtering as during training
            if self.feature_indices is not None:
                features = features[:, self.feature_indices]
                logger.debug(f"Applied feature filtering: {len(self.feature_indices)} features")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Get decision function (distance from hyperplane)
        decision = self.model.decision_function(features_scaled)
        
        # Convert to probabilities using sigmoid
        # This is an approximation since LinearSVC doesn't provide probabilities
        probs_positive = 1 / (1 + np.exp(-decision))
        probs = np.column_stack([1 - probs_positive, probs_positive])
        
        return probs
    
    def save(self, save_dir: str) -> None:
        """Save model and scaler."""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_dir, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
        # Save feature metadata
        import json
        metadata = {
            "feature_indices": self.feature_indices,
            "feature_names": self.feature_names,
            "use_stage2_only": self.use_stage2_only,
            "features_stage2_path": self.features_stage2_path,
            "features_stage4_path": self.features_stage4_path,
        }
        with open(os.path.join(save_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved SVM model to %s", save_dir)
    
    def load(self, load_dir: str) -> None:
        """Load model and scaler."""
        self.model = joblib.load(os.path.join(load_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(load_dir, "scaler.joblib"))
        # Load feature metadata
        import json
        metadata_path = os.path.join(load_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.feature_indices = metadata.get("feature_indices")
            self.feature_names = metadata.get("feature_names")
            self.use_stage2_only = metadata.get("use_stage2_only", False)
            self.features_stage2_path = metadata.get("features_stage2_path")
            self.features_stage4_path = metadata.get("features_stage4_path")
        self.is_fitted = True
        logger.info("Loaded SVM model from %s", load_dir)


__all__ = ["SVMBaseline"]
