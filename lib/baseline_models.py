"""
Baseline models for video classification:
- Logistic Regression on handcrafted features
- Linear SVM on handcrafted features
- Naive CNN over uniformly sampled frames
"""

from __future__ import annotations

import os
import logging
from typing import Optional
import numpy as np
import polars as pl
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import joblib

from .handcrafted_features import HandcraftedFeatureExtractor

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline using handcrafted features.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, num_frames: int = 8):
        """
        Initialize baseline model.
        
        Args:
            cache_dir: Directory to cache extracted features
            num_frames: Number of frames to sample per video
        """
        self.feature_extractor = HandcraftedFeatureExtractor(cache_dir, num_frames)
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, project_root: str) -> None:
        """
        Train the model.
        
        Args:
            df: DataFrame with video_path and label columns
            project_root: Project root directory
        """
        logger.info("Extracting handcrafted features for %d videos...", df.height)
        
        video_paths = df["video_path"].to_list()
        labels = df["label"].to_list()
        
        # Extract features (ultra conservative batch size for OOM safety)
        from .mlops_utils import aggressive_gc
        features = self.feature_extractor.extract_batch(
            video_paths,
            project_root,
            batch_size=3,  # Ultra conservative: reduced from 5
        )
        
        # Aggressive GC after feature extraction
        aggressive_gc(clear_cuda=False)
        
        # Convert labels to binary (0/1)
        label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        y = np.array([label_map[label] for label in labels])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        logger.info("Training Logistic Regression...")
        self.model.fit(features_scaled, y)
        self.is_fitted = True
        
        logger.info("✓ Logistic Regression trained")
    
    def predict(self, df: pl.DataFrame, project_root: str) -> np.ndarray:
        """
        Predict labels for videos.
        
        Args:
            df: DataFrame with video_path column
            project_root: Project root directory
        
        Returns:
            Predicted probabilities (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        video_paths = df["video_path"].to_list()
        
        # Extract features (ultra conservative batch size for OOM safety)
        from .mlops_utils import aggressive_gc
        features = self.feature_extractor.extract_batch(
            video_paths,
            project_root,
            batch_size=3,  # Ultra conservative: reduced from 5
        )
        
        # Aggressive GC after feature extraction
        aggressive_gc(clear_cuda=False)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Predict probabilities
        probs = self.model.predict_proba(features_scaled)
        
        return probs
    
    def save(self, save_dir: str) -> None:
        """Save model and scaler."""
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(self.model, os.path.join(save_dir, "model.joblib"))
        joblib.dump(self.scaler, os.path.join(save_dir, "scaler.joblib"))
        logger.info("Saved Logistic Regression model to %s", save_dir)
    
    def load(self, load_dir: str) -> None:
        """Load model and scaler."""
        self.model = joblib.load(os.path.join(load_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(load_dir, "scaler.joblib"))
        self.is_fitted = True
        logger.info("Loaded Logistic Regression model from %s", load_dir)


class SVMBaseline:
    """
    Linear SVM baseline using handcrafted features.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, num_frames: int = 8):
        """
        Initialize baseline model.
        
        Args:
            cache_dir: Directory to cache extracted features
            num_frames: Number of frames to sample per video
        """
        self.feature_extractor = HandcraftedFeatureExtractor(cache_dir, num_frames)
        self.scaler = StandardScaler()
        self.model = LinearSVC(max_iter=1000, random_state=42)
        self.is_fitted = False
    
    def fit(self, df: pl.DataFrame, project_root: str) -> None:
        """
        Train the model.
        
        Args:
            df: DataFrame with video_path and label columns
            project_root: Project root directory
        """
        logger.info("Extracting handcrafted features for %d videos...", df.height)
        
        video_paths = df["video_path"].to_list()
        labels = df["label"].to_list()
        
        # Extract features (ultra conservative batch size for OOM safety)
        from .mlops_utils import aggressive_gc
        features = self.feature_extractor.extract_batch(
            video_paths,
            project_root,
            batch_size=3,  # Ultra conservative: reduced from 5
        )
        
        # Aggressive GC after feature extraction
        aggressive_gc(clear_cuda=False)
        
        # Convert labels to binary (0/1)
        label_map = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        y = np.array([label_map[label] for label in labels])
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train model
        logger.info("Training Linear SVM...")
        self.model.fit(features_scaled, y)
        self.is_fitted = True
        
        logger.info("✓ Linear SVM trained")
    
    def predict(self, df: pl.DataFrame, project_root: str) -> np.ndarray:
        """
        Predict labels for videos.
        
        Args:
            df: DataFrame with video_path column
            project_root: Project root directory
        
        Returns:
            Predicted probabilities (n_samples, 2)
            Note: SVM doesn't provide probabilities by default, so we use decision function
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        video_paths = df["video_path"].to_list()
        
        # Extract features
        features = self.feature_extractor.extract_batch(
            video_paths,
            project_root,
            batch_size=10
        )
        
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
        logger.info("Saved SVM model to %s", save_dir)
    
    def load(self, load_dir: str) -> None:
        """Load model and scaler."""
        self.model = joblib.load(os.path.join(load_dir, "model.joblib"))
        self.scaler = joblib.load(os.path.join(load_dir, "scaler.joblib"))
        self.is_fitted = True
        logger.info("Loaded SVM model from %s", load_dir)


class NaiveCNNBaseline(nn.Module):
    """
    Naive CNN baseline that processes frames independently and averages predictions.
    """
    
    def __init__(self, num_frames: int = 8, num_classes: int = 2):
        """
        Initialize CNN model.
        
        Args:
            num_frames: Number of frames to process
            num_classes: Number of classes (2 for binary)
        """
        super().__init__()
        
        # Simple 2D CNN for per-frame processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)
        
        self.num_frames = num_frames
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (N, C, T, H, W) or (N, T, C, H, W)
        
        Returns:
            Logits (N, num_classes)
        """
        # Handle different input formats
        if x.dim() == 5:
            if x.shape[1] == 3:  # (N, C, T, H, W)
                # Rearrange to (N*T, C, H, W) for per-frame processing
                N, C, T, H, W = x.shape
                x = x.permute(0, 2, 1, 3, 4).contiguous()  # (N, T, C, H, W)
                x = x.view(N * T, C, H, W)
            else:  # (N, T, C, H, W)
                N, T, C, H, W = x.shape
                x = x.view(N * T, C, H, W)
        else:
            # Already in (N*T, C, H, W) format
            pass
        
        # Process each frame independently
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classification
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)  # (N*T, num_classes)
        
        # Reshape back to (N, T, num_classes) and average over frames
        if x.dim() == 2 and logits.shape[0] % self.num_frames == 0:
            N = logits.shape[0] // self.num_frames
            logits = logits.view(N, self.num_frames, -1)
            logits = logits.mean(dim=1)  # Average over frames
        
        return logits


__all__ = [
    "LogisticRegressionBaseline",
    "SVMBaseline",
    "NaiveCNNBaseline",
]

