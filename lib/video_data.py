"""
Data utilities for the FVC video classifier.

Includes:
- Loading and cleaning the metadata CSV
- Stratified train/val/test splits (with dup_group awareness to avoid leakage)
- K-fold cross-validation splits
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

import os
import numpy as np
import polars as pl

logger = logging.getLogger(__name__)


@dataclass
class SplitConfig:
    """Configuration for dataset splits."""

    val_size: float = 0.2
    test_size: float = 0.1
    random_state: int = 42


def load_metadata(csv_path: str) -> pl.DataFrame:
    """Load the video metadata CSV using Polars."""
    df = pl.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    return df


def filter_existing_videos(df: pl.DataFrame, project_root: str, check_frames: bool = False) -> pl.DataFrame:
    """
    Filter DataFrame to only include rows where video files exist.
    
    Optionally checks that videos have frames (slower, but prevents runtime errors).
    By default, only checks file existence for speed. Set check_frames=True to
    also validate that videos have at least 1 frame.
    
    Uses centralized path resolution from video_paths module.
    
    Args:
        df: Polars DataFrame with 'video_path' column
        project_root: Root directory for resolving relative paths
        check_frames: If True, also check that videos have frames (slower, memory-intensive)
        
    Returns:
        Filtered DataFrame with only existing (and optionally non-empty) videos
        
    Raises:
        ValueError: If no videos exist after filtering
    """
    from .video_paths import check_video_path_exists, get_video_path_candidates, resolve_video_path
    
    def _check_path(video_rel: str) -> bool:
        """Check if video file exists."""
        if video_rel is None or (isinstance(video_rel, float) and np.isnan(video_rel)):
            return False
        return check_video_path_exists(str(video_rel).strip(), project_root)
    
    # First pass: check file existence (fast)
    try:
        existing_mask = df["video_path"].map_elements(_check_path, return_dtype=pl.Boolean)
    except (AttributeError, TypeError):
        # Fallback for older Polars versions
        existing_mask = df["video_path"].map(_check_path)
    
    filtered = df.filter(existing_mask)
    
    # Second pass: optionally check for frames (slower, but prevents runtime errors)
    if check_frames:
        from .video_modeling import _read_video_wrapper
        import gc
        
        def _check_has_frames(video_rel: str) -> bool:
            """Check if video has frames."""
            try:
                video_path = resolve_video_path(str(video_rel).strip(), project_root)
                # Read video with minimal memory footprint
                video = _read_video_wrapper(video_path)
                has_frames = video.shape[0] > 0
                # Clean up immediately
                del video
                gc.collect()  # Force garbage collection
                return has_frames
            except Exception:
                return False
        
        logger.info("Checking video frames (this may take a while and use memory)...")
        try:
            frame_mask = filtered["video_path"].map_elements(_check_has_frames, return_dtype=pl.Boolean)
        except (AttributeError, TypeError):
            frame_mask = filtered["video_path"].map(_check_has_frames)
        
        filtered = filtered.filter(frame_mask)
        logger.info("Frame check complete.")
    
    # Log how many videos were filtered out
    filtered_count = df.height - filtered.height
    if filtered_count > 0:
        logger.warning(
            "Filtered out %d invalid videos (missing files%s). Keeping %d valid videos.",
            filtered_count,
            " or empty videos" if check_frames else "",
            filtered.height
        )
    
    # Validate that we have videos after filtering
    if filtered.height == 0:
        # Try to find at least one example path to show in error message
        from .video_paths import get_video_path_candidates
        sample_paths = []
        if df.height > 0:
            sample_row = df.row(0, named=True)
            video_rel = sample_row.get("video_path", "")
            if video_rel:
                candidates = get_video_path_candidates(str(video_rel), project_root)
                sample_paths.append(f"  Example: '{video_rel}' -> tried:")
                for candidate in candidates:
                    sample_paths.append(f"    - {candidate}")
        
        error_msg = (
            f"No existing video files found after filtering. "
            f"Original dataset had {df.height} videos. "
            f"Project root: {project_root}\n"
        )
        if sample_paths:
            error_msg += "\n".join(sample_paths)
        raise ValueError(error_msg)
    
    return filtered


def _build_strat_key(df: pl.DataFrame) -> pl.Series:
    """Build a stratification key on 'label' and optional 'platform'."""
    key = df["label"].cast(str)
    if "platform" in df.columns:
        key = key + "__" + df["platform"].cast(str)
    return key


def train_val_test_split(
    df: pl.DataFrame,
    cfg: SplitConfig,
    save_dir: Optional[str] = None,
) -> Dict[str, pl.DataFrame]:
    """
    Stratified train/val/test split.

    If 'dup_group' exists, we group by dup_group first to avoid leaking near-duplicates
    across splits, then perform stratified split on group-level labels.
    """
    rng = np.random.default_rng(cfg.random_state)

    if "dup_group" in df.columns:
        # Work at dup_group level to keep near-duplicates together.
        group_df = df.select(["dup_group", "label", *([c for c in df.columns if c == "platform"])]) \
            .unique(subset=["dup_group"])
        strat_key = _build_strat_key(group_df).to_numpy()

        groups = group_df["dup_group"].to_numpy()
        n_groups = len(groups)

        # Shuffle indices
        indices = np.arange(n_groups)
        rng.shuffle(indices)

        # Split into train/val/test based on stratification key proportions.
        # Approximate stratification by slicing within each key group.
        train_mask = np.zeros(n_groups, dtype=bool)
        val_mask = np.zeros(n_groups, dtype=bool)
        test_mask = np.zeros(n_groups, dtype=bool)

        for key in np.unique(strat_key):
            key_idx = indices[strat_key[indices] == key]
            n = len(key_idx)
            n_test = int(round(n * cfg.test_size))
            n_val = int(round(n * cfg.val_size))
            n_train = n - n_test - n_val
            train_mask[key_idx[:n_train]] = True
            val_mask[key_idx[n_train : n_train + n_val]] = True
            test_mask[key_idx[n_train + n_val : n_train + n_val + n_test]] = True

        train_groups = groups[train_mask]
        val_groups = groups[val_mask]
        test_groups = groups[test_mask]

        train_df = df.filter(pl.col("dup_group").is_in(train_groups.tolist()))
        val_df = df.filter(pl.col("dup_group").is_in(val_groups.tolist()))
        test_df = df.filter(pl.col("dup_group").is_in(test_groups.tolist()))
    else:
        strat_key = _build_strat_key(df).to_numpy()
        n = df.height
        indices = np.arange(n)
        rng.shuffle(indices)

        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)

        for key in np.unique(strat_key):
            key_idx = indices[strat_key[indices] == key]
            m = len(key_idx)
            n_test = int(round(m * cfg.test_size))
            n_val = int(round(m * cfg.val_size))
            n_train = m - n_test - n_val
            train_mask[key_idx[:n_train]] = True
            val_mask[key_idx[n_train : n_train + n_val]] = True
            test_mask[key_idx[n_train + n_val : n_train + n_val + n_test]] = True

        train_df = df[train_mask.tolist()]
        val_df = df[val_mask.tolist()]
        test_df = df[test_mask.tolist()]

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    # Optionally persist splits to Arrow/Feather for reproducibility and reuse.
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for name, split_df in splits.items():
            path = os.path.join(save_dir, "{}.feather".format(name))
            # Polars uses write_ipc() for Arrow/Feather format
            split_df.write_ipc(path)

    return splits


def stratified_kfold(
    df: pl.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Stratified K-fold splits on labels.

    Returns a list of (train_df, val_df) for each fold.
    """
    key = _build_strat_key(df).to_numpy()
    n = df.height
    indices = np.arange(n)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)

    # Distribute indices to folds in a stratified round-robin manner.
    folds_indices: List[List[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(key):
        label_idx = indices[key[indices] == label]
        for i, idx in enumerate(label_idx):
            folds_indices[i % n_splits].append(int(idx))

    # Ensure every fold has at least one validation sample.
    # For very small datasets (e.g., tiny test mode) the initial stratified
    # assignment can leave some folds empty; we fix that here by borrowing
    # one index from the largest folds.
    fold_sizes = [len(fi) for fi in folds_indices]
    for k in range(n_splits):
        if fold_sizes[k] == 0:
            # Find a donor fold with at least 2 samples to spare
            donor = max(
                (i for i in range(n_splits) if fold_sizes[i] > 1),
                key=lambda i: fold_sizes[i],
                default=None,
            )
            if donor is not None:
                moved_idx = folds_indices[donor].pop()
                folds_indices[k].append(moved_idx)
                fold_sizes[donor] -= 1
                fold_sizes[k] += 1
    
    folds: List[Tuple[pl.DataFrame, pl.DataFrame]] = []
    for k in range(n_splits):
        val_idx = np.array(folds_indices[k], dtype=int)
        train_idx = np.setdiff1d(indices, val_idx)
        train_df = df[train_idx.tolist()]
        val_df = df[val_idx.tolist()]
        
        # Verify balanced splits and log warnings if imbalanced
        if "label" in train_df.columns:
            train_label_counts = train_df["label"].value_counts().sort_index()
            val_label_counts = val_df["label"].value_counts().sort_index()
            
            # Check if all classes are present in both train and val
            train_labels = set(train_label_counts.keys().to_list())
            val_labels = set(val_label_counts.keys().to_list())
            all_labels = train_labels | val_labels
            
            if len(all_labels) > 1:  # Binary or multi-class
                # Calculate class balance ratios
                train_total = train_df.height
                val_total = val_df.height
                
                for label in all_labels:
                    train_count = train_label_counts.get(label, 0)
                    val_count = val_label_counts.get(label, 0)
                    
                    train_ratio = train_count / train_total if train_total > 0 else 0.0
                    val_ratio = val_count / val_total if val_total > 0 else 0.0
                    
                    # Warn if class is missing or severely imbalanced
                    if train_count == 0:
                        logger.warning(
                            "Fold %d: Class %d missing in training set (val has %d)",
                            k + 1, label, val_count
                        )
                    elif val_count == 0:
                        logger.warning(
                            "Fold %d: Class %d missing in validation set (train has %d)",
                            k + 1, label, train_count
                        )
                    elif abs(train_ratio - val_ratio) > 0.2:  # More than 20% difference
                        logger.warning(
                            "Fold %d: Class %d imbalance - train: %.1f%%, val: %.1f%%",
                            k + 1, label, train_ratio * 100, val_ratio * 100
                        )
        
        folds.append((train_df, val_df))
    
    return folds


def maybe_limit_to_small_test_subset(
    df: pl.DataFrame,
    max_per_class: int = 5,
    env_var: str = "FVC_TEST_MODE",
) -> pl.DataFrame:
    """
    Optionally limit the dataset to a very small, balanced subset for test runs.

    When the environment variable specified by ``env_var`` is set to a truthy value
    (\"1\", \"true\", \"yes\", \"y\" - case-insensitive), this function returns a
    subset containing at most ``max_per_class`` samples per class (based on the
    'label' column). This is intended for quick end-to-end sanity checks on SLURM.

    Args:
        df: Input Polars DataFrame (must contain a 'label' column for class labels)
        max_per_class: Maximum number of samples to keep per class
        env_var: Name of the environment variable that enables test mode

    Returns:
        Either the original DataFrame (if test mode is disabled or label column
        is missing) or a reduced, balanced subset.
    """
    flag = os.environ.get(env_var, "").strip().lower()
    if flag not in ("1", "true", "yes", "y"):
        return df

    if "label" not in df.columns:
        logger.warning(
            "Test mode enabled via %s, but 'label' column is missing. "
            "Skipping dataset downsampling.",
            env_var,
        )
        return df

    logger.info(
        "Test mode enabled via %s. Limiting dataset to at most %d samples per class.",
        env_var,
        max_per_class,
    )

    subsets = []
    unique_labels = df["label"].unique().to_list()
    for label_val in unique_labels:
        sub = df.filter(pl.col("label") == label_val).head(max_per_class)
        if sub.height > 0:
            subsets.append(sub)

    if not subsets:
        logger.warning(
            "Test mode enabled but no samples were selected. Returning original DataFrame."
        )
        return df

    limited = pl.concat(subsets)
    logger.info(
        "Test mode dataset size: %d rows (labels: %s)",
        limited.height,
        [str(v) for v in unique_labels],
    )
    return limited

def make_balanced_batch_sampler(
    df: pl.DataFrame,
    batch_size: int,
    samples_per_class: Optional[int] = None,
    shuffle: bool = True,
    random_state: int = 42,
):
    """
    Create a sampler that ensures balanced batches (equal number of each class per batch).
    
    This is more memory-efficient than gradient accumulation for ensuring class balance,
    as it guarantees balanced batches without needing to accumulate gradients.
    
    Args:
        df: Polars DataFrame with 'label' column
        batch_size: Total batch size (must be even for binary classification)
        samples_per_class: Number of samples per class per batch (defaults to batch_size // 2)
        shuffle: Whether to shuffle within each class
        random_state: Random seed
        
    Returns:
        A sampler that yields balanced batches
        
    Raises:
        ValueError: If batch_size is odd or not enough samples per class
    """
    import torch
    from torch.utils.data import Sampler
    
    if "label" not in df.columns:
        raise ValueError("DataFrame must have 'label' column")
    
    # Get indices for each class
    labels = df["label"].to_numpy()
    class_0_indices = np.where(labels == 0)[0].tolist()
    class_1_indices = np.where(labels == 1)[0].tolist()
    
    if samples_per_class is None:
        samples_per_class = batch_size // 2
    
    if batch_size % 2 != 0:
        raise ValueError(f"batch_size must be even for balanced sampling, got {batch_size}")
    
    # Adapt to tiny datasets: clamp samples_per_class to what is actually available
    # instead of failing entirely. This avoids warnings like:
    # "Not enough samples: need 8 per class, but have 1 class 0 and 2 class 1."
    max_possible = min(len(class_0_indices), len(class_1_indices))
    if max_possible == 0:
        raise ValueError("No samples available for at least one class; cannot build balanced sampler.")
    if samples_per_class > max_possible:
        logger = logging.getLogger(__name__)
        logger.info(
            "Reducing samples_per_class from %d to %d for balanced sampling "
            "(available: %d class 0, %d class 1).",
            samples_per_class,
            max_possible,
            len(class_0_indices),
            len(class_1_indices),
        )
        samples_per_class = max_possible
    
    class BalancedBatchSampler(Sampler):
        def __init__(self, class_0_idx, class_1_idx, samples_per_class, shuffle, random_state):
            self.class_0_idx = class_0_idx
            self.class_1_idx = class_1_idx
            self.samples_per_class = samples_per_class
            self.shuffle = shuffle
            self.rng = np.random.default_rng(random_state)
            
            # Calculate number of batches
            min_class_size = min(len(class_0_idx), len(class_1_idx))
            self.num_batches = min_class_size // samples_per_class
        
        def __iter__(self):
            # Shuffle indices for each class
            class_0_shuffled = self.class_0_idx.copy()
            class_1_shuffled = self.class_1_idx.copy()
            
            if self.shuffle:
                self.rng.shuffle(class_0_shuffled)
                self.rng.shuffle(class_1_shuffled)
            
            # Generate balanced batches
            for batch_idx in range(self.num_batches):
                start_0 = batch_idx * self.samples_per_class
                end_0 = start_0 + self.samples_per_class
                start_1 = batch_idx * self.samples_per_class
                end_1 = start_1 + self.samples_per_class
                
                batch_indices = (
                    class_0_shuffled[start_0:end_0] + 
                    class_1_shuffled[start_1:end_1]
                )
                
                # Shuffle the batch to mix classes
                if self.shuffle:
                    self.rng.shuffle(batch_indices)
                
                # DataLoader expects batch_sampler to yield a list/sequence of indices
                yield batch_indices
        
        def __len__(self):
            return self.num_batches * (self.samples_per_class * 2)
    
    return BalancedBatchSampler(
        class_0_indices, class_1_indices, samples_per_class, shuffle, random_state
    )


__all__ = [
    "SplitConfig",
    "load_metadata",
    "filter_existing_videos",
    "train_val_test_split",
    "stratified_kfold",
    "make_balanced_batch_sampler",
]


