"""
Training utilities for the FVC video classifier.

Includes:
- Optimizer / scheduler builders
- Layer freezing / unfreezing helpers
- Class-imbalance handling (class weights, WeightedRandomSampler)
- Standard training / evaluation loops
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import logging
import os
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from .video_modeling import VariableARVideoModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Early stopping / checkpointing
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Early stops the training if validation metric doesn't improve."""

    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode  # "max" for metrics like accuracy/F1, "min" for loss
        self.best: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, value: float) -> None:
        if self.best is None:
            self.best = value
            self.counter = 0
            return

        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


class CheckpointManager:
    """Saves best model checkpoints to disk."""

    def __init__(self, directory: Optional[str]) -> None:
        self.directory = directory
        self.best_metric: Optional[float] = None

    def _path(self) -> Optional[str]:
        if self.directory is None:
            return None
        return f"{self.directory.rstrip('/')}/best_model.pt"

    def maybe_save(self, model: nn.Module, metric: float) -> None:
        path = self._path()
        if path is None:
            return
        if self.best_metric is None or metric > self.best_metric:
            self.best_metric = metric
            os.makedirs(self.directory, exist_ok=True)  # type: ignore[arg-type]
            torch.save(model.state_dict(), path)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OptimConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class TrainConfig:
    num_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    use_class_weights: bool = True
    use_amp: bool = True
    checkpoint_dir: Optional[str] = None
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1  # Accumulate gradients over N batches before updating
    # For severe imbalance you might prefer focal loss; here we keep BCEWithLogitsLoss
    # but you can plug in your own criterion if needed.


# ---------------------------------------------------------------------------
# Freezing / unfreezing
# ---------------------------------------------------------------------------


def freeze_all(model: nn.Module) -> None:
    """Freeze all parameters (no gradients)."""
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters (enable gradients)."""
    for p in model.parameters():
        p.requires_grad = True


def freeze_backbone_unfreeze_head(model: VariableARVideoModel) -> None:
    """
    Typical transfer-learning behavior:
    - Freeze stem and Inception blocks
    - Keep final FC trainable
    """
    for m in [model.stem, model.incept1, model.incept2]:
        for p in m.parameters():
            p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True


def trainable_params(model: nn.Module) -> Iterable[torch.nn.Parameter]:
    """Return an iterator over parameters that require gradients."""
    return (p for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Class imbalance handling
# ---------------------------------------------------------------------------


def compute_class_counts(loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Compute how many samples per class appear in the loader."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, labels in loader:
        # labels are assumed to be integer class indices
        for c in labels.view(-1):
            counts[c] += 1
    return counts


def make_class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Inverse-frequency class weights."""
    counts = counts.float()
    # Avoid division by zero
    counts[counts == 0] = 1.0
    inv_freq = 1.0 / counts
    return inv_freq / inv_freq.sum() * len(counts)


def make_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler given a vector of labels.
    Each sample is weighted by inverse class frequency.
    """
    unique, counts = labels.unique(return_counts=True)
    class_weights = make_class_weights(counts)
    weight_map: Dict[int, float] = {
        int(c.item()): float(w.item()) for c, w in zip(unique, class_weights)
    }
    sample_weights = torch.tensor([weight_map[int(l.item())] for l in labels])
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights))


# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------


def build_optimizer(model: nn.Module, cfg: OptimConfig) -> Optimizer:
    return torch.optim.Adam(
        trainable_params(model),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=cfg.betas,
    )


def build_scheduler(optimizer: Optimizer, step_size: int = 10, gamma: float = 0.1) -> _LRScheduler:
    """Simple step scheduler."""
    return StepLR(optimizer, step_size=step_size, gamma=gamma)


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    device: str,
    use_class_weights: bool = True,
    use_amp: bool = True,
    epoch: int = 0,
    log_interval: int = 10,
    gradient_accumulation_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    
    # Clear CUDA cache at start of epoch (less frequent with fixed-size videos)
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        # Log GPU memory info for monitoring
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.debug("GPU memory at epoch start: %.2f GB allocated, %.2f GB reserved, %.2f GB total", 
                        allocated, reserved, total)
    
    # Log model parameter info for debugging
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d total params, %d trainable", total_params, trainable_params)

    # Loss criterion will be inferred on the first batch based on BOTH
    # label distribution and model output shape:
    # - If logits are 1D: binary BCEWithLogitsLoss
    # - If logits are 2D with C>1: multi-class CrossEntropyLoss
    first_batch = True
    criterion: Optional[nn.Module] = None

    # Use new API if available, otherwise fall back to deprecated API
    if use_amp and device.startswith("cuda"):
        try:
            # New API (PyTorch 2.0+)
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            # Fallback to old API for older PyTorch versions
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Initialize gradient tracking variables
    grad_norm_after_backward = 0.0
    grad_count_after_backward = 0
    
    for batch_idx, (clips, labels) in enumerate(loader):
        # Aggressive GC every 5 batches to prevent OOM
        if batch_idx > 0 and batch_idx % 5 == 0:
            import gc
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        clips = clips.to(device)
        labels = labels.to(device)

        # Only zero gradients at the start of accumulation cycle
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Debug: Check if inputs are actually different (first batch only)
        if batch_idx == 0:
            with torch.no_grad():
                # Check if clips are different
                clip_std = clips.std().item()
                clip_mean = clips.mean().item()
                logger.info(
                    "First batch input stats: mean=%.3f, std=%.3f, shape=%s",
                    clip_mean, clip_std, list(clips.shape)
                )
                if clip_std < 1e-6:
                    logger.error("⚠ CRITICAL: All input clips are identical! Check data loading.")

        # Use new autocast API if available, otherwise fall back to deprecated API
        if use_amp and device.startswith("cuda"):
            try:
                # New API (PyTorch 2.0+) - test if available
                _ = torch.amp.autocast('cuda')
                use_new_autocast = True
            except (AttributeError, TypeError):
                # Fallback to old API for older PyTorch versions
                use_new_autocast = False
        else:
            use_new_autocast = None  # No AMP
        
        # Apply autocast based on availability
        if use_new_autocast is True:
            # New API
            with torch.amp.autocast('cuda'):
                logits = model(clips)
        elif use_new_autocast is False:
            # Old API
            with torch.cuda.amp.autocast():
                logits = model(clips)
        else:
            # No AMP
            logits = model(clips)

        # Derive criterion once, after seeing logits shape
        if first_batch:
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                # Binary classification with scalar logits → BCEWithLogitsLoss
                if logits.ndim == 2:
                    logits = logits.squeeze(-1)
                if use_class_weights:
                    # pos_weight is ratio of negative to positive examples
                    pos_count = (labels == 1).sum().float()
                    neg_count = (labels == 0).sum().float()
                    if pos_count == 0 or neg_count == 0:
                        pos_weight = torch.tensor(1.0, device=device)
                        logger.warning(
                            "⚠ First batch has only one class (pos=%d, neg=%d). "
                            "pos_weight set to 1.0. This may cause training issues.",
                            int(pos_count.item()),
                            int(neg_count.item()),
                        )
                    else:
                        pos_weight = neg_count / (pos_count + 1e-6)
                    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    logger.info(
                        "BCEWithLogitsLoss with pos_weight=%.3f (pos=%d, neg=%d)",
                        pos_weight.item(),
                        int(pos_count.item()),
                        int(neg_count.item()),
                    )
                else:
                    criterion = nn.BCEWithLogitsLoss()
                    logger.info("BCEWithLogitsLoss (no class weights)")
            else:
                # Multi-class (including 2-logit case like naive_cnn)
                num_classes = logits.shape[1] if logits.ndim > 1 else int(labels.max().item() + 1)
                if use_class_weights:
                    with torch.no_grad():
                        # Compute counts over single batch as an approximation
                        counts = torch.zeros(num_classes, device=device)
                        for c in labels.view(-1):
                            counts[c] += 1
                        class_weights = make_class_weights(counts.cpu()).to(device)
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                    logger.info(
                        "CrossEntropyLoss with class weights: %s",
                        class_weights.cpu().tolist(),
                    )
                else:
                    criterion = nn.CrossEntropyLoss()
                    logger.info("CrossEntropyLoss (no class weights)")
            first_batch = False

        # If using BCE, ensure logits are 1D; if using CE, ensure 2D
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)
        else:
            # CrossEntropyLoss expects (N, C)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)

        # Compute loss (same for all cases)
        # Initialize unscaled_loss_value for all code paths
        unscaled_loss_value = 0.0
        
        if logits.ndim == 1:
            # Binary classification
            targets = labels.float()
            
            # Check for single-class batch (all same label)
            unique_labels = torch.unique(labels)
            if len(unique_labels) == 1:
                logger.warning(
                    "⚠ Batch %d has only one class (label=%d)! "
                    "This prevents the model from learning class boundaries. "
                    "Consider: 1) Increasing batch_size, 2) Using weighted sampling, "
                    "3) Using gradient accumulation.",
                    batch_idx + 1, int(unique_labels[0].item())
                )
            
            # Check for identical logits (model not learning)
            # Handle batch_size=1 case
            if batch_idx == 0:
                if logits.numel() > 1:
                    logits_std = logits.std().item()
                    if logits_std < 1e-6:
                        logger.error(
                            "⚠ CRITICAL: All logits are identical (std=%.6f)! "
                            "Model is not learning. Check model initialization and gradients.",
                            logits_std
                        )
                else:
                    logger.warning(
                        "⚠ Batch size is 1 - cannot compute logit variance. "
                        "This severely limits learning. Consider increasing batch_size or using gradient accumulation."
                    )
            
            loss = criterion(logits, targets)  # type: ignore[arg-type]
            
            # ALWAYS compute manual loss as the source of truth (more reliable)
            # The criterion loss can sometimes have numerical issues
            with torch.no_grad():
                probs_verify = torch.sigmoid(logits)
                manual_loss_verify = float(
                    -(targets * torch.log(probs_verify + 1e-8) + 
                      (1 - targets) * torch.log(1 - probs_verify + 1e-8)).mean().item()
                )
            
            # Use manual computation as the primary value, criterion as backup
            unscaled_loss_value = manual_loss_verify
            
            # Log if there's a significant mismatch (for debugging)
            criterion_loss_value = float(loss.item())
            if abs(criterion_loss_value - manual_loss_verify) > 0.1:
                logger.warning(
                    "Loss mismatch: criterion=%.6f, manual=%.6f. Using manual computation.",
                    criterion_loss_value, manual_loss_verify
                )
            
            # Scale loss by accumulation steps for gradient accumulation
            # Use the criterion loss for backprop (it has gradients), but log the manual value
            loss = loss / gradient_accumulation_steps
        else:
            # Multi-class (not typical for this project, but supported)
            loss = criterion(logits, labels)  # type: ignore[arg-type]
            unscaled_loss_value = float(loss.item())
            loss = loss / gradient_accumulation_steps

        try:
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Check gradients immediately after backward() (before optimizer step)
            # This is critical for gradient accumulation - we need to check before stepping
            grad_norm_after_backward = 0.0
            grad_count_after_backward = 0
            has_nan_grad = False
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_count_after_backward += 1
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            has_nan_grad = True
                        else:
                            grad_norm_after_backward += p.grad.data.norm().item() ** 2
            grad_norm_after_backward = grad_norm_after_backward ** 0.5 if grad_norm_after_backward > 0 else 0.0
            
            if has_nan_grad:
                logger.error("⚠ NaN or Inf gradients detected! Skipping optimizer step.")
                if scaler is not None:
                    scaler.update()  # Update scaler even if we skip step
                else:
                    optimizer.zero_grad()  # Clear gradients
                continue
            
            # Only step optimizer at the end of accumulation cycle
            if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(loader):
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                # Ultra aggressive GC after optimizer step to free memory immediately
                from .mlops_utils import aggressive_gc
                aggressive_gc(clear_cuda=device.startswith("cuda"))
        except RuntimeError as e:
            error_str = str(e).lower()
            if any(oom_indicator in error_str for oom_indicator in 
                   ["out of memory", "cuda out of memory", "oom", "allocation failed"]):
                logger.error("CUDA OOM at batch %d. Performing ultra aggressive cleanup...", batch_idx + 1)
                
                # Use ultra aggressive GC from mlops_utils
                from .mlops_utils import aggressive_gc
                aggressive_gc(clear_cuda=True)
                
                # Log memory stats
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(0) / 1e9
                    reserved = torch.cuda.memory_reserved(0) / 1e9
                    logger.error("GPU Memory after cleanup: %.2f GB allocated, %.2f GB reserved", 
                               allocated, reserved)
                
                # Skip this batch
                continue
            else:
                raise
        
        # Use the unscaled loss value we stored earlier (before dividing by accumulation_steps)
        loss_value = unscaled_loss_value
        
        # Debug: Log the actual values to diagnose the issue
        if (batch_idx + 1) % log_interval == 0:
            logger.debug(
                "DEBUG: unscaled_loss_value=%.6f, loss_value=%.6f, gradient_accumulation_steps=%d",
                unscaled_loss_value, loss_value, gradient_accumulation_steps
            )
        
        # Verify loss_value is reasonable (use abs() check instead of == 0.0)
        # Note: loss_value should already be from manual computation, but double-check
        if abs(loss_value) < 1e-6 and batch_idx > 0:
            logger.error(
                "⚠ CRITICAL: loss_value is suspiciously small (%.6f)! unscaled_loss_value=%.6f. "
                "This should not happen. Recomputing loss manually...",
                loss_value, unscaled_loss_value
            )
            # Fallback: compute loss manually for this batch (shouldn't be needed, but safety check)
            with torch.no_grad():
                if logits.ndim == 1:
                    targets_fallback = labels.float()
                    probs_fallback = torch.sigmoid(logits)
                    manual_loss_fallback = float(
                        -(targets_fallback * torch.log(probs_fallback + 1e-8) + 
                          (1 - targets_fallback) * torch.log(1 - probs_fallback + 1e-8)).mean().item()
                    )
                    logger.error("Using manual loss computation as fallback: %.6f", manual_loss_fallback)
                    loss_value = manual_loss_fallback
                    unscaled_loss_value = manual_loss_fallback  # Update this too for consistency
        
        # Check for negative loss (shouldn't happen with BCEWithLogitsLoss)
        if loss_value < 0:
            logger.error(
                "⚠ CRITICAL: Negative loss detected (%.6f)! This should not happen with BCEWithLogitsLoss. "
                "Check loss computation.",
                loss_value
            )
        
        total_loss += loss_value
        
        # Clear intermediate tensors to free GPU memory
        del loss
        if device.startswith("cuda"):
            # Clear cache less frequently with fixed-size videos (every 20 batches)
            # Fixed-size videos are more memory-efficient, so we can reduce cache clearing
            if (batch_idx + 1) % 20 == 0:
                torch.cuda.empty_cache()
            # Synchronize less frequently with larger batches (every 10 batches)
            # Larger batches are more efficient, so we don't need to sync as often
            if batch_idx % 10 == 0:
                torch.cuda.synchronize()

        if (batch_idx + 1) % log_interval == 0:
            # Add diagnostics: logits range, predictions, and actual loss value
            with torch.no_grad():
                logits_min = float(logits.min().item())
                logits_max = float(logits.max().item())
                logits_mean = float(logits.mean().item())
                # Handle batch_size=1 case for std() computation
                if logits.numel() > 1:
                    logits_std = float(logits.std().item())
                else:
                    logits_std = 0.0  # Single element, no variance
                logits_unique = len(torch.unique(logits))
                probs = logits.sigmoid()
                probs_min = float(probs.min().item())
                probs_max = float(probs.max().item())
                probs_mean = float(probs.mean().item())
                preds = (probs >= 0.5).long()
                correct = int((preds == labels).sum().item())
                total = int(labels.numel())
                acc = correct / max(1, total)
                
                # Check label distribution
                label_counts = {}
                for lbl in labels.cpu().tolist():
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1
            
            # Check gradient flow (only for trainable parameters)
            # IMPORTANT: Check gradients AFTER backward() but before optimizer step
            grad_norm = 0.0
            param_norm = 0.0
            trainable_count = 0
            grad_count = 0
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        trainable_count += 1
                        param_norm += p.data.norm().item() ** 2
                        if p.grad is not None:
                            grad_count += 1
                            grad_norm += p.grad.data.norm().item() ** 2
            grad_norm = grad_norm ** 0.5 if grad_norm > 0 else 0.0
            param_norm = param_norm ** 0.5 if param_norm > 0 else 0.0
            
            # Warn if no gradients are present
            if grad_count == 0 and batch_idx > 0:
                logger.error(
                    "⚠ CRITICAL: No gradients found for any trainable parameters! "
                    "Model is not learning. Check: 1) loss.backward() is called, "
                    "2) model parameters have requires_grad=True, 3) model is in train mode."
                )
            
            # Memory monitoring
            mem_info = ""
            if device.startswith("cuda"):
                allocated = torch.cuda.memory_allocated(0) / 1e9
                reserved = torch.cuda.memory_reserved(0) / 1e9
                mem_info = f", GPU: {allocated:.2f}GB/{reserved:.2f}GB"
            
            # Enhanced logging with more diagnostics
            logger.info(
                "Epoch %d Batch %d/%d Loss %.6f | "
                "Logits: [%.3f, %.3f] mean=%.3f std=%.3f unique=%d | "
                "Probs: [%.3f, %.3f] mean=%.3f | "
                "Acc=%.1f%% | Labels: %s | "
                "GradNorm=%.4f ParamNorm=%.2f TrainableParams=%d%s",
                epoch,
                batch_idx + 1,
                len(loader),
                loss_value,
                logits_min,
                logits_max,
                logits_mean,
                logits_std,
                logits_unique,
                probs_min,
                probs_max,
                probs_mean,
                acc * 100,
                label_counts,
                grad_norm,
                param_norm,
                trainable_count,
                mem_info,
            )
            
            # Warn if logits are too similar (model might not be learning)
            if logits_std < 0.01 and batch_idx > 0:
                logger.warning(
                    "⚠ Logits have very low std (%.4f) - model may not be learning! "
                    "Check if gradients are flowing (GradNorm=%.4f)",
                    logits_std, grad_norm
                )
            
            # Warn if loss is suspiciously small
            if abs(loss_value) < 1e-6 and batch_idx > 0:
                logger.warning(
                    "⚠ Loss is extremely small (%.6f) - this may indicate a problem! "
                    "Expected loss for random predictions: ~0.693 (ln(2))",
                    loss_value
                )
            
            # Clear logits from GPU after logging to save memory
            del logits, probs, preds
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    avg_loss = total_loss / max(1, len(loader))
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[float, float]:
    """Return (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    criterion: Optional[nn.Module] = None
    first_batch = True
    
    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
    
        # Use autocast for mixed precision (optimized: check once, reuse)
        if device.startswith("cuda"):
            try:
                # New API (PyTorch 2.0+)
                with torch.amp.autocast('cuda'):
                    logits = model(clips)
            except (AttributeError, TypeError):
                # Fallback to old API
                with torch.cuda.amp.autocast():
                    logits = model(clips)
        else:
            logits = model(clips)

        # Infer criterion on first batch, mirroring train_one_epoch logic
        if first_batch:
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                # Binary case
                if logits.ndim == 2:
                    logits = logits.squeeze(-1)
                criterion = nn.BCEWithLogitsLoss()
            else:
                # Multi-class (including 2-logit case)
                criterion = nn.CrossEntropyLoss()
            first_batch = False

        if isinstance(criterion, nn.BCEWithLogitsLoss):
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)
            targets = labels.float()
            loss = criterion(logits, targets)
            preds = (logits.sigmoid() >= 0.5).long()
        else:
            # CrossEntropyLoss expects (N, C)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            loss = criterion(logits, labels)  # type: ignore[arg-type]
            preds = logits.argmax(dim=1)
    
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.numel())
        total_loss += float(loss.item())
        
        # Clear intermediate tensors to save memory
        del logits, preds, loss, clips, labels
        
        # Ultra aggressive GC after every evaluation batch
        from .mlops_utils import aggressive_gc
        aggressive_gc(clear_cuda=device.startswith("cuda"))
    
    avg_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_samples)
    
    # Final aggressive GC after evaluation completes
    aggressive_gc(clear_cuda=device.startswith("cuda"))
    
    return avg_loss, acc


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optim_cfg: OptimConfig,
    train_cfg: TrainConfig,
) -> nn.Module:
    """High-level training loop with optional validation."""
    device = train_cfg.device
    model.to(device)
    
    # Verify model is in train mode
    model.train()
    if not model.training:
        logger.warning("⚠ Model is not in training mode! Setting to train mode.")
        model.train()

    optimizer = build_optimizer(model, optim_cfg)
    scheduler = build_scheduler(optimizer)
    
    # Log optimizer configuration
    logger.info("Optimizer: %s, LR: %.6f, Weight Decay: %.6f", 
                type(optimizer).__name__, optim_cfg.lr, optim_cfg.weight_decay)
    
    # Verify trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found! All parameters are frozen. "
            "Check model configuration (freeze_backbone setting)."
        )
    trainable_param_count = sum(p.numel() for p in trainable_params)
    logger.info("Trainable parameters: %d (%.2fM)", 
                trainable_param_count, trainable_param_count / 1e6)
    
    # Check initial parameter values (for debugging)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                if abs(param_mean) > 10 or param_std > 10:
                    logger.warning(
                        "⚠ Parameter %s has large values: mean=%.3f, std=%.3f",
                        name, param_mean, param_std
                    )
                break  # Just check first trainable param

    best_val_acc = 0.0
    best_state: Optional[Dict[str, torch.Tensor]] = None

    early_stopper = EarlyStopping(
        patience=train_cfg.early_stopping_patience, mode="max"
    )
    ckpt_manager = CheckpointManager(train_cfg.checkpoint_dir)

    for epoch in range(1, train_cfg.num_epochs + 1):
        try:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                use_class_weights=train_cfg.use_class_weights,
                epoch=epoch,
                log_interval=train_cfg.log_interval,
                gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("CUDA OOM during training epoch %d. Clearing cache and stopping.", epoch)
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "Training stopped due to CUDA OOM. "
                    "Try reducing batch_size, num_frames, or model size."
                ) from e
            else:
                raise

        # Step scheduler at the end of the epoch (AFTER optimizer.step() calls in train_one_epoch)
        # This ensures scheduler.step() is called after optimizer.step() in the same epoch
        if train_loss is not None:
            scheduler.step()

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device=device)
            logger.info(
                "Epoch %d/%d train_loss=%.4f val_loss=%.4f val_acc=%.4f",
                epoch,
                train_cfg.num_epochs,
                train_loss,
                val_loss,
                val_acc,
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict()

            ckpt_manager.maybe_save(model, val_acc)
            early_stopper.step(val_acc)
            if early_stopper.should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d with best val_acc=%.4f",
                    epoch,
                    early_stopper.best if early_stopper.best is not None else -1.0,
                )
                break
        else:
            logger.info(
                "Epoch %d/%d train_loss=%.4f",
                epoch,
                train_cfg.num_epochs,
                train_loss,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


__all__ = [
    "OptimConfig",
    "TrainConfig",
    "freeze_all",
    "unfreeze_all",
    "freeze_backbone_unfreeze_head",
    "build_optimizer",
    "build_scheduler",
    "train_one_epoch",
    "evaluate",
    "fit",
]


