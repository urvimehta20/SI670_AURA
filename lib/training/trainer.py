"""
Core training utilities.

Provides:
- Optimizer and scheduler builders
- Training and evaluation loops
- Configuration dataclasses
- Early stopping
- Model freezing utilities
- Class weight computation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable, Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import _LRScheduler, StepLR, CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler
# VariableARVideoModel import removed - not needed (freeze_backbone_unfreeze_head uses generic nn.Module)

logger = logging.getLogger(__name__)


@dataclass
class OptimConfig:
    """Optimizer configuration."""
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    # Differential learning rates for pretrained models
    backbone_lr: Optional[float] = None  # If None, uses lr
    head_lr: Optional[float] = None  # If None, uses lr * 10 (common practice)
    # Gradient clipping
    max_grad_norm: float = 1.0  # Clip gradients to this norm (0 = disabled)


@dataclass
class TrainConfig:
    """Training configuration."""
    num_epochs: int = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_interval: int = 10
    use_class_weights: bool = True
    use_amp: bool = True
    checkpoint_dir: Optional[str] = None
    early_stopping_patience: int = 5
    gradient_accumulation_steps: int = 1
    # Learning rate scheduling
    scheduler_type: str = "cosine"  # "cosine", "step", or "none"
    warmup_epochs: int = 2  # Number of warmup epochs
    warmup_factor: float = 0.1  # Initial LR = base_lr * warmup_factor
    # Gradient monitoring
    log_grad_norm: bool = True  # Log gradient norms for debugging


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


def freeze_all(model: nn.Module) -> None:
    """Freeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def freeze_backbone_unfreeze_head(model: nn.Module) -> None:
    """Freeze backbone but unfreeze head (for pretrained models)."""
    if hasattr(model, 'backbone'):
        freeze_all(model.backbone)
    if hasattr(model, 'head'):
        unfreeze_all(model.head)
    elif hasattr(model, 'classifier'):
        unfreeze_all(model.classifier)
    elif hasattr(model, 'fc'):  # Some models use 'fc' for final layer
        unfreeze_all(model.fc)


def trainable_params(model: nn.Module) -> Iterable[torch.nn.Parameter]:
    """Get trainable parameters."""
    return (p for p in model.parameters() if p.requires_grad)


def compute_class_counts(loader: DataLoader, num_classes: int) -> torch.Tensor:
    """Compute class counts from a DataLoader."""
    counts = torch.zeros(num_classes, dtype=torch.long)
    for _, labels in loader:
        for label in labels:
            counts[label.item()] += 1
    return counts


def make_class_weights(counts: torch.Tensor) -> torch.Tensor:
    """Make class weights from counts (inverse frequency)."""
    total = counts.sum().float()
    weights = total / (counts.float() + 1e-6)  # Add epsilon to avoid division by zero
    weights = weights / weights.sum() * len(weights)  # Normalize
    return weights


def make_weighted_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """Make weighted random sampler from labels."""
    unique_labels = torch.unique(labels)
    counts = torch.bincount(labels)
    weights = make_class_weights(counts)
    sample_weights = weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def build_optimizer(
    model: nn.Module, 
    config: OptimConfig,
    use_differential_lr: bool = False
) -> Optimizer:
    """
    Build optimizer from config.
    
    Args:
        model: Model to optimize
        config: Optimizer configuration
        use_differential_lr: If True, use different LRs for backbone and head (for pretrained models)
    
    Returns:
        Optimizer instance
    """
    if use_differential_lr and (config.backbone_lr is not None or config.head_lr is not None):
        # Separate parameter groups for backbone and head
        backbone_params = []
        head_params = []
        
        # Try to identify backbone and head
        if hasattr(model, 'backbone') and hasattr(model, 'fc'):
            # Standard pretrained model structure
            backbone_params = list(model.backbone.parameters())
            head_params = list(model.fc.parameters())
        elif hasattr(model, 'backbone') and hasattr(model, 'head'):
            backbone_params = list(model.backbone.parameters())
            head_params = list(model.head.parameters())
        elif hasattr(model, 'backbone') and hasattr(model, 'classifier'):
            backbone_params = list(model.backbone.parameters())
            head_params = list(model.classifier.parameters())
        else:
            # For ViT-based models, try to separate backbone from temporal head
            if hasattr(model, 'vit_backbone'):
                backbone_params = list(model.vit_backbone.parameters())
                # Get all other parameters as head
                head_params = [
                    p for name, p in model.named_parameters()
                    if 'vit_backbone' not in name
                ]
            else:
                # Fallback: use all parameters with same LR
                use_differential_lr = False
        
        if use_differential_lr and backbone_params and head_params:
            backbone_lr = config.backbone_lr if config.backbone_lr is not None else config.lr
            head_lr = config.head_lr if config.head_lr is not None else config.lr * 10.0
            
            param_groups = [
                {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
                {'params': head_params, 'lr': head_lr, 'name': 'head'}
            ]
            
            logger.info(
                f"Using differential learning rates: backbone={backbone_lr:.2e}, head={head_lr:.2e}"
            )
            
            # Use AdamW for better weight decay handling
            return AdamW(
                param_groups,
                lr=config.lr,  # Base LR (overridden by param groups)
                weight_decay=config.weight_decay,
                betas=config.betas
            )
    
    # Standard optimizer for all parameters
    # Use AdamW instead of Adam for better weight decay (decoupled from gradient)
    return AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas
    )


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_epochs: int = 20,
    warmup_epochs: int = 2,
    warmup_factor: float = 0.1,
    step_size: int = 10,
    gamma: float = 0.1
) -> _LRScheduler:
    """
    Build learning rate scheduler with optional warmup.
    
    Args:
        optimizer: Optimizer instance
        scheduler_type: "cosine", "step", or "none"
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        warmup_factor: Initial LR multiplier during warmup
        step_size: Step size for StepLR
        gamma: LR decay factor for StepLR
    
    Returns:
        Learning rate scheduler
    """
    if scheduler_type == "none":
        # No scheduling
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    # Create base scheduler
    if scheduler_type == "cosine":
        base_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    elif scheduler_type == "step":
        base_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}. Using StepLR.")
        base_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Add warmup if requested
    if warmup_epochs > 0:
        # Use a simpler approach: combine warmup with base scheduler
        # We'll manually handle warmup in the training loop, then use base scheduler
        # For now, return base scheduler and handle warmup in fit()
        logger.info(f"Warmup will be handled in training loop ({warmup_epochs} epochs)")
        return base_scheduler
    else:
        return base_scheduler


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
    max_grad_norm: float = 1.0,
    log_grad_norm: bool = False,
) -> float:
    """
    Train model for one epoch.
    
    Returns:
        Average training loss
    """
    from lib.utils.memory import aggressive_gc
    
    model.train()
    total_loss = 0.0
    
    # Clear CUDA cache at start
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    
    # Loss criterion (inferred on first batch)
    first_batch = True
    criterion: Optional[nn.Module] = None
    
    # AMP scaler
    if use_amp and device.startswith("cuda"):
        try:
            scaler = torch.amp.GradScaler('cuda')
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    
    for batch_idx, (clips, labels) in enumerate(loader):
        # Aggressive GC every 5 batches
        if batch_idx > 0 and batch_idx % 5 == 0:
            import gc
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        
        # GPU-optimized data transfer with non_blocking
        if device.startswith("cuda"):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            clips = clips.to(device)
            labels = labels.to(device)
        
        # Zero gradients at start of accumulation cycle
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()
        
        # Infer criterion on first batch
        if first_batch:
            with torch.no_grad():
                logits = model(clips)
                if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                    criterion = nn.BCEWithLogitsLoss()
                else:
                    criterion = nn.CrossEntropyLoss()
            first_batch = False
        
        # Forward pass with AMP and OOM handling
        try:
            if scaler is not None:
                try:
                    with torch.amp.autocast('cuda'):
                        logits = model(clips)
                except (AttributeError, TypeError):
                    with torch.cuda.amp.autocast():
                        logits = model(clips)
            else:
                logits = model(clips)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg and "memory" in error_msg:
                # OOM during forward pass - clear cache and re-raise
                from lib.utils.memory import handle_oom_error
                handle_oom_error(e, f"forward pass batch {batch_idx}")
                # Clear gradients to free memory
                if batch_idx % gradient_accumulation_steps == 0:
                    optimizer.zero_grad()
                # Re-raise to let caller handle (may reduce batch size or skip batch)
                raise
            else:
                # Not OOM - re-raise original error
                raise
        
        # Compute loss
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)
            targets = labels.float()
            loss = criterion(logits, targets)
        else:
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            loss = criterion(logits, labels)
        
        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights at end of accumulation cycle
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Gradient clipping (before optimizer step)
            if max_grad_norm > 0:
                if scaler is not None:
                    # Unscale gradients before clipping (required for AMP)
                    scaler.unscale_(optimizer)
                    # Clip gradients
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=max_grad_norm
                    )
                else:
                    # Clip gradients directly
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        max_norm=max_grad_norm
                    )
                
                # Log gradient norm if requested
                if log_grad_norm and (batch_idx + 1) % log_interval == 0:
                    logger.info(f"Gradient norm: {grad_norm:.4f} (clipped to {max_grad_norm})")
            
            # Optimizer step
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            aggressive_gc(clear_cuda=device.startswith("cuda"))
        
        total_loss += float(loss.item() * gradient_accumulation_steps)
        
        # Logging
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(loader)}, "
                f"Loss: {loss.item() * gradient_accumulation_steps:.4f}"
            )
    
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, Any]:
    """
    Evaluate model on validation/test set.
    
    Returns:
        Dictionary with metrics:
        {
            "loss": average_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "per_class": {
                "0": {"precision": ..., "recall": ..., "f1": ...},
                "1": {"precision": ..., "recall": ..., "f1": ...}
            }
        }
    """
    from lib.utils.memory import aggressive_gc
    from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_fscore_support
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    # Collect all predictions and labels for metric computation
    all_preds = []
    all_labels = []
    
    criterion: Optional[nn.Module] = None
    first_batch = True
    
    for clips, labels in loader:
        # GPU-optimized data transfer
        if device.startswith("cuda"):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            clips = clips.to(device)
            labels = labels.to(device)
        
        # Use autocast for mixed precision
        if device.startswith("cuda"):
            try:
                with torch.amp.autocast('cuda'):
                    logits = model(clips)
            except (AttributeError, TypeError):
                with torch.cuda.amp.autocast():
                    logits = model(clips)
        else:
            logits = model(clips)
        
        # Infer criterion on first batch
        if first_batch:
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                if logits.ndim == 2:
                    logits = logits.squeeze(-1)
                criterion = nn.BCEWithLogitsLoss()
            else:
                criterion = nn.CrossEntropyLoss()
            first_batch = False
        
        # Compute loss and predictions
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            if logits.ndim == 2 and logits.shape[1] == 1:
                logits = logits.squeeze(-1)
            targets = labels.float()
            loss = criterion(logits, targets)
            preds = (logits.sigmoid() >= 0.5).long()
        else:
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
        
        # Collect predictions and labels
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
        total_correct += int((preds == labels).sum().item())
        total_samples += int(labels.numel())
        total_loss += float(loss.item())
        
        # Clear intermediate tensors
        del logits, preds, loss, clips, labels
        aggressive_gc(clear_cuda=device.startswith("cuda"))
    
    avg_loss = total_loss / max(1, len(loader))
    acc = total_correct / max(1, total_samples)
    
    # Compute comprehensive metrics
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    # Overall metrics (binary classification)
    precision = float(precision_score(all_labels, all_preds, average='binary', zero_division=0))
    recall = float(recall_score(all_labels, all_preds, average='binary', zero_division=0))
    f1 = float(f1_score(all_labels, all_preds, average='binary', zero_division=0))
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    
    per_class_metrics = {}
    for class_idx in range(len(precision_per_class)):
        per_class_metrics[str(class_idx)] = {
            "precision": float(precision_per_class[class_idx]),
            "recall": float(recall_per_class[class_idx]),
            "f1": float(f1_per_class[class_idx]),
            "support": int(support[class_idx])
        }
    
    # Final aggressive GC
    aggressive_gc(clear_cuda=device.startswith("cuda"))
    
    return {
        "loss": avg_loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class": per_class_metrics
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optim_cfg: OptimConfig,
    train_cfg: TrainConfig,
    use_differential_lr: bool = False,
) -> nn.Module:
    """
    High-level training loop with optional validation.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
        optim_cfg: Optimizer configuration
        train_cfg: Training configuration
        use_differential_lr: Use different LRs for backbone and head (for pretrained models)
    
    Returns:
        Trained model
    """
    device = train_cfg.device
    model.to(device)
    model.train()  # Ensure model is in training mode
    
    # Build optimizer with optional differential learning rates
    optimizer = build_optimizer(model, optim_cfg, use_differential_lr=use_differential_lr)
    
    # Build scheduler with warmup
    scheduler = build_scheduler(
        optimizer,
        scheduler_type=train_cfg.scheduler_type,
        num_epochs=train_cfg.num_epochs,
        warmup_epochs=train_cfg.warmup_epochs,
        warmup_factor=train_cfg.warmup_factor
    )
    
    # Early stopping
    early_stopping = None
    if train_cfg.early_stopping_patience > 0 and val_loader is not None:
        early_stopping = EarlyStopping(patience=train_cfg.early_stopping_patience, mode="max")
    
    best_val_acc = 0.0
    best_model_state = None
    initial_lr = optimizer.param_groups[0]['lr']
    
    try:
        for epoch in range(1, train_cfg.num_epochs + 1):
            # Handle warmup manually
            if train_cfg.warmup_epochs > 0 and epoch <= train_cfg.warmup_epochs:
                # Linear warmup: warmup_factor -> 1.0
                warmup_progress = epoch / train_cfg.warmup_epochs
                lr_scale = train_cfg.warmup_factor + (1.0 - train_cfg.warmup_factor) * warmup_progress
                for param_group in optimizer.param_groups:
                    param_group['lr'] = initial_lr * lr_scale
            else:
                # After warmup, use base scheduler
                scheduler.step()
            
            # Ensure model is in training mode (BatchNorm, Dropout active)
            model.train()
            
            # Train
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device=device,
                use_class_weights=train_cfg.use_class_weights,
                use_amp=train_cfg.use_amp,
                epoch=epoch,
                log_interval=train_cfg.log_interval,
                gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
                max_grad_norm=optim_cfg.max_grad_norm,
                log_grad_norm=train_cfg.log_grad_norm,
            )
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Epoch {epoch}/{train_cfg.num_epochs}, Train Loss: {train_loss:.4f}, LR: {current_lr:.2e}"
            )
            
            # Validate
            if val_loader is not None:
                # Ensure model is in eval mode (BatchNorm uses running stats, Dropout disabled)
                model.eval()
                val_metrics = evaluate(model, val_loader, device=device)
                val_loss = val_metrics["loss"]
                val_acc = val_metrics["accuracy"]
                val_f1 = val_metrics["f1"]
                val_precision = val_metrics["precision"]
                val_recall = val_metrics["recall"]
                logger.info(
                    f"Epoch {epoch}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"Val F1: {val_f1:.4f}, Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}"
                )
                
                # Early stopping check
                if early_stopping is not None:
                    early_stopping.step(val_acc)
                    if early_stopping.should_stop:
                        logger.info(f"Early stopping triggered at epoch {epoch}")
                        break
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save best model state
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    logger.info(f"New best validation accuracy: {best_val_acc:.4f}")
        
        # Restore best model state
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            logger.info("Restored best model state based on validation accuracy")
        
        return model
    except Exception as e:
        # Ensure GPU cleanup on error
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        # Re-raise exception after cleanup
        raise
    finally:
        # Final cleanup
        if device.startswith("cuda"):
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

