"""
Metrics and evaluation helpers for the FVC video classifier.

Includes:
- Accuracy, precision, recall, F1
- ROC-AUC (if sklearn is available)
- Confusion matrix
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_logits_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run model over loader and collect logits and labels."""
    model.eval()
    all_logits = []
    all_labels = []
    for clips, labels in loader:
        clips = clips.to(device)
        labels = labels.to(device)
        logits = model(clips)
        # For binary models that output a single logit per sample, squeeze it;
        # for multi-class models (including 2-logit naive_cnn) keep (N, C).
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(-1)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())
    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def basic_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute accuracy, precision, recall, F1 for binary classification.

    Supports both:
    - Binary models with a single logit per sample (logits shape (N,))
    - Multi-class models with logits shape (N, C) (we treat class 1 vs 0)
    """
    if logits.ndim == 1:
        # Single-logit binary case
        probs = logits.sigmoid()
        preds = (probs >= threshold).long()
    else:
        # Multi-class case (including 2-logit naive_cnn)
        # Use softmax over classes and take argmax as predicted class.
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

    # Ensure labels are 1D tensor of class indices
    if labels.ndim > 1:
        labels = labels.view(-1)

    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def confusion_matrix(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return 2x2 confusion matrix for binary classification."""
    if logits.ndim == 1:
        probs = logits.sigmoid()
        preds = (probs >= threshold).long()
    else:
        probs = logits.softmax(dim=1)
        preds = probs.argmax(dim=1)

    if labels.ndim > 1:
        labels = labels.view(-1)

    cm = torch.zeros((2, 2), dtype=torch.long)
    for p, y in zip(preds, labels):
        cm[int(y.item()), int(p.item())] += 1
    return cm


def roc_auc(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Compute ROC-AUC using sklearn if available, else return -1.0."""
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore[import]
    except Exception:
        return -1.0

    if logits.ndim == 1:
        probs = logits.sigmoid().numpy()
    else:
        # Use probability of class 1 from softmax
        probs = logits.softmax(dim=1)[:, 1].numpy()
    y = labels.numpy()
    try:
        return float(roc_auc_score(y, probs))
    except ValueError:
        # Happens when only one class is present
        return -1.0


__all__ = [
    "collect_logits_and_labels",
    "basic_classification_metrics",
    "confusion_matrix",
    "roc_auc",
]


