
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

def get_predictions(outputs, threshold=0.5):
    """
    Convert raw model outputs (logits) into binary predictions.
    """
    probs = torch.sigmoid(outputs).detach().cpu().numpy()
    return (probs >= threshold).astype(int)

def compute_metrics(outputs, targets, threshold=0.5):
    """
    Compute evaluation metrics for multi-label classification.

    Args:
        outputs (Tensor): raw logits from the model, shape (batch_size, num_classes)
        targets (Tensor): ground truth one-hot labels, shape (batch_size, num_classes)
        threshold (float): threshold to binarize probabilities

    Returns:
        dict: containing accuracy, precision, recall, f1, and mAP
    """
    preds = get_predictions(outputs, threshold)
    targets = targets.cpu().numpy()

    metrics = {}

    # Accuracy per image (exact match ratio)
    metrics["accuracy"] = accuracy_score(targets, preds)

    # Precision, Recall, F1 (macro = treats all classes equally)
    metrics["precision_macro"] = precision_score(targets, preds, average="macro", zero_division=0)
    metrics["recall_macro"] = recall_score(targets, preds, average="macro", zero_division=0)
    metrics["f1_macro"] = f1_score(targets, preds, average="macro", zero_division=0)

    # Micro = treats all samples equally (better for imbalanced datasets)
    metrics["precision_micro"] = precision_score(targets, preds, average="micro", zero_division=0)
    metrics["recall_micro"] = recall_score(targets, preds, average="micro", zero_division=0)
    metrics["f1_micro"] = f1_score(targets, preds, average="micro", zero_division=0)

    # Mean Average Precision (mAP)
    try:
        metrics["mAP"] = average_precision_score(targets, preds, average="macro")
    except ValueError:
        metrics["mAP"] = 0.0  # happens if no positive samples

    return metrics

def print_metrics(metrics: dict, prefix=""):
    """
    Nicely format and print metrics.
    """
    print(f"{prefix} Accuracy: {metrics['accuracy']:.4f}")
    print(f"{prefix} Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"{prefix} Recall (macro): {metrics['recall_macro']:.4f}")
    print(f"{prefix} F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"{prefix} Precision (micro): {metrics['precision_micro']:.4f}")
    print(f"{prefix} Recall (micro): {metrics['recall_micro']:.4f}")
    print(f"{prefix} F1 (micro): {metrics['f1_micro']:.4f}")
    print(f"{prefix} mAP: {metrics['mAP']:.4f}")