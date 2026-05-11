"""
utils.py — Shared helper functions for the fraud detection project.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)
from datetime import datetime


def load_dataset(name: str, data_dir: str) -> pd.DataFrame:
    """Load a CSV dataset from the data directory."""
    path = os.path.join(data_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def save_results(results: dict, output_path: str) -> None:
    """Save a results dictionary as JSON."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert numpy types to native Python types for JSON serialization
    serializable = {}
    for k, v in results.items():
        if isinstance(v, (np.integer,)):
            serializable[k] = int(v)
        elif isinstance(v, (np.floating,)):
            serializable[k] = float(v)
        elif isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        else:
            serializable[k] = v
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute all evaluation metrics for binary classification."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
    }


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple:
    """Find the classification threshold that maximizes F1 score.

    Returns:
        (best_threshold, best_f1)
    """
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh), float(best_f1)


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path):
    """Plot and save a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(results_dict, save_path):
    """Plot ROC curves for multiple models on the same axes."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_prob) in results_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pr_curves(results_dict, save_path):
    """Plot Precision-Recall curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_prob) in results_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{name} (AP={ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves Comparison")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def get_timestamp() -> str:
    """Return a timestamp string for file naming."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
