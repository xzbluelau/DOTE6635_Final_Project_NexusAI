"""
07_feature_attribution.py - Feature importance and gradient-based attribution for 1D-CNN.

Generates:
  1. Permutation importance bar chart (model-agnostic)
  2. Gradient x Input attribution summary plot (SHAP-equivalent interpretation)

Usage: python code/07_feature_attribution.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")
model_mod = import_module("03_model_1dcnn")

OUTPUT_FIGURES = setup.OUTPUT_FIGURES
DATA_PROCESSED = setup.DATA_PROCESSED
MODELS_DIR = setup.MODELS_DIR

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11})


def load_model_and_data():
    """Load trained 1D-CNN and test data."""
    # Load model
    state = torch.load(os.path.join(MODELS_DIR, "1dcnn_best.pt"),
                        map_location="cpu", weights_only=True)
    num_features = state["classifier.weight"].shape[1]
    model = model_mod.FraudDetectionCNN1D(num_features=num_features)
    model.load_state_dict(state)
    model.eval()

    # Load test data
    X_test = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_test.parquet")).values.astype(np.float32)
    y_test = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_test.parquet")).values.ravel()

    # Load feature names
    with open(os.path.join(DATA_PROCESSED, "metadata.json")) as f:
        meta = json.load(f)
    feature_names = list(meta["feature_cols"])

    return model, X_test, y_test, feature_names


def predict_proba(model, X):
    """Get fraud probability from model for numpy array X."""
    model.eval()
    with torch.no_grad():
        tensor = torch.FloatTensor(X).unsqueeze(1)  # (N, 1, 46)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return probs


# =============================================================================
# 1. Permutation Importance
# =============================================================================
def compute_permutation_importance(model, X, y, feature_names, n_repeats=5):
    """Compute permutation importance using AUC-ROC drop."""
    from sklearn.metrics import roc_auc_score

    print("Computing permutation importance...")
    baseline_probs = predict_proba(model, X)
    baseline_auc = roc_auc_score(y, baseline_probs)
    print(f"  Baseline AUC-ROC: {baseline_auc:.4f}")

    importances = np.zeros(len(feature_names))
    importances_std = np.zeros(len(feature_names))

    for i, fname in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, i] = np.random.permutation(X_perm[:, i])
            perm_probs = predict_proba(model, X_perm)
            perm_auc = roc_auc_score(y, perm_probs)
            drops.append(baseline_auc - perm_auc)
        importances[i] = np.mean(drops)
        importances_std[i] = np.std(drops)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(feature_names)}] features done")

    return importances, importances_std


def plot_permutation_importance(importances, importances_std, feature_names, top_n=20):
    """Plot top-N permutation importance as horizontal bar chart."""
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_imp = importances[top_idx][::-1]
    top_std = importances_std[top_idx][::-1]
    top_names = [feature_names[i] for i in top_idx][::-1]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(top_n), top_imp, xerr=top_std, color="#2196F3", alpha=0.85,
            edgecolor="white", linewidth=0.5, capsize=3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.set_xlabel("Importance (AUC-ROC Drop)", fontsize=12)
    ax.set_title("1D-CNN Permutation Feature Importance (Top 20)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    path = os.path.join(OUTPUT_FIGURES, "permutation_importance_1dcnn.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Permutation importance plot saved: {path}")
    return top_idx, top_imp


# =============================================================================
# 2. Gradient x Input Attribution (SHAP-equivalent)
# =============================================================================
def compute_gradient_attribution(model, X, n_samples=500):
    """Compute gradient x input attribution for each feature.

    This computes d(output_fraud_prob)/d(input) * input for each feature,
    which gives a local attribution similar to SHAP values.
    """
    print(f"Computing gradient x input attribution (n={n_samples})...")

    model.eval()
    # Sample subset
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(n_samples, len(X)), replace=False)
    X_sub = X[idx]

    tensor = torch.FloatTensor(X_sub).unsqueeze(1)  # (N, 1, 46)
    tensor.requires_grad_(True)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)
    fraud_probs = probs[:, 1]  # fraud class probability

    # Sum of fraud probs as scalar loss for backward
    fraud_probs.sum().backward()

    gradients = tensor.grad.detach().numpy()[:, 0, :]  # (N, 46)

    # Attribution = gradient * input
    attribution = gradients * X_sub  # (N, 46)

    return attribution, idx


def plot_attribution_summary(attribution, feature_names, y, idx, top_n=20):
    """Plot gradient attribution summary (SHAP-style beeswarm plot).

    Left panel: mean absolute attribution bar chart (feature importance ranking)
    Right panel: beeswarm-style scatter showing attribution direction per sample
    """
    abs_mean = np.mean(np.abs(attribution), axis=0)
    top_feat_idx = np.argsort(abs_mean)[::-1][:top_n]
    top_feat_idx_sorted = top_feat_idx[np.argsort(abs_mean[top_feat_idx])[::-1]]

    # --- Figure 1: Bar chart of mean |attribution| ---
    fig, ax = plt.subplots(figsize=(10, 7))
    names = [feature_names[i] for i in top_feat_idx_sorted][::-1]
    values = abs_mean[top_feat_idx_sorted][::-1]

    colors_bar = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))
    ax.barh(range(len(names)), values, color=colors_bar, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Mean |Attribution| (Gradient x Input)", fontsize=12)
    ax.set_title("1D-CNN Feature Attribution (Top 20)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_FIGURES, "gradient_attribution_bar_1dcnn.png")
    plt.savefig(path1, bbox_inches="tight")
    plt.close()
    print(f"  Attribution bar chart saved: {path1}")

    # --- Figure 2: SHAP-style beeswarm / summary plot ---
    fig, ax = plt.subplots(figsize=(10, 8))

    y_sub = y[idx]

    for rank, fi in enumerate(top_feat_idx_sorted):
        feat_attr = attribution[:, fi]
        feat_val = np.zeros(len(feat_attr))  # we don't have raw values handy, use standardized
        # Use the attribution values directly for color
        feat_vals_sample = attribution[:, fi]

        # Jitter y positions
        jitter = np.random.uniform(-0.3, 0.3, size=len(feat_attr))
        y_pos = rank + jitter

        # Color by attribution sign: positive (red) = increases fraud prob, negative (blue) = decreases
        scatter = ax.scatter(feat_attr, y_pos, c=feat_vals_sample,
                           cmap="coolwarm", vmin=-np.max(np.abs(attribution[:, top_feat_idx_sorted])),
                           vmax=np.max(np.abs(attribution[:, top_feat_idx_sorted])),
                           alpha=0.5, s=8, linewidths=0)

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=10)
    ax.set_xlabel("Attribution Value (Gradient x Input)", fontsize=12)
    ax.set_title("1D-CNN Feature Attribution Summary (SHAP-style)", fontsize=13, fontweight="bold")

    # Add vertical line at 0
    ax.axvline(x=0, color="gray", linewidth=0.8, linestyle="--", alpha=0.7)

    # Color bar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Attribution Value", fontsize=10)

    # Add annotation
    ax.text(0.98, 0.02, "Red: increases fraud probability\nBlue: decreases fraud probability",
            transform=ax.transAxes, fontsize=9, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(OUTPUT_FIGURES, "gradient_attribution_shap_1dcnn.png")
    plt.savefig(path2, bbox_inches="tight")
    plt.close()
    print(f"  SHAP-style attribution plot saved: {path2}")

    return top_feat_idx_sorted, abs_mean


def main():
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    print("=" * 60)
    print("1D-CNN FEATURE ATTRIBUTION ANALYSIS")
    print("=" * 60)

    # Load
    model, X_test, y_test, feature_names = load_model_and_data()
    print(f"Model loaded: {X_test.shape[1]} features, {len(X_test)} test samples")
    print(f"Feature names: {len(feature_names)}")

    # 1. Permutation importance
    print("\n--- Permutation Importance ---")
    perm_imp, perm_std = compute_permutation_importance(
        model, X_test, y_test, feature_names, n_repeats=5
    )
    top_idx, top_vals = plot_permutation_importance(perm_imp, perm_std, feature_names, top_n=20)

    print("\n  Top-10 by Permutation Importance:")
    sorted_idx = np.argsort(perm_imp)[::-1]
    for i, idx in enumerate(sorted_idx[:10]):
        print(f"    {i+1}. {feature_names[idx]}: {perm_imp[idx]:.6f} (+/- {perm_std[idx]:.6f})")

    # 2. Gradient x Input attribution
    print("\n--- Gradient x Input Attribution ---")
    attribution, sample_idx = compute_gradient_attribution(model, X_test, n_samples=500)
    top_attr_idx, abs_mean = plot_attribution_summary(
        attribution, feature_names, y_test, sample_idx, top_n=20
    )

    print("\n  Top-10 by Mean |Attribution|:")
    for i, idx in enumerate(top_attr_idx[:10]):
        print(f"    {i+1}. {feature_names[idx]}: {abs_mean[idx]:.6f}")

    # Save results
    results = {
        "permutation_importance": {
            feature_names[i]: {"importance": float(perm_imp[i]), "std": float(perm_std[i])}
            for i in range(len(feature_names))
        },
        "gradient_attribution_mean_abs": {
            feature_names[i]: float(np.mean(np.abs(attribution[:, i])))
            for i in range(len(feature_names))
        }
    }
    out_path = os.path.join(setup.OUTPUT_TABLES, "feature_attribution_1dcnn.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved: {out_path}")

    print("\n" + "=" * 60)
    print("FEATURE ATTRIBUTION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
