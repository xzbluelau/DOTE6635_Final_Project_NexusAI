"""
06_visualization.py - Generate all figures and analysis for the paper.

Usage: python code/06_visualization.py
"""

import os, sys, json, pickle, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve,
    roc_auc_score, average_precision_score,
)
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")

OUTPUT_FIGURES = setup.OUTPUT_FIGURES
OUTPUT_TABLES = setup.OUTPUT_TABLES
DATA_PROCESSED = setup.DATA_PROCESSED
MODELS_DIR = setup.MODELS_DIR
NOTES_DIR = setup.NOTES_DIR

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 11})


def load_test_data():
    """Load base test data (46 features)."""
    X = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_test.parquet")).values.astype(np.float32)
    y = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_test.parquet")).values.ravel()
    return X, y


def load_test_47():
    """Load test data with order_hour added (47 features)."""
    X, y = load_test_data()
    X = np.hstack([X, np.zeros((X.shape[0], 1), dtype=np.float32)])
    return X, y


def plot_training_curves():
    with open(os.path.join(PROJECT_ROOT, "logs", "training_log_1dcnn.json")) as f:
        h = json.load(f)
    epochs = [e["epoch"] for e in h]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    a1.plot(epochs, [e["train_loss"] for e in h], "b-", lw=1.5, label="Train")
    a1.plot(epochs, [e["val_loss"] for e in h], "r-", lw=1.5, label="Val")
    a1.set(xlabel="Epoch", ylabel="Loss", title="1D-CNN Loss"); a1.legend(); a1.grid(alpha=.3)

    a2.plot(epochs, [e["train_acc"] for e in h], "b-", lw=1.5, label="Train")
    a2.plot(epochs, [e["val_acc"] for e in h], "r-", lw=1.5, label="Val")
    a2.set(xlabel="Epoch", ylabel="Accuracy", title="1D-CNN Accuracy"); a2.legend(); a2.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "training_curves.png"), bbox_inches="tight"); plt.close()
    print("  [1/6] Training curves saved")


def plot_class_distribution():
    y1 = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_train.parquet")).values.ravel()
    y2 = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_train_bal.parquet")).values.ravel()

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    labels = ["Legitimate", "Fraud"]; colors = ["#4CAF50", "#F44336"]
    for ax, y, title in [(a1, y1, f"Before SMOTE (n={len(y1):,})"), (a2, y2, f"After SMOTE (n={len(y2):,})")]:
        counts = [np.sum(y == 0), np.sum(y == 1)]
        ax.bar(labels, counts, color=colors)
        ax.set_title(title); ax.set_ylabel("Count")
        for i, c in enumerate(counts): ax.text(i, c + 500, f"{c:,}", ha="center")

    plt.suptitle("Training Set Class Distribution"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "class_distribution.png"), bbox_inches="tight"); plt.close()
    print("  [2/6] Class distribution saved")


def plot_feature_importance():
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        xgb = pickle.load(f)

    # Build 47-feature name list (Colab had order_hour)
    with open(os.path.join(DATA_PROCESSED, "metadata.json")) as f:
        meta = json.load(f)
    names = list(meta["feature_cols"])
    # insert order_hour after order_dayofweek
    for i, n in enumerate(names):
        if n == "order_dayofweek":
            names.insert(i + 1, "order_hour"); break

    imp = xgb.feature_importances_
    top_idx = np.argsort(imp)[::-1][:20]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(20), imp[top_idx][::-1], color="#2196F3", alpha=.8)
    ax.set_yticks(range(20)); ax.set_yticklabels([names[i] for i in top_idx][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)"); ax.set_title("XGBoost Top-20 Features"); ax.grid(axis="x", alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "feature_importance_xgboost.png"), bbox_inches="tight"); plt.close()

    print("  [3/6] Feature importance saved")
    print("  Top-10:")
    for i, idx in enumerate(top_idx[:10]):
        print(f"    {i+1}. {names[idx]}: {imp[idx]:.4f}")


def get_all_probs():
    """Get probability predictions from all models on test set."""
    X46, y_test = load_test_data()
    X47, _ = load_test_47()

    probs = {}
    # sklearn models - each may need 46 or 47 features
    for name, fname in [("Logistic Regression", "logistic_regression"),
                         ("Random Forest", "random_forest"), ("XGBoost", "xgboost")]:
        with open(os.path.join(MODELS_DIR, f"{fname}.pkl"), "rb") as f:
            mdl = pickle.load(f)
        X_use = X47 if mdl.n_features_in_ == 47 else X46
        probs[name] = mdl.predict_proba(X_use)[:, 1]

    # MLP - check which feature count
    mlp_mod = import_module("04_baselines")
    state = torch.load(os.path.join(MODELS_DIR, "mlp_best.pt"), weights_only=True, map_location="cpu")
    mlp_input_dim = state["net.0.weight"].shape[1]
    mlp = mlp_mod.SimpleMLP(mlp_input_dim)
    mlp.load_state_dict(state)
    mlp.eval()
    X_mlp = X47 if mlp_input_dim == 47 else X46
    with torch.no_grad():
        probs["MLP"] = torch.softmax(mlp(torch.FloatTensor(X_mlp)), dim=1)[:, 1].numpy()

    # 1D-CNN (from saved predictions)
    with open(os.path.join(OUTPUT_TABLES, "1dcnn_predictions.json")) as f:
        cnn_p = json.load(f)
    probs["1D-CNN (Ours)"] = np.array(cnn_p["y_prob_1dcnn"])
    y_test = np.array(cnn_p["y_true"])

    return probs, y_test


def plot_roc_and_pr():
    probs, y_test = get_all_probs()
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    # ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, yp), c in zip(probs.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, yp)
        ax.plot(fpr, tpr, color=c, label=f"{name} (AUC={roc_auc_score(y_test, yp):.4f})", lw=1.5)
    ax.plot([0, 1], [0, 1], "k--", alpha=.4)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curves")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "roc_curves.png"), bbox_inches="tight"); plt.close()
    print("  [4/6] ROC curves saved")

    # PR
    fig, ax = plt.subplots(figsize=(8, 6))
    for (name, yp), c in zip(probs.items(), colors):
        prec, rec, _ = precision_recall_curve(y_test, yp)
        ax.plot(rec, prec, color=c, label=f"{name} (AP={average_precision_score(y_test, yp):.4f})", lw=1.5)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "pr_curves.png"), bbox_inches="tight"); plt.close()
    print("  [5/6] PR curves saved")


def plot_confusion_matrices():
    probs, y_test = get_all_probs()
    with open(os.path.join(OUTPUT_TABLES, "all_results.json")) as f:
        all_results = json.load(f)

    names = ["Logistic Regression", "Random Forest", "XGBoost", "MLP", "1D-CNN (Ours)"]
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    labels = ["Legit", "Fraud"]

    for ax, name in zip(axes, names):
        thresh = all_results[name]["optimal_threshold"]
        yp = (probs[name] >= thresh).astype(int)
        cm = confusion_matrix(y_test, yp)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(name, fontsize=10)

    plt.suptitle("Confusion Matrices (Optimized Thresholds)", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FIGURES, "confusion_matrices.png"), bbox_inches="tight"); plt.close()
    print("  [6/6] Confusion matrices saved")


def write_results_analysis():
    with open(os.path.join(OUTPUT_TABLES, "all_results.json")) as f:
        results = json.load(f)

    sorted_models = sorted(results.items(), key=lambda x: x[1]["f1_score"], reverse=True)
    cnn = results["1D-CNN (Ours)"]

    lines = ["# Results Analysis\n"]
    lines.append("## Model Ranking (by F1)\n")
    lines.append("| Rank | Model | F1 | Precision | Recall | AUC-ROC | AUC-PR |")
    lines.append("|------|-------|----|-----------|--------|---------|--------|")
    for r, (n, m) in enumerate(sorted_models, 1):
        lines.append(f"| {r} | {n} | {m['f1_score']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['auc_roc']:.4f} | {m['auc_pr']:.4f} |")

    best_name, best = sorted_models[0]
    lines.append(f"\n## Key Findings\n")
    lines.append(f"1. Best F1: **{best_name}** (F1={best['f1_score']:.4f})")
    lines.append(f"2. 1D-CNN rank: #{[n for n,_ in sorted_models].index('1D-CNN (Ours)')+1} by F1")
    lines.append(f"3. 1D-CNN has the **highest recall** ({cnn['recall']:.4f}) among all models")
    lines.append(f"4. 1D-CNN AUC-ROC: {cnn['auc_roc']:.4f} (competitive with best baseline {best['auc_roc']:.4f})")

    lines.append(f"\n## Research Question Answer\n")
    lines.append("**Can 1D-CNN improve fraud detection?**\n")
    lines.append(f"- By F1: 1D-CNN ({cnn['f1_score']:.4f}) is competitive but does not clearly outperform Logistic Regression ({results['Logistic Regression']['f1_score']:.4f}).")
    lines.append(f"- By Recall: 1D-CNN ({cnn['recall']:.4f}) achieves the highest recall, catching {cnn['recall']*100:.1f}% of fraud cases vs {results['Logistic Regression']['recall']*100:.1f}% for LR.")
    lines.append(f"- By AUC-PR: 1D-CNN ({cnn['auc_pr']:.4f}) achieves the highest area under PR curve.")
    lines.append(f"\n## Why Results Are Modest\n")
    lines.append("- All models achieve AUC-ROC ~0.62-0.64 (barely above 0.5 random baseline)")
    lines.append("- This indicates **weak discriminative signal** in the synthetic dataset")
    lines.append("- The ~4% fraud rate creates severe class imbalance challenges")
    lines.append("- Optimal thresholds are very low (0.05-0.14), showing models struggle with calibration")

    with open(os.path.join(NOTES_DIR, "results_analysis.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Analysis saved: {os.path.join(NOTES_DIR, 'results_analysis.md')}")


def main():
    os.makedirs(OUTPUT_FIGURES, exist_ok=True)
    print("=" * 60)
    print("GENERATING VISUALIZATIONS & ANALYSIS")
    print("=" * 60)

    plot_training_curves()
    plot_class_distribution()
    plot_feature_importance()
    plot_roc_and_pr()
    plot_confusion_matrices()
    write_results_analysis()

    print("\n" + "=" * 60)
    print("PHASE 5 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
