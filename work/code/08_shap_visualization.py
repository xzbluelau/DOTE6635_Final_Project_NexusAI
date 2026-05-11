"""
08_shap_visualization.py - SHAP value computation and visualization for 1D-CNN.

Uses shap.Explainer (model-agnostic) to compute SHAP values, then generates
beeswarm and bar plots following the SHAP documentation pattern.

Usage: python code/08_shap_visualization.py
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")
model_mod = import_module("03_model_1dcnn")

DATA_PROCESSED = setup.DATA_PROCESSED
MODELS_DIR = setup.MODELS_DIR
OUTPUT_FIGURES = setup.OUTPUT_FIGURES
OUTPUT_TABLES = setup.OUTPUT_TABLES


def load_model_and_data():
    """Load 1D-CNN model and test data."""
    state = torch.load(
        os.path.join(MODELS_DIR, "1dcnn_best.pt"),
        map_location="cpu", weights_only=True
    )
    num_features = state["classifier.weight"].shape[1]
    model = model_mod.FraudDetectionCNN1D(num_features=num_features)
    model.load_state_dict(state)
    model.eval()

    X_test = pd.read_parquet(
        os.path.join(DATA_PROCESSED, "X_test.parquet")
    ).values.astype(np.float32)
    y_test = pd.read_parquet(
        os.path.join(DATA_PROCESSED, "y_test.parquet")
    ).values.ravel()

    with open(os.path.join(DATA_PROCESSED, "metadata.json")) as f:
        meta = json.load(f)
    feature_names = list(meta["feature_cols"])

    return model, X_test, y_test, feature_names


def predict_proba(x):
    """Wrapper for shap.Explainer — returns fraud probability."""
    model = predict_proba.model
    model.eval()
    with torch.no_grad():
        t = torch.FloatTensor(np.array(x, dtype=np.float32)).unsqueeze(1)
        logits = model(t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    return probs


def main():
    print("=" * 60)
    print("SHAP ANALYSIS FOR 1D-CNN")
    print("=" * 60)

    model, X_test, y_test, feature_names = load_model_and_data()
    print(f"Model loaded. Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    # Attach model to predict function
    predict_proba.model = model

    # Subsample for SHAP (model-agnostic is expensive)
    rng = np.random.RandomState(42)
    n_background = 100
    n_explain = 500

    bg_idx = rng.choice(len(X_test), size=n_background, replace=False)
    explain_idx = rng.choice(len(X_test), size=n_explain, replace=False)

    X_background = X_test[bg_idx]
    X_explain = X_test[explain_idx]

    print(f"\nBackground samples: {n_background}")
    print(f"Explain samples: {n_explain}")

    # Compute SHAP values using model-agnostic Explainer (Permutation)
    print("\nComputing SHAP values (this may take a few minutes)...")
    explainer = shap.Explainer(predict_proba, X_background, algorithm="permutation")
    shap_values = explainer(X_explain)

    print(f"SHAP values shape: {shap_values.values.shape}")
    print(f"Base value: {shap_values.base_values[0]:.4f}")

    # Wrap in Explanation with feature names for nice plots
    shap_exp = shap.Explanation(
        values=shap_values.values,
        base_values=shap_values.base_values,
        data=X_explain,
        feature_names=feature_names,
    )

    # ====== Plot 1: SHAP Beeswarm Plot ======
    print("\nGenerating SHAP beeswarm plot...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(shap_exp, max_display=20, show=False)
    plt.title("1D-CNN SHAP Feature Attribution (Beeswarm)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path_beeswarm = os.path.join(OUTPUT_FIGURES, "shap_beeswarm_1dcnn.png")
    plt.savefig(path_beeswarm, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path_beeswarm}")

    # ====== Plot 2: SHAP Bar Plot ======
    print("Generating SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.bar(shap_exp, max_display=20, show=False)
    plt.title("1D-CNN SHAP Feature Importance (Mean |SHAP|)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path_bar = os.path.join(OUTPUT_FIGURES, "shap_bar_1dcnn.png")
    plt.savefig(path_bar, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  Saved: {path_bar}")

    # ====== Save SHAP values for reference ======
    mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
    shap_results = {
        "method": "shap.Explainer (permutation, model-agnostic)",
        "n_background": n_background,
        "n_explain": n_explain,
        "base_value": float(shap_values.base_values[0]),
        "mean_abs_shap": {
            feature_names[i]: float(mean_abs_shap[i])
            for i in range(len(feature_names))
        },
    }
    out_path = os.path.join(OUTPUT_TABLES, "shap_values_1dcnn.json")
    with open(out_path, "w") as f:
        json.dump(shap_results, f, indent=2)
    print(f"  SHAP values saved: {out_path}")

    # Print top features
    sorted_shap = sorted(
        shap_results["mean_abs_shap"].items(),
        key=lambda x: x[1], reverse=True
    )
    print("\nTop-15 features by mean |SHAP|:")
    for i, (name, val) in enumerate(sorted_shap[:15]):
        print(f"  {i+1:2d}. {name}: {val:.6f}")

    print("\n" + "=" * 60)
    print("SHAP ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
