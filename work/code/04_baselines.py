"""
04_baselines.py — Train and evaluate baseline models for fraud detection.

Baselines:
1. Logistic Regression
2. Random Forest
3. XGBoost
4. Simple 2-layer MLP (PyTorch)

All models use the same SMOTE-balanced training data.
Thresholds are optimized on the validation set (maximizing F1),
then evaluated on the held-out test set.

Usage: python code/04_baselines.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")
from utils import compute_metrics, save_results, find_optimal_threshold

DATA_PROCESSED = setup.DATA_PROCESSED
OUTPUT_TABLES = setup.OUTPUT_TABLES
MODELS_DIR = setup.MODELS_DIR
SEED = setup.SEED
DEVICE = setup.DEVICE


# ===========================================================================
# Simple MLP baseline
# ===========================================================================
class SimpleMLP(nn.Module):
    """2-layer MLP: input -> 128 -> 64 -> 2."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=50, batch_size=128):
    """Train the MLP baseline."""
    model = SimpleMLP(input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    X_t = torch.FloatTensor(np.array(X_train, dtype=np.float32)).to(DEVICE)
    y_t = torch.LongTensor(np.array(y_train, dtype=np.int64)).to(DEVICE)
    X_v = torch.FloatTensor(np.array(X_val, dtype=np.float32)).to(DEVICE)
    y_v = torch.LongTensor(np.array(y_val, dtype=np.int64)).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(X_t)

        model.eval()
        with torch.no_grad():
            val_out = model(X_v)
            val_loss = criterion(val_out, y_v).item()

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    model = model.to(DEVICE)
    return model


def predict_with_optimal_threshold(model, X_val, y_val, X_test, is_sklearn=True):
    """Find optimal threshold on validation set, apply to test set.

    Returns: (y_pred_test, y_prob_test, optimal_threshold)
    """
    if is_sklearn:
        val_prob = model.predict_proba(X_val)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
    else:
        # PyTorch model
        model.eval()
        with torch.no_grad():
            X_v = torch.FloatTensor(np.array(X_val, dtype=np.float32)).to(DEVICE)
            X_t = torch.FloatTensor(np.array(X_test, dtype=np.float32)).to(DEVICE)
            val_prob = torch.softmax(model(X_v), dim=1)[:, 1].cpu().numpy()
            test_prob = torch.softmax(model(X_t), dim=1)[:, 1].cpu().numpy()

    # Find optimal threshold on validation set
    opt_thresh, val_f1 = find_optimal_threshold(y_val, val_prob)
    print(f"  Optimal threshold (from val): {opt_thresh:.2f} | Val F1: {val_f1:.4f}")

    # Apply to test set
    test_pred = (test_prob >= opt_thresh).astype(int)
    return test_pred, test_prob, opt_thresh


def main():
    setup.set_seed()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    X_train = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_train_bal.parquet")).values
    y_train = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_train_bal.parquet")).values.ravel()
    X_val = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_val.parquet")).values
    y_val = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_val.parquet")).values.ravel()
    X_test = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_test.parquet")).values
    y_test = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_test.parquet")).values.ravel()

    input_dim = X_train.shape[1]
    print(f"  Train (SMOTE): {X_train.shape}")
    print(f"  Val:  {X_val.shape}")
    print(f"  Test: {X_test.shape}")
    print(f"  Features: {input_dim}")

    X_train = np.array(X_train, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int64)
    y_val = np.array(y_val, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    results = {}

    # ------------------------------------------------------------------
    # 1. Logistic Regression
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. LOGISTIC REGRESSION")
    print("=" * 60)

    t0 = time.time()
    lr = LogisticRegression(max_iter=1000, random_state=SEED, n_jobs=-1)
    lr.fit(X_train, y_train)
    lr_pred, lr_prob, lr_thresh = predict_with_optimal_threshold(lr, X_val, y_val, X_test, is_sklearn=True)
    lr_time = time.time() - t0
    results["Logistic Regression"] = compute_metrics(y_test, lr_pred, lr_prob)
    results["Logistic Regression"]["train_time_sec"] = round(lr_time, 2)
    results["Logistic Regression"]["optimal_threshold"] = lr_thresh
    print(f"  Test F1={results['Logistic Regression']['f1_score']:.4f} | "
          f"AUC-ROC={results['Logistic Regression']['auc_roc']:.4f} | "
          f"Time={lr_time:.1f}s")

    # ------------------------------------------------------------------
    # 2. Random Forest
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. RANDOM FOREST")
    print("=" * 60)

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred, rf_prob, rf_thresh = predict_with_optimal_threshold(rf, X_val, y_val, X_test, is_sklearn=True)
    rf_time = time.time() - t0
    results["Random Forest"] = compute_metrics(y_test, rf_pred, rf_prob)
    results["Random Forest"]["train_time_sec"] = round(rf_time, 2)
    results["Random Forest"]["optimal_threshold"] = rf_thresh
    print(f"  Test F1={results['Random Forest']['f1_score']:.4f} | "
          f"AUC-ROC={results['Random Forest']['auc_roc']:.4f} | "
          f"Time={rf_time:.1f}s")

    # ------------------------------------------------------------------
    # 3. XGBoost
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("3. XGBOOST")
    print("=" * 60)

    t0 = time.time()
    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=SEED, n_jobs=-1, eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_train, y_train)
    xgb_pred, xgb_prob, xgb_thresh = predict_with_optimal_threshold(xgb, X_val, y_val, X_test, is_sklearn=True)
    xgb_time = time.time() - t0
    results["XGBoost"] = compute_metrics(y_test, xgb_pred, xgb_prob)
    results["XGBoost"]["train_time_sec"] = round(xgb_time, 2)
    results["XGBoost"]["optimal_threshold"] = xgb_thresh
    print(f"  Test F1={results['XGBoost']['f1_score']:.4f} | "
          f"AUC-ROC={results['XGBoost']['auc_roc']:.4f} | "
          f"Time={xgb_time:.1f}s")

    # ------------------------------------------------------------------
    # 4. MLP
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("4. MLP (PyTorch)")
    print("=" * 60)

    t0 = time.time()
    mlp = train_mlp(X_train, y_train, X_val, y_val, input_dim, epochs=50, batch_size=128)
    mlp_pred, mlp_prob, mlp_thresh = predict_with_optimal_threshold(mlp, X_val, y_val, X_test, is_sklearn=False)
    mlp_time = time.time() - t0
    results["MLP"] = compute_metrics(y_test, mlp_pred, mlp_prob)
    results["MLP"]["train_time_sec"] = round(mlp_time, 2)
    results["MLP"]["optimal_threshold"] = mlp_thresh
    print(f"  Test F1={results['MLP']['f1_score']:.4f} | "
          f"AUC-ROC={results['MLP']['auc_roc']:.4f} | "
          f"Time={mlp_time:.1f}s")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING BASELINE RESULTS")
    print("=" * 60)

    os.makedirs(OUTPUT_TABLES, exist_ok=True)
    out_path = os.path.join(OUTPUT_TABLES, "baseline_results.json")
    save_results(results, out_path)
    print(f"  Saved: {out_path}")

    import pickle
    for name, mdl in [("logistic_regression", lr), ("random_forest", rf), ("xgboost", xgb)]:
        p = os.path.join(MODELS_DIR, f"{name}.pkl")
        with open(p, "wb") as f:
            pickle.dump(mdl, f)
        print(f"  Saved: {p}")

    mlp_path = os.path.join(MODELS_DIR, "mlp_best.pt")
    torch.save(mlp.state_dict(), mlp_path)
    print(f"  Saved: {mlp_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINE RESULTS SUMMARY (with optimal thresholds)")
    print("=" * 60)
    print(f"{'Model':<22} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>7} {'AUC-ROC':>8} {'AUC-PR':>8} {'Thresh':>7} {'Time(s)':>8}")
    print("-" * 100)
    for name, m in results.items():
        print(f"{name:<22} {m['accuracy']:>9.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['f1_score']:>7.4f} {m['auc_roc']:>8.4f} {m['auc_pr']:>8.4f} {m['optimal_threshold']:>7.2f} {m['train_time_sec']:>8.1f}")


if __name__ == "__main__":
    main()
