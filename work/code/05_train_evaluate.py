"""
05_train_evaluate.py — Train 1D-CNN, evaluate all models, produce final comparison.

1. Train the 1D-CNN with residual connections + self-attention
2. Load baseline results from 04_baselines.py
3. Evaluate 1D-CNN on test set
4. Produce final comparison table (all 5 models)
5. Save all results

Usage: python code/05_train_evaluate.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")
cnn_module = import_module("03_model_1dcnn")
from utils import compute_metrics, save_results, find_optimal_threshold

FraudDetectionCNN1D = cnn_module.FraudDetectionCNN1D
FraudDataset = cnn_module.FraudDataset

DATA_PROCESSED = setup.DATA_PROCESSED
OUTPUT_TABLES = setup.OUTPUT_TABLES
MODELS_DIR = setup.MODELS_DIR
LOGS_DIR = setup.LOGS_DIR
SEED = setup.SEED
DEVICE = setup.DEVICE


def train_1dcnn(
    model, train_loader, val_loader,
    epochs=100, lr=1e-3, weight_decay=1e-4,
    patience=15, save_path=None,
):
    """Train the 1D-CNN model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    print(f"  Training on {DEVICE} | epochs={epochs} | lr={lr} | patience={patience}")
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(bx)
            train_correct += (out.argmax(dim=1) == by).sum().item()
            train_total += len(by)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- Validate ----
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                out = model(bx)
                loss = criterion(out, by)
                val_loss += loss.item() * len(bx)
                val_correct += (out.argmax(dim=1) == by).sum().item()
                val_total += len(by)

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)

        # Record
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(DEVICE)

    if save_path:
        torch.save(best_state, save_path)
        print(f"  Best model saved to: {save_path}")

    return model, history


def evaluate_model(model, test_loader):
    """Evaluate model on test set, return predictions and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(DEVICE)
            out = model(bx)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(by.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def main():
    setup.set_seed()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    with open(os.path.join(DATA_PROCESSED, "metadata.json")) as f:
        meta = json.load(f)
    num_features = meta["n_features"]
    print(f"  Features: {num_features}")

    X_train = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_train_bal.parquet")).values
    y_train = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_train_bal.parquet")).values.ravel()
    X_val = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_val.parquet")).values
    y_val = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_val.parquet")).values.ravel()
    X_test = pd.read_parquet(os.path.join(DATA_PROCESSED, "X_test.parquet")).values
    y_test = pd.read_parquet(os.path.join(DATA_PROCESSED, "y_test.parquet")).values.ravel()

    print(f"  Train (SMOTE): {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Build datasets and loaders
    train_ds = FraudDataset(X_train, y_train)
    val_ds = FraudDataset(X_val, y_val)
    test_ds = FraudDataset(X_test, y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    # ------------------------------------------------------------------
    # Train 1D-CNN
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING 1D-CNN WITH RESIDUAL + SELF-ATTENTION")
    print("=" * 60)

    model = FraudDetectionCNN1D(num_features=num_features).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    t0 = time.time()
    model_path = os.path.join(MODELS_DIR, "1dcnn_best.pt")
    model, history = train_1dcnn(
        model, train_loader, val_loader,
        epochs=30, lr=1e-3, weight_decay=1e-4,
        patience=7, save_path=model_path,
    )
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Save training history
    os.makedirs(LOGS_DIR, exist_ok=True)
    hist_path = os.path.join(LOGS_DIR, "training_log_1dcnn.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  Training log saved: {hist_path}")

    # ------------------------------------------------------------------
    # Evaluate 1D-CNN on test set (with optimal threshold)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING 1D-CNN ON TEST SET")
    print("=" * 60)

    # Get probabilities on validation set for threshold tuning
    val_loader_all = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)
    y_val_true, _, y_val_prob = evaluate_model(model, val_loader_all)
    opt_thresh, val_f1 = find_optimal_threshold(y_val_true, y_val_prob)
    print(f"  Optimal threshold (from val): {opt_thresh:.2f} | Val F1: {val_f1:.4f}")

    # Evaluate on test set
    y_true, _, y_prob = evaluate_model(model, test_loader)
    y_pred = (y_prob >= opt_thresh).astype(int)
    cnn_metrics = compute_metrics(y_true, y_pred, y_prob)
    cnn_metrics["train_time_sec"] = round(train_time, 2)
    cnn_metrics["optimal_threshold"] = opt_thresh
    print(f"  Accuracy:  {cnn_metrics['accuracy']:.4f}")
    print(f"  Precision: {cnn_metrics['precision']:.4f}")
    print(f"  Recall:    {cnn_metrics['recall']:.4f}")
    print(f"  F1:        {cnn_metrics['f1_score']:.4f}")
    print(f"  AUC-ROC:   {cnn_metrics['auc_roc']:.4f}")
    print(f"  AUC-PR:    {cnn_metrics['auc_pr']:.4f}")

    # ------------------------------------------------------------------
    # Load baseline results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING BASELINE RESULTS")
    print("=" * 60)

    baseline_path = os.path.join(OUTPUT_TABLES, "baseline_results.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as f:
            all_results = json.load(f)
        print(f"  Loaded baselines: {list(all_results.keys())}")
    else:
        print("  WARNING: baseline_results.json not found. Run 04_baselines.py first.")
        all_results = {}

    # Add 1D-CNN results
    all_results["1D-CNN (Ours)"] = cnn_metrics

    # ------------------------------------------------------------------
    # Final comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON (optimal thresholds)")
    print("=" * 60)

    comparison_rows = []
    for name, m in all_results.items():
        comparison_rows.append({
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1_score"],
            "AUC-ROC": m["auc_roc"],
            "AUC-PR": m["auc_pr"],
            "Threshold": m.get("optimal_threshold", ""),
            "Time(s)": m.get("train_time_sec", ""),
        })

    comp_df = pd.DataFrame(comparison_rows)
    # Sort by F1 descending
    comp_df = comp_df.sort_values("F1", ascending=False).reset_index(drop=True)

    # Print table
    print(comp_df.to_string(index=False))

    # Save as CSV
    comp_path = os.path.join(OUTPUT_TABLES, "model_comparison.csv")
    comp_df.to_csv(comp_path, index=False)
    print(f"\n  Saved: {comp_path}")

    # Save full results as JSON
    full_path = os.path.join(OUTPUT_TABLES, "all_results.json")
    save_results(all_results, full_path)
    print(f"  Saved: {full_path}")

    # Save test predictions for visualization
    pred_data = {
        "y_true": y_true.tolist(),
        "y_pred_1dcnn": y_pred.tolist(),
        "y_prob_1dcnn": y_prob.tolist(),
    }
    pred_path = os.path.join(OUTPUT_TABLES, "1dcnn_predictions.json")
    with open(pred_path, "w") as f:
        json.dump(pred_data, f)
    print(f"  Saved: {pred_path}")


if __name__ == "__main__":
    main()
