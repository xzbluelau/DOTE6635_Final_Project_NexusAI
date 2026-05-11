"""
02_feature_engineering.py — Feature engineering for fraud detection.

Steps:
1. Load joined dataset
2. Drop non-feature columns (IDs, other targets)
3. Extract temporal features from order_date and registration_date
4. One-hot encode categorical columns
5. Train/Val/Test split (stratified, before any transformation)
6. Standardize numerical features (fit on train only)
7. SMOTE oversampling (training set only)
8. Save all processed splits + scaler + metadata

Usage: python code/02_feature_engineering.py
"""

import os
import sys
import json
import pickle

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")

DATA_PROCESSED = setup.DATA_PROCESSED
NOTES_DIR = setup.NOTES_DIR
MODELS_DIR = setup.MODELS_DIR
SEED = setup.SEED
TARGET_COL = setup.TARGET_COL
TRAIN_RATIO = setup.TRAIN_RATIO
VAL_RATIO = setup.VAL_RATIO


def main():
    setup.set_seed()

    # ------------------------------------------------------------------
    # Step 1: Load joined dataset
    # ------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading joined dataset")
    print("=" * 60)

    input_path = os.path.join(DATA_PROCESSED, "joined_raw.parquet")
    df = pd.read_parquet(input_path)
    print(f"  Loaded: {df.shape}")

    # ------------------------------------------------------------------
    # Step 2: Drop non-feature columns
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Dropping non-feature columns")
    print("=" * 60)

    cols_to_drop = [
        "transaction_id",       # identifier
        "customer_id",          # identifier
        "product_id",           # identifier
        "churn_label",          # different target (from customers)
        "behavior_churn_signal", # potential label leakage (from behavior)
    ]
    # Only drop columns that exist
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped: {cols_to_drop}")
    print(f"  Remaining columns ({len(df.columns)}): {list(df.columns)}")

    # ------------------------------------------------------------------
    # Step 3: Temporal feature extraction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Temporal feature extraction")
    print("=" * 60)

    # From order_date
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_hour"] = df["order_date"].dt.hour
    df["order_dayofweek"] = df["order_date"].dt.dayofweek    # 0=Mon, 6=Sun
    df["order_month"] = df["order_date"].dt.month
    df["order_is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)

    # From registration_date — customer tenure at time of order
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    df["customer_tenure_days"] = (df["order_date"] - df["registration_date"]).dt.days

    # Drop original date columns
    df = df.drop(columns=["order_date", "registration_date"])

    print("  Derived features:")
    print(f"    order_hour: {df['order_hour'].min()}-{df['order_hour'].max()}")
    print(f"    order_dayofweek: {df['order_dayofweek'].min()}-{df['order_dayofweek'].max()} (0=Mon, 6=Sun)")
    print(f"    order_month: {df['order_month'].min()}-{df['order_month'].max()}")
    print(f"    order_is_weekend: {df['order_is_weekend'].value_counts().to_dict()}")
    print(f"    customer_tenure_days: {df['customer_tenure_days'].min()}-{df['customer_tenure_days'].max()}")

    # ------------------------------------------------------------------
    # Step 4: Identify column types
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: Column type classification")
    print("=" * 60)

    categorical_cols = [
        "payment_method",
        "device_type",
        "gender",
        "country",
        "category",
    ]
    # Only keep those that exist
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    binary_cols = ["order_is_weekend"]

    numerical_cols = [
        col for col in df.select_dtypes(include=["int64", "float64"]).columns
        if col != TARGET_COL and col not in binary_cols
    ]

    print(f"  Numerical ({len(numerical_cols)}): {numerical_cols}")
    print(f"  Categorical ({len(categorical_cols)}): {categorical_cols}")
    print(f"  Binary ({len(binary_cols)}): {binary_cols}")
    print(f"  Target: {TARGET_COL}")

    # ------------------------------------------------------------------
    # Step 5: One-hot encode categoricals
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: One-hot encoding")
    print("=" * 60)

    print(f"  Encoding {len(categorical_cols)} columns...")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
    print(f"  Shape after encoding: {df.shape}")

    # Re-identify numerical cols (includes new dummy cols which are uint8, treat as numeric)
    # The original numerical cols are already identified; new dummy cols are 0/1, no scaling needed
    all_feature_cols = [c for c in df.columns if c != TARGET_COL]
    # Numerical cols to scale = original numerical cols (continuous)
    # Dummy cols and binary cols stay as-is
    cols_to_scale = [c for c in numerical_cols if c in df.columns]

    print(f"  Columns to scale: {len(cols_to_scale)}")
    print(f"  Total features: {len(all_feature_cols)}")

    # ------------------------------------------------------------------
    # Step 6: Train/Val/Test split (BEFORE scaling and SMOTE)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 6: Train/Validation/Test split")
    print("=" * 60)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # First split: 85% temp, 15% test
    test_size = 1 - TRAIN_RATIO - VAL_RATIO  # 0.15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    # Second split from temp: proportional to get 70/15 overall
    val_ratio_of_temp = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)  # 0.15/0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, random_state=SEED, stratify=y_temp
    )

    print(f"  Train:      {X_train.shape[0]:,} samples (fraud: {y_train.sum():,}, {y_train.mean()*100:.2f}%)")
    print(f"  Validation: {X_val.shape[0]:,} samples (fraud: {y_val.sum():,}, {y_val.mean()*100:.2f}%)")
    print(f"  Test:       {X_test.shape[0]:,} samples (fraud: {y_test.sum():,}, {y_test.mean()*100:.2f}%)")

    # ------------------------------------------------------------------
    # Step 7: Standardize numerical features (fit on train only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 7: Standardizing numerical features")
    print("=" * 60)

    scaler = StandardScaler()
    X_train[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    print(f"  Scaled {len(cols_to_scale)} features: fit on train, applied to all splits")

    # ------------------------------------------------------------------
    # Step 8: SMOTE oversampling (training set only)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 8: SMOTE oversampling on training set")
    print("=" * 60)

    print(f"  Before SMOTE: {X_train.shape[0]:,} samples (fraud: {y_train.sum():,}, legit: {len(y_train)-y_train.sum():,})")
    smote = SMOTE(random_state=SEED)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"  After SMOTE:  {X_train_bal.shape[0]:,} samples (fraud: {y_train_bal.sum():,}, legit: {len(y_train_bal)-y_train_bal.sum():,})")

    # ------------------------------------------------------------------
    # Step 9: Save everything
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 9: Saving processed data")
    print("=" * 60)

    # Save data splits
    for name, data in [
        ("X_train_bal", X_train_bal),
        ("y_train_bal", y_train_bal),
        ("X_train", X_train),
        ("y_train", y_train),
        ("X_val", X_val),
        ("y_val", y_val),
        ("X_test", X_test),
        ("y_test", y_test),
    ]:
        path = os.path.join(DATA_PROCESSED, f"{name}.parquet")
        if isinstance(data, pd.Series):
            data.to_frame(name=name).to_parquet(path, index=False)
        else:
            data.to_parquet(path, index=False)
        print(f"  Saved: {name} → {path}")

    # Save scaler
    os.makedirs(MODELS_DIR, exist_ok=True)
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved: scaler → {scaler_path}")

    # Save metadata
    metadata = {
        "feature_cols": list(X_train.columns),
        "cols_to_scale": cols_to_scale,
        "categorical_cols_original": categorical_cols,
        "numerical_cols": numerical_cols,
        "binary_cols": binary_cols,
        "target_col": TARGET_COL,
        "n_features": len(X_train.columns),
        "seed": SEED,
        "train_size": int(len(X_train_bal)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test)),
        "train_fraud_before_smote": int(y_train.sum()),
        "train_fraud_after_smote": int(y_train_bal.sum()),
        "test_fraud": int(y_test.sum()),
        "fraud_rate_test": float(y_test.mean()),
    }
    meta_path = os.path.join(DATA_PROCESSED, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved: metadata → {meta_path}")

    # ------------------------------------------------------------------
    # Step 10: Feature engineering documentation
    # ------------------------------------------------------------------
    report = []
    report.append("# Feature Engineering Report\n")
    report.append(f"## Feature Summary\n")
    report.append(f"- Total features: {len(X_train.columns)}")
    report.append(f"- Features to scale: {len(cols_to_scale)}")
    report.append(f"- One-hot encoded categories: {categorical_cols}")
    report.append(f"- Binary features: {binary_cols}")
    report.append(f"- Target: {TARGET_COL}\n")

    report.append("## Features by Source Table\n")
    report.append("### From transactions.csv")
    report.append("| Feature | Type | Preprocessing |")
    report.append("|---------|------|---------------|")
    txn_features = {
        "order_value": "Numerical / StandardScaler",
        "discount_applied": "Numerical / StandardScaler",
        "shipping_delay_days": "Numerical / StandardScaler",
        "payment_method": "Categorical / One-hot (5 values)",
        "device_type": "Categorical / One-hot (3 values)",
        "order_hour": "Numerical (derived) / StandardScaler",
        "order_dayofweek": "Numerical (derived) / StandardScaler",
        "order_month": "Numerical (derived) / StandardScaler",
        "order_is_weekend": "Binary (derived) / Keep as-is",
    }
    for feat, desc in txn_features.items():
        report.append(f"| {feat} | {desc} |")

    report.append("\n### From customers.csv")
    cust_features = {
        "age": "Numerical / StandardScaler",
        "gender": "Categorical / One-hot (3 values)",
        "country": "Categorical / One-hot (8 values)",
        "loyalty_score": "Numerical / StandardScaler",
        "lifetime_value": "Numerical / StandardScaler",
        "customer_tenure_days": "Numerical (derived) / StandardScaler",
    }
    report.append("| Feature | Type | Preprocessing |")
    report.append("|---------|------|---------------|")
    for feat, desc in cust_features.items():
        report.append(f"| {feat} | {desc} |")

    report.append("\n### From products.csv")
    prod_features = {
        "category": "Categorical / One-hot (8 values)",
        "price": "Numerical / StandardScaler",
        "margin_percentage": "Numerical / StandardScaler",
        "popularity_score": "Numerical / StandardScaler",
    }
    report.append("| Feature | Type | Preprocessing |")
    report.append("|---------|------|---------------|")
    for feat, desc in prod_features.items():
        report.append(f"| {feat} | {desc} |")

    report.append("\n### From behavior.csv")
    beh_features = {
        "avg_session_time": "Numerical / StandardScaler",
        "pages_per_session": "Numerical / StandardScaler",
        "cart_abandon_rate": "Numerical / StandardScaler",
        "return_rate": "Numerical / StandardScaler",
        "support_tickets": "Numerical / StandardScaler",
        "review_score": "Numerical / StandardScaler",
    }
    report.append("| Feature | Type | Preprocessing |")
    report.append("|---------|------|---------------|")
    for feat, desc in beh_features.items():
        report.append(f"| {feat} | {desc} |")

    report.append(f"\n### Dropped columns")
    report.append("- `transaction_id`, `customer_id`, `product_id` — identifiers")
    report.append("- `churn_label` — different target variable")
    report.append("- `behavior_churn_signal` — potential label leakage")
    report.append("- `order_date`, `registration_date` — replaced by derived temporal features")

    report.append(f"\n## Data Splits (stratified)")
    report.append(f"| Split | Samples | Fraud | Legit | Fraud Rate |")
    report.append(f"|-------|---------|-------|-------|------------|")
    report.append(f"| Train (pre-SMOTE) | {len(X_train):,} | {int(y_train.sum()):,} | {int(len(y_train)-y_train.sum()):,} | {y_train.mean()*100:.2f}% |")
    report.append(f"| Train (post-SMOTE) | {len(X_train_bal):,} | {int(y_train_bal.sum()):,} | {int(len(y_train_bal)-y_train_bal.sum()):,} | {y_train_bal.mean()*100:.2f}% |")
    report.append(f"| Validation | {len(X_val):,} | {int(y_val.sum()):,} | {int(len(y_val)-y_val.sum()):,} | {y_val.mean()*100:.2f}% |")
    report.append(f"| Test | {len(X_test):,} | {int(y_test.sum()):,} | {int(len(y_test)-y_test.sum()):,} | {y_test.mean()*100:.2f}% |")

    report_path = os.path.join(NOTES_DIR, "feature_engineering.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"  Final feature count: {len(X_train.columns)}")
    print(f"  Training samples (balanced): {len(X_train_bal):,}")
    print(f"  Validation samples: {len(X_val):,}")
    print(f"  Test samples: {len(X_test):,}")


if __name__ == "__main__":
    main()
