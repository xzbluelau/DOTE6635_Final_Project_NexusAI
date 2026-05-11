"""
01_data_loading.py — Load 4 Kaggle CSVs, join into a single DataFrame, profile data.

Usage: python code/01_data_loading.py
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "code"))

from importlib import import_module
setup = import_module("00_setup")
from utils import load_dataset

NOTES_DIR = setup.NOTES_DIR
DATA_RAW = setup.DATA_RAW
DATA_PROCESSED = setup.DATA_PROCESSED
SEED = setup.SEED


def main():
    setup.set_seed()

    # -----------------------------------------------------------------------
    # Step 1: Load all 4 CSVs
    # -----------------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading raw CSVs")
    print("=" * 60)

    transactions = load_dataset("transactions", DATA_RAW)
    customers = load_dataset("customers", DATA_RAW)
    products = load_dataset("products", DATA_RAW)
    behavior = load_dataset("behavior", DATA_RAW)

    tables = {
        "transactions": transactions,
        "customers": customers,
        "products": products,
        "behavior": behavior,
    }

    for name, df in tables.items():
        print(f"\n--- {name}.csv ---")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Dtypes:\n{df.dtypes.to_string()}")
        print(f"  Missing values:\n{df.isnull().sum().to_string()}")
        print(f"  Sample (first 3 rows):\n{df.head(3).to_string()}")

    # -----------------------------------------------------------------------
    # Step 2: Join tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Joining tables")
    print("=" * 60)

    # Start from transactions (fact table)
    df = transactions.merge(customers, on="customer_id", how="left")
    print(f"  After joining customers: {df.shape}")

    df = df.merge(products, on="product_id", how="left")
    print(f"  After joining products: {df.shape}")

    df = df.merge(behavior, on="customer_id", how="left")
    print(f"  After joining behavior: {df.shape}")

    # -----------------------------------------------------------------------
    # Step 3: Data audit
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: Data audit")
    print("=" * 60)

    # Missing values
    missing = df.isnull().sum()
    missing_pct = (df.isnull().mean() * 100).round(2)
    print("\n  Missing values (count / %):")
    for col in df.columns:
        if missing[col] > 0:
            print(f"    {col}: {missing[col]} ({missing_pct[col]}%)")
    if missing.sum() == 0:
        print("    No missing values in any column.")

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\n  Duplicate rows: {dup_count}")

    # Target distribution
    fraud_count = df["fraud_label"].sum()
    fraud_rate = df["fraud_label"].mean() * 100
    print(f"\n  Target (fraud_label):")
    print(f"    Legitimate (0): {len(df) - fraud_count} ({100 - fraud_rate:.2f}%)")
    print(f"    Fraud (1): {fraud_count} ({fraud_rate:.2f}%)")

    # Numeric summary
    print(f"\n  Numeric columns summary:")
    print(df.describe().to_string())

    # Categorical value counts
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print(f"\n  Categorical columns value counts:")
    for col in cat_cols:
        n_unique = df[col].nunique()
        print(f"    {col}: {n_unique} unique values")
        if n_unique <= 20:
            print(f"      {df[col].value_counts().to_dict()}")

    # -----------------------------------------------------------------------
    # Step 4: Save data profile report
    # -----------------------------------------------------------------------
    report = []
    report.append("# Data Profile Report\n")
    report.append(f"Generated from 4 CSVs joined on customer_id / product_id.\n")
    report.append(f"## Joined Dataset: {df.shape[0]} rows x {df.shape[1]} columns\n")

    report.append("## Source Tables\n")
    report.append("| Table | Rows | Cols | Key Columns |")
    report.append("|-------|------|------|-------------|")
    report.append(f"| transactions | {transactions.shape[0]} | {transactions.shape[1]} | transaction_id, customer_id, product_id, fraud_label |")
    report.append(f"| customers | {customers.shape[0]} | {customers.shape[1]} | customer_id (PK) |")
    report.append(f"| products | {products.shape[0]} | {products.shape[1]} | product_id (PK) |")
    report.append(f"| behavior | {behavior.shape[0]} | {behavior.shape[1]} | customer_id (FK) |")

    report.append("\n## Missing Values\n")
    if missing.sum() == 0:
        report.append("No missing values detected.\n")
    else:
        report.append("| Column | Missing | % |")
        report.append("|--------|---------|---|")
        for col in df.columns:
            if missing[col] > 0:
                report.append(f"| {col} | {missing[col]} | {missing_pct[col]}% |")

    report.append(f"\n## Duplicate Rows: {dup_count}\n")

    report.append(f"\n## Target Distribution\n")
    report.append(f"- Legitimate (0): {len(df) - fraud_count} ({100 - fraud_rate:.2f}%)")
    report.append(f"- Fraud (1): {fraud_count} ({fraud_rate:.2f}%)")
    report.append(f"- Fraud rate: {fraud_rate:.2f}%\n")

    report.append("## Column Types\n")
    report.append("### Numeric Columns\n")
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        report.append(f"- `{col}`: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

    report.append("\n### Categorical Columns\n")
    for col in cat_cols:
        report.append(f"- `{col}`: {df[col].nunique()} unique values")

    report_path = os.path.join(NOTES_DIR, "data_profile.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n  Data profile saved to: {report_path}")

    # -----------------------------------------------------------------------
    # Step 5: Save joined dataset
    # -----------------------------------------------------------------------
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    output_path = os.path.join(DATA_PROCESSED, "joined_raw.parquet")
    df.to_parquet(output_path, index=False)
    print(f"  Joined dataset saved to: {output_path}")
    print(f"\n  Final shape: {df.shape}")

    return df


if __name__ == "__main__":
    main()
