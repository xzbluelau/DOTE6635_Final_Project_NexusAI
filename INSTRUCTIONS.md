# Replicating and Extending "A Deep Residual 1D-CNN with Self-Attention for Fraud Transaction Detection in Virtual Economies"

## Project Overview

Replicate and extend Mohammed et al. (2026). The paper proposes a 1D-CNN architecture with residual connections and a self-attention (SE-block style) mechanism for classifying financial transactions by risk level. We adapt this method to a new domain — enterprise e-commerce fraud detection — using the Kaggle "Enterprise E-Commerce Intelligence" dataset, which contains 150K transactions with a binary fraud label (~4% fraud rate). Features from four relational tables (transactions, customers, products, behavior) are integrated to create a rich, multi-source feature set.

**Core Research Question**: Can the 1D-CNN with self-attention method from Mohammed et al. (2026) improve fraud detection performance on e-commerce data compared to traditional ML baselines?

**Tasks**:
1. Replicate the 1D-CNN architecture in Python/PyTorch (with **true residual/skip connections**, which the original MATLAB code omitted)
2. Integrate features from all 4 Kaggle tables (transactions, customers, products, behavior) into a unified fraud detection pipeline
3. Compare against baselines (Logistic Regression, Random Forest, XGBoost, simple MLP) to evaluate whether 1D-CNN offers improvement
4. Write a complete research paper answering the research question

**Original paper**: Mohammed et al. (2026) — "A deep residual 1D-CNN with self-attention for fraud transaction detection in virtual economies"
**Original code (MATLAB)**: https://github.com/Kamel123654/Real-Time-Risk-Classification-of-Metaverse-Financial-Transactions-Using-Enhanced-1D-CNN
**New dataset**: https://www.kaggle.com/datasets/jayjoshi37/enterprise-e-commerce-intelligence (4 CSVs: transactions.csv, customers.csv, products.csv, behavior.csv)

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python + PyTorch | Practical for research, easy sklearn/XGBoost integration |
| Task | Binary classification (fraud vs legitimate) | Dataset has binary fraud_label |
| Residual connections | True skip connections | Paper claims them but original code omits them |
| Class imbalance | SMOTE oversampling on training set only | Matches paper's oversampling approach; ~4% fraud rate needs balancing |
| Baselines | LR, Random Forest, XGBoost, MLP | Full baselines to rigorously answer the research question |
| Feature integration | Join all 4 tables on customer_id/product_id | Star schema: transactions as fact table |

---

## Stop-and-Check Protocol

Each phase ends with a mandatory 🛑 checkpoint. At every checkpoint:
1. Summarize what was completed
2. Present deliverables for review
3. List issues or concerns
4. **Critical Corrections**: discoveries that change the approach
5. **Wait for human approval** before proceeding

---

## PHASE 0: Project Setup

### Task 0.1: Create directory structure
```
fraud-detection-1dcnn/
├── INSTRUCTIONS.md
├── CLAUDE.md
├── original/
│   ├── paper/           # Source PDF (already copied)
│   ├── code/            # Original MATLAB code (reference)
│   └── data/raw/
└── work/
    ├── requirements.txt   # (already created)
    ├── .env.example       # (already created)
    ├── code/
    │   ├── 00_setup.py          # (already created)
    │   ├── 01_data_loading.py
    │   ├── 02_feature_engineering.py
    │   ├── 03_model_1dcnn.py
    │   ├── 04_baselines.py
    │   ├── 05_train_evaluate.py
    │   ├── 06_visualization.py
    │   └── utils.py              # (already created)
    ├── data/{raw,processed,extension}/
    ├── notes/
    ├── models/
    ├── output/{tables,figures,paper}/
    └── logs/
```

### Task 0.2: Download data
Download all 4 CSVs from https://www.kaggle.com/datasets/jayjoshi37/enterprise-e-commerce-intelligence and place them in `work/data/raw/`:
- `transactions.csv` (150K rows, 10 cols)
- `customers.csv` (25K rows, 7 cols)
- `products.csv` (2K rows, 5 cols)
- `behavior.csv` (25K rows, 8 cols)

### Task 0.3: Environment setup
```bash
pip install -r work/requirements.txt
```
Verify: `python work/code/00_setup.py` should print project root, device, and seed info without errors.

## 🛑 CHECKPOINT 0
Confirm: structure created, 4 CSVs downloaded to `work/data/raw/`, `00_setup.py` runs successfully.
Deliver: data file list, `requirements.txt` installed.
Critical Corrections: ____

---

## PHASE 1: Data Loading & Exploration

### Task 1.1: Implement `01_data_loading.py`

**Step 1 — Load all 4 CSVs** using `utils.load_dataset()`:
```python
transactions = load_dataset("transactions", DATA_RAW)
customers = load_dataset("customers", DATA_RAW)
products = load_dataset("products", DATA_RAW)
behavior = load_dataset("behavior", DATA_RAW)
```

**Step 2 — Record raw data profiles** for each table:
| Table | Rows | Cols | Key columns | Target |
|-------|------|------|-------------|--------|
| transactions | ~150K | 10 | transaction_id, customer_id, product_id, order_date, order_value, payment_method, device_type, discount_applied, shipping_delay_days | **fraud_label** |
| customers | ~25K | 7 | customer_id (PK), age, gender, country, registration_date, loyalty_score, lifetime_value, churn_label | — |
| products | ~2K | 5 | product_id (PK), category, price, margin_percentage, popularity_score | — |
| behavior | ~25K | 8 | customer_id (FK), avg_session_time, pages_per_session, cart_abandon_rate, return_rate, support_tickets, review_score, behavior_churn_signal | — |

**Step 3 — Join tables** into a single flat DataFrame:
```python
df = transactions.merge(customers, on="customer_id", how="left")
df = df.merge(products, on="product_id", how="left")
df = df.merge(behavior, on="customer_id", how="left")
```
Expected result: ~150K rows × ~27 feature columns + fraud_label

**Step 4 — Basic data audit**:
- Missing values per column
- Duplicate rows
- Data types (numeric, categorical, datetime)
- Target distribution: count and percentage of fraud_label = 1
- Save data profile report to `notes/data_profile.md`

### Task 1.2: Save joined dataset
Save the joined (but not yet processed) DataFrame to `work/data/processed/joined_raw.parquet` for reproducibility.

## 🛑 CHECKPOINT 1
Confirm: all 4 CSVs loaded, joined successfully, no unexpected data loss from joins, fraud rate ~4%.
Deliver: `notes/data_profile.md`, `data/processed/joined_raw.parquet`.
Critical Corrections: ____

---

## PHASE 2: Feature Engineering

### Task 2.1: Implement `02_feature_engineering.py`

**Step 1 — Drop non-feature columns**:
- `transaction_id`, `customer_id`, `product_id` (identifiers, not predictive)
- `churn_label` (from customers — this is a different target, not usable for fraud prediction)
- `behavior_churn_signal` (from behavior — also a different target, **keep only if you decide it's a legitimate behavioral feature; by default, drop to avoid label leakage concerns**)

**Step 2 — Temporal feature extraction** from `order_date`:
```python
df["order_date"] = pd.to_datetime(df["order_date"])
df["order_hour"] = df["order_date"].dt.hour        # 0-23
df["order_dayofweek"] = df["order_date"].dt.dayofweek  # 0=Mon, 6=Sun
df["order_month"] = df["order_date"].dt.month       # 1-12
df["order_is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)
```
Also extract features from `registration_date`:
```python
df["registration_date"] = pd.to_datetime(df["registration_date"])
df["customer_tenure_days"] = (df["order_date"] - df["registration_date"]).dt.days
```
Drop original date columns after extraction.

**Step 3 — Identify column types**:

Numerical columns (to be scaled):
- `order_value`, `discount_applied`, `shipping_delay_days`
- `age`, `loyalty_score`, `lifetime_value`
- `price`, `margin_percentage`, `popularity_score`
- `avg_session_time`, `pages_per_session`, `cart_abandon_rate`, `return_rate`, `support_tickets`, `review_score`
- `order_hour`, `order_dayofweek`, `order_month`, `customer_tenure_days`

Categorical columns (to be one-hot encoded):
- `payment_method`, `device_type`
- `gender`, `country`
- `category` (product category)

Binary columns (keep as-is):
- `order_is_weekend`

**Step 4 — One-hot encode categoricals**:
```python
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
```

**Step 5 — Train/Val/Test split** (BEFORE scaling and SMOTE, to prevent data leakage):
```python
from sklearn.model_selection import train_test_split
X = df.drop(columns=["fraud_label"])
y = df["fraud_label"]

# First split: 85% train+val, 15% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y
)
# Second split: ~70% train, ~15% val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=SEED, stratify=y_temp
)
```

**Step 6 — Standardize numerical features** (fit on train only):
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])
```

**Step 7 — SMOTE oversampling** (training set only):
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=SEED)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
```

**Step 8 — Save processed data**:
- Save `X_train_bal`, `y_train_bal`, `X_val`, `y_val`, `X_test`, `y_test` to `data/processed/`
- Save the scaler and feature column list for reproducibility

### Task 2.2: Feature summary
Create a table documenting all features, their source table, type, and preprocessing applied. Save to `notes/feature_engineering.md`.

## 🛑 CHECKPOINT 2
Confirm: features engineered correctly, no data leakage (split → scale → SMOTE order), feature counts documented.
Deliver: processed data files, scaler object, `notes/feature_engineering.md`.
Critical Corrections: ____

---

## PHASE 3: Model Architecture — 1D-CNN with Self-Attention

### Task 3.1: Implement `03_model_1dcnn.py`

Build the PyTorch model with the following architecture:

```
Input: (batch_size, num_features)
  → reshape to (batch_size, 1, num_features)  # 1 channel, num_features length

Conv Block 1:
  Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
  BatchNorm1d(64)
  ReLU
  Conv1d(64, 64, kernel_size=3, padding=1)
  BatchNorm1d(64)
  ReLU
  Output: (batch_size, 64, num_features)

Residual Block (with TRUE skip connection):
  identity = x
  Conv1d(64, 64, kernel_size=3, stride=2, padding=1)
  BatchNorm1d(64)
  ReLU
  Conv1d(64, 64, kernel_size=3, padding=1)
  BatchNorm1d(64)
  identity shortcut = Conv1d(64, 64, kernel_size=1, stride=2)(identity)  # match dims
  x = ReLU(x + identity)  # ← TRUE skip connection
  Output: (batch_size, 64, num_features // 2)

Self-Attention (SE-block style):
  GlobalAvgPool1d → (batch_size, 64)
  FC(64 → num_features // 2) → ReLU
  FC(num_features // 2 → 64) → Sigmoid
  Multiply: x * attention_weights (channel-wise)
  Output: (batch_size, 64, num_features // 2) — reweighted

Classifier:
  GlobalAvgPool1d → (batch_size, 64)
  FC(64 → 2)
  Output: logits for binary classification
```

**Implementation notes**:
- The input is reshaped from a flat feature vector to a 1D "sequence" of length `num_features` with 1 channel. This is the same approach as the original paper — treating each feature dimension as a position in a 1D signal.
- The residual block uses a **projection shortcut** (1×1 Conv1d with stride=2) to match dimensions before adding. This is the standard ResNet approach and fixes the missing skip connections in the original code.
- The self-attention is SE-block style: squeeze (global avg pool) → excitation (FC → ReLU → FC → sigmoid) → scale. This matches the paper's description.
- Use `nn.CrossEntropyLoss()` (binary classification with 2 output classes).

### Task 3.2: Data preparation for 1D-CNN
Create a PyTorch `Dataset` and `DataLoader`:
```python
class FraudDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        # X: numpy array (num_samples, num_features)
        # Reshape to (num_samples, 1, num_features) for Conv1d
        self.X = torch.FloatTensor(X).unsqueeze(1)  # add channel dim
        self.y = torch.LongTensor(y.values if hasattr(y, 'values') else y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

### Task 3.3: Verify model forward pass
- Create a dummy batch and verify shapes at each layer
- Print model summary (total params, trainable params)
- Save architecture diagram to `output/figures/model_architecture.png`

## 🛑 CHECKPOINT 3
Confirm: model builds without errors, forward pass produces correct output shape (batch_size, 2), parameter count is reasonable (~50K-200K params expected).
Deliver: `03_model_1dcnn.py`, architecture diagram, parameter count.
Critical Corrections: ____

---

## PHASE 4: Training & Evaluation

### Task 4.1: Implement `05_train_evaluate.py`

**Step 1 — Training loop for 1D-CNN**:
```python
# Configuration (from 00_setup.py)
model = FraudDetectionCNN1D(num_features=X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=LR_DROP_FACTOR, patience=LR_DROP_PATIENCE
)
criterion = nn.CrossEntropyLoss()
```

Training loop:
- For each epoch: train on batches → validate → step LR scheduler → log metrics
- Track: train_loss, val_loss, train_acc, val_acc, val_f1 per epoch
- Save best model (lowest val_loss) to `models/1dcnn_best.pt`
- Early stopping: patience=15 epochs (stop if val_loss doesn't improve)
- Log training curves to `logs/training_log.json`

**Step 2 — Implement baseline models** in `04_baselines.py`:

All baselines use the **same** SMOTE-balanced training data and the **same** test set.

| Model | Implementation | Key Params |
|-------|---------------|------------|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | max_iter=1000, class_weight='balanced' |
| Random Forest | `sklearn.ensemble.RandomForestClassifier` | n_estimators=200, random_state=42 |
| XGBoost | `xgboost.XGBClassifier` | n_estimators=200, max_depth=6, learning_rate=0.1 |
| MLP | PyTorch 2-layer: FC(num_features→128→64→2) | Same optimizer/lr as 1D-CNN |

For each baseline, train on `X_train_bal` / `y_train_bal` and predict on `X_test`.

**Step 3 — Evaluate all models** on the **original (unbalanced) test set**:

Compute for each model:
```python
metrics = compute_metrics(y_test, y_pred, y_prob)
# Returns: accuracy, precision, recall, f1_score, auc_roc, auc_pr
```

**Step 4 — Compile comparison table**:
| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|-------|----------|-----------|--------|----|---------|--------|
| Logistic Regression | — | — | — | — | — | — |
| Random Forest | — | — | — | — | — | — |
| XGBoost | — | — | — | — | — | — |
| MLP | — | — | — | — | — | — |
| **1D-CNN (Ours)** | — | — | — | — | — | — |

Save to `output/tables/model_comparison.csv`

## 🛑 CHECKPOINT 4
Confirm: all 5 models trained, metrics computed on held-out test set, comparison table generated.
Deliver: trained model files, `output/tables/model_comparison.csv`, training logs.
Critical Corrections: ____

---

## PHASE 5: Visualization & Analysis

### Task 5.1: Implement `06_visualization.py`

Generate these figures:
1. **Training curves** (1D-CNN): loss and accuracy over epochs — `output/figures/training_curves.png`
2. **Confusion matrices**: one per model (5 subplots) — `output/figures/confusion_matrices.png`
3. **ROC curves**: all 5 models on one plot — `output/figures/roc_curves.png`
4. **Precision-Recall curves**: all 5 models on one plot — `output/figures/pr_curves.png`
5. **Feature importance** (for tree-based baselines): XGBoost top-20 features — `output/figures/feature_importance_xgboost.png`
6. **Class distribution**: before/after SMOTE — `output/figures/class_distribution.png`

### Task 5.2: Analysis notes
Write `notes/results_analysis.md` answering:
- Does 1D-CNN outperform baselines on key metrics (F1, AUC-ROC, AUC-PR)?
- On which metrics does it excel or underperform?
- Is the improvement (if any) practically significant given the ~4% fraud rate?
- What does the feature importance from XGBoost tell us about which features (from which tables) matter most?

## 🛑 CHECKPOINT 5
Confirm: all figures generated, results analyzed, comparison table complete.
Deliver: all figures in `output/figures/`, `notes/results_analysis.md`.
Critical Corrections: ____

---

## PHASE 6: Paper Writing

### Task 6.1: Write the research paper

**Structure and approximate word counts**:

1. **Abstract** (~200 words)
   - Research question, method summary, key result, contribution

2. **Introduction** (~1500 words)
   - Problem: e-commerce fraud detection, cost of undetected fraud
   - Gap: can deep learning (1D-CNN) improve on traditional methods?
   - Paper contribution: adapt Mohammed et al.'s 1D-CNN to multi-source e-commerce data
   - Paper structure overview

3. **Literature Review** (~1500 words)
   - Traditional fraud detection methods (rule-based, logistic regression)
   - ML approaches (Random Forest, XGBoost, ensemble methods)
   - Deep learning for fraud detection (CNNs, RNNs, transformers)
   - Mohammed et al. (2026) — the paper we extend
   - Multi-source feature integration in fraud detection

4. **Data** (~1000 words)
   - Data source: Kaggle Enterprise E-Commerce Intelligence
   - Four tables and their schemas
   - Join strategy and feature integration
   - Target variable: fraud_label (~4% fraud rate)
   - Data preprocessing pipeline: temporal features, encoding, scaling, SMOTE
   - Summary statistics table

5. **Methodology** (~2000 words)
   - 1D-CNN architecture with detailed layer descriptions
   - Residual connections (explain true skip connections vs original code)
   - Self-attention mechanism (SE-block)
   - Training procedure: optimizer, loss, LR schedule, early stopping
   - Baseline models description
   - Evaluation metrics and why they matter for imbalanced data

6. **Results** (~2000 words)
   - Model comparison table (all metrics)
   - ROC and PR curve analysis
   - Confusion matrix analysis
   - Feature importance discussion
   - Training convergence analysis

7. **Discussion** (~1000 words)
   - Answer to research question: does 1D-CNN improve fraud detection?
   - Why it works or doesn't (feature interactions, sequential pattern extraction)
   - Limitations: synthetic data, single domain, no temporal sequence modeling
   - Practical implications for e-commerce fraud detection

8. **Conclusion** (~300 words)
   - Summary of findings
   - Future work directions

### Task 6.2: Generate final tables and figures for paper
- Format comparison table as LaTeX or Word-ready
- Label all figures with captions

## 🛑 CHECKPOINT 6
Deliver: complete paper draft in `output/paper/`, all tables and figures.
Critical Corrections: ____

---

## PHASE 7: Final Deliverables

### Task 7.1: Organize and clean up
- Ensure all numbered scripts (00–06) run end-to-end: `python 01_data_loading.py && python 02_feature_engineering.py && ...`
- Verify all output files are present
- Create `README.md` with project summary, how to run, and results highlight

## 🛑 FINAL CHECKPOINT
Confirm: paper, code, data, documentation all present and reproducible.

---

## Appendix A: Original Paper — Method Summary

### Architecture (Mohammed et al. 2026)
The paper proposes an "Enhanced 1D-CNN" with three key components:

1. **Initial Convolutional Block**: Two stacked Conv1D layers (64 filters, kernel size 3) with BatchNorm and ReLU activation
2. **Residual Block**: Two Conv1D layers with stride-2 downsampling. The paper describes skip connections but the released MATLAB code does not implement them. We implement true residual connections with projection shortcuts.
3. **Self-Attention**: SE-block (Squeeze-and-Excitation) style attention — Global Average Pooling → FC → Softmax → channel reweighting
4. **Output**: FC layer → Softmax → Classification

### Key Adaptations for Our Project
| Aspect | Original Paper | Our Project |
|--------|---------------|-------------|
| Domain | Metaverse transactions | E-commerce transactions |
| Classification | 3-class (high/low/moderate risk) | Binary (fraud/legitimate) |
| Data source | Single CSV (metaverse_transactions_dataset.csv) | 4 joined CSVs (transactions, customers, products, behavior) |
| Language | MATLAB | Python + PyTorch |
| Residual connections | Described but not implemented | True skip connections with projection shortcuts |
| Balancing | Oversampling (replication) | SMOTE |
| Evaluation | Basic (accuracy implied) | Comprehensive (P, R, F1, AUC-ROC, AUC-PR, confusion matrices) |

## Appendix B: Dataset Feature Map

### Features by Source Table

**From transactions.csv (core)**:
| Feature | Type | Preprocessing |
|---------|------|---------------|
| order_value | Numerical | StandardScaler |
| discount_applied | Numerical | StandardScaler |
| shipping_delay_days | Numerical | StandardScaler |
| payment_method | Categorical | One-hot encode |
| device_type | Categorical | One-hot encode |
| order_hour | Numerical (derived) | StandardScaler |
| order_dayofweek | Numerical (derived) | StandardScaler |
| order_month | Numerical (derived) | StandardScaler |
| order_is_weekend | Binary (derived) | Keep as-is |

**From customers.csv**:
| Feature | Type | Preprocessing |
|---------|------|---------------|
| age | Numerical | StandardScaler |
| gender | Categorical | One-hot encode |
| country | Categorical | One-hot encode |
| loyalty_score | Numerical | StandardScaler |
| lifetime_value | Numerical | StandardScaler |
| customer_tenure_days | Numerical (derived) | StandardScaler |

**From products.csv**:
| Feature | Type | Preprocessing |
|---------|------|---------------|
| category | Categorical | One-hot encode |
| price | Numerical | StandardScaler |
| margin_percentage | Numerical | StandardScaler |
| popularity_score | Numerical | StandardScaler |

**From behavior.csv**:
| Feature | Type | Preprocessing |
|---------|------|---------------|
| avg_session_time | Numerical | StandardScaler |
| pages_per_session | Numerical | StandardScaler |
| cart_abandon_rate | Numerical | StandardScaler |
| return_rate | Numerical | StandardScaler |
| support_tickets | Numerical | StandardScaler |
| review_score | Numerical | StandardScaler |

**Dropped columns**:
- `transaction_id`, `customer_id`, `product_id` (identifiers)
- `churn_label` (from customers — different target)
- `behavior_churn_signal` (from behavior — potential label leakage)
- `order_date`, `registration_date` (replaced by derived features)

## Appendix C: Hyperparameter Summary

| Parameter | Value | Source |
|-----------|-------|--------|
| Conv filters | 64 | Paper original |
| Kernel size | 3 | Paper original |
| Residual stride | 2 | Paper original |
| Batch size | 16 | Paper original |
| Max epochs | 100 | Paper original |
| Learning rate (initial) | 0.001 | Paper original |
| Weight decay (L2) | 0.0001 | Paper original |
| LR drop factor | 0.5 | Paper original |
| LR drop patience | 5 epochs | Adapted from paper's piecewise decay |
| Early stopping patience | 15 epochs | Our addition |
| Optimizer | Adam | Paper original |
| SMOTE strategy | auto (equalize) | Our adaptation |
| Random seed | 42 | Standard |
| Train/Val/Test | 70/15/15 | Paper original |

---

## Quality Standards

- **Statistical**: All metrics with 95% CI where applicable; seeds reported; SMOTE applied to training set only (never val/test)
- **Writing**: Active prose; define variables on first use; consistent notation
- **Reproducibility**: All results from code; no manual steps; paths relative to PROJECT_ROOT
- **Citations**: Every paper verified via web search; no hallucinated references; APA format with DOI
