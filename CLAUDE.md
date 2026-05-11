# Fraud Detection with 1D-CNN — Project Context

## Project Overview
Replicate and extend Mohammed et al. (2026) "A deep residual 1D-CNN with self-attention for fraud transaction detection in virtual economies". Apply their 1D-CNN architecture to a new e-commerce dataset to answer: **Can this 1D-CNN method improve fraud detection performance?**

## Research Question
Can a deep residual 1D-CNN with self-attention, originally designed for metaverse financial transactions, effectively detect fraud in enterprise e-commerce data when compared to traditional ML baselines?

## Repository Structure
```
fraud-detection-1dcnn/
├── CLAUDE.md                    # This file — project context
├── INSTRUCTIONS.md              # Master workflow (primary deliverable)
├── original/
│   ├── paper/                   # Source PDF
│   ├── code/                    # Original MATLAB code reference
│   └── data/raw/                # Original dataset reference
└── work/                        # ALL code and output
    ├── code/
    │   ├── 00_setup.py          # Paths, seeds, constants
    │   ├── 01_data_loading.py   # Load & join 4 Kaggle CSVs
    │   ├── 02_feature_engineering.py  # Clean, encode, normalize, SMOTE
    │   ├── 03_model_1dcnn.py    # PyTorch 1D-CNN architecture
    │   ├── 04_baselines.py      # LR, RF, XGBoost, MLP baselines
    │   ├── 05_train_evaluate.py # Training loop + evaluation
    │   ├── 06_visualization.py  # Plots, confusion matrices, ROC curves
    │   └── utils.py             # Shared helpers
    ├── data/{raw,processed,extension}/
    ├── notes/
    ├── models/
    ├── output/{tables,figures,paper}/
    └── logs/
```

## Key Technical Context

### Model Architecture (1D-CNN with Self-Attention)
- **Input**: Flattened feature vector per transaction (num_features channels, length 1)
  - Alternatively: reshape to (num_features // k, k) for sequence-like input
- **Conv Block 1**: 2× [Conv1d(in→64, kernel=3, padding=1) → BatchNorm → ReLU]
- **Residual Block**: 2× [Conv1d(64→64, kernel=3, stride=2, padding=1) → BN → ReLU] + **true skip connection** (Conv1d shortcut with stride=2)
- **Self-Attention (SE-style)**: GlobalAvgPool1d → FC(num_features) → Softmax → channel reweighting
- **Classifier**: Flatten → FC(64→2) → Softmax (binary: fraud/legitimate)

### Hyperparameters (from original paper)
- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- LR Schedule: ReduceLROnPlateau (factor=0.5, patience=5) — adapted from original piecewise decay
- Batch size: 16
- Max epochs: 100
- Dropout: not used in original (add optionally)
- Data split: 70/15/15 train/val/test with stratification
- Class balancing: SMOTE on training set only

### Data: Enterprise E-Commerce Intelligence (Kaggle)
- **transactions.csv** (150K rows): transaction_id, customer_id, product_id, order_date, order_value, payment_method, device_type, discount_applied, shipping_delay_days, **fraud_label**
- **customers.csv** (25K rows): customer_id, age, gender, country, registration_date, loyalty_score, lifetime_value, churn_label
- **products.csv** (2K rows): product_id, category, price, margin_percentage, popularity_score
- **behavior.csv** (25K rows): customer_id, avg_session_time, pages_per_session, cart_abandon_rate, return_rate, support_tickets, review_score, behavior_churn_signal
- **Target**: fraud_label (binary, ~4% fraud rate)
- **Join**: transactions ← customers (customer_id), transactions ← products (product_id), transactions ← behavior (customer_id)

### Baselines for Comparison
- Logistic Regression (sklearn)
- Random Forest (sklearn)
- XGBoost
- Simple 2-layer MLP (PyTorch)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR
- Confusion matrices, ROC curves, PR curves
- All metrics computed on the **held-out test set** (never SMOTE-balanced)

## Dependencies
- Python 3.9+
- PyTorch, scikit-learn, xgboost, imbalanced-learn (SMOTE)
- pandas, numpy, matplotlib, seaborn

## Coding Conventions
- All paths relative to PROJECT_ROOT (resolved in 00_setup.py)
- Seeds: global seed 42, set in numpy, torch, random
- Numbered scripts 00–06, run in order
- Figures saved to output/figures/, tables to output/tables/
- Use type hints, docstrings for public functions
- No hardcoded absolute paths
