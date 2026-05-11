# Fraud Detection with 1D-CNN: A Replication and Extension Study

> **Course**: DOTE6635 AI for Business — Final Project
> **Team**: NexusAI

## Research Question

Can a deep residual 1D-CNN with self-attention, originally designed for metaverse transaction risk classification (Mohammed et al., 2026), improve fraud detection performance on enterprise e-commerce data?

## Key Results

| Model | F1 | Recall | AUC-ROC | AUC-PR |
|-------|----|--------|---------|--------|
| **1D-CNN (Ours)** | 0.1215 | **0.4116** | 0.6347 | **0.0704** |
| Logistic Regression | **0.1260** | 0.3789 | 0.6301 | 0.0696 |
| Random Forest | 0.1217 | 0.2916 | 0.6210 | 0.0644 |
| XGBoost | 0.1206 | 0.2916 | 0.6347 | 0.0692 |
| MLP | 0.1193 | 0.3758 | 0.6375 | 0.0663 |

- **1D-CNN achieves the highest recall (41.2%)** — catches the most fraud cases
- **1D-CNN achieves the highest AUC-PR (0.0704)** — best precision-recall tradeoff

## Project Structure

```
fraud-detection-1dcnn/
├── README.md
├── CLAUDE.md                    # Project context
├── INSTRUCTIONS.md              # Master workflow (phase-by-phase)
├── original/
│   └── paper/                   # Source PDF
└── work/                        # All code, data, and output
    ├── requirements.txt
    ├── .env.example
    ├── code/
    │   ├── 00_setup.py           # Paths, seeds, constants
    │   ├── 01_data_loading.py    # Load & join 4 Kaggle CSVs
    │   ├── 02_feature_engineering.py  # Clean, encode, scale, SMOTE
    │   ├── 03_model_1dcnn.py     # PyTorch 1D-CNN architecture
    │   ├── 04_baselines.py       # LR, RF, XGBoost, MLP
    │   ├── 05_train_evaluate.py  # Train 1D-CNN + final comparison
    │   ├── 06_visualization.py   # All figures
    │   └── utils.py              # Shared helpers
    ├── data/
    │   ├── raw/                  # 4 CSVs from Kaggle
    │   └── processed/            # Joined, split, scaled, SMOTE-balanced
    ├── models/                   # Trained model files
    ├── output/
    │   ├── tables/               # model_comparison.csv, results JSON
    │   ├── figures/              # 6 figures (ROC, PR, confusion matrices, etc.)
    │   └── paper/                # technical_report.md (8,100 words)
    ├── notes/                    # Data profile, feature engineering, analysis
    └── logs/                     # Training log
```

## How to Run

```bash
# 1. Install dependencies
pip install -r work/requirements.txt

# 2. Download data from Kaggle
#    https://www.kaggle.com/datasets/jayjoshi37/enterprise-e-commerce-intelligence
#    Place 4 CSVs in work/data/raw/

# 3. Run pipeline in order
cd work
python code/00_setup.py          # Verify setup
python code/01_data_loading.py   # Load & join tables
python code/02_feature_engineering.py  # Preprocess & split
python code/03_model_1dcnn.py    # Verify model architecture
python code/04_baselines.py      # Train baselines (LR, RF, XGBoost, MLP)
python code/05_train_evaluate.py  # Train 1D-CNN + compare all models
python code/06_visualization.py   # Generate all figures
```

## Methodology

- **Architecture**: 1D-CNN with true residual connections (projection shortcuts) + SE-style self-attention, 44K parameters
- **Data**: Enterprise E-Commerce Intelligence (Kaggle) — 150K transactions, 4 tables joined (transactions, customers, products, behavior), 46 features
- **Preprocessing**: SMOTE oversampling on training set only, StandardScaler fit on train only, threshold optimization on validation set
- **Baselines**: Logistic Regression, Random Forest, XGBoost, 2-layer MLP

## Key Findings

1. The 1D-CNN **does not clearly outperform** simpler baselines on F1 score
2. The 1D-CNN achieves the **highest recall** (41.2%) — most effective at catching actual fraud
3. All models show modest absolute performance (AUC-ROC ~0.63) due to weak discriminative signal in the synthetic dataset
4. Threshold optimization is **essential** — at default threshold 0.5, all models predict zero fraud

## References

- Mohammed, K. et al. (2026). A deep residual 1D-CNN with self-attention for fraud transaction detection in virtual economies.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition (ResNet).
- Hu, J. et al. (2018). Squeeze-and-Excitation Networks.
- Chawla, N. V. et al. (2002). SMOTE: Synthetic Minority Over-sampling Technique.
