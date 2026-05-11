# Results Analysis

## Model Ranking (by F1)

| Rank | Model | F1 | Precision | Recall | AUC-ROC | AUC-PR |
|------|-------|----|-----------|--------|---------|--------|
| 1 | Logistic Regression | 0.1260 | 0.0756 | 0.3789 | 0.6301 | 0.0696 |
| 2 | Random Forest | 0.1217 | 0.0769 | 0.2916 | 0.6210 | 0.0644 |
| 3 | 1D-CNN (Ours) | 0.1215 | 0.0712 | 0.4116 | 0.6347 | 0.0704 |
| 4 | XGBoost | 0.1206 | 0.0760 | 0.2916 | 0.6347 | 0.0692 |
| 5 | MLP | 0.1193 | 0.0709 | 0.3758 | 0.6375 | 0.0663 |

## Key Findings

1. Best F1: **Logistic Regression** (F1=0.1260)
2. 1D-CNN rank: #3 by F1
3. 1D-CNN has the **highest recall** (0.4116) among all models
4. 1D-CNN AUC-ROC: 0.6347 (competitive with best baseline 0.6301)

## Research Question Answer

**Can 1D-CNN improve fraud detection?**

- By F1: 1D-CNN (0.1215) is competitive but does not clearly outperform Logistic Regression (0.1260).
- By Recall: 1D-CNN (0.4116) achieves the highest recall, catching 41.2% of fraud cases vs 37.9% for LR.
- By AUC-PR: 1D-CNN (0.0704) achieves the highest area under PR curve.

## Why Results Are Modest

- All models achieve AUC-ROC ~0.62-0.64 (barely above 0.5 random baseline)
- This indicates **weak discriminative signal** in the synthetic dataset
- The ~4% fraud rate creates severe class imbalance challenges
- Optimal thresholds are very low (0.05-0.14), showing models struggle with calibration