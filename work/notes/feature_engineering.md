# Feature Engineering Report

## Feature Summary

- Total features: 47
- Features to scale: 16
- One-hot encoded categories: ['payment_method', 'device_type', 'gender', 'country', 'category']
- Binary features: ['order_is_weekend']
- Target: fraud_label

## Features by Source Table

### From transactions.csv
| Feature | Type | Preprocessing |
|---------|------|---------------|
| order_value | Numerical / StandardScaler |
| discount_applied | Numerical / StandardScaler |
| shipping_delay_days | Numerical / StandardScaler |
| payment_method | Categorical / One-hot (5 values) |
| device_type | Categorical / One-hot (3 values) |
| order_hour | Numerical (derived) / StandardScaler |
| order_dayofweek | Numerical (derived) / StandardScaler |
| order_month | Numerical (derived) / StandardScaler |
| order_is_weekend | Binary (derived) / Keep as-is |

### From customers.csv
| Feature | Type | Preprocessing |
|---------|------|---------------|
| age | Numerical / StandardScaler |
| gender | Categorical / One-hot (3 values) |
| country | Categorical / One-hot (8 values) |
| loyalty_score | Numerical / StandardScaler |
| lifetime_value | Numerical / StandardScaler |
| customer_tenure_days | Numerical (derived) / StandardScaler |

### From products.csv
| Feature | Type | Preprocessing |
|---------|------|---------------|
| category | Categorical / One-hot (8 values) |
| price | Numerical / StandardScaler |
| margin_percentage | Numerical / StandardScaler |
| popularity_score | Numerical / StandardScaler |

### From behavior.csv
| Feature | Type | Preprocessing |
|---------|------|---------------|
| avg_session_time | Numerical / StandardScaler |
| pages_per_session | Numerical / StandardScaler |
| cart_abandon_rate | Numerical / StandardScaler |
| return_rate | Numerical / StandardScaler |
| support_tickets | Numerical / StandardScaler |
| review_score | Numerical / StandardScaler |

### Dropped columns
- `transaction_id`, `customer_id`, `product_id` — identifiers
- `churn_label` — different target variable
- `behavior_churn_signal` — potential label leakage
- `order_date`, `registration_date` — replaced by derived temporal features

## Data Splits (stratified)
| Split | Samples | Fraud | Legit | Fraud Rate |
|-------|---------|-------|-------|------------|
| Train (pre-SMOTE) | 104,999 | 4,431 | 100,568 | 4.22% |
| Train (post-SMOTE) | 201,136 | 100,568 | 100,568 | 50.00% |
| Validation | 22,500 | 950 | 21,550 | 4.22% |
| Test | 22,501 | 950 | 21,551 | 4.22% |