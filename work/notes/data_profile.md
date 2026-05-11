# Data Profile Report

Generated from 4 CSVs joined on customer_id / product_id.

## Joined Dataset: 150000 rows x 28 columns

## Source Tables

| Table | Rows | Cols | Key Columns |
|-------|------|------|-------------|
| transactions | 150000 | 10 | transaction_id, customer_id, product_id, fraud_label |
| customers | 25000 | 8 | customer_id (PK) |
| products | 2000 | 5 | product_id (PK) |
| behavior | 25000 | 8 | customer_id (FK) |

## Missing Values

No missing values detected.


## Duplicate Rows: 0


## Target Distribution

- Legitimate (0): 143669 (95.78%)
- Fraud (1): 6331 (4.22%)
- Fraud rate: 4.22%

## Column Types

### Numeric Columns

- `order_value`: min=2.56, max=999.72, mean=383.33
- `discount_applied`: min=0.0, max=0.5, mean=0.25
- `shipping_delay_days`: min=0, max=10, mean=5.01
- `fraud_label`: min=0, max=1, mean=0.04
- `age`: min=18, max=70, mean=44.00
- `loyalty_score`: min=0.0, max=100.0, mean=49.97
- `lifetime_value`: min=100.33, max=19998.03, mean=10023.36
- `churn_label`: min=0, max=1, mean=0.22
- `price`: min=5.13, max=999.72, mean=511.33
- `margin_percentage`: min=5.0, max=39.98, mean=22.27
- `popularity_score`: min=0.0, max=99.77, mean=49.08
- `avg_session_time`: min=1.0, max=20.0, mean=10.45
- `pages_per_session`: min=1, max=25, mean=13.08
- `cart_abandon_rate`: min=0.0, max=1.0, mean=0.50
- `return_rate`: min=0.0, max=0.5, mean=0.25
- `support_tickets`: min=0, max=10, mean=5.01
- `review_score`: min=1.0, max=5.0, mean=3.00
- `behavior_churn_signal`: min=0, max=1, mean=0.34

### Categorical Columns

- `transaction_id`: 150000 unique values
- `customer_id`: 24938 unique values
- `product_id`: 2000 unique values
- `order_date`: 2001 unique values
- `payment_method`: 5 unique values
- `device_type`: 3 unique values
- `gender`: 3 unique values
- `country`: 8 unique values
- `registration_date`: 2001 unique values
- `category`: 8 unique values