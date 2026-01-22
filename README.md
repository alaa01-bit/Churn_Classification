# Customer Churn Dataset
## File: customer_churn_dataset.csv

## Description
This dataset contains information about 1000 telecom customers and whether they churned (left the company).

## Columns
1. customer_id - Unique identifier
2. age - Customer age in years
3. tenure_months - Number of months with the company
4. monthly_charges - Monthly service charges ($)
5. total_charges - Total amount charged ($)
6. contract_type - Contract type (Monthly/Yearly/Two-year)
7. internet_service - Type of internet service (DSL/Fiber/None)
8. paperless_billing - Paperless billing option (Yes/No)
9. payment_method - Payment method used
10. dependents - Number of dependents
11. phone_service - Has phone service (Yes/No)
12. tech_support - Technical support subscription
13. monthly_data_gb - Monthly data usage in GB (some missing)
14. customer_service_calls - Number of customer service calls
15. late_payments_last_year - Number of late payments
16. churn - TARGET: 1=customer left, 0=customer stayed

## Task
Build a classification model to predict customer churn.

## Challenges
- Missing values in monthly_data_gb and tech_support
- Imbalanced target variable (~30% churn)
- Mix of numerical and categorical features
- Need for proper preprocessing and feature engineering
