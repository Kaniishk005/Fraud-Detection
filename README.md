Fraud Transaction Detection API
1. Overview

A production-ready Fraud Detection system built using Random Forest and deployed via FastAPI.
The model detects fraudulent financial transactions with optimized threshold tuning and imbalance handling.

2. Model Performance

-ROC-AUC: 0.989
-Fraud Recall (threshold 0.3): 0.81
-Class imbalance handled using class_weight='balanced'

3. Feature Engineering

-Log transformation of transaction amount
-Balance difference features:
-org_balance_diff
-dest_balance_diff
-One-hot encoding of transaction type
-Correlation-based feature reduction

4. API Endpoints
Endpoint	Method	Description
/predict	POST	Returns fraud probability & classification
/model-info	GET	Returns model metadata
/	GET	Health check
5. Tech Stack

-Python
-Scikit-learn
-FastAPI
-Uvicorn
-Pandas / NumPy


