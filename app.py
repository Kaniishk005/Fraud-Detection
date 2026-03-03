from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

model = joblib.load("fraud_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")


def preprocess_input(data_dict):
    df_input = pd.DataFrame([data_dict])

    df_input['amount_log'] = np.log1p(df_input['amount'])
    df_input['org_balance_diff'] = df_input['oldbalanceOrg'] - df_input['newbalanceOrig']
    df_input['dest_balance_diff'] = df_input['newbalanceDest'] - df_input['oldbalanceDest']

    df_input = pd.get_dummies(df_input, columns=['type'], drop_first=True)

    for col in feature_columns:
        if col not in df_input:
            df_input[col] = 0

    df_input = df_input[feature_columns]

    return df_input


@app.post("/predict")
def predict(transaction: dict):

    processed_data = preprocess_input(transaction)

    probability = model.predict_proba(processed_data)[:,1]
    prediction = (probability > 0.3).astype(int)

    return {
        "fraud_probability": float(probability[0]),
        "prediction": int(prediction[0])
    }
@app.get("/model-info")
def model_info():
    return {
        "model": "RandomForestClassifier",
        "roc_auc": 0.989,
        "threshold": 0.3,
        "features_used": len(feature_columns)
    }
@app.get("/")
def home():
    return {"message": "Fraud Detection API Running"}