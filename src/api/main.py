# Date: 2025-06-27
# Created by: Addisu Taye
# Purpose: Expose a REST API for real-time credit risk prediction using FastAPI.
# Key features:
# - Loads trained model and preprocessor
# - Accepts POST requests with customer transaction history
# - Preprocesses and transforms data into model-ready format
# - Returns risk probability and classification
# - Uses Pydantic models for input/output validation
# Task Number: 6

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pydantic import BaseModel
import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from datetime import datetime

app = FastAPI(title="Credit Risk API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    model = mlflow.sklearn.load_model("models:/random_forest/Production")
except Exception as e:
    raise RuntimeError(f"Failed to load model from MLflow: {e}")

# Load preprocessor
try:
    with open("models/preprocessor.pkl", 'rb') as f:
        preprocessor = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError("Preprocessor not found at preprocessor.pkl")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Credit Risk Prediction API"}


class TransactionRequest(BaseModel):
    Amount: float
    Value: float
    CountryCode: int
    ProviderId: str
    PricingStrategy: str
    ProductCategory: str
    ChannelId: str
    TransactionStartTime: str  # ISO format date string


class CustomerRiskRequest(BaseModel):
    transactions: List[TransactionRequest]


class RiskPredictionResponse(BaseModel):
    risk_probability: float
    is_high_risk: bool


@app.post("/predict", response_model=RiskPredictionResponse)
def predict_credit_risk(customer_data: CustomerRiskRequest):
    try:
        # Convert transaction data to DataFrame
        df = pd.DataFrame([t.dict() for t in customer_data.transactions])
        
        # Convert timestamp
        df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
        
        # Time-based features
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        df['transaction_year'] = df['TransactionStartTime'].dt.year
        df['is_weekend'] = df['TransactionStartTime'].dt.dayofweek >= 5
        
        # Aggregate features
        latest = df.iloc[df['TransactionStartTime'].argmax()]
        total_trans = len(df)
        avg_amount = df['Amount'].mean()
        std_amount = df['Amount'].std()
        total_value = df['Value'].sum()
        unique_categories = df['ProductCategory'].nunique()
        
        # Snapshot date for RFM
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        recency = (snapshot_date - df['TransactionStartTime'].max()).days
        frequency = len(df)
        monetary = df['Value'].sum()
        
        # Create single row for prediction
        input_data = pd.DataFrame({
            'Amount': [latest['Amount']],
            'Value': [latest['Value']],
            'CountryCode': [latest['CountryCode']],
            'ProviderId': [latest['ProviderId']],
            'PricingStrategy': [latest['PricingStrategy']],
            'ProductCategory': [latest['ProductCategory']],
            'ChannelId': [latest['ChannelId']],
            'transaction_hour': [latest['transaction_hour']],
            'transaction_day': [latest['transaction_day']],
            'transaction_month': [latest['transaction_month']],
            'total_transactions': [total_trans],
            'avg_transaction_amount': [avg_amount],
            'std_transaction_amount': [std_amount],
            'total_transaction_value': [total_value],
            'unique_product_categories': [unique_categories],
            'recency': [recency],
            'frequency': [frequency],
            'monetary': [monetary]
        })
        
        # Apply preprocessing
        processed_data = preprocessor.transform(input_data)
        
        # Make prediction
        risk_prob = model.predict_proba(processed_data)[:, 1][0]
        is_high_risk = risk_prob > 0.5  # Threshold
        
        return {
            "risk_probability": round(float(risk_prob), 4),
            "is_high_risk": bool(is_high_risk)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))