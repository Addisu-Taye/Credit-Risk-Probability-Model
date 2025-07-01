# Date: 2025-04-05
# Created by: Addisu Taye
# Purpose: Define request and response models for FastAPI endpoint validation.
# Key features:
# - Defines TransactionRequest for individual transaction data
# - Defines CustomerRiskRequest for batch transaction history
# - Defines RiskPredictionResponse for API output
# Task Number: 6
from typing import List, Optional
from pydantic import BaseModel


class TransactionRequest(BaseModel):
    """
    Model for transaction data in request
    """
    Amount: float
    Value: float
    CountryCode: int
    ProviderId: str
    PricingStrategy: str
    ProductCategory: str
    ChannelId: str
    TransactionStartTime: str  # ISO format date string


class CustomerRiskRequest(BaseModel):
    """
    Model for customer data in request
    """
    transactions: List[TransactionRequest]


class RiskPredictionResponse(BaseModel):
    """
    Model for risk prediction response
    """
    risk_probability: float
    is_high_risk: bool