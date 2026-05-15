from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import joblib
import pandas as pd
import numpy as np
import uvicorn

app = FastAPI(
    title="Credit Card Retention Campaign API",
    description="Predicts offer acceptance probability for SME credit card customers",
    version="1.0.0"
)

# Load the serialized pipeline at startup
pipeline = joblib.load("models/offer_acceptance_pipeline.joblib")

NUMERIC_FEATURES = [
    "Customer_Age", "Annual_Income_USD", "Credit_Score", "Discount_Percent",
    "Credit_Limit_USD", "Avg_Monthly_Spend_USD", "Profit_Margin_Percent",
    "Spend_to_Income_Ratio", "Credit_Utilization", "Is_High_Value",
    "Month_Sin", "Month_Cos"
]

CATEGORICAL_FEATURES = [
    "Offer_Type", "Campaign_Channel", "Region", "Customer_Segment",
    "Income_Band", "Offer_Discount_Group", "Age_Group"
]


class CustomerInput(BaseModel):
    Customer_Age: int
    Annual_Income_USD: float
    Credit_Score: int
    Discount_Percent: float
    Credit_Limit_USD: float
    Avg_Monthly_Spend_USD: float
    Profit_Margin_Percent: float
    Spend_to_Income_Ratio: float
    Credit_Utilization: float
    Is_High_Value: int
    Month_Sin: float
    Month_Cos: float
    Offer_Type: str
    Campaign_Channel: str
    Region: str
    Customer_Segment: str
    Income_Band: str
    Offer_Discount_Group: str
    Age_Group: str


class PredictionResponse(BaseModel):
    acceptance_probability: float
    propensity_tier: str
    recommended_action: str


def get_tier_and_action(prob: float, is_high_value: int):
    if prob >= 0.85 and is_high_value:
        tier = "Hot Lead"
        action = "Priority Outreach: Premium Offer + Dedicated RM Call"
    elif prob >= 0.85:
        tier = "Hot Lead"
        action = "Digital Outreach: Cashback or Travel Rewards Offer"
    elif prob >= 0.70:
        tier = "Warm Lead"
        action = "Email Campaign: Dining Rewards or Low Interest Offer"
    elif prob >= 0.50:
        tier = "Cold Lead"
        action = "SMS Nudge: Discount-Heavy Offer"
    else:
        tier = "Not Likely"
        action = "No Campaign: Include in Next Cycle Review"
    return tier, action


@app.get("/")
def root():
    return {"status": "running", "service": "Retention Campaign Predictor"}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput):
    try:
        input_df = pd.DataFrame([customer.dict()])
        prob = float(pipeline.predict_proba(input_df)[:, 1][0])
        tier, action = get_tier_and_action(prob, customer.Is_High_Value)
        return PredictionResponse(
            acceptance_probability=round(prob, 4),
            propensity_tier=tier,
            recommended_action=action
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
def batch_predict(customers: list[CustomerInput]):
    results = []
    input_df = pd.DataFrame([c.dict() for c in customers])
    probas = pipeline.predict_proba(input_df)[:, 1]
    for prob, customer in zip(probas, customers):
        tier, action = get_tier_and_action(float(prob), customer.Is_High_Value)
        results.append({
            "acceptance_probability": round(float(prob), 4),
            "propensity_tier": tier,
            "recommended_action": action
        })
    return {"predictions": results}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
