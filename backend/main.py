"""
Credit Card Fraud Detection — FastAPI Backend
Loads the serialized model + scaler produced by notebook/fraud_training.ipynb
and exposes a /predict endpoint for the Streamlit frontend.

Run with:
    uvicorn main:app --reload --port 8007
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Serves real-time fraud predictions for individual transactions.",
    version="1.0.0",
)

# Allow the Streamlit frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
scaler = None

V_FIELDS = [f"V{i}" for i in range(1, 29)]
FEATURE_ORDER = ["Time"] + V_FIELDS + ["Amount"]


@app.on_event("startup")
def load_artifacts():
    global model, scaler
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError(
            "Model artifacts not found. Run notebook/fraud_training.ipynb first, "
            "then copy fraud_model.pkl and scaler.pkl into the backend/ folder."
        )
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)


class Transaction(BaseModel):
    Time: float = Field(..., description="Seconds elapsed since the first transaction")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0, description="Transaction amount, must be non-negative")

    @field_validator("Amount")
    @classmethod
    def amount_must_be_non_negative(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "Time": 10000,
                **{f"V{i}": 0.1 if i % 2 == 0 else -0.1 for i in range(1, 29)},
                "Amount": 150.75,
            }
        }


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    message: str


def _predict_single(payload: dict) -> PredictionResponse:
    row = pd.DataFrame([payload])[FEATURE_ORDER]
    row[["Time", "Amount"]] = scaler.transform(row[["Time", "Amount"]])

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1])

    message = (
        "Fraudulent transaction detected"
        if prediction == 1
        else "Transaction appears legitimate"
    )
    return PredictionResponse(prediction=prediction, probability=round(probability, 4), message=message)


@app.get("/")
def root():
    return {"status": "ok", "service": "Credit Card Fraud Detection API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"model_loaded": model is not None, "scaler_loaded": scaler is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")
    try:
        return _predict_single(transaction.model_dump())
    except Exception as exc:  # defensive: never leak a raw 500 with no context
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}")
