from pathlib import Path
import json

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from src.schemas import CustomerFeatures


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "churn_model.joblib"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "model_metadata.json"
)


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Intelligence API",
    description=(
        "API for telecom customer churn prediction "
        "and retention risk intelligence."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# Load trained production model
# --------------------------------------------------

model = joblib.load(
    MODEL_PATH
)


# --------------------------------------------------
# Load model metadata
# --------------------------------------------------

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8",
) as file:
    model_metadata = json.load(file)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root():
    """
    Basic API information endpoint.
    """

    return {
        "message": (
            "Customer Churn Intelligence API"
        ),
        "docs": "/docs",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health():
    """
    Check whether the API and model are available.
    """

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": model_metadata[
            "model_name"
        ],
        "model_type": model_metadata[
            "model_type"
        ],
        "threshold": model_metadata[
            "selected_threshold"
        ],
    }


# --------------------------------------------------
# Risk segmentation
# --------------------------------------------------

def assign_risk_segment(
    probability: float,
) -> str:
    """
    Convert churn probability into a
    business-friendly risk segment.
    """

    if probability >= 0.90:
        return "Critical"

    if probability >= model_metadata[
        "selected_threshold"
    ]:
        return "High"

    if probability >= 0.10:
        return "Medium"

    return "Low"


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(
    customer: CustomerFeatures,
):
    """
    Predict customer churn probability,
    binary churn prediction, and risk segment.
    """

    try:
        # Convert validated Pydantic input
        # into a one-row Pandas DataFrame.
        customer_data = pd.DataFrame(
            [
                customer.model_dump()
            ]
        )

        # Generate churn probability.
        churn_probability = float(
            model.predict_proba(
                customer_data
            )[0, 1]
        )

        # Read our selected decision threshold.
        threshold = float(
            model_metadata[
                "selected_threshold"
            ]
        )

        # Convert probability into
        # binary churn prediction.
        churn_prediction = int(
            churn_probability
            >= threshold
        )

        # Convert probability into
        # business risk segment.
        risk_segment = (
            assign_risk_segment(
                churn_probability
            )
        )

        # Return prediction response.
        return {
            "churn_probability": round(
                churn_probability,
                4,
            ),
            "churn_prediction": (
                churn_prediction
            ),
            "risk_segment": risk_segment,
            "threshold": threshold,
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from error