from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.logger import get_logger
from src.schemas import CustomerFeatures


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"

logger = get_logger("churn_api")


app = FastAPI(
    title="Customer Churn Intelligence API",
    description=(
        "API for telecom customer churn prediction "
        "and risk segmentation."
    ),
    version="1.0.0",
)


try:
    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r") as file:
        metadata = json.load(file)

    logger.info(
        "Production model and metadata loaded successfully."
    )

except Exception:
    logger.exception(
        "Failed to load production model or metadata."
    )
    raise


def assign_risk_segment(probability: float) -> str:
    threshold = metadata["selected_threshold"]

    if probability >= 0.90:
        return "Critical"

    if probability >= threshold:
        return "High"

    if probability >= 0.10:
        return "Medium"

    return "Low"


@app.get("/")
def root():
    return {
        "message": "Customer Churn Intelligence API",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": metadata["model_name"],
        "model_type": metadata["model_type"],
        "threshold": metadata["selected_threshold"],
    }


@app.post("/predict")
def predict(customer: CustomerFeatures):
    try:
        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        churn_probability = float(
            model.predict_proba(customer_df)[0, 1]
        )

        threshold = metadata["selected_threshold"]

        churn_prediction = int(
            churn_probability >= threshold
        )

        risk_segment = assign_risk_segment(
            churn_probability
        )

        logger.info(
            "Prediction completed successfully | "
            "prediction=%s | risk_segment=%s",
            churn_prediction,
            risk_segment,
        )

        return {
            "churn_probability": round(
                churn_probability,
                4,
            ),
            "churn_prediction": churn_prediction,
            "risk_segment": risk_segment,
            "threshold": threshold,
        }

    except Exception as exc:
        logger.exception(
            "Prediction failed due to an internal error."
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc