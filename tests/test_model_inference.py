from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "churn_model.joblib"


def create_sample_customer():
    return pd.DataFrame(
        [
            {
                "call_failure": 8,
                "complains": 1,
                "subscription_length": 30,
                "charge_amount": 3,
                "seconds_of_use": 1500,
                "frequency_of_use": 25,
                "frequency_of_sms": 10,
                "distinct_called_numbers": 15,
                "age_group": 2,
                "tariff_plan": 1,
                "status": 1,
                "age": 30,
                "customer_value": 500.0,
            }
        ]
    )


def test_model_file_exists():
    assert MODEL_PATH.exists()


def test_model_loads_as_pipeline():
    model = joblib.load(MODEL_PATH)

    assert isinstance(model, Pipeline)
    assert "preprocessor" in model.named_steps
    assert "classifier" in model.named_steps


def test_model_predicts_probability():
    model = joblib.load(MODEL_PATH)
    customer = create_sample_customer()

    probabilities = model.predict_proba(customer)

    assert probabilities.shape == (1, 2)

    churn_probability = probabilities[0, 1]

    assert 0.0 <= churn_probability <= 1.0
