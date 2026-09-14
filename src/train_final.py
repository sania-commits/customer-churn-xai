from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_preprocessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

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

RANDOM_STATE = 42
SELECTED_THRESHOLD = 0.51


def build_final_model(scale_pos_weight):
    """
    Build the selected production churn model.

    The pipeline contains preprocessing and the
    tuned Weighted XGBoost classifier.
    """

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=7,
                    learning_rate=0.1,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=RANDOM_STATE,
                    eval_metric="logloss",
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def main() -> None:
    """
    Train the final production model using all
    available labeled customer data.
    """

    print("Loading processed customer data...")

    df = pd.read_csv(DATA_PATH)

    # Separate features and target
    X = df.drop(columns="churn")
    y = df["churn"]

    # Calculate class imbalance weight using
    # the complete labeled production dataset
    negative_count = (y == 0).sum()
    positive_count = (y == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"Production training samples: {len(X)}"
    )

    print(
        f"Non-churn customers: {negative_count}"
    )

    print(
        f"Churn customers: {positive_count}"
    )

    print(
        f"scale_pos_weight: {scale_pos_weight:.4f}"
    )

    # Build selected final model
    model = build_final_model(
        scale_pos_weight
    )

    print("\nTraining final production model...")

    # Train using all available labeled data
    model.fit(
        X,
        y,
    )

    # Ensure models directory exists
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save complete preprocessing + model pipeline
    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        "\nModel saved to:",
        MODEL_PATH,
    )

    # Save production model metadata
    metadata = {
        "model_name": (
            "Weighted XGBoost Customer Churn Model"
        ),
        "model_type": "XGBClassifier",
        "selected_threshold": SELECTED_THRESHOLD,
        "scale_pos_weight": round(
            float(scale_pos_weight),
            4,
        ),
        "training_samples": int(
            len(X)
        ),
        "features": list(
            X.columns
        ),
        "hyperparameters": {
            "n_estimators": 300,
            "max_depth": 7,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
        )

    print(
        "Metadata saved to:",
        METADATA_PATH,
    )


if __name__ == "__main__":
    main()
