from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from features import build_preprocessor
from split_data import split_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

def build_baseline_model() -> Pipeline:
    """Build the Logistic Regression baseline pipeline."""

    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    return pipeline

def main() -> None:
    """Train and evaluate the baseline churn model."""

    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    model = build_baseline_model()

    print("Training baseline Logistic Regression model...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print("\n--- BASELINE MODEL PERFORMANCE ---")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
        )
    )

    roc_auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print(f"ROC-AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()


