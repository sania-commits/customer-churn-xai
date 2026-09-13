from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from features import build_preprocessor
from split_data import split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

RANDOM_STATE = 42
SELECTED_THRESHOLD = 0.51


def build_final_model(scale_pos_weight):
    """
    Build the selected tuned Weighted XGBoost model.
    """

    return Pipeline(
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


def assign_risk_segment(probability):
    """
    Convert churn probability into an actionable
    customer risk segment.
    """

    if probability >= 0.90:
        return "Critical"

    if probability >= SELECTED_THRESHOLD:
        return "High"

    if probability >= 0.10:
        return "Medium"

    return "Low"


def main() -> None:
    """
    Train the final churn model and segment customers
    according to their predicted churn probabilities.
    """

    # Load processed dataset
    df = pd.read_csv(DATA_PATH)

    # Use our leakage-safe train/test split
    X_train, X_test, y_train, y_test = split_data(df)

    # Calculate class imbalance weight using training data only
    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    # Build and train selected final model
    model = build_final_model(
        scale_pos_weight
    )

    print("Training final churn model...")

    model.fit(
        X_train,
        y_train,
    )

    # Generate churn probabilities for test customers
    churn_probability = (
        model.predict_proba(X_test)[:, 1]
    )

    probability_series = pd.Series(
        churn_probability,
        name="churn_probability",
    )

    # Inspect probability distribution
    print(
        "\n--- CHURN PROBABILITY DISTRIBUTION ---"
    )

    print(
        probability_series
        .describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
            ]
        )
        .round(4)
    )

    # Count customers classified as churners
    print(
        "\nCustomers above selected "
        f"threshold ({SELECTED_THRESHOLD}):",
        (
            churn_probability
            >= SELECTED_THRESHOLD
        ).sum(),
    )

    # Create customer-level risk results
    risk_results = X_test.copy()

    risk_results["actual_churn"] = (
        y_test.to_numpy()
    )

    risk_results["churn_probability"] = (
        churn_probability
    )

    risk_results["risk_segment"] = (
        risk_results["churn_probability"]
        .apply(assign_risk_segment)
    )

    # Build business-level segment summary
    segment_summary = (
        risk_results
        .groupby(
            "risk_segment",
            observed=True,
        )
        .agg(
            customers=(
                "churn_probability",
                "size",
            ),
            average_probability=(
                "churn_probability",
                "mean",
            ),
            actual_churn_rate=(
                "actual_churn",
                "mean",
            ),
        )
    )

    # Display segments in business-friendly order
    segment_order = [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]

    segment_summary = (
        segment_summary
        .reindex(segment_order)
    )

    print(
        "\n--- CUSTOMER RISK SEGMENTS ---"
    )

    print(
        segment_summary
        .round(4)
        .to_string()
    )

    OUTPUT_PATH = (
        PROJECT_ROOT
        / "reports"
        / "customer_risk_segments.csv"
    )

    risk_results.to_csv(
        OUTPUT_PATH,
        index=True,
    )

    print(
        "\nCustomer risk segmentation saved to:",
        OUTPUT_PATH,
    )

if __name__ == "__main__":
    main()
