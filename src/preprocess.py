from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_PATH = (
    PROJECT_ROOT / "data" / "raw" / "Customer Churn.csv"
)

PROCESSED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "customer_churn_clean.csv"
)


COLUMN_MAPPING = {
    "Call  Failure": "call_failure",
    "Complains": "complains",
    "Subscription  Length": "subscription_length",
    "Charge  Amount": "charge_amount",
    "Seconds of Use": "seconds_of_use",
    "Frequency of use": "frequency_of_use",
    "Frequency of SMS": "frequency_of_sms",
    "Distinct Called Numbers": "distinct_called_numbers",
    "Age Group": "age_group",
    "Tariff Plan": "tariff_plan",
    "Status": "status",
    "Age": "age",
    "Customer Value": "customer_value",
    "Churn": "churn",
}



def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply deterministic preprocessing to the raw churn dataset.

    The raw dataset is left unchanged.
    """

    df = df.copy()

    # Standardize column names
    df = df.rename(columns=COLUMN_MAPPING)

    # Validate target
    if not set(df["churn"].unique()).issubset({0, 1}):
        raise ValueError("Churn target contains unexpected values.")

    # Validate missing values
    if df.isnull().any().any():
        raise ValueError("Missing values detected during preprocessing.")

    return df


def main() -> None:
    """Load raw data, preprocess it, and save the cleaned dataset."""

    print("Loading raw dataset...")

    df = pd.read_csv(RAW_DATA_PATH)

    print(f"Raw shape: {df.shape}")

    processed_df = preprocess_data(df)

    PROCESSED_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    processed_df.to_csv(
        PROCESSED_DATA_PATH,
        index=False
    )

    print(f"Processed shape: {processed_df.shape}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")


if __name__ == "__main__":
    main()