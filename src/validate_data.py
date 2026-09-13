from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Customer Churn.csv"


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the raw customer churn dataset."""
    return pd.read_csv(data_path)


def validate_data(df: pd.DataFrame) -> None:
    """Validate the raw customer churn dataset."""

    expected_columns = [
        "Call  Failure",
        "Complains",
        "Subscription  Length",
        "Charge  Amount",
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Age Group",
        "Tariff Plan",
        "Status",
        "Age",
        "Customer Value",
        "Churn",
    ]

    allowed_values = {
        "Complains": {0, 1},
        "Charge  Amount": set(range(0, 11)),
        "Age Group": {1, 2, 3, 4, 5},
        "Tariff Plan": {1, 2},
        "Status": {1, 2},
        "Churn": {0, 1},
    }

    print("\n--- VALIDATING DATASET ---")

    # 1. Check dataset shape
    if df.shape[1] != 14:
        raise ValueError(
            f"Expected 14 columns, found {df.shape[1]}"
        )

    # 2. Check column names
    if list(df.columns) != expected_columns:
        raise ValueError("Dataset columns do not match expected schema.")

    # 3. Check missing values
    if df.isnull().any().any():
        raise ValueError("Missing values detected in raw dataset.")

    # 4. Check allowed categorical values
    for column, valid_values in allowed_values.items():
        actual_values = set(df[column].unique())

        if not actual_values.issubset(valid_values):
            invalid_values = actual_values - valid_values

            raise ValueError(
                f"Invalid values detected in '{column}': "
                f"{invalid_values}"
            )

    # 5. Check numeric values are non-negative
    numeric_columns = [
        "Call  Failure",
        "Subscription  Length",
        "Seconds of Use",
        "Frequency of use",
        "Frequency of SMS",
        "Distinct Called Numbers",
        "Age",
        "Customer Value",
    ]

    for column in numeric_columns:
        if (df[column] < 0).any():
            raise ValueError(
                f"Negative values detected in '{column}'."
            )

    # 6. Report duplicates but do not remove them
    duplicate_count = df.duplicated().sum()

    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Extra duplicate rows: {duplicate_count}")

    print("\nTarget distribution:")
    print(df["Churn"].value_counts())

    print("\nValidation successful.")


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    validate_data(df)
    
print("\n--- DUPLICATE ANALYSIS ---")

duplicate_mask = df.duplicated(keep=False)
duplicate_rows = df[duplicate_mask]

print("Rows involved in duplicate groups:", duplicate_mask.sum())
print("Extra duplicate rows:", df.duplicated().sum())
print("Unique rows:", len(df.drop_duplicates()))

print("\nMost frequently repeated records:")
print(df.value_counts().head(10))    

print("\n--- CATEGORICAL / ENCODED FEATURE VALUES ---")

categorical_columns = [
    "Complains",
    "Charge  Amount",
    "Age Group",
    "Tariff Plan",
    "Status",
    "Churn",
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(sorted(df[column].unique()))
