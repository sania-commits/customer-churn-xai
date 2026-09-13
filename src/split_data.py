from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

TARGET = "churn"

N_SPLITS = 5
RANDOM_STATE = 42


def load_processed_data() -> pd.DataFrame:
    """Load the cleaned customer churn dataset."""
    return pd.read_csv(DATA_PATH)


def split_data(df: pd.DataFrame):
    """
    Create a stratified, group-aware train/test split.

    Identical feature profiles are kept entirely within either
    the training set or the testing set to reduce leakage.
    """

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Give identical feature profiles the same group ID
    groups = (
        X.groupby(
            list(X.columns),
            sort=False,
        )
        .ngroup()
    )

    splitter = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_processed_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("Full dataset:", df.shape)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    print(
        "\nOverall churn rate:",
        round(df[TARGET].mean() * 100, 2),
        "%"
    )

    print(
        "Training churn rate:",
        round(y_train.mean() * 100, 2),
        "%"
    )

    print(
        "Testing churn rate:",
        round(y_test.mean() * 100, 2),
        "%"
    )
    train_profiles = set(
        map(tuple, X_train.to_numpy())
    )

    test_profiles = set(
        map(tuple, X_test.to_numpy())
    )

    overlapping_profiles = (
        train_profiles.intersection(test_profiles)
    )

    print(
        "\nIdentical feature profiles shared between "
        "train and test:",
        len(overlapping_profiles),
    )
