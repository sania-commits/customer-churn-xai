from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
)
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


def create_groups(X):
    """Assign identical feature profiles to the same group."""

    return (
        X.groupby(
            list(X.columns),
            sort=False,
        )
        .ngroup()
    )


def build_cv():
    return StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


RF_PARAMS = {
    "classifier__n_estimators": [
        100,
        200,
        300,
        500,
    ],
    "classifier__max_depth": [
        None,
        5,
        10,
        20,
    ],
    "classifier__min_samples_split": [
        2,
        5,
        10,
    ],
    "classifier__min_samples_leaf": [
        1,
        2,
        4,
    ],
    "classifier__max_features": [
        "sqrt",
        "log2",
    ],
}


XGB_PARAMS = {
    "classifier__n_estimators": [
        100,
        200,
        300,
        500,
    ],
    "classifier__max_depth": [
        3,
        5,
        7,
    ],
    "classifier__learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2,
    ],
    "classifier__subsample": [
        0.7,
        0.8,
        1.0,
    ],
    "classifier__colsample_bytree": [
        0.7,
        0.8,
        1.0,
    ],
}



def tune_model(
    model_name,
    classifier,
    param_distributions,
    X_train,
    y_train,
    groups,
):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="f1",
        cv=build_cv(),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )

    print(f"\nTuning {model_name}...")

    search.fit(
        X_train,
        y_train,
        groups=groups,
    )

    print(
        f"{model_name} best CV F1: "
        f"{search.best_score_:.4f}"
    )

    print(
        f"{model_name} best parameters:"
    )

    for parameter, value in search.best_params_.items():
        print(f"  {parameter}: {value}")

    return search



def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    groups = create_groups(X_train)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.4f}"
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    xgboost = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )

    tune_model(
        "Random Forest",
        random_forest,
        RF_PARAMS,
        X_train,
        y_train,
        groups,
    )

    tune_model(
        "Weighted XGBoost",
        xgboost,
        XGB_PARAMS,
        X_train,
        y_train,
        groups,
    )


if __name__ == "__main__":
    main()
