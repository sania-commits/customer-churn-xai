from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
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

def build_model(scale_pos_weight):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
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


def create_groups(X):
    return (
        X.groupby(
            list(X.columns),
            sort=False,
        )
        .ngroup()
    )


def get_oof_probabilities(
    model,
    X_train,
    y_train,
    groups,
):
    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probability = np.zeros(len(X_train))

    for fold, (train_idx, val_idx) in enumerate(
        cv.split(X_train, y_train, groups),
        start=1,
    ):
        fold_model = clone(model)

        fold_model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
        )

        oof_probability[val_idx] = (
            fold_model.predict_proba(
                X_train.iloc[val_idx]
            )[:, 1]
        )

        print(f"Completed fold {fold}")

    return oof_probability

def find_best_threshold(
    y_true,
    probabilities,
):
    results = []

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "Recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "F1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    results_df = pd.DataFrame(results)

    best_row = results_df.loc[
        results_df["F1"].idxmax()
    ]

    return results_df, best_row


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    groups = create_groups(X_train)

    model = build_model(
        scale_pos_weight
    )

    print(
        "Generating out-of-fold "
        "training probabilities..."
    )

    oof_probability = get_oof_probabilities(
        model,
        X_train,
        y_train,
        groups,
    )

    results, best_row = find_best_threshold(
        y_train,
        oof_probability,
    )

    default_row = results.iloc[
        (results["Threshold"] - 0.50)
        .abs()
        .argsort()[:1]
    ].iloc[0]

    print("\n--- DEFAULT THRESHOLD ---")
    print(default_row.round(4).to_string())

    print("\n--- BEST F1 THRESHOLD ---")
    print(best_row.round(4).to_string())

    best_threshold = float(
    best_row["Threshold"]
    )

    print(
        f"\nEvaluating fixed threshold "
        f"{best_threshold:.2f} on test set..."
    )

    model.fit(
        X_train,
        y_train,
    )

    test_probability = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_probability >= best_threshold
    ).astype(int)

    test_precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    print(
        "\n--- OPTIMIZED THRESHOLD TEST PERFORMANCE ---"
    )

    print(
        "Confusion Matrix:"
    )
    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    print(
        f"\nPrecision: {test_precision:.4f}"
    )
    print(
        f"Recall:    {test_recall:.4f}"
    )
    print(
        f"F1:        {test_f1:.4f}"
    )


if __name__ == "__main__":
    main()


