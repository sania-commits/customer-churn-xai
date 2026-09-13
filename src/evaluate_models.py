from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


def evaluate_model(
    model_name,
    classifier,
    X_train,
    X_test,
    y_train,
    y_test,
):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ]
    )

    print(f"\nTraining {model_name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n--- {model_name} ---")

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

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
        "PR-AUC": average_precision_score(
            y_test,
            y_probability,
        ),
    }

    return metrics


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    models = {
        "Tuned Random Forest":
            RandomForestClassifier(
                n_estimators=200,
                min_samples_split=5,
                min_samples_leaf=1,
                max_features="sqrt",
                max_depth=None,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),

        "Tuned Weighted XGBoost":
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
    }

    results = []

    for model_name, classifier in models.items():

        metrics = evaluate_model(
            model_name,
            classifier,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    print("\n--- FINAL MODEL COMPARISON ---")

    print(
        results_df
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()


