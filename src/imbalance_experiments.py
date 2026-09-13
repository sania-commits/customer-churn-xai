from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
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

    print(f"Training {model_name}...")

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_probability = pipeline.predict_proba(X_test)[:, 1]

    return {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(
            y_test,
            y_probability,
        ),
    }



def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    print(
        "Training class counts:",
        f"non-churn={negative_count},",
        f"churn={positive_count}",
    )

    print(
        f"XGBoost scale_pos_weight: "
        f"{scale_pos_weight:.4f}\n"
    )

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),

        "Random Forest Balanced":
            RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),

        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),

        "XGBoost Weighted": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }

    results = []

    for model_name, classifier in models.items():
        result = evaluate_model(
            model_name,
            classifier,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    print("\n--- CLASS IMBALANCE EXPERIMENTS ---")

    print(
        results_df
        .sort_values(
            by="F1",
            ascending=False,
        )
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()




