from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from features import build_preprocessor
from split_data import split_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)


MODELS = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    ),
    
    "XGBoost": XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
    ),
}

def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
) -> pd.DataFrame:

    results = []

    for model_name, classifier in MODELS.items():

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

        results.append(
            {
                "Model": model_name,
                "Accuracy": accuracy_score(
                    y_test,
                    y_pred,
                ),
                "Precision": precision_score(
                    y_test,
                    y_pred,
                ),
                "Recall": recall_score(
                    y_test,
                    y_pred,
                ),
                "F1": f1_score(
                    y_test,
                    y_pred,
                ),
                "ROC-AUC": roc_auc_score(
                    y_test,
                    y_probability,
                ),
            }
        )

    return pd.DataFrame(results)

def main() -> None:

    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    results = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\n--- MODEL COMPARISON ---")

    print(
        results
        .sort_values(
            by="ROC-AUC",
            ascending=False,
        )
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()


