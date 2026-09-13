import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import shap

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

def build_final_model(scale_pos_weight):
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

def main() -> None:
    df = pd.read_csv(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = (
        negative_count / positive_count
    )

    model = build_final_model(
        scale_pos_weight
    )

    print("Training final XGBoost model...")

    model.fit(
        X_train,
        y_train,
    )

    preprocessor = model.named_steps[
        "preprocessor"
    ]

    classifier = model.named_steps[
        "classifier"
    ]

    X_test_transformed = (
        preprocessor.transform(X_test)
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    print(
        "Transformed test shape:",
        X_test_transformed.shape,
    )

    print(
        "Number of feature names:",
        len(feature_names),
    )
    
    explainer = shap.TreeExplainer(
        classifier
    )

    shap_values = explainer(
        X_test_transformed
    )

    shap_values.feature_names = (
        feature_names
    )

    print(
        "SHAP values shape:",
        shap_values.values.shape,
    )
    
    global_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": abs(
                shap_values.values
            ).mean(axis=0),
        }
    )

    global_importance = (
        global_importance
        .sort_values(
            by="mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n--- GLOBAL SHAP FEATURE IMPORTANCE ---"
    )

    print(
        global_importance
        .head(10)
        .round(4)
        .to_string(index=False)
    )
    
    REPORTS_DIR = (
        PROJECT_ROOT
        / "reports"
    )

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    shap.summary_plot(
        shap_values.values,
        X_test_transformed,
        feature_names=feature_names,
        show=False,
    )

    plt.tight_layout()

    plot_path = (
        REPORTS_DIR
        / "shap_summary.png"
    )

    plt.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        "\nSHAP summary plot saved to:",
        plot_path,
    )


    churn_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    highest_risk_position = (
        churn_probabilities.argmax()
    )

    highest_risk_probability = (
        churn_probabilities[
            highest_risk_position
        ]
    )

    customer_shap = pd.DataFrame(
        {
            "feature": feature_names,
            "feature_value":
                X_test_transformed[
                    highest_risk_position
                ],
            "shap_value":
                shap_values.values[
                    highest_risk_position
                ],
        }
    )

    customer_shap["abs_shap"] = (
        customer_shap["shap_value"].abs()
    )

    customer_shap = (
        customer_shap
        .sort_values(
            "abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print(
        "\n--- HIGHEST-RISK CUSTOMER ---"
    )

    print(
        f"Test row position: "
        f"{highest_risk_position}"
    )

    print(
        f"Predicted churn probability: "
        f"{highest_risk_probability:.4f}"
    )

    print(
        "\n--- TOP CUSTOMER SHAP DRIVERS ---"
    )

    print(
        customer_shap[
            [
                "feature",
                "feature_value",
                "shap_value",
            ]
        ]
        .head(10)
        .round(4)
        .to_string(index=False)
    )
    
    
if __name__ == "__main__":
    main()
