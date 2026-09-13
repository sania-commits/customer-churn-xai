from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "churn"

NUMERIC_FEATURES = [
    "call_failure",
    "subscription_length",
    "seconds_of_use",
    "frequency_of_use",
    "frequency_of_sms",
    "distinct_called_numbers",
    "age",
    "customer_value",
]

CATEGORICAL_FEATURES = [
    "complains",
    "age_group",
    "tariff_plan",
    "status",
]

ORDINAL_FEATURES = [
    "charge_amount",
]


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing transformer used by ML models.
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "ordinal",
                "passthrough",
                ORDINAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor