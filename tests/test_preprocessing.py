import pandas as pd

from src.features import (
    TARGET,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    ORDINAL_FEATURES,
    build_preprocessor,
)


def test_target_not_in_features():
    """
    Target must never be included among model features.
    """

    all_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ORDINAL_FEATURES
    )

    assert TARGET not in all_features


def test_expected_feature_count():
    """
    The model should receive exactly 13 raw features.
    """

    all_features = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + ORDINAL_FEATURES
    )

    assert len(all_features) == 13
    assert len(set(all_features)) == 13


def test_preprocessor_transforms_data():
    """
    The preprocessing transformer should successfully
    fit and transform valid customer data.
    """

    sample = pd.DataFrame(
        {
            "call_failure": [8, 2],
            "subscription_length": [30, 40],
            "seconds_of_use": [1500, 5000],
            "frequency_of_use": [25, 80],
            "frequency_of_sms": [10, 70],
            "distinct_called_numbers": [15, 40],
            "age": [30, 45],
            "customer_value": [500.0, 900.0],
            "complains": [1, 0],
            "age_group": [2, 3],
            "tariff_plan": [1, 2],
            "status": [1, 2],
            "charge_amount": [3, 5],
        }
    )

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(
        sample
    )

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 13
