from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RISK_DATA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "customer_risk_segments.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "customer_retention_actions.csv"
)

RETENTION_STRATEGY = {
    "Low": {
        "priority": "Routine",
        "action": "Maintain standard customer engagement",
    },

    "Medium": {
        "priority": "Monitor",
        "action": "Send proactive engagement and satisfaction check",
    },

    "High": {
        "priority": "High",
        "action": "Initiate targeted retention outreach",
    },

    "Critical": {
        "priority": "Immediate",
        "action": "Escalate for immediate retention intervention",
    },
}



DRIVER_ACTIONS = {
    "complains": (
        "Review unresolved complaints and prioritize "
        "customer-support follow-up"
    ),

    "call_failure": (
        "Investigate service reliability and call-quality issues"
    ),

    "frequency_of_use": (
        "Review declining engagement and consider "
        "proactive re-engagement"
    ),

    "seconds_of_use": (
        "Review reduced service usage and customer engagement"
    ),

    "frequency_of_sms": (
        "Monitor declining communication activity"
    ),

    "customer_value": (
        "Consider customer value when prioritizing "
        "retention resources"
    ),
}



def add_retention_actions(df):
    """
    Add business priority and recommended retention
    action based on each customer's risk segment.
    """

    result = df.copy()

    result["retention_priority"] = (
        result["risk_segment"]
        .map(
            lambda segment:
            RETENTION_STRATEGY[segment]["priority"]
        )
    )

    result["recommended_action"] = (
        result["risk_segment"]
        .map(
            lambda segment:
            RETENTION_STRATEGY[segment]["action"]
        )
    )

    return result


def main() -> None:
    risk_data = pd.read_csv(
        RISK_DATA_PATH
    )

    retention_data = add_retention_actions(
        risk_data
    )

    print(
        "--- RETENTION ACTION SUMMARY ---"
    )

    summary = (
        retention_data
        .groupby(
            [
                "risk_segment",
                "retention_priority",
                "recommended_action",
            ],
            observed=True,
        )
        .size()
        .reset_index(
            name="customers"
        )
    )

    segment_order = {
        "Low": 0,
        "Medium": 1,
        "High": 2,
        "Critical": 3,
    }

    summary["segment_order"] = (
        summary["risk_segment"]
        .map(segment_order)
    )

    summary = (
        summary
        .sort_values("segment_order")
        .drop(columns="segment_order")
    )

    print(
        summary.to_string(
            index=False
        )
    )

    retention_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nRetention recommendations saved to:",
        OUTPUT_PATH,
    )

    print(
        "\n--- EXAMPLE DRIVER-BASED ACTIONS ---"
    )

    example_drivers = [
        "categorical__complains_0",
        "numeric__call_failure",
        "numeric__frequency_of_use",
        "numeric__seconds_of_use",
    ]

    for driver in example_drivers:
        print(
            f"{driver}: "
            f"{get_driver_action(driver)}"
        )


def get_driver_action(feature_name):
    """
    Map an important churn driver to a business-friendly
    investigation or retention recommendation.
    """

    for driver, action in DRIVER_ACTIONS.items():
        if driver in feature_name:
            return action

    return "Review customer profile and perform targeted retention assessment"



if __name__ == "__main__":
    main()
