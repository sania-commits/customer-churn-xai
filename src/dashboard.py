from pathlib import Path

import pandas as pd
import requests
import streamlit as st


# ==================================================
# CONFIGURATION
# ==================================================

API_URL = "https://customer-churn-xai.onrender.com/predict"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RISK_DATA_PATH = (
    PROJECT_ROOT
    / "reports"
    / "customer_risk_segments.csv"
)

SHAP_IMAGE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "shap_summary.png"
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
)


# ==================================================
# DASHBOARD HEADER
# ==================================================

st.title(
    "Customer Churn Intelligence Dashboard"
)

st.write(
    "An explainable machine learning system for "
    "telecom churn prediction, customer risk "
    "segmentation, and retention decision support."
)


# ==================================================
# SECTION 1 — INDIVIDUAL CUSTOMER PREDICTION
# ==================================================

st.header(
    "Individual Customer Prediction"
)

st.write(
    "Enter customer information to estimate "
    "churn probability and retention risk."
)


# --------------------------------------------------
# Customer input form
# --------------------------------------------------

with st.form("customer_form"):

    col1, col2, col3 = st.columns(3)

    # ----------------------------------------------
    # Column 1
    # ----------------------------------------------

    with col1:

        call_failure = st.number_input(
            "Call Failures",
            min_value=0,
            value=0,
            step=1,
        )

        complains = st.selectbox(
            "Complaints",
            options=[0, 1],
            format_func=lambda x: (
                "Yes"
                if x == 1
                else "No"
            ),
        )

        subscription_length = st.number_input(
            "Subscription Length",
            min_value=0,
            value=30,
            step=1,
        )

        charge_amount = st.number_input(
            "Charge Amount",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

        seconds_of_use = st.number_input(
            "Seconds of Use",
            min_value=0,
            value=3000,
            step=100,
        )

    # ----------------------------------------------
    # Column 2
    # ----------------------------------------------

    with col2:

        frequency_of_use = st.number_input(
            "Frequency of Use",
            min_value=0,
            value=50,
            step=1,
        )

        frequency_of_sms = st.number_input(
            "Frequency of SMS",
            min_value=0,
            value=20,
            step=1,
        )

        distinct_called_numbers = st.number_input(
            "Distinct Called Numbers",
            min_value=0,
            value=20,
            step=1,
        )

        age_group = st.selectbox(
            "Age Group",
            options=[
                1,
                2,
                3,
                4,
                5,
            ],
        )

    # ----------------------------------------------
    # Column 3
    # ----------------------------------------------

    with col3:

        tariff_plan = st.selectbox(
            "Tariff Plan",
            options=[
                1,
                2,
            ],
        )

        status = st.selectbox(
            "Status",
            options=[
                1,
                2,
            ],
        )

        age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=30,
            step=1,
        )

        customer_value = st.number_input(
            "Customer Value",
            min_value=0.0,
            value=500.0,
            step=10.0,
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk"
    )


# --------------------------------------------------
# Prediction through FastAPI
# --------------------------------------------------

if submitted:

    customer_data = {
        "call_failure": call_failure,
        "complains": complains,
        "subscription_length": (
            subscription_length
        ),
        "charge_amount": charge_amount,
        "seconds_of_use": seconds_of_use,
        "frequency_of_use": (
            frequency_of_use
        ),
        "frequency_of_sms": (
            frequency_of_sms
        ),
        "distinct_called_numbers": (
            distinct_called_numbers
        ),
        "age_group": age_group,
        "tariff_plan": tariff_plan,
        "status": status,
        "age": age,
        "customer_value": customer_value,
    }

    try:

        response = requests.post(
            API_URL,
            json=customer_data,
            timeout=90,
        )

        response.raise_for_status()

        result = response.json()

        probability = result[
            "churn_probability"
        ]

        prediction = result[
            "churn_prediction"
        ]

        risk_segment = result[
            "risk_segment"
        ]

        threshold = result[
            "threshold"
        ]

        # ------------------------------------------
        # Prediction results
        # ------------------------------------------

        st.divider()

        st.header(
            "Churn Risk Assessment"
        )

        metric1, metric2, metric3 = (
            st.columns(3)
        )

        with metric1:

            st.metric(
                "Churn Probability",
                f"{probability:.2%}",
            )

        with metric2:

            st.metric(
                "Prediction",
                (
                    "Likely to Churn"
                    if prediction == 1
                    else "Likely to Stay"
                ),
            )

        with metric3:

            st.metric(
                "Risk Segment",
                risk_segment,
            )

        st.write(
            "Model decision threshold: "
            f"**{threshold:.2f}**"
        )

        # ------------------------------------------
        # Retention recommendation
        # ------------------------------------------

        st.subheader(
            "Recommended Retention Action"
        )

        if risk_segment == "Critical":

            st.error(
                "Immediate retention intervention "
                "recommended. Prioritize this "
                "customer for proactive outreach."
            )

        elif risk_segment == "High":

            st.warning(
                "Targeted retention outreach "
                "recommended."
            )

        elif risk_segment == "Medium":

            st.info(
                "Monitor the customer and consider "
                "proactive engagement or a "
                "satisfaction check."
            )

        else:

            st.success(
                "Low churn risk. Maintain "
                "standard customer engagement."
            )

    # ----------------------------------------------
    # API connection error
    # ----------------------------------------------

    except requests.exceptions.ConnectionError:

        st.error(
            "Could not connect to the prediction "
            "API. Make sure FastAPI is running on "
            "http://127.0.0.1:8000."
        )

    # ----------------------------------------------
    # API timeout
    # ----------------------------------------------

    except requests.exceptions.Timeout:

        st.error(
            "The prediction API took too long "
            "to respond."
        )

    # ----------------------------------------------
    # Other API errors
    # ----------------------------------------------

    except requests.exceptions.RequestException as error:

        st.error(
            f"Prediction request failed: {error}"
        )


# ==================================================
# SECTION 2 — BUSINESS RISK OVERVIEW
# ==================================================

st.divider()

st.header(
    "Business Risk Overview"
)

st.write(
    "Risk segmentation results from the held-out "
    "customer evaluation dataset."
)


# --------------------------------------------------
# Load customer risk results
# --------------------------------------------------

if RISK_DATA_PATH.exists():

    risk_data = pd.read_csv(
        RISK_DATA_PATH
    )

    total_customers = len(
        risk_data
    )

    high_risk_customers = (
        risk_data[
            "risk_segment"
        ]
        .isin(
            [
                "High",
                "Critical",
            ]
        )
        .sum()
    )

    critical_customers = (
        risk_data[
            "risk_segment"
        ]
        .eq(
            "Critical"
        )
        .sum()
    )

    high_risk_percentage = (
        high_risk_customers
        / total_customers
    )

    # ----------------------------------------------
    # Business KPI metrics
    # ----------------------------------------------

    metric1, metric2, metric3, metric4 = (
        st.columns(4)
    )

    with metric1:

        st.metric(
            "Evaluated Customers",
            total_customers,
        )

    with metric2:

        st.metric(
            "High + Critical Risk",
            int(
                high_risk_customers
            ),
        )

    with metric3:

        st.metric(
            "Critical Risk",
            int(
                critical_customers
            ),
        )

    with metric4:

        st.metric(
            "Priority Population",
            f"{high_risk_percentage:.1%}",
        )

    # ----------------------------------------------
    # Risk distribution chart
    # ----------------------------------------------

    st.subheader(
        "Customer Risk Distribution"
    )

    segment_order = [
        "Low",
        "Medium",
        "High",
        "Critical",
    ]

    risk_counts = (
        risk_data[
            "risk_segment"
        ]
        .value_counts()
        .reindex(
            segment_order
        )
        .fillna(0)
        .astype(int)
    )

    risk_chart_data = pd.DataFrame(
        {
            "Risk Segment": segment_order,
            "Customers": [
                int(
                    risk_counts[
                        segment
                    ]
                )
                for segment in segment_order
            ],
        }
    )

    risk_chart_data[
        "Risk Segment"
    ] = pd.Categorical(
        risk_chart_data[
            "Risk Segment"
        ],
        categories=segment_order,
        ordered=True,
    )

    risk_chart_data = (
        risk_chart_data
        .set_index(
            "Risk Segment"
        )
    )

    st.bar_chart(
        risk_chart_data
    )

    # ----------------------------------------------
    # Risk segment summary
    # ----------------------------------------------

    st.subheader(
        "Risk Segment Summary"
    )

    risk_summary = (
        risk_data
        .groupby(
            "risk_segment",
            observed=True,
        )
        .agg(
            customers=(
                "churn_probability",
                "size",
            ),
            average_probability=(
                "churn_probability",
                "mean",
            ),
            actual_churn_rate=(
                "actual_churn",
                "mean",
            ),
        )
    )

    risk_summary = (
        risk_summary
        .reindex(
            segment_order
        )
    )

    risk_summary[
        "average_probability"
    ] = (
        risk_summary[
            "average_probability"
        ]
        .map(
            lambda value:
            f"{value:.2%}"
        )
    )

    risk_summary[
        "actual_churn_rate"
    ] = (
        risk_summary[
            "actual_churn_rate"
        ]
        .map(
            lambda value:
            f"{value:.2%}"
        )
    )

    st.dataframe(
        risk_summary,
        width="stretch",
    )

    st.caption(
        "These results represent the held-out "
        "evaluation customers used during model "
        "assessment, not the complete production "
        "customer population."
    )


else:

    st.warning(
        "Customer risk segmentation report "
        "is not available."
    )


# ==================================================
# SECTION 3 — MODEL EXPLAINABILITY
# ==================================================

st.divider()

st.header(
    "Model Explainability"
)

st.write(
    "SHAP explanations help show which features "
    "have the greatest influence on the model's "
    "churn predictions."
)


# --------------------------------------------------
# SHAP summary visualization
# --------------------------------------------------

if SHAP_IMAGE_PATH.exists():

    st.subheader(
        "Global SHAP Feature Impact"
    )

    st.image(
        str(
            SHAP_IMAGE_PATH
        ),
        width="stretch",
    )

    st.caption(
        "SHAP values describe how model features "
        "influence predictions. They represent "
        "model behavior and should not be "
        "interpreted as causal effects."
    )

    # ----------------------------------------------
    # Business interpretation
    # ----------------------------------------------

    st.subheader(
        "Key Model Insights"
    )

    insight1, insight2 = st.columns(2)

    with insight1:

        st.markdown(
            """
            **Engagement signals**

            Lower frequency of use and lower
            seconds of use were important signals
            associated with higher model-predicted
            churn risk.

            **Service experience**

            Call failures were among the strongest
            features influencing churn predictions.
            """
        )

    with insight2:

        st.markdown(
            """
            **Customer complaints**

            Complaint-related information was an
            important churn signal in both the EDA
            and model explanations.

            **Retention use**

            These signals can help retention teams
            prioritize investigation and outreach,
            but they should not be interpreted as
            proof that a feature causes churn.
            """
        )


else:

    st.warning(
        "SHAP explanation image is not available."
    )


# ==================================================
# FOOTER
# ==================================================

st.divider()

st.caption(
    "Customer Churn Intelligence | "
    "Weighted XGBoost + FastAPI + Streamlit + SHAP"
)
