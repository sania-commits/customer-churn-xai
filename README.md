# Customer Churn Intelligence — Explainable ML for Telecom Retention

An end-to-end machine learning system for predicting telecom customer churn, explaining model decisions, segmenting customers by churn risk, and supporting targeted retention actions.

The project goes beyond model training by implementing a production ML workflow with leakage-aware validation, threshold optimization, explainable AI, FastAPI, Streamlit, automated testing, Docker, CI, and cloud deployment.

## Live Application

- **Streamlit Dashboard:** https://sania-customer-churn-xai.streamlit.app
- **FastAPI Backend:** https://customer-churn-xai.onrender.com
- **Interactive API Docs:** https://customer-churn-xai.onrender.com/docs

> The backend uses a free Render instance and may require additional time for the first request after a period of inactivity.

---

## Business Problem

Customer churn directly affects recurring revenue and customer acquisition costs.

The objective of this project is not simply to classify customers as churners or non-churners. The system is designed to help a retention team answer four practical questions:

1. Which customers are most likely to churn?
2. How confident is the model about that risk?
3. Which customer characteristics contributed most strongly to the prediction?
4. Which customers should receive priority retention attention?

The final application therefore combines churn prediction with explainability, risk segmentation, and retention recommendations.

---

## Dataset

The project uses the **UCI Iranian Telecom Customer Churn dataset**.

- **Customers:** 3,150
- **Predictive features:** 13
- **Target:** `Churn`
- **Non-churn customers:** 2,655
- **Churn customers:** 495
- **Churn rate:** 15.71%

The dataset contains customer usage, subscription, complaint, demographic, tariff, and customer-value information.

Examples include:

- call failures
- complaints
- subscription length
- seconds of use
- frequency of use
- frequency of SMS
- distinct called numbers
- age and age group
- tariff plan
- customer status
- customer value

---

## Data Quality and Leakage Prevention

Initial profiling identified repeated feature profiles in the dataset.

Because the dataset does not contain a reliable customer identifier, identical rows were not blindly deleted: they can potentially represent different customers with the same recorded characteristics.

A standard random split initially resulted in **84 identical feature profiles appearing in both training and test data**.

To prevent this form of train-test leakage, the final workflow uses:

`StratifiedGroupKFold`

Customers with identical predictive feature profiles are assigned to the same group so that a profile cannot appear in both training and test partitions.

Final split:

- Training samples: **2,517**
- Test samples: **633**
- Training churn rate: **15.69%**
- Test churn rate: **15.80%**
- Shared identical feature profiles: **0**

---

## Exploratory Data Analysis

EDA identified several strong behavioral differences between retained and churned customers.

Examples:

| Feature | Retained Customers | Churned Customers |
|---|---:|---:|
| Seconds of Use | 5,014.22 | 1,566.63 |
| Frequency of Use | 76.98 | 29.13 |
| Frequency of SMS | 83.87 | 15.80 |

Customers with recorded complaints also showed substantially higher observed churn in this dataset.

These relationships are treated as **associations**, not causal effects.

---

## Machine Learning Pipeline

The preprocessing and model are combined in a reusable Scikit-learn pipeline.

### Numeric Features

Numeric variables are standardized using:

`StandardScaler`

### Categorical Features

Categorical variables are transformed using:

`OneHotEncoder(handle_unknown="ignore")`

### Ordinal Feature

`charge_amount` is retained as an ordinal encoded feature.

Keeping preprocessing inside the ML pipeline helps ensure that transformations are fitted only on the appropriate training data.

---

## Model Development

Four candidate models were compared:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9052 | 0.8226 | 0.5100 | 0.6296 | 0.9435 |
| Decision Tree | 0.9542 | 0.8515 | 0.8600 | 0.8557 | 0.9177 |
| Random Forest | 0.9700 | 0.9263 | 0.8800 | 0.9026 | 0.9948 |
| XGBoost | 0.9684 | 0.9255 | 0.8700 | 0.8969 | 0.9933 |

Because churn is the minority class, additional class-imbalance experiments were performed.

Weighted XGBoost improved churn recall to approximately **93%** before final tuning.

---

## Hyperparameter Tuning

The two strongest candidates were:

- Random Forest
- Weighted XGBoost

Hyperparameters were tuned using:

- `RandomizedSearchCV`
- 5-fold `StratifiedGroupKFold`
- F1 score as the optimization metric

The grouping strategy was retained during cross-validation to avoid identical customer profiles leaking between folds.

### Selected XGBoost Configuration

```python
XGBClassifier(
    n_estimators=300,
    max_depth=7,
    learning_rate=0.1,
    subsample=1.0,
    colsample_bytree=1.0,
    scale_pos_weight=5.3722,
    random_state=42,
    eval_metric="logloss",
    n_jobs=-1,
)
```

---

## Final Model Evaluation

The tuned weighted XGBoost model was selected as the final model.

Performance on the untouched test set:

| Metric | Result |
|---|---:|
| Accuracy | 96.84% |
| Precision | 89.22% |
| Recall | **91.00%** |
| F1 Score | **90.10%** |
| ROC-AUC | **0.9941** |
| PR-AUC | **0.9685** |

Confusion matrix:

```text
[[522, 11],
 [  9, 91]]
```

The model correctly identified **91 of 100 churners** in the held-out test set.

---

## Decision Threshold Optimization

Instead of automatically assuming a classification threshold of `0.50`, the threshold was evaluated using group-aware out-of-fold predictions from the training data.

The selected threshold was:

**0.51**

Threshold selection was performed without using the final test labels.

The threshold was then fixed before final evaluation.

---

## Explainable AI with SHAP

SHAP is used to understand the model's predictions at both global and customer levels.

The strongest global model drivers included:

1. Frequency of use
2. Customer status
3. Call failures
4. Seconds of use
5. Complaints
6. Subscription length
7. Customer value
8. Distinct called numbers
9. Frequency of SMS

SHAP values explain how features influence the model prediction. They should not be interpreted as evidence of causal relationships.

---

## Customer Risk Segmentation

Predicted churn probabilities are converted into business-oriented risk segments.

| Segment | Probability |
|---|---|
| Low | `< 0.10` |
| Medium | `0.10 – < 0.51` |
| High | `0.51 – < 0.90` |
| Critical | `>= 0.90` |

Held-out evaluation population:

| Risk Segment | Customers | Avg. Probability | Actual Churn Rate |
|---|---:|---:|---:|
| Low | 502 | 0.0030 | 0.20% |
| Medium | 29 | 0.2097 | 27.59% |
| High | 15 | 0.7771 | 66.67% |
| Critical | 87 | 0.9873 | 93.10% |

This converts raw model probabilities into a prioritization framework that can be used by retention teams.

---

## Retention Decision Support

Each risk group receives an operational recommendation:

| Risk | Priority | Recommended Action |
|---|---|---|
| Low | Routine | Maintain standard customer engagement |
| Medium | Monitor | Proactive engagement and satisfaction check |
| High | High | Targeted retention outreach |
| Critical | Immediate | Immediate retention intervention |

These recommendations are decision-support rules based on predicted risk, not claims of causal treatment effectiveness.

---

## Production Architecture

```text
                        GitHub Repository
                               |
                        GitHub Actions CI
                               |
                +--------------+--------------+
                |                             |
                v                             v
          Render Cloud               Streamlit Community Cloud
                |                             |
                v                             v
        Docker Container               Streamlit Dashboard
                |
                v
           FastAPI API
                |
                v
       Scikit-learn Pipeline
                |
                v
      Weighted XGBoost Model
                |
                v
       Churn Probability
                |
                v
          Risk Segment
                |
                v
    Retention Recommendation
```

---

## FastAPI Prediction Service

The trained production pipeline is exposed through FastAPI.

Main endpoints:

```text
GET  /
GET  /health
POST /predict
GET  /docs
```

Example prediction request:

```json
{
  "call_failure": 8,
  "complains": 1,
  "subscription_length": 30,
  "charge_amount": 3,
  "seconds_of_use": 1500,
  "frequency_of_use": 25,
  "frequency_of_sms": 10,
  "distinct_called_numbers": 15,
  "age_group": 2,
  "tariff_plan": 1,
  "status": 1,
  "age": 30,
  "customer_value": 500.0
}
```

Example response:

```json
{
  "churn_probability": 0.9991,
  "churn_prediction": 1,
  "risk_segment": "Critical",
  "threshold": 0.51
}
```

---

## Streamlit Dashboard

The frontend provides:

- individual customer churn prediction
- churn probability
- churn classification
- risk segmentation
- recommended retention action
- business risk overview
- SHAP model insights

**Live dashboard:**  
https://sania-customer-churn-xai.streamlit.app

---

## Automated Testing

The project includes automated tests covering:

- preprocessing
- risk-segment boundaries
- production model loading and inference
- FastAPI health endpoint
- valid prediction requests
- schema validation
- missing/invalid inputs
- controlled internal prediction failures

Tests can be executed with:

```bash
python -m pytest tests/ -v
```

---

## Docker

The FastAPI backend is containerized using Docker.

Build:

```bash
docker build -t customer-churn-api:1.0 .
```

Run:

```bash
docker run --name customer-churn-api \
  -p 8000:8000 \
  customer-churn-api:1.0
```

Local API:

```text
http://127.0.0.1:8000
```

---

## CI

GitHub Actions automatically creates a clean Python environment, installs project dependencies, and executes the automated test suite for pushes and pull requests to `main`.

This provides automated validation outside the local development environment.

---

## Cloud Deployment

### Backend

The Dockerized FastAPI prediction service is deployed on **Render**.

https://customer-churn-xai.onrender.com

### Frontend

The Streamlit dashboard is deployed on **Streamlit Community Cloud**.

https://sania-customer-churn-xai.streamlit.app

---

## Project Structure

```text
customer-churn-xai/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── churn_model.joblib
│   └── model_metadata.json
├── notebooks/
│   └── 01_eda_business_insights.ipynb
├── reports/
├── src/
│   ├── api.py
│   ├── dashboard.py
│   ├── evaluate_models.py
│   ├── explain_model.py
│   ├── features.py
│   ├── imbalance_experiments.py
│   ├── logger.py
│   ├── optimize_threshold.py
│   ├── preprocess.py
│   ├── retention_strategy.py
│   ├── risk_segmentation.py
│   ├── schemas.py
│   ├── train_baseline.py
│   ├── train_final.py
│   ├── tune_models.py
│   └── validate_data.py
├── tests/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Tech Stack

**Machine Learning:** Python, Pandas, NumPy, Scikit-learn, XGBoost  
**Explainability:** SHAP  
**Backend:** FastAPI, Pydantic, Uvicorn  
**Frontend:** Streamlit  
**Testing:** Pytest  
**Containerization:** Docker  
**CI:** GitHub Actions  
**Deployment:** Render, Streamlit Community Cloud  
**Version Control:** Git, GitHub

---

## Key Engineering Decisions

This project intentionally demonstrates more than predictive performance:

- leakage-aware train/test splitting
- group-aware cross-validation
- class-imbalance handling
- threshold optimization using training-only OOF predictions
- untouched final test evaluation
- reusable preprocessing/model pipeline
- SHAP explainability
- business-oriented risk segmentation
- input validation and error handling
- application logging
- automated testing
- containerized inference
- CI validation
- cloud deployment

---

## Important Modeling Note

After model selection and evaluation were completed, the production pipeline was retrained using all **3,150 labeled customers**.

The deployed production artifact is therefore intended for inference.

Reported performance metrics in this README come from the earlier **untouched 633-customer test set**, not from evaluating the production model on the same data used to train it.

---

## Author

**Sania Anjum**

Data Science | Machine Learning | Explainable AI | Production ML
