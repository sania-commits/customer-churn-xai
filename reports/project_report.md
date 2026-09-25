# Customer Churn Intelligence
## Explainable Machine Learning for Telecom Customer Retention

**Author:** Sania Anjum

---

## 1. Executive Summary

This project develops an end-to-end machine learning system for telecom customer churn prediction and retention decision support.

The solution combines:

- leakage-aware model validation
- class-imbalance handling
- hyperparameter tuning
- decision-threshold optimization
- SHAP explainability
- customer risk segmentation
- retention recommendations
- production model training
- FastAPI inference
- Streamlit visualization
- automated testing
- Docker containerization
- GitHub Actions CI
- cloud deployment

The final selected model is a weighted XGBoost classifier.

On the untouched test population, the selected model achieved:

| Metric | Result |
|---|---:|
| Accuracy | 96.84% |
| Precision | 89.22% |
| Recall | 91.00% |
| F1 Score | 90.10% |
| ROC-AUC | 0.9941 |
| PR-AUC | 0.9685 |

The deployed application converts churn probabilities into Low, Medium, High, and Critical risk segments to support customer-retention prioritization.

---

## 2. Business Objective

Customer churn can reduce recurring revenue and increase the cost of customer acquisition.

The project addresses four practical questions:

1. Which customers are at risk of churn?
2. What is their estimated churn probability?
3. Which customer characteristics influence the model's prediction?
4. How should customers be prioritized for retention attention?

The project therefore treats machine learning as part of a larger decision-support system rather than as an isolated classification exercise.

---

## 3. Dataset

The project uses the UCI Iranian Telecom Customer Churn dataset.

Dataset characteristics:

- 3,150 customer records
- 13 predictive features
- binary churn target
- 2,655 non-churn observations
- 495 churn observations
- overall churn rate: 15.71%

The predictors include customer usage, complaints, subscription characteristics, tariff information, demographic variables, and customer value.

---

## 4. Data Quality and Leakage Prevention

Initial validation found no missing values.

The dataset contained repeated feature profiles. Because no reliable customer identifier was available, duplicate-looking observations were not automatically removed.

An initial conventional stratified split resulted in 84 identical predictive profiles being shared between training and test sets.

This creates a leakage risk because the model may effectively encounter the same feature profile during both training and evaluation.

To address this, identical predictive profiles were assigned group identifiers and the final split used StratifiedGroupKFold.

Final split:

- Training observations: 2,517
- Test observations: 633
- Training churn rate: 15.69%
- Test churn rate: 15.80%
- Identical feature profiles shared across train/test: 0

The same group-aware principle was retained during model tuning and threshold optimization.

---

## 5. Exploratory Analysis

Behavioral differences were observed between retained and churned customers.

Examples:

| Feature | Retained | Churned |
|---|---:|---:|
| Seconds of Use | 5,014.22 | 1,566.63 |
| Frequency of Use | 76.98 | 29.13 |
| Frequency of SMS | 83.87 | 15.80 |

Complaint status also showed a strong association with churn in the observed dataset.

These findings are descriptive associations and are not interpreted as causal effects.

---

## 6. Preprocessing

The preprocessing workflow is implemented using a Scikit-learn ColumnTransformer.

Numeric features are standardized using StandardScaler.

Categorical features are encoded using OneHotEncoder with unknown-category handling.

The encoded charge amount variable is retained as an ordinal feature.

Preprocessing and classification are combined inside a Scikit-learn Pipeline so transformations can be applied consistently during training and inference.

---

## 7. Model Development

The initial candidate models were:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost

Initial held-out comparison:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9052 | 0.8226 | 0.5100 | 0.6296 | 0.9435 |
| Decision Tree | 0.9542 | 0.8515 | 0.8600 | 0.8557 | 0.9177 |
| Random Forest | 0.9700 | 0.9263 | 0.8800 | 0.9026 | 0.9948 |
| XGBoost | 0.9684 | 0.9255 | 0.8700 | 0.8969 | 0.9933 |

Random Forest and XGBoost were selected for further experimentation.

---

## 8. Class Imbalance

The churn class represented approximately 15.7% of the dataset.

Class weighting was evaluated instead of relying only on overall accuracy.

For XGBoost, scale_pos_weight was derived from the ratio of non-churn to churn observations in the training population.

The weighted XGBoost experiment increased churn recall while maintaining strong precision and discrimination.

---

## 9. Hyperparameter Tuning

Random Forest and weighted XGBoost were tuned using RandomizedSearchCV.

The tuning procedure used:

- 20 randomized configurations
- F1 optimization
- five-fold StratifiedGroupKFold
- group-aware validation based on repeated predictive profiles

The selected XGBoost hyperparameters were:

- n_estimators: 300
- max_depth: 7
- learning_rate: 0.1
- subsample: 1.0
- colsample_bytree: 1.0

The training-set class weight was approximately 5.3722.

---

## 10. Final Model Evaluation

The tuned weighted XGBoost model was selected as the final evaluated model.

Confusion matrix:

```text
[[522, 11],
 [  9, 91]]
```

This corresponds to:

- 522 true negatives
- 11 false positives
- 9 false negatives
- 91 true positives

Final test metrics:

- Accuracy: 96.84%
- Precision: 89.22%
- Recall: 91.00%
- F1: 90.10%
- ROC-AUC: 0.9941
- PR-AUC: 0.9685

The model identified 91 of the 100 churners in the held-out test population.

---

## 11. Decision Threshold Optimization

A fixed probability threshold of 0.50 was not assumed automatically.

Instead, group-aware out-of-fold predictions were generated from the training data.

Thresholds from 0.10 to 0.90 were evaluated using F1 score.

The selected operating threshold was:

**0.51**

The final test set was not used to choose this threshold.

---

## 12. Explainable AI

SHAP was used to analyze model behavior.

Leading global model drivers included:

1. frequency of use
2. customer status
3. call failures
4. seconds of use
5. complaints
6. subscription length
7. customer value
8. distinct called numbers
9. frequency of SMS

The SHAP analysis describes how features contribute to model predictions.

It does not establish that changing a particular feature will causally prevent churn.

---

## 13. Risk Segmentation

Predicted probabilities were mapped into operational risk bands:

| Risk | Probability |
|---|---|
| Low | < 0.10 |
| Medium | 0.10 to < 0.51 |
| High | 0.51 to < 0.90 |
| Critical | >= 0.90 |

Held-out test segmentation:

| Segment | Customers | Average Probability | Actual Churn Rate |
|---|---:|---:|---:|
| Low | 502 | 0.0030 | 0.20% |
| Medium | 29 | 0.2097 | 27.59% |
| High | 15 | 0.7771 | 66.67% |
| Critical | 87 | 0.9873 | 93.10% |

The segmentation layer converts continuous probabilities into a format that can be used for operational prioritization.

---

## 14. Retention Decision Support

Risk segments are mapped to retention recommendations.

| Segment | Priority | Recommendation |
|---|---|---|
| Low | Routine | Maintain standard customer engagement |
| Medium | Monitor | Proactive engagement and satisfaction check |
| High | High | Initiate targeted retention outreach |
| Critical | Immediate | Escalate for immediate retention intervention |

These are operational decision-support rules rather than experimentally validated causal treatment effects.

---

## 15. Production Model

After model selection, tuning, threshold selection, and final evaluation were complete, the production pipeline was retrained using all 3,150 labeled observations.

The production artifact contains both preprocessing and the XGBoost classifier.

The deployed model is therefore intended for inference.

Reported model performance remains based on the untouched 633-observation test set used before full-data production retraining.

---

## 16. Production Architecture

```text
                     GitHub Repository
                            |
                     GitHub Actions CI
                            |
             +--------------+--------------+
             |                             |
             v                             v
       Render Cloud              Streamlit Community Cloud
             |                             |
       Docker Container              Streamlit UI
             |
          FastAPI
             |
       ML Pipeline
             |
    Weighted XGBoost
             |
     Churn Probability
             |
      Risk Segment
             |
 Retention Recommendation
```

---

## 17. API Layer

FastAPI exposes the production model through:

- GET /
- GET /health
- POST /predict
- GET /docs

Pydantic validation is used to reject invalid customer inputs.

Application logging captures model loading, successful predictions, and internal prediction errors.

---

## 18. Frontend

The Streamlit dashboard provides:

- customer-level prediction
- churn probability
- predicted churn status
- risk segment
- retention recommendation
- portfolio-level risk analysis
- SHAP insights

The frontend communicates with the deployed FastAPI service through HTTPS.

---

## 19. Software Quality

Automated tests cover major application components including:

- preprocessing
- risk segmentation boundaries
- production model loading
- inference
- API health
- prediction responses
- invalid inputs
- missing inputs
- controlled internal prediction errors

The tests are executed with Pytest.

---

## 20. MLOps

The API is packaged inside a Docker container.

GitHub Actions provides continuous integration by:

1. checking out the repository
2. creating a clean Python environment
3. installing dependencies
4. executing the automated tests

This verifies the application outside the local development environment.

---

## 21. Cloud Deployment

The production system is deployed using two cloud services.

### Backend

Render hosts the Dockerized FastAPI service.

https://customer-churn-xai.onrender.com

### Frontend

Streamlit Community Cloud hosts the user interface.

https://sania-customer-churn-xai.streamlit.app

### API Documentation

https://customer-churn-xai.onrender.com/docs

---

## 22. Key Technical Lessons

This project demonstrates several important machine-learning engineering principles:

- strong predictive metrics do not remove the need to investigate leakage
- imbalanced classification should not be evaluated using accuracy alone
- cross-validation strategy must reflect the structure of the data
- threshold selection should be separated from final test evaluation
- model explanations should not be presented as causal conclusions
- ML predictions become more useful when translated into operational decisions
- preprocessing should be packaged with the production model
- production ML requires validation, testing, APIs, logging, containerization, CI, and deployment in addition to model training

---

## 23. Limitations

Important limitations include:

- the dataset contains only 3,150 observations
- no reliable customer identifier is available
- repeated predictive profiles required group-aware handling
- the dataset represents a particular telecom context and may not generalize to other populations
- retention recommendations are rule-based and have not been validated through randomized intervention experiments
- SHAP explanations describe model behavior rather than causal effects
- real-world deployment would require drift monitoring, retraining policies, access control, security review, and ongoing performance monitoring

---

## 24. Future Improvements

Possible extensions include:

- model and data drift monitoring
- automated retraining pipelines
- experiment tracking and model registry
- batch prediction support
- authenticated API access
- cloud artifact storage
- retention campaign outcome tracking
- treatment-effect or uplift modeling
- cost-sensitive threshold optimization using real retention economics

---

## 25. Conclusion

The project demonstrates the complete lifecycle of an applied machine-learning system: from business framing and data validation through leakage-safe modeling, explainability, decision support, production engineering, automated testing, containerization, CI, and cloud deployment.

The final result is not only a trained churn classifier but a publicly accessible customer churn intelligence application designed to translate machine-learning predictions into actionable retention priorities.
