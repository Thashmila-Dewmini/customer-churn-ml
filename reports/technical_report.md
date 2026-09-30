# Customer Churn Prediction Using Machine Learning

## 1. Executive Summary

This project develops a supervised machine learning model to identify customers who are likely to churn.

The project was developed as a simulated freelance client assignment for portfolio practice. The stated business objective was to identify as many actual churn customers as possible so that potential retention activities could be prioritized.

Because missing a customer who actually churns was considered more important than incorrectly flagging a customer, **recall was used as the primary evaluation metric** during model tuning and threshold selection.

Three classification algorithms were evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

The final model was a Logistic Regression model with class balancing and a probability threshold of `0.40`.

On the held-out test set, the final model achieved:

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  |     70.35% |
| Precision |     46.80% |
| Recall    | **86.10%** |
| F1-Score  |     60.64% |
| AUC-ROC   |     83.55% |

The model identified **161 of 187 actual churn customers** in the test set.

---

## 2. Problem Definition

Customer churn refers to customers ending their relationship with a service provider.

The objective of this project is to build a binary classification model that predicts whether a customer will churn based on available demographic, service, contract, billing, and account information.

The simulated client objective was:

> Identify as many actual churn customers as possible.

Therefore, recall was prioritized over accuracy alone.

A high-recall model is useful for this scenario because it reduces the number of actual churn customers that are missed. However, increasing recall can also result in more false positives, meaning some customers who would not churn may also be flagged.

---

## 3. Dataset

The project uses the publicly available **IBM Telco Customer Churn dataset**.

The dataset contains:

* **7,043 rows**
* **21 columns**
* **1,869 churned customers**
* **5,174 non-churned customers**

The target variable is:

```text
Churn
```

where:

* `Yes` represents a churned customer
* `No` represents a customer who did not churn

The dataset contains demographic, service, contract, billing, and account-related features.

`customerID` was retained for customer identification but excluded from model training because it is an identifier rather than a predictive feature.

---

## 4. Data Cleaning

The dataset was inspected for:

* Missing values
* Duplicate rows
* Duplicate customer IDs
* Data type issues
* Invalid target values

### TotalCharges

The `TotalCharges` column was initially stored as a string rather than a numeric variable.

There were **11 blank values** in this column.

All 11 records had:

```text
tenure = 0
```

These blank values were converted to missing numeric values and then filled with `0`, representing zero accumulated charges for customers with zero recorded tenure.

After cleaning:

* No missing values remained.
* No duplicate rows were found.
* No duplicate customer IDs were found.
* The dataset retained its original 7,043 rows.

---

## 5. Exploratory Data Analysis

Exploratory analysis was performed to understand relationships between customer characteristics and churn.

Several notable patterns were observed.

### Tenure

Customers who churned had a lower average tenure than customers who did not churn.

### Monthly Charges

Customers who churned had higher average monthly charges than customers who did not churn.

### Contract Type

Observed churn rates differed substantially across contract types.

The observed churn rates were:

| Contract       | Observed Churn Rate |
| -------------- | ------------------: |
| Month-to-month |              42.71% |
| One year       |              11.27% |
| Two year       |               2.83% |

These are descriptive associations in the dataset and should not be interpreted as evidence that contract type directly causes churn.

---

## 6. Data Splitting

The dataset was divided into training, validation, and test sets using stratified sampling.

The split was:

| Dataset    | Records |
| ---------- | ------: |
| Training   |   5,634 |
| Validation |     704 |
| Test       |     705 |

The split ratio was:

* 80% training
* 10% validation
* 10% test

A fixed random seed of `42` was used.

The test set was kept separate until the final evaluation so that it would not influence model or threshold selection.

---

## 7. Feature Preprocessing

The input features were divided into numerical and categorical variables.

### Numerical features

* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `TotalCharges`

Numerical features were standardized using:

```text
StandardScaler
```

### Categorical features

The categorical features were encoded using:

```text
OneHotEncoder(handle_unknown="ignore")
```

The preprocessing operations were implemented inside Scikit-learn pipelines.

This ensured that preprocessing was fitted as part of the model training workflow rather than being performed separately on the complete dataset.

---

## 8. Baseline Model Comparison

Three classification models were evaluated using the validation set:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

The initial validation results were:

| Model               | Accuracy |  Precision |     Recall | F1-Score | AUC-ROC |
| ------------------- | -------: | ---------: | ---------: | -------: | ------: |
| Logistic Regression |   81.25% |     66.87% |     58.29% |   62.29% |  84.94% |
| Random Forest       |   77.84% |     57.08% | **66.84%** |   61.58% |  83.15% |
| Gradient Boosting   |   81.39% | **68.18%** |     56.15% |   61.58% |  84.77% |

These results show that the models produced different precision-recall trade-offs.

Since the project objective was focused on identifying actual churn customers, recall was given greater importance during subsequent tuning.

---

## 9. Hyperparameter Tuning

Hyperparameter optimization was performed using `GridSearchCV`.

The tuning process used:

```text
5-fold cross-validation
Scoring metric: recall
```

The test set was not used during hyperparameter tuning.

### Logistic Regression

The selected configuration was:

```text
C = 1
class_weight = balanced
max_iter = 1000
```

Cross-validation recall:

```text
80.40%
```

### Random Forest

The selected configuration was:

```text
n_estimators = 300
max_depth = 10
min_samples_split = 5
class_weight = balanced
```

Cross-validation recall:

```text
75.65%
```

### Gradient Boosting

The selected configuration was:

```text
n_estimators = 100
learning_rate = 0.1
max_depth = 3
```

Cross-validation recall:

```text
52.84%
```

The tuned Logistic Regression model provided the strongest cross-validation recall among the evaluated configurations.

---

## 10. Classification Threshold Selection

The default classification threshold of `0.50` was not assumed to be optimal for the business objective.

Instead, several probability thresholds were evaluated on the validation set.

The selected operating threshold was:

```text
0.40
```

for the tuned Logistic Regression model.

At this threshold on the validation set:

| Metric    | Validation Result |
| --------- | ----------------: |
| Precision |            46.15% |
| Recall    |            86.63% |
| F1-Score  |            60.22% |

The threshold was selected using the validation set.

The test set was not used for threshold selection.

---

## 11. Final Model Training

After selecting the model configuration and classification threshold, the final Logistic Regression pipeline was retrained using the combined training and validation data.

The final configuration was:

```text
Model: Logistic Regression
C: 1
class_weight: balanced
max_iter: 1000
Threshold: 0.40
```

The final model was saved using Joblib:

```text
models/final_churn_model.joblib
```

The saved package contains:

* The trained Scikit-learn pipeline
* The selected probability threshold

This allows the preprocessing and prediction workflow to be reused consistently.

---

## 12. Final Test Evaluation

The final model was evaluated once on the held-out test set.

The results were:

| Metric    | Test Result |
| --------- | ----------: |
| Accuracy  |      70.35% |
| Precision |      46.80% |
| Recall    |  **86.10%** |
| F1-Score  |      60.64% |
| AUC-ROC   |      83.55% |

### Confusion Matrix

```text
                 Predicted No    Predicted Yes
Actual No              335              183
Actual Yes              26              161
```

The test set contained **187 actual churn customers**.

The model correctly identified:

```text
161 / 187
```

actual churn customers.

This produced a recall of:

```text
86.10%
```

The model missed 26 actual churn customers.

At the same time, 183 non-churn customers were incorrectly flagged as potential churners.

This reflects the precision-recall trade-off created by using a lower threshold to prioritize churn detection.

---

## 13. Error Analysis

Post-hoc error analysis was performed after the final test evaluation.

The overall prediction counts were:

| Prediction Type | Count |
| --------------- | ----: |
| True Negative   |   335 |
| False Positive  |   183 |
| True Positive   |   161 |
| False Negative  |    26 |

### False Negatives

There were 26 false negatives.

These were customers who actually churned but were predicted as non-churners at the selected threshold.

Their average characteristics included:

* Average tenure: approximately 37.1 months
* Average monthly charges: approximately 62.54
* Average total charges: approximately 2,867.27
* Average predicted churn probability: approximately 0.254

Some false-negative probabilities were close to the selected threshold, while others were considerably lower. This indicates that some actual churners were difficult to distinguish from non-churners using the available features.

### False Positives

There were 183 false positives.

These were non-churn customers who were flagged as potential churners.

Their average characteristics included:

* Average tenure: approximately 21.1 months
* Average monthly charges: approximately 73.31
* Average total charges: approximately 1,810.35
* Average predicted churn probability: approximately 0.630

The relatively large number of false positives is consistent with the high-recall operating point.

---

## 14. Group-Level Error Analysis

Recall was also examined across selected customer groups.

### Contract

| Contract       | Recall |
| -------------- | -----: |
| Month-to-month | 92.77% |
| One year       | 41.18% |
| Two year       |  0.00% |

The two-year contract group had 0% recall in the test set, but the number of actual churners in that group was very small. Therefore, this result should be treated as a diagnostic observation rather than a general conclusion about all customers with two-year contracts.

### Internet Service

| Internet Service    | Recall |
| ------------------- | -----: |
| DSL                 | 74.55% |
| Fiber optic         | 95.00% |
| No internet service | 50.00% |

### Payment Method

| Payment Method            | Recall |
| ------------------------- | -----: |
| Bank transfer (automatic) | 81.82% |
| Credit card (automatic)   | 66.67% |
| Electronic check          | 94.79% |
| Mailed check              | 79.41% |

These results indicate that model performance was not uniform across all customer segments.

---

## 15. Model Interpretation

The final model uses Logistic Regression, allowing its learned coefficients to be examined.

The strongest absolute coefficients included:

| Feature                       | Coefficient |
| ----------------------------- | ----------: |
| `tenure`                      |      -1.285 |
| `Contract_Two year`           |      -0.737 |
| `InternetService_Fiber optic` |      +0.629 |
| `Contract_Month-to-month`     |      +0.607 |
| `TotalCharges`                |      +0.592 |
| `InternetService_DSL`         |      -0.570 |
| `MonthlyCharges`              |      -0.514 |

Other notable positive coefficients included:

* `PaymentMethod_Electronic check`
* `StreamingMovies_Yes`
* `StreamingTV_Yes`
* `OnlineSecurity_No`

The strongest negative coefficient was associated with `tenure`.

These coefficients represent associations learned by the fitted multivariable Logistic Regression model. They should not be interpreted as causal effects.

In particular, individual coefficient signs should not be interpreted as simple one-variable relationships because the model considers multiple features simultaneously.

---

## 16. Key Findings

The main findings are:

1. The dataset contains a substantial class imbalance, with churn representing approximately 26.5% of observations.
2. Accuracy alone does not fully represent performance for the stated business objective.
3. Recall was prioritized during model tuning and threshold selection.
4. Class weighting substantially increased churn recall for Logistic Regression.
5. Threshold selection provided an additional mechanism for controlling the precision-recall trade-off.
6. A threshold of `0.40` produced 86.10% recall on the held-out test set.
7. The final model detected 161 of 187 actual churn customers.
8. The high-recall operating point resulted in 46.80% precision.
9. Model performance varied across customer segments.
10. Tenure, contract type, internet service, and payment-related features were among the stronger signals in the fitted model.

---

## 17. Limitations

Several limitations should be considered.

### Public Dataset

The project uses a public dataset rather than real client data. Therefore, the results may not represent the behavior of customers in a specific business.

### Threshold Trade-Off

The selected threshold prioritizes recall. This results in a larger number of false positives and may increase the number of customers requiring additional retention review.

### Group-Level Sample Sizes

Some customer groups contain relatively few actual churners. Recall estimates for these groups can therefore be unstable.

### Model Interpretation

Logistic Regression coefficients describe associations within the fitted model and do not establish causality.

### No Deployment

The project does not include an API, web application, cloud deployment, or real-time prediction system.

---

## 18. Future Improvements

Potential future improvements include:

* Probability calibration
* More extensive hyperparameter optimization
* Cost-sensitive threshold selection using actual business costs
* Additional model comparison
* SHAP-based explainability
* Model performance monitoring
* Deployment through an API or web application
* Evaluation on a larger or business-specific dataset

---

## 19. Reproducibility

The project uses:

```text
random_state = 42
```

The preprocessing and model are implemented using Scikit-learn pipelines.

The final trained model and selected threshold are stored using Joblib.

Required packages are listed in:

```text
requirements.txt
```

They can be installed using:

```bash
pip install -r requirements.txt
```

The complete analysis and modeling workflow is available in:

```text
notebooks/01_customer_churn_analysis.ipynb
```

---

## 20. Conclusion

This project demonstrates a complete customer churn classification workflow, from data cleaning and exploratory analysis through model comparison, hyperparameter tuning, threshold selection, final evaluation, error analysis, interpretation, and model persistence.

The final Logistic Regression model achieved **86.10% recall** on the held-out test set using a classification threshold of `0.40`. This allowed the model to identify 161 of the 187 actual churn customers in the test set.

The result demonstrates how model selection and probability-threshold decisions should be connected to the intended business objective rather than relying on accuracy alone.
