# Customer Churn Prediction using Machine Learning

A machine learning project for identifying customers who are likely to churn. The project covers data cleaning, exploratory data analysis, preprocessing, model comparison, hyperparameter tuning, probability-threshold optimization, final evaluation, error analysis, and model interpretation.

## Project Overview

Customer churn prediction can help businesses identify customers who may leave a service and prioritize them for retention activities.

In this project, a supervised classification approach is used to predict whether a customer will churn based on demographic, service, contract, billing, and account information.

The primary objective of the simulated client scenario is to **identify as many actual churn customers as possible**. Therefore, recall was treated as the primary evaluation metric during model selection and threshold analysis.

> **Dataset note:** The original client scenario was simulated for portfolio practice. The implementation uses the publicly available **IBM** Telco Customer Churn dataset rather than a private client dataset.

---

## Objectives

- Clean and validate the customer dataset
- Perform exploratory data analysis
- Prepare numerical and categorical features
- Compare multiple classification algorithms
- Optimize models with recall as the primary tuning metric
- Evaluate different classification thresholds
- Select a final operating threshold based on the stated business objective
- Evaluate the final model on a held-out test set
- Perform post-hoc error analysis
- Interpret Logistic Regression coefficients
- Save the final trained model for future predictions

---

## Dataset

The project uses the **IBM** Telco Customer Churn dataset.

### Dataset characteristics

- **Rows:** 7,**043**
- **Columns:** 21
- **Target:** `Churn`
- **Churn = Yes:** 1,**869** customers (26.54%)
- **Churn = No:** 5,**174** customers (73.46%)

### Main feature groups

**Customer information**

- Gender
- Senior citizen status
- Partner
- Dependents

**Service information**

- Phone service
- Multiple lines
- Internet service
- Online security
- Online backup
- Device protection
- Tech support
- Streaming TV
- Streaming movies

**Contract and billing information**

- Contract
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Tenure

`customerID` is retained for customer identification but is excluded from model training.

---

## Project Structure

```
customer-churn-ml/
├── .gitignore
├── README.md
├── requirements.txt
├── data/
│   ├── processed/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── models/
│   └── final_churn_model.joblib
└── notebooks/
    └── 01_customer_churn_analysis.ipynb
```
---

## Methodology

### 1. Data Cleaning

The dataset was checked for:

- Missing values
- Duplicate rows
- Duplicate customer IDs
- Incorrect data types
- Invalid target values

`TotalCharges` was originally stored as a string. It was converted to numeric values.

There were 11 blank `TotalCharges` values. All 11 belonged to customers with `tenure = 0`, so they were treated as zero accumulated charges.

```python
df_clean["TotalCharges"] = pd.to_numeric(
    df_clean["TotalCharges"],
    errors=coerce
)
```

```python
df_clean["TotalCharges"] = df_clean["TotalCharges"].fillna(0) 
```

No duplicate rows or duplicate customer IDs were found.

---

## 2. Exploratory Data Analysis

Several relationships between customer characteristics and churn were investigated.

Some notable observations were:

- Churned customers had lower average tenure.
- Churned customers had higher average monthly charges.
- Month-to-month contracts had a higher observed churn rate than one-year and two-year contracts.
- Churn rates varied across internet service and payment-method groups.

These observations describe associations in the dataset and should not be interpreted as causal relationships.

---

## 3. Train / Validation / Test Split

The data was split using stratified sampling:

- **80% training**
- **10% validation**
- **10% test**


```
Training:       5,634 customers 
Validation:     704 customers 
Test:           705 customers 
```

A fixed random seed of `42` was used for reproducibility.

The test set was kept separate until final model evaluation.

---

## 4. Preprocessing

The preprocessing pipeline uses:

### Numerical features

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`
- `TotalCharges`

These were standardized using `StandardScaler`.

### Categorical features

Categorical variables were encoded using:

```python OneHotEncoder(handle_unknown="ignore") ```

The preprocessing steps were included inside the Scikit-learn model pipeline to ensure that transformations were learned from the appropriate training data.

---

## 5. Models Compared

Three classification algorithms were evaluated:

- Logistic Regression
- Random Forest 
- Gradient Boosting

The initial validation results were:

| Model               | Accuracy | Precision | Recall |     F1 | AUC-ROC |
| ------------------- | -------: | --------: | -----: | -----: | ------: |
| Logistic Regression |   81.25% |    66.87% | 58.29% | 62.29% |  84.94% |
| Random Forest       |   77.84% |    57.08% | 66.84% | 61.58% |  83.15% |
| Gradient Boosting   |   81.39% |    68.18% | 56.15% | 61.58% |  84.77% |

Because the primary objective was to identify actual churn customers, recall was prioritized rather than accuracy alone.

---

## 6. Hyperparameter Tuning

Hyperparameter tuning was performed using 5-fold cross-validation with **recall as the scoring metric**.

### Logistic Regression

The selected configuration was:

```
C = 1 
class_weight = balanced 
max_iter = 1000 
```

Cross-validation recall:

```80.40% ```

### Random Forest

The selected configuration was:

```
n_estimators = 300 
max_depth = 10 
min_samples_split = 5 
class_weight = balanced 
```

Cross-validation recall:

```75.65% ```

Gradient Boosting did not provide a comparable improvement in recall.

---

## 7. Classification Threshold Selection

Instead of automatically using the default probability threshold of `0.50`, multiple thresholds were evaluated using the validation set.

The selected operating threshold was:

```0.40 ```

for the tuned Logistic Regression model.

At this validation threshold:

```
Precision: 46.15% 
Recall:    86.63% 
F1-Score:  60.22% 
```

The threshold was selected because the simulated client's primary objective was to detect as many churn customers as possible.

The test set was not used to select the threshold.

---

## 8. Final Model

The final model was retrained using the combined training and validation datasets.

### Final configuration

```
Model: Logistic Regression 
C: 1 
class_weight: balanced 
max_iter: 1000 
Threshold: 0.40 
```

The final model and threshold are saved in:

```models/final_churn_model.joblib ```

The saved file contains both the trained pipeline and the selected classification threshold.

---

## 9. Final Test Results

The final model was evaluated once on the held-out test set.

| Metric    | Test Result |
| --------- | ----------: |
| Accuracy  |  **70.35%** |
| Precision |  **46.80%** |
| Recall    |  **86.10%** |
| F1-Score  |  **60.64%** |
| AUC-ROC   |  **83.55%** |

### Confusion Matrix

```
            Predicted No      Predicted Yes
Actual No       335                 183
Actual Yes       26                 161
```

The model identified:

```161 / 187 ```

actual churn customers in the test set.

Therefore, the final recall was:

```86.10% ```

The model missed 26 actual churn customers.

The relatively low precision reflects the high-recall operating point: the model also flagged a substantial number of customers who did not churn.

---

## 10. Error Analysis

Post-hoc error analysis was performed on the final test predictions.

### Overall prediction counts

```
True Negatives:   335 
False Positives:  183 
True Positives:   161 
False Negatives:   26 
```

Some group-level recall results were:

### Contract

| Contract       | Recall |
| -------------- | -----: |
| Month-to-month | 92.77% |
| One year       | 41.18% |
| Two year       |  0.00% |

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

These group-level results show that model performance is not uniform across all customer segments.

The two-year contract group produced 0% recall in the test set, but the number of actual churners in that group was very small. Therefore, this result should be treated as a diagnostic observation rather than a general conclusion about all two-year customers.

---

## 11. Model Interpretation

Because the final model is Logistic Regression, its coefficients were examined to understand which transformed features contributed most strongly to the model's predictions.

### Stronger positive coefficients

Features associated with higher predicted churn probability included:

- `InternetService_Fiber optic`
- `Contract_Month-to-month`
- `TotalCharges`
- `PaymentMethod_Electronic check`
- `StreamingMovies_Yes`

### Stronger negative coefficients

Features associated with lower predicted churn probability included:

- `tenure`
- `Contract_Two year`
- `InternetService_DSL`
- `MonthlyCharges`
- `InternetService_No`

The strongest coefficient was associated with `tenure`, with a coefficient of approximately `-1.285`.

These coefficients describe associations learned by this particular model. They should not be interpreted as evidence that an individual feature directly causes or prevents customer churn.

---

## 12. Key Findings

The main findings from the project are:

- Customer churn is an imbalanced classification problem in this dataset.
- Accuracy alone does not adequately represent the project's business objective.
- Recall was prioritized because the simulated client wanted to identify as many potential churners as possible.
- Class weighting substantially improved churn recall for Logistic Regression.
- Probability threshold selection provided an additional way to control the precision-recall trade-off.
- The final threshold of `0.40` achieved **86.10% recall** on the held-out test set.
- The model identified **161** of **187** actual churn customers.
- The high-recall configuration resulted in a relatively low precision of **46.80%**.
- Model performance varied across customer segments.
- Tenure, contract type, internet service, and payment-related features were among the stronger signals in the final Logistic Regression model.

---

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Joblib
- Jupyter Notebook

---

## Reproducibility

The project uses a fixed random seed:

``` random_state = 42 ```

The preprocessing and model are implemented using Scikit-learn pipelines.

The final trained model is stored using Joblib.

To install the required Python packages:

```pip install -r requirements.txt ```

---

## Limitations

- The project uses a public dataset rather than real client data.
- The dataset represents a specific customer population and may not generalize to other businesses.
- The selected threshold prioritizes recall and therefore produces more false positives.
- Group-level performance can be unstable when a subgroup contains few actual churners.
- Model coefficients represent associations within the fitted model and do not establish causality.
- No deployment, **API** or real-time prediction interface is included in this project.

---

## Future Improvements

Possible future improvements include:

- Probability calibration
- More extensive hyperparameter optimization
- Cost-sensitive threshold selection using actual business costs
- Additional model comparison
- Explainability using **SHAP**
- Monitoring model performance over time
- Deployment through an **API** or web application
- Evaluation on a larger or business-specific dataset

---

## Project Status

**Completed**

The project currently includes the complete machine learning workflow from dataset investigation through final model evaluation, error analysis, model interpretation and model persistence.