# Customer Churn Prediction

This project focuses on predicting customer churn using a bank's customer dataset. It implements machine learning models in Python and compares them against automated modeling results from RapidMiner Studio. The pipeline includes data preprocessing, visualization, class balancing (SMOTE), and model evaluation using various metrics.

---

## Repository Contents

- `py-code.ipynb` – Main Jupyter notebook for data analysis, preprocessing, model training, and evaluation.
- `churn_modelling.csv` – Dataset used for the churn analysis.
- `RapidMiner.rmp` – RapidMiner AutoModel workflow for benchmarking results with no-code ML tools.

---

## Problem Statement

Customer churn refers to the percentage of customers who stop using a company's product or service over a specific period. Retaining customers is more cost-effective than acquiring new ones. Hence, identifying customers likely to churn can help a business proactively improve customer satisfaction and retention.

---

## Data Summary

The dataset contains **10,000 rows** and the following key features:

- **Customer Profile**: Credit Score, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
- **Target**: `Exited` (0 = Stayed, 1 = Churned)

---

## Exploratory Data Analysis (EDA)

EDA revealed:

- Churn rate is around **20%**
- Geography and Gender show significant correlation with churn
- Age and number of products also contribute meaningfully to predictions

Visualizations include:

- Class distributions
- Churn by gender and geography
- Age histograms
- Correlation heatmaps

---

## Preprocessing Steps

- Removed non-predictive columns: `RowNumber`, `CustomerId`, `Surname`
- Categorical Encoding:
  - Binary encoding for `Gender`
  - One-hot encoding for `Geography`
- Handled class imbalance using **SMOTE**
- Scaled numerical features with `StandardScaler`

---

## Models Implemented (Python)

| Model                 | Accuracy | F1-Score (Churn) | Recall (Churn) |
|----------------------|----------|------------------|----------------|
| Logistic Regression  | 72.6%    | 0.60             | 0.67           |
| Random Forest        | 84.2%    | 0.61             | 0.61           |
| Decision Tree        | 75.4%    | 0.55             | 0.53           |
| XGBoost              | 84.7%    | 0.64             | 0.57           |

**Best performing model:** `XGBoost` in Python (84.7% accuracy)

Evaluation techniques used:
- Confusion Matrices
- Classification Reports
- ROC Curves
- Feature Importance (XGBoost)

---

## RapidMiner AutoModel Results

The same dataset was used in RapidMiner's visual AutoModel environment. Performance was slightly better due to optimized pipelines:

| Model             | Accuracy | Precision | F1-score | Recall |
|------------------|----------|-----------|----------|--------|
| Logistic Regression | 81.1%  | 0.81      | 0.89     | 0.99   |
| Decision Tree       | 81.8%  | 0.85      | 0.89     | 0.93   |
| Random Forest       | 82.0%  | 0.85      | 0.89     | 0.92   |
| XGBoost             | 86.6%  | 0.87      | 0.92     | 0.97   |

---

## How to Use

### Clone the Repository

```bash
git clone https://github.com/athulrj02/Churn-Modelling.git
cd Churn-Modelling


