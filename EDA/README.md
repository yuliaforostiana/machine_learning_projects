# Credit Risk EDA: Home Credit Default Risk

## 📌 Overview

This project is an end-to-end **Exploratory Data Analysis (EDA)** of a loan application dataset (~307K records, 122 features), aimed at identifying the factors that drive loan repayment difficulties. The analysis follows a structured data science workflow: data cleaning, missing value treatment, outlier detection, and univariate/bivariate/multivariate analysis against the target variable.

## 🎯 Problem Statement

Financial institutions need to assess the risk of a client defaulting on a loan **before** approval. Poor risk assessment leads to either:
- Approving loans to high-risk clients (financial losses), or
- Rejecting capable clients (lost business opportunities).

The raw application data contains missing values, inconsistent encodings, data-type mismatches, and outliers — all of which must be resolved before the data can support reliable risk modeling.

## 🎯 Goal

- Clean and prepare a large, messy real-world credit application dataset for downstream machine learning.
- Identify which client attributes (demographic, financial, behavioral) are associated with **payment difficulties** (`TARGET = 1`) vs. **on-time payments** (`TARGET = 0`).
- Surface actionable client segments that require deeper risk review before loan approval.

## 🔍 Approach

1. **Data structure audit** — reviewed shape, dtypes, and column composition (16 categorical / 106 numerical features).
2. **Data type correction** — fixed inconsistent types (e.g., ID columns, binary flag columns downcast to `int8` for memory efficiency).
3. **Missing value strategy**:
   - Dropped columns with >40% missing values (49 columns removed).
   - For columns with <40% missing, applied targeted imputation per column (mode, median, or explicit "Unknown"/"NA" categories), guided by distribution shape, skew, and outlier presence — not blanket rules.
4. **Data quality fixes** — corrected invalid categorical codes (e.g., `XNA` gender values), converted negative day-based columns (`DAYS_BIRTH`, `DAYS_EMPLOYED`, etc.) into human-readable positive day/year counts, and handled a known sentinel anomaly (`DAYS_EMPLOYED = 365243`, representing retired/unemployed clients).
5. **Outlier detection** — applied the IQR method across key numeric features (income, credit amount, family size, annuity) with visual diagnostics (boxplots, distribution plots).
6. **Feature engineering** — derived interpretable features such as `YEARS_BIRTH`, `YEARS_EMPLOYED`, age buckets, and price/income binning.
7. **Target imbalance check** — confirmed a ~92% / 8% class imbalance between on-time and defaulting clients.
8. **Univariate, bivariate, and multivariate analysis** — built reusable analysis functions to systematically compare each categorical and numerical feature against `TARGET`, including correlation heatmaps and cross-segment breakdowns (e.g., income × education × gender).

## 📊 Key Findings

Client segments associated with a higher share of payment difficulties:
- Income above ~900K, or in the 75K–100K range (in local currency units)
- Loan / goods amounts in the 250K–650K range
- Loan annuity above 100K, notably among pensioners
- Clients younger than 45
- Single/unmarried clients or those in civil marriages
- Clients with secondary (complete or incomplete) education
- Working clients — particularly laborers and drivers
- Clients employed for less than ~2,200 days prior to application
- Sales department employees applying for revolving loans

These findings point to concrete features and segments worth prioritizing in a downstream credit scoring model.

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`
- **Environment:** Jupyter Notebook (originally developed in Google Colab)

## 📁 Repository Structure
├── credit_eda.ipynb     # Full EDA notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn jupyter
jupyter notebook credit_eda.ipynb
```

> Note: the dataset (`application_data.csv`) is not included in this repository and must be provided separately.

## 📈 Next Steps

- Feature selection based on correlation and signal strength identified in this EDA
- Encoding of categorical variables and handling of class imbalance (e.g., SMOTE, class weighting)
- Building and evaluating classification models (Logistic Regression, Gradient Boosting) for credit default prediction
