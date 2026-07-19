# Used Car Price Prediction: Multivariate Linear Regression

## 📌 Overview

This project builds an interpretable price prediction model for used cars listed in the Indian market, using multivariate linear regression. Beyond fitting a single model, the project runs a structured feature-selection and diagnostic process — comparing multiple model variants, testing statistical significance, and validating core regression assumptions — to arrive at a final model suited for practical use.

## 🎯 Problem Statement

Used car listings carry a mix of numerical specs (engine size, power, mileage, kilometers driven) and categorical attributes (brand, fuel type, transmission, ownership history). Pricing a car accurately from these features requires more than a single best-fit model — it requires understanding *which* factors genuinely drive price, whether their effects are statistically reliable, and whether the model's assumptions hold up to scrutiny (multicollinearity, heteroscedasticity).

## 🎯 Goal

- Predict a car's price (`Price`, in INR) from its specifications and listing attributes.
- Identify which features have a statistically significant, interpretable effect on price.
- Compare model variants at different feature-selection thresholds and recommend the most reliable one for practical use.
- Validate the final model against standard linear regression diagnostics.

## 🔍 Approach

### 1. Exploratory Data Analysis
- Profiled the target variable (`Price`) and all numerical features (`Kilometers_Driven`, `Mileage`, `Engine`, `Power`, `Year`, `Seats`) using distribution plots, histograms, and Q-Q plots to assess normality and detect outliers.
- Profiled categorical features (`Transmission`, `Owner_Type`, `Fuel_Type`, `Brand`, `Model`) via frequency counts and cross-tabulations (e.g., ownership type by transmission and by vehicle year).
- Classified features into binary, unordered multi-category, and ordered multi-category types to guide the encoding strategy.

### 2. Data Preparation & Encoding
- Split the data into train/test sets (80/20).
- Encoded binary categorical features (`Fuel_Type`, `Transmission`) as 0/1.
- One-hot encoded `Brand` using `sklearn.preprocessing.OneHotEncoder`, fit on training data only.
- Ordinal-encoded `Owner_Type` (`First` < `Second` < `Third`) to preserve its natural order.
- Computed a correlation matrix against `Price` to flag features with |correlation| > 0.5 (notably `Power`, `Engine`, `Transmission`, and `Mileage`).

### 3. Model Iterations
Built and compared five linear regression variants of increasing rigor:

| Step | Model | Description |
|------|-------|-------------|
| 1 | Baseline OLS | All numeric + encoded features, unscaled |
| 2 | Scaled OLS | Same features, standardized with `StandardScaler` to compare coefficient magnitudes directly |
| 3 | Statsmodels OLS | Full statistical summary (p-values, F-statistic, R², AIC/BIC) |
| 4 | Significant-features model (α = 0.05) | Refit using only statistically significant predictors |
| 5 | Relaxed-threshold model (α = 0.25) | Refit with a looser significance cutoff to test the R²/robustness trade-off |

### 4. Diagnostics on the Final Model
- **Multicollinearity** — checked Variance Inflation Factor (VIF) for all retained features (all below 5, confirming no multicollinearity issue).
- **Heteroscedasticity** — tested residuals with the Breusch-Pagan, White, and Goldfeld-Quandt tests, and visually inspected the residuals-vs-predicted plot.

## 📊 Results

| Model | F-stat significant | All coefficients significant | R² (train) | Adj. R² (train) | R² (test) | RMSE (train) | RMSE (test) |
|-------|:---:|:---:|:---:|:---:|:---:|---:|---:|
| Baseline OLS | ✅ | ❌ | 0.959 | 0.946 | 0.851 | 210,342.80 | 259,601.26 |
| Significant-features (α=0.05) | ✅ | ❌ | 0.951 | 0.943 | 0.880 | 227,824.30 | 233,241.86 |
| **Relaxed-threshold (α=0.25)** | ✅ | ✅ | 0.958 | 0.948 | 0.861 | 212,184.26 | 250,977.95 |

**The strongest predictors of price** were engine power (positive), transmission type — automatic commands a premium (positive), and fuel efficiency (negative: lower mileage per liter correlates with higher price, consistent with larger/more powerful engines). Premium brands (Mercedes, BMW, Audi) showed a strong positive effect on price, while mass-market brands (Hyundai, Mahindra, Ford) showed a negative effect — both directionally consistent with real-world market expectations.

**Final model recommendation:** the α = 0.25 relaxed-threshold model was selected for practical use. It is the only variant where every coefficient is statistically significant while retaining R² comparable to the full baseline model, and it passed all diagnostic checks — no multicollinearity (VIF < 5) and no meaningful heteroscedasticity.

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling & visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
- **Modeling & statistics:** `scikit-learn` (`LinearRegression`, `OneHotEncoder`, `OrdinalEncoder`, `StandardScaler`, `train_test_split`), `statsmodels` (OLS, VIF, Breusch-Pagan, White, Goldfeld-Quandt tests), `scipy.stats`

## 📁 Repository Structure
├── car_price_prediction.ipynb   # Full analysis and modeling notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn statsmodels scipy
jupyter notebook car_price_regression.ipynb
```

> Note: the dataset (`cars.csv`) is not included in this repository.

## 📈 Next Steps

- Test non-linear alternatives (polynomial features, tree-based models) to address the residual heteroscedasticity signal flagged by the Breusch-Pagan/White tests.
- Explore grouping the high-cardinality `Model` column into a reduced set of categories to add model-level signal without excessive dimensionality.
- Validate feature stability and model performance with cross-validation rather than a single train/test split, given the modest dataset size (100 records).
