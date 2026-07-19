# Medical Charges Prediction: Simple Linear Regression from Scratch

## 📌 Overview

This project implements simple (single-feature) linear regression to model medical insurance charges based on age, comparing three different estimation methods built from first principles alongside a `scikit-learn` benchmark. The goal is not just to fit a model, but to demonstrate how the Ordinary Least Squares closed-form solution, gradient descent, and `scikit-learn`'s implementation relate to one another — and where a single-feature model's limitations become apparent.

## 🎯 Problem Statement

Medical charges vary widely between individuals, and a key question for an insurer is how much of that variation can be explained by a single, easily available attribute — age. Understanding the strength (and limits) of an age-only model is a useful first step before layering in more predictors, and it also serves as a testbed for validating that manually implemented estimation methods (OLS via linear algebra, gradient descent) match established library results.

## 🎯 Goal

- Fit `charges ~ age` using three independent estimation approaches and confirm they converge to consistent results.
- Evaluate model fit quality (RMSE, R²) and residual behavior (normality, homoscedasticity) for both non-smokers and smokers.
- Determine whether a single-feature (age-only) model is adequate for practical use, or whether additional predictors are needed.

## 🔍 Approach

### 1. Non-smokers: comparing three estimation methods
- **Ordinary Least Squares (closed-form)** — implemented directly with `numpy` linear algebra (`(XᵀX)⁻¹Xᵀy`), without any ML library.
- **Full-batch gradient descent** — implemented from scratch with `numpy`, tested across six learning rates (1e-4 through 1e-12) over 1,000 epochs to identify the largest stable rate and visualize convergence via the error-vs-iteration curve.
- **scikit-learn `LinearRegression`** — fit as a reference implementation.

For each method: extracted the fitted coefficients, generated predictions, computed RMSE and R², and inspected residuals (scatter plot, distribution histogram, Q-Q plot) for signs of non-normality and heteroscedasticity.

### 2. Model comparison
- Overlaid all three fitted regression lines on a single scatter plot of `age` vs. `charges` to visually compare fit quality.
- Compared coefficients and error metrics across methods to confirm the closed-form OLS and `scikit-learn` solutions agree exactly, while gradient descent converges to a comparable but slightly less precise result depending on the learning rate.

### 3. Smokers: independent model
- Repeated the `scikit-learn` linear regression workflow on the smoker subgroup to test whether the age-charges relationship — and the model's practical usefulness — holds for a segment with fundamentally different cost drivers.

## 📊 Results

| Segment | Method | R² | RMSE ($) |
|---|---|---:|---:|
| Non-smokers | OLS (closed-form) | 0.3943 | 4,662.51 |
| Non-smokers | Full-batch GD (best: lr=1e-4) | 0.3811 | 4,713.25 |
| Non-smokers | scikit-learn `LinearRegression` | 0.3943 | 4,662.51 |
| Smokers | scikit-learn `LinearRegression` | 0.1356 | 10,711.00 |

**Key findings:**
- The closed-form OLS solution and `scikit-learn`'s `LinearRegression` produce identical coefficients and metrics, as expected for this convex, well-conditioned problem.
- Gradient descent diverges for learning rates ≥ 1e-2 in this problem's (unscaled) feature space; among the tested rates, **1e-4** gave the best convergence, though it slightly underperforms the closed-form solution. Rates of 1e-8 and below fail to converge meaningfully within 1,000 epochs.
- For non-smokers, medical charges increase by approximately **$267.25 per year of age**, but age alone only explains ~39% of the variance in charges — the residual distribution is non-normal, signaling that other factors are needed for a statistically robust model.
- For smokers, an age-only model performs substantially worse (R² = 0.136), confirming that smoking status interacts strongly with other cost drivers and that age alone is not a practically usable predictor for this segment.

**Conclusion:** a single-feature (age-only) model is a useful pedagogical and diagnostic baseline, but is not adequate for production use in either segment — particularly for smokers. Multivariate modeling (incorporating BMI, smoking status, region, etc.) is the clear next step.

## 🛠️ Tech Stack

- **Language:** Python
- **Core libraries:** `numpy` (from-scratch OLS and gradient descent), `pandas`, `matplotlib`
- **Modeling & metrics:** `scikit-learn` (`LinearRegression`, `mean_squared_error`, `r2_score`), `scipy.stats` (Q-Q plots via `probplot`)

## 📁 Repository Structure
├── medical_charges_prediction.ipynb   # Full analysis and modeling notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib scikit-learn scipy
jupyter notebook medical_charges_regression.ipynb
```

> Note: the dataset (`medical-charges.csv`) is not included in this repository.

## 📈 Next Steps

- Extend to a multivariate model incorporating BMI, sex, number of children, and region alongside age and smoking status.
- Apply feature scaling before gradient descent to allow larger, more efficient learning rates.
- Investigate the non-normal residual distribution further — potential log-transformation of `charges` given the right-skewed nature typical of medical cost data.
