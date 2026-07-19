# Bank Customer Churn Prediction: A Progressive Model Benchmarking Project

## 📌 Overview

This project tackles a binary classification task — predicting whether a bank customer will churn (`Exited`) — as a progressive series of experiments, each building on the last. It starts with a baseline logistic regression, consolidates the workflow into a reusable `scikit-learn` pipeline, explores model complexity through polynomial features and regularization, moves to interpretable decision trees, benchmarks distance-based classification (kNN) with systematic hyperparameter tuning, and finishes with gradient boosting (XGBoost, LightGBM) tuned via Bayesian optimization. Each stage's model is benchmarked against the ones before it using AUROC, culminating in submissions to a Kaggle competition leaderboard.

## 🎯 Problem Statement

Predicting customer churn from account and demographic attributes (credit score, geography, age, balance, product usage, activity status) is a common but deceptively tricky classification problem: different model families make very different trade-offs between interpretability, training cost, and their tendency to overfit. Rather than picking one algorithm upfront, this project treats model selection itself as an empirical question — benchmarking linear, tree-based, distance-based, and gradient-boosted approaches under a consistent, reusable preprocessing pipeline to find what actually generalizes best.

## 🎯 Goal

- Build a reusable, well-tested preprocessing pipeline that can feed multiple model families without duplicated logic.
- Establish a logistic regression baseline and understand how feature scaling and polynomial feature expansion affect it.
- Diagnose and control overfitting, both in linear models (via regularization) and in tree-based models (via depth/leaf constraints).
- Benchmark kNN against decision trees using systematic hyperparameter search (`GridSearchCV`, `RandomizedSearchCV`).
- Push performance further with gradient boosting (XGBoost, LightGBM), tuned via Bayesian hyperparameter optimization (`hyperopt`), and compare native categorical handling against the shared encoded/scaled pipeline.
- Track model performance across every stage and submit the strongest model to the Kaggle competition leaderboard.

## 🔍 Project Stages

### 1. Logistic Regression Baseline (`01_logistic_regression.ipynb`)
- Stratified train/validation split; categorical encoding (binary mapping for `Gender`, dummy-encoded one-hot for `Geography`).
- Compared `LogisticRegression` under four numeric-scaling conditions (none, `StandardScaler`, `RobustScaler`, `MinMaxScaler`) to identify the most stable configuration.
- Validated the model against a naive majority-class baseline and serialized the final model with `joblib` for reuse.
- **Result:** standardized features gave the most stable model, reaching **AUROC 0.88** on validation.

### 2. Pipelines, Polynomial Features & Regularization (`02_pipelines_polynomial_features.ipynb`)
- Consolidated the full preprocessing + modeling workflow into a single `sklearn.Pipeline` using `ColumnTransformer`.
- Extended the pipeline with `PolynomialFeatures` at degree 2 and degree 4 to test whether added non-linearity improves churn prediction.
- Used a separate synthetic regression dataset to cleanly demonstrate overfitting at high polynomial degree (5 and 20), and tested whether `Lasso`, `Ridge`, `ElasticNet`, feature selection, or PCA could rescue an over-parameterized model.
- Tested the high-cardinality `Surname` feature for added predictive value.
- **Result:** degree-2 polynomial features generalized well (**train/val AUROC both 0.93**); degree-4 overfit (**0.95 train vs. 0.92 val**). On the regression side, no regularization strategy fully compensated for a degree-20 feature space.

### 3. Decision Trees (`03_decision_trees.ipynb`)
- Refactored preprocessing into a standalone, reusable module — `process_bank_churn.py` — with `preprocess_data()` and `preprocess_new_data()` functions shared across all subsequent notebooks.
- Trained an unconstrained baseline `DecisionTreeClassifier`, visualized its top splits, and ranked feature importances.
- Ran a systematic sweep of `max_depth` (1–20) plus a manual search over `max_depth`/`max_leaf_nodes` combinations to control overfitting.
- **Result:** the baseline tree overfit severely (100% train accuracy, 0.77 val AUROC). Depth-based tuning found a clear generalization peak at **`max_depth=5`**, achieving a validation AUROC of **0.9203**.

### 4. kNN, Cross-Validation & Hyperparameter Tuning (`04_knn_cross_validation.ipynb`)
- Benchmarked a baseline `KNeighborsClassifier` against the tuned decision tree from stage 3.
- Used `GridSearchCV` (5-fold CV) to tune kNN's `n_neighbors`, and separately tuned `DecisionTreeClassifier` via both `GridSearchCV` (3-fold) and `RandomizedSearchCV` (40 iterations over a wider hyperparameter space).
- Ran a final, expanded `RandomizedSearchCV` (60 iterations, `StratifiedKFold` cross-validation) to push decision tree performance further, then generated a Kaggle submission from the best model found.
- **Result:** tuned kNN reached **0.87 val AUROC**. Randomized search on decision trees (expanded grid + stratified CV) surpassed the manually tuned tree, becoming that stage's final submitted model.

### 5. Gradient Boosting: XGBoost & LightGBM (`05_gradient_boosting.ipynb`)
- Trained `XGBClassifier` directly on raw inputs using native categorical feature support (`enable_categorical=True`), with `scale_pos_weight` set to correct for class imbalance.
- Tuned XGBoost hyperparameters (tree depth, learning rate, subsampling, regularization terms) using Bayesian optimization via `hyperopt` (`fmin` with the Tree-structured Parzen Estimator).
- Repeated the same approach with `LightGBM` (`LGBMClassifier`), using its native categorical feature indices and comparable hyperparameter tuning.
- Re-ran both boosting models on the shared, encoded/scaled preprocessing pipeline (via `process_bank_churn.py`) to test whether standardized, pre-encoded inputs outperform each model's native categorical handling.
- **Result:** the baseline XGBoost model overfit (0.9797 train vs. 0.9285 val AUROC) and, at that stage, underperformed the previously tuned decision tree. Hyperparameter tuning with `hyperopt` improved XGBoost's quality (with some residual overfitting); LightGBM outperformed the equivalent untuned XGBoost model, and its tuned version was marginally stronger than tuned XGBoost. The **XGBoost model tuned on the shared encoded/scaled pipeline** was ultimately selected as the most robust performer and used for the final Kaggle submission.

## 📊 Results Summary

| Stage | Model | Best Validation AUROC |
|---|---|---:|
| 1 | Logistic Regression (standardized) | 0.88 |
| 2 | Logistic Regression + Polynomial features (degree 2) | 0.93 |
| 3 | Decision Tree (`max_depth=5`, manually tuned) | 0.9203 |
| 4 | kNN (`GridSearchCV`-tuned `n_neighbors`) | 0.87 |
| 4 | Decision Tree (`RandomizedSearchCV`, expanded grid + `StratifiedKFold`) | Best of stage 4 |
| 5 | XGBoost (baseline, native categorical handling) | 0.9285 (overfit) |
| 5 | XGBoost (`hyperopt`-tuned) | Improved, some overfitting remained |
| 5 | LightGBM (`hyperopt`-tuned) | Slightly ahead of tuned XGBoost |
| 5 | **XGBoost (`hyperopt`-tuned, shared encoded/scaled pipeline)** | **Final submitted model — most robust overall** |

**Key takeaway:** across every stage, the project consistently found that **more complexity ≠ better generalization on its own** — degree-4 polynomial features, degree-20 regression features, and untuned gradient boosting all showed clear overfitting signatures. The strongest, most reliable result came not from the most powerful algorithm alone, but from combining a capable model family (gradient boosting) with disciplined hyperparameter search and the project's shared, well-tested preprocessing pipeline.

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling & visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`
- **Pipelines & preprocessing:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `FunctionTransformer`, `OneHotEncoder`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`, `PolynomialFeatures`)
- **Modeling:** `scikit-learn` (`LogisticRegression`, `LinearRegression`, `Lasso`, `Ridge`, `ElasticNet`, `DecisionTreeClassifier`, `KNeighborsClassifier`), `XGBoost`, `LightGBM`
- **Complexity control & tuning:** `SelectFromModel`, `PCA`, `GridSearchCV`, `RandomizedSearchCV`, `StratifiedKFold`, `hyperopt` (Bayesian/TPE hyperparameter optimization)
- **Evaluation:** confusion matrix, ROC/AUROC, F1-score, accuracy, RMSE, R²
- **Model persistence:** `joblib`

## 📁 Repository Structure

├── 01_logistic_regression.ipynb          # Baseline logistic regression + scaling comparison

├── 02_pipelines_polynomial_features.ipynb # Pipeline consolidation, polynomial features, regularization

├── 03_decision_trees.ipynb                # Decision tree modeling, overfitting analysis, feature importance

├── 04_knn_cross_validation.ipynb          # kNN benchmarking, GridSearchCV/RandomizedSearchCV tuning

├── 05_gradient_boosting.ipynb             # XGBoost & LightGBM, Bayesian hyperparameter tuning with hyperopt

├── process_bank_churn.py                  # Shared, reusable preprocessing module

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy joblib xgboost lightgbm hyperopt
jupyter notebook 01_logistic_regression.ipynb
```

Run the notebooks in order (01 → 05); notebooks 03–05 depend on `process_bank_churn.py` being present in the same directory. LightGBM with GPU support requires a separate build step (see the notebook for the CUDA build commands) — a standard CPU install via `pip install lightgbm` is sufficient to reproduce the core results.

> Note: the datasets (`train.csv`, `test.csv`, `sample_submission.csv`) are from a Kaggle bank churn prediction competition and are not included in this repository. The synthetic `regression_data.csv` used for the regularization experiments in notebook 02 is also not included.

## 📈 Next Steps

- Ensemble the top-performing models from stages 3–5 (e.g., a blend of the tuned decision tree, XGBoost, and LightGBM) to see whether combining predictions improves robustness further.
- Apply `GridSearchCV`/`RandomizedSearchCV`-equivalent tuning consistently to logistic regression and kNN, for a fully consistent tuning methodology across all model families.
- Investigate feature engineering directly within the shared `process_bank_churn.py` module (e.g., interaction terms, frequency-encoded `Surname`) so all models — including the boosting models — benefit consistently.
