# Customer Segmentation: Imbalanced Multiclass Classification

## 📌 Overview

This project builds a multiclass classification pipeline to predict customer segment (A/B/C/D) for a company expanding into new markets, using logistic regression under two multiclass strategies (One-vs-Rest and One-vs-One). Since segment classes are imbalanced, the project systematically compares multiple resampling techniques from `imbalanced-learn` — including categorical-aware variants — to determine which combination of resampling and multiclass strategy produces the most reliable classifier.

## 🎯 Problem Statement

A company wants to expand into new markets using a customer segmentation strategy (segments A–D) that worked well in its existing market, and needs to predict the correct segment for ~2,600 new potential customers based on demographic and behavioral attributes. Two challenges complicate this: the dataset mixes numeric and categorical features with missing values requiring targeted imputation, and the segment classes are imbalanced — a standard classifier risks underperforming on minority segments unless the class imbalance is explicitly addressed.

## 🎯 Goal

- Clean and encode a mixed-type (numeric + categorical) customer dataset, handling missing values with feature-appropriate imputation strategies.
- Apply and compare multiple resampling strategies (SMOTE, SMOTE-Tomek, SMOTENC, SMOTEN) to address class imbalance, including techniques that correctly handle categorical features.
- Train multiclass logistic regression models under both One-vs-Rest and One-vs-One strategies, across all resampling variants.
- Identify which combination of multiclass strategy and resampling technique yields the best, most balanced classification performance — with particular attention to minority-class recall.

## 🔍 Approach

### 1. Data Cleaning & Encoding
- Split data into train/validation sets (80/20) with stratification on the target (`Segmentation`).
- Handled missing values feature-by-feature, matching the imputation strategy to each feature's distribution and semantics:
  - `Family_Size` and categorical binary features (`Ever_Married`, `Graduated`) — imputed with mode, given a low missing-value rate.
  - `Work_Experience` — imputed with median, appropriate for its skewed numeric distribution.
  - `Profession` — imputed with an explicit `'Unknown'` category rather than mode, to avoid fabricating unrealistic profiles for genuinely ambiguous cases.
  - `Var_1` (an anonymized categorical feature) — imputed with mode.
- Encoded features according to their type: binary mapping for `Gender`/`Ever_Married`/`Graduated`, `LabelEncoder` for nominal features (`Profession`, `Var_1`), and `OrdinalEncoder` for the naturally ordered `Spending_Score` (Low < Average < High).
- Scaled numeric features with `MinMaxScaler`.

### 2. Class Imbalance: Resampling Strategies
Applied and compared four resampling approaches on the training set only:
- **SMOTE** — applied to numeric features only (base SMOTE cannot correctly handle categorical data).
- **SMOTE-Tomek** — combined over- and under-sampling (SMOTE oversampling + Tomek link removal), also on numeric features.
- **SMOTENC** — SMOTE variant designed for mixed numeric/categorical data, using explicit categorical feature indices.
- **SMOTEN** — SMOTE variant designed for purely categorical data, applied to the categorical feature subset.

### 3. Multiclass Modeling & Comparison
- Trained `LogisticRegression` under both **One-vs-Rest** (`OneVsRestClassifier`) and **One-vs-One** (`OneVsOneClassifier`) strategies, for each of: original (unresampled) data, SMOTE, SMOTE-Tomek, SMOTENC, and SMOTEN.
- Evaluated every model with `classification_report`, using **weighted F1-score** as the primary metric (balancing overall performance across all classes) and **per-class recall** as a secondary metric, with particular attention to minority-class recall.

## 📊 Results

| Strategy | Best-performing resampling | Weakest resampling |
|---|---|---|
| One-vs-Rest | **SMOTENC** — best weighted F1-score and best per-class recall | SMOTE-Tomek — weakest weighted F1-score |
| One-vs-One | **SMOTENC** (tied with original/unresampled data on weighted F1-score) — best minority-class recall | SMOTE-Tomek — weakest weighted F1-score |

**Key findings:**
- **SMOTENC consistently outperformed the other resampling methods** under both multiclass strategies, since it correctly synthesizes new samples for mixed numeric/categorical data rather than treating categorical features as continuous or ignoring them.
- SMOTE and SMOTE-Tomek showed no meaningful difference from each other in this dataset — likely because Tomek link removal had little effect on the actual class balance here.
- Under One-vs-One, the unresampled (original) data performed comparably to SMOTENC on weighted F1-score, but SMOTENC still delivered noticeably better recall on the minority class, making it the preferred choice when minority-segment identification matters.

**Conclusion:** for this mixed-type, imbalanced multiclass problem, **SMOTENC combined with either multiclass strategy** was the most reliable choice, meaningfully improving minority-class recall without sacrificing overall weighted performance — validating that resampling methods must match the data's feature types to be effective.

## 🛠️ Tech Stack

- **Language:** Python
- **Data handling & visualization:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy.stats`
- **Preprocessing:** `scikit-learn` (`MinMaxScaler`, `OrdinalEncoder`, `LabelEncoder`, `train_test_split`)
- **Imbalanced-data handling:** `imbalanced-learn` (`SMOTE`, `SMOTENC`, `SMOTEN`, `SMOTETomek`)
- **Modeling:** `scikit-learn` (`LogisticRegression`, `OneVsRestClassifier`, `OneVsOneClassifier`)
- **Evaluation:** `classification_report` (precision, recall, F1-score per class and weighted average)

## 📁 Repository Structure

├── customer_segmentation_classification.ipynb   # Full preprocessing, resampling, and modeling notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy imbalanced-learn
jupyter notebook customer_segmentation_classification.ipynb
```

> Note: the dataset (`customer_segmentation_train.csv`) is not included in this repository.

## 📈 Next Steps

- Extend the comparison to non-linear multiclass models (e.g., Random Forest, gradient boosting with native multiclass support) to see whether they benefit from SMOTENC resampling as much as logistic regression did.
- Tune classification thresholds or use cost-sensitive learning (`class_weight='balanced'`) as an alternative or complement to resampling.
- Apply the final SMOTENC-based pipeline to the actual new-market prospect data (2,627 customers) referenced in the original business problem, once available.
