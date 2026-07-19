# Retail Demand Forecasting: Time Series Modeling with Darts

## 📌 Overview

This project explores time series forecasting techniques for retail sales data, using the [Store Item Demand Forecasting](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview) dataset — 5 years of daily sales across 50 items in 10 stores. The core objective is to forecast **daily sales one month ahead** and to identify, through systematic experimentation, which modeling approach delivers the most reliable forecasts.

To keep the analysis focused and reproducible, the deep-dive modeling work concentrates on a single item/store combination, with the resulting approach designed to generalize across the full 50-item × 10-store product catalog.

## 🎯 Problem Statement

Retailers need accurate short-term demand forecasts to optimize inventory, avoid stockouts, and reduce overstock costs. Sales data is noisy, seasonal at multiple frequencies (weekly and yearly), and varies significantly in volatility across items and stores — a naive one-size-fits-all forecasting approach is unlikely to perform well across the full assortment.

## 🎯 Goal

- Understand the structure, seasonality, and volatility patterns in multi-store, multi-item retail sales data.
- Benchmark a range of forecasting approaches — from naive baselines to statistical and machine learning models — on a representative time series.
- Identify the best-performing model and validate its robustness with backtesting.
- Propose a scalable modeling strategy for forecasting all 500 item-store combinations.

## 🔍 Approach

1. **Data preparation** — loaded and indexed sales data by date; converted to `darts.TimeSeries` objects for consistent time series handling.
2. **Exploratory analysis**:
   - Visualized average sales dynamics with min–max ranges across stores per item to detect shared seasonal patterns.
   - Built a coefficient-of-variation heatmap (item × store) to flag unstable, hard-to-forecast segments.
   - Segmented items by demand level vs. volatility to surface high-risk forecasting targets.
3. **Single series deep-dive** (item 1, store 1):
   - Additive seasonal decomposition (trend / seasonality / residuals) via `statsmodels`.
   - Train/validation split anchored at a fixed cutoff date.
   - Partial autocorrelation (PACF) analysis to identify significant lags (1, 7, 14, 21 days).
   - Seasonality detection via `darts.utils.statistics.check_seasonality`.
4. **Model benchmarking** — trained and evaluated a wide range of forecasting models using MAPE:
   - **Naive baselines**: Seasonal, Drift, and weighted combinations of both (including a linear-regression-optimized blend of weekly/yearly seasonal components).
   - **XGBoost**: base model and an enhanced version with engineered date-based past covariates (day of week, cyclical day-of-year encoding).
   - **Exponential Smoothing**: base and seasonally-configured variants.
   - **ARIMA / SARIMAX**: manually tuned and enriched with calendar covariates (weekday dummies, cyclical yearly features).
   - **AutoARIMA**: automated order search, with and without seasonal configuration.
   - **Prophet**: base model and a multi-seasonality variant (additive weekly + multiplicative yearly components).
   - **RNN (LSTM)**: deep learning approach via `darts.models.RNNModel`.
5. **Backtesting** — validated the top-performing model with a rolling 30-day-ahead backtest over a full year, using `historical_forecasts`.

## 📊 Results

| Model | MAPE |
|-------|------|
| Naive Seasonal | 38.17% |
| Naive Seasonal & Drift | 39.91% |
| Naive Seasonal & Drift (two seasonal components) | 136.71% |
| Naive Seasonal & Drift (weighted seasonal components) | 26.28% |
| XGBoost (basic) | 28.82% |
| XGBoost (with past covariates) | 24.48% |
| Exponential Smoothing (basic) | 39.01% |
| Exponential Smoothing (tuned) | 39.00% |
| ARIMA | 39.03% |
| SARIMAX (with covariates) | 25.30% |
| AutoARIMA | 39.49% |
| AutoARIMA (seasonal) | 35.10% |
| Prophet (basic) | 23.80% |
| **Prophet (multi-seasonality)** | **22.97%** |
| RNN (LSTM) | 27.67% |

**Prophet with explicit weekly (additive) and yearly (multiplicative) seasonality components achieved the best accuracy**, best capturing both trend and seasonal structure in the data. A rolling 30-day backtest over the final year confirmed the model reliably tracks the overall trend, though it loses some accuracy on intra-month fluctuations compared to a single validation-period evaluation.

### Key takeaways
- Naive models with properly weighted seasonal components can be a surprisingly strong and cheap baseline.
- Adding calendar-based covariates (day-of-week, cyclical yearly encoding) meaningfully improves both tree-based (XGBoost) and statistical (SARIMAX) models.
- Prophet's explicit multi-seasonality decomposition outperformed both classical statistical models (ARIMA family) and a deep learning baseline (LSTM) on this series.

## 🚀 Scaling to the Full Catalog

Two strategies were considered for forecasting all 50 items × 10 stores:
1. **Per-series modeling** — train an individual Prophet model per item-store combination. Best suited given the observed heterogeneity in volatility across items/stores, at the cost of training and maintaining ~500 models.
2. **Global modeling** — train a single XGBoost model across all series with item/store as categorical features. Faster to train and scale than per-series Prophet, and a strong secondary candidate based on benchmark results.

## 🛠️ Tech Stack

- **Language:** Python
- **Core libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `plotly`
- **Time series & forecasting:** `darts` (TimeSeries, NaiveSeasonal, NaiveDrift, XGBModel, ExponentialSmoothing, ARIMA, AutoARIMA, Prophet, RNNModel), `statsmodels` (seasonal decomposition)
- **ML utilities:** `scikit-learn`, `scipy`

## 📁 Repository Structure
├── `store_time_series_analysis.ipynb`   # Full analysis and modeling notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn plotly statsmodels darts scikit-learn scipy
jupyter notebook time_series_analysis.ipynb
```

> Note: the dataset (`train.csv`) is from the [Store Item Demand Forecasting Challenge](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/overview) on Kaggle and is not included in this repository.

## 📈 Next Steps

- Extend the winning approach (Prophet multi-seasonality) across the full item-store catalog and evaluate aggregate MAPE/SMAPE.
- Benchmark a global XGBoost model with item/store embeddings against the per-series Prophet approach on cost/accuracy trade-offs.
- Incorporate external regressors (promotions, holidays, pricing) where available.
