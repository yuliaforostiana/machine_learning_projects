# Airline Passenger Forecasting with LSTM (PyTorch)

## 📌 Overview

This project implements an LSTM-based neural network from scratch in PyTorch to forecast international airline passenger volume — a classic univariate time series benchmark. Beyond training a single model, the project investigates *why* a straightforward single-step LSTM struggles with this data, testing whether increasing model capacity alone can compensate for a modeling choice (a 1-month lookback window) that structurally prevents the model from capturing the data's clear yearly seasonality.

## 🎯 Problem Statement

Monthly airline passenger counts (1949–1960) follow a strong upward trend with clear yearly seasonality (summer peaks, winter troughs). A neural sequence model like LSTM is well-suited to time series in principle, but its ability to capture seasonal patterns depends heavily on how much historical context it's given at each prediction step. This project tests whether a minimal one-step lookback is sufficient, and whether simply scaling up the network (more hidden units) can substitute for giving it more temporal context.

## 🎯 Goal

- Build a complete PyTorch data pipeline for time series forecasting: windowing, tensor formatting, batching, and chronological (non-shuffled) train/test splitting.
- Implement and train a custom LSTM regression model (`AirModel`) from scratch using `torch.nn`.
- Evaluate forecast quality quantitatively (RMSE on train/test) and visually (predicted vs. actual passenger counts over time).
- Test whether increasing model capacity (`hidden_size`) improves forecast quality, and diagnose what's actually limiting performance.

## 🔍 Approach

### 1. Data Preparation
- Loaded the classic airline passengers dataset (144 monthly observations, Jan 1949 – Dec 1960) and converted passenger counts to a `float32` NumPy array for neural network training.
- Split chronologically into 67% train / 33% test — critical for time series, where validation must come from later periods than training, unlike a random split.
- Implemented a `create_dataset()` windowing function that transforms the raw series into (X, y) tensor pairs using a configurable `lookback` window — the number of prior time steps used to predict the next one.

### 2. Model Architecture
- Built a custom `AirModel` class (`torch.nn.Module`) wrapping a single-layer LSTM (`batch_first=True`) followed by a linear output layer, configurable via `hidden_size` and `num_layers`.
- Verified the model's forward pass on a single synthetic input before training, to confirm shapes and randomly-initialized outputs behaved as expected.

### 3. Training Pipeline
- Wrapped the training tensors in a `TensorDataset` + `DataLoader` (batch size 8, shuffled per epoch — shuffling batches during training is fine even for time series, since the windowed samples are already ordered pairs).
- Trained with the Adam optimizer and MSE loss over 2,000 epochs, tracking average epoch loss and periodically (every 100 epochs) evaluating train/test RMSE in `eval()` mode with gradients disabled.
- Visualized the training loss curve to check for convergence, and plotted actual vs. predicted passenger counts (train and test regions shown separately, correctly offset for the lookback window) to assess forecast quality visually.

### 4. Capacity Experiment
- Retrained the same architecture with `hidden_size` increased from 50 to 100, keeping all other hyperparameters fixed, to isolate the effect of model capacity alone.

## 📊 Results

| Configuration | Behavior |
|---|---|
| `hidden_size=50`, `lookback=1` | Training loss plateaus around epoch 400; RMSE remains large relative to the data scale (~100, i.e. ~100,000 passengers of error) |
| `hidden_size=100`, `lookback=1` | Nearly identical loss curve and RMSE to the 50-unit model — no meaningful improvement |

**Key findings:**
- The training loss plateaus and appears to converge, but the resulting RMSE is large relative to the passenger counts being predicted — a case where "the loss curve looks fine" does not mean "the forecasts are good."
- Visual inspection of predictions vs. actual values confirmed the model **fails to capture yearly seasonality**, which is expected: with `lookback=1`, the model only ever sees the immediately preceding month, giving it no way to learn a 12-month seasonal pattern.
- **Doubling the hidden layer size did not help** — confirming the bottleneck isn't model capacity, it's the lack of sufficient temporal context in the input window. This is a useful diagnostic lesson: when a sequence model underperforms, check whether it can even "see" the pattern you're asking it to learn before scaling up the model.

**Conclusion:** the natural next experiment is to extend `lookback` to span at least one full seasonal cycle (e.g., 12 months) rather than increasing hidden size, since the current architecture is structurally blind to yearly seasonality regardless of capacity.

## 🛠️ Tech Stack

- **Language:** Python
- **Deep learning:** PyTorch (`torch.nn`, `nn.LSTM`, `nn.Linear`, `TensorDataset`, `DataLoader`, `torch.optim.Adam`, `nn.MSELoss`)
- **Data handling & visualization:** `pandas`, `numpy`, `matplotlib`

## 📁 Repository Structure

├── lstm_airline_passengers_forecast.ipynb   # Full data pipeline, model, training, and evaluation notebook

└── README.md

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib torch
jupyter notebook lstm_airline_passengers_forecast.ipynb
```

> Note: the dataset is fetched directly from a public URL within the notebook (`airline-passengers.csv`), so no separate download is required.

## 📈 Next Steps

- Extend `lookback` to 12+ months so the model has enough temporal context to learn yearly seasonality, and re-evaluate RMSE and visual fit.
- Normalize/scale the passenger counts (e.g., min-max or log transform) before training, since neural networks typically train more stably on normalized targets than raw counts in the hundreds.
- Compare this LSTM approach against the classical time series models (Prophet, ARIMA/SARIMAX, exponential smoothing) benchmarked in the related time series forecasting project, to see how a from-scratch neural approach stacks up against domain-specific forecasting methods.
