#  Stock Price Prediction — Random Forest Model

##  Overview

This project predicts the **next 5 days of stock prices** for selected Indian companies (such as **HDFCBANK**, **ICICIBANK**, and **BHARTIARTL**) using a **Random Forest Regressor** trained on historical market data.

It includes:

* Automated **data collection** using *yfinance*
* **Data preprocessing** and **feature engineering**
* **Model training**, **evaluation**, and **saving**
* **5-day future price forecasting**


## Requirements

Install all dependencies before running the notebook:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn yfinance joblib
```

---

* streamlit
* pandas
* numpy
* scikit-learn
* joblib
* yfinance
* matplotlib
* seaborn


##  Workflow

### 1️. Data Collection

Historical data is collected from Yahoo Finance using *yfinance*:

```python
import yfinance as yf
data = yf.download(["HDFCBANK.NS", "ICICIBANK.NS", "BHARTIARTL.NS"],
                   start="2018-01-01", end="2025-01-01")
```

The collected data is saved as:

```
indian_stocks.csv
```

---

### 2️. Feature Engineering

Features created:

* `Close_lag_1`, `Close_lag_2`, `Close_lag_3`
* `Close_roll_mean_3`, `Close_roll_mean_5`, `Close_roll_mean_10`
* `Open`, `High`, `Low`, `Volume`
* Encoded `Ticker` values

---

### 3️. Model Training

The **Random Forest Regressor** is trained as follows:

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    max_depth=12
)
model.fit(X_train_scaled, y_train)
print(" Model training complete.")
```

The trained model, scaler, and label encoder are saved as:

```
model/stock_model.pkl
model/stock_scaler.pkl
model/stock_label_encoder.pkl
```

---

### 4️. Model Evaluation

Performance metrics:

* **MAE:** 21.53
* **RMSE:** 87.75
* **R²:** 0.9916

```python
r2 = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {r2 * 100:.2f}%")
```

---

### 5️. Forecasting

Predicts the next **5 days** of closing prices for each stock:

Example output (saved in `indian_stock_forecast_5day.csv`):

| Date       | Ticker   | Predicted_Close |
| ---------- | -------- | --------------- |
| 2025-01-01 | HDFCBANK | 874.57          |
| 2025-01-02 | HDFCBANK | 876.68          |
| 2025-01-03 | HDFCBANK | 873.66          |

---

###  Visualization

Visualize the actual vs predicted stock prices:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10,5))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted', alpha=0.7)
plt.legend()
plt.title(" Actual vs Predicted Stock Prices")
plt.xlabel("Samples")
plt.ylabel("Close Price")
plt.show()
```

---

##  Future Improvements

* Add **technical indicators** (RSI, MACD, Bollinger Bands)
* Try **LSTM/GRU models** for time-series forecasting
* Deploy a **Streamlit dashboard** for real-time prediction visualization

