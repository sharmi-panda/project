import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
# import warnings

# warnings.filterwarnings("ignore")

# ==============================
# 🌈 Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="📈 Indian Stock Price Forecast",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ==============================
# 🧩 Load Model and Preprocessors
# ==============================
@st.cache_resource
def load_resources():
    model = joblib.load("model/stock_model.pkl")
    scaler = joblib.load("model/stock_scaler.pkl")
    le = joblib.load("model/stock_label_encoder.pkl")
    return model, scaler, le

model, scaler, le = load_resources()

# ==============================
# 🗃️ Load Dataset
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("data/indian_stocks.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

data = load_data()

# ==============================
# 🧮 Feature Engineering
# ==============================
def create_features(df):
    for lag in [1, 2, 3]:
        df[f"Close_lag_{lag}"] = df["Close"].shift(lag)
    for window in [3, 5, 10]:
        df[f"Close_roll_mean_{window}"] = df["Close"].rolling(window).mean()
    df = df.dropna().copy()
    df["Ticker_enc"] = le.transform(df["Ticker"])
    return df

def forecast_next_days(df, days=5):
    df = create_features(df)
    feature_cols = [
        "Ticker_enc", "Open", "High", "Low", "Volume",
        "Close_lag_1", "Close_lag_2", "Close_lag_3",
        "Close_roll_mean_3", "Close_roll_mean_5", "Close_roll_mean_10"
    ]
    last_row = df.iloc[-1:].copy()
    preds = []

    for _ in range(days):
        X = last_row[feature_cols]
        X_scaled = scaler.transform(X)
        pred_close = model.predict(X_scaled)[0]
        preds.append(pred_close)

        # simulate next day
        new_row = last_row.copy()
        new_row["Close"] = pred_close
        for lag in [3, 2, 1]:
            new_row[f"Close_lag_{lag}"] = last_row[f"Close_lag_{lag-1}"] if lag > 1 else pred_close
        for window in [3, 5, 10]:
            values = np.append(df["Close"].values[-(window-1):], pred_close)
            new_row[f"Close_roll_mean_{window}"] = np.mean(values)
        df = pd.concat([df, new_row], ignore_index=True)
        last_row = new_row.copy()

    future_dates = pd.date_range(df["Date"].iloc[-1], periods=days+1, freq="D")[1:]
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds})
    return forecast_df

# ==============================
# 🎛️ Sidebar Controls
# ==============================
st.sidebar.markdown("## ⚙️ Settings")
company = st.sidebar.selectbox("🏢 Select Company", sorted(data["Ticker"].unique()))
forecast_days = st.sidebar.slider("🗓 Forecast Days", 1, 10, 5)
theme = st.sidebar.radio("🎨 Theme", ["Light", "Dark"], index=0)
st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit & Scikit-learn")

# ==============================
# 🌐 Main Layout
# ==============================
if theme == "Dark":
    plt.style.use("dark_background")
else:
    plt.style.use("seaborn-v0_8")

st.title("📊 Indian Stock Price Prediction Dashboard")
st.markdown(f"### 🔮 {forecast_days}-Day Forecast for **{company}**")

tabs = st.tabs(["📈 Forecast Chart", "📊 Data Table", "ℹ️ Model Info"])

# ==============================
# 📈 Run Forecast
# ==============================
df_company = data[data["Ticker"] == company].copy().sort_values("Date")
forecast_df = forecast_next_days(df_company, forecast_days)

recent_df = df_company.tail(60)[["Date", "Close"]].copy()
merged_df = pd.concat([
    recent_df,
    pd.DataFrame({"Date": forecast_df["Date"], "Close": forecast_df["Predicted_Close"]})
]).reset_index(drop=True)

# Convert dates properly
merged_df["Date"] = pd.to_datetime(merged_df["Date"])
forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

# ==============================
# 🪄 Summary Cards
# ==============================
col1, col2, col3 = st.columns(3)
col1.metric("📅 Forecast Days", forecast_days)
col2.metric("💰 Last Close Price (₹)", round(df_company["Close"].iloc[-1], 2))
col3.metric("📈 Predicted Next Close (₹)", round(forecast_df["Predicted_Close"].iloc[0], 2))

# ==============================
# 📊 Tab 1: Chart
# ==============================
with tabs[0]:
    st.markdown("#### 📉 Actual vs Predicted Price Movement")
    fig, ax = plt.subplots(figsize=(6, 3))  # 👈 smaller chart dimensions
    ax.plot(merged_df["Date"], merged_df["Close"], label="Actual Close", linewidth=1.8)
    ax.plot(
        forecast_df["Date"],
        forecast_df["Predicted_Close"],
        label="Predicted Close",
        linestyle="--",
        marker="o",
        markersize=4
    )
    ax.set_xlabel("Date", fontsize=9)
    ax.set_ylabel("Price (₹)", fontsize=9)
    ax.set_title(f"{company} - {forecast_days}-Day Forecast", fontsize=11)
    ax.legend(fontsize=8)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    st.pyplot(fig, use_container_width=False) 

# ==============================
# 📊 Tab 2: Data Table
# ==============================
with tabs[1]:
    st.markdown("#### 🧾 Forecast Data")
    st.dataframe(forecast_df.style.highlight_max(axis=0, color="lightgreen"))

# ==============================
# 📘 Tab 3: Model Info
# ==============================
with tabs[2]:
    st.markdown("### ⚙️ Model Information")
    st.markdown("""
    **Model Used:** Random Forest Regressor  
    **Trained On:** Historical Indian stock data  
    **Features Used:**  
    - Open, High, Low, Volume  
    - 3-day, 5-day, and 10-day rolling averages  
    - Lag values for previous 3 closes  
    """)
    st.success("✅ Model and preprocessing pipeline loaded successfully.")

# ==============================
# 💾 Save Forecast File
# ==============================
# forecast_df.to_csv(f"{company}_forecast_{forecast_days}day.csv", index=False)
# st.download_button(
#     label="💾 Download Forecast CSV",
#     data=forecast_df.to_csv(index=False).encode("utf-8"),
#     file_name=f"{company}_forecast_{forecast_days}day.csv",
#     mime="text/csv"
# )
