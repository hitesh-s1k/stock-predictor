```python
# app.py
# 📈 Streamlit Stock Price Predictor — refurbished (fixes pandas bdate_range error)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from pandas.tseries.offsets import BDay
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor (30-day ahead)")
st.caption("Time-series safe Ridge model with technical features & walk-forward validation.")


# =========================
# Sidebar controls
# =========================
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input(
        "Stock Symbol",
        value="AAPL",
        help="For NSE, use suffix .NS (e.g., RELIANCE.NS, TCS.NS).",
    ).strip()
    start_date = st.date_input("History Start Date", value=pd.to_datetime("2015-01-01"))
    horizon = st.number_input(
        "Forecast Horizon (business days)",
        min_value=5,
        max_value=90,
        value=30,
        step=5,
    )
    alpha = st.slider("Ridge α (regularization)", 0.1, 10.0, 2.0, 0.1)
    run_btn = st.button("Run / Refresh")


# =========================
# Helpers
# =========================
@st.cache_data(show_spinner=False)
def load_data(sym: str, start) -> pd.DataFrame:
    df = yf.download(sym, start=str(start), auto_adjust=True, progress=False)
    # Ensure DateTimeIndex is tz-naive for plotting consistency
    if not df.empty and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    return df


def compute_features_from_prices(prices: pd.Series) -> pd.DataFrame:
    """Compute technical features & lags from a price series."""
    tmp = pd.DataFrame(index=prices.index)
    tmp["Close"] = prices

    # Returns
    tmp["return_1d"] = tmp["Close"].pct_change(1)
    tmp["return_5d"] = tmp["Close"].pct_change(5)
    tmp["return_10d"] = tmp["Close"].pct_change(10)

    # Moving averages
    tmp["sma_5"] = tmp["Close"].rolling(5).mean()
    tmp["sma_10"] = tmp["Close"].rolling(10).mean()
    tmp["sma_20"] = tmp["Close"].rolling(20).mean()

    # EMAs + MACD
    tmp["ema_12"] = tmp["Close"].ewm(span=12, adjust=False).mean()
    tmp["ema_26"] = tmp["Close"].ewm(span=26, adjust=False).mean()
    tmp["macd"] = tmp["ema_12"] - tmp["ema_26"]

    # RSI(14) (simple robust formulation)
    delta = tmp["Close"].diff()
    gain = (delta.clip(lower=0)).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    tmp["rsi_14"] = 100 - (100 / (1 + rs))

    # Lags
    for lag in [1, 2, 3, 5, 10, 20]:
        tmp[f"lag_close_{lag}"] = tmp["Close"].shift(lag)
        tmp[f"lag_ret_{lag}"] = tmp["return_1d"].shift(lag)

    return tmp


def feature_cols():
    return [
        "return_1d",
        "return_5d",
        "return_10d",
        "sma_5",
        "sma_10",
        "sma_20",
        "ema_12",
        "ema_26",
        "macd",
        "rsi_14",
        "lag_close_1",
        "lag_close_2",
        "lag_close_3",
        "lag_close_5",
        "lag_close_10",
        "lag_close_20",
        "lag_ret_1",
        "lag_ret_2",
        "lag_ret_3",
        "lag_ret_5",
        "lag_ret_10",
        "lag_ret_20",
    ]


def safe_timeseries_splits(n_samples: int, target_splits: int = 5) -> int:
    """Choose a valid number of TimeSeriesSplit folds given sample size."""
    # need at least target_splits+1 blocks; keep at least ~60 samples per fold if possible
    if n_samples < 200:
        return max(2, min(4, n_samples // 50)) or 2
    return min(target_splits, n_samples // 100) or 2


# =========================
# Main run
# =========================
if run_btn or "ran_once" not in st.session_state:
    st.session_state["ran_once"] = True

    if not symbol:
        st.warning("Please enter a stock symbol.")
        st.stop()

    data = load_data(symbol, start_date)
    if data.empty:
        st.error("No data returned. Check the symbol (include .NS for NSE) and date range.")
        st.stop()

    st.success(f"Loaded {len(data)} rows. Latest: {data.index[-1].date()}")
    st.dataframe(data.tail())

    # Basic sanity: ensure enough rows for features + horizon
    min_needed = 26 + 20 + int(horizon)  # EMA26 + SMA20 + horizon buffer
    if len(data) < min_needed:
        st.error(
            f"Not enough history to compute features and forecast {int(horizon)} days. "
            f"Need at least ~{min_needed} rows, have {len(data)}."
        )
        st.stop()

    # Feature engineering
    H = int(horizon)
    feat_all = compute_features_from_prices(data["Close"])
    feat_all["target_close_h"] = feat_all["Close"].shift(-H)
    feat_all = feat_all.dropna().copy()

    feats = feature_cols()
    # Ensure all features exist (defensive)
    missing = [c for c in feats if c not in feat_all.columns]
    if missing:
        st.error(f"Missing features: {missing}")
        st.stop()

    X = feat_all[feats].values
    y = feat_all["target_close_h"].values
    dates = feat_all.index

    # TimeSeriesSplit (no leakage)
    n_splits = max(2, min(5, safe_timeseries_splits(len(X))))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("model", Ridge(alpha=float(alpha), random_state=42)),
        ]
    )

    # Walk through splits; keep last as "test"
    last_split = None
    for tr, te in tscv.split(X):
        last_split = (tr, te)

    tr, te = last_split
    X_train, X_test = X[tr], X[te]
    y_train, y_test = y[tr], y[te]
    dates_train, dates_test = dates[tr], dates[te]

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", f"{mse:.2f}")
    c2.metric("RMSE", f"{np.sqrt(mse):.2f}")
    c3.metric("R²", f"{r2:.3f}")
    c4.metric("MAPE", f"{mape*100:.2f}%")

    # Plot test fold results
    fig1, ax1 = plt.subplots(figsize=(11, 4))
    ax1.plot(dates_test, y_test, label="Actual (t + H Close)")
    ax1.plot(dates_test, y_pred, label="Predicted (t + H Close)")
    ax1.set_title(f"{symbol} — {H}-day Ahead Close: Actual vs Predicted (Test Fold)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1)

    # Refit on all data and forecast next H business days (FIX: use BDay to avoid 'closed' arg)
    pipe.fit(X, y)

    hist = data.copy()
    last_date = hist.index[-1]

    # Generate future business dates robustly (no deprecated 'closed' argument)
    future_dates = pd.bdate_range(start=last_date + BDay(1), periods=H)

    prices_extended = hist["Close"].copy()
    future_preds = []

    for d in future_dates:
        feat_df = compute_features_from_prices(prices_extended)
        row = feat_df.iloc[-1:]
        row = row[feats].dropna(axis=0, how="any")  # ensure complete feature row
        if row.empty:
            next_price = float(prices_extended.iloc[-1])
        else:
            next_price = float(pipe.predict(row.values)[0])
        prices_extended.loc[d] = next_price
        future_preds.append(next_price)

    forecast_df = pd.DataFrame({"Predicted_Close": future_preds}, index=future_dates)

    st.subheader("Forecast")
    st.dataframe(forecast_df.round(2))

    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button(
        "Download forecast CSV",
        data=csv,
        file_name=f"{symbol.replace('.', '_')}_forecast_{H}d.csv",
        mime="text/csv",
    )

    # Plot history + forecast
    fig2, ax2 = plt.subplots(figsize=(11, 4))
    ax2.plot(hist.index[-200:], hist["Close"].tail(200), label="History (last ~200 bars)")
    ax2.plot(forecast_df.index, forecast_df["Predicted_Close"], label="Forecast (next days)")
    ax2.set_title(f"{symbol} — Forecast vs Recent History")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    st.info("Tip: Try different α values, symbols (e.g., RELIANCE.NS, TCS.NS), and horizons.")
```
