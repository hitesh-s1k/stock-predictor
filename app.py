import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error


st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Price Predictor (30-day ahead)")
st.caption("Time-series safe Ridge model with technical features & walk-forward validation.")

# ---------------------------
# Sidebar controls
# ---------------------------
with st.sidebar:
    st.header("Settings")
    symbol = st.text_input("Stock Symbol", value="AAPL", help="For NSE, use suffix .NS e.g., RELIANCE.NS")
    start_date = st.date_input("History Start Date", value=pd.to_datetime("2015-01-01"))
    horizon = st.number_input("Forecast Horizon (business days)", min_value=5, max_value=90, value=30, step=5)
    alpha = st.slider("Ridge α (regularization)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    run_btn = st.button("Run / Refresh")

st.markdown("---")

# ---------------------------
# Helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def load_data(sym: str, start: str):
    df = yf.download(sym, start=str(start), auto_adjust=True, progress=False)
    return df

def compute_features_from_prices(prices: pd.Series) -> pd.DataFrame:
    tmp = pd.DataFrame(index=prices.index)
    tmp["Close"] = prices

    tmp["return_1d"]  = tmp["Close"].pct_change(1)
    tmp["return_5d"]  = tmp["Close"].pct_change(5)
    tmp["return_10d"] = tmp["Close"].pct_change(10)

    tmp["sma_5"]  = tmp["Close"].rolling(5).mean()
    tmp["sma_10"] = tmp["Close"].rolling(10).mean()
    tmp["sma_20"] = tmp["Close"].rolling(20).mean()

    tmp["ema_12"] = tmp["Close"].ewm(span=12, adjust=False).mean()
    tmp["ema_26"] = tmp["Close"].ewm(span=26, adjust=False).mean()
    tmp["macd"]   = tmp["ema_12"] - tmp["ema_26"]

    tmp["rsi_14"] = (
        pd.Series(np.where(tmp["Close"].diff() > 0, tmp["Close"].diff(), 0), index=tmp.index)
          .rolling(14).mean()
        /
        (tmp["Close"].diff().abs().rolling(14).mean() + 1e-9)
    ) * 100

    for lag in [1, 2, 3, 5, 10, 20]:
        tmp[f"lag_close_{lag}"] = tmp["Close"].shift(lag)
        tmp[f"lag_ret_{lag}"]   = tmp["return_1d"].shift(lag)

    return tmp

def feature_cols():
    return [
        "return_1d","return_5d","return_10d",
        "sma_5","sma_10","sma_20","ema_12","ema_26","macd","rsi_14",
        "lag_close_1","lag_close_2","lag_close_3","lag_close_5","lag_close_10","lag_close_20",
        "lag_ret_1","lag_ret_2","lag_ret_3","lag_ret_5","lag_ret_10","lag_ret_20",
    ]

# ---------------------------
# Main run
# ---------------------------
if run_btn or "ran_once" not in st.session_state:
    st.session_state["ran_once"] = True

    data = load_data(symbol, start_date)
    if data.empty:
        st.error("No data returned. Check the symbol (include .NS for NSE).")
        st.stop()

    st.success(f"Loaded {len(data)} rows. Latest: {data.index[-1].date()}")
    st.dataframe(data.tail())

    # Feature engineering
    df = data.copy()
    HORIZON = int(horizon)
    df_feat = compute_features_from_prices(df["Close"])
    df_feat["target_close_h"] = df_feat["Close"].shift(-HORIZON)
    df_feat = df_feat.dropna().copy()

    feats = feature_cols()
    X = df_feat[feats].values
    y = df_feat["target_close_h"].values
    dates = df_feat.index

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model",  Ridge(alpha=float(alpha), random_state=42)),
    ])

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
    r2  = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MSE", f"{mse:.2f}")
    c2.metric("RMSE", f"{np.sqrt(mse):.2f}")
    c3.metric("R²", f"{r2:.3f}")
    c4.metric("MAPE", f"{mape*100:.2f}%")

    # Plot test fold
    fig1, ax1 = plt.subplots(figsize=(11,4))
    ax1.plot(dates_test, y_test, label="Actual (t+h Close)")
    ax1.plot(dates_test, y_pred, label="Predicted (t+h Close)")
    ax1.set_title(f"{symbol} — {HORIZON}-day Ahead Close: Actual vs Predicted (Test Fold)")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.legend()
    st.pyplot(fig1)

    # Refit on all data and forecast
    pipe.fit(X, y)

    hist = data.copy()
    last_date = hist.index[-1]
    future_dates = pd.bdate_range(last_date, periods=HORIZON+1, closed="right")

    prices_extended = hist["Close"].copy()
    future_preds = []

    for d in future_dates:
        feat_df = compute_features_from_prices(prices_extended)
        row = feat_df.iloc[-1:]
        row = row[feats].dropna()
        if row.empty:
            next_price = prices_extended.iloc[-1]
        else:
            next_price = float(pipe.predict(row.values)[0])
        prices_extended.loc[d] = next_price
        future_preds.append(next_price)

    forecast_df = pd.DataFrame({"Predicted_Close": future_preds}, index=future_dates)

    st.subheader("Forecast")
    st.dataframe(forecast_df.round(2))

    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button("Download forecast CSV", data=csv, file_name=f"{symbol}_forecast_{HORIZON}d.csv", mime="text/csv")

    # Plot history + forecast
    fig2, ax2 = plt.subplots(figsize=(11,4))
    ax2.plot(hist.index[-200:], hist["Close"].tail(200), label="History (last ~200 bars)")
    ax2.plot(forecast_df.index, forecast_df["Predicted_Close"], label="Forecast (next days)")
    ax2.set_title(f"{symbol} — Forecast vs Recent History")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Price")
    ax2.legend()
    st.pyplot(fig2)

    st.info("Tip: Try different α values, symbols (e.g., RELIANCE.NS, TCS.NS), and horizons.")
