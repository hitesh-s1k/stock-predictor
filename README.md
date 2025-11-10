# 📈 Stock Price Predictor (Streamlit)

Time-series-safe 30-day-ahead stock price predictor with feature engineering and walk-forward validation, wrapped in Streamlit.

## Features
- Chronological validation (no leakage) via `TimeSeriesSplit`
- Technical features (returns, SMAs, EMAs, MACD, RSI, lags)
- Ridge regression with scaling
- Date-aligned plots & metrics
- Recursive multi-step forecasting for the next N business days
- CSV download

## Local Setup

```bash
# 1) Create and activate a virtual env (optional but recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run
streamlit run streamlit_app.py
```

## Deploy to Streamlit Community Cloud
1. Push this folder to a public GitHub repo, e.g.
   - `https://github.com/<your-username>/stock-price-predictor-streamlit`
2. In Streamlit Cloud, click **Deploy an app**:
   - **Repository**: `<your-username>/stock-price-predictor-streamlit`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
3. Click **Deploy**.

> If you want a one-click URL to share, it will look like:
> `https://stock-price-predictor-streamlit-<hash>.streamlit.app`

## Example GitHub Repo Link
Replace with your own after pushing:
- https://github.com/your-username/stock-price-predictor-streamlit

## Notes
- For NSE symbols, remember the `.NS` suffix (e.g., `RELIANCE.NS`, `TCS.NS`).
- This model is for **educational purposes only** and not financial advice.
