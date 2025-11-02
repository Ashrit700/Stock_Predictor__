# next_move_ml_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# Optional imports (wrapped)
try:
    import xgboost as xgb
except Exception:
    xgb = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
except Exception:
    tf = None

# Optional sentiment libs
try:
    from textblob import TextBlob
    import requests
except Exception:
    TextBlob = None
    requests = None

st.set_page_config(page_title="The Next Move – ML Enhanced Predictor", layout="wide")
st.title("📈 The Next Move – ML Enhanced Predictor")
st.write("Pattern-based predictor + ML (XGBoost / LSTM) with volume, technicals and optional news sentiment.")

# Sidebar controls
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker (e.g. RELIANCE.NS)", value="RELIANCE.NS")
period = st.sidebar.selectbox("History Period", ["180d", "365d", "730d"], index=0)
interval = st.sidebar.selectbox("Data Interval", ["1d", "1h"], index=0)
model_choice = st.sidebar.selectbox("Model", ["XGBoost (tabular)", "LSTM (sequence)"])
use_news = st.sidebar.checkbox("Use News Sentiment (optional)", value=False)
news_api_key = st.sidebar.text_input("NewsAPI Key (if using)", value="")
test_size = st.sidebar.slider("Test size (%)", 10, 40, 20)
random_state = st.sidebar.number_input("Random state", min_value=0, value=42)

st.sidebar.markdown("---")
st.sidebar.write("Note: XGBoost is recommended for speed. LSTM needs TensorFlow and is slower.")

# ----- helper indicators -----
def add_technical_indicators(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Vol"] = df["Volume"].fillna(0)
    # Moving averages
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA10"]= df["Close"].rolling(10).mean()
    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    # RSI
    delta = df["Close"].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))
    # Volatility
    df["Volatility"] = df["Return"].rolling(10).std()
    # Fill
    df = df.fillna(method="bfill").fillna(method="ffill").fillna(0)
    return df

# News sentiment helper (very simple)
def fetch_news_sentiment(query, api_key, max_articles=5):
    if not requests or not TextBlob or not api_key:
        return 0.0
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "apiKey": api_key,
        "pageSize": max_articles,
        "language": "en",
        "sortBy": "relevancy"
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        js = r.json()
        articles = js.get("articles", [])
        if not articles:
            return 0.0
        scores = []
        for a in articles:
            txt = (a.get("title","") or "") + ". " + (a.get("description") or "")
            polarity = TextBlob(txt).sentiment.polarity
            scores.append(polarity)
        return float(np.mean(scores)) if scores else 0.0
    except Exception:
        return 0.0

# ----- fetch data -----
with st.spinner("Downloading data..."):
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
if df.empty:
    st.error("No data. Try different ticker/period/interval.")
    st.stop()

df = add_technical_indicators(df)

st.subheader("Data preview")
st.dataframe(df.tail(8))

# create features and target (next-day close)
df_features = df.copy()
# features we will use
feats = ["Close", "Vol", "MA5", "MA10", "MACD", "RSI14", "Volatility"]
X = df_features[feats].shift(1).dropna()   # use previous candle features to predict next close
y = df_features["Close"].loc[X.index]      # target is current Close (i.e., predict close from previous features)

# optional news sentiment added as feature
if use_news and news_api_key.strip():
    with st.spinner("Fetching news sentiment..."):
        sentiment = fetch_news_sentiment(ticker, news_api_key.strip(), max_articles=5)
    st.sidebar.write(f"News sentiment score: {sentiment:.3f}")
    X["news_sentiment"] = sentiment
else:
    X["news_sentiment"] = 0.0

# train-test split
test_frac = test_size / 100.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_frac, shuffle=False)

st.write(f"Training rows: {len(X_train)}, Test rows: {len(X_test)}")

# scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- train selected model -----
predicted_next = None
y_pred_test = None
train_time = 0
if model_choice.startswith("XGBoost"):
    if xgb is None:
        st.error("XGBoost not installed. Install with `pip install xgboost` or choose LSTM.")
    else:
        dtrain = xgb.DMatrix(X_train_scaled, label=y_train.values)
        dtest = xgb.DMatrix(X_test_scaled, label=y_test.values)
        params = {"objective":"reg:squarederror", "eval_metric":"rmse", "seed":random_state}
        t0 = time.time()
        bst = xgb.train(params, dtrain, num_boost_round=150, evals=[(dtrain,"train")], verbose_eval=False)
        train_time = time.time() - t0
        y_pred_test = bst.predict(dtest)
        # predict next (use last row of X)
        X_last = scaler.transform(X.iloc[[-1]])
        predicted_next = float(bst.predict(xgb.DMatrix(X_last))[0])

elif model_choice.startswith("LSTM"):
    if tf is None:
        st.error("TensorFlow not installed. Install with `pip install tensorflow` or choose XGBoost.")
    else:
        # prepare sequences: for LSTM we'll create sequences of length seq_len from X (sliding)
        seq_len = 10
        def create_sequences(X_arr, y_arr, seq_len):
            Xs, ys = [], []
            for i in range(len(X_arr) - seq_len):
                Xs.append(X_arr[i:i+seq_len])
                ys.append(y_arr[i+seq_len])
            return np.array(Xs), np.array(ys)
        # scale full X
        X_all_scaled = scaler.fit_transform(X)
        Xs_all, ys_all = create_sequences(X_all_scaled, y.values, seq_len)
        split_idx = int((1-test_frac) * len(Xs_all))
        Xs_train, Xs_test = Xs_all[:split_idx], Xs_all[split_idx:]
        ys_train, ys_test = ys_all[:split_idx], ys_all[split_idx:]
        # build model
        model = Sequential([
            LSTM(64, input_shape=(seq_len, Xs_all.shape[2]), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation="relu"),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mse")
        t0 = time.time()
        model.fit(Xs_train, ys_train, epochs=25, batch_size=32, validation_split=0.1, verbose=0)
        train_time = time.time() - t0
        y_pred_test = model.predict(Xs_test).flatten()
        # predict next: take last seq
        last_seq = X_all_scaled[-seq_len:].reshape(1, seq_len, X_all_scaled.shape[1])
        predicted_next = float(model.predict(last_seq)[0][0])

else:
    st.error("Unknown model choice.")
    st.stop()

# ----- evaluation -----
if y_pred_test is not None:
    # align shapes
    if model_choice.startswith("LSTM"):
        y_test_vals = ys_test
    else:
        y_test_vals = y_test.values
    mse = mean_squared_error(y_test_vals, y_pred_test)
    mae = mean_absolute_error(y_test_vals, y_pred_test)
    # directional accuracy
    direction_true = np.sign(np.diff(y_test_vals))
    direction_pred = np.sign(np.diff(y_pred_test))
    if len(direction_true) > 0 and len(direction_pred) > 0:
        dir_acc = np.mean(direction_true == direction_pred) * 100
    else:
        dir_acc = np.nan
else:
    mse = mae = dir_acc = None

# ----- display results -----
st.subheader("Prediction Results")
st.write(f"Model: **{model_choice}** — training time: {train_time:.2f}s")
st.metric("Predicted next close", f"{predicted_next:.2f}")

st.markdown("**Backtest metrics (on holdout):**")
st.write(f"MSE: {mse:.4f}  |  MAE: {mae:.4f}  |  Direction accuracy: {dir_acc:.2f}%")

# plot actual vs predicted for test set
fig = go.Figure()
if model_choice.startswith("LSTM"):
    # plot last part corresponding to test
    test_index = df.index[-len(ys_test):]
    fig.add_trace(go.Scatter(x=test_index, y=ys_test, mode="lines", name="Actual (test)"))
    fig.add_trace(go.Scatter(x=test_index, y=y_pred_test, mode="lines", name="Predicted (test)"))
else:
    test_index = X_test.index
    fig.add_trace(go.Scatter(x=test_index, y=y_test, mode="lines", name="Actual (test)"))
    fig.add_trace(go.Scatter(x=test_index, y=y_pred_test, mode="lines", name="Predicted (test)"))

# add last point prediction on top of actual series
fig.add_trace(go.Scatter(x=[df.index[-1]], y=[df["Close"].iloc[-1]], mode="markers", name="Last Close"))
fig.add_trace(go.Scatter(x=[df.index[-1] + pd.Timedelta(days=1)], y=[predicted_next], mode="markers+text", name="Predicted Next", text=["Next"]))

fig.update_layout(title=f"{ticker} — Actual vs Predicted (Test)", template="plotly_dark")
st.plotly_chart(fig, use_container_width=True)

# show summary table for next day prediction
st.subheader("Next-Day Prediction Summary")
summary_df = pd.DataFrame({
    "Metric":["Last Close","Predicted Next Close","Predicted Change (%)"],
    "Value":[round(df["Close"].iloc[-1],2), round(predicted_next,2), round((predicted_next - df["Close"].iloc[-1]) / df["Close"].iloc[-1] * 100,3)]
})
st.table(summary_df)

st.info("""
**Notes & tips**
- XGBoost: fast for tabular features. LSTM: sequence model that may capture temporal dynamics but needs more data/time.
- You can add more features (volume diff, on-balance volume, additional technical indicators) to improve accuracy.
- News sentiment is optional and noisy — use sparingly and with API limits in mind.
""")

