import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="The Next Move - NIFTY Predictor", layout="wide")

st.title("📈 The Next Move – AI Candle Pattern Predictor")
st.write("Predicting short-term trends for NIFTY 50 using pattern similarity from past data.")

# ----------------------------------------------------------
# Step 1: Fetch NIFTY / Stock data
# ----------------------------------------------------------
st.sidebar.header("⚙ Configuration")
period = st.sidebar.selectbox("Select Time Period", ["60d", "90d", "180d"])
interval = st.sidebar.selectbox("Select Interval", ["1h", "30m", "15m", "1d"])
ticker = st.sidebar.text_input("Enter Stock Symbol (e.g. RELIANCE.NS or ^NSEI)", value="RELIANCE.NS")

st.sidebar.info("Fetching live data from Yahoo Finance...")

data = yf.download(ticker, period=period, interval=interval)
data.dropna(inplace=True)

# Add percentage returns
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Debug Info
st.subheader("🧩 Debug Info")
st.dataframe(data.head())
st.write("Data shape:", data.shape)

# ----------------------------------------------------------
# Step 2: Prepare data for pattern matching
# ----------------------------------------------------------
window_size = 10  # number of candles to compare
current_pattern = data['Return'].iloc[-window_size:].values

# Find similar past patterns
similarities = []
for i in range(len(data) - window_size - 5):
    past_pattern = data['Return'].iloc[i:i + window_size].values
    sim = np.corrcoef(current_pattern, past_pattern)[0, 1]
    similarities.append((i, sim))

# Sort by highest similarity
top_matches = sorted(similarities, key=lambda x: x[1], reverse=True)[:5]

# ----------------------------------------------------------
# Step 3: Predict next move based on past matches
# ----------------------------------------------------------
future_moves = []
for idx, sim in top_matches:
    next_returns = data['Return'].iloc[idx + window_size: idx + window_size + 5].values
    if len(next_returns) == 5:
        future_moves.append(next_returns)

if future_moves:
    # Flatten nested series into clean numeric list
    future_moves_flat = [float(np.mean(x)) for x in future_moves]

    avg_future = np.mean(future_moves_flat)
    last_price = data['Close'].iloc[-1]

    # Predict next 5 future prices
    predicted_prices = [last_price * (1 + avg_future * (i + 1)) for i in range(5)]

    # Create a fake future index
    future_index = pd.date_range(start=data.index[-1], periods=6, freq=interval)[1:]
else:
    predicted_prices = []
    future_index = []

# ----------------------------------------------------------
# Step 4: Plot the actual and predicted data
# ----------------------------------------------------------
fig = go.Figure()

# Actual price line
fig.add_trace(go.Scatter(
    x=data.index,
    y=data['Close'],
    mode='lines',
    name=f'{ticker} Actual',
    line=dict(color='skyblue')
))

# Predicted price line
if len(predicted_prices) > 0:
    fig.add_trace(go.Scatter(
        x=future_index,
        y=predicted_prices,
        mode='lines+markers',
        name='Predicted Move',
        line=dict(color='orange', dash='dot')
    ))

fig.update_layout(
    title=f"{ticker} Price Prediction (Based on Pattern Similarity)",
    xaxis_title="Time",
    yaxis_title="Price",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------------------
# Step 5: Prediction Summary
# ----------------------------------------------------------
st.subheader("📊 Prediction Summary")
if len(top_matches) > 0:
    st.write(f"Based on {len(top_matches)} most similar patterns found in the past {period} period.")
    st.write(f"Model used pattern correlation to estimate the next few moves in {ticker}.")
else:
    st.warning("Not enough data to make a prediction. Try increasing the period or decreasing the window size.")


#streamlit run next_move_app.py
