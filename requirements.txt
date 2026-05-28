import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ================= APP CONFIG =================
st.set_page_config(page_title="AI Trading Dashboard", layout="wide")
st.title("📊 AI Trading Dashboard")

stock = st.text_input("Enter Stock Symbol", "AAPL")

# ================= DATA =================
data = yf.download(stock, start="2022-01-01")

if data.empty:
    st.error("No data found for this symbol.")
    st.stop()

# CLEAN DATA
data = data[['Close']].dropna()
close_array = data['Close'].to_numpy().flatten()

# ================= CHART =================
st.subheader("📈 Price Chart")
st.line_chart(data['Close'])

# ================= SENTIMENT =================
def get_sentiment(stock):
    try:
        url = f"https://query1.finance.yahoo.com/v1/finance/search?q={stock}"
        res = requests.get(url, timeout=5)

        analyzer = SentimentIntensityAnalyzer()
        scores = []
        headlines = []

        if res.status_code == 200:
            news = res.json().get("news", [])
        else:
            news = []

        for item in news[:10]:
            title = item.get("title", "")
            if title:
                scores.append(
                    analyzer.polarity_scores(title)["compound"]
                )
                headlines.append(title)

        if len(scores) == 0:
            return 0.0, headlines

        return float(np.mean(scores)), headlines

    except:
        return 0.0, []

sentiment_score, headlines = get_sentiment(stock)

# ================= NEWS =================
st.subheader("📰 News Headlines")

if headlines:
    for h in headlines:
        st.write("•", h)
else:
    st.write("No news available")

# ================= SENTIMENT SCORE =================
st.subheader("🧠 Sentiment Score")
st.metric("Market Sentiment", f"{sentiment_score:.3f}")

# ================= PREP DATA =================
scaler = MinMaxScaler()

scaled = scaler.fit_transform(
    close_array.reshape(-1, 1)
)

time_step = 60

X = []
y = []

for i in range(time_step, len(scaled)):
    X.append(scaled[i-time_step:i, 0])
    y.append(scaled[i, 0])

X = np.array(X)
y = np.array(y)

X = X.reshape(X.shape[0], X.shape[1], 1)

# ================= MODEL =================
model = Sequential([
    LSTM(50, return_sequences=True,
         input_shape=(60, 1)),

    LSTM(50),

    Dense(25),

    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.fit(
    X,
    y,
    epochs=2,
    batch_size=32,
    verbose=0
)

# ================= PREDICTION =================
last_60 = scaled[-60:]

X_test = np.array([last_60]).reshape(1, 60, 1)

pred = model.predict(X_test, verbose=0)

pred_price = float(
    scaler.inverse_transform(pred).flatten()[0]
)

current_price = float(close_array[-1])

# ================= SIGNAL ENGINE =================
diff = pred_price - current_price

lstm_signal = 1 if diff > 0 else -1

sent_signal = (
    1 if sentiment_score > 0.05
    else -1 if sentiment_score < -0.05
    else 0
)

final_score = (
    (0.6 * lstm_signal)
    + (0.4 * sent_signal)
)

if final_score > 0:
    signal = "📈 BUY"

elif final_score < 0:
    signal = "📉 SELL"

else:
    signal = "⏸ HOLD"

confidence = min(abs(final_score) * 100, 100)

# ================= OUTPUT =================
st.subheader("📊 Results")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Price",
    f"${current_price:.2f}"
)

col2.metric(
    "Predicted Price",
    f"${pred_price:.2f}"
)

col3.metric(
    "Signal",
    signal
)

st.subheader("🧠 Confidence")

st.progress(int(confidence))

st.write(
    f"Confidence Score: {confidence:.2f}%")
