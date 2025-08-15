import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta
import joblib

# Paths
RAW_DATA_PATH = os.path.join("data", "raw", "nifty50_2010_2019.csv")
PROCESSED_PATH = os.path.join("data", "processed")

os.makedirs(PROCESSED_PATH, exist_ok=True)

# Load data
df = pd.read_csv(RAW_DATA_PATH, header=[0, 1], index_col=0, parse_dates=True)

# Get tickers
tickers = df.columns.levels[0]

for ticker in tickers:
    print(f"Processing {ticker}...")

    data = df[ticker].copy()

    # Technical Indicators
    data['Return_%'] = data['Close'].pct_change() * 100
    data['Vol_Change_%'] = data['Volume'].pct_change() * 100
    data['SMA_10'] = ta.sma(data['Close'], length=10)
    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['RSI_14'] = ta.rsi(data['Close'], length=14)
    macd = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data = pd.concat([data, macd], axis=1)

    # Replace inf with NaN, then drop NaNs
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    # RL state features
    features = [
        'Close', 'Return_%', 'Vol_Change_%', 'SMA_10', 'SMA_50',
        'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9'
    ]

    # Add RL placeholders
    data['Holdings'] = 0.0
    data['Cash'] = 1.0
    data['Unrealized_PnL'] = 0.0

    # Scale
    scaler = MinMaxScaler()
    data[features + ['Holdings', 'Cash', 'Unrealized_PnL']] = scaler.fit_transform(
        data[features + ['Holdings', 'Cash', 'Unrealized_PnL']]
    )

    # Save
    out_csv = os.path.join(PROCESSED_PATH, f"{ticker.replace('.', '_')}_state.csv")
    out_scaler = os.path.join(PROCESSED_PATH, f"{ticker.replace('.', '_')}_scaler.pkl")

    data.to_csv(out_csv)
    joblib.dump(scaler, out_scaler)

print("✅ Processing complete! Files saved in:", PROCESSED_PATH)
