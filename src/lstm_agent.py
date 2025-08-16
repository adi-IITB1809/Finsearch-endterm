# lstm_agent.py — train and save LSTM models for all tickers
# Run: python lstm_agent.py

import os
import glob
import pickle
import pandas as pd
import tensorflow as tf
from lstm_model import train_lstm_for_ticker

# ===== PATHS =====
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Finsearch/
TRAIN_SPLIT_DIR = os.path.join(ROOT_DIR, "data", "split")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== TRAINING CONFIG =====
WINDOW = 20
EPOCHS = 8
BATCH_SIZE = 64
LR = 1e-3

# ===== LOAD TICKERS =====
train_files = glob.glob(os.path.join(TRAIN_SPLIT_DIR, "*_train.csv"))
tickers = [os.path.basename(f).replace("_train.csv","") for f in train_files]
tickers.sort()
print(f"Found {len(tickers)} tickers for training: {tickers}")

# ===== TRAIN AND SAVE =====
for TICKER in tickers:
    print(f"\nTraining LSTM for {TICKER}...")
    try:
        # Train LSTM
        bundle = train_lstm_for_ticker(
            split_dir=TRAIN_SPLIT_DIR,
            ticker=TICKER,
            window=WINDOW,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR
        )
        # Save the bundle
        save_path = os.path.join(RESULTS_DIR, f"lstm_{TICKER}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"Saved LSTM model to {save_path}")
    except Exception as e:
        print(f"Failed to train LSTM for {TICKER}: {e}")

print("\n✅ All LSTM models trained and saved!")
