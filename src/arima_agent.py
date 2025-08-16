# arima_agent.py — train and save ARIMA models for all tickers
# Run: python arima_agent.py

import os
import glob
import pickle
import pandas as pd
from arima_model import ARIMA, _find_price_col

# ===== PATHS =====
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Finsearch/
TRAIN_SPLIT_DIR = os.path.join(ROOT_DIR, "data", "split")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ARIMA_ORDER = (1,1,1)  # You can change this if needed

# ===== HELPERS =====
def _load_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df.sort_values("Date", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df

# ===== TRAIN AND SAVE =====
train_files = glob.glob(os.path.join(TRAIN_SPLIT_DIR, "*_train.csv"))
tickers = [os.path.basename(f).replace("_train.csv","") for f in train_files]
tickers.sort()
print(f"Found {len(tickers)} tickers for training: {tickers}")

for TICKER in tickers:
    train_path = os.path.join(TRAIN_SPLIT_DIR, f"{TICKER}_train.csv")
    df_train = _load_file(train_path)
    price_col = _find_price_col(df_train)
    train_prices = df_train[price_col].astype(float).values

    print(f"Training ARIMA({ARIMA_ORDER}) for {TICKER}...")

    try:
        model = ARIMA(train_prices, order=ARIMA_ORDER)
        fitted_model = model.fit()
        # Save model
        save_path = os.path.join(RESULTS_DIR, f"arima_{TICKER}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(fitted_model, f)
        print(f"Saved model to {save_path}")
    except Exception as e:
        print(f"Failed to train {TICKER}: {e}")

print("\n✅ All ARIMA models trained and saved!")