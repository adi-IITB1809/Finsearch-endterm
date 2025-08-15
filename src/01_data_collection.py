import os
import pandas as pd
import yfinance as yf

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
TICKERS_FILE = os.path.join(DATA_DIR, "nifty50_tickers.txt")

# Date range
START_DATE = "2010-01-01"
END_DATE = "2019-06-30"

def main():
    # Create raw folder if not exists
    os.makedirs(RAW_DIR, exist_ok=True)

    # Read tickers
    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    print(f"Downloading {len(tickers)} NIFTY50 tickers from {START_DATE} to {END_DATE}...")

    # Download all tickers
    df = yf.download(tickers, start=START_DATE, end=END_DATE, group_by='ticker', auto_adjust=False)

    # Save raw file
    raw_path = os.path.join(RAW_DIR, "nifty50_2010_2019.csv")
    df.to_csv(raw_path)
    print(f"✅ Saved raw data to {raw_path}")

if __name__ == "__main__":
    main()
