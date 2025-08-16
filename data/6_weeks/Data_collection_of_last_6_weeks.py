# Data_collection.py
"""
Download last 6 weeks of market data for a hard-coded list of NIFTY / large-cap tickers.
Saves CSVs into the configured ROOT_DIR as <TICKER>_test.csv (so your env_trading can load them with split="test").
If files already exist, they are skipped.
"""

import os
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf

# --- Configuration ---
# Use the 6_Weeks data folder directly (no ENDTERM subdirectory will be created)
ROOT_DIR = "C:\\Users\\adity\\OneDrive\\Desktop\\Finsearch\\data\\6_Weeks"

# Ensure output directory exists (we will save files directly here)
OUT_DIR = ROOT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Download window: last 6 weeks
END_DATE = dt.date.today()
START_DATE = END_DATE - dt.timedelta(weeks=6)

# === HARD-CODED TICKERS (common NIFTY / large-cap tickers). Edit this list if you want different symbols. ===
# Most entries include ".NS" so yfinance fetches NSE symbols.
HARD_CODED_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "HDFC.NS",
    "ITC.NS", "HINDUNILVR.NS", "KOTAKBANK.NS", "SBIN.NS", "AXISBANK.NS", "LT.NS",
    "BHARTIARTL.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "BAJAJ-AUTO.NS", "MARUTI.NS",
    "TITAN.NS", "ULTRACEMCO.NS", "ONGC.NS", "POWERGRID.NS", "NTPC.NS", "SUNPHARMA.NS",
    "DIVISLAB.NS", "WIPRO.NS", "HCLTECH.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "JSWSTEEL.NS", "TATASTEEL.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS", "GRASIM.NS",
    "CIPLA.NS", "BRITANNIA.NS", "EICHERMOT.NS", "HINDALCO.NS", "BAJAJFINSV.NS",
    "TATAMOTORS.NS", "SRF.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "M&M.NS", "INDUSINDBK.NS", "TVSMOTOR.NS", "NESTLEIND.NS", "COLPAL.NS", "GODREJCP.NS",
    "VEDL.NS", "TATACONSUM.NS", "ADANIGREEN.NS", "PIDILITIND.NS", "BANDHANBNK.NS",
    "HINDPETRO.NS", "LAURUSLABS.NS", "SIEMENS.NS", "TORNTPHARM.NS", "TATAELXSI.NS",
    "ACC.NS", "LICI.NS"
]

# You can reduce or expand the list above. MAX_TICKERS will limit actual attempts.
MAX_TICKERS = 100

def download_and_save_ticker(ticker: str, start_date: dt.date, end_date: dt.date, out_dir: str):
    """Download OHLCV using yfinance and save to CSV as <ticker>_test.csv."""
    safe_ticker = ticker.replace("/", "_").replace("\\", "_")
    out_path = os.path.join(out_dir, f"{safe_ticker}_train.csv")
    if os.path.exists(out_path):
        print(f"[skip] {safe_ticker} already exists at {out_path}")
        return out_path
    try:
        print(f"[download] {safe_ticker}: {start_date} -> {end_date}")
        # yfinance uses end-exclusive date, so add one day to include END_DATE
        df = yf.download(safe_ticker, start=str(start_date), end=str(end_date + dt.timedelta(days=1)), progress=False, auto_adjust=False)
        if df is None or df.empty:
            print(f"[warn] No data returned for {safe_ticker}.")
            return None
        df = df.reset_index()
        # Ensure numeric conversion where appropriate
        for c in df.columns:
            if c != "Date":
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().reset_index(drop=True)
        df.to_csv(out_path, index=False)
        print(f"[saved] {out_path} (rows={len(df)})")
        return out_path
    except Exception as e:
        print(f"[error] Failed to download {safe_ticker}: {e}")
        return None

def main():
    print("Configured output directory:", OUT_DIR)
    print(f"Downloading period: {START_DATE} -> {END_DATE} (approx. 6 weeks)")

    tickers = HARD_CODED_TICKERS[:MAX_TICKERS]
    downloaded_files = []
    for t in tickers:
        out = download_and_save_ticker(t, START_DATE, END_DATE, OUT_DIR)
        if out:
            downloaded_files.append(out)

    if not downloaded_files:
        # final fallback: download NIFTY index ^NSEI if nothing succeeded
        print("[warning] No tickers downloaded from the hard-coded list. Falling back to ^NSEI index.")
        out = download_and_save_ticker("^NSEI", START_DATE, END_DATE, OUT_DIR)
        if out:
            downloaded_files.append(out)

    print(f"[done] Downloaded {len(downloaded_files)} files into {OUT_DIR}")

if __name__ == "__main__":
    main()
