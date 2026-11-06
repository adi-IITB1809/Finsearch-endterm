import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROC_DIR = os.path.join(DATA_DIR, "processed")
SPLIT_DIR = os.path.join(DATA_DIR, "split")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
TICKERS_FILE = os.path.join(DATA_DIR, "nifty50_tickers.txt")
for _d in (RAW_DIR, PROC_DIR, SPLIT_DIR, RESULTS_DIR, MODELS_DIR):
    os.makedirs(_d, exist_ok=True)

# Data window (per the problem statement: Nifty 50, Jan-2010 to Jun-2019)
START_DATE = "2010-01-01"
END_DATE_EXCLUSIVE = "2019-07-01"

# Chronological split
TRAIN_END = "2016-12-31"
VAL_END = "2018-06-30"      # test = everything after this
MIN_TRAIN_ROWS = 500        # drop stocks without enough history
MIN_TEST_ROWS = 150

FEATURES = ["ret_1d", "ret_5d", "ret_10d", "vol_10d", "vol_chg",
            "sma10_gap", "sma50_gap", "rsi", "macd", "macd_hist"]

HISTORY_LEN = 10     # DQN observation window
LSTM_WINDOW = 20
COST = 0.001         # 0.1% per side transaction cost
SEED = 42
