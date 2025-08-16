#!/usr/bin/env python3
"""
Model_testing.py

- Evaluate pretrained DQN (if available), LSTM benchmark (pretrained if found, else trained locally),
  and ARIMA rolling benchmark on the CSV files produced by Data_collection.py (ENDTERM/*_endterm.csv).
- Writes results to ENDTERM/compare_summary_last6w.csv and prints per-ticker + aggregated totals.
- Keeps existing TradingEnv + behaviour; only adds robustness and printing.

Run from project root (Finsearch/):
    python ENDTERM/Model_testing.py
"""
import os
import sys
import glob
import re
import warnings
import traceback
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------- detect project root and src ----------------
THIS_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(THIS_FILE), ".."))
if not os.path.isdir(PROJECT_ROOT):
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(THIS_FILE), "..", ".."))

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if not os.path.isdir(SRC_DIR):
    # try alternative relative locations
    SRC_DIR = os.path.join(os.path.dirname(THIS_FILE), "..", "src")
SRC_DIR = os.path.abspath(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

ENDTERM_DIR = os.path.join(PROJECT_ROOT, "ENDTERM")
if not os.path.isdir(ENDTERM_DIR):
    os.makedirs(ENDTERM_DIR, exist_ok=True)

# Use downloaded_data or data folder if present
DOWNLOADS_DIR = os.path.join(ENDTERM_DIR, "data")
if not os.path.isdir(DOWNLOADS_DIR):
    DOWNLOADS_DIR = ENDTERM_DIR  # fallback

RESULTS_CSV = os.path.join(ENDTERM_DIR, "compare_summary_last6w.csv")

print("[info] PROJECT_ROOT:", PROJECT_ROOT)
print("[info] SRC_DIR:", SRC_DIR)
print("[info] ENDTERM_DIR:", ENDTERM_DIR)
print("[info] DOWNLOADS_DIR:", DOWNLOADS_DIR)

# ---------------- import RL & TradingEnv ----------------
try:
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception as e:
    print("[error] stable-baselines3 import failed. Make sure stable-baselines3 is installed:", e)
    raise

# import TradingEnv from src
try:
    from env_trading import TradingEnv
except Exception as e:
    print("[error] Cannot import TradingEnv from src/env_trading.py. Please ensure file exists and src is on sys.path.")
    raise

# ---------------- utility functions (kept and improved) ----------------
def _series_from_column_like(df, col):
    s = df.loc[:, col]
    if isinstance(s, pd.Series):
        return pd.to_numeric(s, errors="coerce")
    if isinstance(s, pd.DataFrame):
        best = None
        best_non_nulls = -1
        for c in s.columns:
            coerced = pd.to_numeric(s[c], errors="coerce")
            non_nulls = int(coerced.notna().sum())
            if non_nulls > best_non_nulls:
                best_non_nulls = non_nulls
                best = coerced
        if best is None:
            first_col = s.iloc[:, 0]
            return pd.to_numeric(first_col, errors="coerce")
        return best
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return pd.Series(dtype=float)

def ensure_price_column_and_clean(fpath):
    """
    Ensure CSV has a numeric 'Close' column; coerce and clean.
    Overwrites CSV with cleaned version when possible.
    """
    try:
        df = pd.read_csv(fpath)
    except Exception as e:
        print(f"[error] Failed to read {fpath}: {e}")
        return False

    if df.empty:
        print(f"[error] {os.path.basename(fpath)} is empty.")
        return False

    # If Close exists, coerce safely
    if "Close" in df.columns:
        try:
            df["Close"] = _series_from_column_like(df, "Close")
        except Exception:
            pass

    # Find a close-like column
    if "Close" not in df.columns or df["Close"].isnull().all():
        found = None
        for c in df.columns:
            if "close" in str(c).lower() and c != "Close":
                found = c
                break
        if found:
            df["Close"] = _series_from_column_like(df, found)
            print(f"[info] Renamed '{found}' -> 'Close' in {os.path.basename(fpath)}")

    # If still no Close, pick numeric column with max variance
    if "Close" not in df.columns or df["Close"].isnull().all():
        numeric_cols = []
        for c in df.columns:
            if c == "Date":
                continue
            try:
                s = _series_from_column_like(df, c)
                if s.notna().sum() > 0:
                    numeric_cols.append((c, s))
            except Exception:
                continue
        if not numeric_cols:
            print(f"[error] No numeric-like columns found in {os.path.basename(fpath)}")
            return False
        best_col, best_series = None, None
        best_var = -1.0
        for c, s in numeric_cols:
            try:
                v = float(s.var(skipna=True))
            except Exception:
                v = -1.0
            if np.isnan(v):
                v = -1.0
            if v > best_var:
                best_var = v
                best_col = c
                best_series = s
        if best_col is None:
            print(f"[error] Could not determine a price column for {os.path.basename(fpath)}")
            return False
        print(f"[warn] No explicit 'Close' column found. Using '{best_col}' as Close for {os.path.basename(fpath)}")
        df["Close"] = best_series

    # Convert other columns to numeric where possible
    for c in df.columns:
        if c == "Date":
            continue
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            try:
                df[c] = _series_from_column_like(df, c)
            except Exception:
                df[c] = pd.Series([np.nan] * len(df))

    # Fill and drop NA
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().reset_index(drop=True)

    if "Close" not in df.columns or df["Close"].isnull().all():
        print(f"[error] After cleaning, no valid Close column in {os.path.basename(fpath)}")
        return False

    try:
        df.to_csv(fpath, index=False)
    except Exception:
        # if can't overwrite (permissions), continue with in-memory df
        pass
    return True

def compute_metrics_from_equity(equity, positions, last_slice):
    eq = np.asarray(equity, dtype=np.float64)
    pos = np.asarray(positions, dtype=int)
    sl = last_slice
    eq_window = eq[sl]
    pos_window = pos[sl]
    eq_ret = eq_window[1:] / eq_window[:-1] - 1.0 if len(eq_window) > 1 else np.array([])
    N = len(eq_ret)
    cum_return = float(eq_window[-1] - 1.0) if len(eq_window) else 0.0
    ann_return = float((eq_window[-1]) ** (252 / max(N, 1)) - 1.0) if N > 0 else 0.0
    ann_vol = float(np.std(eq_ret, ddof=1) * np.sqrt(252)) if N > 1 else 0.0
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")
    roll_max = np.maximum.accumulate(eq_window) if len(eq_window) else np.array([1.0])
    drawdown = eq_window / roll_max - 1.0 if len(eq_window) else np.array([0.0])
    max_dd = float(np.min(drawdown)) if len(drawdown) else 0.0
    total_trades = int(np.sum(np.diff(pos_window) != 0)) if len(pos_window) > 1 else 0
    in_pos = pos_window[:-1] == 1 if len(pos_window) > 1 else np.array([])
    wins = int((eq_ret[in_pos] > 0).sum()) if in_pos.size else 0
    win_rate = float(wins / in_pos.sum()) if in_pos.size and in_pos.sum() > 0 else float("nan")
    return {
        "CumReturn": cum_return,
        "AnnReturn": ann_return,
        "AnnVol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "TotalTrades": total_trades,
        "WinRate": win_rate
    }

def pad_or_truncate_obs(obs_arr, expected_dim):
    obs = np.asarray(obs_arr)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    n_env, curr_dim = obs.shape
    if expected_dim is None:
        return obs
    if curr_dim == expected_dim:
        return obs
    elif curr_dim < expected_dim:
        pad_width = expected_dim - curr_dim
        pad = np.zeros((n_env, pad_width), dtype=obs.dtype)
        return np.concatenate([obs, pad], axis=1)
    else:
        return obs[:, :expected_dim]

# ---------------- model discovery ----------------
def find_dqn_model_path(src_dir, project_root):
    # 1) parse src/rl_agent.py for a zip path
    rl = os.path.join(src_dir, "rl_agent.py")
    if os.path.isfile(rl):
        try:
            txt = open(rl, "r", encoding="utf-8", errors="ignore").read()
            # first try absolute-like string
            m = re.search(r'["\']([A-Za-z0-9_:\\/.\- ]+dqn_trading_model[^"\']*\.zip)["\']', txt, flags=re.IGNORECASE)
            if m:
                candidate = m.group(1)
                cand_abs = candidate if os.path.isabs(candidate) else os.path.abspath(os.path.join(project_root, candidate))
                if os.path.isfile(cand_abs):
                    return cand_abs
            # then any .zip in rl_agent
            m2 = re.search(r'["\']([A-Za-z0-9_:\\/.\- ]+\.zip)["\']', txt)
            if m2:
                candidate = m2.group(1)
                cand_abs = candidate if os.path.isabs(candidate) else os.path.abspath(os.path.join(project_root, candidate))
                if os.path.isfile(cand_abs):
                    return cand_abs
        except Exception:
            pass
    # 2) check results/ for expected names
    results_dir = os.path.join(project_root, "results")
    if os.path.isdir(results_dir):
        files = sorted(os.listdir(results_dir))
        # prefer exact name
        prefer = ["dqn_trading_model.zip", "dqn_trading_model-1.zip", "dqn_trading_model-v1.zip"]
        for p in prefer:
            pabs = os.path.join(results_dir, p)
            if os.path.isfile(pabs):
                return pabs
        # search for dqn*.zip
        for fname in files:
            if fname.lower().endswith(".zip") and "dqn" in fname.lower():
                return os.path.join(results_dir, fname)
        # fallback: any zip
        for fname in files:
            if fname.lower().endswith(".zip"):
                return os.path.join(results_dir, fname)
    # 3) global search for zip in project
    all_zip = glob.glob(os.path.join(project_root, "**", "*.zip"), recursive=True)
    if all_zip:
        return all_zip[0]
    return None

def find_lstm_model_path(project_root):
    """Search results/ or project for common LSTM model files (.h5, .pt, .pth, .pkl)."""
    results_dir = os.path.join(project_root, "results")
    exts = (".h5", ".hdf5", ".pt", ".pth", ".pkl")
    candidates = []
    if os.path.isdir(results_dir):
        for fname in sorted(os.listdir(results_dir)):
            if fname.lower().endswith(exts):
                candidates.append(os.path.join(results_dir, fname))
    if not candidates:
        # search entire project (first 20)
        all_files = []
        for ext in exts:
            all_files.extend(glob.glob(os.path.join(project_root, "**", f"*{ext}"), recursive=True))
        if all_files:
            candidates.extend(sorted(all_files)[:20])
    return candidates[0] if candidates else None

# ---------------- try loading models ----------------
DQN_MODEL_PATH = find_dqn_model_path(SRC_DIR, PROJECT_ROOT)
if DQN_MODEL_PATH:
    print("[info] DQN model candidate:", DQN_MODEL_PATH)
else:
    print("[warn] No DQN model found automatically. Expected at results/dqn_trading_model.zip or in src/rl_agent.py")

dqn_model = None
EXPECTED_DIM = None
if DQN_MODEL_PATH:
    try:
        dqn_model = DQN.load(DQN_MODEL_PATH)
        EXPECTED_DIM = int(dqn_model.policy.observation_space.shape[0])
        print("[info] Loaded DQN model. Expected obs dim:", EXPECTED_DIM)
    except Exception as e:
        print("[error] Failed to load DQN model from", DQN_MODEL_PATH)
        traceback.print_exc()
        dqn_model = None
        EXPECTED_DIM = None

# LSTM pre-saved (optional)
LSTM_MODEL_PATH = find_lstm_model_path(PROJECT_ROOT)
if LSTM_MODEL_PATH:
    print("[info] Found candidate LSTM model file (will attempt to load if format supported):", LSTM_MODEL_PATH)
else:
    print("[info] No pretrained LSTM model found; script will train a small local LSTM benchmark per ticker (PyTorch required).")

# ---------------- ARIMA rolling strategy ----------------
def arima_rolling_strategy(prices, order=(1,1,1), min_train=10):
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except Exception:
        raise RuntimeError("statsmodels package required for ARIMA evaluations.")
    n = len(prices)
    equity = [1.0]
    positions = []
    for t in range(0, n-1):
        if t < min_train:
            positions.append(0)
            equity.append(equity[-1])
            continue
        try:
            series = prices[:t+1]
            model = ARIMA(series, order=order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=1)[0]
            pos = 1 if forecast > prices[t] else 0
        except Exception:
            pos = 0
        ret = 0.0 if prices[t] == 0 else (prices[t+1] / prices[t] - 1.0)
        profit = pos * ret
        equity.append(equity[-1] * (1.0 + profit))
        positions.append(pos)
    if len(positions) < len(equity):
        positions.append(positions[-1] if positions else 0)
    return {"equity_curve": equity, "positions": positions, "slice": slice(max(0, len(equity)-31), len(equity))}

# ---------------- simple PyTorch LSTM benchmark ----------------
def lstm_simple_strategy(prices, seq_len=10, epochs=20, hidden=32):
    try:
        import torch
        import torch.nn as nn
    except Exception:
        raise RuntimeError("PyTorch is required for the simple LSTM benchmark. Install torch or provide a pretrained LSTM model.")

    prices = np.asarray(prices, dtype=np.float32)
    n = len(prices)
    if n < seq_len + 5:
        return {"equity_curve": [1.0] * n, "positions": [0] * n, "slice": slice(0, n)}

    split = int(n * 0.7)
    train_prices = prices[:split]

    def make_xy(series):
        X, Y = [], []
        for i in range(len(series) - seq_len):
            X.append(series[i:i+seq_len])
            Y.append(series[i+seq_len])
        return np.array(X), np.array(Y)

    X_train, y_train = make_xy(train_prices)
    if len(X_train) == 0:
        return {"equity_curve": [1.0] * n, "positions": [0] * n, "slice": slice(0, n)}

    X_train_t = torch.tensor(X_train).unsqueeze(-1)
    y_train_t = torch.tensor(y_train).unsqueeze(-1)

    class SmallLSTM(nn.Module):
        def __init__(self, input_size=1, hidden_size=hidden, num_layers=1):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.fc(out)

    model = SmallLSTM()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        optim.zero_grad()
        out = model(X_train_t)
        loss = loss_fn(out, y_train_t)
        loss.backward()
        optim.step()

    equity = [1.0]
    positions = []
    for t in range(seq_len, n-1):
        window = prices[t-seq_len+1:t+1] if t-seq_len+1 >= 0 else prices[:t+1]
        if len(window) < seq_len:
            pad = np.full(seq_len - len(window), window[0])
            window = np.concatenate([pad, window])
        x = torch.tensor(window.reshape(1, seq_len, 1)).float()
        model.eval()
        with torch.no_grad():
            pred = model(x).item()
        pos = 1 if pred > prices[t] else 0
        ret = 0.0 if prices[t] == 0 else (prices[t+1] / prices[t] - 1.0)
        profit = pos * ret
        equity.append(equity[-1] * (1.0 + profit))
        positions.append(pos)
    if len(positions) < len(equity):
        positions.append(positions[-1] if positions else 0)
    return {"equity_curve": equity, "positions": positions, "slice": slice(max(0, len(equity)-31), len(equity))}

# ---------------- prepare list of test files (copy endterm -> _test.csv) ----------------
endterm_endings = sorted(glob.glob(os.path.join(DOWNLOADS_DIR, "*_endterm.csv")) + glob.glob(os.path.join(ENDTERM_DIR, "*_endterm.csv")))
if not endterm_endings:
    print("[error] No *_endterm.csv found in ENDTERM/data or ENDTERM. Make sure Data_collection.py ran.")
    # also check for *_test.csv already
    test_files = sorted(glob.glob(os.path.join(ENDTERM_DIR, "*_test.csv")))
    if not test_files:
        print("[error] No *_test.csv found either. Exiting.")
        sys.exit(1)
else:
    # ensure we have *_test.csv copies for each endterm csv
    test_files = []
    for p in endterm_endings:
        basename = os.path.basename(p).replace("_endterm.csv", "")
        test_name = os.path.join(ENDTERM_DIR, f"{basename}_test.csv")
        if not os.path.isfile(test_name):
            try:
                df_tmp = pd.read_csv(p)
                df_tmp.to_csv(test_name, index=False)
                print(f"[info] Created test copy: {test_name} from {p}")
            except Exception as e:
                print(f"[warn] Failed to create test copy for {p}: {e}")
                continue
        test_files.append(test_name)

test_files = sorted(test_files)
if not test_files:
    print("[error] No test files available for evaluation after copying. Exiting.")
    sys.exit(1)

print(f"[info] Found {len(test_files)} test files for evaluation.")

# ---------------- evaluate each file ----------------
records = []
# accumulators for totals per model
totals = {
    "DQN": {"TotalTrades": 0, "Wins": 0, "InPosDays": 0, "CumReturns": []},
    "LSTM": {"TotalTrades": 0, "Wins": 0, "InPosDays": 0, "CumReturns": []},
    "ARIMA": {"TotalTrades": 0, "Wins": 0, "InPosDays": 0, "CumReturns": []},
}

for fpath in test_files:
    ticker = os.path.basename(fpath).replace("_test.csv", "")
    print(f"\n=== Evaluating {ticker} ===")
    # ensure price column
    ok = ensure_price_column_and_clean(fpath)
    if not ok:
        print(f"[warn] Skipping {ticker} due to failed preprocessing.")
        for model_name in ("DQN", "LSTM", "ARIMA"):
            records.append({"Ticker": ticker, "Model": model_name, "CumReturn": np.nan, "Sharpe": np.nan, "TotalTrades": np.nan, "WinRate": np.nan})
        continue

    # prepare env for DQN (TradingEnv reads <ticker>_test.csv from ENDTERM_DIR)
    def make_env():
        return TradingEnv(ticker=ticker, split="test", data_dir=ENDTERM_DIR, episode_length=100000, random_start=False)
    vec_env = DummyVecEnv([make_env])

    # reset
    reset_ret = vec_env.reset()
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        obs_raw, _ = reset_ret
    else:
        obs_raw = reset_ret

    # DQN evaluation
    dqn_equity = []
    dqn_pos = []
    if dqn_model is not None:
        obs = pad_or_truncate_obs(obs_raw, EXPECTED_DIM)
        step_count = 0
        while True:
            action, _ = dqn_model.predict(obs, deterministic=True)
            step_ret = vec_env.step(action)
            if len(step_ret) == 5:
                obs_raw, rewards, terminateds, truncateds, infos = step_ret
                dones = np.logical_or(terminateds, truncateds)
            else:
                obs_raw, rewards, dones, infos = step_ret
            obs = pad_or_truncate_obs(obs_raw, EXPECTED_DIM) if EXPECTED_DIM is not None else obs_raw
            info = infos[0] if isinstance(infos, (list, tuple, np.ndarray)) else infos
            dqn_equity.append(info.get("equity", np.nan))
            dqn_pos.append(info.get("position", np.nan))
            done = dones[0] if isinstance(dones, (list, tuple, np.ndarray)) else dones
            step_count += 1
            if bool(done) or step_count > 200_000:
                break
    else:
        print("[warn] DQN model not loaded; skipping DQN evaluation.")
    vec_env.close()

    if len(dqn_equity) > 0:
        T = len(dqn_equity)
        N = min(31, T)
        sl = slice(T - N, T)
        dqn_metrics = compute_metrics_from_equity(dqn_equity, dqn_pos, sl)
    else:
        dqn_metrics = {k: float("nan") for k in ["CumReturn", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "TotalTrades", "WinRate"]}

    # read prices for ARIMA/LSTM
    df_all = pd.read_csv(fpath)
    if "Close" not in df_all.columns:
        print(f"[warn] {ticker} missing Close column after cleaning; skipping benchmarks.")
        prices = []
    else:
        prices = df_all["Close"].astype(float).values

    # ARIMA
    try:
        arima_out = arima_rolling_strategy(prices, order=(1,1,1), min_train=8)
        arima_metrics = compute_metrics_from_equity(arima_out["equity_curve"], arima_out["positions"], arima_out["slice"])
    except Exception as e:
        print("[warn] ARIMA eval failed:", e)
        arima_metrics = {k: float("nan") for k in ["CumReturn", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "TotalTrades", "WinRate"]}

    # LSTM: try using pretrained file if available (limited format support), else local train-run
    lstm_metrics = {k: float("nan") for k in ["CumReturn", "AnnReturn", "AnnVol", "Sharpe", "MaxDD", "TotalTrades", "WinRate"]}
    if LSTM_MODEL_PATH:
        # attempt to load depending on extension
        ext = os.path.splitext(LSTM_MODEL_PATH)[1].lower()
        try:
            if ext in (".h5", ".hdf5"):
                # try Keras load and run (requires building a test runner - but formats vary)
                from tensorflow import keras
                model = keras.models.load_model(LSTM_MODEL_PATH)
                # We'll produce a simple strategy similar to lstm_simple_strategy but using the keras model
                # Prepare features: here we only have price series, so use sliding window predictions
                seq_len = 8
                if len(prices) >= seq_len + 2:
                    equity = [1.0]
                    positions = []
                    for t in range(seq_len, len(prices)-1):
                        window = prices[t-seq_len+1:t+1]
                        x = np.array(window).reshape(1, seq_len, 1)
                        p = model.predict(x, verbose=0)[0,0]
                        pos = 1 if p > prices[t] else 0
                        ret = 0.0 if prices[t] == 0 else (prices[t+1] / prices[t] - 1.0)
                        equity.append(equity[-1] * (1.0 + pos * ret))
                        positions.append(pos)
                    if len(positions) < len(equity):
                        positions.append(positions[-1] if positions else 0)
                    lstm_metrics = compute_metrics_from_equity(equity, positions, slice(max(0, len(equity)-31), len(equity)))
                else:
                    lstm_metrics = {k: float("nan") for k in lstm_metrics}
            elif ext in (".pt", ".pth"):
                # try torch load (user must implement expected architecture); fallback to local training
                try:
                    import torch
                    model = torch.load(LSTM_MODEL_PATH, map_location="cpu")
                    # we do not know architecture; attempt inference if model is a callable nn.Module with .eval()
                    if hasattr(model, "eval"):
                        seq_len = 8
                        equity = [1.0]
                        positions = []
                        for t in range(seq_len, len(prices)-1):
                            window = prices[t-seq_len+1:t+1]
                            if len(window) < seq_len:
                                pad = np.full(seq_len - len(window), window[0])
                                window = np.concatenate([pad, window])
                            x = torch.tensor(window.reshape(1, seq_len, 1)).float()
                            model.eval()
                            with torch.no_grad():
                                out = model(x)
                                try:
                                    p = float(out.squeeze().cpu().numpy())
                                except Exception:
                                    p = float(out[0].item())
                            pos = 1 if p > prices[t] else 0
                            ret = 0.0 if prices[t] == 0 else (prices[t+1] / prices[t] - 1.0)
                            equity.append(equity[-1] * (1.0 + pos * ret))
                            positions.append(pos)
                        if len(positions) < len(equity):
                            positions.append(positions[-1] if positions else 0)
                        lstm_metrics = compute_metrics_from_equity(equity, positions, slice(max(0, len(equity)-31), len(equity)))
                    else:
                        raise RuntimeError("Loaded torch object is not a model instance")
                except Exception:
                    lstm_metrics = {k: float("nan") for k in lstm_metrics}
            else:
                lstm_metrics = {k: float("nan") for k in lstm_metrics}
        except Exception as e:
            print("[warn] Failed to run pretrained LSTM model; falling back to local LSTM benchmark. Error:", e)
            lstm_metrics = {k: float("nan") for k in lstm_metrics}

    # If pretrained LSTM couldn't be used or not found, run local LSTM benchmark
    if np.isnan(lstm_metrics["CumReturn"]):
        try:
            lstm_out = lstm_simple_strategy(prices, seq_len=8, epochs=40, hidden=32)
            lstm_metrics = compute_metrics_from_equity(lstm_out["equity_curve"], lstm_out["positions"], lstm_out["slice"])
        except Exception as e:
            print("[warn] Local LSTM benchmark failed:", e)
            lstm_metrics = {k: float("nan") for k in lstm_metrics}

    # print per-ticker summary (nicely)
    def pretty(x):
        return "N/A" if (x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))) else f"{x:.6f}" if isinstance(x, float) else str(x)

    print("Ticker:", ticker)
    print("  DQN   -> CumReturn:", pretty(dqn_metrics["CumReturn"]), "Trades:", int(dqn_metrics["TotalTrades"]) if not np.isnan(dqn_metrics["TotalTrades"]) else "N/A", "WinRate:", pretty(dqn_metrics["WinRate"]))
    print("  LSTM  -> CumReturn:", pretty(lstm_metrics["CumReturn"]), "Trades:", int(lstm_metrics["TotalTrades"]) if not np.isnan(lstm_metrics["TotalTrades"]) else "N/A", "WinRate:", pretty(lstm_metrics["WinRate"]))
    print("  ARIMA -> CumReturn:", pretty(arima_metrics["CumReturn"]), "Trades:", int(arima_metrics["TotalTrades"]) if not np.isnan(arima_metrics["TotalTrades"]) else "N/A", "WinRate:", pretty(arima_metrics["WinRate"]))

    # append to records
    records.append({
        "Ticker": ticker, "Model": "DQN",
        "CumReturn": dqn_metrics["CumReturn"],
        "Sharpe": dqn_metrics["Sharpe"],
        "TotalTrades": dqn_metrics["TotalTrades"],
        "WinRate": dqn_metrics["WinRate"]
    })
    records.append({
        "Ticker": ticker, "Model": "LSTM",
        "CumReturn": lstm_metrics["CumReturn"],
        "Sharpe": lstm_metrics["Sharpe"],
        "TotalTrades": lstm_metrics["TotalTrades"],
        "WinRate": lstm_metrics["WinRate"]
    })
    records.append({
        "Ticker": ticker, "Model": "ARIMA",
        "CumReturn": arima_metrics["CumReturn"],
        "Sharpe": arima_metrics["Sharpe"],
        "TotalTrades": arima_metrics["TotalTrades"],
        "WinRate": arima_metrics["WinRate"]
    })

    # accumulate totals
    for name, m in (("DQN", dqn_metrics), ("LSTM", lstm_metrics), ("ARIMA", arima_metrics)):
        try:
            totals[name]["TotalTrades"] += int(m["TotalTrades"]) if not np.isnan(m["TotalTrades"]) else 0
        except Exception:
            pass
        try:
            # wins/inposdays estimation:
            # recreate wins/inposdays from WinRate & TotalTrades? Not directly available.
            # Instead estimate inposdays from WinRate and counts when possible:
            if not np.isnan(m["WinRate"]) and not np.isnan(m["TotalTrades"]) and m["TotalTrades"] > 0:
                # this is approximate; prefer to recompute from equity/positions if needed.
                pass
        except Exception:
            pass
        try:
            if not np.isnan(m["CumReturn"]):
                totals[name]["CumReturns"].append(float(m["CumReturn"]))
        except Exception:
            pass

# ---------------- compute aggregated totals in a sensible way ----------------
# For wins/inposdays we cannot precisely reconstruct without storing per-step pos/returns for every ticker.
# We do have TotalTrades per ticker aggregated above, and collected CumReturns for averaging.
agg_rows = []
for name in ("DQN", "LSTM", "ARIMA"):
    total_trades = totals[name]["TotalTrades"]
    avg_cum = np.mean(totals[name]["CumReturns"]) if len(totals[name]["CumReturns"]) else float("nan")
    # Wins/InPosDays/WinRate unknown exactly; set as NaN where not computable.
    agg_rows.append({
        "Ticker": "ALL",
        "Model": name,
        "CumReturn": avg_cum,
        "Sharpe": float("nan"),
        "TotalTrades": total_trades,
        "WinRate": float("nan")
    })

# append aggregated totals to records (so CSV contains them)
for r in agg_rows:
    records.append(r)

# ---------------- save results safely ----------------
df_out = pd.DataFrame(records)
try:
    df_out.to_csv(RESULTS_CSV, index=False)
    print("\n✅ Saved comparison summary to", RESULTS_CSV)
except PermissionError:
    # Try a safe save to temp then move
    tmp = RESULTS_CSV + ".tmp"
    try:
        df_out.to_csv(tmp, index=False)
        # attempt move
        try:
            os.replace(tmp, RESULTS_CSV)
            print("\n✅ Saved comparison summary to", RESULTS_CSV)
        except Exception as e:
            print("[error] Could not move temp results into place (permission?). Temp file saved at:", tmp)
    except Exception as e:
        print("[error] Failed to write results CSV due to permission error. Close Excel/OneDrive or save location elsewhere.")
        print("Exception:", e)
        print("Temp path attempted:", tmp)

# Print aggregated summary nicely
print("\n=== AGGREGATED SUMMARY (averages where applicable) ===")
for a in agg_rows:
    print(f"{a['Model']}: AvgCumReturn={a['CumReturn']:.6f}  TotalTrades={a['TotalTrades']}")

print("\nDone.")
