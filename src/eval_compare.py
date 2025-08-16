# eval_compare.py — fixed evaluator (vectorized + obs shape safe)
# Run: python eval_compare.py

# ===== SILENCE WARNINGS & LOGS =====
import os
import warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
np.seterr(divide='ignore', invalid='ignore')

# ===== IMPORTS =====
import glob
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env_trading import TradingEnv
from lstm_model import train_lstm_for_ticker, lstm_strategy_equity
from arima_model import arima_strategy_equity

# ========= PATHS =========
ROOT_DIR    = os.path.dirname(os.path.dirname(__file__))  # Finsearch/
SPLIT_DIR   = os.path.join(ROOT_DIR, "data", "split")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_trading_model.zip")
DETERMINISTIC = True           # Greedy for evaluation
LAST_N_DAYS   = 30             # ~6 weeks of trading days

# ========= HELPERS =========
def _find_price_col(df: pd.DataFrame) -> str:
    for c in ["Close", "Adj Close", "Adj_Close", "close", "adj_close"]:
        if c in df.columns:
            return c
    raise ValueError("Price col not found.")

def _load_split(split_dir: str, ticker: str, split: str) -> pd.DataFrame:
    p = os.path.join(split_dir, f"{ticker}_{split}.csv")
    if not os.path.isfile(p):
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def compute_metrics_from_equity(equity, positions, last_slice):
    eq = np.asarray(equity)
    pos = np.asarray(positions)
    sl = last_slice

    eq_window = eq[sl]
    pos_window = pos[sl]

    eq_ret = eq_window[1:] / eq_window[:-1] - 1.0
    N = len(eq_ret)

    cum_return = eq_window[-1] - 1.0
    ann_return = (eq_window[-1]) ** (252 / max(N, 1)) - 1.0 if N > 0 else 0.0
    ann_vol = (np.std(eq_ret, ddof=1) * np.sqrt(252)) if N > 1 else 0.0
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan

    roll_max = np.maximum.accumulate(eq_window)
    drawdown = eq_window / roll_max - 1.0
    max_dd = np.min(drawdown) if len(drawdown) else 0.0

    total_trades = int(np.sum(np.diff(pos_window) != 0)) if len(pos_window) > 1 else 0
    in_pos = pos_window[:-1] == 1 if len(pos_window) > 1 else np.array([])
    wins = (eq_ret[in_pos] > 0).sum() if in_pos.size else 0
    win_rate = (wins / in_pos.sum()) if in_pos.size and in_pos.sum() > 0 else np.nan

    return {
        "CumReturn": float(cum_return),
        "AnnReturn": float(ann_return),
        "AnnVol": float(ann_vol),
        "Sharpe": float(sharpe),
        "MaxDD": float(max_dd),
        "TotalTrades": int(total_trades),
        "WinRate": float(win_rate) if not np.isnan(win_rate) else np.nan
    }

def save_dqn_plots(ticker, price_curve, positions, equity_curve):
    plt.figure(figsize=(10, 4))
    plt.plot(equity_curve)
    plt.title(f"{ticker} – Equity Curve (DQN, last window shown in metrics)")
    plt.xlabel("Step")
    plt.ylabel("Equity (× initial)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{ticker}_DQN_equity.png"), dpi=150)
    plt.close()

    pos = np.array(positions)
    price = np.array(price_curve)
    pos_change = np.diff(pos)
    buy_steps = np.where(pos_change == 1)[0] + 1
    sell_steps = np.where(pos_change == -1)[0] + 1

    plt.figure(figsize=(12, 5))
    plt.plot(price, label="Price")
    if len(buy_steps) > 0:
        plt.scatter(buy_steps, price[buy_steps], marker="^", s=60, label="Buy")
    if len(sell_steps) > 0:
        plt.scatter(sell_steps, price[sell_steps], marker="v", s=60, label="Sell")
    plt.legend()
    plt.title(f"{ticker} – DQN Price with Trades")
    plt.xlabel("Step")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{ticker}_DQN_price_trades.png"), dpi=150)
    plt.close()

# Pad/truncate obs to expected model dimension
def pad_or_truncate_obs(obs_arr, expected_dim):
    obs = np.asarray(obs_arr)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    n_env, curr_dim = obs.shape
    if curr_dim == expected_dim:
        return obs
    elif curr_dim < expected_dim:
        pad_width = expected_dim - curr_dim
        pad = np.zeros((n_env, pad_width), dtype=obs.dtype)
        # zeros appended on the right (features absent)
        return np.concatenate([obs, pad], axis=1)
    else:  # curr_dim > expected_dim
        return obs[:, :expected_dim]

# ========= MAIN =========
if __name__ == "__main__":
    print("Using model:", MODEL_PATH)
    model = DQN.load(MODEL_PATH)

    # expected dim from the loaded model/policy
    obs_shape = model.policy.observation_space.shape
    if len(obs_shape) != 1:
        raise RuntimeError(f"Unexpected policy observation shape: {obs_shape}")
    EXPECTED_DIM = int(obs_shape[0])
    print(f"Model expects observation vector of length: {EXPECTED_DIM}")

    test_files = glob.glob(os.path.join(SPLIT_DIR, "*_test.csv"))
    tickers = [os.path.basename(f).replace("_test.csv", "") for f in test_files]
    tickers.sort()
    print(f"Found {len(tickers)} tickers for evaluation: {tickers}")

    records = []
    total_trades_all = 0
    wins_all = 0
    inpos_all = 0

    for TICKER in tickers:
        print(f"\n=== {TICKER} ===")
        # Build a vectorized env so the model receives (n_env, obs_dim)
        def make_env():
            return TradingEnv(
                ticker=TICKER,
                split="test",
                data_dir=SPLIT_DIR,
                episode_length=10_000,
                random_start=False
            )
        vec_env = DummyVecEnv([make_env])

        # Reset (handle different reset signatures)
        reset_ret = vec_env.reset()
        if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
            obs_raw, _ = reset_ret
        else:
            obs_raw = reset_ret

        obs = pad_or_truncate_obs(obs_raw, EXPECTED_DIM)

        dqn_equity = []
        dqn_price = []
        dqn_pos = []

        step_count = 0
        while True:
            action, _ = model.predict(obs, deterministic=DETERMINISTIC)
            step_ret = vec_env.step(action)

            # Normalize step returns (support 5-tuple and 4-tuple)
            if len(step_ret) == 5:
                obs_raw, rewards, terminateds, truncateds, infos = step_ret
                dones = np.logical_or(terminateds, truncateds)
            elif len(step_ret) == 4:
                obs_raw, rewards, dones, infos = step_ret
            else:
                raise RuntimeError(f"Unexpected return from vec_env.step(): got {len(step_ret)} items")

            obs = pad_or_truncate_obs(obs_raw, EXPECTED_DIM)

            # index first (only) env
            reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
            done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
            info = infos[0] if isinstance(infos, (list, np.ndarray)) else infos

            dqn_equity.append(info.get("equity", np.nan))
            dqn_price.append(info.get("price", np.nan))
            dqn_pos.append(info.get("position", np.nan))

            step_count += 1
            if bool(done):
                break
            if step_count > 200_000:
                warnings.warn("Step count exceeded 200k — breaking.")
                break

        vec_env.close()

        # Convert arrays and compute metrics
        dqn_equity = np.array(dqn_equity, dtype=np.float64)
        dqn_price = np.array(dqn_price, dtype=np.float64)
        dqn_pos = np.array(dqn_pos, dtype=int)

        T = len(dqn_equity)
        if T == 0:
            print(f"No steps recorded for {TICKER} — skip.")
            continue
        N = min(LAST_N_DAYS + 1, T)
        sl = slice(T - N, T)

        dqn_metrics = compute_metrics_from_equity(dqn_equity, dqn_pos, sl)

        # Print summary
        wr = dqn_metrics["WinRate"]
        wr_str = f"{wr*100:.2f}%" if not np.isnan(wr) else "N/A"
        print(f"Trades: {dqn_metrics['TotalTrades']}, Wins: {wr_str}")
        print(f"Cumulative return: {dqn_metrics['CumReturn']:.4f} ({dqn_metrics['CumReturn']*100:.2f}%)")
        print(f"Sharpe ratio: {dqn_metrics['Sharpe']:.2f}")

        total_trades_all += dqn_metrics["TotalTrades"]
        eq_window = dqn_equity[sl]
        eq_ret = eq_window[1:] / eq_window[:-1] - 1.0
        pos_window = dqn_pos[sl]
        in_pos_mask = pos_window[:-1] == 1
        wins_all += int((eq_ret[in_pos_mask] > 0).sum()) if in_pos_mask.size else 0
        inpos_all += int(in_pos_mask.sum()) if in_pos_mask.size else 0

        save_dqn_plots(TICKER, dqn_price, dqn_pos, dqn_equity)

        # LSTM evaluation (best-effort; keep your original behaviour)
        try:
            bundle = train_lstm_for_ticker(
                split_dir=SPLIT_DIR,
                ticker=TICKER,
                window=20,
                epochs=8,
                batch_size=64,
                lr=1e-3
            )
            df_test = _load_split(SPLIT_DIR, TICKER, "test")
            lstm_out = lstm_strategy_equity(bundle, df_test, last_n_days=LAST_N_DAYS)
            lstm_metrics = compute_metrics_from_equity(
                lstm_out["equity_curve"], lstm_out["positions"], lstm_out["slice"]
            )
        except Exception as e:
            print(f"[LSTM] {TICKER} failed: {e}")
            lstm_metrics = {k: np.nan for k in ["CumReturn","AnnReturn","AnnVol","Sharpe","MaxDD","TotalTrades","WinRate"]}

        # ARIMA evaluation
        try:
            df_train = _load_split(SPLIT_DIR, TICKER, "train")
            df_test  = _load_split(SPLIT_DIR, TICKER, "test")
            arima_out = arima_strategy_equity(df_train, df_test, last_n_days=LAST_N_DAYS, order=(1,1,1))
            arima_metrics = compute_metrics_from_equity(
                arima_out["equity_curve"], arima_out["positions"], arima_out["slice"]
            )
        except Exception as e:
            print(f"[ARIMA] {TICKER} failed: {e}")
            arima_metrics = {k: np.nan for k in ["CumReturn","AnnReturn","AnnVol","Sharpe","MaxDD","TotalTrades","WinRate"]}

        records.append({
            "Ticker": TICKER,
            "DQN_CumReturn": dqn_metrics["CumReturn"],
            "DQN_AnnReturn": dqn_metrics["AnnReturn"],
            "DQN_AnnVol": dqn_metrics["AnnVol"],
            "DQN_Sharpe": dqn_metrics["Sharpe"],
            "DQN_MaxDD": dqn_metrics["MaxDD"],
            "DQN_TotalTrades": dqn_metrics["TotalTrades"],
            "DQN_WinRate": dqn_metrics["WinRate"],
            "LSTM_CumReturn": lstm_metrics["CumReturn"],
            "LSTM_AnnReturn": lstm_metrics["AnnReturn"],
            "LSTM_AnnVol": lstm_metrics["AnnVol"],
            "LSTM_Sharpe": lstm_metrics["Sharpe"],
            "LSTM_MaxDD": lstm_metrics["MaxDD"],
            "LSTM_TotalTrades": lstm_metrics["TotalTrades"],
            "LSTM_WinRate": lstm_metrics["WinRate"],
            "ARIMA_CumReturn": arima_metrics["CumReturn"],
            "ARIMA_AnnReturn": arima_metrics["AnnReturn"],
            "ARIMA_AnnVol": arima_metrics["AnnVol"],
            "ARIMA_Sharpe": arima_metrics["Sharpe"],
            "ARIMA_MaxDD": arima_metrics["MaxDD"],
            "ARIMA_TotalTrades": arima_metrics["TotalTrades"],
            "ARIMA_WinRate": arima_metrics["WinRate"],
        })

    # Save summary CSV
    summary_df = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "compare_summary_last6w.csv")
    summary_df.to_csv(out_csv, index=False)
    print(f"\n✅ Saved comparison summary to {out_csv}")

    win_rate_all = (wins_all / inpos_all) if inpos_all > 0 else float('nan')
    print(f"\n=== DQN Totals over last {LAST_N_DAYS} days across all tickers ===")
    print(f"Total trades: {total_trades_all}")
    if np.isnan(win_rate_all):
        print("Win rate: N/A")
    else:
        print(f"Total wins: {wins_all} / {inpos_all}  -> Win rate: {win_rate_all:.2%}")
