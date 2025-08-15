import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from env_trading import TradingEnv

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Finsearch/
SPLIT_DIR = os.path.join(ROOT_DIR, "data", "split")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_trading_model.zip")
print("Using model:", MODEL_PATH)

# Load model
model = DQN.load(MODEL_PATH)

# All tickers in split/test folder
tickers = [f.replace("_test.csv", "") for f in os.listdir(SPLIT_DIR) if f.endswith("_test.csv")]
print(f"Found {len(tickers)} tickers for evaluation: {tickers}")

# For total stats
total_trades_all = 0
total_wins_all = 0
total_trading_days_in_pos = 0

for TICKER in tickers:
    print(f"\n=== Evaluating {TICKER} ===")
    
    env = TradingEnv(
        ticker=TICKER,
        split="test",
        data_dir=SPLIT_DIR,
        episode_length=10_000,
        random_start=False
    )

    obs, _ = env.reset()
    done = False

    equity_curve = [env.equity]
    price_curve = [env._price(env.current_step)]
    positions = [env.position]

    # Step through the test set
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        equity_curve.append(info["equity"])
        price_curve.append(info["price"])
        positions.append(info["position"])

    equity_curve = np.array(equity_curve)
    price_curve = np.array(price_curve)
    positions = np.array(positions)

    # ===== Metrics =====
    eq_ret = equity_curve[1:] / equity_curve[:-1] - 1.0
    cum_return = equity_curve[-1] - 1.0
    N = len(eq_ret)
    ann_return = (equity_curve[-1]) ** (252 / N) - 1.0 if N > 0 else 0.0
    ann_vol = np.std(eq_ret, ddof=1) * np.sqrt(252) if N > 1 else 0.0
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan
    roll_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / roll_max - 1.0
    max_dd = np.min(drawdown)

    total_trades = int(np.sum(np.diff(positions) != 0))
    in_pos = positions[:-1] == 1
    wins = (eq_ret[in_pos] > 0).sum()
    win_rate = (wins / in_pos.sum()) if in_pos.sum() > 0 else np.nan

    # Accumulate totals
    total_trades_all += total_trades
    total_wins_all += wins
    total_trading_days_in_pos += in_pos.sum()

    # ===== Print summary =====
    print(f"Cumulative return: {cum_return:.4f} ({cum_return*100:.2f}%)")
    print(f"Annualized return: {ann_return:.4f}")
    print(f"Annualized volatility: {ann_vol:.4f}")
    print(f"Sharpe ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.4f}")
    print(f"Total trades: {total_trades}")
    print(f"Win rate (while long): {win_rate:.2%}")

# ===== Final overall summary =====
overall_win_rate = (total_wins_all / total_trading_days_in_pos) if total_trading_days_in_pos > 0 else np.nan

print("\n========== OVERALL SUMMARY ==========")
print(f"Total trades across all tickers: {total_trades_all}")
print(f"Overall win rate (while long) across all tickers: {overall_win_rate:.2%}")
print("=====================================")
