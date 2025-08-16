
import os
import numpy as np
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env_trading import TradingEnv
import warnings

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Finsearch/
SPLIT_DIR = os.path.join(ROOT_DIR, "data", "split")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_trading_model.zip")
print("Using model:", MODEL_PATH)
model = DQN.load(MODEL_PATH)

# Expected observation dimension according to the loaded policy
expected_shape = model.policy.observation_space.shape
if len(expected_shape) != 1:
    raise RuntimeError(f"Unexpected policy observation shape: {expected_shape}")
EXPECTED_DIM = int(expected_shape[0])
print(f"Model expects observation vector of length: {EXPECTED_DIM}")

# Collect tickers
tickers = [f.replace("_test.csv", "") for f in os.listdir(SPLIT_DIR) if f.endswith("_test.csv")]
tickers.sort()
print(f"Found {len(tickers)} tickers for evaluation: {tickers}")

# Overall accumulators
total_trades_all = 0
total_wins_all = 0
total_trading_days_in_pos = 0

def ensure_close_numeric_and_clean(path):
    df = pd.read_csv(path)
    if "Close" not in df.columns:
        raise ValueError(f"'Close' column not found in {path}")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    # Drop rows where Close is NaN (can't trade those)
    df.dropna(subset=["Close"], inplace=True)
    df.to_csv(path, index=False)

def pad_or_truncate_obs(obs_arr, expected_dim):
    """
    obs_arr: numpy array with shape (n_env, dim) or (dim,)
    returns: numpy array with shape (n_env, expected_dim)
    """
    obs = np.asarray(obs_arr)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    n_env, curr_dim = obs.shape
    if curr_dim == expected_dim:
        return obs
    elif curr_dim < expected_dim:
        pad_width = expected_dim - curr_dim
        pad = np.zeros((n_env, pad_width), dtype=obs.dtype)
        obs_padded = np.concatenate([obs, pad], axis=1)
        warnings.warn(f"Padding observation from {curr_dim} -> {expected_dim} (zeros appended).")
        return obs_padded
    else:  # curr_dim > expected_dim
        obs_trunc = obs[:, :expected_dim]
        warnings.warn(f"Truncating observation from {curr_dim} -> {expected_dim} (last features dropped).")
        return obs_trunc

for TICKER in tickers:
    print(f"\n=== Evaluating {TICKER} ===")
    test_path = os.path.join(SPLIT_DIR, f"{TICKER}_test.csv")

    # 1) Ensure CSV OK
    try:
        ensure_close_numeric_and_clean(test_path)
    except Exception as e:
        print(f"Skipping {TICKER} due to CSV problem: {e}")
        continue

    # 2) Build env (vectorized) — match training setup as closely as possible
    def make_env():
        return TradingEnv(
            ticker=TICKER,
            split="test",
            data_dir=SPLIT_DIR,
            episode_length=10_000,
            random_start=False
        )

    vec_env = DummyVecEnv([make_env])

    # 3) Reset and get initial obs (handle different reset return signatures)
    reset_ret = vec_env.reset()
    # Some VecEnvs return only obs; some (with gymnasium special wrapper) might return (obs, infos)
    if isinstance(reset_ret, tuple) and len(reset_ret) == 2:
        obs, _reset_info = reset_ret
    else:
        obs = reset_ret

    obs = pad_or_truncate_obs(obs, EXPECTED_DIM)

    # storage
    equity_curve = []
    price_curve = []
    positions = []

    done_flag = False
    step_count = 0

    # We will loop until the vectorized env signals done for the single env.
    # Because it's vectorized with n_env=1, we look at the first element of the arrays returned.
    while True:
        # Model expects vectorized obs shape (n_env, obs_dim)
        action, _ = model.predict(obs, deterministic=True)

        # Step the vec env. Different VecEnv implementations return different tuple shapes:
        step_ret = vec_env.step(action)

        # Normalize the output into obs, rewards, dones (boolean array), infos
        if len(step_ret) == 5:
            # Likely: obs, rewards, terminateds, truncateds, infos
            obs_raw, rewards, terminateds, truncateds, infos = step_ret
            dones = np.logical_or(terminateds, truncateds)
        elif len(step_ret) == 4:
            # Likely: obs, rewards, dones, infos
            obs_raw, rewards, dones, infos = step_ret
        else:
            raise RuntimeError(f"Unexpected return from vec_env.step(): got {len(step_ret)} items")

        # Pad/truncate obs to expected dim (should be stable across steps)
        obs = pad_or_truncate_obs(obs_raw, EXPECTED_DIM)

        # Because n_env==1, index the first env's outputs
        reward = rewards[0] if isinstance(rewards, (list, np.ndarray)) else rewards
        done = dones[0] if isinstance(dones, (list, np.ndarray)) else dones
        info = infos[0] if isinstance(infos, (list, np.ndarray)) else infos

        # Save metrics from info (your env returns "equity","price","position")
        equity_curve.append(info.get("equity", np.nan))
        price_curve.append(info.get("price", np.nan))
        positions.append(info.get("position", np.nan))

        step_count += 1

        if bool(done):
            break

        # safety: avoid infinite loops
        if step_count > 200_000:
            warnings.warn("Step count exceeded 200k — breaking.")
            break

    # Convert to arrays and compute stats (align with how you computed previously)
    equity_curve = np.array(equity_curve)
    price_curve = np.array(price_curve)
    positions = np.array(positions)

    if len(equity_curve) < 2:
        print(f"Not enough data produced for {TICKER} (len={len(equity_curve)}). Skipping metrics.")
        continue

    eq_ret = equity_curve[1:] / equity_curve[:-1] - 1.0
    cum_return = equity_curve[-1] - 1.0
    N = len(eq_ret)
    ann_return = (equity_curve[-1]) ** (252 / N) - 1.0 if N > 0 else 0.0
    ann_vol = np.std(eq_ret, ddof=1) * np.sqrt(252) if N > 1 else 0.0
    sharpe = (ann_return / ann_vol) if ann_vol > 0 else np.nan
    roll_max = np.maximum.accumulate(equity_curve)
    drawdown = equity_curve / roll_max - 1.0
    max_dd = np.min(drawdown)

    # Trades & win-rate while long
    total_trades = int(np.sum(np.diff(positions) != 0))
    in_pos = positions[:-1] == 1  # days where previous step position was long
    wins = int(np.sum(eq_ret[in_pos] > 0)) if in_pos.sum() > 0 else 0
    win_rate = (wins / in_pos.sum()) if in_pos.sum() > 0 else np.nan

    # Accumulate
    total_trades_all += total_trades
    total_wins_all += wins
    total_trading_days_in_pos += in_pos.sum()

    # Print per-ticker summary
    print(f"Cumulative return: {cum_return:.6f} ({cum_return*100:.2f}%)")
    print(f"Annualized return: {ann_return:.6f}")
    print(f"Annualized volatility: {ann_vol:.6f}")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Max Drawdown: {max_dd:.6f}")
    print(f"Total trades: {total_trades}")
    if np.isnan(win_rate):
        print(f"Win rate (while long): N/A (never in position)")
    else:
        print(f"Win rate (while long): {win_rate:.2%}")

# Final overall summary
overall_win_rate = (total_wins_all / total_trading_days_in_pos) if total_trading_days_in_pos > 0 else np.nan
print("\n========== OVERALL SUMMARY ==========")
print(f"Total trades across all tickers: {total_trades_all}")
if np.isnan(overall_win_rate):
    print("Overall win rate (while long) across all tickers: N/A")
else:
    print(f"Overall win rate (while long) across all tickers: {overall_win_rate:.2%}")
print("=====================================")
