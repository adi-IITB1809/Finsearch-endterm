"""Validation-only DQN sweep and a chronological 2010-2019 split comparison.

Run after the regular preprocessing step. The alternate split is trained from
scratch; neither experiment reads the held-out test returns for model selection.
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from statsmodels.tsa.arima.model import ARIMA

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from config import *  # noqa: E402,F403
from common import (  # noqa: E402
    _assemble, backtest, load_dataset, metrics, read_used, safe_name, sharpe,
)
from env_trading import TradingEnv, rollout_positions  # noqa: E402
from lstm_agent import PriceLSTM, make_sequences, lstm_positions  # noqa: E402

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ALT_TRAIN_END = "2015-12-31"
ALT_VAL_END = "2016-12-31"
ALT_STEPS = 40_000
SWEEP_STEPS = 10_000


def mean_val_sharpe(model, data, history_len):
    env = TradingEnv(data, split="val", random_start=False, shaped=False,
                     history_len=history_len)
    policy = lambda obs: int(model.predict(obs, deterministic=True)[0])
    vals = []
    for ticker in env.tickers:
        pos = rollout_positions(env, policy, ticker, "val")
        a, b = data[ticker]["bounds"]["val"]
        vals.append(np.nan_to_num(sharpe(backtest(pos, data[ticker]["ret"][a:b]))))
    return float(np.mean(vals))


def train_dqn(data, params, steps, history_len, seed, save_path=None):
    env = Monitor(TradingEnv(
        data, split="train", episode_length=252, random_start=True, shaped=True,
        loss_penalty=params["loss_penalty"], history_len=history_len,
    ))
    model = DQN(
        "MlpPolicy", env, verbose=0, seed=seed, device="cpu",
        learning_rate=params["learning_rate"], buffer_size=50_000,
        learning_starts=min(2_000, steps // 5), batch_size=128,
        gamma=params["gamma"], tau=1.0, target_update_interval=1_000,
        train_freq=4, exploration_initial_eps=1.0, exploration_final_eps=0.05,
        exploration_fraction=0.3, policy_kwargs=dict(net_arch=[128, 128]),
    )
    model.learn(total_timesteps=steps)
    score = mean_val_sharpe(model, data, history_len)
    if save_path:
        model.save(save_path)
    return model, score


def load_alternate_split():
    data = {}
    for ticker in read_used():
        name = safe_name(ticker)
        proc = PROC_DIR + "/" + name + ".csv"
        if not os.path.exists(proc):
            continue
        frame = pd.read_csv(proc, index_col=0, parse_dates=True)
        train = frame.loc[frame.index <= ALT_TRAIN_END]
        val = frame.loc[(frame.index > ALT_TRAIN_END) & (frame.index <= ALT_VAL_END)]
        test = frame.loc[frame.index > ALT_VAL_END]
        if len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        mean = train[FEATURES].mean()
        std = train[FEATURES].std().replace(0, 1.0).fillna(1.0)
        scaler = {"mean": mean.to_dict(), "std": std.to_dict()}
        data_item = _assemble(frame, scaler, {
            "train": (None, ALT_TRAIN_END),
            "val": (ALT_TRAIN_END, ALT_VAL_END),
            "test": (ALT_VAL_END, None),
        })
        if data_item and all(key in data_item["bounds"] for key in ("train", "val", "test")):
            data[ticker] = data_item
    return data


def fit_lstm(data, seed=SEED, epochs=15, patience=4):
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr, ytr = map(torch.from_numpy, make_sequences(data, "train"))
    Xva, yva = map(torch.from_numpy, make_sequences(data, "val"))
    model = PriceLSTM(Xtr.shape[-1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = torch.nn.HuberLoss()
    best_loss, best_state, patience_count = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr))
        for start in range(0, len(order), 256):
            batch = order[start:start + 256]
            optimizer.zero_grad()
            loss_fn(model(Xtr[batch]), ytr[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            value = float(loss_fn(model(Xva), yva))
        if value < best_loss:
            best_loss = value
            best_state = {key: item.clone() for key, item in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_loss


def eval_alternate(data, dqn, lstm, arima_params):
    env = TradingEnv(data, split="test", random_start=False, shaped=False)
    policy = lambda obs: int(dqn.predict(obs, deterministic=True)[0])
    nets = {strategy: {} for strategy in ("DQN", "LSTM", "ARIMA", "BuyHold")}
    records = []
    for ticker, item in data.items():
        a, b = item["bounds"]["test"]
        positions = {
            "DQN": rollout_positions(env, policy, ticker, "test"),
            "LSTM": lstm_positions(lstm, item, "test"),
            "ARIMA": arima_positions(arima_params[ticker], item),
            "BuyHold": np.ones(b - a, dtype=int),
        }
        for strategy, pos in positions.items():
            net = backtest(pos, item["ret"][a:b])
            nets[strategy][ticker] = pd.Series(net, index=item["dates"][a:b])
            records.append({"Ticker": ticker, "Strategy": strategy, **metrics(net, pos)})
    portfolio = pd.DataFrame({
        strategy: pd.DataFrame(per_ticker).mean(axis=1)
        for strategy, per_ticker in nets.items()
    })
    summary = pd.DataFrame({strategy: metrics(portfolio[strategy].to_numpy())
                            for strategy in portfolio}).T
    tag = "alternate_2010_2019"
    pd.DataFrame(records).to_csv(os.path.join(RESULTS_DIR, f"per_ticker_{tag}.csv"), index=False)
    summary.to_csv(os.path.join(RESULTS_DIR, f"portfolio_{tag}.csv"))
    return summary


def main():
    primary = load_dataset()
    # One-factor-at-a-time settings around the original configuration. Each
    # candidate is trained on the train split and ranked only on validation.
    trials = [
        ("baseline", 1e-4, 0.99, 0.50, 10),
        ("learning_rate_3e-4", 3e-4, 0.99, 0.50, 10),
        ("gamma_0.95", 1e-4, 0.95, 0.50, 10),
        ("loss_penalty_0.25", 1e-4, 0.99, 0.25, 10),
        ("history_5", 1e-4, 0.99, 0.50, 5),
    ]
    sweep_rows = []
    for idx, (name, lr, gamma, penalty, history) in enumerate(trials):
        params = {"learning_rate": lr, "gamma": gamma, "loss_penalty": penalty}
        _, score = train_dqn(primary, params, SWEEP_STEPS, history, SEED + idx)
        sweep_rows.append({"trial": name, "learning_rate": lr, "gamma": gamma,
                           "loss_penalty": penalty, "history_len": history,
                           "training_steps": SWEEP_STEPS,
                           "mean_validation_sharpe": score})
        print(f"sweep {name}: validation Sharpe={score:.4f}", flush=True)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(RESULTS_DIR, "validation_sweep.csv"), index=False)

    # Second, independent chronological segmentation; all three models are
    # retrained and the final period remains untouched until evaluation.
    alternate = load_alternate_split()
    if not alternate:
        raise RuntimeError("No stocks satisfy minimum history for the alternate split")
    alt_params = {"learning_rate": 1e-4, "gamma": 0.99, "loss_penalty": 0.50}
    alt_dqn, val_sharpe = train_dqn(
        alternate, alt_params, ALT_STEPS, HISTORY_LEN, SEED,
        os.path.join(MODELS_DIR, "dqn_alternate_2010_2019"),
    )
    alt_lstm, lstm_val_loss = fit_lstm(alternate)
    arima_params = {}
    for ticker, item in alternate.items():
        a, b = item["bounds"]["train"]
        log_returns = np.diff(np.log(item["price"]), prepend=np.log(item["price"][0])) * 100
        try:
            fitted = ARIMA(log_returns[a + 1:b + 1], order=(1, 0, 1), trend="c").fit()
            arima_params[ticker] = np.asarray(fitted.params)
        except Exception:
            arima_params[ticker] = None
    result = eval_alternate(alternate, alt_dqn, alt_lstm, arima_params)
    pd.DataFrame([{
        "train_end": ALT_TRAIN_END, "validation_end": ALT_VAL_END,
        "test_start": "2017-01-01", "test_end": "2019-06-28",
        "stocks": len(alternate), "dqn_steps": ALT_STEPS,
        "dqn_validation_sharpe": val_sharpe, "lstm_best_validation_loss": lstm_val_loss,
    }]).to_csv(os.path.join(RESULTS_DIR, "alternate_split_metadata.csv"), index=False)
    print(f"Alternate split: {len(alternate)} stocks; DQN validation Sharpe={val_sharpe:.4f}; "
          f"LSTM validation loss={lstm_val_loss:.5f}", flush=True)
    print(result.round(4).to_string())


def arima_positions(params, item):
    a, b = item["bounds"]["test"]
    if params is None:
        return np.zeros(b - a, dtype=int)
    log_returns = np.diff(np.log(item["price"]), prepend=np.log(item["price"][0])) * 100
    fitted = np.asarray(ARIMA(log_returns, order=(1, 0, 1), trend="c").smooth(params).fittedvalues)
    return (fitted[a + 1:b + 1] > 0).astype(int)


if __name__ == "__main__":
    main()
