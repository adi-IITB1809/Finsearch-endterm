import os, argparse
import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from config import *
from common import load_dataset, backtest, sharpe
from env_trading import TradingEnv, rollout_positions

# Small MLP batches run faster without PyTorch oversubscribing CPU threads.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def val_score(model, data, env):
    policy = lambda o: int(model.predict(o, deterministic=True)[0])
    scores = []
    for t in env.tickers:
        pos = rollout_positions(env, policy, t, "val")
        a, b = data[t]["bounds"]["val"]
        scores.append(np.nan_to_num(sharpe(backtest(pos, data[t]["ret"][a:b]))))
    return float(np.mean(scores))


class ValCallback(BaseCallback):
    def __init__(self, data, every, path):
        super().__init__()
        self.data, self.every, self.path, self.best = data, every, path, -np.inf
        self.env = TradingEnv(data, split="val", random_start=False, shaped=False)

    def _on_step(self):
        if self.n_calls % self.every == 0:
            s = val_score(self.model, self.data, self.env)
            tag = ""
            if s > self.best:
                self.best, tag = s, "  <- new best, saved"
                self.model.save(self.path)
            print(f"[step {self.num_timesteps}] mean val Sharpe = {s:.3f}{tag}")
        return True


def main(steps):
    data = load_dataset()
    print(f"Training DQN on {len(data)} stocks for {steps} steps")
    env = Monitor(TradingEnv(data, split="train", episode_length=252, random_start=True, shaped=True))
    model = DQN(
        "MlpPolicy", env, verbose=0, seed=SEED, device="cpu",
        learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000, batch_size=128,
        gamma=0.99, tau=1.0, target_update_interval=1_000, train_freq=4,
        exploration_initial_eps=1.0, exploration_final_eps=0.05, exploration_fraction=0.3,
        policy_kwargs=dict(net_arch=[128, 128]),
    )
    best_path = os.path.join(MODELS_DIR, "dqn_best")
    cb = ValCallback(data, every=max(steps // 15, 5_000), path=best_path)
    model.learn(total_timesteps=steps, callback=cb)
    model.save(os.path.join(MODELS_DIR, "dqn_final"))
    print(f"Done. Best val Sharpe {cb.best:.3f}. Saved {best_path}.zip")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400_000)
    main(p.parse_args().steps)
