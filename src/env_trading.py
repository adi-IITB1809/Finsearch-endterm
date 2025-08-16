import os
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

class TradingEnv(gym.Env):
    """
    Long/flat trading environment:
    Actions: 0=Hold, 1=Go Long, 2=Go Flat
    Reward shaped to encourage active, profitable trading.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, ticker, split, data_dir, episode_length=50, random_start=True, history_len=10):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of {'train','val','test'}")

        filename = f"{ticker}_{split}.csv"  # Example: RELIANCE_NS_train.csv
        self.file_path = os.path.join(data_dir, filename)
        if not os.path.isfile(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found. Please run 03_split_data.py first.")

        # === Load CSV ===
        df = pd.read_csv(self.file_path)
        self.has_date = "Date" in df.columns
        if self.has_date:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # === Force numeric conversion for all non-date columns ===
        for col in df.columns:
            if col != "Date":
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # === Identify price column ===
        price_candidates = ["Close", "Adj Close", "Adj_Close", "close", "adj_close"]
        self.price_col = next((c for c in price_candidates if c in df.columns), None)
        if self.price_col is None:
            raise ValueError("No price column found.")

        if self.has_date:
            df = df.sort_values("Date").reset_index(drop=True)

        # Keep numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.price_col not in numeric_cols:
            raise ValueError(f"Price column '{self.price_col}' is not numeric after conversion.")

        # Replace inf and NaN, forward/backward fill
        df = df[numeric_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna().reset_index(drop=True)

        self.df = df
        self.n = len(df)
        if self.n < history_len + 2:
            raise ValueError("Not enough rows after cleaning.")

        self.numeric_cols = numeric_cols
        self.action_space = spaces.Discrete(3)

        # Observation now includes `history_len` previous steps
        self.history_len = history_len
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(len(numeric_cols) * history_len,),
            dtype=np.float32
        )

        self.episode_length = int(episode_length)
        self.random_start = bool(random_start)
        self.reset()

    def _price(self, idx):
        return float(self.df.iloc[idx][self.price_col])

    def _get_obs(self):
        start_idx = max(self.current_step - self.history_len + 1, 0)
        frames = self.df.iloc[start_idx:self.current_step+1][self.numeric_cols].values
        # Pad if not enough history
        if len(frames) < self.history_len:
            pad = np.zeros((self.history_len - len(frames), len(self.numeric_cols)))
            frames = np.vstack((pad, frames))
        return frames.flatten().astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        if self.episode_length >= self.n - 1:
            self.start_index = 0
            self.end_index = self.n - 1
        else:
            if self.random_start:
                hi = self.n - self.episode_length - 1
                self.start_index = np.random.randint(0, hi + 1) if hi > 0 else 0
            else:
                self.start_index = 0
            self.end_index = self.start_index + self.episode_length

        self.current_step = self.start_index
        self.position = 0
        self.equity = 1.0
        self.trades = 0
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)
        if action not in (0, 1, 2):
            action = 0

        desired_pos = 1 if action == 1 else 0 if action == 2 else self.position
        if desired_pos != self.position:
            self.trades += 1
        self.position = desired_pos

        done = False
        p0 = self._price(self.current_step)
        p1 = self._price(self.current_step + 1) if self.current_step + 1 < self.n else p0

        # Return calculation
        ret = 0.0 if p0 == 0 else (p1 / p0) - 1.0
        reward = float(self.position) * ret

        # Reward shaping to encourage active & correct trading
        if self.position == 0:
            reward -= 0.0001  # penalty for staying flat
        if self.position == 1 and ret < 0:
            reward -= 0.001   # penalty for wrong long

        self.equity *= (1.0 + reward)
        self.current_step += 1

        if self.current_step >= self.end_index:
            done = True

        obs = self._get_obs()
        info = {
            "equity": self.equity,
            "position": self.position,
            "trades": self.trades,
            "price": self._price(self.current_step)
        }
        return obs, reward, done, False, info

    def render(self):
        print(f"step={self.current_step} pos={self.position} eq={self.equity:.4f} price={self._price(self.current_step):.2f}")
