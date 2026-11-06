import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config import HISTORY_LEN, COST


class TradingEnv(gym.Env):
    """
    Long/flat trading env. Actions: 0=Hold, 1=Go long, 2=Go flat.
    Observation: last `history_len` days of standardized features + current position.
    Decision at close of day t; position earns the return from close t to close t+1 (no look-ahead).
    info['net_return'] is the TRUE return after costs; the shaped reward is for training only.
    """
    metadata = {"render_modes": []}

    def __init__(self, data, split="train", history_len=HISTORY_LEN, episode_length=252,
                 random_start=True, cost=COST, shaped=True, loss_penalty=0.5, reward_scale=100.0):
        super().__init__()
        self.data, self.split = data, split
        self.h, self.episode_length = history_len, episode_length
        self.random_start, self.cost = random_start, cost
        self.shaped, self.loss_penalty, self.reward_scale = shaped, loss_penalty, reward_scale
        self.tickers = [t for t, d in data.items() if split in d["bounds"]]
        if not self.tickers:
            raise ValueError(f"No tickers have split '{split}'")
        self.n_feat = next(iter(data.values()))["X"].shape[1]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.n_feat * self.h + 1,), dtype=np.float32)

    def _obs(self):
        w = self.d["X"][self.t - self.h + 1: self.t + 1]
        return np.concatenate([w.ravel(), [float(self.position)]]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        split = options.get("split", self.split)
        cands = [t for t, d in self.data.items() if split in d["bounds"]]
        self.ticker = str(options.get("ticker") or self.np_random.choice(cands))
        self.d = self.data[self.ticker]
        a, b = self.d["bounds"][split]
        if options.get("full", not self.random_start):
            self.t, self.end = a, b
        else:
            L = min(self.episode_length, b - a)
            s = int(self.np_random.integers(a, b - L + 1))
            self.t, self.end = s, s + L
        self.position, self.equity = 0, 1.0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        desired = 1 if action == 1 else 0 if action == 2 else self.position
        turn = abs(desired - self.position)
        r = float(self.d["ret"][self.t])
        net = desired * r - self.cost * turn
        reward = net
        if self.shaped:  # extra penalty on losing long days -> downside aversion
            reward = net - self.loss_penalty * max(0.0, -desired * r)
        reward *= self.reward_scale
        self.position = desired
        self.equity *= (1 + net)
        self.t += 1
        terminated = self.t >= self.end
        info = dict(net_return=net, position=self.position, equity=self.equity, ticker=self.ticker)
        return self._obs(), float(reward), terminated, False, info


def rollout_positions(env, policy_fn, ticker, split):
    """Run a policy over the full segment of one ticker; returns position chosen on each decision day."""
    obs, _ = env.reset(options={"ticker": ticker, "split": split, "full": True})
    pos, done = [], False
    while not done:
        obs, _, done, _, info = env.step(policy_fn(obs))
        pos.append(info["position"])
    return np.array(pos)
