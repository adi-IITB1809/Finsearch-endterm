import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from env_trading import TradingEnv

# === Path settings ===
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Finsearch/
DATA_DIR = os.path.join(ROOT_DIR, "data")
SPLIT_DIR = os.path.join(DATA_DIR, "split")  # change if not using 'split'

# Pick one stock for now
TICKER = "RELIANCE_NS"

# Create environment
def make_env():
    return TradingEnv(
        ticker=TICKER,
        split="train",  # train or test
        data_dir=SPLIT_DIR,
        episode_length=50
    )

env = DummyVecEnv([make_env])

# === RL Agent ===
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0005,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=0.99,
    gamma=0.95,
    target_update_interval=500,
    train_freq=4,
    exploration_fraction=0.7,       # Slower decay (50% of training time)
    exploration_final_eps=0.15,      # Final exploration rate (was 0.05)
    exploration_initial_eps=1.0,    # Start with full exploration
)

TOTAL_TIMESTEPS = 500000  # adjust for your hardware

print("Using CUDA device" if model.device.type == "cuda" else "Using CPU")
model.learn(total_timesteps=TOTAL_TIMESTEPS)

# Save the model
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
model.save(os.path.join(RESULTS_DIR, "dqn_trading_model"))
print("✅ Model trained and saved.")
