import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from config import *
from common import load_dataset, load_recent, backtest, metrics
from env_trading import TradingEnv, rollout_positions
from lstm_agent import load_model, lstm_positions
from arima_agent import load_params, arima_positions

warnings.filterwarnings("ignore")
STRATS = ["DQN", "LSTM", "ARIMA", "BuyHold"]


def main(recent):
    data = load_recent() if recent else load_dataset()
    tag = "recent6w" if recent else "test"
    dqn = DQN.load(os.path.join(MODELS_DIR, "dqn_best"), device="cpu")
    lstm, arima = load_model(), load_params()
    env = TradingEnv(data, split="test", random_start=False, shaped=False)
    policy = lambda o: int(dqn.predict(o, deterministic=True)[0])

    rows, nets = [], {s: {} for s in STRATS}
    for sym, d in data.items():
        a, b = d["bounds"]["test"]
        ret = d["ret"][a:b]
        dates = d["dates"][a:b]
        pos = {
            "DQN": rollout_positions(env, policy, sym, "test"),
            "LSTM": lstm_positions(lstm, d, "test"),
            "ARIMA": arima_positions(arima.get(sym), d, "test"),
            "BuyHold": np.ones(b - a, dtype=int),
        }
        for s in STRATS:
            net = backtest(pos[s], ret)
            nets[s][sym] = pd.Series(net, index=dates)
            rows.append(dict(Ticker=sym, Strategy=s, **metrics(net, pos[s])))

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(RESULTS_DIR, f"per_ticker_{tag}.csv"), index=False)

    # ---- cross-sectional summary ----
    summ = per.groupby("Strategy")[["CumReturn", "Sharpe", "AnnVol", "MaxDD", "Trades", "Exposure", "WinRate"]].mean()
    bh = per[per.Strategy == "BuyHold"].set_index("Ticker")["Sharpe"]
    summ["%StocksBeatingBuyHoldSharpe"] = [
        100 * (per[per.Strategy == s].set_index("Ticker")["Sharpe"] > bh).mean() for s in summ.index]
    summ.loc[STRATS].to_csv(os.path.join(RESULTS_DIR, f"summary_{tag}.csv"))

    # ---- equal-weight portfolio across stocks ----
    port = {s: pd.DataFrame(nets[s]).mean(axis=1) for s in STRATS}
    pm = pd.DataFrame({s: metrics(port[s].values) for s in STRATS}).T
    pm.to_csv(os.path.join(RESULTS_DIR, f"portfolio_{tag}.csv"))

    pd.set_option("display.width", 200)
    print(f"\n=== {tag}: {len(data)} stocks | mean across stocks ===")
    print(summ.loc[STRATS].round(4).to_string())
    print(f"\n=== {tag}: equal-weight portfolio ===")
    print(pm.round(4).to_string())
    if recent:
        print("\nNOTE: ~30 trading days only -> anecdotal; annualised numbers are not meaningful.")

    # ---- plots ----
    fig, ax = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for s in STRATS:
        eq = (1 + port[s]).cumprod()
        ax[0].plot(eq.index, eq.values, label=s)
        ax[1].plot(eq.index, eq.values / eq.cummax().values - 1, label=s)
    ax[0].set_title(f"Equal-weight portfolio equity ({tag}, after {COST*100:.1f}% costs)"); ax[0].legend()
    ax[1].set_title("Drawdown")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, f"equity_{tag}.png"), dpi=150); plt.close()

    plt.figure(figsize=(8, 5))
    plt.boxplot([per[per.Strategy == s]["Sharpe"].dropna() for s in STRATS], labels=STRATS)
    plt.title(f"Per-stock Sharpe distribution ({tag})"); plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"sharpe_box_{tag}.png"), dpi=150); plt.close()
    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--recent", action="store_true", help="evaluate on the last 6 weeks (downloads fresh data)")
    main(p.parse_args().recent)
