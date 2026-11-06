# Trading Strategy Optimization

This project evaluates a DQN long/flat trading policy against LSTM, ARIMA, and buy-and-hold using daily adjusted prices for a supplied Nifty 50 ticker list. It is a historical course experiment, not a live trading system.

## Data and sample

The ticker list contains 46 symbols. The recorded fresh run downloaded 45; 43 remained after the minimum-history filters. `data/raw/` and `data/processed/` are generated locally and ignored by Git. The `data/split/` scaler metadata and model checkpoints are small and may be committed.

## Install and run

Use Python 3.11 or newer and install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
python src/run_all.py
```

`run_all.py` performs, in order: data download, feature construction, chronological splitting and train-only scaling, DQN training, LSTM training, ARIMA fitting, the historical test comparison, and a fresh recent-six-week comparison. It trains the DQN for 400,000 steps. The recent check downloads fresh data, so its period and reported numbers change on rerun.

Run the additional validation-only DQN sweep and second chronological split after the main pipeline:

```powershell
python src/validation_experiments.py
```

The experiment script trains each candidate using only the training data and ranks DQN settings on validation data. It retrains models from scratch for the alternate split. Outputs are saved under `results/`; the alternate DQN checkpoint is saved in `models/`.

## Primary historical test results

Equal-weight portfolio metrics on 1 July 2018–28 June 2019:

| Strategy | Cumulative return | Annualized return | Annualized volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|---:|
| DQN | 2.55% | 2.66% | 3.76% | 0.718 | -2.33% |
| LSTM | 10.86% | 11.39% | 12.95% | 0.898 | -11.99% |
| ARIMA | 5.38% | 5.63% | 10.72% | 0.564 | -11.77% |
| Buy-and-hold | 12.94% | 13.57% | 14.09% | 0.974 | -13.45% |

Buy-and-hold led on cumulative return and Sharpe in this one held-out period. LSTM was the strongest active baseline. DQN had lower volatility and drawdown, with lower return. Read the [Checkpoint 6 report](checkpoint6_report.md) for the method, second split, tuning sweep, recent-window result, and limitations.

## Repository layout

- `src/`: collection, preprocessing, training, evaluation, and validation experiments.
- `data/nifty50_tickers.txt`: input ticker universe.
- `models/`: trained model artifacts.
- `results/`: generated tables and plots.
- `checkpoint6_report.md`: report with methodology, results, references, and caveats.
