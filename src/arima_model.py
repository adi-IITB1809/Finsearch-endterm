import os
import numpy as np
import pandas as pd
from typing import Dict, Any

from statsmodels.tsa.arima.model import ARIMA


def _find_price_col(df: pd.DataFrame) -> str:
    for c in ["Close-p", "Adj Close", "Adj_Close", "close", "adj_close"]:
        if c in df.columns:
            return c
    raise ValueError("Price column not found.")


def arima_strategy_equity(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    last_n_days: int = 30,
    order=(1,1,1)
) -> Dict[str, Any]:
    """
    Simple ARIMA baseline:
      - Fit on train price
      - One-step rolling forecast over test
      - Long if forecasted price_{t+1} > price_t else flat
    """
    # sort by date if present
    for d in (df_train, df_test):
        if "Date" in d.columns:
            d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
            d.sort_values("Date", inplace=True)
            d.reset_index(drop=True, inplace=True)

    price_col = _find_price_col(df_train)

    train_p = df_train[price_col].astype(float).values
    test_p  = df_test[price_col].astype(float).values

    # Fit ARIMA on train
    # For stability on short series, catch failures and fallback to naive (hold last)
    try:
        model = ARIMA(train_p, order=order)
        fitted = model.fit()
    except Exception:
        fitted = None

    # Rolling forecast over test
    T = len(test_p)
    preds = np.zeros(T, dtype=float)
    if fitted is not None:
        history = list(train_p)
        for t in range(T):
            try:
                m = ARIMA(history, order=order).fit()
                preds[t] = m.forecast(steps=1)[0]
            except Exception:
                preds[t] = history[-1]
            history.append(test_p[t])
    else:
        preds[:] = np.concatenate([train_p[-1:], test_p[:-1]])[:T]

    # Returns from prices
    p0 = test_p[:-1]
    p1 = test_p[1:]
    ret = np.zeros_like(test_p, dtype=np.float64)
    ret[1:] = np.where(p0 == 0.0, 0.0, (p1 / p0) - 1.0)

    # Positions: 1 if forecast up
    pos = np.zeros(T, dtype=int)
    for t in range(T - 1):
        pos[t] = 1 if preds[t+1] > test_p[t] else 0
    pos[-1] = pos[-2] if T > 1 else 0

    # Equity
    equity = np.ones(T, dtype=np.float64)
    for t in range(T - 1):
        equity[t+1] = equity[t] * (1.0 + pos[t] * ret[t+1])

    N = min(last_n_days, T)
    sl = slice(T - N, T)
    return {
        "equity_curve": equity,
        "positions": pos,
        "returns": ret,
        "slice": sl,
    }
