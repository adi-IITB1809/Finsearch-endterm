import os, pickle, warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from config import *
from common import load_dataset

warnings.filterwarnings("ignore")
ORDER = (1, 0, 1)   # on log returns == ARIMA(1,1,1) on log prices
PATH = os.path.join(MODELS_DIR, "arima_params.pkl")


def log_ret_pct(price):
    return np.diff(np.log(price), prepend=np.log(price[0])) * 100


def fit_all():
    data = load_dataset()
    params = {}
    for sym, d in data.items():
        lr = log_ret_pct(d["price"])
        tb = d["bounds"]["train"][1]
        try:
            res = ARIMA(lr[1:tb + 1], order=ORDER, trend="c").fit()   # TRAIN data only
            params[sym] = np.asarray(res.params)
        except Exception as e:
            print("ARIMA failed", sym, e)
            params[sym] = None
    with open(PATH, "wb") as f:
        pickle.dump(params, f)
    print(f"Fitted ARIMA{ORDER} for {sum(v is not None for v in params.values())}/{len(params)} stocks")


def load_params():
    with open(PATH, "rb") as f:
        return pickle.load(f)


def arima_positions(params, d, split):
    a, b = d["bounds"][split]
    if params is None:
        return np.zeros(b - a, dtype=int)
    lr = log_ret_pct(d["price"])
    res = ARIMA(lr, order=ORDER, trend="c").smooth(params)       # fixed params, no refit, no look-ahead
    fitted = np.asarray(res.fittedvalues)                         # fitted[i] = forecast of lr[i] from data < i
    return (fitted[a + 1: b + 1] > 0).astype(int)                 # decision on day t forecasts return of t+1


if __name__ == "__main__":
    fit_all()
