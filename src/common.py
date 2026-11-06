import os, re, json, itertools
import numpy as np
import pandas as pd
from config import *

SEGMENTS = {"train": (None, TRAIN_END), "val": (TRAIN_END, VAL_END), "test": (VAL_END, None)}
MIN_START = max(HISTORY_LEN, LSTM_WINDOW) - 1


def safe_name(sym):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", sym)


def read_tickers():
    out = []
    with open(TICKERS_FILE) as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#"):
                continue
            if "." not in t:
                t += ".NS"
            out.append(t)
    return out


def read_used():
    with open(os.path.join(SPLIT_DIR, "tickers_used.txt")) as f:
        return [l.strip() for l in f if l.strip()]


def download_prices(sym, start, end):
    import yfinance as yf
    try:
        cache_dir = os.path.join(DATA_DIR, ".yfinance_cache")
        os.makedirs(cache_dir, exist_ok=True)
        yf.set_tz_cache_location(cache_dir)
        df = yf.download(sym, start=start, end=end, auto_adjust=True, progress=False)
    except Exception as exc:
        print(f"  download error for {sym}: {exc}")
        return None
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])


def build_features(df):
    """Stationary, ratio-style features. Everything at row t uses data up to close of day t only."""
    c = df["Close"].astype(float)
    v = df["Volume"].astype(float)
    lc = np.log(c)
    lr = lc.diff()
    out = pd.DataFrame(index=df.index)
    out["Close"] = c
    out["ret_1d"] = lr
    out["ret_5d"] = lc.diff(5)
    out["ret_10d"] = lc.diff(10)
    out["vol_10d"] = lr.rolling(10).std()
    out["vol_chg"] = np.log((v + 1) / (v.shift(1) + 1))
    out["sma10_gap"] = c / c.rolling(10).mean() - 1
    out["sma50_gap"] = c / c.rolling(50).mean() - 1
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    au = up.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    ad = dn.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    out["rsi"] = (100 * au / (au + ad) - 50) / 50
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    out["macd"] = macd / c
    out["macd_hist"] = (macd - sig) / c
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    return out[out["Close"] > 0]


def _assemble(df, sc, segments):
    mean = np.array([sc["mean"][f] for f in FEATURES])
    std = np.array([sc["std"][f] for f in FEATURES])
    X = np.clip((df[FEATURES].values - mean) / std, -5, 5).astype(np.float32)
    price = df["Close"].values.astype(float)
    n = len(df)
    ret = np.zeros(n)
    ret[:-1] = price[1:] / price[:-1] - 1.0          # ret[t] = return earned by holding from close t to close t+1
    dates = df.index
    bounds = {}
    for name, (lo, hi) in segments.items():
        a = 0 if lo is None else int(dates.searchsorted(pd.Timestamp(lo), side="right"))
        b = n if hi is None else int(dates.searchsorted(pd.Timestamp(hi), side="right"))
        a, b = max(a, MIN_START), min(b, n - 1)
        if b - a >= 2:
            bounds[name] = (a, b)               # decision days a..b-1
    if not bounds:
        return None
    return dict(dates=dates, X=X, price=price, ret=ret, bounds=bounds)


def load_dataset():
    data = {}
    for sym in read_used():
        name = safe_name(sym)
        df = pd.read_csv(os.path.join(PROC_DIR, name + ".csv"), index_col=0, parse_dates=True)
        with open(os.path.join(SPLIT_DIR, name + "_scaler.json")) as f:
            sc = json.load(f)
        item = _assemble(df, sc, SEGMENTS)
        if item:
            data[sym] = item
    return data


def load_recent(weeks=6, warmup_days=450):
    """Fresh data for the last `weeks` weeks (Checkpoint 6). Uses train-period scalers."""
    today = pd.Timestamp.today().normalize()
    cutoff = today - pd.Timedelta(weeks=weeks)
    start = (cutoff - pd.Timedelta(days=warmup_days)).strftime("%Y-%m-%d")
    end = (today + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = {}
    for sym in read_used():
        raw = download_prices(sym, start, end)
        if raw is None:
            print("  no recent data:", sym)
            continue
        feat = build_features(raw)
        with open(os.path.join(SPLIT_DIR, safe_name(sym) + "_scaler.json")) as f:
            sc = json.load(f)
        item = _assemble(feat, sc, {"test": (cutoff, None)})
        if item:
            data[sym] = item
    print(f"Recent window: decision days after {cutoff.date()} up to {today.date()}")
    return data


# ---------------- backtest + metrics (shared by ALL strategies) ----------------
def backtest(pos, ret_next, cost=COST):
    """pos[i] = position chosen at close of decision day i, earns ret_next[i]. Cost charged on every position change."""
    pos = np.asarray(pos, float)
    prev = np.concatenate([[0.0], pos[:-1]])
    return pos * np.asarray(ret_next) - cost * np.abs(pos - prev)


def sharpe(net):
    net = np.asarray(net, float)
    s = net.std(ddof=1) if len(net) > 1 else 0.0
    return float(net.mean() / s * np.sqrt(252)) if s > 1e-12 else np.nan


def trade_stats(pos, net):
    wins = total = 0
    for k, g in itertools.groupby(zip(pos, net), key=lambda x: x[0]):
        if k == 1:
            r = np.prod([1 + x[1] for x in g]) - 1
            total += 1
            wins += int(r > 0)
    return total, (wins / total if total else np.nan)


def metrics(net, pos=None):
    net = np.asarray(net, float)
    n = len(net)
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    out = dict(
        CumReturn=float(eq[-1] - 1),
        AnnReturn=float(eq[-1] ** (252 / n) - 1) if eq[-1] > 0 else -1.0,
        AnnVol=float(net.std(ddof=1) * np.sqrt(252)) if n > 1 else np.nan,
        Sharpe=sharpe(net),
        MaxDD=float((eq / peak - 1).min()),
    )
    if pos is not None:
        pos = np.asarray(pos)
        trades, wr = trade_stats(pos, net)
        out.update(Trades=trades, Exposure=float(pos.mean()), WinRate=wr)
    return out
