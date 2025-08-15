import os
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------- Utilities ----------
def _find_price_col(df: pd.DataFrame) -> str:
    candidates = ["Close", "Adj Close", "Adj_Close", "close", "adj_close"]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError("No price column found in dataframe. Expected one of "
                     "['Close','Adj Close','Adj_Close','close','adj_close'].")

def _load_split_csv(split_dir: str, ticker: str, split: str) -> pd.DataFrame:
    path = os.path.join(split_dir, f"{ticker}_{split}.csv")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found.")
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)
    return df

def _numericize(df: pd.DataFrame) -> pd.DataFrame:
    # Drop non-numeric columns except Date (kept only for slicing, not used in features)
    num = df.select_dtypes(include=[np.number]).copy()
    num = num.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    num = num.dropna().reset_index(drop=True)
    return num


# ---------- Dataset ----------
class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------- Model ----------
class PriceLSTM(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)      # predict next-day return
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last).squeeze(1)  # [B]


# ---------- Training & Inference ----------
def _build_sequences(arr: np.ndarray, targets: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(len(arr) - window):
        X.append(arr[i:i+window])
        y.append(targets[i+window])
    return np.array(X), np.array(y)

def _standardize(train_arr: np.ndarray, val_arr: np.ndarray, test_arr: np.ndarray):
    # train_arr: [N, F]; per-feature standardization
    mean = train_arr.mean(axis=0, keepdims=True)
    std = train_arr.std(axis=0, keepdims=True)
    std[std == 0.0] = 1.0
    return (train_arr - mean) / std, (val_arr - mean) / std, (test_arr - mean) / std, (mean, std)

@torch.no_grad()
def lstm_predict_series(model: PriceLSTM, series_3d: np.ndarray, device: str = "cpu") -> np.ndarray:
    """
    series_3d: [N, T, F] sequences already standardized
    Returns next-day return predictions aligned to each sequence end: [N]
    """
    model.eval()
    tensor = torch.from_numpy(series_3d.astype(np.float32)).to(device)
    preds = model(tensor).cpu().numpy()
    return preds


def train_lstm_for_ticker(
    split_dir: str,
    ticker: str,
    window: int = 20,
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str | None = None
) -> Dict[str, Any]:
    """
    Trains an LSTM to predict next-day return (based on price_col) from numeric features.
    Returns a dict with model, scalers, feature metadata, etc.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load splits
    df_train = _load_split_csv(split_dir, ticker, "train")
    df_val   = _load_split_csv(split_dir, ticker, "val")

    # Identify price column & numeric features
    price_col = _find_price_col(df_train)
    num_train = _numericize(df_train)
    num_val   = _numericize(df_val)
    feat_cols = [c for c in num_train.columns]  # include price_col; model learns returns implicitly

    # Targets: next-day return from price_col
    def price_ret(d: pd.DataFrame):
        p = d[price_col].astype(float).values
        # safe division
        p0 = p[:-1]
        p1 = p[1:]
        r = np.zeros_like(p)
        r[1:] = np.where(p0 == 0.0, 0.0, (p1 / p0) - 1.0)
        return r

    y_train_full = price_ret(df_train)
    y_val_full   = price_ret(df_val)

    # Align numeric frames with targets length
    # After return calc, the first return is 0; we keep lengths consistent
    X_train_mat = num_train[feat_cols].values
    X_val_mat   = num_val[feat_cols].values

    # Standardize by train stats
    X_train_std, X_val_std, _, (mu, sigma) = _standardize(X_train_mat, X_val_mat, X_val_mat)

    # Build sequences
    Xtr, ytr = _build_sequences(X_train_std, y_train_full, window)
    Xva, yva = _build_sequences(X_val_std,   y_val_full,   window)

    # Dataloaders
    train_ds = SeqDataset(Xtr, ytr)
    val_ds   = SeqDataset(Xva, yva)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    # Model
    n_features = Xtr.shape[-1]
    model = PriceLSTM(n_features=n_features, hidden_size=64, num_layers=1, dropout=0.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = np.inf
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * len(xb)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * len(xb)

        tr_loss /= max(1, len(train_ds))
        va_loss /= max(1, len(val_ds))
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "model": model,
        "device": device,
        "feat_cols": feat_cols,
        "price_col": price_col,
        "window": window,
        "mu": mu,          # (1, F)
        "sigma": sigma,    # (1, F)
    }


def lstm_strategy_equity(
    model_bundle: Dict[str, Any],
    df_test: pd.DataFrame,
    last_n_days: int = 30
) -> Dict[str, Any]:
    """
    Builds equity curve for a simple long/flat strategy:
      - Predict next-day return; if pred > 0 -> long (1), else flat (0)
    Uses only last_n_days of the test set for metrics.
    """
    # unpack
    model = model_bundle["model"]
    device = model_bundle["device"]
    feat_cols = model_bundle["feat_cols"]
    price_col = model_bundle["price_col"]
    window = model_bundle["window"]
    mu = model_bundle["mu"]
    sigma = model_bundle["sigma"]

    # Prepare numeric features
    df = df_test.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").reset_index(drop=True)

    num = _numericize(df)
    # Ensure feature presence alignment (if test lost columns, intersect)
    cols = [c for c in feat_cols if c in num.columns]
    num = num[cols].copy()
    # Standardize using train stats
    X = num.values
    X_std = (X - mu[:, :len(cols)]) / sigma[:, :len(cols)]

    # Price returns for PnL
    p = df[price_col].astype(float).values
    p0 = p[:-1]
    p1 = p[1:]
    ret = np.zeros_like(p, dtype=np.float64)
    ret[1:] = np.where(p0 == 0.0, 0.0, (p1 / p0) - 1.0)

    # Build sequences for predictions (aligned to ret index)
    X_seq, _ = _build_sequences(X_std, ret, window)  # target unused here
    preds = lstm_predict_series(model, X_seq, device)

    # Align preds to original timeline
    # seq ending at index t uses data [t-window+1 ... t]; it predicts return at t+1
    # We map decision at t to position for next step
    T = len(p)
    pos = np.zeros(T, dtype=int)
    for t in range(window-1, T-1):
        seq_idx = t - (window - 1)
        pred = preds[seq_idx] if 0 <= seq_idx < len(preds) else 0.0
        pos[t] = 1 if pred > 0.0 else 0
    pos[-1] = pos[-2] if T > 1 else 0

    # Equity compounding
    equity = np.ones(T, dtype=np.float64)
    for t in range(T - 1):
        equity[t+1] = equity[t] * (1.0 + pos[t] * ret[t+1])

    # Take only the last_n_days slice for metrics
    N = min(last_n_days, T)
    sl = slice(T - N, T)
    return {
        "equity_curve": equity,
        "positions": pos,
        "returns": ret,
        "slice": sl,
    }
