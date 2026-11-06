import os
import numpy as np
import torch
import torch.nn as nn
from config import *
from common import load_dataset

torch.set_num_threads(max(1, os.cpu_count() // 2))


class PriceLSTM(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.head(o[:, -1]).squeeze(-1)


def make_sequences(data, split, window=LSTM_WINDOW):
    Xs, ys = [], []
    for d in data.values():
        if split not in d["bounds"]:
            continue
        a, b = d["bounds"][split]
        Xs.append(np.stack([d["X"][i - window + 1: i + 1] for i in range(a, b)]))
        ys.append(d["ret"][a:b] * 100)                    # next-day return in %
    return np.concatenate(Xs).astype(np.float32), np.clip(np.concatenate(ys), -10, 10).astype(np.float32)


def train(epochs=25, patience=5, bs=256):
    torch.manual_seed(SEED); np.random.seed(SEED)
    data = load_dataset()
    Xtr, ytr = make_sequences(data, "train")
    Xva, yva = make_sequences(data, "val")
    Xtr, ytr, Xva, yva = map(torch.from_numpy, (Xtr, ytr, Xva, yva))
    model = PriceLSTM(Xtr.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.HuberLoss()
    best, best_state, bad = np.inf, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss_fn(model(Xtr[idx]), ytr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(Xva), yva))
        print(f"epoch {ep:02d}  val loss {vl:.5f}")
        if vl < best:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print("early stop"); break
    model.load_state_dict(best_state)
    torch.save({"state": model.state_dict(), "n_feat": Xtr.shape[-1]}, os.path.join(MODELS_DIR, "lstm.pt"))
    print("Saved models/lstm.pt")


def load_model():
    ck = torch.load(os.path.join(MODELS_DIR, "lstm.pt"), map_location="cpu")
    m = PriceLSTM(ck["n_feat"]); m.load_state_dict(ck["state"]); m.eval()
    return m


@torch.no_grad()
def lstm_positions(model, d, split, window=LSTM_WINDOW):
    a, b = d["bounds"][split]
    X = np.stack([d["X"][i - window + 1: i + 1] for i in range(a, b)]).astype(np.float32)
    pred = model(torch.from_numpy(X)).numpy()
    return (pred > 0).astype(int)                         # long if predicted next-day return > 0


if __name__ == "__main__":
    train()
