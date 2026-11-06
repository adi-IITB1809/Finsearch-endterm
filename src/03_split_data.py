import os, json
import pandas as pd
from config import *
from common import read_tickers, safe_name


def main():
    used, rows = [], []
    for sym in read_tickers():
        path = os.path.join(PROC_DIR, safe_name(sym) + ".csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        tr = df[df.index <= TRAIN_END]
        va = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
        te = df[df.index > VAL_END]
        if len(tr) < MIN_TRAIN_ROWS or len(te) < MIN_TEST_ROWS:
            print(f"  skip {sym}: train={len(tr)} test={len(te)}")
            continue
        mean = tr[FEATURES].mean()
        std = tr[FEATURES].std().replace(0, 1.0).fillna(1.0)       # TRAIN-ONLY statistics
        with open(os.path.join(SPLIT_DIR, safe_name(sym) + "_scaler.json"), "w") as f:
            json.dump({"mean": mean.to_dict(), "std": std.to_dict()}, f)
        used.append(sym)
        rows.append(dict(Ticker=sym, Train=len(tr), Val=len(va), Test=len(te),
                         First=df.index[0].date(), Last=df.index[-1].date()))
    with open(os.path.join(SPLIT_DIR, "tickers_used.txt"), "w") as f:
        f.write("\n".join(used))
    pd.DataFrame(rows).to_csv(os.path.join(SPLIT_DIR, "split_summary.csv"), index=False)
    print(f"\nSTOCKS USED: {len(used)}   <-- this is the number to put on your resume")
    print(f"Train <= {TRAIN_END} | Val <= {VAL_END} | Test > {VAL_END}")


if __name__ == "__main__":
    main()
