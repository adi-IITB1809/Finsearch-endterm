import os
import pandas as pd
from config import *
from common import build_features


def main():
    n = 0
    for fn in sorted(os.listdir(RAW_DIR)):
        if not fn.endswith(".csv"):
            continue
        raw = pd.read_csv(os.path.join(RAW_DIR, fn), index_col=0, parse_dates=True)
        feat = build_features(raw)
        feat.to_csv(os.path.join(PROC_DIR, fn))
        n += 1
        print(f"{fn}: {len(feat)} rows")
    print(f"\nProcessed {n} files -> {PROC_DIR}  (unscaled; scaling happens in 03 using TRAIN rows only)")


if __name__ == "__main__":
    main()
