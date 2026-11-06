import os
from config import *
from common import read_tickers, download_prices, safe_name


def main():
    tickers = read_tickers()
    print(f"Downloading {len(tickers)} tickers, {START_DATE} -> {END_DATE_EXCLUSIVE} (adjusted prices)")
    ok, failed = 0, []
    for t in tickers:
        df = download_prices(t, START_DATE, END_DATE_EXCLUSIVE)
        if df is None or len(df) < 100:
            failed.append(t)
            print("  FAILED/too short:", t)
            continue
        df.to_csv(os.path.join(RAW_DIR, safe_name(t) + ".csv"))
        ok += 1
        print(f"  {t}: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"\nSaved {ok} tickers. Failed: {failed}")


if __name__ == "__main__":
    main()
