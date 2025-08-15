import os
import pandas as pd

# Define paths
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))  # Finsearch/
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
SPLIT_DIR = os.path.join(ROOT_DIR, "data", "split")

# Create split folder if it doesn't exist
os.makedirs(SPLIT_DIR, exist_ok=True)

def split_data(file_path, train_ratio=0.8, val_ratio=0.1):
    """Split data into train, validation, and test CSV files."""
    df = pd.read_csv(file_path)
    total_len = len(df)

    train_end = int(total_len * train_ratio)
    val_end = int(total_len * (train_ratio + val_ratio))

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    return df_train, df_val, df_test

def main():
    print(f"Looking for processed files in: {PROCESSED_DIR}")
    files = [f for f in os.listdir(PROCESSED_DIR) if f.endswith("_state.csv")]

    if not files:
        print("❌ No processed files found. Please run 02_preprocessing.py first.")
        return

    for file_name in files:
        # Remove "_state" so output filenames match what rl_agent.py expects
        ticker = file_name.replace("_state.csv", "")
        file_path = os.path.join(PROCESSED_DIR, file_name)

        train_df, val_df, test_df = split_data(file_path)

        train_df.to_csv(os.path.join(SPLIT_DIR, f"{ticker}_train.csv"), index=False)
        val_df.to_csv(os.path.join(SPLIT_DIR, f"{ticker}_val.csv"), index=False)
        test_df.to_csv(os.path.join(SPLIT_DIR, f"{ticker}_test.csv"), index=False)

        print(f"✅ Split {ticker}: train {len(train_df)} rows, val {len(val_df)}, test {len(test_df)}")

    print("\n🎯 Data splitting complete. Files saved in:", SPLIT_DIR)

if __name__ == "__main__":
    main()
