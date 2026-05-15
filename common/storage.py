import os
import pandas as pd


def save_parquet(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def save_trades(trades: pd.DataFrame, results_dir: str, filename: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    trades.to_csv(os.path.join(results_dir, filename), index=True)
