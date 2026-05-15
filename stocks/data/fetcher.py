"""
Stocks data fetcher — pulls historical OHLCV from Yahoo Finance via yfinance.
"""
import os
import pandas as pd
import yfinance as yf
from loguru import logger
from common.features import add_features, add_target_triple_barrier
from common.storage import save_parquet, load_parquet

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "stocks", "data")

# Starter universe — adjust as needed
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT"]

# Triple-barrier target: within 3 trading days, did price hit +1.5% (win)
# before -3% (stop) or time-out? Matches actual trade execution semantics.
TARGET_HORIZON = 3
TARGET_UPPER = 0.015   # take-profit barrier
TARGET_LOWER = 0.03    # stop-loss barrier (matches STOP_LOSS_PCT)


def fetch_ticker(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Fetch raw OHLCV for one ticker."""
    logger.info(f"Fetching {ticker} ({period}, {interval})")
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    if df.empty:
        logger.warning(f"No data returned for {ticker}")
        return df
    # yfinance returns MultiIndex columns for single tickers in newer versions
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    return df


def fetch_and_store(ticker: str, period: str = "5y") -> pd.DataFrame:
    """Fetch a ticker, add features, save to Parquet, return the DataFrame."""
    df = fetch_ticker(ticker, period=period)
    if df.empty:
        return df
    df = add_features(df)
    df = add_target_triple_barrier(df, horizon=TARGET_HORIZON, upper=TARGET_UPPER, lower=TARGET_LOWER)
    path = os.path.join(DATA_DIR, f"{ticker}.parquet")
    save_parquet(df, path)
    logger.info(f"Saved {ticker}: {len(df)} rows -> {path}")
    return df


def load_ticker(ticker: str) -> pd.DataFrame:
    """Load a previously stored ticker with features."""
    return load_parquet(os.path.join(DATA_DIR, f"{ticker}.parquet"))


def fetch_universe(tickers: list[str] | None = None, period: str = "5y") -> dict[str, pd.DataFrame]:
    """Fetch and store the full universe."""
    tickers = tickers or DEFAULT_TICKERS
    return {t: fetch_and_store(t, period=period) for t in tickers}


if __name__ == "__main__":
    fetch_universe()
