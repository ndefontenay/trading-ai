"""
Crypto data fetcher — pulls historical OHLCV from Coinbase Advanced.

Public candle endpoints don't require auth, so we can fetch data without keys.
Auth keys are only needed in Phase 3 when we go live; paper trading is simulated
locally because Coinbase deprecated their sandbox.
"""
import os
import time
import pandas as pd
import requests
from loguru import logger
from common.features import add_features
from common.storage import save_parquet, load_parquet

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "crypto", "data")

# Starter universe — major liquid pairs on Coinbase
DEFAULT_PAIRS = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD"]

# Coinbase public market data endpoint (Exchange/legacy public API, no auth needed)
COINBASE_PUBLIC_URL = "https://api.exchange.coinbase.com"

# Granularity in seconds
GRANULARITY_DAILY = 86400
MAX_CANDLES_PER_REQUEST = 300  # Coinbase limit


def fetch_pair(symbol: str, days: int = 365 * 5, granularity: int = GRANULARITY_DAILY) -> pd.DataFrame:
    """
    Fetch OHLCV candles for one pair.
    Coinbase returns max 300 candles per request, so we paginate backwards.
    """
    logger.info(f"Fetching {symbol} ({days} days, granularity={granularity}s)")
    end = pd.Timestamp.utcnow().floor("D")
    start_target = end - pd.Timedelta(days=days)

    all_candles: list = []
    cursor_end = end
    step = pd.Timedelta(seconds=granularity * MAX_CANDLES_PER_REQUEST)

    while cursor_end > start_target:
        cursor_start = max(cursor_end - step, start_target)
        params = {
            "start": cursor_start.isoformat(),
            "end": cursor_end.isoformat(),
            "granularity": granularity,
        }
        resp = requests.get(f"{COINBASE_PUBLIC_URL}/products/{symbol}/candles", params=params, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Coinbase error for {symbol}: {resp.status_code} {resp.text}")
            break
        batch = resp.json()
        if not batch:
            break
        all_candles.extend(batch)
        cursor_end = cursor_start
        time.sleep(0.25)  # be polite to public endpoint

    if not all_candles:
        logger.warning(f"No candles for {symbol}")
        return pd.DataFrame()

    # Coinbase format: [time, low, high, open, close, volume]
    df = pd.DataFrame(all_candles, columns=["time", "low", "high", "open", "close", "volume"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df[["open", "high", "low", "close", "volume"]]


def fetch_and_store(symbol: str, days: int = 365 * 5) -> pd.DataFrame:
    """Fetch a pair, add features, save to Parquet."""
    df = fetch_pair(symbol, days=days)
    if df.empty:
        return df
    df = add_features(df)
    path = os.path.join(DATA_DIR, f"{symbol}.parquet")
    save_parquet(df, path)
    logger.info(f"Saved {symbol}: {len(df)} rows -> {path}")
    return df


def load_pair(symbol: str) -> pd.DataFrame:
    return load_parquet(os.path.join(DATA_DIR, f"{symbol}.parquet"))


def fetch_universe(pairs: list[str] | None = None, days: int = 365 * 5) -> dict[str, pd.DataFrame]:
    pairs = pairs or DEFAULT_PAIRS
    return {p: fetch_and_store(p, days=days) for p in pairs}


if __name__ == "__main__":
    fetch_universe()
