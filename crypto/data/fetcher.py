"""
Crypto data fetcher — pulls historical OHLCV from Kraken public API.

Kraken Pro has 0.16% taker fees (vs Coinbase 0.4%) and supports paper trading
via their demo environment. Public OHLC endpoint requires no auth.
"""
import os
import time
import pandas as pd
import requests
from loguru import logger
from common.features import add_features, add_target
from common.storage import save_parquet, load_parquet

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "crypto", "data")

# Display name -> Kraken pair code. Kraken uses XBT for Bitcoin and prefixes
# legacy pairs with X/Z. We store data under the friendly display name.
KRAKEN_PAIRS = {
    "BTC-USD": "XBTUSD",
    "ETH-USD": "ETHUSD",
    "SOL-USD": "SOLUSD",
    "XRP-USD": "XRPUSD",
    "ADA-USD": "ADAUSD",
}
DEFAULT_PAIRS = list(KRAKEN_PAIRS.keys())

KRAKEN_PUBLIC_URL = "https://api.kraken.com/0/public/OHLC"
INTERVAL_DAILY_MIN = 1440  # daily in minutes

# Target: predict whether close in 10 days will be >4% higher.
# Kraken Pro fees ≈ 0.16% × 2 = 0.32% round-trip, so 4% target leaves
# plenty of room. Longer horizon also cuts trade frequency further.
TARGET_HORIZON = 10
TARGET_THRESHOLD = 0.04


def fetch_pair(symbol: str, days: int = 365 * 5, interval: int = INTERVAL_DAILY_MIN) -> pd.DataFrame:
    """
    Fetch OHLCV candles for one pair via Kraken's public OHLC endpoint.

    Kraken's `since` param is a timestamp; the response returns up to 720
    candles from that point forward. For daily data, 720 days = ~2 years,
    so we paginate to cover the full lookback.
    """
    logger.info(f"Fetching {symbol} ({days} days, interval={interval}min)")
    kraken_pair = KRAKEN_PAIRS.get(symbol, symbol)
    seconds_per_candle = interval * 60

    end_ts = int(pd.Timestamp.utcnow().timestamp())
    since = end_ts - days * 86400

    all_rows: list = []
    cursor = since
    pair_key: str | None = None

    while cursor < end_ts:
        resp = requests.get(KRAKEN_PUBLIC_URL, params={"pair": kraken_pair, "interval": interval, "since": cursor}, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Kraken HTTP {resp.status_code} for {symbol}: {resp.text}")
            break
        payload = resp.json()
        if payload.get("error"):
            logger.error(f"Kraken error for {symbol}: {payload['error']}")
            break

        result = payload["result"]
        # Kraken returns the pair under its canonical key (e.g. XXBTZUSD for XBTUSD)
        if pair_key is None:
            pair_key = next(k for k in result.keys() if k != "last")
        rows = result.get(pair_key, [])
        if not rows:
            break

        all_rows.extend(rows)
        new_cursor = int(result["last"])
        if new_cursor <= cursor:
            break
        cursor = new_cursor
        time.sleep(0.5)  # Kraken rate limits public endpoint

    if not all_rows:
        logger.warning(f"No candles for {symbol}")
        return pd.DataFrame()

    # Kraken row format: [time, open, high, low, close, vwap, volume, count]
    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s")
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    return df[["open", "high", "low", "close", "volume"]]


def fetch_and_store(symbol: str, days: int = 365 * 5) -> pd.DataFrame:
    """Fetch a pair, add features and target, save to Parquet."""
    df = fetch_pair(symbol, days=days)
    if df.empty:
        return df
    df = add_features(df)
    df = add_target(df, horizon=TARGET_HORIZON, threshold=TARGET_THRESHOLD)
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
