"""
Market regime gate.

For stocks: SPY > SPY's 200-day SMA → "risk-on", trade entries allowed.
For crypto: BTC > BTC's 200-day SMA → "risk-on" for crypto entries.

Single binary signal applied across the whole respective universe — when
the broad market is in a downtrend, the bot stands aside. The backtest
shows our model still issues plenty of "buy" probabilities in bear
windows; those windows are where the edge dies and tax drag stings most.
"""
import pandas as pd
import yfinance as yf
import requests
import time


REGIME_WINDOW = 200


def stocks_regime(period: str = "2y") -> pd.Series:
    """Boolean series indexed by date: True when SPY > SPY's 200d SMA."""
    df = yf.download("SPY", period=period, interval="1d", progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    sma = df["close"].rolling(REGIME_WINDOW).mean()
    return (df["close"] > sma).rename("regime_ok")


def crypto_regime(days: int = 365 * 2) -> pd.Series:
    """Boolean series indexed by date: True when BTC > BTC's 200d SMA."""
    end_ts = int(pd.Timestamp.utcnow().timestamp())
    since = end_ts - days * 86400
    cursor = since
    all_rows: list = []
    pair_key: str | None = None

    while cursor < end_ts:
        resp = requests.get(
            "https://api.kraken.com/0/public/OHLC",
            params={"pair": "XBTUSD", "interval": 1440, "since": cursor},
            timeout=15,
        )
        payload = resp.json()
        if payload.get("error"):
            break
        result = payload["result"]
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
        time.sleep(0.5)

    if not all_rows:
        return pd.Series(dtype=bool, name="regime_ok")

    df = pd.DataFrame(all_rows, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s")
    df["close"] = pd.to_numeric(df["close"])
    df = df.set_index("time").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    sma = df["close"].rolling(REGIME_WINDOW).mean()
    return (df["close"] > sma).rename("regime_ok")
