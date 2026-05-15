"""
Cross-asset context features — what the broader market is doing.

Adding SPY/VIX/sector-ETF features turns each row from "what does AAPL look
like in isolation" into "what does AAPL look like inside today's market
environment." Most common-mode variance in individual-stock returns comes
from market and sector factors; ignoring them leaves edge on the table.
"""
import pandas as pd
import yfinance as yf
from loguru import logger


# Stock → its primary sector ETF (Select Sector SPDRs).
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "NVDA": "XLK",
    "GOOGL": "XLC", "META": "XLC",
    "AMZN": "XLY", "TSLA": "XLY",
    "WMT": "XLP",
    "V": "XLF", "JPM": "XLF",
}


def fetch_market_context(period: str = "5y") -> dict[str, pd.DataFrame]:
    """Pre-fetch SPY, VIX, and all unique sector ETFs in SECTOR_MAP."""
    tickers = ["SPY", "^VIX"] + sorted(set(SECTOR_MAP.values()))
    ctx: dict[str, pd.DataFrame] = {}
    for t in tickers:
        logger.info(f"Fetching market context: {t}")
        df = yf.download(t, period=period, interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            logger.warning(f"No data for {t}")
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        ctx[t] = df
    return ctx


def add_market_features(df: pd.DataFrame, symbol: str, ctx: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Append market-environment features to a per-stock DataFrame.
    Requires `return_5d` to already be present (from add_features).
    """
    df = df.copy()
    spy = ctx.get("SPY")
    vix = ctx.get("^VIX")
    sec_ticker = SECTOR_MAP.get(symbol, "SPY")
    sec = ctx.get(sec_ticker, spy)

    if spy is not None:
        spy_close = spy["close"].reindex(df.index, method="ffill")
        df["spy_return_1d"] = spy_close.pct_change(1)
        df["spy_return_5d"] = spy_close.pct_change(5)
        df["spy_return_20d"] = spy_close.pct_change(20)
        if "return_5d" in df.columns:
            df["rs_vs_spy_5d"] = df["return_5d"] - df["spy_return_5d"]

    if vix is not None:
        vix_close = vix["close"].reindex(df.index, method="ffill")
        df["vix_level"] = vix_close
        df["vix_change_5d"] = vix_close.pct_change(5)

    if sec is not None:
        sec_close = sec["close"].reindex(df.index, method="ffill")
        df["sector_return_5d"] = sec_close.pct_change(5)
        df["sector_return_20d"] = sec_close.pct_change(20)
        if "return_5d" in df.columns:
            df["rs_vs_sector_5d"] = df["return_5d"] - df["sector_return_5d"]

    return df


def market_feature_columns() -> list[str]:
    return [
        "spy_return_1d", "spy_return_5d", "spy_return_20d",
        "vix_level", "vix_change_5d",
        "sector_return_5d", "sector_return_20d",
        "rs_vs_spy_5d", "rs_vs_sector_5d",
    ]
