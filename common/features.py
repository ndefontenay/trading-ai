"""
Technical indicator feature engineering.

The indicators themselves are asset-agnostic — the same RSI/MACD logic applies
to AAPL or BTC. Keeping this shared avoids drift between the bots' feature
definitions, while each bot still owns its own model, data, and trade results.
"""
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to an OHLCV DataFrame.

    Expects columns: open, high, low, close, volume (lowercase).
    Returns a copy with feature columns appended.
    """
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Returns
    df["return_1d"] = df["close"].pct_change()
    df["return_5d"] = df["close"].pct_change(5)
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

    # Moving averages
    df["sma_10"] = SMAIndicator(df["close"], window=10).sma_indicator()
    df["sma_50"] = SMAIndicator(df["close"], window=50).sma_indicator()
    df["ema_12"] = EMAIndicator(df["close"], window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(df["close"], window=26).ema_indicator()

    # RSI
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()

    # MACD
    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()

    # Bollinger Bands
    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_pct"] = bb.bollinger_pband()  # position within bands [0, 1]

    # ATR (volatility)
    df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # Volume features
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    # Target: next-day return direction (1 = up, 0 = down)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

    return df


def feature_columns() -> list[str]:
    """Columns to feed the model — excludes raw OHLCV and the target."""
    return [
        "return_1d", "return_5d", "log_return_1d",
        "sma_10", "sma_50", "ema_12", "ema_26",
        "rsi_14",
        "macd", "macd_signal", "macd_diff",
        "bb_high", "bb_low", "bb_pct",
        "atr_14",
        "volume_sma_20", "volume_ratio",
    ]
