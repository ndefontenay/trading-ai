"""
Technical indicator feature engineering and target construction.

Indicators are asset-agnostic; the target (what we're predicting) is tuned
per asset class. Stocks see smaller moves and zero commission, so a tight
target works; crypto needs a larger threshold to clear fees.
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
    Returns a copy with feature columns appended (no target yet).
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
    df["bb_pct"] = bb.bollinger_pband()

    # ATR (volatility)
    df["atr_14"] = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()

    # Volume features
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]

    return df


def add_target(df: pd.DataFrame, horizon: int, threshold: float) -> pd.DataFrame:
    """
    Add a binary target: 1 if the forward `horizon`-day return exceeds `threshold`.

    Examples:
      - Stocks: horizon=3, threshold=0.01  (predict +1% over 3 days)
      - Crypto: horizon=5, threshold=0.02  (predict +2% over 5 days)

    Larger threshold → fewer positive examples, but each one is a meaningful
    move that can overcome trading friction. The target is forward-looking,
    so future rows will be NaN; training will drop them.
    """
    df = df.copy()
    forward_return = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target"] = (forward_return > threshold).astype(int)
    # Mark the last `horizon` rows as NaN so they don't enter training
    df.loc[df.index[-horizon:], "target"] = np.nan
    df["forward_return"] = forward_return
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
