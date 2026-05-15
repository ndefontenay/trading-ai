"""
Generate today's trading signals from the latest data and trained models.
"""
import os
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from common.features import add_features, feature_columns
from common.training import load_model
from stocks.data.fetcher import fetch_ticker, DEFAULT_TICKERS, TARGET_HORIZON

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "stocks", "models")
SIGNAL_THRESHOLD = 0.60


@dataclass
class Signal:
    symbol: str
    proba: float
    last_close: float
    enter: bool


def signal_for(symbol: str, lookback_period: str = "200d") -> Signal | None:
    """
    Pull the latest ~200 trading days, compute features, ask the model for
    today's probability of a +1% move in the next 3 days.
    """
    df = fetch_ticker(symbol, period=lookback_period)
    if df.empty:
        logger.warning(f"No fresh data for {symbol}")
        return None
    df = add_features(df).dropna(subset=feature_columns())
    if df.empty:
        logger.warning(f"Feature window too short for {symbol}")
        return None

    model_path = os.path.join(MODEL_DIR, f"{symbol}.joblib")
    if not os.path.exists(model_path):
        logger.warning(f"No trained model for {symbol} at {model_path}")
        return None
    model = load_model(model_path)

    latest = df.iloc[[-1]]
    proba = float(model.predict_proba(latest[feature_columns()])[0, 1])
    return Signal(
        symbol=symbol,
        proba=proba,
        last_close=float(latest["close"].iloc[0]),
        enter=proba >= SIGNAL_THRESHOLD,
    )


def signals_for_universe(tickers: list[str] | None = None) -> list[Signal]:
    tickers = tickers or DEFAULT_TICKERS
    out: list[Signal] = []
    for t in tickers:
        s = signal_for(t)
        if s is not None:
            out.append(s)
            logger.info(f"{t}: proba={s.proba:.3f} {'ENTER' if s.enter else 'skip'} (close={s.last_close:.2f})")
    return out
