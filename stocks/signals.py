"""
Generate today's trading signals from the latest data and trained models.
Gated by:
  - Active universe (only tickers that passed selection-period filter)
  - Broad-market regime (SPY > SPY 200d SMA)
  - Probability threshold (default 0.60)
"""
import os
import json
from dataclasses import dataclass

import pandas as pd
from loguru import logger

from common.features import add_features, feature_columns
from common.training import load_model
from common.regime import stocks_regime
from stocks.data.fetcher import fetch_ticker, DEFAULT_TICKERS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "results", "stocks", "models")
UNIVERSE_FILE = os.path.join(BASE_DIR, "results", "stocks", "active_universe.json")
SIGNAL_THRESHOLD = 0.60


@dataclass
class Signal:
    symbol: str
    proba: float
    last_close: float
    enter: bool


def active_universe() -> list[str]:
    if os.path.exists(UNIVERSE_FILE):
        return json.load(open(UNIVERSE_FILE)).get("tickers", [])
    logger.warning("No active_universe.json — falling back to full default universe")
    return list(DEFAULT_TICKERS)


def signal_for(symbol: str, regime_ok: bool, lookback_period: str = "200d") -> Signal | None:
    df = fetch_ticker(symbol, period=lookback_period)
    if df.empty:
        return None
    df = add_features(df).dropna(subset=feature_columns())
    if df.empty:
        return None
    model_path = os.path.join(MODEL_DIR, f"{symbol}.joblib")
    if not os.path.exists(model_path):
        logger.warning(f"No trained model for {symbol}")
        return None

    model = load_model(model_path)
    latest = df.iloc[[-1]]
    proba = float(model.predict_proba(latest[feature_columns()])[0, 1])
    enter = (proba >= SIGNAL_THRESHOLD) and regime_ok
    return Signal(
        symbol=symbol,
        proba=proba,
        last_close=float(latest["close"].iloc[0]),
        enter=enter,
    )


def signals_for_universe(tickers: list[str] | None = None) -> list[Signal]:
    tickers = tickers or active_universe()

    regime = stocks_regime(period="2y")
    regime_ok = bool(regime.iloc[-1]) if not regime.empty else False
    logger.info(f"Stocks regime: {'RISK-ON' if regime_ok else 'RISK-OFF (no entries)'}")

    out: list[Signal] = []
    for t in tickers:
        s = signal_for(t, regime_ok=regime_ok)
        if s is None:
            continue
        out.append(s)
        logger.info(f"{t}: proba={s.proba:.3f} {'ENTER' if s.enter else 'skip'} (close={s.last_close:.2f})")
    return out
