"""
Live signal generation using the pooled cross-sectional model.

Per cycle:
  1. Fetch latest market context (SPY, VIX, sectors).
  2. For each ticker in the universe: fetch ~200 days, compute features
     (per-stock + cross-asset), predict with the pooled model.
  3. Gate by SPY regime + probability threshold.
"""
import os
import json
from dataclasses import dataclass

from loguru import logger

from common.features import add_features, feature_columns_in
from common.market_features import fetch_market_context, add_market_features
from common.cross_section import load_pooled
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


def signals_for_universe(tickers: list[str] | None = None) -> list[Signal]:
    tickers = tickers or active_universe()
    model, feature_cols = load_pooled(MODEL_DIR)

    regime = stocks_regime(period="2y")
    regime_ok = bool(regime.iloc[-1]) if not regime.empty else False
    logger.info(f"Stocks regime: {'RISK-ON' if regime_ok else 'RISK-OFF (no entries)'}")

    ctx = fetch_market_context(period="1y")
    out: list[Signal] = []
    for t in tickers:
        df = fetch_ticker(t, period="1y")
        if df.empty:
            continue
        df = add_features(df)
        df = add_market_features(df, t, ctx)
        df = df.dropna(subset=feature_cols)
        if df.empty:
            logger.warning(f"{t}: no rows after feature dropna")
            continue
        latest = df.iloc[[-1]]
        proba = float(model.predict_proba(latest[feature_cols])[0, 1])
        enter = (proba >= SIGNAL_THRESHOLD) and regime_ok
        out.append(Signal(
            symbol=t,
            proba=proba,
            last_close=float(latest["close"].iloc[0]),
            enter=enter,
        ))
        logger.info(f"{t}: proba={proba:.3f} {'ENTER' if enter else 'skip'} (close={latest['close'].iloc[0]:.2f})")
    return out
