"""
Train XGBoost models for the crypto universe.

Workflow mirrors the stocks bot:
  1. Fetch BTC regime series (200d SMA) — used to gate all crypto entries.
  2. Walk-forward train + split backtest (selection vs evaluation) per pair.
  3. Filter universe by selection-period Sharpe and after-tax net return.

Run: python -m crypto.model.train
"""
import os
import json
from loguru import logger

from common.training import train_symbol
from common.backtest import backtest_symbol
from common.regime import crypto_regime
from crypto.data.fetcher import DEFAULT_PAIRS, load_pair, TARGET_HORIZON

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "results", "crypto", "models")
REPORT_DIR = os.path.join(BASE_DIR, "results", "crypto", "reports")
UNIVERSE_FILE = os.path.join(BASE_DIR, "results", "crypto", "active_universe.json")

# Kraken Pro taker fee ≈ 0.16%
CRYPTO_FEE_RATE = 0.0016

MIN_SELECTION_SHARPE = 0.30
MIN_SELECTION_NET_RETURN = 0.0


def train_all(pairs: list[str] | None = None) -> None:
    pairs = pairs or DEFAULT_PAIRS
    logger.info("Fetching BTC regime series for entry gating")
    regime = crypto_regime(days=365 * 2)

    surviving: list[str] = []
    for pair in pairs:
        try:
            df = load_pair(pair)
        except FileNotFoundError:
            logger.warning(f"No data for {pair}, skipping")
            continue
        train_symbol(pair, df, MODEL_DIR, REPORT_DIR)
        bt = backtest_symbol(
            pair, df, REPORT_DIR,
            fees=CRYPTO_FEE_RATE,
            max_hold=TARGET_HORIZON,
            stop_loss=0.08,
            regime=regime,
        )
        if bt.selection.sharpe >= MIN_SELECTION_SHARPE and bt.selection.after_tax_return > MIN_SELECTION_NET_RETURN:
            surviving.append(pair)
            logger.info(f"{pair} PASSES filter (sel sharpe={bt.selection.sharpe:.2f}, net={bt.selection.after_tax_return:.2%})")
        else:
            logger.info(f"{pair} FAILS filter (sel sharpe={bt.selection.sharpe:.2f}, net={bt.selection.after_tax_return:.2%})")

    os.makedirs(os.path.dirname(UNIVERSE_FILE), exist_ok=True)
    with open(UNIVERSE_FILE, "w") as f:
        json.dump({"pairs": surviving}, f, indent=2)
    logger.info(f"Active universe ({len(surviving)} pairs): {surviving}")


if __name__ == "__main__":
    train_all()
