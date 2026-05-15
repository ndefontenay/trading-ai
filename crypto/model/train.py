"""
Train XGBoost models for the crypto universe.

Run: python -m crypto.model.train
"""
import os
from loguru import logger

from common.training import train_symbol
from common.backtest import backtest_symbol
from crypto.data.fetcher import DEFAULT_PAIRS, load_pair

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "crypto", "models")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "crypto", "reports")

# Coinbase taker fee ≈ 0.4% — realistic for a small account
CRYPTO_FEE_RATE = 0.004


def train_all(pairs: list[str] | None = None) -> None:
    pairs = pairs or DEFAULT_PAIRS
    for pair in pairs:
        try:
            df = load_pair(pair)
        except FileNotFoundError:
            logger.warning(f"No data for {pair}, skipping")
            continue
        train_symbol(pair, df, MODEL_DIR, REPORT_DIR)
        backtest_symbol(pair, df, REPORT_DIR, fees=CRYPTO_FEE_RATE)


if __name__ == "__main__":
    train_all()
