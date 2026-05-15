"""
Train XGBoost models for the stocks universe.

Run: python -m stocks.model.train
"""
import os
from loguru import logger

from common.training import train_symbol
from common.backtest import backtest_symbol
from stocks.data.fetcher import DEFAULT_TICKERS, load_ticker, TARGET_HORIZON

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "stocks", "models")
REPORT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results", "stocks", "reports")

# Alpaca is commission-free for stocks; slippage only
STOCKS_FEE_RATE = 0.0


def train_all(tickers: list[str] | None = None) -> None:
    tickers = tickers or DEFAULT_TICKERS
    for ticker in tickers:
        try:
            df = load_ticker(ticker)
        except FileNotFoundError:
            logger.warning(f"No data for {ticker}, skipping")
            continue
        train_symbol(ticker, df, MODEL_DIR, REPORT_DIR)
        backtest_symbol(ticker, df, REPORT_DIR, fees=STOCKS_FEE_RATE, max_hold=TARGET_HORIZON, stop_loss=0.05)


if __name__ == "__main__":
    train_all()
