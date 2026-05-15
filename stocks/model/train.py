"""
Train XGBoost models for the stocks universe.

Workflow:
  1. Fetch SPY regime series (200d SMA) — used to gate entries.
  2. For each ticker: walk-forward train + split backtest (selection vs evaluation).
  3. Filter: only keep models whose SELECTION-period Sharpe >= threshold AND
     after-tax net return > 0. The evaluation period is then the honest OOS
     estimate of what live performance should look like.

Run: python -m stocks.model.train
"""
import os
import json
from loguru import logger

from common.training import train_symbol
from common.backtest import backtest_symbol
from common.regime import stocks_regime
from stocks.data.fetcher import DEFAULT_TICKERS, load_ticker, TARGET_HORIZON

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "results", "stocks", "models")
REPORT_DIR = os.path.join(BASE_DIR, "results", "stocks", "reports")
UNIVERSE_FILE = os.path.join(BASE_DIR, "results", "stocks", "active_universe.json")

# Alpaca is commission-free; slippage handled in run_backtest
STOCKS_FEE_RATE = 0.0

# Universe filter: drop names whose SELECTION-period results don't meet the bar.
MIN_SELECTION_SHARPE = 0.30
MIN_SELECTION_NET_RETURN = 0.0


def train_all(tickers: list[str] | None = None) -> None:
    tickers = tickers or DEFAULT_TICKERS
    logger.info("Fetching SPY regime series for entry gating")
    regime = stocks_regime(period="5y")

    surviving: list[str] = []
    for ticker in tickers:
        try:
            df = load_ticker(ticker)
        except FileNotFoundError:
            logger.warning(f"No data for {ticker}, skipping")
            continue
        train_symbol(ticker, df, MODEL_DIR, REPORT_DIR)
        bt = backtest_symbol(
            ticker, df, REPORT_DIR,
            fees=STOCKS_FEE_RATE,
            max_hold=TARGET_HORIZON,
            stop_loss=0.03,
            regime=regime,
        )
        if bt.selection.sharpe >= MIN_SELECTION_SHARPE and bt.selection.after_tax_return > MIN_SELECTION_NET_RETURN:
            surviving.append(ticker)
            logger.info(f"{ticker} PASSES filter (sel sharpe={bt.selection.sharpe:.2f}, net={bt.selection.after_tax_return:.2%})")
        else:
            logger.info(f"{ticker} FAILS filter (sel sharpe={bt.selection.sharpe:.2f}, net={bt.selection.after_tax_return:.2%})")

    os.makedirs(os.path.dirname(UNIVERSE_FILE), exist_ok=True)
    with open(UNIVERSE_FILE, "w") as f:
        json.dump({"tickers": surviving}, f, indent=2)
    logger.info(f"Active universe ({len(surviving)} names): {surviving}")


if __name__ == "__main__":
    train_all()
