"""
Stocks paper-trading bot.

Daily cycle:
  1. Skip if market is closed.
  2. Generate signals for the universe using trained models.
  3. For each new BUY signal: size position by RISK_PER_TRADE and submit a
     bracket order (stop-loss + take-profit). Skip if we already hold it or
     have hit MAX_OPEN_POSITIONS.
  4. Log every entry decision to results/stocks/trades.csv.

Runs as a daemon: schedules itself daily at 09:35 ET (5 min after open).
For manual testing: `python -m stocks.bot --once`.
"""
import os
import sys
import csv
import time
import argparse
from datetime import datetime, timezone

import schedule
from loguru import logger

from common.config import (
    RISK_PER_TRADE,
    MAX_OPEN_POSITIONS,
    STOP_LOSS_PCT,
    STOCKS_RESULTS_DIR,
)
from common.logger import setup_logger
from stocks.signals import signals_for_universe
from stocks.broker.alpaca_client import AlpacaBroker

setup_logger("stocks", STOCKS_RESULTS_DIR)

TRADE_LOG = os.path.join(STOCKS_RESULTS_DIR, "trades.csv")
TAKE_PROFIT_PCT = 0.03  # ~3% above entry; ties to our 3-day +1% target


def _ensure_trade_log() -> None:
    if os.path.exists(TRADE_LOG):
        return
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    with open(TRADE_LOG, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp_utc", "symbol", "proba", "entry_price",
            "qty", "stop_loss_pct", "take_profit_pct", "order_id", "note",
        ])


def _log_trade(symbol: str, proba: float, price: float, qty: int, order_id: str, note: str) -> None:
    _ensure_trade_log()
    with open(TRADE_LOG, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now(timezone.utc).isoformat(),
            symbol, f"{proba:.4f}", f"{price:.2f}", qty,
            f"{STOP_LOSS_PCT:.4f}", f"{TAKE_PROFIT_PCT:.4f}", order_id, note,
        ])


def _position_size(portfolio_value: float, entry_price: float) -> int:
    """
    Position size is the minimum of:
      - Risk-based: lose at most RISK_PER_TRADE of portfolio at the stop
      - Allocation-based: at most 1/MAX_OPEN_POSITIONS of portfolio notional
    so we never over-concentrate even if stops are tight.
    """
    if entry_price <= 0:
        return 0
    risk_dollars = portfolio_value * RISK_PER_TRADE
    per_share_risk = entry_price * STOP_LOSS_PCT
    risk_based_qty = int(risk_dollars / per_share_risk) if per_share_risk > 0 else 0
    alloc_cap_qty = int((portfolio_value / MAX_OPEN_POSITIONS) / entry_price)
    return max(min(risk_based_qty, alloc_cap_qty), 0)


def run_cycle() -> None:
    logger.info("=== Stocks trading cycle ===")
    broker = AlpacaBroker()

    if not broker.is_market_open():
        logger.info("Market closed, skipping cycle")
        return

    account = broker.account()
    held = broker.positions()
    logger.info(f"Portfolio ${account.portfolio_value:,.2f} | cash ${account.cash:,.2f} | open positions: {len(held)}")

    signals = signals_for_universe()
    enters = [s for s in signals if s.enter and s.symbol not in held]

    slots_left = max(MAX_OPEN_POSITIONS - len(held), 0)
    enters = sorted(enters, key=lambda s: s.proba, reverse=True)[:slots_left]
    if not enters:
        logger.info("No new entries this cycle")
        return

    for sig in enters:
        qty = _position_size(account.portfolio_value, sig.last_close)
        if qty <= 0:
            logger.warning(f"{sig.symbol}: position size 0, skipping")
            continue
        try:
            order_id = broker.submit_bracket_buy(
                sig.symbol, qty,
                stop_loss_pct=STOP_LOSS_PCT,
                take_profit_pct=TAKE_PROFIT_PCT,
            )
            _log_trade(sig.symbol, sig.proba, sig.last_close, qty, order_id, "entry")
        except Exception as e:
            logger.error(f"Order failed for {sig.symbol}: {e}")
            _log_trade(sig.symbol, sig.proba, sig.last_close, qty, "", f"error: {e}")

    logger.info("Cycle complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    if args.once:
        run_cycle()
        return

    logger.info("Stocks bot starting (scheduled mode)")
    schedule.every().day.at("09:35").do(run_cycle)  # 5 min after NYSE open
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
