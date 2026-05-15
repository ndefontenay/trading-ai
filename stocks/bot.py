"""
Stocks bot entry point.
Runs the daily signal → order cycle for US equities via Alpaca.
"""
import schedule
import time
from loguru import logger
from common.config import STOCKS_RESULTS_DIR
from common.logger import setup_logger

setup_logger("stocks", STOCKS_RESULTS_DIR)


def run_cycle():
    logger.info("Starting stocks trading cycle")
    # Phase 2: fetch latest data, generate signal
    # Phase 3: submit order to Alpaca paper account
    logger.info("Stocks cycle complete")


if __name__ == "__main__":
    logger.info("Stocks bot starting")
    schedule.every().day.at("09:35").do(run_cycle)  # 5 min after NYSE open
    while True:
        schedule.run_pending()
        time.sleep(60)
