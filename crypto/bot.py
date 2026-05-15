"""
Crypto bot entry point.
Runs the daily signal → order cycle for crypto pairs via Binance testnet.
"""
import schedule
import time
from loguru import logger
from common.config import CRYPTO_RESULTS_DIR
from common.logger import setup_logger

setup_logger("crypto", CRYPTO_RESULTS_DIR)


def run_cycle():
    logger.info("Starting crypto trading cycle")
    # Phase 2: fetch latest data, generate signal
    # Phase 3: simulate order locally (paper) or submit to Coinbase (live)
    logger.info("Crypto cycle complete")


if __name__ == "__main__":
    logger.info("Crypto bot starting")
    schedule.every().day.at("00:05").do(run_cycle)  # 5 min after UTC midnight (daily close)
    while True:
        schedule.run_pending()
        time.sleep(60)
