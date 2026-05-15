import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Binance
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
BINANCE_TESTNET = os.getenv("BINANCE_TESTNET", "true").lower() == "true"

# Shared trading params
RISK_PER_TRADE = 0.02       # 2% of capital per trade
MAX_OPEN_POSITIONS = 5
STOP_LOSS_PCT = 0.03        # 3% stop-loss

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STOCKS_RESULTS_DIR = os.path.join(RESULTS_DIR, "stocks")
CRYPTO_RESULTS_DIR = os.path.join(RESULTS_DIR, "crypto")
