import os
from dotenv import load_dotenv

load_dotenv()

# Alpaca
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"

# Coinbase Advanced
# Public market data needs no auth; keys only required for Phase 5 live trading.
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
# Crypto paper trading is simulated locally (Coinbase has no sandbox).
CRYPTO_PAPER = os.getenv("CRYPTO_PAPER", "true").lower() == "true"

# Shared trading params
RISK_PER_TRADE = 0.02       # 2% of capital per trade
MAX_OPEN_POSITIONS = 5
STOP_LOSS_PCT = 0.03        # 3% stop-loss

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
STOCKS_RESULTS_DIR = os.path.join(RESULTS_DIR, "stocks")
CRYPTO_RESULTS_DIR = os.path.join(RESULTS_DIR, "crypto")
