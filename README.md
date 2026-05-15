# trading-ai

AI-powered trading bots for US stocks and crypto, with a futures module planned for later.

## Structure

```
trading-ai/
├── common/       # Shared config, logging, storage
├── stocks/       # US equities bot (Alpaca)
├── crypto/       # Crypto bot (Binance)
├── results/      # Trade logs and backtest reports (gitignored)
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
cp .env.example .env        # then fill in your API keys
```

## Running the bots

```bash
# Stocks bot (runs daily at 09:35 ET)
python -m stocks.bot

# Crypto bot (runs daily at 00:05 UTC)
python -m crypto.bot
```

## Phases

- [x] Phase 1 — Project scaffold & data pipeline
- [ ] Phase 2 — Model training & backtesting
- [ ] Phase 3 — Paper trading loop
- [ ] Phase 4 — Monitoring dashboard
- [ ] Phase 5 — Live trading migration
- [ ] Phase 6 — Futures module
