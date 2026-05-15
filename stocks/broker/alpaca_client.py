"""
Alpaca paper/live trading client.

Wraps alpaca-py with the few operations we need: account info, current
positions, latest quote, market-order entry with bracket (stop-loss + take-
profit). Same code path works for paper and live — only the URL/keys change.
"""
from dataclasses import dataclass
from datetime import datetime

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopLossRequest, TakeProfitRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from loguru import logger

from common.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_PAPER


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    market_value: float
    unrealized_pl: float


@dataclass
class Account:
    cash: float
    portfolio_value: float
    buying_power: float


class AlpacaBroker:
    def __init__(self) -> None:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env")
        self.trading = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=ALPACA_PAPER)
        self.data = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
        logger.info(f"Alpaca client ready (paper={ALPACA_PAPER})")

    def account(self) -> Account:
        a = self.trading.get_account()
        return Account(
            cash=float(a.cash),
            portfolio_value=float(a.portfolio_value),
            buying_power=float(a.buying_power),
        )

    def positions(self) -> dict[str, Position]:
        result: dict[str, Position] = {}
        for p in self.trading.get_all_positions():
            result[p.symbol] = Position(
                symbol=p.symbol,
                qty=float(p.qty),
                avg_entry=float(p.avg_entry_price),
                market_value=float(p.market_value),
                unrealized_pl=float(p.unrealized_pl),
            )
        return result

    def latest_price(self, symbol: str) -> float:
        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = self.data.get_stock_latest_quote(req)[symbol]
        # Use mid-price; fall back to ask if bid missing
        bid, ask = float(quote.bid_price or 0), float(quote.ask_price or 0)
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return ask or bid

    def submit_bracket_buy(self, symbol: str, qty: int, stop_loss_pct: float, take_profit_pct: float) -> str:
        """Submit a market buy with attached stop-loss and take-profit."""
        price = self.latest_price(symbol)
        stop_price = round(price * (1 - stop_loss_pct), 2)
        target_price = round(price * (1 + take_profit_pct), 2)

        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.BRACKET,
            stop_loss=StopLossRequest(stop_price=stop_price),
            take_profit=TakeProfitRequest(limit_price=target_price),
        )
        order = self.trading.submit_order(order_data=req)
        logger.info(f"Bracket BUY {symbol} qty={qty} @~{price:.2f} (SL {stop_price}, TP {target_price}) id={order.id}")
        return str(order.id)

    def close_position(self, symbol: str) -> None:
        self.trading.close_position(symbol)
        logger.info(f"Closed position {symbol}")

    def recent_orders(self, since: datetime | None = None) -> list:
        req = GetOrdersRequest(status=QueryOrderStatus.ALL, after=since)
        return self.trading.get_orders(filter=req)

    def is_market_open(self) -> bool:
        return self.trading.get_clock().is_open
