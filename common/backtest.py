"""
Shared backtesting harness using vectorbt.

Strategy: long-only, enter when model predicts up (proba > threshold),
exit when proba flips below threshold or stop-loss hits. Each bot supplies
its own model predictions, this harness simulates trades and reports stats.
"""
import os
import json
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import vectorbt as vbt
from loguru import logger

from common.features import feature_columns
from common.training import load_model


@dataclass
class BacktestResult:
    symbol: str
    start: str
    end: str
    n_trades: int
    total_return: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    final_value: float
    initial_cash: float


def generate_signals(model, df: pd.DataFrame, threshold: float = 0.60) -> pd.Series:
    """
    Generate boolean entry signals from model probabilities.

    Note on lookahead: this assumes the model was trained on data prior to
    the backtest window. For honest evaluation, use walk-forward predictions
    instead of a single model — see `walk_forward_signals` below.
    """
    cols = feature_columns()
    df = df.dropna(subset=cols).copy()
    proba = model.predict_proba(df[cols])[:, 1]
    return pd.Series(proba > threshold, index=df.index, name="signal")


def walk_forward_signals(df: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.5, threshold: float = 0.60) -> pd.Series:
    """
    Generate out-of-sample signals using the same walk-forward scheme as training.
    This is the honest version for backtest evaluation — no lookahead.
    """
    from common.training import prepare_xy, _xgb_model
    X, y = prepare_xy(df)
    n = len(X)
    min_train = int(n * min_train_frac)
    test_size = (n - min_train) // n_folds

    proba_full = pd.Series(np.nan, index=X.index)
    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = test_start + test_size if fold < n_folds - 1 else n
        model = _xgb_model()
        model.fit(X.iloc[:test_start], y.iloc[:test_start])
        p = model.predict_proba(X.iloc[test_start:test_end])[:, 1]
        proba_full.iloc[test_start:test_end] = p

    signal = (proba_full > threshold).fillna(False)
    return signal


def run_backtest(symbol: str, df: pd.DataFrame, signals: pd.Series, initial_cash: float = 10_000.0, fees: float = 0.001, slippage: float = 0.0005, max_hold: int | None = None, stop_loss: float | None = None) -> BacktestResult:
    """
    Run vectorbt backtest on a long-only signal series.

    `fees` and `slippage` are fractional (0.001 = 0.1%).
    `max_hold`: optional cap on holding period in bars (forces exit after N days).
    `stop_loss`: optional fractional stop (e.g. 0.05 = 5% stop-loss).
    """
    df = df.loc[signals.index].copy()
    close = df["close"]

    # Long when signal flips True, exit when signal flips False
    entries = signals & ~signals.shift(1).fillna(False)
    exits = ~signals & signals.shift(1).fillna(False)

    # Add a time-based exit: force exit `max_hold` bars after every entry
    if max_hold is not None and max_hold > 0:
        time_exit = entries.shift(max_hold).fillna(False)
        exits = exits | time_exit

    pf_kwargs = dict(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        freq="1D",
    )
    if stop_loss is not None:
        pf_kwargs["sl_stop"] = stop_loss
    pf = vbt.Portfolio.from_signals(**pf_kwargs)

    stats = pf.stats()
    n_trades = int(stats.get("Total Trades", 0))
    return BacktestResult(
        symbol=symbol,
        start=str(close.index[0]),
        end=str(close.index[-1]),
        n_trades=n_trades,
        total_return=float(stats.get("Total Return [%]", 0.0)) / 100.0,
        sharpe=float(stats.get("Sharpe Ratio", 0.0)) if not pd.isna(stats.get("Sharpe Ratio", 0.0)) else 0.0,
        max_drawdown=float(stats.get("Max Drawdown [%]", 0.0)) / 100.0,
        win_rate=float(stats.get("Win Rate [%]", 0.0)) / 100.0 if n_trades > 0 else 0.0,
        final_value=float(pf.value().iloc[-1]),
        initial_cash=initial_cash,
    )


def save_backtest(result: BacktestResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)


def backtest_symbol(symbol: str, df: pd.DataFrame, report_dir: str, fees: float = 0.001, threshold: float = 0.60, max_hold: int | None = None, stop_loss: float | None = 0.05) -> BacktestResult:
    """Run walk-forward signals + backtest + persist report."""
    logger.info(f"Backtesting {symbol}")
    signals = walk_forward_signals(df, threshold=threshold)
    result = run_backtest(symbol, df, signals, fees=fees, max_hold=max_hold, stop_loss=stop_loss)
    save_backtest(result, os.path.join(report_dir, f"{symbol}_backtest.json"))
    logger.info(
        f"{symbol}: trades={result.n_trades} return={result.total_return:.2%} "
        f"sharpe={result.sharpe:.2f} dd={result.max_drawdown:.2%} win={result.win_rate:.2%}"
    )
    return result
