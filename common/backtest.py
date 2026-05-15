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

from common.features import feature_columns_in
from common.training import load_model
from common.config import TAX_RATE


def compute_taxes(trades_df: pd.DataFrame, tax_rate: float) -> float:
    """
    Sum short-term capital gains tax across years.

    Each trade is realized on its exit date. Gains and losses within the same
    calendar year net out; net losses carry forward to offset future years
    (simplified — no $3k/year ordinary-income offset, no wash sales).
    """
    if trades_df.empty or tax_rate <= 0:
        return 0.0
    df = trades_df[["Exit Timestamp", "PnL"]].copy()
    df["year"] = pd.to_datetime(df["Exit Timestamp"]).dt.year
    annual = df.groupby("year")["PnL"].sum().sort_index()

    taxes = 0.0
    carry = 0.0
    for pnl in annual.values:
        taxable = pnl + carry
        if taxable > 0:
            taxes += taxable * tax_rate
            carry = 0.0
        else:
            carry = taxable
    return float(taxes)


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
    # After-tax view (short-term capital gains at TAX_RATE applied annually,
    # with losses carried forward to the next year)
    tax_rate: float = 0.0
    total_taxes: float = 0.0
    after_tax_return: float = 0.0
    after_tax_final_value: float = 0.0


def generate_signals(model, df: pd.DataFrame, threshold: float = 0.60) -> pd.Series:
    """
    Generate boolean entry signals from model probabilities.

    Note on lookahead: this assumes the model was trained on data prior to
    the backtest window. For honest evaluation, use walk-forward predictions
    instead of a single model — see `walk_forward_signals` below.
    """
    cols = feature_columns_in(df)
    df = df.dropna(subset=cols).copy()
    proba = model.predict_proba(df[cols])[:, 1]
    return pd.Series(proba > threshold, index=df.index, name="signal")


def walk_forward_signals(df: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.5, threshold: float = 0.60, regime: pd.Series | None = None) -> pd.Series:
    """
    Out-of-sample signal series using walk-forward training, optionally gated
    by a regime series (True = risk-on, False = no entries that day).
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
    if regime is not None:
        aligned = regime.reindex(signal.index).ffill().fillna(False).astype(bool)
        signal = signal & aligned
    return signal


def run_backtest(symbol: str, df: pd.DataFrame, signals: pd.Series, initial_cash: float = 10_000.0, fees: float = 0.001, slippage: float = 0.0005, max_hold: int | None = None, stop_loss: float | None = None, tax_rate: float = TAX_RATE) -> BacktestResult:
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
    final_value = float(pf.value().iloc[-1])

    taxes = compute_taxes(pf.trades.records_readable, tax_rate) if n_trades > 0 else 0.0
    after_tax_final = final_value - taxes
    after_tax_return = (after_tax_final - initial_cash) / initial_cash

    return BacktestResult(
        symbol=symbol,
        start=str(close.index[0]),
        end=str(close.index[-1]),
        n_trades=n_trades,
        total_return=float(stats.get("Total Return [%]", 0.0)) / 100.0,
        sharpe=float(stats.get("Sharpe Ratio", 0.0)) if not pd.isna(stats.get("Sharpe Ratio", 0.0)) else 0.0,
        max_drawdown=float(stats.get("Max Drawdown [%]", 0.0)) / 100.0,
        win_rate=float(stats.get("Win Rate [%]", 0.0)) / 100.0 if n_trades > 0 else 0.0,
        final_value=final_value,
        initial_cash=initial_cash,
        tax_rate=tax_rate,
        total_taxes=taxes,
        after_tax_return=after_tax_return,
        after_tax_final_value=after_tax_final,
    )


def save_backtest(result: BacktestResult, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)


@dataclass
class SplitBacktest:
    """A symbol's backtest split into in-sample (selection) and OOS (evaluation)."""
    symbol: str
    selection: BacktestResult
    evaluation: BacktestResult


def backtest_symbol(
    symbol: str, df: pd.DataFrame, report_dir: str,
    fees: float = 0.001, threshold: float = 0.60,
    max_hold: int | None = None, stop_loss: float | None = 0.05,
    regime: pd.Series | None = None,
    selection_frac: float = 0.60,
) -> SplitBacktest:
    """
    Run walk-forward signals, split into selection (early 60%) and evaluation
    (late 40%) windows, backtest each, persist a combined report.

    The split is for HONEST universe selection: we pick names based on
    `selection` numbers and judge live deployment by `evaluation` numbers.
    """
    logger.info(f"Backtesting {symbol}")
    signals = walk_forward_signals(df, threshold=threshold, regime=regime)
    aligned_df = df.loc[signals.index]

    split_idx = int(len(aligned_df) * selection_frac)
    split_date = aligned_df.index[split_idx]

    sel_signals = signals.loc[:split_date].iloc[:-1]
    sel_df = aligned_df.loc[sel_signals.index]
    sel_result = run_backtest(f"{symbol}_sel", sel_df, sel_signals, fees=fees, max_hold=max_hold, stop_loss=stop_loss)

    eval_signals = signals.loc[split_date:]
    eval_df = aligned_df.loc[eval_signals.index]
    eval_result = run_backtest(f"{symbol}_eval", eval_df, eval_signals, fees=fees, max_hold=max_hold, stop_loss=stop_loss)

    combined = SplitBacktest(symbol=symbol, selection=sel_result, evaluation=eval_result)
    _save_split(combined, os.path.join(report_dir, f"{symbol}_backtest.json"))
    logger.info(
        f"{symbol} SEL: trades={sel_result.n_trades} net={sel_result.after_tax_return:.2%} "
        f"sharpe={sel_result.sharpe:.2f} | EVAL: trades={eval_result.n_trades} "
        f"net={eval_result.after_tax_return:.2%} sharpe={eval_result.sharpe:.2f}"
    )
    return combined


def _save_split(split: SplitBacktest, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "symbol": split.symbol,
        "selection": asdict(split.selection),
        "evaluation": asdict(split.evaluation),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
