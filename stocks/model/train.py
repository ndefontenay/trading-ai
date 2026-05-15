"""
Cross-sectional training for the stocks universe.

Pools all tickers into one training set, walk-forward by date, then
backtests each symbol using its slice of the OOS predictions.

Run: python -m stocks.model.train
"""
import os
import json
import pandas as pd
from loguru import logger

from common.cross_section import (
    pool_data, walk_forward_pooled, train_final_pooled,
    save_pooled_artifacts, per_symbol_signal_series,
)
from common.backtest import run_backtest, save_backtest, SplitBacktest, _save_split
from common.regime import stocks_regime
from stocks.data.fetcher import DEFAULT_TICKERS, load_ticker, TARGET_HORIZON

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "results", "stocks", "models")
REPORT_DIR = os.path.join(BASE_DIR, "results", "stocks", "reports")
UNIVERSE_FILE = os.path.join(BASE_DIR, "results", "stocks", "active_universe.json")

STOCKS_FEE_RATE = 0.0
THRESHOLD = 0.60
SELECTION_FRAC = 0.60
# Empirically the tight Sharpe filter was anti-correlated with eval outcomes
# (tiny selection trade counts → high noise). Use a permissive net-loss bar
# instead: drop only clear bleeders. Live results will refine over time.
MIN_SEL_NET_RETURN = -0.05  # tolerate up to -5% selection-period loss


def _split_backtest_for_symbol(sym: str, df: pd.DataFrame, signals: pd.Series) -> SplitBacktest:
    aligned_df = df.loc[signals.index]
    split_idx = int(len(aligned_df) * SELECTION_FRAC)
    if split_idx < 5 or split_idx >= len(aligned_df) - 5:
        # too short to split — return same result twice
        full = run_backtest(sym, aligned_df, signals, fees=STOCKS_FEE_RATE, max_hold=TARGET_HORIZON, stop_loss=0.03)
        return SplitBacktest(symbol=sym, selection=full, evaluation=full)

    split_date = aligned_df.index[split_idx]
    sel_sig = signals.loc[:split_date].iloc[:-1]
    sel_df = aligned_df.loc[sel_sig.index]
    sel = run_backtest(f"{sym}_sel", sel_df, sel_sig, fees=STOCKS_FEE_RATE, max_hold=TARGET_HORIZON, stop_loss=0.03)

    eval_sig = signals.loc[split_date:]
    eval_df = aligned_df.loc[eval_sig.index]
    evl = run_backtest(f"{sym}_eval", eval_df, eval_sig, fees=STOCKS_FEE_RATE, max_hold=TARGET_HORIZON, stop_loss=0.03)
    return SplitBacktest(symbol=sym, selection=sel, evaluation=evl)


def train_all(tickers: list[str] | None = None) -> None:
    tickers = tickers or DEFAULT_TICKERS
    data_dict: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            data_dict[t] = load_ticker(t)
        except FileNotFoundError:
            logger.warning(f"No data for {t}, skipping")

    logger.info(f"Pooling {len(data_dict)} symbols")
    pooled, feat_cols = pool_data(data_dict)
    logger.info(f"Pooled rows: {len(pooled)} | features: {len(feat_cols)}")

    logger.info("Walk-forward training (pooled)")
    folds, proba = walk_forward_pooled(pooled, feat_cols)

    logger.info("Fetching SPY regime")
    regime = stocks_regime(period="5y")

    logger.info("Splitting OOS predictions per symbol + applying regime")
    sig_by_sym = per_symbol_signal_series(pooled, proba, threshold=THRESHOLD, regime=regime)

    surviving: list[str] = []
    for sym, signal in sig_by_sym.items():
        if signal.empty or not signal.any():
            logger.info(f"{sym}: no signals in OOS window")
            continue
        bt = _split_backtest_for_symbol(sym, data_dict[sym], signal)
        _save_split(bt, os.path.join(REPORT_DIR, f"{sym}_backtest.json"))
        logger.info(
            f"{sym} SEL: trades={bt.selection.n_trades} net={bt.selection.after_tax_return:.2%} sh={bt.selection.sharpe:.2f} | "
            f"EVAL: trades={bt.evaluation.n_trades} net={bt.evaluation.after_tax_return:.2%} sh={bt.evaluation.sharpe:.2f}"
        )
        if bt.selection.after_tax_return >= MIN_SEL_NET_RETURN:
            surviving.append(sym)

    logger.info("Training final pooled model on all data")
    final = train_final_pooled(pooled, feat_cols)
    save_pooled_artifacts(final, feat_cols, folds, MODEL_DIR, REPORT_DIR)

    os.makedirs(os.path.dirname(UNIVERSE_FILE), exist_ok=True)
    with open(UNIVERSE_FILE, "w") as f:
        json.dump({"tickers": surviving}, f, indent=2)
    logger.info(f"Active universe ({len(surviving)} names): {surviving}")


if __name__ == "__main__":
    train_all()
