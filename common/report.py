"""Print a summary table of all training + split backtest reports."""
import os
import json
import glob


def summarize(report_dir: str, label: str, universe_file: str | None = None) -> None:
    if not os.path.isdir(report_dir):
        print(f"No reports in {report_dir}")
        return

    active: set[str] = set()
    if universe_file and os.path.exists(universe_file):
        data = json.load(open(universe_file))
        active = set(data.get("tickers") or data.get("pairs") or [])

    print(f"\n=== {label} ===")
    print(f"{'Symbol':<10} {'Acc':>6} {'AUC':>6}  || {'sel Trd':>7} {'sel Net':>9} {'sel Sh':>7}  || {'eval Trd':>8} {'eval Net':>9} {'eval Sh':>8} {'eval DD':>8}  Active")
    print("-" * 115)

    symbols = sorted({os.path.basename(p).replace("_backtest.json", "").replace(".json", "")
                      for p in glob.glob(os.path.join(report_dir, "*.json"))})

    for sym in symbols:
        train_path = os.path.join(report_dir, f"{sym}.json")
        bt_path = os.path.join(report_dir, f"{sym}_backtest.json")
        train = json.load(open(train_path)) if os.path.exists(train_path) else {}
        bt = json.load(open(bt_path)) if os.path.exists(bt_path) else {}
        sel = bt.get("selection", {})
        evl = bt.get("evaluation", {})

        active_marker = "*" if sym in active else " "
        print(f"{sym:<10} {train.get('mean_accuracy', 0):>6.3f} {train.get('mean_auc', 0):>6.3f}  || "
              f"{sel.get('n_trades', 0):>7d} {sel.get('after_tax_return', 0):>8.2%} {sel.get('sharpe', 0):>7.2f}  || "
              f"{evl.get('n_trades', 0):>8d} {evl.get('after_tax_return', 0):>8.2%} {evl.get('sharpe', 0):>8.2f} {evl.get('max_drawdown', 0):>7.2%}   {active_marker}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summarize(
        os.path.join(base, "results", "stocks", "reports"),
        "STOCKS",
        os.path.join(base, "results", "stocks", "active_universe.json"),
    )
    summarize(
        os.path.join(base, "results", "crypto", "reports"),
        "CRYPTO",
        os.path.join(base, "results", "crypto", "active_universe.json"),
    )
