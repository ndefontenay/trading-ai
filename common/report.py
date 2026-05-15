"""Print a summary table of all training + backtest reports."""
import os
import json
import glob


def summarize(report_dir: str, label: str) -> None:
    if not os.path.isdir(report_dir):
        print(f"No reports in {report_dir}")
        return

    print(f"\n=== {label} ===")
    print(f"{'Symbol':<10} {'Acc':>6} {'AUC':>6} {'Trades':>7} {'Gross':>9} {'Tax$':>9} {'Net':>9} {'Sharpe':>7} {'MaxDD':>8} {'Win%':>7}")
    print("-" * 92)

    symbols = sorted({os.path.basename(p).replace("_backtest.json", "").replace(".json", "")
                      for p in glob.glob(os.path.join(report_dir, "*.json"))})

    for sym in symbols:
        train_path = os.path.join(report_dir, f"{sym}.json")
        bt_path = os.path.join(report_dir, f"{sym}_backtest.json")
        train = json.load(open(train_path)) if os.path.exists(train_path) else {}
        bt = json.load(open(bt_path)) if os.path.exists(bt_path) else {}

        print(f"{sym:<10} {train.get('mean_accuracy', 0):>6.3f} {train.get('mean_auc', 0):>6.3f} "
              f"{bt.get('n_trades', 0):>7d} {bt.get('total_return', 0):>8.2%} "
              f"${bt.get('total_taxes', 0):>7,.0f} {bt.get('after_tax_return', 0):>8.2%} "
              f"{bt.get('sharpe', 0):>7.2f} {bt.get('max_drawdown', 0):>7.2%} "
              f"{bt.get('win_rate', 0):>6.2%}")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    summarize(os.path.join(base, "results", "stocks", "reports"), "STOCKS")
    summarize(os.path.join(base, "results", "crypto", "reports"), "CRYPTO")
