"""
Cross-sectional ML: pool all symbols into one training set, walk forward
by DATE (not by symbol-row), generate OOS predictions per symbol-date.

Why pooled: instead of training 10 small models on ~1000 examples each,
we train 1 model on ~10000 examples — much better statistical power
and the model learns market-wide patterns (e.g., "high VIX + oversold
RSI tends to bounce" generalizes across names).

Time-based walk-forward avoids the bias of randomly splitting symbols
or rows; we always train strictly in the past relative to the test window.
"""
import os
import json
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

from common.features import feature_columns_in


@dataclass
class FoldStats:
    fold: int
    train_end: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    auc: float


def _xgb_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def pool_data(data_dict: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """
    Stack per-symbol DataFrames into one long DataFrame with columns:
      symbol, date, <features...>, target

    Returns (pooled_df, feature_cols).
    """
    if not data_dict:
        raise ValueError("Empty data_dict")
    feature_cols = feature_columns_in(next(iter(data_dict.values())))
    needed = feature_cols + ["target", "close"]

    frames = []
    for sym, df in data_dict.items():
        d = df[needed].dropna(subset=feature_cols + ["target"]).copy()
        d["symbol"] = sym
        d["date"] = d.index
        frames.append(d)

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(["date", "symbol"]).reset_index(drop=True)
    pooled["target"] = pooled["target"].astype(int)
    return pooled, feature_cols


def walk_forward_pooled(pooled: pd.DataFrame, feature_cols: list[str], n_folds: int = 5, min_train_frac: float = 0.5) -> tuple[list[FoldStats], pd.Series]:
    """
    Walk forward by date. Returns (fold_stats, oos_proba_per_row aligned to pooled).
    """
    dates = np.sort(pooled["date"].unique())
    n_dates = len(dates)
    min_train_dates = max(int(n_dates * min_train_frac), 50)
    test_window = max((n_dates - min_train_dates) // n_folds, 1)

    proba_out = pd.Series(np.nan, index=pooled.index, dtype=float)
    folds: list[FoldStats] = []

    for fold in range(n_folds):
        train_end_idx = min_train_dates + fold * test_window
        test_end_idx = train_end_idx + test_window if fold < n_folds - 1 else n_dates
        if train_end_idx >= n_dates:
            break
        train_end_date = dates[train_end_idx - 1]
        test_end_date = dates[test_end_idx - 1]

        train_mask = pooled["date"] <= train_end_date
        test_mask = (pooled["date"] > train_end_date) & (pooled["date"] <= test_end_date)
        if not test_mask.any():
            continue

        X_train, y_train = pooled.loc[train_mask, feature_cols], pooled.loc[train_mask, "target"]
        X_test, y_test = pooled.loc[test_mask, feature_cols], pooled.loc[test_mask, "target"]

        model = _xgb_model()
        model.fit(X_train, y_train)
        p = model.predict_proba(X_test)[:, 1]
        proba_out.loc[test_mask] = p

        preds = (p > 0.5).astype(int)
        auc = float(roc_auc_score(y_test, p)) if y_test.nunique() > 1 else float("nan")
        stats = FoldStats(
            fold=fold,
            train_end=str(train_end_date),
            test_end=str(test_end_date),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask.sum()),
            accuracy=float(accuracy_score(y_test, preds)),
            auc=auc,
        )
        folds.append(stats)
        logger.info(f"Fold {fold} (train≤{stats.train_end[:10]}, n_train={stats.n_train}): acc={stats.accuracy:.3f} auc={stats.auc:.3f}")

    return folds, proba_out


def train_final_pooled(pooled: pd.DataFrame, feature_cols: list[str]) -> XGBClassifier:
    """Train one model on ALL pooled data — used for live inference."""
    model = _xgb_model()
    model.fit(pooled[feature_cols], pooled["target"])
    return model


def save_pooled_artifacts(model: XGBClassifier, feature_cols: list[str], folds: list[FoldStats], model_dir: str, report_dir: str) -> None:
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, os.path.join(model_dir, "pooled.joblib"))
    report = {
        "n_folds": len(folds),
        "mean_accuracy": float(np.mean([f.accuracy for f in folds])) if folds else 0.0,
        "mean_auc": float(np.nanmean([f.auc for f in folds])) if folds else 0.0,
        "folds": [asdict(f) for f in folds],
        "feature_cols": feature_cols,
    }
    with open(os.path.join(report_dir, "pooled.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)


def load_pooled(model_dir: str) -> tuple[XGBClassifier, list[str]]:
    payload = joblib.load(os.path.join(model_dir, "pooled.joblib"))
    return payload["model"], payload["feature_cols"]


def per_symbol_signal_series(pooled: pd.DataFrame, proba: pd.Series, threshold: float = 0.60, regime: pd.Series | None = None) -> dict[str, pd.Series]:
    """
    Split pooled OOS probabilities back into per-symbol signal Series indexed by date.
    Optional regime mask is applied (date-wise) before thresholding.
    """
    df = pooled[["symbol", "date"]].copy()
    df["proba"] = proba.values

    if regime is not None:
        aligned = regime.reindex(df["date"]).ffill().fillna(False).astype(bool).values
        df["signal"] = (df["proba"] > threshold) & aligned
    else:
        df["signal"] = df["proba"] > threshold

    out: dict[str, pd.Series] = {}
    for sym, sub in df.groupby("symbol"):
        s = pd.Series(sub["signal"].values, index=pd.DatetimeIndex(sub["date"]).rename(None), name=sym).fillna(False)
        out[sym] = s
    return out
