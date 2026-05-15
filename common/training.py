"""
Shared XGBoost training utilities with walk-forward validation.

The indicators are identical between stocks and crypto, but each bot trains
its OWN model on its OWN data — no cross-pollination. This module just gives
both bots the same scientific training/validation harness.
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

from common.features import feature_columns


@dataclass
class FoldResult:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    accuracy: float
    auc: float


@dataclass
class TrainingReport:
    symbol: str
    folds: list[FoldResult]
    mean_accuracy: float
    mean_auc: float
    final_model_n_train: int


def _xgb_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def prepare_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows with NaN features/target and return X, y."""
    cols = feature_columns()
    df = df.dropna(subset=cols + ["target"]).copy()
    return df[cols], df["target"].astype(int)


def walk_forward(df: pd.DataFrame, n_folds: int = 5, min_train_frac: float = 0.5) -> list[FoldResult]:
    """
    Expanding-window walk-forward validation.

    Splits the post-warmup data into n_folds test windows; each fold trains on
    everything before its test window. This mimics how the model would be
    retrained in production and avoids lookahead bias.
    """
    X, y = prepare_xy(df)
    n = len(X)
    if n < 200:
        raise ValueError(f"Not enough samples for walk-forward: {n}")

    min_train = int(n * min_train_frac)
    test_size = (n - min_train) // n_folds
    results: list[FoldResult] = []

    for fold in range(n_folds):
        test_start = min_train + fold * test_size
        test_end = test_start + test_size if fold < n_folds - 1 else n
        X_train, y_train = X.iloc[:test_start], y.iloc[:test_start]
        X_test, y_test = X.iloc[test_start:test_end], y.iloc[test_start:test_end]

        model = _xgb_model()
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba > 0.5).astype(int)

        results.append(FoldResult(
            fold=fold,
            train_start=str(X_train.index[0]),
            train_end=str(X_train.index[-1]),
            test_start=str(X_test.index[0]),
            test_end=str(X_test.index[-1]),
            n_train=len(X_train),
            n_test=len(X_test),
            accuracy=float(accuracy_score(y_test, preds)),
            auc=float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else float("nan"),
        ))
        logger.info(f"Fold {fold}: acc={results[-1].accuracy:.3f} auc={results[-1].auc:.3f}")

    return results


def train_final_model(df: pd.DataFrame) -> XGBClassifier:
    """Train one model on all available data — used for live inference."""
    X, y = prepare_xy(df)
    model = _xgb_model()
    model.fit(X, y)
    return model


def save_model(model: XGBClassifier, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> XGBClassifier:
    return joblib.load(path)


def save_report(report: TrainingReport, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {**asdict(report), "folds": [asdict(f) for f in report.folds]}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def train_symbol(symbol: str, df: pd.DataFrame, model_dir: str, report_dir: str) -> TrainingReport:
    """Walk-forward validate + train final model + persist everything."""
    logger.info(f"Training {symbol}")
    folds = walk_forward(df)
    final_model = train_final_model(df)
    save_model(final_model, os.path.join(model_dir, f"{symbol}.joblib"))

    report = TrainingReport(
        symbol=symbol,
        folds=folds,
        mean_accuracy=float(np.mean([f.accuracy for f in folds])),
        mean_auc=float(np.nanmean([f.auc for f in folds])),
        final_model_n_train=len(prepare_xy(df)[0]),
    )
    save_report(report, os.path.join(report_dir, f"{symbol}.json"))
    logger.info(f"{symbol}: mean acc={report.mean_accuracy:.3f} mean auc={report.mean_auc:.3f}")
    return report
