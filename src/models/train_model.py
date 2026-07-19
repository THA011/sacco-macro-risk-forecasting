"""
Train baseline classifiers for `migrates_worse` (does this account slip into
a worse SASRA bucket next month?) on a TIME-based train/test split.

Why time-based and not random: the entire premise of this project is
forecasting behavior under a macro shock the model has not seen yet. A
random split leaks future shock months into training and will produce
metrics that look good and mean nothing in production. Train on the earlier
months, test on the later ones — the same discipline as backtesting.

Usage:
    python -m src.models.train_model [--test-frac 0.25]
"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import FEATURES_CSV, MODELS_DIR, MODEL_METRICS_JSON, RANDOM_SEED

NON_FEATURE_COLS = {
    "account_id",
    "month",
    "sector",
    "classification",
    "class_rank",
    "next_class_rank",
    "migrates_worse",
}


def _feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS and df[c].dtype != "object"]


def time_based_split(df: pd.DataFrame, test_frac: float = 0.25):
    df = df.sort_values("month").reset_index(drop=True)
    cutoff_idx = int(len(df) * (1 - test_frac))
    cutoff_month = df.loc[cutoff_idx, "month"]
    train = df[df["month"] < cutoff_month]
    test = df[df["month"] >= cutoff_month]
    return train, test, cutoff_month


def train(test_frac: float = 0.25) -> dict:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["month"])
    feature_cols = _feature_columns(df)

    train_df, test_df, cutoff_month = time_based_split(df, test_frac=test_frac)

    X_train, y_train = train_df[feature_cols], train_df["migrates_worse"]
    X_test, y_test = test_df[feature_cols], test_df["migrates_worse"]

    results = {"cutoff_month": str(cutoff_month.date()), "n_train": len(train_df), "n_test": len(test_df)}

    # --- Baseline: Logistic Regression, class-balanced -------------------
    # Scaled inside a Pipeline: loan_amount (KES ~10k-500k) sitting next to
    # 0/1 flags on an unscaled feature matrix stalls lbfgs convergence.
    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    logit.fit(X_train, y_train)
    logit_proba = logit.predict_proba(X_test)[:, 1]
    results["logistic_regression_auc"] = float(roc_auc_score(y_test, logit_proba))
    joblib.dump(logit, MODELS_DIR / "logistic_regression.joblib")

    # --- XGBoost, scale_pos_weight for imbalance -------------------------
    n_pos = max(y_train.sum(), 1)
    n_neg = max(len(y_train) - n_pos, 1)
    scale_pos_weight = n_neg / n_pos

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        eval_metric="auc",
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    results["xgboost_auc"] = float(roc_auc_score(y_test, xgb_proba))
    joblib.dump(xgb, MODELS_DIR / "xgboost.joblib")

    feature_importance = sorted(
        zip(feature_cols, xgb.feature_importances_.tolist()), key=lambda x: -x[1]
    )
    results["xgboost_top_features"] = feature_importance[:10]
    results["feature_columns"] = feature_cols

    with open(MODEL_METRICS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-frac", type=float, default=0.25)
    args = parser.parse_args()

    results = train(test_frac=args.test_frac)
    print(f"Split cutoff month: {results['cutoff_month']} "
          f"(train={results['n_train']} rows, test={results['n_test']} rows)")
    print(f"Logistic Regression AUC: {results['logistic_regression_auc']:.3f}")
    print(f"XGBoost AUC:             {results['xgboost_auc']:.3f}")
    print("Top XGBoost features:")
    for name, importance in results["xgboost_top_features"][:5]:
        print(f"  {name}: {importance:.4f}")


if __name__ == "__main__":
    main()
