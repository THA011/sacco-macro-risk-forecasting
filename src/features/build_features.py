"""
Merge the loan ledger panel with the macro panel and engineer the feature
table the models train on, including the forward-looking target label:
"does this account migrate to a worse SASRA bucket next month?"

Usage:
    python -m src.features.build_features
"""

import numpy as np
import pandas as pd

from src.config import (
    CLASSIFICATION_ORDER,
    FEATURES_CSV,
    LEDGER_CSV,
    MACRO_CSV,
    SECTOR_FUEL_SENSITIVITY,
)

CLASS_RANK = {label: i for i, label in enumerate(CLASSIFICATION_ORDER)}


def _add_macro_lags(macro: pd.DataFrame, lags=(1, 2, 3)) -> pd.DataFrame:
    macro = macro.sort_values("month").reset_index(drop=True)
    macro["fuel_pct_change"] = macro["fuel_price_kes_per_litre"].pct_change().fillna(0.0)
    for lag in lags:
        macro[f"fuel_pct_change_lag{lag}"] = macro["fuel_pct_change"].shift(lag).fillna(0.0)
        macro[f"inflation_lag{lag}"] = macro["inflation_pct_yoy"].shift(lag).fillna(
            macro["inflation_pct_yoy"].iloc[0]
        )
    return macro


def _add_account_level_features(ledger: pd.DataFrame) -> pd.DataFrame:
    ledger = ledger.sort_values(["account_id", "month"]).reset_index(drop=True)

    ledger["class_rank"] = ledger["classification"].map(CLASS_RANK)
    ledger["par30_flag"] = (ledger["days_in_arrears"] >= 30).astype(int)
    ledger["par90_flag"] = (ledger["days_in_arrears"] >= 90).astype(int)
    ledger["sector_fuel_sensitivity"] = ledger["sector"].map(SECTOR_FUEL_SENSITIVITY)

    # 3-month rolling mean of days-in-arrears, per account — a smoothed
    # trend signal in addition to the point-in-time value.
    ledger["days_in_arrears_roll3"] = (
        ledger.groupby("account_id")["days_in_arrears"]
        .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        .round(2)
    )

    # Forward-looking label: does next month's bucket rank exceed this
    # month's rank? Computed within each account's own time-ordered rows.
    ledger["next_class_rank"] = ledger.groupby("account_id")["class_rank"].shift(-1)
    ledger["migrates_worse"] = (
        ledger["next_class_rank"] > ledger["class_rank"]
    ).astype("Int64")

    # Drop the final observed month per account — there is no "next month"
    # to label it against.
    ledger = ledger.dropna(subset=["next_class_rank"]).copy()
    ledger["migrates_worse"] = ledger["migrates_worse"].astype(int)
    return ledger


def build_features() -> pd.DataFrame:
    ledger = pd.read_csv(LEDGER_CSV, parse_dates=["month"])
    macro = pd.read_csv(MACRO_CSV, parse_dates=["month"])
    macro = _add_macro_lags(macro)

    ledger = _add_account_level_features(ledger)

    merged = ledger.merge(
        macro.drop(columns=["fuel_pct_change"]),  # avoid duplicate column, already on ledger via lags below
        on="month",
        how="left",
        suffixes=("", "_macro"),
    )

    # One-hot encode sector for modeling
    sector_dummies = pd.get_dummies(merged["sector"], prefix="sector")
    merged = pd.concat([merged, sector_dummies], axis=1)

    merged.to_csv(FEATURES_CSV, index=False)
    return merged


def main() -> None:
    df = build_features()
    print(f"Wrote {len(df)} feature rows -> {FEATURES_CSV}")
    print(f"Positive label rate (migrates_worse=1): {df['migrates_worse'].mean():.3f}")


if __name__ == "__main__":
    main()
