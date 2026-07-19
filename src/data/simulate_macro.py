"""
Generate a synthetic monthly macro panel: EPRA-style pump price (KES/litre,
diesel), KNBS-style headline inflation (% y/y), and CBK-style USD/KES
indicative exchange rate.

This stands in for real EPRA/KNBS/CBK data during local development. Swap it
for `fetch_external.py` output once you have real-source data — the column
names are deliberately identical so `build_features.py` doesn't care which
one it's reading.

Usage:
    python -m src.data.simulate_macro [--months 48] [--shock-start 30]
"""

import argparse

import numpy as np
import pandas as pd

from src.config import MACRO_CSV, RANDOM_SEED


def simulate_macro(n_months: int = 48, shock_start: int = 30, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Build a monthly macro panel of length `n_months`.

    A "shock window" starting at month index `shock_start` raises the drift
    and volatility of the fuel-price series, standing in for an oil-market
    disruption (e.g. a Gulf-region supply shock) working its way through
    EPRA's monthly pricing formula. Inflation and FX drift upward with a lag,
    consistent with imported fuel costs feeding into the CPI basket and
    import bills pressuring the currency.
    """
    rng = np.random.default_rng(seed)
    months = pd.date_range("2022-01-01", periods=n_months, freq="MS")

    fuel_price = np.empty(n_months)
    fuel_price[0] = 180.0  # KES/litre, illustrative starting diesel price
    fx_rate = np.empty(n_months)
    fx_rate[0] = 145.0  # KES per USD, illustrative
    inflation = np.empty(n_months)
    inflation[0] = 6.5  # % y/y, illustrative

    for t in range(1, n_months):
        in_shock = t >= shock_start
        fuel_drift = 3.5 if in_shock else 0.4
        fuel_vol = 4.5 if in_shock else 1.5
        fuel_price[t] = max(
            fuel_price[t - 1] + rng.normal(fuel_drift, fuel_vol), 100.0
        )

        fx_drift = 0.6 if in_shock else 0.15
        fx_rate[t] = max(fx_rate[t - 1] + rng.normal(fx_drift, 0.8), 100.0)

        # Inflation responds to fuel price momentum with a one-month lag
        fuel_pct_change = (fuel_price[t] - fuel_price[t - 1]) / fuel_price[t - 1]
        inflation[t] = max(
            inflation[t - 1] + fuel_pct_change * 15 + rng.normal(0, 0.3), 0.5
        )

    df = pd.DataFrame(
        {
            "month": months,
            "fuel_price_kes_per_litre": fuel_price.round(2),
            "usd_kes_rate": fx_rate.round(2),
            "inflation_pct_yoy": inflation.round(2),
            "shock_period": [t >= shock_start for t in range(n_months)],
        }
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", type=int, default=48)
    parser.add_argument("--shock-start", type=int, default=30)
    args = parser.parse_args()

    df = simulate_macro(n_months=args.months, shock_start=args.shock_start)
    df.to_csv(MACRO_CSV, index=False)
    print(f"Wrote {len(df)} monthly macro rows -> {MACRO_CSV}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
