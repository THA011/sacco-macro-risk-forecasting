"""
Generate a synthetic, member-anonymous SACCO loan ledger panel: one row per
(account, month) with sector, loan attributes, and a days-in-arrears path
that responds to the macro fuel-price shock with a sector-specific
sensitivity multiplier (see src/config.SECTOR_FUEL_SENSITIVITY).

No name field exists anywhere in this schema, by design — see README §6 and
docs/methodology.md §5. Only a synthetic `account_id` identifies a row.

Usage:
    python -m src.data.simulate_ledger [--n-accounts 800] [--months 48]
"""

import argparse

import numpy as np
import pandas as pd

from src.config import (
    LEDGER_CSV,
    MACRO_CSV,
    RANDOM_SEED,
    SASRA_CLASSIFICATION_THRESHOLDS,
    SECTOR_FUEL_SENSITIVITY,
    SECTORS,
)
from src.data.simulate_macro import simulate_macro

SECTOR_WEIGHTS = {
    "salaried_formal": 0.35,
    "agriculture": 0.25,
    "retail_trade": 0.15,
    "transport": 0.10,
    "boda_boda": 0.10,
    "other": 0.05,
}
SECTOR_BASE_LOAN_MEAN = {
    "salaried_formal": 180_000,
    "agriculture": 120_000,
    "retail_trade": 90_000,
    "transport": 250_000,
    "boda_boda": 70_000,
    "other": 100_000,
}
TENURE_CHOICES_MONTHS = [12, 24, 36, 48]


def days_to_classification(days: int) -> str:
    """Map days-in-arrears to a SASRA bucket using the shared threshold table."""
    for label, lower, upper in SASRA_CLASSIFICATION_THRESHOLDS:
        if upper is None:
            if days >= lower:
                return label
        elif lower <= days <= upper:
            return label
    return "Normal"  # defensive fallback, should be unreachable


def _load_or_build_macro(n_months: int) -> pd.DataFrame:
    if MACRO_CSV.exists():
        macro = pd.read_csv(MACRO_CSV, parse_dates=["month"])
        if len(macro) >= n_months:
            return macro.iloc[:n_months].reset_index(drop=True)
    macro = simulate_macro(n_months=n_months)
    return macro


def simulate_ledger(
    n_accounts: int = 800, n_months: int = 48, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    macro = _load_or_build_macro(n_months)
    macro = macro.sort_values("month").reset_index(drop=True)
    macro["fuel_pct_change"] = macro["fuel_price_kes_per_litre"].pct_change().fillna(0.0)

    sectors = list(SECTOR_WEIGHTS.keys())
    weights = np.array([SECTOR_WEIGHTS[s] for s in sectors])
    weights = weights / weights.sum()

    rows = []
    for acc_idx in range(n_accounts):
        account_id = f"ACC-{acc_idx + 1:05d}"
        sector = rng.choice(sectors, p=weights)
        tenure = int(rng.choice(TENURE_CHOICES_MONTHS))
        disburse_month_idx = int(rng.integers(0, max(n_months - tenure, 1)))
        loan_amount = float(
            max(rng.lognormal(mean=np.log(SECTOR_BASE_LOAN_MEAN[sector]), sigma=0.4), 10_000)
        )
        monthly_installment = round(loan_amount / tenure * 1.12, 2)  # flat add-on rate, illustrative
        sensitivity = SECTOR_FUEL_SENSITIVITY[sector]

        days_in_arrears = 0.0
        active_months = min(tenure, n_months - disburse_month_idx)
        for m in range(active_months):
            month_idx = disburse_month_idx + m
            fuel_shock = macro.loc[month_idx, "fuel_pct_change"]

            shock_effect = sensitivity * fuel_shock * 900.0
            catch_up = rng.random() < 0.06  # borrower clears arrears this month
            noise = rng.normal(0, 4)

            if catch_up:
                days_in_arrears = max(days_in_arrears * 0.15, 0.0)
            else:
                days_in_arrears = max(days_in_arrears * 0.88 + shock_effect + noise, 0.0)
            days_in_arrears = min(days_in_arrears, 500.0)

            classification = days_to_classification(days_in_arrears)

            rows.append(
                {
                    "account_id": account_id,
                    "month": macro.loc[month_idx, "month"],
                    "sector": sector,
                    "loan_amount": round(loan_amount, 2),
                    "tenure_months": tenure,
                    "months_on_book": m + 1,
                    "monthly_installment": monthly_installment,
                    "days_in_arrears": round(days_in_arrears, 1),
                    "classification": classification,
                }
            )

    panel = pd.DataFrame(rows)
    return panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-accounts", type=int, default=800)
    parser.add_argument("--months", type=int, default=48)
    args = parser.parse_args()

    panel = simulate_ledger(n_accounts=args.n_accounts, n_months=args.months)
    panel.to_csv(LEDGER_CSV, index=False)
    print(f"Wrote {len(panel)} account-month rows across {args.n_accounts} accounts -> {LEDGER_CSV}")
    print(panel["classification"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
