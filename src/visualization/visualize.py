"""
Reusable plotting helpers for notebooks and reports: PAR trend vs fuel price
shock, and model feature importance. Kept separate from evaluate.py because
these are exploratory/reporting plots, not part of the model evaluation
contract.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPORTS_FIGURES


def plot_par_vs_fuel_price(features_df: pd.DataFrame, out_name: str = "par_vs_fuel_price.png"):
    """Monthly PAR30 rate overlaid against fuel price, to visualize co-movement."""
    monthly = (
        features_df.groupby("month")
        .agg(par30_rate=("par30_flag", "mean"), fuel_price=("fuel_price_kes_per_litre", "mean"))
        .reset_index()
    )

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax2 = ax1.twinx()

    ax1.plot(monthly["month"], monthly["par30_rate"], color="firebrick", label="PAR30 rate")
    ax2.plot(monthly["month"], monthly["fuel_price"], color="steelblue", linestyle="--", label="Fuel price (KES/L)")

    ax1.set_ylabel("PAR30 rate", color="firebrick")
    ax2.set_ylabel("Fuel price (KES/litre)", color="steelblue")
    ax1.set_xlabel("Month")
    plt.title("Portfolio-at-Risk (30d) vs Fuel Price")
    fig.tight_layout()

    out_path = REPORTS_FIGURES / out_name
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_importance(feature_names, importances, out_name: str = "feature_importance.png", top_n: int = 10):
    """Horizontal bar chart of the top-N most important model features."""
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1])[-top_n:]
    names, values = zip(*pairs)

    plt.figure(figsize=(7, 5))
    plt.barh(names, values, color="seagreen")
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances — XGBoost")
    plt.tight_layout()

    out_path = REPORTS_FIGURES / out_name
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
