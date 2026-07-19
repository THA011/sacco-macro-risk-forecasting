"""
Evaluate the trained XGBoost model on the held-out (later-months) test split:
classification report on the positive ("migrates to a worse bucket") class,
ROC curve, calibration curve, and a business-framed capital-at-risk metric.

Usage:
    python -m src.models.evaluate [--threshold 0.5]
"""

import argparse

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import classification_report, roc_curve

from src.config import FEATURES_CSV, MODELS_DIR, REPORTS_FIGURES
from src.models.train_model import _feature_columns, time_based_split


def evaluate(threshold: float = 0.5) -> None:
    df = pd.read_csv(FEATURES_CSV, parse_dates=["month"])
    feature_cols = _feature_columns(df)
    _, test_df, cutoff_month = time_based_split(df, test_frac=0.25)

    X_test = test_df[feature_cols]
    y_test = test_df["migrates_worse"]

    model = joblib.load(MODELS_DIR / "xgboost.joblib")
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    print(f"Test period: from {cutoff_month.date()} onward ({len(test_df)} account-months)\n")
    print("Classification report (1 = migrates to a worse SASRA bucket next month):")
    print(classification_report(y_test, preds, digits=3))

    # --- ROC curve ---------------------------------------------------
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC — Migration to Worse Bucket")
    plt.legend()
    plt.tight_layout()
    roc_path = REPORTS_FIGURES / "roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # --- Calibration curve --------------------------------------------
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, marker="o", label="XGBoost")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration — Migration to Worse Bucket")
    plt.legend()
    plt.tight_layout()
    calib_path = REPORTS_FIGURES / "calibration_curve.png"
    plt.savefig(calib_path, dpi=150)
    plt.close()

    # --- Business framing: capital-at-risk flagged vs realized ---------
    capital_flagged = test_df.loc[preds == 1, "loan_amount"].sum()
    capital_realized = test_df.loc[y_test == 1, "loan_amount"].sum()
    capital_caught = test_df.loc[(preds == 1) & (y_test == 1), "loan_amount"].sum()

    print("\nCapital-at-risk framing (test period, KES):")
    print(f"  Flagged by model:                 {capital_flagged:,.0f}")
    print(f"  Actually migrated worse:           {capital_realized:,.0f}")
    print(f"  Correctly flagged before migrating: {capital_caught:,.0f} "
          f"({capital_caught / capital_realized:.1%} of at-risk capital caught)"
          if capital_realized > 0 else "  (no positive cases in test period)")

    print(f"\nSaved: {roc_path}")
    print(f"Saved: {calib_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    evaluate(threshold=args.threshold)


if __name__ == "__main__":
    main()
