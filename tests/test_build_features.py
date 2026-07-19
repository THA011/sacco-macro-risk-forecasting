import pandas as pd

from src.config import CLASSIFICATION_ORDER
from src.features.build_features import CLASS_RANK, _add_account_level_features


def _toy_ledger():
    # One account, 4 months, classification worsens then improves.
    return pd.DataFrame(
        {
            "account_id": ["ACC-00001"] * 4,
            "month": pd.date_range("2024-01-01", periods=4, freq="MS"),
            "sector": ["transport"] * 4,
            "loan_amount": [100_000] * 4,
            "tenure_months": [24] * 4,
            "months_on_book": [1, 2, 3, 4],
            "monthly_installment": [5000] * 4,
            "days_in_arrears": [0, 40, 100, 20],
            "classification": ["Normal", "Watch", "Substandard", "Watch"],
        }
    )


def test_migrates_worse_label_direction():
    df = _add_account_level_features(_toy_ledger())

    # Row 0 (Normal -> Watch): worse next month => 1
    # Row 1 (Watch -> Substandard): worse next month => 1
    # Row 2 (Substandard -> Watch): improves next month => 0
    # Row 3 has no "next month" and must be dropped.
    assert len(df) == 3
    labels = df.sort_values("months_on_book")["migrates_worse"].tolist()
    assert labels == [1, 1, 0]


def test_class_rank_is_monotonic_with_classification_order():
    ranks = [CLASS_RANK[c] for c in CLASSIFICATION_ORDER]
    assert ranks == sorted(ranks)


def test_par_flags_consistent_with_days_in_arrears():
    df = _add_account_level_features(_toy_ledger())
    assert (df.loc[df["days_in_arrears"] >= 30, "par30_flag"] == 1).all()
    assert (df.loc[df["days_in_arrears"] < 30, "par30_flag"] == 0).all()
