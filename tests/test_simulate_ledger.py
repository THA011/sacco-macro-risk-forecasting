import pandas as pd

from src.config import CLASSIFICATION_ORDER, SECTORS
from src.data.simulate_ledger import days_to_classification, simulate_ledger


def test_days_to_classification_boundaries():
    assert days_to_classification(0) == "Normal"
    assert days_to_classification(30) == "Normal"
    assert days_to_classification(31) == "Watch"
    assert days_to_classification(90) == "Watch"
    assert days_to_classification(91) == "Substandard"
    assert days_to_classification(180) == "Substandard"
    assert days_to_classification(181) == "Doubtful"
    assert days_to_classification(360) == "Doubtful"
    assert days_to_classification(361) == "Loss"
    assert days_to_classification(10_000) == "Loss"


def test_simulate_ledger_schema_and_no_name_field():
    df = simulate_ledger(n_accounts=25, n_months=12, seed=1)

    assert len(df) > 0
    expected_cols = {
        "account_id", "month", "sector", "loan_amount", "tenure_months",
        "months_on_book", "monthly_installment", "days_in_arrears", "classification",
    }
    assert expected_cols.issubset(set(df.columns))

    # No PII field should ever exist on this schema.
    forbidden = {"name", "member_name", "full_name", "id_number", "phone", "national_id"}
    assert forbidden.isdisjoint(set(c.lower() for c in df.columns))

    assert set(df["classification"].unique()).issubset(set(CLASSIFICATION_ORDER))
    assert set(df["sector"].unique()).issubset(set(SECTORS))
    assert (df["days_in_arrears"] >= 0).all()
    assert (df["loan_amount"] > 0).all()


def test_simulate_ledger_reproducible_with_seed():
    df1 = simulate_ledger(n_accounts=10, n_months=6, seed=7)
    df2 = simulate_ledger(n_accounts=10, n_months=6, seed=7)
    pd.testing.assert_frame_equal(df1, df2)
