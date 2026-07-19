# Methodology

## 1. Target Variable Construction

The SASRA loan classification framework used across regulated Kenyan SACCOs
buckets accounts by days past due:

| Bucket | Days past due |
|---|---|
| Normal | 0–30 |
| Watch | 31–90 |
| Substandard | 91–180 |
| Doubtful | 181–360 |
| Loss | 361+ |

`src/config.SASRA_CLASSIFICATION_THRESHOLDS` is the single source of truth
for these boundaries; both the synthetic label generator and any future
real-data pipeline read from it, so the classification logic cannot drift
between the two.

The model's target, `migrates_worse`, is binary: 1 if an account's bucket
rank next month is strictly worse than its rank this month, 0 otherwise
(including "stays the same" and "improves"). This was chosen over a
5-class multinomial target for two reasons: (a) the business action is
largely the same regardless of which worse bucket an account lands in —
escalate for review — and (b) the rarer classes (Doubtful, Loss) do not
have enough support in most SACCO portfolios to train a reliable multinomial
classifier without heavy oversampling that would itself be a bigger
modeling risk than collapsing to binary.

## 2. Time-Based Validation

Loan performance is a time series problem masquerading as a tabular one. A
random train/test split would let the model see rows from the same shock
period in both train and test, inflating apparent performance because nearby
months are correlated. `src/models/train_model.time_based_split` instead
sorts by month and cuts the panel at a fixed quantile — train is strictly
earlier months, test is strictly later ones. This mirrors how the model
would actually be used: trained on history, scored against a future the
SACCO hasn't experienced yet.

## 3. Class Imbalance

Migration to a worse bucket is, by construction, the minority outcome in any
healthy portfolio. Two independent mitigations are used rather than one:

- Logistic regression uses `class_weight="balanced"`.
- XGBoost uses `scale_pos_weight` computed from the *training* split's own
  class ratio (never the full dataset's ratio, which would leak test-period
  information about how bad the shock turned out to be).

Accuracy is not reported as a headline metric for this reason — a model that
never predicts migration can exceed 90% accuracy on a healthy portfolio
while being useless. ROC-AUC and the classification report's precision/recall
on the positive class are the metrics that matter here.

## 4. Business Framing: Capital-at-Risk

A ranked list of flagged accounts is only useful to a credit committee if it
can be sized in currency terms. `src/models/evaluate.py` reports three
figures for the test period: total loan capital flagged by the model, total
loan capital that actually migrated to a worse bucket, and the overlap
between the two (capital correctly caught before migrating). This translates
a probability threshold into a number a credit committee actually budgets
against.

## 5. Data Governance and Ethics

This repository is intended to live on a public GitHub profile as a
portfolio artifact, which changes the data rules relative to internal SACCO
work:

- **No real member data in any form.** Not raw exports, not aggregates, not
  "anonymized" tables. Kenya's Data Protection Act, 2019 treats financial
  and loan-repayment data as personal data requiring a lawful basis and
  purpose limitation; a public portfolio repo is not that basis. Separately,
  small structured financial datasets (sector + loan size + tenure + rough
  geography) are frequently re-identifiable even after names are stripped,
  so "anonymization" alone is not treated here as sufficient — the synthetic
  generator sidesteps the question entirely by never touching a real record.
- **No institution-specific figures.** Internal PAR30, total portfolio size,
  or officer-concentration figures are never hard-coded into this repo, even
  as "realistic-looking" constants, so that nothing here could be mistaken
  for, or reverse-engineered toward, a real institution's actual exposure.
- **Synthetic data is disclosed, not hidden.** Every notebook and the README
  states plainly that the ledger is simulated. A portfolio project that
  implies real data without saying so misrepresents the work.

## 6. Known Limitations (stated plainly, not hidden in an appendix)

- The synthetic ledger's sector fuel-sensitivity multipliers
  (`SECTOR_FUEL_SENSITIVITY`) are illustrative assumptions, not estimated
  elasticities. Once real EPRA/KNBS/CBK series and a real (properly
  governed, non-public) ledger are available, these should be replaced with
  regression-estimated sensitivities.
- The macro shock in `simulate_macro.py` is a stylized step-change in drift
  and volatility, not a calibrated model of any specific historical event.
  It exists to give the classifier a shock to detect, not to forecast actual
  fuel prices — that forecasting problem is scoped separately (see the
  SARIMAX project in the wider portfolio).
- A single train/test cut is used for simplicity. A production version
  should use walk-forward (rolling-origin) validation across multiple
  cutoffs before any deployment decision.
