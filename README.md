# SACCO Macro-Energy Portfolio Risk & Arrears Forecasting

A credit-risk data science project that predicts which SACCO loan accounts are
likely to migrate into higher SASRA risk classifications (Watch → Substandard →
Doubtful → Loss) when macroeconomic shocks — principally EPRA fuel price
movements — compress borrower disposable income.

## 1. Problem Statement

A SACCO's Portfolio at Risk (PAR) does not move independently of the economy.
When diesel and petrol prices rise, borrowers in transport, agriculture, and
boda boda sectors see their operating margins compressed first, and loan
servicing is typically the first discretionary payment to slip. Static credit
appraisal — the industry default — has no mechanism to anticipate this. It
scores a borrower once, at disbursement, and does not revisit that score as
macro conditions change.

This project builds an early-warning classifier: given a loan account's
attributes (sector, tenure, exposure) and a rolling window of macro indicators
(fuel price deltas, inflation, FX), predict the probability that the account
transitions to a worse SASRA classification bucket within the next reporting
period. The output is a ranked watchlist a credit committee can act on before
the arrears materialize, not after.

## 2. Why This Project (and not the other three candidates)

Four project directions were scoped: (1) this forecasting model, (2) a
SARIMAX fuel-price/inflation elasticity model in R, (3) an automated
risk-based credit appraisal engine with a macro-stress modifier, and (4) a
Power BI PAR/macro stability dashboard. This one was prioritized because:

- It is the only one of the four that is a genuine predictive modeling
  exercise — the appraisal engine (#3) and dashboard (#4) are valuable but sit
  closer to engineering and BI than to statistical inference, and are already
  represented elsewhere in this portfolio.
- It maps directly onto SASRA's existing loan classification framework, which
  means the target variable is real and auditable, not invented for the sake
  of having a label.
- It compounds on domain context already held from live PAR30, portfolio
  size, and officer-concentration work, without touching any live member
  data (see §6, Data Governance).
- It is defensible in an interview: time-aware validation, class imbalance,
  and macro feature engineering are exactly what a credit-risk-data-science
  screening will probe.

## 3. Data

| Source | Content | Access in this repo |
|---|---|---|
| EPRA Monthly Petroleum Pricing | Pump prices (petrol, diesel, kerosene) by month | `src/data/fetch_external.py` (stub, real URL, run locally) |
| KNBS CPI / Inflation releases | Monthly inflation, cost-of-living indices | `src/data/fetch_external.py` (stub) |
| CBK Indicative Exchange Rates | USD/KES monthly average | `src/data/fetch_external.py` (stub) |
| SACCO loan ledger | Member-level loan, sector, arrears, classification | **Synthetic only** — `src/data/simulate_ledger.py` |

**Nothing in this repository is, or is derived from, a real SACCO member
record.** The ledger generator produces statistically realistic but entirely
synthetic accounts. See `docs/methodology.md §5` for why this boundary is
non-negotiable for a public repo.

The three macro fetchers are shipped as working stubs with the real
government source URLs documented in code comments. They are not executed in
CI or in the sandbox this repo was scaffolded in, because that environment's
network egress is restricted to package registries. Run them from a normal
internet connection: `python -m src.data.fetch_external --all`.

## 4. Method

1. **Simulate / ingest** — synthetic ledger (`simulate_ledger.py`) +
   synthetic macro series (`simulate_macro.py`) for local development;
   swap in real macro series via `fetch_external.py` when available.
2. **Feature engineering** (`build_features.py`) — sector exposure encoding,
   month-on-month and quarter-on-quarter fuel price deltas, lagged macro
   features (t-1, t-2, t-3), rolling PAR30/PAR90 flags per account.
3. **Modeling** (`train_model.py`) — baseline logistic regression, then
   gradient boosting (XGBoost), both trained on a **time-based split**
   (train on earlier months, validate on later ones — never a random split,
   since the whole point is forecasting under a shock the model hasn't seen
   yet). Class weights handle the natural rarity of the Loss bucket.
4. **Evaluation** (`evaluate.py`) — precision/recall on the at-risk classes
   specifically (not just overall accuracy, which is misleading under class
   imbalance), ROC-AUC, calibration curve, and a business-framed metric:
   capital-at-risk flagged versus capital-at-risk realized.

Full methodology, including the exact SASRA day-past-due thresholds used to
construct the classification labels, is in `docs/methodology.md`.

## 5. Repository Map

```
sacco-macro-risk-forecasting/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # simulated ledger + macro CSVs land here (gitignored)
│   ├── external/          # real EPRA/KNBS/CBK pulls land here (gitignored)
│   ├── interim/           # merged, pre-feature-engineering (gitignored)
│   └── processed/         # model-ready feature tables (gitignored)
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   └── 02_modeling_and_evaluation.ipynb
├── src/
│   ├── config.py                    # paths, SASRA thresholds, sector list, seed
│   ├── data/
│   │   ├── simulate_ledger.py       # synthetic loan ledger generator
│   │   ├── simulate_macro.py        # synthetic fuel/CPI/FX series
│   │   └── fetch_external.py        # real-data fetch stubs (EPRA/KNBS/CBK)
│   ├── features/
│   │   └── build_features.py        # merge + lag/delta feature engineering
│   ├── models/
│   │   ├── train_model.py           # logistic regression + XGBoost, time split
│   │   └── evaluate.py              # metrics, ROC, calibration, capital-at-risk
│   └── visualization/
│       └── visualize.py             # PAR trend, feature importance, ROC plots
├── tests/
│   ├── test_simulate_ledger.py
│   └── test_build_features.py
├── docs/
│   └── methodology.md               # SASRA thresholds, stats methodology, ethics
├── reports/
│   └── figures/                     # exported charts (gitignored except .gitkeep)
├── presentation/
│   └── (pitch deck lives here — see chat deliverable)
└── .github/workflows/ci.yml         # pytest on every push
```

## 6. Data Governance (read before you push this publicly)

This repo is designed to be pushed to a **public** GitHub profile as a
portfolio artifact. Two rules keep that safe:

1. **No real member data, ever** — not raw, not aggregated, not "anonymized."
   Anonymization of small, structured financial datasets is reversible more
   often than people assume (sector + tenure + loan amount + rough location
   is frequently enough to re-identify someone). The synthetic generator
   exists specifically so the statistical *shape* of the problem is realistic
   without any real individual behind a row.
2. **No RUPSA-specific figures** — PAR30, portfolio size, or officer
   concentration numbers used in internal work stay internal. This repo
   never hard-codes them; `config.py` exposes them as parameters you can set
   locally (and gitignore) if you ever want to re-run this against real
   figures for your own analysis, but the defaults and all committed data
   are synthetic.

## 7. Setup

```bash
git clone https://github.com/THA011/sacco-macro-risk-forecasting.git
cd sacco-macro-risk-forecasting
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.simulate_ledger
python -m src.data.simulate_macro
python -m src.features.build_features
python -m src.models.train_model
python -m src.models.evaluate

pytest -q
```

## 8. Roadmap

- [ ] v0.1 — synthetic pipeline + baseline logistic regression (this repo)
- [ ] v0.2 — XGBoost + PyCaret experiment comparison, SHAP feature attribution
- [ ] v0.3 — swap in real EPRA/KNBS/CBK series via `fetch_external.py`
- [ ] v0.4 — FastAPI scoring endpoint (`/predict`) for a ranked watchlist
- [ ] v0.5 — survival/hazard framing (time-to-default) as a stretch model

## Author

Zacheas Mwatha (Mwatha Maina) — Credit Officer/Intern, RUPSA Regulated NWDT
Sacco Ltd · [github.com/THA011](https://github.com/THA011)
