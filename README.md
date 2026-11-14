# Case Study III — Millennium BCP: Daily Account Balance Forecasting

Forecast each client's daily account balance for the **next 30 days**, evaluated by **MAPE** (see `problem-description.md`).

## Data
| File | Contents |
|---|---|
| `data/raw/small_dataset.csv` | 50 accounts × 1095 days (subset: the first 50 accounts of the medium dataset) |
| `data/raw/medium-dataset.csv` | 500 accounts × 1095 days |

- *Wide* format: one row per `ACCOUNT_ID`, one column per day (raw labels 2020-01-02 → 2022-12-31). The data is fictitious.
- The **final evaluation uses a large dataset** (not yet available). The small/medium datasets are for development; all code must scale to the large one (vectorised, memory-aware) and run on the group's Lenovo ThinkStation PGX (ARM64 Linux).
- The final forecasts are for the 30 days following the end of the data, for the same accounts.
- How MAPE treats zero/negative balances is not specified — any assumption must be documented.

### Key findings so far (Sprint 1, medium dataset)
- **Date convention changes in Aug 2022:** until label 2022-08-06 a label *D* holds the closing balance of *D − 1*; from 2022-08-07 it is the same-day close. Fixed by realigning the calendar (prepared data: 2020-01-01 → 2022-12-31, 1096 days).
- **Corrupted day:** label 2022-08-05 is the sum of two closing balances; repaired exactly.
- **Missing values (~1%) are all leading** — 25 accounts opened later; no imputation.
- **MAPE risk:** 22% of balances negative, 17% with |balance| < €100, 358/500 accounts change sign.
- **Calendar:** activity on business days only; inflows peak at month end (salaries) and on day 8 (pensions); outflows early in the month.
- **Dynamics:** levels ≈ random walk with level shifts; first differences stationary (d = 1); weak cross-account correlation, confirmed by prewhitened CCFs (no lead-lag pattern).
- **Volatility:** ARCH effects in ~37% of accounts on raw changes but ~27% on AR(p)-prewhitened residuals (the version to cite, per Cowpertwait & Metcalfe §7.4); not enough to justify GARCH for point forecasts.
- **COVID:** the first lockdown (Mar–May 2020) shows a clear dip in activity and outflows; the second (Jan–Feb 2021) a smaller one. Decision: no treatment for models on the recent regime; interruption flag or down-weighting to be tested in Sprint 2 for models that use the whole history.
- **Small vs medium:** the first 50 accounts (= small dataset) give the same structural picture; too few accounts to show the pension pattern or stable baseline scores.
- **Paydays:** largest inflows on the 8th (moved to the next business day at weekends; 8 Jan 2023 is a Sunday → 9 Jan) and on the 28th–30th; ~1 in 5 recurring amounts falls on a fixed day of the month.
- **The 8th behaves like pensions:** in all 36 months the large inflows of days 5–11 peak on the 8th, or on Monday 10th/9th when the 8th is a Saturday/Sunday (the payment is not brought forward on the 8 December holiday). ~13% of accounts are paid there in ≥ 80% of their months; for them the July and December amounts are ~2× an ordinary month (14 payments a year) and the ordinary amount rises ~1%/year.
- **January 2023 insight:** for those accounts the January inflow should arrive on **Monday 9 Jan 2023** with the amount of an **ordinary month, about half of December's** — a model that copies the last monthly cycle would forecast twice the real inflow. July and December are special months.
- **Structure:** monthly cycle ≈30.4 days with harmonics (15.2, 10.1); frequent level shifts (half the accounts changed level in the last ~2 months); January dips like a deep ordinary month; accounts independent (no transfers/duplicates); no credit-limit floors; no long memory (Hurst ≈ 0.37; reading it as anti-persistence is a possible interpretation, not covered by the course references).
- **Dynamics, refined:** random walk day to day, but strong mean reversion over a month (variance ratio ≈ 0.24 at 21 business days); once one level break is allowed (Zivot–Andrews) most accounts are stationary → forecast the level the balance oscillates around.
- **Two kinds of accounts:** the feature map splits accounts mainly by distance from zero — ~230 stable positive accounts (naive median MAPE ≈ 26%) vs ~270 accounts living around zero (≈ 260%).
- **Modelling hints:** flows are lumpy (Syntetos–Boylan) → forecast levels, not flows; seasonal patterns stable across years but too weak for seasonal differencing → calendar regressors/Fourier terms; asinh scaling for global models; ~40% of accounts keep non-linear structure (BDS) → room for tree-based ML.
- **Oracle:** the best constant level per account reaches an official MAPE of ~44% vs ~1 300% for the naive forecast — most of the error is choosing the wrong level for near-zero accounts.
- **Baselines (mean of 5 folds):** median of last 30 days → 71.5% MAPE on |actual| ≥ €100; official pooled MAPE > 1 000% for all baselines (dominated by near-zero actuals).

All Sprint 1 results are saved as CSV in `data/sprint1/` (start with `00_index.csv` and `00_account_features.csv`).

## Deliverables & Tooling
- Each sprint is delivered as a self-contained Jupyter notebook (`.ipynb`) that runs with only the professor's CSV.
- Language: Python 3.13.

## Repository Structure
```
data/raw/          original CSV files (not versioned)
data/processed/    cached/cleaned data generated by the code (not versioned)
data/sprintN/      analysis results saved as CSV by each sprint notebook (00_index.csv lists them)
notebooks/         one self-contained notebook per sprint (the deliverables), developed
                   as percent-format .py scripts and converted with jupytext
reports/figures/   exported figures
```

## Setup
```bash
python3.13 -m venv .venv
# Windows: .venv\Scripts\activate    Linux (PGX): source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name eqc-case-study-iii

# Build and run a sprint notebook from its .py source
cd notebooks
jupytext --to ipynb 01_data_preparation.py
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=eqc-case-study-iii 01_data_preparation.ipynb
```

---

## Project Plan

| Milestone | Due date |
|---|---|
| Sprint 1 — Data Preparation | 2026-11-19 (Thu) |
| Sprint 2 — Modelling: Forecasting Models | 2026-11-26 (Thu) |
| Sprint 3 — Anomalies + Optimizations | 2026-12-03 (Thu) |
| Final Submission | 2026-12-17 (Thu) |

### Sprint 1 — Data Preparation (due 2026-11-19)
Inspect the time series and decide which changes are needed. Deliverable: `notebooks/01_data_preparation.ipynb`.

Done:
- [x] Project setup: folder structure, Python environment, self-contained notebook workflow
- [x] Missing values: located (all leading — accounts opened later); decision: no imputation
- [x] Calendar audit: Aug-2022 date-convention change detected and realigned; corrupted 2022-08-05 repaired
- [x] Zeros, negatives and near-zero balances quantified; three-metric evaluation protocol defined
- [x] Duplicates, impossible values (none), extreme moves (heavy tails; mostly permanent level shifts)
- [x] Descriptive statistics per account and globally, normality (Jarque–Bera) and distributions
- [x] Calendar relationships: weekday, day of month, month, holidays (heatmaps + Kruskal–Wallis)
- [x] Decomposition (MSTL, weekly + monthly) and component strengths
- [x] Stationarity (ADF/KPSS on levels and differences), ACF/PACF, Ljung–Box, ARCH-LM
- [x] Structural changes across years
- [x] Correlations between accounts on daily changes vs levels
- [x] Account segmentation (dormant, near-zero/overdraft, stable positive, volatile positive)
- [x] Documented data-preparation decisions, each backed by evidence
- [x] Validation scheme (rolling origin, 5 folds incl. January 2022) and baseline MAPE
- [x] All analysis results saved as CSV in `data/sprint1/`

- [x] Advanced analyses (Part D): transaction forensics (Benford, cents, round amounts), recurring payments, paydays and the weekend rule for the 8th, overdraft floors, transfers/duplicates (permutation test), spectral analysis, change points, January paths, oracle bounds, error drivers, zero-crossing risk, Hurst exponent
- [x] Specialist pattern-finding tools (Part D, D14–D22): feature map of all accounts (PCA + clustering), Syntetos–Boylan flow classification, Zivot–Andrews unit-root test with a break, seasonal-differencing rule, Lo–MacKinlay variance ratios, asinh/Yeo–Johnson transformations, BDS non-linearity test, seasonal subseries and lag plots, matrix profile (motifs and discords)
- [x] Notebook rewritten as storytelling, with a short summary at the top
- [x] Course-method additions: robust STL per account (§9: flags >10% of a typical account's days, so not usable as an anomaly filter here — fixed 30-day period drifts against real months, lumpy flows), rolling mean/std plots (§10), ARCH-LM on AR(p)-prewhitened residuals next to the raw test (§12), COVID lockdowns as an interrupted series (§13, Hyndman & Rostami-Tabar), prewhitened cross-correlations (§14), caution note on the Hurst exponent (D13)
- [x] Pension evidence for the 8th: month-by-month table, weekend/holiday rule, pension-like accounts, July/December doubling, January 2023 consequence (D3)
- [x] Small dataset check: key facts for the first 50 accounts vs all accounts (§19)

To do:
- [ ] Re-run the notebook after the additions above: build the `.ipynb` from the `.py` with jupytext, execute it and refresh `data/sprint1/` (the committed `.ipynb` and CSVs predate them)
- [ ] Group review of the notebook and its conclusions

### Sprint 2 — Modelling: Forecasting Models (due 2026-11-26)
- [x] Baselines: naive (last value), seasonal naive, mean/median (done in Sprint 1, §17)
- [ ] Statistical models: ETS, ARIMA/SARIMA
- [ ] Inflow of the 8th: July/December as special months; January 2023 inflow on Monday 9 with an ordinary amount (D3)
- [ ] COVID interruption: test an interruption flag or down-weighting (Mar–May 2020, Jan–Feb 2021) for models that use the whole history (§13)
- [ ] ML / global models (e.g. LightGBM with lags and calendar features)
- [ ] Compare models by MAPE under the defined validation scheme

### Sprint 3 — Anomalies + Optimizations (due 2026-12-03)
- [ ] Identify and explain the anomalies found (in the data and in forecast errors)
- [ ] Optimize the best model(s): hyperparameters, features, per-segment treatment of accounts, ensembles
- [ ] Re-evaluate the impact of the optimizations on MAPE

### Final Submission (due 2026-12-17)
- [ ] Final 30-day forecasts for all accounts
- [ ] Delivery of all work (code, results and report/presentation)
