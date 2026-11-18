# Case Study III — Millennium BCP: Daily Account Balance Forecasting

Forecast each client's daily account balance for the **next 30 days**, evaluated by **MAPE** (see `problem-description.md`).

## Data
| File | Contents |
|---|---|
| `data/raw/small-dataset.csv` | 50 accounts × 1095 days (subset: the first 50 accounts of the medium dataset) |
| `data/raw/medium-dataset.csv` | 500 accounts × 1095 days |

- *Wide* format: one row per `ACCOUNT_ID`, one column per day (raw labels 2020-01-02 → 2022-12-31). The data is fictitious.
- The **final evaluation uses a large dataset** (not yet available; the professor said it is about 20 times the medium one, i.e. roughly 10 000 accounts). The small/medium datasets are for development; all code must scale to the large one (vectorised, memory-aware) and run on the group's Lenovo ThinkStation PGX (ARM64 Linux).
- The final forecasts are for the 30 days following the end of the data, for the same accounts.
- How MAPE treats zero/negative balances is not specified by the organisers. Following the professor's indication we add €1 to the absolute balance in the denominator: APE = |A − F| / (|A| + €1). The professor's reference value of 224 matches the naive forecast for the last month under this rule (~238% on medium, Dec 2022).

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
- **Oracle:** the best constant level per account reaches an official MAPE of ~43% vs ~740% for the naive forecast — most of the error is choosing the wrong level for near-zero accounts.
- **Baselines (mean of 5 folds, MAPE with |A| + €1):** the ranking depends on the metric. Official MAPE: last value wins (736% vs 2 075% for the 30-day median). Median per-account MAPE: 30-day median wins (66.0% vs 68.1%). MAPE on |actual| ≥ €100: 30-day median wins (71.2% vs 88.3%). The official MAPE is dominated by near-zero actuals.
- **Decision (model selection):** the official MAPE (denominator |A| + €1) ranks models, but a model is accepted only if it does not get worse than the reference on the median per-account MAPE and on the MAPE for |actual| ≥ €100 (`passes_decision_rule`, notebook §17).
- **Decision (segmentation):** the four rule-based segments of §15 (near-zero/overdraft, stable positive, volatile positive, dormant). The two-cluster map of D14 (~270 near-zero, ~230 stable positive) is a consistency check, not the segmentation.
- **Zero floor (Sprint 2):** forecasting exactly 0 gives a MAPE of exactly 100% on every account, so for near-zero accounts every model must beat 100%; no constant level does (weighted median ≈ 100–120% official). The gated calendar profile gets to ~96%; ARIMA is the best model for stable accounts and LightGBM for volatile ones; the full pipeline reaches 72.5% official without the largest contributor (637% with it).
- **EDA statistics for time series:** levels are random-walk-like, so §4 describes levels only by median, min, max and share negative, and reports mean, std, skewness and kurtosis on the daily changes. Jarque–Bera was removed (changes are autocorrelated and ARCH); normality is tested on model residuals in Sprint 2. §7 reports effect sizes (eta squared on per-date cross-account aggregates) instead of Kruskal–Wallis p-values.

All Sprint 1 results are saved as CSV in `data/sprint1/` (start with `00_index.csv` and `00_account_features.csv`).

## Deliverables & Tooling
- Each sprint is delivered as a self-contained Jupyter notebook (`.ipynb`) that runs with only the professor's CSV.
- Language: Python 3.13.

## Repository Structure
```
data/raw/          original CSV files (versioned; the data is fictitious)
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
- [x] Descriptive statistics per account and globally (moments on daily changes, not levels) and distributions
- [x] Calendar relationships: weekday, day of month, month, holidays (heatmaps + effect sizes)
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
- [x] Why the 18 D14 features, and the alternatives tried (D14b): each feature group justified by an earlier section; one-change-at-a-time comparison of feature sets (drop each group, only distance-from-zero, tsfresh with 744 automatic features), scalings (rank-normal, standardised, robust) and algorithms (k-means, Ward, Gaussian mixture), judged by silhouette, Calinski–Harabasz, Davies–Bouldin, subsample stability and the share of forecast difficulty explained. Distance from zero drives the map (eta² 0.34 → 0.18 without it); robust scaling wins on silhouette but explains nothing (0.02); Ward is unstable (0.44); tsfresh loses on every measure; the rule-based segments explain the most difficulty (0.51)
- [x] D14 clusters on the principal components keeping 80% of the variance (5 components: persistence, monthly cycle, inactivity near zero, two short-term dynamics); D14b compares no PCA / 70% / 90% (same groups, ARI 0.97–0.98, same explained difficulty) and tsfresh with PCA (still 0.21)
- [x] MAPE switched to the professor's rule (|A| + €1 in the denominator) in Sprint 1 (§5, D9, D10 oracle) and Sprint 2 (metric, MAPE-optimal weighted median, GBM weights); Sprint 1 re-run
- [x] Why four segments when the data show two clusters (D14b): near-zero and stable segments sit on opposite sides of the k = 2 border (178/195 and 125/138), volatile accounts straddle it (79/87); forcing k = 3 or 4 does not recover the segments (ARI 0.20–0.23) nor explain more difficulty (≈0.35 vs 0.51); border accounts (calm small balances clustered with stable ones) have the near-zero error level, so the rules classify better for forecasting. Clusters renamed by behaviour: "persistent, calm" vs "erratic, crossing zero"

To do:
- [x] Re-run the notebook after the additions above: build the `.ipynb` from the `.py` with jupytext, execute it and refresh `data/sprint1/` (the committed `.ipynb` and CSVs predate them)
- [x] Notebook rebuilt and re-run after the EDA corrections (§4, §7), the decision rule (§17) and the segmentation text (§15); `data/sprint1/` refreshed
- [ ] Group review of the notebook and its conclusions

### Sprint 2 — Modelling: Forecasting Models (due 2026-11-26)
- [x] Baselines: naive (last value), seasonal naive, mean/median (done in Sprint 1, §17)
- [x] Level experiment for near-zero accounts (`notebooks/02_modelling.ipynb`, results in `data/sprint2/`): the MAPE-optimal weighted median cuts the near-zero official MAPE from ~1 600% (last value) to ~120%, but a forecast of exactly 0 scores 100% on every measure and nothing constant beats it. The two-step sign classifier + MAE size (~610%) and + MAPE size (~160%) both lose to the plain weighted median. On stable positive accounts the weighted median is worse than the last value, so segmenting is justified.
- [x] Beat the zero floor on near-zero accounts with a calendar profile (weighted median of same-day-of-month balances) gated by sign agreement: official MAPE 95.9% (tuned gate) / 96.7% (q=0.9) vs 100% for zero, lower than zero in 5/5 and 3/5 folds, but the gate lets the profile through on only 11–30% of account-days
- [x] Stable accounts ending near zero: one unpredictable account (€134k → €2) produces ~all of one fold's segment error; a warning flag (min of last year < 20% of median) isolates accounts that are not really stable (typical error 26% vs 8%)
- [x] Revised segment rule (stable accounts must keep their lowest balance of the last year above 20% of the median; flagged accounts go to the near-zero group) and results reported with and without the largest contributor
- [x] Volatile positive accounts: global LightGBM (scaled deviations from the 30-day median, MAPE-weighted L1 loss, calendar-profile feature, small forecasts zeroed with a tuned threshold) → official 97.7%, typical 55.2%, |A|≥100 53.0%; accepted vs last value and 30-day median. Dormant accounts: none in the medium dataset, strategy = last value (untested)
- [x] Segment pipeline (gated calendar profile + guards for near-zero, ARIMA for stable, GBM for volatile): 637% official on all accounts (one account at €134k → €2 decides a fold) / 72.5% without the largest contributor; accepted only vs the zero forecast — not vs last value, because the median per-account MAPE is 74.5% vs 68.8% (medians do not add up: near-zero accounts the last value forecast at 30–50% move to ~100%, even though every segment's median improves)
- [x] WAPE added as a reported (not deciding) measure: the pipeline misses ~17% of the euros vs ~12% for the last value; forecasting zero is cheap in MAPE and expensive in euros (§2, §7, §13)
- [x] Per-account choice (last value / guarded profile / zero) in the near-zero group on the two inner windows (§12): ~45 accounts per fold get their last value back, median per-account MAPE 74.5% → 70.4%, but official MAPE 637% → 664% (72.5% → 81.5% without the largest contributor) — not accepted; kept as a measured trade-off
- [x] ETS (additive, weekly season) and AutoARIMA (weekly season) per account with statsforecast, in parallel, on the last 365 days (§11): on stable accounts ARIMA beats the last value (official 16.3% vs 17.0% without the largest contributor, typical 7.2% vs 7.7%) and is adopted; ETS ≈ last value; on volatile accounts ARIMA beats the last value but not the GBM (296%/64% vs 98%/55%)
- [x] More comparisons in `02_modelling.py` (not yet re-run): warning-flag threshold sensitivity (5–50%, §9), LightGBM loss and tree-size variants (unweighted L1, L2, 7/31 leaves, §10), ETS/ARIMA with and without the weekly season (§11), an optional tsfresh-features GBM (`RUN_TSFRESH_GBM = False` by default; on medium it was rejected: 99.3% vs 97.7% official with zeroing, worse WAPE, ~10 min extra), and a decision log of every choice with the alternatives compared (§14)
- [ ] Re-run `02_modelling` (jupytext + nbconvert, ~12–14 min) — needed for the new MAPE (|A| + €1) too — and update the numbers quoted in the §7–§13 text from the new tables
- [x] Ask the professor how the MAPE treats zeros: add €1 to |actual| in the denominator (adopted in both notebooks)
- [ ] Confirm with the professor that the reference value of 224 is the naive forecast under the |A| + €1 rule (we get ~238% on medium for Dec 2022)
- [x] Metric sensitivity to the zero offset on the baselines (€0.01 to €10, Sprint 1 D9b)
- [ ] Metric sensitivity, rest: sMAPE and MASE as references, and the offset sensitivity applied to the Sprint 2 pipeline — does the ranking of models depend on the metric?
- [ ] Level from the current regime only: compute levels (last value, medians, weighted medians, profiles) on the history since the last change point (D7, D16) instead of fixed 30/90/365-day windows; compare with the decision rule
- [ ] Calendar table for the models: business days, Portuguese holidays, the 8th with the weekend rule, paydays on the 28th–30th, July/December double pensions; feed it to SARIMAX, the GBM and the calendar profile
- [ ] asinh transformation of the balance (D19) for ARIMA/ETS and the GBM on volatile accounts, with back-transformation bias correction
- [ ] Inflow of the 8th: July/December as special months; January 2023 inflow on Monday 9 with an ordinary amount (D3)
- [ ] COVID interruption: test an interruption flag or down-weighting (Mar–May 2020, Jan–Feb 2021) for models that use the whole history (§13)
- [x] ML / global models: LightGBM for volatile accounts (§10)
- [x] tsfresh features for the volatile-account GBM: tested (30 of 339 relevant features, selected before the first fold), no gain on any MAPE measure — kept as an optional experiment in §10
- [x] Compare models by MAPE under the defined validation scheme, using the decision rule (official MAPE ranks, no worsening on the two companion measures), with and without the largest contributor
- [ ] Test normality (Jarque–Bera / Q–Q) on the residuals of the ARIMA models

### Sprint 3 — Anomalies + Optimizations (due 2026-12-03)
- [ ] Identify and explain the anomalies found (in the data and in forecast errors)
- [ ] Suspicious-activity scan (daily balances only, so each typology is searched through its signature in the net daily change):
  - [ ] Structuring / smurfing: repeated inflows of similar size just below a reporting threshold (e.g. €9 500–9 990) on consecutive days
  - [ ] Pass-through / money-mule accounts: large inflow followed by a near-equal outflow within 1–3 days; turnover far above the average balance
  - [ ] Unexplained large inflows: net change many times the account's usual income (robust z-score against its own history)
  - [ ] Account draining / takeover: sudden fall to near zero after a stable history, at once or in several withdrawals (candidate: the €134k → €2 account of Sprint 2 §9)
  - [ ] Salami slicing: many small, near-identical debits; cents analysis (extends D1)
  - [ ] Transfers between accounts and round-tripping: equal and opposite net changes in two accounts with a 0–2 day lag; cycles A→B→C→A in the transfer graph (extends D5, which tested lag 0 globally)
  - [ ] Funnel accounts: one account whose inflows match outflows of many others
  - [ ] Kiting: pairs of accounts with mirrored oscillations (strong negative correlation of daily changes)
  - [ ] Dormant accounts reactivated: long flat spell followed by a burst of activity
  - [ ] Bust-out / overdraft abuse: a new minimum far below the historical floor with no recovery (D4 found no floors)
  - [ ] Pension or payroll fraud: duplicated payments on the 8th, or pension accounts that keep accumulating with no outflows (D3)
  - [ ] Movements on weekends or holidays, when the bank does not process (§7: activity on business days only)
  - [ ] Fabricated amounts: Benford, round amounts and repeated exact amounts per account (extends D1)
  - [ ] Change of pattern: loss of the monthly cycle or of the salary, abrupt change of profile (D7, D16)
- [ ] Suspicion score per account combining the indicators above; inspect the top accounts by hand and report them as suspicious patterns (no ground truth to confirm fraud)
- [ ] Use the flags in the forecasts: exclude or replace flagged days (pass-through spikes, one-off inflows) before computing levels and profiles; forecast pass-through accounts at their base level; restart drained accounts from their new level
- [ ] Anomaly cleaning of the training data: a simple rule for isolated jumps that revert within a few days (the robust STL of §9 flags too many days to be used as a filter)
- [ ] Near-zero group: a better way to find the accounts that a simple level forecasts well (more than two inner windows, or account features instead of recent errors only)
- [ ] Optimize the best model(s): hyperparameters, features, per-segment treatment of accounts, ensembles
- [ ] Re-evaluate the impact of the optimizations on MAPE

### Final Submission (due 2026-12-17)
- [ ] Final 30-day forecasts for all accounts
- [ ] Delivery of all work (code, results and report/presentation)
