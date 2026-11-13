# CLAUDE.md

Master's course project (EQC, FEP × Millennium BCP): forecast each account's daily balance for the **next 30 days**, scored by **MAPE**. Full brief in `problem-description.md`; plan, dates and checklists in `README.md`.

## Hard rules
- **Everything written to the project is in English** (code, comments, markdown, plots, reports). The user chats in Portuguese — replies in chat follow the user's language.
- **Each sprint deliverable is one self-contained `.ipynb`** that runs end-to-end with only the professor's CSV: no imports from local `.py` modules, no pre-built caches. All helper functions live in the notebook. The data path is a single `DATA_PATH` parameter in the first code cell.
- **Code must scale to the large dataset** (final evaluation; not yet in the repo; run on the group's ThinkStation PGX — ARM64 Linux, NVIDIA GB10, 128 GB). Vectorise over the accounts × days matrix; no per-account Python loops in pandas where avoidable; per-account models run in parallel; N×N or per-account plots use a fixed random sample.
- **MAPE treatment of zero/near-zero actuals is unspecified** by the organisers — any choice must be explicit and justified in the notebook.

## Notebook workflow
1. Develop as a percent-format script `notebooks/NN_name.py` (`# %%` code cells, `# %% [markdown]` text cells).
2. Run it as a script on the small dataset, then medium.
3. Build the notebook: `.venv\Scripts\jupytext --to ipynb notebooks/NN_name.py`
4. Verify it runs headless and save outputs: `.venv\Scripts\jupyter nbconvert --to notebook --execute --inplace notebooks/NN_name.ipynb`
The `.py` is the source of truth; never hand-edit the `.ipynb`.

## Data
- `data/raw/small_dataset.csv` (50 accounts) — exactly the first 50 rows of `data/raw/medium-dataset.csv` (500 accounts).
- Wide format: `ACCOUNT_ID` + one column per day, 2020-01-02 → 2022-12-31 (1095 days).
- Known issues: ~1% missing, ~22% negative balances, a few exact zeros; balances often flat for days.

## Environment
- Python 3.13 venv in `.venv` (Python 3.14 is the system default — don't use it; some TS libraries lag). Pinned deps in `requirements.txt`.
- Windows dev machine (PowerShell / Git Bash); final runs on Linux ARM64 — avoid Windows-only paths (use `pathlib`, relative paths).

## Course methods (use their vocabulary)
EDA for time series; stationarity (ADF/KPSS) and ACF/PACF; decomposition and smoothing (STL, exponential smoothing); AR/ARMA/ARIMA/SARIMA and forecasting with SARIMA; multivariate time series; ML for time series.

## Schedule
Sprint 1 Data Preparation — 2026-11-19 · Sprint 2 Modelling — 2026-11-26 · Sprint 3 Anomalies + Optimizations — 2026-12-03 · Final submission — 2026-12-17.

## Related repo
Previous course projects: GitHub `therealblitz00/ECQ` (branches `Projeto-I`, `parfois`; old branches archived as `archive/*` tags — never delete those tags).
