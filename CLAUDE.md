# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Team case-study competition project (Moodle course, section 8608): predict a customer's
**Billing (BIL)** system configuration from their **CRM** system configuration. A 4-Play
telecom subscription gets provisioned across several systems (CRM, TV, Internet, Billing);
human/automatic errors let these drift out of sync, so the model's job is to predict the
*correct* BIL configuration and help catch provisioning errors.

- Input: 745 binary `CRM_*` columns. Output: 731 binary `BIL_*` columns.
- Metric: **Exact Matching Ratio (EMR)** — a row only counts as correct if all 731 BIL labels
  match. No partial credit, so per-label accuracy/F1 do not align with what's being optimized.
- `train.csv` contains real provisioning errors; the brief says test pairs are the correct
  ones. CRM/BIL share no id vocabulary, so this is a real learning problem.

## Working rules (agreed with the team lead — follow them)

- **Experiments are Python scripts in `scripts/`.** Do **not** edit notebooks unless the user
  explicitly approves it in the current conversation.
- **Work step by step:** propose the next step, run it after approval, report, stop.
- **Log every step** (plan → run → result → decision + justification) in
  `docs/sprint3/SPRINT3_DIAGNOSTIC_LOG.md`; it is the audit trail presented to the professor.
- **`solution.csv` is never used to learn or select** (no training, features, settings,
  thresholds, or test-error breakdowns). It is read only at the end of
  `scripts/train_final.py` as a post-selection diagnostic. Test *inputs* (`test.csv`, no
  labels) may be compared with train inputs; disclose it when it shapes a choice.

## Setup and commands

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
```

Always run from the repo root with the venv's Python (`.venv/Scripts/python`); the global
Python lacks the right packages. No test suite or linter exists.

```bash
.venv/Scripts/python scripts/train_final.py 4 5      # current champion -> data/derived/submission_sprint3_k4_mcs5.csv
.venv/Scripts/python scripts/make_leaderboard.py     # regenerate docs/EXPERIMENT_LEADERBOARD.md
.venv/Scripts/python scripts/exp8_retune.py 42 5:1   # example experiment: seed, min_child_samples:reg_lambda
```

Model fits take ~2–4 min each — run long jobs in the background. GitHub CLI is installed at
`C:\Program Files\GitHub CLI\gh.exe` (logged in as `therealblitz00`, repo `therealblitz00/ECQ`);
the working branch is `ManuelS`.

## Code layout

- `src/` — shared library, imported by every script:
  `data.py` (loads raw + Sprint 1 exports once into `data/cache/arrays.npz`, uint8),
  `validation.py` (packbits row keys, consensus targets, `test_like_mask`, `rare_pack_count`,
  `clean_like_mask`, grouped split), `metrics.py` (EMR, Hamming, `multilabel_report`,
  `top_offending_columns`), `models.py` (`BinaryRelevanceLGBM`, `CoOccurrenceChainLGBM`),
  `experiment.py` (`setup()` shared split/targets/masks, `score()`, `log()` → `data/cache/results.jsonl`).
- `scripts/` — one script per experiment (`exp1`–`exp10`), diagnostic (`diag1`–`diag3`),
  `train_final.py`, `make_leaderboard.py`. Kept flat: the Sprint 2 notebook imports
  `scripts.exp1_lookup_knn`, and reads `data/derived/sprint2_experiment_results.csv`
  (don't rename either).
- `docs/` — `ARCHITECTURE_AND_ROUTING.md` (module/script map), `EXPERIMENT_LEADERBOARD.md`
  (generated — never edit by hand), `sprint1/`, `sprint2/`, `sprint3/` write-ups.
- `notebooks/` — one notebook per sprint (+ `archive/`, `exports/`).

## Data

`data/*.csv`, `data/cache/` and `data/derived/submission_*.csv` are gitignored. Place
`train.csv` (187,442 rows), `test.csv` (97,100), `solution.csv` and `sampleSubmission.csv`
in `data/`. Columns are strictly binary — always load with an explicit `int8`/`uint8` dtype
map (use `src/data.load()`), never pandas' default `int64`.

Column families: one-hot categorical blocks `BUSINESS_LINE`, `SUBSCRIBER_TYPE`,
`SUBSCRIBER_STATUS`, and hundreds of sparse `*_<id>_PACK` flags. Pack ids encode no usable
grouping; the CRM→BIL pack mapping is many-to-one.

## Methodology (what has been established — don't re-derive)

- **Validation:** `GroupShuffleSplit` on the exact CRM configuration (20%, seed 42; confirm
  on seed 7). Only 0.006% of test rows have a CRM config seen in train, so lookup approaches
  don't transfer. Score against **consensus targets** (modal full BIL configuration per CRM
  config).
- **Primary metric: clean-like validation EMR** (test-like rows with < 3 CRM packs rarer
  than 0.2%); it tracks test EMR. A choice must hold on both seeds; within one standard
  error (~0.1 pt) prefer the simpler / more regularised setting.
- **Training data corruption:** ~7–8% of train rows have randomly injected rare packs (CRM
  and BIL) and/or impossible one-hot categoricals (two statuses, value `0`). Drop rows that
  are not test-like or carry ≥ 4 rare packs before training.
- **Current champion:** per-label LightGBM on the raw 745 CRM columns, unique configs +
  consensus labels, corrupted rows dropped, `min_child_samples=5`, `reg_lambda=1`
  (never 0 — rare columns blow up), threshold 0.5, then the 42 additive rules from
  `data/derived/additive_rules_v2.json`. Clean-like val 96.81% / 96.33%; test diagnostic
  97.27% (benchmark ~98%).
- **Tried and rejected** (see leaderboard / logs): PCA/SVD features, Section 10 column
  collapsing, multiplicity weights, k-NN, label powerset, classifier chains, blends,
  rare-column thresholds, confident-learning cleaning, repairing corrupted rows, more capacity.

## Sprint plan

| Sprint | Focus | Weight | Due | Status |
|---|---|---|---|---|
| 1 | Pre-processing (`notebooks/sprint1_preprocessing_v3.ipynb`) | 40% | 22/09/2026 | done |
| 2 | Modeling (`notebooks/sprint2_modeling.ipynb`) | 30% | 29/09/2026 | done (committed copy needs re-execution) |
| 3 | Optimization & Explainability | 30% | 06/10/2026 | optimisation done; **next: Step 6, SHAP explainability** (`scripts/explain_shap.py`), then the Sprint 3 notebook after approval |
