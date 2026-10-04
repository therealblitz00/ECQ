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
.venv/Scripts/python scripts/sprint3/exp8_retune.py 42 5:1   # example experiment: seed, min_child_samples:reg_lambda
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
- `scripts/` — `train_final.py` and `make_leaderboard.py` at the top; experiments grouped
  by sprint in `scripts/sprint2/` (exp1–exp5) and `scripts/sprint3/` (diag1–3, exp6–exp11,
  `explain_shap.py`). Scripts in subfolders put the repo root on `sys.path` (three
  `dirname`s) and import each other as `scripts.sprintN.<module>`. The Sprint 2 notebook
  imports `scripts.sprint2.exp1_lookup_knn` and reads
  `data/derived/sprint2_experiment_results.csv` (don't rename either).
- `docs/` — `ARCHITECTURE_AND_ROUTING.md` (module/script map), `EXPERIMENT_LEADERBOARD.md`
  (generated — never edit by hand), `sprint1/`, `sprint2/`, `sprint3/` write-ups.
- `notebooks/` — one notebook per sprint; `sprint3_final.ipynb` is the final deliverable
  (+ `archive/`, `exports/`).

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
- **Final model** (in `notebooks/sprint3_final.ipynb`): per-label LightGBM on the raw 745 CRM
  columns, unique configs + consensus labels, corrupted rows dropped, `min_child_samples=5`,
  `reg_lambda=1` (never 0 — rare columns blow up), threshold 0.5, **no additive rules**.
  Clean-like val 96.79% / 96.30%; test 97.21% (benchmark ~98%). The Sprint 1 rules
  (`additive_rules_v2.json`) add only +0.02 on validation and were mined on all of train, so
  they were dropped; `scripts/train_final.py 4 5` still applies them (test 97.27%).
- **The final notebook may read only `train.csv`, `test.csv` and `solution.csv`** (the
  professor's files; `solution.csv` only in the last scoring cell) and must not import from
  `src/` or `scripts/`. Earlier experiment results appear there as markdown tables.
- **Uncertainty (Step 9):** rows of one configuration are not independent — compute SEs and
  comparisons by resampling configurations (paired bootstrap in
  `scripts/sprint3/review_checks.py`). Validation SE ≈ 0.35–0.38 pt; 5-fold GroupKFold
  96.83% ± 0.20; test 97.21% [97.11, 97.31] (every test row is its own configuration).
- **Adoption rule:** a change is adopted only if paired-significant on both seeds, ≥ ~0.1 pt,
  leak-free, and it holds in 5-fold. The rare-column threshold (0.45) is real but +0.03 →
  not adopted.
- **Business use (Steps 9–10):** flag customers whose actual billing disagrees with the
  prediction; queue rare-product customers last, then fewer disagreeing items first. The rule
  was designed on seed 42 (after confidence ranking failed there), so seed 7 is the honest
  number: top 2% catch 83% [73–92%] of known errors (~40× random). Synthetic 1–3-item errors:
  ~92%, but ~80% for unique-configuration customers.
- **Baselines (same metric):** logistic regression untuned 95.97 / 94.61, tuned (C = 10)
  96.56 / 96.15 — LightGBM better by +0.23 / +0.16 (borderline on seed 7); rules floor
  67.9 / 74.2; k-NN 56.8 / 49.6.
- **Tried and rejected** (see leaderboard / logs): PCA/SVD features, Section 10 column
  collapsing, multiplicity weights, k-NN, label powerset, classifier chains, blends,
  rare-column thresholds, confident-learning cleaning, repairing corrupted rows, more capacity,
  basket-size count features. Not built (by decision): multi-output neural net.

## Sprint plan

| Sprint | Focus | Weight | Due | Status |
|---|---|---|---|---|
| 1 | Pre-processing (`notebooks/sprint1_preprocessing_v3.ipynb`) | 40% | 22/09/2026 | done |
| 2 | Modeling (`notebooks/sprint2_modeling.ipynb`) | 30% | 29/09/2026 | done (executed with outputs) |
| 3 | Optimization & Explainability | 30% | 06/10/2026 | done — `notebooks/sprint3_final.ipynb`: Sprints 1–3, professor's CSVs only, ~10 min run, test 97.21% |
