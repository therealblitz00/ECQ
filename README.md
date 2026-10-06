# Case Study I — CRM → Billing Configuration Prediction

Team project for the case-study competition (Moodle course, section 8608).

A 4-Play telecom subscription is provisioned in several systems (CRM, TV, Internet, Billing)
that drift out of sync through human and automatic errors. From a customer's **CRM**
configuration (745 binary `CRM_*` columns), we predict the correct **Billing (BIL)**
configuration (731 binary `BIL_*` columns), to help catch provisioning errors.

**Metric: Exact Match Ratio (EMR).** A customer counts only if all 731 billing items are right.

- The case-study brief: [`docs/PROBLEM_DESCRIPTION.md`](docs/PROBLEM_DESCRIPTION.md)
- Short on time? Read [`docs/EXECUTIVE_SUMMARY.md`](docs/EXECUTIVE_SUMMARY.md) (2 pages).

## Result

| Model | Test EMR (`solution.csv`) |
|---|---|
| Sprint 2 model | 95.04% |
| Sprint 3, Stage 1: per-label LightGBM, corrupted training rows removed | 97.21% |
| **Final model**: Stage 1 + long-tail specialists (Stage 2) | **97.51%** (95% CI 97.41–97.61; 94,681 of 97,100 customers exactly right) |

Test scores were computed only after each model was frozen; `solution.csv` was never used to
train or choose anything.

## Start here: the final notebook

[`notebooks/03_sprint3_final.ipynb`](notebooks/03_sprint3_final.ipynb) tells the whole story,
Sprint 1 to Sprint 3, in 12 chapters with three appendices. It is fully self-contained:
- it reads **only the professor's files**: `train.csv`, `test.csv` and `solution.csv` (the last
  one only at the end, to score the frozen model);
- all code lives in the notebook (no imports from `src/` or `scripts/`);
- it runs end to end in about 25 minutes on a 16-core machine and writes
  `data/derived/submission_sprint3_final.csv`.

```bash
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
# .venv\Scripts\activate            # Windows (cmd / PowerShell)
# source .venv/Scripts/activate     # Windows (Git Bash)
pip install -r requirements.txt     # pinned versions, tested with Python 3.14
# put train.csv, test.csv and solution.csv in data/
jupyter notebook notebooks/03_sprint3_final.ipynb
```

## Key findings

1. **The test set needs generalisation, not lookup.** Only 6 of 97,100 test customers have a
   CRM configuration seen in training, so validation is split by exact configuration and
   scored against consensus targets (the most frequent bill per configuration).
2. **Billing is close to additive per CRM product.** A tuned logistic regression per billing
   item gets within 0.15–0.23 points of LightGBM. One LightGBM per billing item is kept (better
   on both splits, but only slightly). Copying similar customers' bills (k-NN) reaches ~50–57%;
   label powerset is capped at ~23%.
3. **Sprint 1's feature transformations hurt.** PCA/SVD costs ~33 points; collapsing
   correlated columns ~6.
4. **~7% of training rows look corrupted by random injection:** impossible one-hot categories
   (two statuses at once, a status called `0`) and bursts of near-uniformly random rare products
   on both the CRM and billing side. Removing them from training is worth **+2.5 test points**
   (94.67% without the filter; live ablation in the notebook).
5. **Part of the long tail can be recovered (Stage 2).** About 330 billing items were almost
   never predicted (macro F1 ~0.52). A third follow CRM product combinations that transfer to
   new customers; most of the rest look like injection residue. Class-weighted specialists that
   may only *add* an item, and only for customers with a rare product, raise EMR and macro F1 on
   both splits and in all 5 folds, and take test EMR from **97.21% to 97.51%** (+290 customers;
   test macro F1 0.70 → 0.77). Threshold tuning, label communities, stacked probabilities, rule
   overrides and targeted classifier chains were tried and rejected.
6. **Business use: error detection.** Flagging customers whose actual bill disagrees with the
   model, with unusual customers queued last, catches **83% of known provisioning errors
   (95% CI 73–92%) by reviewing 2% of customers**, about 40× better than random review.
7. **Explainability (SHAP).** On the 233 billing items with a clear CRM cause, the model relies
   on that cause (~96% top-1 agreement with an independent check), and it does not exploit the
   injected noise.

Terms like *test-like*, *clean-like* and *consensus target* are defined in the glossary at the
end of the final notebook (Appendix C).

## Repository layout

```
data/                    the professor's CSVs (gitignored)
├── derived/             small tracked outputs (Sprint 1 summaries, rules, experiment results)
└── cache/               local arrays, experiment probabilities and results.jsonl (gitignored)
notebooks/               one notebook per sprint, in reading order
├── 01_sprint1_preprocessing.ipynb
├── 02_sprint2_modeling.ipynb
├── 03_sprint3_final.ipynb      ← final deliverable
└── archive/             superseded Sprint 1 versions
src/                     shared library for the experiment scripts (data, validation, metrics, models)
scripts/                 train_final.py, make_leaderboard.py, sprint2/ and sprint3/ experiments
docs/                    brief, executive summary, architecture, leaderboard, one folder per sprint
```

Full map of every module and script: [`docs/ARCHITECTURE_AND_ROUTING.md`](docs/ARCHITECTURE_AND_ROUTING.md).

## Documentation

| Document | What it contains |
|---|---|
| `docs/PROBLEM_DESCRIPTION.md` | The case-study brief: task, data, metric, sprint plan |
| `docs/EXECUTIVE_SUMMARY.md` | 2-page summary: problem, findings, result, business use, limitations |
| `docs/ARCHITECTURE_AND_ROUTING.md` | How code, data and documents fit together; every module and script |
| `docs/sprint3/SPRINT3_DIAGNOSTIC_LOG.md` | Sprint 3 audit trail: Steps 1–16, every experiment, design choices and why |
| `docs/sprint3/REVIEW_RESPONSE_AND_ROADMAP.md` | Answers to the mock jury reviews |
| `docs/EXPERIMENT_LEADERBOARD.md` | Every experiment run, generated from the results log (do not edit by hand) |
| `docs/sprint2/ITERATION_LOG.md` | Sprint 2 modelling iterations |
| `docs/sprint1/SECTION11_12_EXPLAINER.md` | Sprint 1 Sections 11–12 explained |

## Reproducing the experiments (optional)

The experiment scripts use the `src/` library and a cache in `data/cache/`. The cache is built
on first run from the raw CSVs **and the Sprint 1 exports** (`train_clean*.csv`,
`test_clean*.csv`, `train_targets_v2.csv`), which are not in git: run
`notebooks/01_sprint1_preprocessing.ipynb` first to produce them. Then, from the repo root:

```bash
python scripts/train_final.py 4 5                    # Stage 1 via the scripts (with Sprint 1 rules: 97.27%)
python scripts/sprint3/diag1_audit.py                # Sprint 3 Step 1 audit
python scripts/sprint3/review_checks.py              # Step 9: uncertainty, paired tests, baselines, error detection
python scripts/sprint3/explain_shap.py               # Step 6 SHAP figures
python scripts/sprint3/exp13_macro_f1.py fit 42 base # Step 12: Stage-1 baseline + out-of-fold probabilities
python scripts/sprint3/exp14_longtail.py diag 42     # Step 13: long-tail label table (needs the line above)
python scripts/sprint3/exp14_longtail.py fit 42 crm-cw
python scripts/sprint3/exp14_longtail.py eval 42 crm-cw 0.5 rare   # Step 13: Stage 2 on one split
python scripts/sprint3/exp15_kfold_longtail.py       # Step 13: 5-fold check of Stage 2 (~1 hour)
python scripts/make_leaderboard.py                   # rebuild the leaderboard from data/cache/results.jsonl
```

Use the venv's Python (`.venv/bin/python`, or `.venv\Scripts\python` on Windows). Each model
fit takes about 2–4 minutes. `make_leaderboard.py` rebuilds the leaderboard from the local
`results.jsonl`, so only run it on a machine whose cache holds the full experiment history.

## Methodology rules

- **Validation:** `GroupShuffleSplit` on the exact CRM configuration (20%, seed 42), every
  choice confirmed on a second split (seed 7), and the final model by 5-fold GroupKFold
  (96.99% ± 0.20; Stage 1 alone 96.83%). The primary metric is EMR on *clean-like* validation
  rows, which tracks the test score.
- **Uncertainty:** customers with an identical CRM configuration are not independent, so
  standard errors and comparisons resample whole configurations (paired bootstrap).
- **Selection:** a change is kept only if it is real on both splits and worth about 0.1 point
  or more (bar set before testing); otherwise the simpler setting wins.
- **`solution.csv`** is never used for training or selection; it only scores frozen models.

## Sprints

| Sprint | Focus | Due | Deliverable |
|---|---|---|---|
| 1 | Pre-processing | 22/09/2026 | `notebooks/01_sprint1_preprocessing.ipynb` |
| 2 | Modelling | 29/09/2026 | `notebooks/02_sprint2_modeling.ipynb` |
| 3 | Optimisation & Explainability | 06/10/2026 | `notebooks/03_sprint3_final.ipynb` |

Work is organised in Microsoft Planner by the team leader; this repository is where the code
is built. One notebook is submitted per sprint on Moodle.
