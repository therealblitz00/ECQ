# Case Study I — CRM → Billing Configuration Prediction

Team project for the case-study competition (Moodle course, section 8608).

A 4-Play telecom subscription is provisioned in several systems (CRM, TV, Internet, Billing)
that drift out of sync through human and automatic errors. From a customer's **CRM**
configuration (745 binary `CRM_*` columns), predict the correct **Billing (BIL)** configuration
(731 binary `BIL_*` columns), to help catch provisioning errors.

**Metric: Exact Match Ratio (EMR)** — a row counts only if all 731 BIL bits are right.

## Result

| Model | Test EMR (`solution.csv`) |
|---|---|
| Sprint 2 champion | 95.04% |
| **Final model** — `notebooks/sprint3_final.ipynb` | **97.21%** (94,391 of 97,100 rows exactly right) |
| Public benchmark (reported) | ~98% |

## Start here: the final notebook

**`notebooks/sprint3_final.ipynb`** tells the whole story, Sprint 1 to Sprint 3, and is fully
self-contained:
- it reads **only the professor's files**: `train.csv`, `test.csv`, and `solution.csv`
  (the last one only in the final cell, to score the frozen model);
- all code lives in the notebook (no imports from `src/` or `scripts/`);
- it runs end to end in about 10 minutes and writes `data/derived/submission_sprint3_final.csv`.

```bash
python -m venv .venv
.venv/Scripts/activate              # Windows; Git Bash: source .venv/Scripts/activate
pip install -r requirements.txt
# put train.csv, test.csv, solution.csv in data/
jupyter notebook notebooks/sprint3_final.ipynb
```

## Key findings

1. **The test set needs generalisation, not lookup.** Only 6 of 97,100 test rows have a CRM
   configuration seen in train, so validation is split by exact configuration and scored
   against consensus targets (the most frequent billing per configuration).
2. **Billing is close to additive per CRM product.** One LightGBM per BIL column on the raw
   columns beats neighbour copying (52%) and label powerset (capped at ~23%).
3. **Sprint 1's feature transformations hurt.** PCA/SVD costs ~33 points; collapsing
   correlated columns ~6.
4. **~7–8% of training rows are corrupted by random injection:** impossible one-hot
   categories (two statuses at once, a status called `0`) and bursts of uniformly random
   rare products on both the CRM and BIL side. Removing them from training: **+2 test
   points**. Re-tuning regularisation on the clean data adds a little more.
5. **Explainability (SHAP):** where a billing item has a clear CRM cause, the model relies on
   exactly that cause (96–99% agreement with an independent check). Its errors are on items
   with no clear CRM cause, and it does not exploit the injected noise.

## Repository layout

```
data/            the professor's CSVs (gitignored) + derived/ small outputs + cache/ (gitignored)
notebooks/       sprint1_preprocessing_v3, sprint2_modeling, sprint3_final (+ archive/, exports/)
src/             shared library for the experiment scripts (loading, validation, metrics, models)
scripts/         train_final.py, make_leaderboard.py, sprint2/ and sprint3/ experiment scripts
docs/            architecture, leaderboard, and one folder of write-ups per sprint
```

Full map of every module and script: `docs/ARCHITECTURE_AND_ROUTING.md`.

## Documentation

| Document | What it contains |
|---|---|
| `docs/sprint3/SPRINT3_DIAGNOSTIC_LOG.md` | Sprint 3 audit trail: Steps 1–8, design choices and why, threats to validity |
| `docs/sprint2/ITERATION_LOG.md` | Sprint 2 lab notebook |
| `docs/EXPERIMENT_LEADERBOARD.md` | every experiment's validation scores (generated) |
| `docs/ARCHITECTURE_AND_ROUTING.md` | how code, data and documents fit together |
| `docs/sprint1/section11_12_explainer.md` | Sprint 1 Sections 11–12 explained |

## Reproducing the experiments (optional)

The experiment scripts use the `src/` library and a cache in `data/cache/`, built on first
run from the raw CSVs and the Sprint 1 exports (`train_clean*.csv`, produced by the Sprint 1
notebook). Run them from the repo root:

```bash
.venv/Scripts/python scripts/train_final.py 4 5             # final model via the scripts (with Sprint 1 rules: 97.27%)
.venv/Scripts/python scripts/sprint3/diag1_audit.py         # Sprint 3 Step 1 audit
.venv/Scripts/python scripts/sprint3/explain_shap.py        # Step 6 SHAP figures
.venv/Scripts/python scripts/make_leaderboard.py            # rebuild the leaderboard
```

Each model fit takes about 2–4 minutes.

## Methodology rules

- **Validation:** `GroupShuffleSplit` on the exact CRM configuration (20%, seed 42), every
  choice confirmed on a second split (seed 7). The primary metric is EMR on *clean-like*
  validation rows, which tracks the test score.
- **Selection:** a change is kept only if it holds on both splits. Within one standard error
  (~0.1 points), the simpler setting wins.
- **`solution.csv`** is never used for training or selection; it only scores frozen models.

## Sprints

| Sprint | Focus | Due | Deliverable |
|---|---|---|---|
| 1 | Pre-processing | 22/09/2026 | `notebooks/sprint1_preprocessing_v3.ipynb` |
| 2 | Modelling | 29/09/2026 | `notebooks/sprint2_modeling.ipynb` |
| 3 | Optimisation & Explainability | 06/10/2026 | `notebooks/sprint3_final.ipynb` |

Work is organised in Microsoft Planner by the team leader; this repository is where the code
is built. One notebook is submitted per sprint on Moodle.
