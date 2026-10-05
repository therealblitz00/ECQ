# Case Study I — CRM → Billing Configuration Prediction

Team project for the case-study competition (Moodle course, section 8608).

A 4-Play telecom subscription is provisioned in several systems (CRM, TV, Internet, Billing)
that drift out of sync through human and automatic errors. From a customer's **CRM**
configuration (745 binary `CRM_*` columns), predict the correct **Billing (BIL)** configuration
(731 binary `BIL_*` columns), to help catch provisioning errors.

**Metric: Exact Match Ratio (EMR)** — a customer counts only if all 731 BIL bits are right.

**Short on time?** Read [`docs/EXECUTIVE_SUMMARY.md`](docs/EXECUTIVE_SUMMARY.md) (2 pages).

## Result

| Model | Test EMR (`solution.csv`) |
|---|---|
| Sprint 2 champion | 95.04% |
| Stage 1 only (per-label LightGBM, corrupted rows removed) | 97.21% |
| **Final model** — `notebooks/sprint3_final.ipynb` (Stage 1 + long-tail specialists) | **97.51%** (95% CI 97.41–97.61; 94,681 of 97,100 customers exactly right) |
| Public benchmark (reported) | ~98% |

## Start here: the final notebook

**`notebooks/sprint3_final.ipynb`** tells the whole story, Sprint 1 to Sprint 3, and is fully
self-contained:
- it reads **only the professor's files**: `train.csv`, `test.csv`, and `solution.csv`
  (the last one only at the end, to score the frozen model);
- all code lives in the notebook (no imports from `src/` or `scripts/`);
- it runs end to end in about 25 minutes and writes `data/derived/submission_sprint3_final.csv`.

```bash
python -m venv .venv
source .venv/bin/activate           # Linux / macOS
# .venv\Scripts\activate            # Windows (cmd / PowerShell)
# source .venv/Scripts/activate     # Windows (Git Bash)
pip install -r requirements.txt     # pinned versions, tested with Python 3.14
# put train.csv, test.csv, solution.csv in data/
jupyter notebook notebooks/sprint3_final.ipynb
```

## Key findings

1. **The test set needs generalisation, not lookup.** Only 6 of 97,100 test customers have a
   CRM configuration seen in train, so validation is split by exact configuration and scored
   against consensus targets (the most frequent billing per configuration).
2. **Billing is close to additive per CRM product.** On the same validation rows, a tuned
   linear model (logistic regression per billing item) gets within 0.15–0.23 points of
   LightGBM. One LightGBM per billing item is kept: better on both splits (paired test), but
   only slightly. Copying similar
   customers' billing (k-NN) reaches ~50–57%; label powerset is capped at ~23%.
3. **Sprint 1's feature transformations hurt.** PCA/SVD costs ~33 points; collapsing
   correlated columns ~6.
4. **~7–8% of training rows look corrupted by random injection:** impossible one-hot
   categories (two statuses at once, a status called `0`) and bursts of near-uniformly random
   rare products on both the CRM and BIL side. Removing them from training is worth **+2.5 test
   points** (the same model without the filter scores 94.67%; live ablation in the notebook),
   confirmed by paired tests on two splits.
5. **Business use — error detection.** Flagging customers whose actual billing disagrees with
   the model, with unusual customers queued last, catches **83% of known provisioning errors
   (95% CI 73–92%) by reviewing 2% of customers**, about 40× better than random review. On
   synthetic 1–3-item errors it catches ~92%, but ~80% for customers with a unique
   configuration (the realistic production case).
6. **Explainability (SHAP):** on the 233 billing items with a clear CRM cause, the model
   relies on that cause (~96% top-1 agreement with an independent check). Errors concentrate
   on items with no clear CRM cause (~2% of customers), and the model does not exploit the
   injected noise.
7. **The long tail is partly recoverable (Stage 2).** About 330 billing items were almost never
   predicted (macro F1 ~0.52). A third follow CRM product combinations that transfer to new
   customers; most of the rest look like injection residue. Class-weighted specialists that
   may only *add* an item, and only for customers with a rare product, raise EMR and macro F1
   on both splits and in all 5 folds, and take test EMR from **97.21% to 97.51%** (+290
   customers; test macro F1 0.70 → 0.77). Threshold tuning, label communities, stacked
   probabilities, rule overrides and targeted chains were tried and rejected.

Terms like *test-like*, *clean-like* and *consensus target* are defined in the glossary at
the top of the final notebook.

## Repository layout

```
data/            the professor's CSVs (gitignored) + derived/ small outputs + cache/ (gitignored)
notebooks/       sprint1_preprocessing_v3, sprint2_modeling, sprint3_final (+ archive/)
src/             shared library for the experiment scripts (loading, validation, metrics, models)
scripts/         train_final.py, make_leaderboard.py, sprint2/ and sprint3/ experiment scripts
docs/            executive summary, architecture, leaderboard, one folder of write-ups per sprint
```

Full map of every module and script: `docs/ARCHITECTURE_AND_ROUTING.md`.

## Documentation

| Document | What it contains |
|---|---|
| `docs/EXECUTIVE_SUMMARY.md` | 2-page summary: problem, findings, result, business use, limitations |
| `docs/sprint3/SPRINT3_DIAGNOSTIC_LOG.md` | Sprint 3 audit trail: Steps 1–9, design choices and why, threats to validity |
| `docs/sprint3/REVIEW_RESPONSE_AND_ROADMAP.md` | point-by-point response to a mock jury review |
| `docs/sprint2/ITERATION_LOG.md` | Sprint 2 lab notebook |
| `docs/EXPERIMENT_LEADERBOARD.md` | every experiment's validation scores (generated) |
| `docs/ARCHITECTURE_AND_ROUTING.md` | how code, data and documents fit together |
| `docs/sprint1/section11_12_explainer.md` | Sprint 1 Sections 11–12 explained |

## Reproducing the experiments (optional)

The experiment scripts use the `src/` library and a cache in `data/cache/`. The cache is
built on first run from the raw CSVs **and the Sprint 1 exports** (`train_clean*.csv`,
`test_clean*.csv`, `train_targets_v2.csv`), which are not in git: run
`notebooks/sprint1_preprocessing_v3.ipynb` first to produce them. Then, from the repo root:

```bash
python scripts/train_final.py 4 5                  # final model via the scripts (with Sprint 1 rules: 97.27%)
python scripts/sprint3/diag1_audit.py              # Sprint 3 Step 1 audit
python scripts/sprint3/review_checks.py            # Step 9: uncertainty, paired tests, baselines, error detection
python scripts/sprint3/explain_shap.py             # Step 6 SHAP figures
python scripts/make_leaderboard.py                 # rebuild the leaderboard
```

Use the venv's Python (`.venv/bin/python`, or `.venv\Scripts\python` on Windows). Each model
fit takes about 2–4 minutes.

## Methodology rules

- **Validation:** `GroupShuffleSplit` on the exact CRM configuration (20%, seed 42), every
  choice confirmed on a second split (seed 7) and the final model by 5-fold GroupKFold
  (96.99% ± 0.20; Stage 1 alone 96.83%). The primary metric is EMR on *clean-like* validation rows, which tracks
  the test score.
- **Uncertainty:** customers with an identical CRM configuration are not independent, so
  standard errors and comparisons resample whole configurations (paired bootstrap).
- **Selection:** a change is kept only if it is real on both splits and worth ≈ 0.1 point or
  more (bar set before testing); otherwise the simpler setting wins.
- **`solution.csv`** is never used for training or selection; it only scores frozen models.

## Sprints

| Sprint | Focus | Due | Deliverable |
|---|---|---|---|
| 1 | Pre-processing | 22/09/2026 | `notebooks/sprint1_preprocessing_v3.ipynb` |
| 2 | Modelling | 29/09/2026 | `notebooks/sprint2_modeling.ipynb` |
| 3 | Optimisation & Explainability | 06/10/2026 | `notebooks/sprint3_final.ipynb` |

Work is organised in Microsoft Planner by the team leader; this repository is where the code
is built. One notebook is submitted per sprint on Moodle.
