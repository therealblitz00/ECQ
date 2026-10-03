# Case Study I — CRM → Billing Configuration Prediction

Team project for the case-study competition (Moodle course, section 8608).

A 4-Play telecom subscription is provisioned across several systems (CRM, TV, Internet,
Billing), and human or automatic errors let them drift out of sync. Given a customer's
**CRM** configuration (745 binary `CRM_*` columns), predict the correct **Billing (BIL)**
configuration (731 binary `BIL_*` columns), to help catch provisioning errors.

- **Metric: Exact Match Ratio (EMR).** A row counts only if all 731 BIL bits are right.
- `train.csv` contains provisioning errors; the test pairs are the correct ones.

## Status

| Sprint | Focus | Due | Status |
|---|---|---|---|
| 1 | Pre-processing | 22/09/2026 | Done — `notebooks/sprint1_preprocessing_v3.ipynb` |
| 2 | Modelling | 29/09/2026 | Done — `notebooks/sprint2_modeling.ipynb` |
| 3 | Optimisation & Explainability | 06/10/2026 | Done — `notebooks/sprint3_final.ipynb` (self-contained, Sprints 1–3) |

| Model | Validation EMR, clean-like (seed 42 / 7) | Test EMR (`solution.csv`, diagnostic) |
|---|---|---|
| Sprint 2 champion | 95.93% / 95.06% | 95.04% |
| Sprint 3 champion with the Sprint 1 additive rules | 96.81% / 96.33% | 97.27% |
| **Final model** (`notebooks/sprint3_final.ipynb`: corrupted training rows removed + re-tuned, no rules; needs only `train.csv` / `test.csv`) | **96.79% / 96.30%** | **97.21%** |
| Public benchmark (reported) | — | ~98% |

## Key findings

1. **The test set needs generalisation, not lookup.** Only 6 of 97,100 test rows have a
   CRM configuration seen in train, so validation is grouped on the exact configuration.
2. **Billing is close to additive per CRM pack.** Per-column models (one LightGBM per BIL
   column) beat neighbour copying and label powerset. Label powerset is capped at ~23%
   because unseen CRM configurations produce unseen BIL configurations.
3. **Neither Sprint 1 feature transformation helps.** PCA/SVD costs ~33 points and column
   collapsing ~6. The model uses the raw 745 binary columns.
4. **~7–8% of training rows are corrupted by random injection** (Sprint 3). These rows have
   impossible one-hot categories (two statuses, a status called `0`) and bursts of
   uniformly random rare products on both the CRM and BIL side. Removing them from
   training: **+2.0 test points**. Re-tuning regularisation afterwards: +0.23. The Sprint 1
   additive rules were then dropped (within-noise gain, mined on all of train), so the final
   model depends only on the professor's CSVs.
5. The remaining errors sit in rows with 1–2 rare products. Further cleaning, repair,
   thresholds and extra capacity were tested and did not help beyond noise.
6. **SHAP explainability** (Step 6): where a BIL item has a clear CRM cause, the model relies
   on exactly that cause (96–99% agreement with an independent check). Errors concentrate
   on items with no clear CRM cause (Spearman +0.86 between "one clear driver" and F1).
   The model does not lean on the injected noise.

Full story: `docs/sprint3/SPRINT3_DIAGNOSTIC_LOG.md` (Sprint 3) and
`docs/sprint2/ITERATION_LOG.md` (Sprint 2). Every experiment: `docs/EXPERIMENT_LEADERBOARD.md`.

## Repository layout

```
data/            raw CSVs (gitignored), derived/ summaries, cache/ (gitignored)
src/             reusable library: data loading, validation, metrics, models, experiment logging
scripts/         one runnable script per experiment / diagnostic (exp*, diag*), train_final, make_leaderboard
notebooks/       sprint notebooks (+ archive/ of old versions, exports/ of HTML renders)
docs/            ARCHITECTURE_AND_ROUTING.md, EXPERIMENT_LEADERBOARD.md, sprint1/ sprint2/ sprint3/
```

Details of every module and script: `docs/ARCHITECTURE_AND_ROUTING.md`.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (Git Bash: source .venv/Scripts/activate)
pip install -r requirements.txt
```

Place the competition files in `data/`: `train.csv` (187,442 × 1,477), `test.csv`
(97,100 × 746), `solution.csv` (97,100 × 2) and `sampleSubmission.csv`. They are large
and gitignored. All CRM/BIL columns are binary: always load them as `int8`/`uint8`
(`src/data.py` does this and caches the arrays in `data/cache/`).

## Reproduce

Run everything from the repo root with the venv's Python.

```bash
# Current champion: fit on train minus suspected-corrupted rows, write the submission,
# report the solution.csv diagnostic  -> data/derived/submission_sprint3_k4_mcs5.csv
.venv/Scripts/python scripts/train_final.py 4 5

# Sprint 2 champion, for comparison -> data/derived/submission_sprint2.csv
.venv/Scripts/python scripts/train_final.py

# Sprint 3 diagnostics (read-only)
.venv/Scripts/python scripts/diag1_audit.py > data/cache/diag1.txt
.venv/Scripts/python scripts/diag2_why.py   > data/cache/diag2.txt

# Rebuild the leaderboard from data/cache/results.jsonl
.venv/Scripts/python scripts/make_leaderboard.py
```

The first run builds `data/cache/arrays.npz` (~1 minute); each model fit takes ~2–4 minutes.

## Methodology rules

- **Validation:** `GroupShuffleSplit` on the exact CRM configuration (20%, seed 42),
  confirmed on a second split (seed 7). Scored against consensus targets (the most
  frequent full BIL configuration per CRM configuration).
- **Primary metric (Sprint 3):** EMR on *clean-like* validation rows (valid one-hot
  categoricals, < 3 CRM packs rarer than 0.2%). This is the validation number that tracks
  the test set.
- **Selection:** settings must hold on both splits. When results are within one standard
  error (~0.1 points), the simpler / more regularised setting wins.
- **`solution.csv`:** never used for training or selection. Read only at the end of
  `scripts/train_final.py` to report a test diagnostic once a model is frozen.
  Test-set *inputs* (no answers) were compared with train inputs to design the
  corruption filters.
- **Workflow (Sprint 3):** experiments run as Python scripts. Notebooks are updated only
  after explicit approval from the team lead.

## Next steps

1. ~~Step 6 — explainability~~ done: `scripts/explain_shap.py`, figures in
   `docs/sprint3/figures/`, per-column drivers in `data/derived/sprint3_shap_drivers.csv`.
2. ~~Sprint 3 notebook~~ done: `notebooks/sprint3_final.ipynb`. It is fully self-contained
   (no imports from `src/` or `scripts/`), runs end to end in ~10 minutes, and writes
   `data/derived/submission_sprint3_final.csv`.
3. **Optional (after approval):** re-execute `notebooks/sprint2_modeling.ipynb` (the committed
   copy has no outputs) and update its pointer `docs/ITERATION_LOG.md` → `docs/sprint2/ITERATION_LOG.md`.
4. Check the submission format against `sampleSubmission.csv`, and confirm with the
   instructor how `solution.csv` may be used.

## Team & process

Work is organised in Microsoft Planner by the team leader; this repo is where the code is
built. One notebook is submitted per sprint on Moodle, and results are presented at the
end of each week.
