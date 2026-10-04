# Architecture & Data Routing

How the code, data and documents fit together. For the project overview see `README.md`.

## Repository map

```
data/                                  # raw competition files and everything built from them
├── train.csv, test.csv, solution.csv  #   the professor's files (gitignored)
├── train_clean*.csv, test_clean*.csv, #   Sprint 1 exports, Section 10 / Section 11 (gitignored)
│   train_targets_v2.csv
├── cache/                             #   gitignored: arrays.npz, results.jsonl, proba_*.npy, logs
└── derived/                           #   tracked small outputs: Sprint 1 summaries,
                                       #   additive_rules_v2.json, sprint2_experiment_results.csv,
                                       #   sprint3_shap_drivers.csv; submission_*.csv (gitignored)
notebooks/
├── sprint1_preprocessing_v3.ipynb     # Sprint 1 deliverable
├── sprint2_modeling.ipynb             # Sprint 2 deliverable (committed without outputs)
├── sprint3_final.ipynb                # FINAL: Sprints 1-3, professor's CSVs only
├── archive/                           # superseded notebook versions
└── exports/                           # HTML renders (gitignored)
src/                                   # shared library used by every script (below)
scripts/
├── train_final.py                     # fit on all train, write submission, solution.csv diagnostic
├── make_leaderboard.py                # results.jsonl -> docs/EXPERIMENT_LEADERBOARD.md + results CSV
├── sprint2/                           # Sprint 2 experiments (exp1-exp5)
└── sprint3/                           # Sprint 3 diagnostics and experiments (diag1-3, exp6-11, SHAP)
docs/
├── ARCHITECTURE_AND_ROUTING.md        # this file
├── EXPERIMENT_LEADERBOARD.md          # generated - never edit by hand
├── sprint1/                           # Section 11/12 explainer
├── sprint2/                           # iteration log, deliverables, early model candidates
└── sprint3/                           # SPRINT3_DIAGNOSTIC_LOG.md (Steps 1-8), figures/,
                                       # teammate_mca_experiments_log.csv (Step 7 input)
```

## Two ways to run the final model

| | `notebooks/sprint3_final.ipynb` | `scripts/train_final.py 4 5` |
|---|---|---|
| Reads | only `train.csv`, `test.csv`, `solution.csv` | the `src/` cache (built from the raw CSVs + Sprint 1 exports) |
| Code | all inside the notebook | `src/` library |
| Sprint 1 additive rules | **not applied** | applied |
| Test EMR | **97.21%** | 97.27% |
| Purpose | the graded, presentable deliverable | experiment harness |

## Data flow (scripts / `src/`)

```
train.csv ──uint8──► src/data.load() ──► X_train (745 CRM), Y_raw (731 BIL)
                            │
                            ├─► validation.factorize_rows(X_train)   exact CRM config id (packbits)
                            ├─► validation.consensus_targets(...)     modal full BIL row per config
                            ├─► validation.test_like_mask(...)        valid one-hot categoricals, no '0'
                            ├─► validation.rare_pack_count(...)       CRM packs in < 0.2% of rows
                            └─► validation.grouped_holdout(...)       split on config id, 20% val
                                        │
      train fold ── drop not test-like or ≥ 4 rare packs (suspected corruption, ~7%)
                 ──► experiment.dedup_configs() ──► unique configs + consensus labels (no weights)
                                        │
                     BinaryRelevanceLGBM(min_child_samples=5, reg_lambda=1)   731 models
                                        │
      predict_proba ──► threshold 0.5 [──► data.apply_additive_rules(), scripts only] ──► 731-bit rows
                                        │
                     experiment.score(): EMR on all / test-like / clean-like validation rows
```

## How Sprint 1's two pipelines route into the models

`src/data.features(d, pipeline)` serves three feature sets from the same cache:
`'raw'` (745 CRM), `'s10'` (Section 10's 734 columns), `'s11'` (Section 11's 742 CRM + 45 SVD).

| | Section 10 | Section 11 | Used in the final model? |
|---|---|---|---|
| Noise handling | majority relabel (share ≥ 0.65) | none | **Replaced** by consensus targets on unique configs + dropping suspected-corrupted rows |
| Feature transform | collapse \|r\| ≥ 0.95 CRM clusters (−6 pts) | Cochran filter + 45 SVD comps (−33 pts) | **No** — raw 745 columns |
| Target handling | all 731 learned | 42 columns by rule | **No** in the final notebook (+0.02 only, mined on all of train); yes in the scripts |
| Validation design (§3b) | — | — | **Yes** — grouped on exact CRM config |

## `src/` modules

| Module | Contents |
|---|---|
| `src/data.py` | `load()` (build/read `.npz` cache, asserts row order vs. Sprint 1 exports), `features()`, `train_target()`, `additive_rules()`, `model_target_mask()`, `apply_additive_rules()` |
| `src/validation.py` | `row_keys()` (packbits), `factorize_rows()`, `consensus_targets()`, `test_like_mask()`, `rare_pack_count()`, `clean_like_mask()`, `grouped_holdout()`, `grouped_kfold()` |
| `src/metrics.py` | `exact_match_ratio()`, `hamming_loss()`, `row_errors()`, `evaluate()`, `multilabel_report()`, `top_offending_columns()` |
| `src/models.py` | `BinaryRelevanceLGBM` (threaded per-label LightGBM), `CoOccurrenceChainLGBM` + `cooccurrence_parents()` |
| `src/experiment.py` | `setup()` (shared split, targets, test-like / clean-like masks), `score()`, `log()` → `results.jsonl`, `Timer`, `dedup_configs()` |

## `scripts/`

Run from the repo root with the project venv, e.g.
`.venv/Scripts/python scripts/sprint3/exp8_retune.py 42 5:1`.

| Script | Purpose |
|---|---|
| `train_final.py [k] [mcs]` | fit on all train, write + validate submission, `solution.csv` diagnostic |
| `make_leaderboard.py` | regenerate `docs/EXPERIMENT_LEADERBOARD.md` and the tracked results CSV |
| `sprint2/exp1_lookup_knn.py` | E1: global mode, exact-lookup coverage, Hamming k-NN (`knn_hamming`) |
| `sprint2/exp2_label_powerset.py` | E2: label-powerset ceilings and decoder |
| `sprint2/exp3_binary_relevance.py [E3a2 E3d ...]` | E3: per-label LightGBM, Section 10 vs 11 |
| `sprint2/exp4_classifier_chain.py [n_parents] [mcs]` | E4: co-occurrence classifier chain |
| `sprint2/exp5_postprocess.py`, `exp5_regularize.py` | E5: blends, thresholds, regularisation, bagging |
| `sprint3/diag1_audit.py` | Step 1: column encoding, one-hot validity, error location, pack-id structure |
| `sprint3/exp6_clean_rows.py <seed>` | Step 2: drop non-test-like training rows |
| `sprint3/diag2_why.py` | Step 2: why — merge test, missed columns, label flips, injection test |
| `sprint3/exp7_drop_corrupted.py <seed> <k,k,...>` | Step 3: drop suspected-corrupted rows, cut-off sweep |
| `sprint3/diag3_remaining_errors.py <probs id>` | Steps 3–4: what is left wrong on clean-like rows |
| `sprint3/exp8_retune.py <seed> <mcs>:<lambda> ...` | Step 4: regularisation re-tuned on clean data |
| `sprint3/exp9_rare_thresholds.py <probs id>` | Step 4: lower threshold for rare BIL columns |
| `sprint3/exp10_step5.py <seed> cl repair cap` | Step 5: confident learning, repair, capacity |
| `sprint3/explain_shap.py` | Step 6: SHAP explainability, figures in `docs/sprint3/figures/` |
| `sprint3/exp11_count_features.py <seed> npk npk_rare` | Step 7: basket-size count features (not adopted) |
