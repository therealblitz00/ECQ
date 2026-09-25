# Sprint 2 — Architecture & Data Routing

## Repository map

```
data/
├── train.csv, test.csv, solution.csv      # raw competition files (gitignored)
├── train_clean*.csv, test_clean*.csv,     # Sprint 1 exports: Section 10 / Section 11 (gitignored)
│   train_targets_v2.csv
├── cache/                                 # gitignored, rebuilt on demand
│   ├── arrays.npz                         #   uint8/float32 arrays of everything above (src/data.py)
│   ├── results.jsonl                      #   every experiment's metrics (src/experiment.log)
│   └── proba_*.npy                        #   saved validation probabilities per experiment
└── derived/
    ├── (Sprint 1 summaries, unchanged)
    ├── additive_rules_v2.json             #   the 42 Section 11 rules, used as post-processing
    ├── sprint2_experiment_results.csv     #   tracked export of results.jsonl
    └── submission_sprint2.csv             #   final submission (gitignored, ~75 MB)
src/                                       # reusable library (below)
scripts/                                   # one runnable script per experiment family
notebooks/
├── sprint1_preprocessing_v3.ipynb
└── sprint2_modeling.ipynb                 # Sprint 2 deliverable, executed end to end
docs/                                      # this file, deliverables, leaderboard, iteration log
```

## Data flow

```
train.csv ──uint8──► src/data.load() ──► X_train (745 CRM) , Y_raw (731 BIL)
                            │
                            ├─► validation.factorize_rows(X_train)   exact CRM config id (packbits)
                            ├─► validation.consensus_targets(...)     modal full BIL row per config
                            └─► validation.grouped_holdout(...)       split on config id, 20% val
                                        │
      train fold ──► experiment.dedup_configs() ──► unique configs + consensus labels (no weights)
                                        │
                     BinaryRelevanceLGBM(min_child_samples=20, reg_lambda=1)   731 models
                                        │
      predict_proba ──► threshold 0.5 ──► data.apply_additive_rules() (42 cols) ──► 731-bit rows
                                        │
                     metrics.evaluate() vs consensus targets  (val EMR, Hamming, 1-bit misses)
```

The final model (`scripts/train_final.py`, notebook §10) runs the same path on all 67,434
unique configurations of train.csv and writes `submission_sprint2.csv`.

## How Sprint 1's two pipelines route into Sprint 2

`src/data.features(d, pipeline)` serves three feature sets from the same cache:
`'raw'` (745 CRM), `'s10'` (Section 10's 734 columns), `'s11'` (Section 11's 742 CRM + 45 SVD).
`src/data.train_target(d, pipeline)` serves each pipeline's own labels (Section 10's
majority-relabeled targets, or raw). Result (see the leaderboard, E3*):

| | Section 10 | Section 11 | Used in champion? |
|---|---|---|---|
| Noise handling | majority relabel (share ≥0.65) | none | **Replaced** by consensus targets on unique configs (same idea, applied to every config) |
| Feature transform | collapse \|r\|≥0.95 CRM clusters (−6 pts) | Cochran filter + 45 SVD comps (−33 pts) | **No** — raw 745 columns |
| Target handling | all 731 learned | 42 columns by rule | **Yes, as post-processing** (+0.2–0.3 pts); rules are not exact on train, so the model still predicts all 731 first |
| Validation design (§3b) | — | — | **Yes** — grouped on exact CRM config |

## `src/` modules

| Module | Contents | Status |
|---|---|---|
| `src/data.py` | `load()` (build/read `.npz` cache, asserts row order vs. Sprint 1 exports), `features()`, `train_target()`, `additive_rules()`, `model_target_mask()`, `apply_additive_rules()` | done |
| `src/validation.py` | `row_keys()` (packbits), `factorize_rows()`, `consensus_targets()`, `grouped_holdout()`, `grouped_kfold()` | done |
| `src/metrics.py` | `exact_match_ratio()`, `hamming_loss()`, `row_errors()`, `evaluate()` (EMR + share of rows 1/2/3+ bits off), `top_offending_columns()` (columns that alone break a row) | done |
| `src/models.py` | `BinaryRelevanceLGBM` (threaded per-label LightGBM), `CoOccurrenceChainLGBM` + `cooccurrence_parents()` | done |
| `src/experiment.py` | `setup()` (shared split + targets), `score()`, `log()` → `results.jsonl`, `Timer`, `dedup_configs()` | done |

## `scripts/`

| Script | Experiments |
|---|---|
| `exp1_lookup_knn.py` | E1*: global mode, exact-lookup coverage, Hamming k-NN (`knn_hamming`) |
| `exp2_label_powerset.py` | E2*: LP ceilings, LP max-likelihood decoder, BR+LP hybrid |
| `exp3_binary_relevance.py [E3a E3a2 E3b E3c E3d E3e]` | E3*: BR LightGBM, Section 10 vs 11 as defined and ablated |
| `exp4_classifier_chain.py [n_parents] [min_child_samples]` | E4*: co-occurrence chain |
| `exp5_postprocess.py` | E5 blends, thresholds, rules, k-NN routing |
| `exp5_regularize.py [seed=N] [E5-reg-mcs20 ...]` | E5 regularization / bagging, second-seed confirmation |
| `train_final.py` | fit champion on all train, write + validate submission, external `solution.csv` diagnostic |
| `make_leaderboard.py` | regenerate `docs/EXPERIMENT_LEADERBOARD.md` and the tracked results CSV |

Run any script from the repo root with the project venv, e.g.
`.venv/Scripts/python scripts/exp3_binary_relevance.py E3a2`.
