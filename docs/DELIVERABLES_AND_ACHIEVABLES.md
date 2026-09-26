# Sprint 2 — Deliverables Checklist

Due 29/09/2026, 30% of grade.

- [x] **Empirical comparison of Sprint 1 Section 10 vs. Section 11 data pipelines** — each
      run as defined (E3b 50.58%, E3c 12.80%) and on identical training rows (E3d 74.31%,
      E3e 47.42% vs raw 80.05%). Notebook §5, `ITERATION_LOG.md` #4.
- [x] **Validation strategy accounting for CRM→BIL ambiguity/provisioning noise** — split
      grouped on the exact CRM configuration (0% val configs seen in train, test 0.006%),
      scored against per-configuration consensus targets; confirmed on a second split seed.
      Notebook §2, §9.
- [x] **All 4 core modeling paradigms**
  - [x] Independent Binary Relevance — `BinaryRelevanceLGBM` (E3*, E5-reg*) — **champion**
  - [x] Multi-Output — Hamming k-NN predicting the full vector jointly (E1*); bagged
        multi-model ensemble (E5-bag3)
  - [x] Classifier Chains — co-occurrence chain (E4-cc30, E4-cc30-mcs20)
  - [x] Label Powerset — max-likelihood decoder over observed configs + ceiling analysis (E2*)
- [x] **Hybrid/lookup post-processing & threshold optimization for EMR** — exact lookup
      coverage, BR+LP snapping, 42 additive rules, thresholds 0.3–0.7, BR/chain blends,
      k-NN routing for rare-pack rows (E2-hyb*, E5*)
- [x] **Clean, reproducible `notebooks/sprint2_modeling.ipynb`** — executed end to end
      with `jupyter nbconvert --execute --inplace`
- [x] **Validated `data/derived/submission_sprint2.csv`** — 97,100 rows, `MSISDN` +
      `Bill_Conf` (731 chars of 0/1), unique MSISDN, zero NaNs (asserted in code)

## Results summary

| | EMR |
|---|---|
| Champion validation (seed 42 / seed 7) | 87.84% / 86.49% |
| Champion on test (`solution.csv`, external diagnostic only) | 95.04% |

## Scope notes

- `sampleSubmission.csv` is not present locally; the format follows `solution.csv` and the
  README (`MSISDN`, `Bill_Conf`). Worth checking against the real sample file before upload.
- `solution.csv` was used once, after model selection was frozen, as an external
  diagnostic. CLAUDE.md asks to confirm with the instructor how it may be used.
- CatBoost is not installed and was not needed; LightGBM covered every tree-based variant.
  A neural/embedding model was not built: the per-pack structure is captured by per-label
  trees, and the remaining errors are variance and rare-pack combinations, not a
  representation problem.
- Environment: the project `.venv` now has `requirements.txt` installed (pandas 3.0.6,
  lightgbm 4.7.0, scikit-learn 1.9.1); all Sprint 2 code runs there.
