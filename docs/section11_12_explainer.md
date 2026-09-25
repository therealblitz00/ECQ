# Explaining Sections 11 and 12 to the teacher

This is a talking-points document for `sprint1_preprocessing_v3.ipynb`, sections **11
(Alternative Multi-Label Preprocessing Pipeline — V2)** and **12 (Multi-Label Baseline
Modeling Architecture & Post-Processing Reconstruction)**. Section 10 is our baseline,
heuristic preprocessing pipeline; Section 11 is a second, more statistically-grounded
pipeline built to be *compared against* it in Sprint 2 — not a replacement decided in
advance. Section 12 defines (but does not yet run) the modeling architecture that will
consume Section 11's output.

---

## 1. Why a second pipeline exists at all

Section 10 made a few pragmatic, judgment-call decisions (e.g., relabeling ambiguous
rows by majority vote, dropping only exactly-constant columns, no dimensionality
reduction). Section 11 revisits each of those decisions with an explicit statistical
justification, so that in Sprint 2 we can measure — via held-out EMR — whether the
extra rigor actually pays off, instead of assuming it does. The notebook includes a
comparison table (just before 11.1) mapping every Section 10 choice to its Section 11
counterpart.

**Key framing for the teacher:** this is not "Section 11 replaces Section 10." It's two
competing candidate pipelines, kept side by side (`train_clean.csv` vs.
`train_clean_v2.csv`) so Sprint 2 can pick a winner empirically.

---

## 2. Section 11 step by step

### 11.1 — Low-variance filter (Cochran's criterion)
- **What:** Drops CRM columns with fewer than 5 positive activations in the training
  set (187,442 rows).
- **Why 5, not some arbitrary cutoff:** Cochran's asymptotic sample-size rule for
  binary/contingency data requires `n · p ≥ 5` for statistical tests on that column to
  be trustworthy. Below that, a column is empirical noise, not a learnable signal.
- **Result:** 3 of 745 CRM columns dropped (`CRM_2396/2397/2398_PACK`), 742 retained.
  This is far more conservative than it sounds — it's not a big pruning step, just a
  principled way to remove the handful of columns too sparse to say anything.

### 11.2 — Detecting deterministic (additive) CRM → BIL rules
- **What:** Builds on the Section 8b finding that some CRM packs *always* turn on
  exactly one BIL flag, with no exceptions, whenever a customer's config changes by
  that one pack (a "distance-1" pair comparison).
- **Criteria for "strictly additive":** 100% consistency (`modal_share == 1.0`) across
  at least 5 observed pairs, adding exactly one BIL column and removing none.
- **Why it matters:** If a rule is 100% deterministic in the data, there's nothing for
  a classifier to learn there — predicting it with `if CRM_X: BIL_Y = 1` is both
  simpler and provably correct on the same distribution the rule was mined from.
- **Result:** 42 deterministic CRM→BIL rules found, covering 42 of the 731 BIL columns.
  Saved as `additive_rules_v2.json`.

### 11.3 — Target partitioning
- **What:** Splits the 731 BIL targets into `Y_additive` (the 42 handled by the
  deterministic rules from 11.2) and `Y_model` (the remaining 689, which genuinely need
  a classifier).
- **Why:** This is the practical payoff of 11.2 — it shrinks the multi-label learning
  problem from 731 to 689 estimators, and those 42 are guaranteed correct by
  construction rather than approximated by a model.

### 11.4 — Vectorized χ² feature selection with Benjamini–Hochberg FDR
- **What:** Tests every (CRM feature, BIL target) pair for statistical association
  using a vectorized Pearson χ² test (1 degree of freedom), computed over sparse
  matrices for speed.
- **Why FDR instead of Bonferroni:** With 742 CRM candidates × 689 remaining targets ≈
  511,000 simultaneous hypothesis tests, an unadjusted α=0.05 would produce ~25,000
  false positives by chance alone. Bonferroni (α/M) would be so conservative it throws
  away real, moderate-strength predictors. Benjamini–Hochberg controls the *expected
  proportion* of false discoveries at 5%, which is the standard, less punishing
  correction for this many simultaneous tests.
- **Result:** All 742 CRM features passed the FDR-adjusted significance threshold, so
  no features were discarded at this step (742 → 742) — the χ² test's job here was to
  formally certify that every retained CRM feature is genuinely informative about at
  least one target, not to shrink the set further.

### 11.5 — Dimensionality reduction (sparse TruncatedSVD, dynamic cutoff)
- **What:** Fits `TruncatedSVD` on the training CRM matrix only (no leakage into test),
  in sparse CSR format for memory efficiency, over a pool of up to 80 candidate
  components.
- **Why a dynamic cutoff instead of a fixed K:** An arbitrary choice like K=20 captured
  under 70% of variance in earlier exploration (Section 7a). Instead, components are
  added until cumulative explained variance crosses 80%, which is a standard,
  defensible threshold rather than a guess.
- **Result:** k\* = 45 components needed to reach ≥80% variance (actual: 80.22%). These
  45 latent components are added as extra features alongside the raw CRM columns.

### 11.6 — Assemble and export
- **What:** Concatenates the filtered CRM columns + 45 SVD components + `MSISDN` into
  `train_clean_v2.csv` / `test_clean_v2.csv`, and the 689 `Y_model` targets into
  `train_targets_v2.csv`. Includes assertions that train/test schemas match and that
  there are no nulls, so a schema mismatch fails loudly instead of silently corrupting
  Sprint 2's modeling.
- **Result shapes:** `train_clean_v2` (187,442 × 788), `test_clean_v2` (97,100 × 788),
  `train_targets_v2` (187,442 × 690) — 788 = 742 CRM features + 45 SVD components +
  `MSISDN`; 690 = 689 model targets + `MSISDN`.

**One-line summary for the teacher:** *Section 11 takes the same raw data as Section
10, but instead of ad-hoc thresholds, every preprocessing decision — which columns to
drop, which targets need a model at all, which features are informative, how many
latent dimensions to keep — is justified by a named statistical criterion (Cochran,
empirical determinism, Benjamini–Hochberg FDR, explained-variance threshold), so the
choices are defensible and reproducible rather than tuned by eye.*

---

## 3. Section 12 step by step

Section 12 does **not** train a model yet — it defines the architecture and evaluation
machinery that Sprint 2 will actually run. Framing it that way to the teacher avoids
the impression that we're reporting live results here; we're reporting a *design*,
validated by definition only (the cell output literally just confirms the functions
were defined successfully).

Three components are defined:

1. **Multi-Label Baseline Pipeline (Binary Relevance with LightGBM).** The stated
   modeling approach for `Y_model`'s 689 targets in Sprint 2: one LightGBM binary
   classifier per target ("binary relevance"), chosen as a strong, fast baseline before
   trying more complex multi-output or chain approaches.

2. **`reconstruct_full_predictions()` — deterministic post-processing.** Takes the
   model's predictions on the 689 `Y_model` columns and recombines them with the 42
   `Y_additive` columns, computed directly from the CRM test features using the rules
   from 11.2 (`bil_to_crm_triggers`), then reorders everything back into the original
   731-column BIL schema. This is what actually realizes the benefit of the
   Section 11.2/11.3 split: the final submission always gets those 42 columns exactly
   right, for free, regardless of how the model performs.

3. **`evaluate_multilabel_performance()` — evaluation suite.** Computes ROC-AUC
   (micro/macro), F1 (micro/macro), and — the metric that actually matters for this
   competition — **Exact Match Ratio (EMR)**: the fraction of rows where *all* 731
   predicted BIL bits match the true row exactly. The other metrics are diagnostic
   (they tell you *how* a model is failing), but EMR is the only one aligned with the
   competition's grading, per Sprint 1's own framing.

**One-line summary for the teacher:** *Section 12 is the modeling contract Sprint 2
will implement against: a per-target LightGBM baseline, a reconstruction step that
stitches deterministic rule-based predictions back in, and an evaluation function
centered on EMR (not per-label accuracy) because that's what the competition actually
scores.*

---

## 4. Anticipated teacher questions

- **"Why not just use Section 11's pipeline as the final one?"** Because it hasn't been
  validated against held-out EMR yet — that comparison is explicitly Sprint 2's job.
  Section 11 is a candidate, not a conclusion.
- **"Why only 42 additive rules, when Section 8b suggested more structure?"** The
  criteria here are deliberately strict (100% consistency, ≥5 supporting pairs, exactly
  one BIL column added, none removed) to avoid baking training-set noise (the
  provisioning errors mentioned in the project brief) into a "deterministic" rule that
  isn't actually always true.
- **"Isn't losing 0 features at the χ² step suspicious?"** No — it's a sanity check
  that the CRM feature space, after the 11.1 filter, is already fairly clean and none
  of the 742 remaining columns are statistical noise relative to the 689 targets; the
  filter still matters because it's the only thing standing between "keep everything"
  and a principled test.
- **"Does Section 12 report real model performance numbers?"** No — it only defines the
  functions that Sprint 2 will use to train and evaluate the actual LightGBM model.
