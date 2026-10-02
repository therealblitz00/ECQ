# Sprint 2 — Iteration Log

Lab notebook. One entry per iteration: hypothesis, what was run, result, and what it
changes about the next step.

---

## Iteration 0 — Discovery: the exact-lookup ceiling is a training-set illusion

**Hypothesis:** Before building anything, check whether Sprint 1's headline
"exact-lookup EMR ceiling = 0.993" (`crm_bil_relationship_summary.csv`) is actually
achievable at test time, since the brief's Exp 1 leans on it as a primary strategy.

**Implementation:** Re-read Sprint 1 notebook cell 18 (`sprint1_preprocessing_v3.ipynb`,
§3), which directly measures how many test-set CRM configurations already appear in
train.csv.

**Result & error analysis:** Only **6 of 97,100 test rows (0.0062%)** have a CRM
configuration seen in train. The 0.993 ceiling describes train-internal determinism
(66,817/67,434 CRM configs map to exactly one BIL config), which is irrelevant to
generalization — it does NOT mean 99.3% of *test* rows are lookup-solvable. Exact lookup
is real but tiny at test time; the k-NN/majority fallback the brief specifies for unseen
configs is where essentially all test-set performance has to come from, and every
learned model (Exp 2–5) must generalize to configurations it has never seen, not just
memorize repeats.

**Next:** Build `src/validation.py` and `src/metrics.py` first (Phase 2), confirm the
GroupKFold split behaves the same way — i.e. that held-out validation folds also see
mostly unseen CRM configs relative to their training fold, so the local validation
number isn't inflated the same way a naive row-split would be (per
`validation_design.csv`). Then run Exp 1 to get the real (not train-internal) lookup +
k-NN coverage and EMR numbers.

---

## Iteration 1 — Harness validation + Section 11 rule audit

**Hypothesis:** A split grouped on the exact CRM configuration reproduces test conditions,
and Section 11's 42 "strictly additive" rules can safely hard-code 42 columns.

**Implementation:** `src/validation.py` (packbits keys, consensus targets, grouped holdout),
`src/data.py` (one-off `.npz` cache of raw + S10 + S11 arrays, ~50 s to build). Rules
applied to every training row and compared with consensus labels.

**Result & error analysis:**
- 1,315 training rows (0.70%) differ from their CRM group's consensus configuration;
  Section 10's relabeled targets already agree with consensus on 99.8% of rows.
- Split: 0.00% of validation rows have a CRM config seen in the training fold (test:
  0.006% vs all of train). Validation is an honest generalization test.
- **0 of 42 Section 11 rules are exact.** Each fails ~200 FP and ~220 FN times regardless
  of prevalence; applied as hard overrides they agree with consensus on only 94.4% of rows
  (i.e. they would cap EMR at ~94%). Failures concentrate in rows with many packs (24 vs 15
  on average). The "distance-1 pair" evidence behind the rules shows the *change* is
  consistent, not that the *level* is determined by one trigger.
- Side finding: 27% of rows have exactly 2 fewer BIL packs than CRM packs — some CRM packs
  systematically produce no billing item.

**Next:** Treat rules as an optional post-processing variant, never as a given.

## Iteration 2 — Exp 1: lookup & k-NN

**Hypothesis:** Near-identical CRM configurations have near-identical BIL configurations,
so a neighbour's consensus label is a strong baseline.

**Implementation:** `scripts/exp1_lookup_knn.py` — Hamming distance by chunked matrix
product over 53,947 unique training configurations.

**Result & error analysis:** Exact lookup covers 0% of val (0.006% of test vs all of train). 1-NN: 0.48%
EMR even though 70% of val rows have a neighbour at Hamming distance ≤1 — and 59% of 1-NN
predictions are exactly one BIL bit wrong. **Changing one CRM pack changes one BIL bit**:
the mapping is close to additive/column-wise, which neighbour copying cannot express. A
k=15 vote reaches 52% EMR by averaging the differing pack away. Section 10 vs raw
features: no difference.

**Next:** Per-label models, which can learn the per-pack mapping directly.

## Iteration 3 — Exp 3: Binary Relevance LightGBM, and a weighting bug

**Hypothesis:** One LightGBM per label learns the pack mapping; training on unique configs
with consensus labels removes provisioning noise.

**Implementation:** `src/models.BinaryRelevanceLGBM` (threaded, ~0.6 s/label),
`scripts/exp3_binary_relevance.py`.

**Result & error analysis (E3a, 58.66%):** 19% of val rows are exactly 1 bit off; 10% have
≥6 wrong bits. Errors are **confident** (median |p − 0.5| = 0.5; only 7% of wrong cells in
[0.2, 0.8]), so plain threshold tuning cannot recover them. `BIL_4420_PACK` alone causes
20% of the one-bit-off rows — always a false positive — predicted on 4.8% of val rows
against 0.12% true prevalence. Root cause: weighting unique configs by multiplicity (max
weight 8,114) lets a few heavy configs dominate leaf values. Unweighted or log-weighted
training cuts that column's cell error ~20× (0.0495 → 0.0026). Same for `BIL_3888`,
`BIL_4096`.

**Pipeline comparison as specified (E3b vs E3c):** Section 10 (all rows, relabeled) 50.6%;
Section 11 (all rows, raw labels, +SVD) 12.8%, 14.9% with rules. All-rows training is
itself implicit multiplicity weighting, so these numbers mix the weighting effect with
the feature-set effect.

**Next:** E3a2 (unweighted), E3d/E3e (S10 / S11 features on identical deduplicated,
unweighted, consensus rows) to isolate the SVD effect. Then Exp 2 (LP decoder on E3a2
probabilities) and Exp 4 (co-occurrence chain).

## Iteration 4 — Weighting fix + clean Section 10 vs Section 11 ablation

**Hypothesis:** Removing multiplicity weights fixes the rare-label false positives, and
training every feature set on identical rows isolates what each Sprint 1 pipeline does.

**Implementation:** `scripts/exp3_binary_relevance.py E3a2 E3d E3e` — unique configs,
consensus labels, no weights; only the feature matrix changes.

**Result & error analysis:**

| Features (same rows, same labels) | Val EMR |
|---|---|
| raw 745 CRM | **80.05%** (80.38% with the 42 rules) |
| Section 10 (734; \|r\|≥0.95 CRM clusters collapsed) | 74.31% |
| Section 11 (742 CRM + 45 SVD) | 47.42% (49.16% with rules) |

- Weighting fix alone: 58.66% → 80.05%.
- Section 10's column collapsing costs ~6 pts: correlated CRM packs each drive a
  *different* BIL pack (Sprint 1 §6b had flagged this), so dropping "redundant" columns
  deletes signal.
- Section 11's CRM subset differs from raw by 3 near-empty columns, so its ~33-pt drop is
  the **SVD components**: continuous, configuration-specific values that trees split on
  to memorise configurations instead of learning the per-pack mapping.
- The 42 rules, used as post-processing on a good model, add +0.3 pts.

**Conclusion for the Sprint 1 comparison deliverable:** keep Sprint 1's *analysis* (the
consensus/noise findings, validation design, the rules) but feed models the raw 745
binary CRM columns; neither pipeline's feature transformations help.

## Iteration 5 — Exp 2 (label powerset) and Exp 4 (classifier chain)

**Hypothesis:** Enforcing joint label structure fixes rows where independent labels
produce inconsistent combinations.

**Implementation:** `scripts/exp2_label_powerset.py` (LP as a max-likelihood decoder over
all 53,369 observed training configs, since multi-class over them is untrainable and the
top-1,000 configs cover <1% of val rows); `scripts/exp4_classifier_chain.py` (each label
sees X + its 30 most co-occurring upstream labels; prevalence order).

**Result & error analysis:** LP reaches 23.09% — essentially its hard ceiling of 23.12%
(share of val rows whose BIL config exists in train). **LP is structurally capped**:
unseen CRM configs mostly produce unseen BIL configs. Snapping only near-certain rows
leaves BR unchanged (80.06%). The chain gives 80.57% (+0.5 over BR).

Error analysis on BR: 6.7% of val rows are 1 bit off, but **9.1% have ≥6 wrong bits**
(~9 FP + ~9 FN each: right pack count, wrong identities). Sprint 1 shows provisioning
noise is small (median 1 bit, max 10), so these are model failures, not label noise.
They are driven by **348 moderately rare CRM packs** (50–200 training rows): val rows
containing one have 2.8% EMR vs 89.7% for the rest. These packs have no dominant BIL
partner (e.g. `CRM_1761` → best partner in only 10.8% of rows), so their billing
depends on pack combinations the trees see too few times to learn. **The test set has
~11× fewer of these packs per row than validation (0.04 vs 0.43)**, so validation EMR
likely understates test EMR.

## Iteration 6 — Exp 5: hybrid post-processing and the variance finding

**Hypothesis:** Rules, a tuned threshold, and blending BR with the chain each add a
little; k-NN may beat trees on rare-pack rows.

**Implementation:** `scripts/exp5_postprocess.py` on saved validation probabilities.

**Result & error analysis:** threshold 0.6 adds ~+0.2 pts per model. Averaging BR and
chain probabilities is 78.1% at t=0.5 (≈ OR of the models) but **87.21% at t=0.6
(≈ AND)**. Verified directly: AND 87.27%, OR 75.58%. The models disagree on 60k cells in
24% of rows, and where they disagree **the truth is 0 in 97.6% of cases** — each model
emits many confident, idiosyncratic false positives the other does not. That is a
variance problem (overfitting rare labels). k-NN routing for rare-pack rows does not help
(86.6% at R=200, 80.7% at R=400).

Caveat: the threshold and AND choice were picked on this validation split, so they need
confirming on an independent split before being trusted.

**Next:** Attack the variance directly — regularized per-label models (min_child_samples
20/50) and seed/feature bagging — then confirm the best pipeline on a second split seed.

## Iteration 7 — Variance reduction (the decisive step)

**Hypothesis:** If the confident false positives are variance, stronger per-leaf
regularization or bagging should remove them without needing a second model to veto them.

**Implementation:** `scripts/exp5_regularize.py` — `min_child_samples` 20 / 50 with
`reg_lambda=1`; 3-model bagging (row/feature subsampling 0.8); then the chain re-run with
the same regularization (`exp4_classifier_chain.py 30 20`) and a BR/chain blend.

**Result & error analysis:**

| Run | Val EMR |
|---|---|
| BR, `min_child_samples=20`, t=0.5, + rules | **87.84%** |
| BR, `min_child_samples=50` | 87.79% |
| 3× bagged BR (default regularization) | 87.28% (3× the time) |
| Chain, `min_child_samples=20` | 87.71% |
| Blend of the two regularized models | 87.72% |

One regularization change is worth +7.5 pts over E3a2+rules and beats the AND ensemble at
the default threshold. Once regularized, chain and blend stop helping: the label
dependencies the chain exploited were mostly compensating for overfitting. One-bit misses
fall to 1.9% of val rows; ~9.5% still have ≥3 wrong bits — the rare-pack rows.

## Iteration 8 — Confirmation, final fit, external diagnostic

**Hypothesis:** The champion's choices (regularization, t=0.5, rules) are not artefacts of
one validation fold.

**Implementation:** Same recipe on an independent split (`exp5_regularize.py seed=7`);
then `scripts/train_final.py` on all 67,434 unique configurations.

**Result & error analysis:** Seed 7: 86.49% (seed 42: 87.84%); t=0.5 beats t=0.6 and the
rules add ~+0.2 pts on both splits. Selection frozen. The submission is validated in code
(97,100 rows, 731-char 0/1 strings, unique MSISDN, no NaNs).

External diagnostic on `solution.csv`, computed only after freezing: **95.04% test EMR**,
Hamming loss 0.00009; 4.1% of rows are one bit off and <1% worse. The gap over validation
is what Iteration 5 predicted: test rows carry ~11× fewer rare CRM packs, and test labels
are clean.

**Next (Sprint 3):** tune regularization and per-label thresholds under grouped CV; attack
rare-pack rows (interaction features, pooling across rare packs); SHAP to check each BIL
column is driven by the CRM packs that should drive it.
