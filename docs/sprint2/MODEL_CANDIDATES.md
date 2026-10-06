# Sprint 2 — Model candidates, ranked

Grounded in Sprint 1's derived data (`crm_bil_relationship_summary.csv`,
`validation_design.csv`, `bil_predictability.csv`, `additivity_summary.csv`,
`coclustering_quality.csv`, `label_imbalance.csv`, `config_coverage_summary.csv`), not
guesses. Ranking = the order to actually implement and benchmark in Sprint 2, from
"build first, costs nothing" to "only if time remains."

**Non-negotiable for every entry below:** benchmark under **GroupKFold /
GroupShuffleSplit on `crm_code`**, not a row-level split. `validation_design.csv` shows
a plain row split would report an inflated EMR (~0.95) purely from memorizing repeated
CRM configs — 68% of validation rows are memorizable that way. This especially matters
for ranks #1–#2, which are literally memorization strategies and would look
artificially strong under a leaky split.

---

## Tier A — Baselines (build first, near-zero cost)

### 1. Exact-lookup / majority-vote table (CRM config → BIL config)
**Why:** 66,817 of 67,434 unique CRM configs (99.1%) map to exactly one BIL config;
training-set lookup ceiling is EMR = 0.993. Costs nothing to build and is the floor
every other model must beat — build this before anything else so every later model has
a real number to compare against.

### 2. k-NN on Hamming/Jaccard distance over CRM vectors
**Why:** Test will contain CRM configs never seen in train, where #1 can't answer.
`coclustering_quality.csv` shows k=10 clusters already capture 99.5% of strong
CRM–BIL associations, so nearest-neighbor majority-vote is a cheap, natural
generalization of the lookup table for unseen configs.

### 3. Logistic Regression One-vs-Rest (fast linear baseline)
**Why:** `bil_predictability.csv` shows many labels are nearly linearly separable from
a single CRM feature (Jaccard ~0.98–0.99 with their best predictor). Cheap ablation
that quantifies how much lift tree models actually buy over linear ones — useful even
if it's not the final model.

---

## Tier B — Core candidates (the real modeling work)

### 4. Expanded deterministic rule layer (association-rule mining)
**Why:** Section 11.2 found only 42 *strictly* perfect rules, but
`additivity_summary.csv` shows up to 538 candidate columns have distance-1 CRM
triggers with **median modal share 0.997** — a much bigger reservoir of
near-deterministic single-pack rules than currently exploited. Mining these with a
confidence threshold (not just modal_share == 1.0) could deterministically cover far
more than 42/731 labels before any model runs, directly raising the EMR ceiling for
free.

### 5. LightGBM Binary Relevance (per-label)
**Why:** The planned Section 12 baseline. 742 binary, sparse CRM features; 607/731 BIL
labels are <1% prevalence (`label_imbalance.csv`). Tree boosting handles sparse binary
input and severe imbalance natively, and per-label training on 187k rows is fast
enough to be practical across ~689 remaining targets.

### 6. Classifier Chains ordered by correlation clusters
**Why:** `near_duplicate_bil_pairs.csv` shows tight label groups (e.g. BIL_3921 /
3922 / 3927 / 3933 / 3934, all pairwise Jaccard > 0.98). Plain Binary Relevance
predicts these independently and can produce internally-inconsistent rows; chaining
predictions forward should recover that joint structure — directly relevant because
EMR needs *all* labels right, not marginal accuracy.

### 7. Community-clustered multi-label models
**Why:** `pack_communities.csv` + `coclustering_quality.csv` show the CRM/BIL
structure is locally clustered (~10–40 tight communities), not globally correlated.
One multi-output model per community matches the actual dependency structure and is
cheaper than either a single global model or 731 fully independent ones.

---

## Tier C — Stretch / comparison candidates

### 8. Label Powerset restricted to correlated communities
**Why:** Within each small correlated community (~10–25 columns), the powerset of
observed combinations is tractable and directly optimizes "all labels in this block
correct" — the closest available proxy to EMR at sub-problem granularity. Global
label powerset is not viable: `config_coverage_summary.csv` needs 29,713 distinct BIL
configs just for 80% coverage, and the top config is only 4.3% of rows.

### 9. CatBoost multi-output (native `MultiLogloss`)
**Why:** Worth a head-to-head against #5 — shared-tree multi-output boosting can
exploit inter-label correlation directly and trains once instead of 689 times, at
some risk of underfitting the rare labels that LightGBM's per-label class weighting
handles more precisely.

### 10. Hybrid pipeline: rules (4) + community chains (6/7) + k-NN fallback (2), blended
**Why:** No single approach above is likely to dominate alone — a meaningful share of
rows are "confidently exact" (singleton CRM configs, or covered by expanded
deterministic rules) while the rest genuinely need learned structure. Evaluating the
*combination* end-to-end through `evaluate_multilabel_performance()`'s EMR is itself
the Sprint 2 deliverable, not just picking one algorithm.
