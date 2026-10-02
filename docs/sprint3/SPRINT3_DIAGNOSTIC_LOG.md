# Sprint 3 — Diagnostic & Improvement Log

Audit trail for the post-Sprint-2 investigation into why EMR is below the ~98% public
benchmark. One entry per step: plan → what was run → what the data showed → decision.

## Ground rules

- **Python scripts only** (`scripts/diag*.py`, `scripts/exp*.py`) for every experiment.
  The notebooks are **not** touched until the team lead gives explicit approval.
- **Model selection uses the grouped validation split only** (GroupShuffleSplit on the exact
  CRM configuration, consensus targets — see `src/validation.py`). `solution.csv` is
  reported as an external diagnostic at milestones, never used to pick features,
  parameters or thresholds, and test errors are not broken down to guide fixes.
  _Confirmed by the team lead on 2026-10-02._
- **Data-usage disclosure.** `solution.csv` is read in exactly one place:
  the end of `scripts/train_final.py`, after the model is trained and the submission is
  written, to log the diagnostic score (verified by searching `src/` and `scripts/`).
  `test.csv` **inputs** (CRM columns, no answers) were compared with train inputs in the
  audits (`diag1`, `diag2`). That comparison shaped two filters: the "test-like"
  definition and the 0.2% rare-pack threshold. This is standard covariate-shift analysis
  and uses no test labels, but those filters were designed knowing what test inputs
  look like.
- Every result lands in this log and in `data/cache/results.jsonl` → `docs/EXPERIMENT_LEADERBOARD.md`.

## Results so far (summary for the presentation)

| Model | Clean-like val EMR (seed 42 / 7) | Test EMR (diagnostic) |
|---|---|---|
| Sprint 2 champion | 95.93% / 95.06% | 95.04% |
| Step 3: same model, ~7% suspected-corrupted training rows removed | 96.57% / 96.15% | 97.04% |
| **Step 4: + regularisation re-tuned on the clean data (`min_child_samples` 20 → 5)** | **96.81% / 96.33%** | **97.27%** |
| Step 5: confident learning / repair / more capacity | none better beyond noise | — (not selected) |
| Public benchmark (reported) | — | ~98% |

**One-line story:** the training data contains rows into which random products were
injected (visible as impossible one-hot categories and bursts of rare packs). Removing them
from training gains +2.0 test points; re-tuning the regularisation that had been chosen
while that corruption was present adds +0.23. The remaining errors sit mostly in rows with
1–2 rare packs, as missed billing items.

## Baseline at the start of Sprint 3

Champion from Sprint 2 (`scripts/train_final.py`): one LightGBM per BIL column, raw 745
binary CRM columns, unique CRM configurations with consensus labels,
`min_child_samples=20`, `reg_lambda=1`, threshold 0.5, plus the 42 Section 11 additive rules.

| Measure | Value |
|---|---|
| Validation EMR (seed 42 / seed 7) | 87.84% / 86.49% |
| Test EMR (`solution.csv`, diagnostic) | 95.04% |
| Test rows 1 bit off / 2+ bits off | 4.12% / 0.84% |
| Validation rows 1 bit off / 6+ bits off | 1.87% / 7.65% |
| F1 micro / macro (validation) | 0.966 / 0.372 |
| Public benchmark (reported by the team) | ~98% |

Gap to close on test: **~3 points, almost all of it single-bit errors.** On validation the
picture differs (multi-bit failures on rows with rare CRM packs), so the two error types
may need different fixes.

## Professor's hints → current status

| Hint | Status at start of Sprint 3 |
|---|---|
| 1. PCA is a bad choice | Already confirmed in Sprint 2: Section 11's 45 SVD components cost ~33 pts (E3e 47.42% vs 80.05% raw). The champion uses **no** PCA/SVD. Step 1 re-verifies this in code. |
| 2. Columns with numbers/letters may be badly one-hot encoded; look for hidden groupings | **Open.** Two candidates: (a) the categorical blocks `BUSINESS_LINE` / `SUBSCRIBER_TYPE` / `SUBSCRIBER_STATUS`, which contain a value literally called `0` (a missing value encoded as a category?) and are predicted column-by-column, so the model can output zero or two statuses for one customer; (b) the `*_<id>_PACK` columns, whose numeric ids may encode product families (Sprint 2 saw consecutive CRM ids `1521/1522/1523/1526` all pointing at the same BIL packs). |
| 3. Other hidden preprocessing problems | **Open.** Known suspects: Sprint 1 dropped `CRM_2396/2398` as "constant in train" although both are active in test; 27% of rows bill exactly 2 fewer packs than the CRM lists (some CRM packs never bill?); consensus-label construction; validation vs test distribution shift. |

---

## Step 1 — Diagnostic audit (read-only)

**Script:** `scripts/diag1_audit.py` (trains nothing, writes nothing; ~1–2 min).
**Run:** `.venv/Scripts/python scripts/diag1_audit.py > data/cache/diag1.txt`

| ID | Question | What would confirm a problem |
|---|---|---|
| D1 | How is every column encoded? Which categorical groups/values exist in CRM vs BIL, train vs test, and is each group a valid one-hot (exactly one active per row)? | Groups with rows where 0 or 2+ values are active; a value `0` behaving like "missing"; values present in test but not train. |
| D2 | Is each BIL categorical block simply a copy of the matching CRM block? | CRM value == BIL value on ~100% of rows → these columns should be copied, not learned. |
| D3 | Where do the champion's validation errors live (categorical vs pack columns), and does it output invalid one-hot rows? Two what-ifs: force exactly one value per BIL group (argmax), and copy CRM categorical values into BIL. | Many rows with 0/2+ predicted statuses; EMR jumps under either what-if. |
| D4 | Which columns alone break otherwise-perfect rows? | A few categorical or pack columns dominating the 1-bit errors. |
| D5 | Is there hidden structure in the numeric pack ids? Strength of each CRM pack's best BIL partner, consecutive ids sharing partners, id-order correlation. | Families of consecutive ids → a grouping feature or a deterministic id mapping the trees cannot see from isolated binary columns. |
| D6 | Pipeline sanity: no PCA/SVD in the champion's features; columns constant in train but active in test; BIL columns that never occur in train; exact duplicate CRM columns; which CRM packs appear in rows that bill fewer packs than they list. | Unlearnable or mishandled columns; a set of "non-billing" CRM packs. |

**Results** (full output: `data/cache/diag1.txt`; the test-like follow-up is section D7 of the same script).
The script reproduces the champion exactly (validation EMR 87.8411%).

1. **The categorical columns are not what limits EMR.** If every categorical BIL column
   were predicted perfectly, validation EMR would rise only to **88.12%** (+0.28). If every
   *pack* column were perfect it would reach **97.75%**. 35,563 of 36,578 wrong cells are
   pack columns. Forcing exactly one predicted status per customer (argmax) gives +0.01;
   copying the CRM status into BIL gives −0.01 (CRM and BIL statuses agree on only 97.8% of
   rows, e.g. 1,072 rows CRM `Active` → BIL multi-status).
2. **But the one-hot blocks expose an anomalous training subpopulation that the test set
   does not have.** In train, 1.0% of rows carry 2+ subscriber statuses, 0.2% carry 2+
   subscriber types, and ~0.1% use the value literally named `0` (or `Deactive`, `Idle`,
   `Pre_Active`). In test, **every** row has exactly one status and one type and the `0`
   value is essentially absent (1 row). Overall: 1.52% of train rows are "not test-like" vs
   0.001% of test rows. These rows look like several subscriptions merged into one record:
   they list **21.2 CRM packs on average vs 12.2** for normal rows (test: 13.4).
   - The champion gets **0.53% EMR** on them (567 val rows); on the test-like val rows it
     gets **89.22%** (35,818 rows).
   - They account for 17% of the validation rows with 6+ wrong bits — a real but partial
     explanation of the validation/test gap.
3. **Validation is also "heavier" than test in a second way.** Validation rows that
   contain a rare CRM pack (11% of val) list a median of 19 packs; a typical test row lists
   13. Status mix also shifts (Barring 10.7% train vs 5.1% test; Block_2_Way 6.0% vs 8.8%).
   So even test-like validation (89.2%) still understates test (95.0%).
4. **Pack ids: no simple numeric family structure.** Only 16 of 726 consecutive-id pairs
   share a BIL partner. But the mapping is clearly **many-to-one / many-to-many**: 383 of 727
   CRM packs have no BIL pack they pair with above 0.2, 483 of 717 BIL packs have no CRM
   partner above 0.5, and 113 BIL packs are the best partner of several CRM packs. Strong
   1:1 pairs show an inverse id ordering (Spearman −0.52). Some CRM packs come as fixed
   bundles (`1529/1532/1533/1541/1542` each in ~79k rows; `1536` and `1539` always together).
5. **"Bills 2 fewer packs than it lists" (27% of rows) is one bundle.** `CRM_1536` +
   `CRM_1539` are present in 98% of those rows vs 0.4% of rows where counts match: that pair
   produces no billing items of its own. This is structure the trees can learn, not a bug.
6. **Sanity checks pass.**
   - The champion uses the 745 raw CRM columns with no PCA/SVD → **hint 1 closed**.
   - `CRM_2396/2398` (never active in train, identical to each other) affect only 2 test
     rows → negligible.
   - No BIL column is unlearnable (every one occurs in train).
   - The top single-bit offenders are pack columns, mostly **missed positives**
     (`BIL_3940`, `4289`, `4297`, `3803`, `3812`), plus `BIL_SUBSCRIBER_STATUS_ORIG_Active` (47 rows).

**Decision / interpretation of the hints:**
- Hint 1 (PCA): closed — not used.
- Hint 2 (one-hot): the status/type columns themselves cost <0.3 pts, but their invalid
  encodings flag a ~1.5% block of merged-looking training records absent from test. That
  is a preprocessing problem worth fixing.
- Hint 3 (hidden problems): (a) the validation population is harder than test, so the
  selection metric should be made test-like; (b) the remaining ceiling is in pack columns
  with many-to-one mappings.

---

## Step log

| Step | Date | Script | Key finding | Val EMR | Decision |
|---|---|---|---|---|---|
| 0 | 2026-10-02 | — | Baseline (Sprint 2 champion) | 87.84% | Start audit |
| 1 | 2026-10-02 | `diag1_audit.py` | Categoricals cap at +0.28; pack columns cap at 97.75%. 1.52% of train rows have invalid one-hot categoricals (0.001% of test), 21 packs/row, 0.5% EMR. Test-like val rows: 89.22%. | 87.84% (89.22% test-like) | Step 2 approved |
| 2 | 2026-10-02 | `exp6_clean_rows.py`, `diag2_why.py` | Dropping non-test-like rows: +0.1. Root cause: ~8% of train rows have random rare CRM + BIL packs injected (uniform usage, 8.0% train vs 0.55% test). Champion on the rest of val: 95.7–95.9% ≈ test. | 87.93% (89.32% test-like) | Step 3 approved |
| 3 | 2026-10-02 | `exp7_drop_corrupted.py`, `diag3_remaining_errors.py`, `train_final.py 4` | Clean-like metric tracks test (95.9 vs 95.0). Dropping rows with ≥4 rare packs or invalid categoricals (7%): clean-like 96.57% / 96.15% (seeds 42/7). **Test 95.04% → 97.04%.** Remaining errors: 78% in rows with 1–2 rare packs, mostly missed items. | 96.57% clean-like (88.41% all) | Step 4 approved |
| 4 | 2026-10-02 | `exp8_retune.py`, `exp9_rare_thresholds.py`, `train_final.py 4 5` | `min_child_samples` 20 → 5 on clean data: clean-like 96.81% / 96.33% (seeds 42/7). `reg_lambda=0` collapses (one rare column fires on 27% of rows). Rare-column thresholds: +0.07/+0.10, within noise, not adopted. **Test 97.04% → 97.27%.** | 96.81% clean-like (88.65% all) | Step 5 approved |
| 5 | 2026-10-02 | `exp10_step5.py` | Confident learning −0.27 / −0.91, repair −0.26, capacity +0.06 (within noise) — mean over both seeds. None adopted; Step 4 model stays champion. Remaining errors look like an irreducible floor for this approach. | 96.81% clean-like (unchanged) | Step 6 agreed for next session |

---

## Step 2 — Test-like metric, dropping anomalous rows, and *why* the errors happen

Approved 2026-10-02. Scripts: `scripts/exp6_clean_rows.py` (experiments),
`scripts/diag2_why.py` (read-only; output `data/cache/diag2.txt`). New shared code:
`src/validation.test_like_mask()`; `src/experiment.score()` now reports `emr_testlike`
for every run, and the leaderboard shows it.

### 2.1 Test-like metric and dropping the anomalous rows

| Run | Val EMR (all) | Val EMR (test-like rows) |
|---|---|---|
| E6-base — Sprint 2 champion, re-scored | 87.84% | 89.22% |
| E6-clean — same model, the 1.52% non-test-like rows dropped from training | **87.93%** | **89.32%** |

Small, consistent gain (+0.1). One-bit misses drop from 1.87% to 1.69% of rows.

### 2.2 Why do the non-test-like rows exist? — *not* merged customers

- **Hypothesis "two customers merged into one record": rejected.** Of 600 sampled
  multi-status rows, **0** are exactly the OR of two normal training configurations.
- **Hypothesis "status changed during the extraction window": unlikely.** `Active` is
  paired with *every* other status at almost the same rate (`Barring` 142, `Block_1_Way` 141,
  … `Idle` 121, and even the missing-value `0` 127), although those statuses differ in
  natural frequency by up to 100×. Real status histories would not be this uniform. This
  points to **errors injected at random** by whoever built the dataset — consistent with the
  brief's "provisioning errors" in train and correct pairs in test.

### 2.3 Why do some pack columns get missed?

For the five most-missed BIL packs:
- `BIL_4289`, `4297`, `3803`: **no CRM pack predicts them** — the best CRM pack gives
  P(BIL pack | CRM pack) ≤ 0.02, a dedicated depth-3 tree recalls 0%, and the missed rows
  look like ordinary customers.
- `BIL_3803` and `BIL_3940` **flip between customers with identical CRM configurations** in
  97% and 84% of the repeated configs where they appear (well-predicted columns: 0–2%).
- But columns of this kind (185 flip in ≥50% of their configs) explain only **0.81%** of
  test-like validation rows. Label noise in single columns is real but minor.

### 2.4 The main cause: random rare packs injected into ~8% of training rows

Call a CRM pack "rare" if it appears in < 0.2% of training rows (471 packs).

| | Train | Test |
|---|---|---|
| Rows with ≥ 3 rare CRM packs | **8.00%** | **0.55%** |
| Distribution of #rare packs per row | long tail: 1,024 rows with 12+ | stops at 7 |

Evidence that these are injected, not real customers:
1. **Uniform usage.** In rows with ≥ 3 rare packs, the 471 rare packs are used almost
   equally often (coefficient of variation 0.12), with little relation to how common each
   pack is in test (Spearman 0.23). Real customers do not pick products uniformly at random.
2. **Matching count, independent identity on the BIL side.** The number of rare BIL packs
   per row tracks the number of rare CRM packs (diagonal of the crosstab: 5,294 rows with
   6+ of each). But Step 1 showed rare CRM packs have no consistent BIL partner (383 CRM
   packs pair with nothing above 0.2). So each injected CRM pack comes with an injected BIL
   pack of *unrelated* identity. That is exactly the catastrophic-row signature from
   Sprint 2: the right number of packs, ~9 false positives + ~9 false negatives.
3. **The status anomaly is part of the same corruption.** Non-test-like rows carry 6.4
   rare CRM and 6.3 rare BIL packs on average; test-like rows 0.5.

**This explains the whole validation/test gap:**

| Validation rows | Share | Champion EMR |
|---|---|---|
| ≥ 3 rare CRM packs | 8.4% | **1.64%** |
| all other rows | 91.6% | **95.73%** |
| other rows that are also test-like | — | **95.93%** (test: 95.04%) |

These two groups together account for **91.3%** of validation rows with 6+ wrong bits.

**Interpretation for the professor's hints:** the "improper one-hot encoding" shows up as
impossible category combinations (two statuses, the value `0`). They are the visible tip
of a corruption that also added random packs, and the corrupted rows do two kinds of
damage:
- they make validation look ~8 points worse than test;
- they teach the model false links for rare packs, which then misfire on legitimate rare
  packs in test (15% of test rows contain at least one rare pack).

**Decision:** pending team review.

---

## Step 3 — Clean-like metric, removing corrupted rows, remaining errors

Approved 2026-10-02. Scripts: `scripts/exp7_drop_corrupted.py` (experiments),
`scripts/diag3_remaining_errors.py` (read-only; output `data/cache/diag3.txt`),
`scripts/train_final.py 4` (final fit + test diagnostic). Shared code:
`src/validation.rare_pack_count()`, `clean_like_mask()`; every run now reports all /
test-like / clean-like validation EMR.

### 3.1 The clean-like metric

Clean-like validation rows = test-like (one value per categorical group, no `0`) **and**
fewer than 3 CRM packs that are rarer than 0.2% of training rows. 33,261 of 36,385
validation rows (91.4%) qualify.

| Model | All val | Test-like | **Clean-like** | Test (diagnostic) |
|---|---|---|---|---|
| Sprint 2 champion, seed 42 | 87.84% | 89.22% | **95.93%** | 95.04% |
| Sprint 2 champion, seed 7 | 86.49% | 88.05% | **95.06%** | — |

Clean-like EMR lands within ~1 point of test, where the all-rows number was 7 points off.
It is now the primary selection metric; the other two are still reported.

### 3.2 Removing suspected-corrupted rows from training

The model and its settings are held fixed (Sprint 2 champion); only the training rows
change. A training row is dropped if it is not test-like **or** carries ≥ k rare CRM packs.

| k (drop rows with ≥ k rare packs) | Train rows dropped | Clean-like val EMR (seed 42) | Seed 7 |
|---|---|---|---|
| no drop (Sprint 2 champion) | 0% | 95.93% | 95.06% |
| ∞ (only non-test-like, = E6-clean) | 1.5% | 96.03% | — |
| 2 | 9.8% | 95.45% | — |
| 3 | 8.1% | 96.41% | — |
| **4** | **6.9%** | **96.57%** | **96.15%** |
| 5 | 5.8% | 96.50% | — |
| 6 | 4.9% | 96.62% | 95.80% |
| 8 | 3.4% | 96.30% | — |

- Too aggressive (k=2) removes real customers and hurts; too lenient (k=8) leaves
  corruption in. Between them is a plateau (k=4–6) where differences on seed 42 (±0.06) are
  smaller than the noise of a 33k-row validation set (one standard error ≈ 0.10 points).
- To choose inside the plateau, k=4 and k=6 were re-run on the independent seed-7 split:
  k=4 +1.09, k=6 +0.73. **k=4 is selected** (best mean over both splits: 96.36%).

**Milestone test diagnostic** (computed once, after k=4 was fixed; not used for selection):
retrained on all of train minus 13,131 suspected-corrupted rows (7.0%):

| | Sprint 2 champion | **k=4 model** |
|---|---|---|
| Test EMR | 95.04% | **97.04%** (+2.00) |
| Test rows 1 bit off | 4.12% | 2.31% |
| Test Hamming loss | 0.000090 | 0.000061 |

Submission file: `data/derived/submission_sprint3_k4.csv` (97,100 rows, validated format).

### 3.3 What is still wrong (k=4 model, clean-like validation rows)

| | |
|---|---|
| EMR | 96.57% (1,142 wrong rows of 33,261) |
| Wrong cells | 4,085 = **628 false positives + 3,457 false negatives** |
| Rows with **no** rare CRM pack | **99.18%** EMR |
| Rows with **1–2** rare CRM packs (7.1% of rows) | **62.17%** EMR → **77.7% of all remaining wrong rows** |
| Wrong only in categorical columns | 5.4% of wrong rows |
| Wrong only in "flipping" columns (inconsistent for identical CRM, likely irreducible) | 16.6% of wrong rows |
| Wrong bits near the threshold (p in 0.35–0.65) | 6.2% → little room for threshold tuning |
| Upper bound if every 1-bit row were fixed | 97.79% |

Top single-bit offenders: `BIL_4167` (44 rows, all missed), `BIL_3940` (41; flips),
`BIL_4289` (36; flips), `BIL_3803` (21; flips). `BIL_4167` is the best BIL partner of the
rare `CRM_1761` (Sprint 2), i.e. a rare-pack mapping.

**Interpretation.** The remaining problem is rows with legitimate rare products, and the
model fails them mainly by **missing** billing items. A likely cause is the Sprint 2
regularisation (`min_child_samples=20`): it was chosen to suppress false positives at a
time when corrupted rows were still in both training and validation. A leaf needs ≥ 20
configurations, so a rare pack with only a few dozen clean examples cannot get its own
rule. This matters for test: 15% of test rows contain at least one rare pack, against 7.1%
of clean-like validation rows.

---

## Design choices and why (for the presentation)

| Choice | Alternatives considered | Why this one |
|---|---|---|
| **Validation split grouped on the exact CRM configuration** | Random row split | Only 0.006% of test rows have a CRM configuration seen in train. A row split lets 68% of validation rows be answered by memorising a twin row (Sprint 1 §3b); grouping reproduces the test condition (0% overlap). |
| **Consensus targets** (most frequent full BIL configuration per CRM configuration) | Raw labels; per-column majority | Train contains provisioning errors; test pairs are correct. Taking the mode of *whole* configurations removes noise in repeated configurations while always producing a configuration that really occurred. |
| **Select on validation only; `solution.csv` reported at milestones** | Tune on `solution.csv` | `solution.csv` is the answer key; using it to choose would turn the test score into a training score. It was computed only after each selection was frozen. |
| **Test-like definition**: exactly one value per categorical group, never `0` | Leave categories as-is | 99.999% of test rows satisfy it vs 98.5% of train. A customer cannot have two statuses or a status called `0`; these are encoding errors (the professor's one-hot hint). Uses CRM features only. |
| **Rare pack = in < 0.2% of training rows** | 0.1%, 0.5% | At 0.1% only 16 packs qualify (too few to catch the injection). At 0.5% legitimate packs dominate (5.6% of test rows would have ≥3). At 0.2% train and test diverge most (8.0% vs 0.55% of rows with ≥3), and the 471 packs are used near-uniformly in heavy rows (coefficient of variation 0.12), the signature of random injection. |
| **Clean-like metric uses ≥3 rare packs, fixed** | Re-define per experiment | Fixed so every experiment is scored on the same rows. ≥3 excludes 8.4% of validation rows but only 0.55% of test rows, so the metric still describes the test population. |
| **Drop corrupted rows (k=4) rather than repair them** | Strip rare packs from corrupted rows; re-weight | Dropping is transparent and needs no guess about *which* packs in a row were injected. Repair is a possible later refinement. k chosen by a sweep (2–8) plus a second split, not by a single best number. |
| **Model settings frozen while cleaning data** | Re-tune together | Changing one thing at a time attributes the gain (+2.0 test points) to the data cleaning alone. Re-tuning came after, in Step 4 (+0.23). |
| **Settings confirmed on two independent splits** | One split | Differences of ~0.1 points are within the noise of one 33k-row validation set; a choice is kept only if it holds on both seed 42 and seed 7. |
| **One-standard-error rule** (`min_child_samples=5`, not 2) | Take the single highest number | When settings are statistically tied, the more regularised one is less likely to overfit; the 0.02-point lead of 2 is a fifth of one standard error. |
| **Thresholds chosen on one split, checked on the other** | Tune and report on the same split | Tuning and scoring on the same rows inflates the gain. The rare-column threshold was not adopted because its checked gain was within noise. |
| **Negative results kept in the log** (Step 5) | Report only what worked | Confident learning, repair and extra capacity were each tested on both splits and rejected with a reason. That tells the professor the ceiling was probed, not assumed. |
| **No PCA/SVD; raw binary columns** | Section 11 pipeline with 45 SVD components | Measured in Sprint 2: SVD features cost ~33 points (trees split on continuous, configuration-specific values and memorise). |

---

## Step 4 — Re-tuning on clean data, rare-column thresholds

Approved 2026-10-02. Scripts: `scripts/exp8_retune.py` (regularisation sweep, both
seeds), `scripts/exp9_rare_thresholds.py` (thresholds), `scripts/diag3_remaining_errors.py
E8-mcs5-l1`, `scripts/train_final.py 4 5` (final fit + milestone test diagnostic). All
runs use the k=4 corrupted-row filter from Step 3.

### 4.1 Regularisation re-tuned on the cleaned training data

| `min_child_samples` | `reg_lambda` | Clean-like seed 42 | Seed 7 | Mean |
|---|---|---|---|---|
| 20 (Step 3 model) | 1 | 96.57% | 96.15% | 96.36% |
| 10 | 1 | 96.78% | 96.11% | 96.45% |
| **5** | **1** | **96.81%** | **96.33%** | **96.57%** |
| 2 | 1 | 96.82% | 96.36% | 96.59% |
| 5 | 5 | 96.61% | 96.25% | 96.43% |
| 5 | 0 | 33.00% | 26.29% | collapse |

- **Smaller leaves help on clean data** (+0.21 on the mean). The Sprint 2 choice of 20 was
  made while corrupted rows were still in training and validation, where small leaves
  mostly memorised injected packs. With them gone, small leaves can learn real rare products.
- **2 vs 5 is a tie** (0.02 points, a fifth of one standard error). **5 is selected** by the
  one-standard-error rule: among statistically tied settings, take the more regularised one.
- **L2 regularisation is essential.** With `reg_lambda=0`, the rare `BIL_4141_PACK` (true
  rate 0.17%) is predicted on 26.7% of rows: tiny pure leaves get extreme values that spread
  to unrelated customers. It is the same failure mechanism as Sprint 2's weighting bug.

**Milestone test diagnostic** (after selection; not used for it): **97.27%** test EMR
(97.04% before; rows 1 bit off 2.31% → 2.12%). Submission:
`data/derived/submission_sprint3_k4_mcs5.csv`.

### 4.2 Lower threshold for rare BIL columns — tested, not adopted

The threshold for BIL columns rarer than 0.2% (or 0.5%) was lowered from 0.5. To avoid
fitting the threshold to the rows it is scored on, it was **chosen on seed 42 and checked on
seed 7**.

| | Seed 42 (choose) | Seed 7 (check) |
|---|---|---|
| t = 0.5 (no change) | 96.81% | 96.33% |
| t = 0.45 for rare columns (best on seed 42) | 96.88% (+0.07) | 96.42% (+0.10) |

**Not adopted.** The gain is consistent in sign but within noise (one standard error ≈ 0.1),
and the two splits disagree on the best value (seed 7 peaks at t = 0.30, seed 42 at 0.45).
This matches the error analysis: only 4.6% of wrong bits have a probability between 0.35
and 0.65, so the errors are confident, not borderline.

### 4.3 What is left (selected model, clean-like validation rows)

| | Step 3 model | **Step 4 model** |
|---|---|---|
| EMR | 96.57% | **96.81%** |
| Wrong cells (false positives / false negatives) | 628 / 3,457 | 612 / 3,369 |
| EMR, rows with no rare CRM pack | 99.18% | 99.18% |
| EMR, rows with 1–2 rare CRM packs (7% of rows) | 62.17% | **65.63%** |
| Share of wrong rows that are rare-pack rows / only "flipping" columns | 77.7% / 16.6% | 76.0% / 16.5% |
| Upper bound if every 1-bit row were fixed | 97.79% | 97.84% |

**Why validation still understates test** (inferred from the single test number only, not a
breakdown of test errors): 11.4% of test rows contain 1–2 rare packs. If they scored the
65.6% they get on validation, test EMR could not exceed ~95.4%, yet the model reaches
97.27%. So validation's rare-pack rows must be harder than test's. The likely reason is
that some still carry *light* corruption (1–2 injected packs), which looks identical to a
genuine rare product and is not caught by the ≥ 4 filter.

---

## Step 5 — Light corruption, repair, capacity: none beats the Step 4 model

Approved 2026-10-02. Script: `scripts/exp10_step5.py` (all three, both seeds). Every run
starts from the Step 4 champion (k=4 filter, `min_child_samples=5`, `reg_lambda=1`).

| Clean-like validation EMR | Seed 42 | Seed 7 | Mean | vs champion |
|---|---|---|---|---|
| **Step 4 champion** | **96.81%** | **96.33%** | **96.57%** | — |
| 5.1 Confident learning, drop configs with ≥ 1 confident disagreement (10.5%) | 96.59% | 96.01% | 96.30% | −0.27 |
| 5.1 Confident learning, ≥ 2 disagreements (9.0%) | 95.38% | 95.93% | 95.66% | −0.91 |
| 5.2 Repair: add back 8.2k corrupted rows with rare packs stripped | 96.60% | 96.03% | 96.31% | −0.26 |
| 5.3 More capacity: 200 trees / 31 leaves | 96.87% | 96.39% | 96.63% | +0.06 |

### Why each idea did not help

- **5.1 Confident learning removes the wrong rows.** The out-of-fold model flags 21.5k
  "unexpected item" cells (an item is present but the model is sure it should not be) and
  only 1.4k "missing item" cells. Most flagged configurations are those with rare packs,
  and the model is confident there precisely *because* it has not learned those products.
  Removing them deletes the legitimate examples of rare products along with the corrupted
  ones. It also destabilises rare columns: with m=2, `BIL_4172_PACK` starts firing falsely
  on 438 validation rows (the same "rare column spreads to unrelated customers" failure seen
  with `reg_lambda=0`).
- **5.2 Repair adds a systematic error.** Stripping every rare pack from a corrupted row also
  strips its *legitimate* rare products. The repaired rows therefore teach "this
  configuration bills nothing for its rare items", which is wrong for real customers who
  own them. 7.8k new configurations of slightly wrong labels cost more than the extra
  common-product signal gains.
- **5.3 Capacity is within noise.** +0.06 on both splits is well inside one standard error
  (~0.1) and doubles the training time (~210 s vs ~120 s). By the one-standard-error rule
  the simpler Step 4 model is kept.

**Decision:** the Step 4 model remains the champion (`E8-mcs5-l1`; test diagnostic 97.27%).
No new test diagnostic was computed, because no new model was selected.

### Where this leaves the gap to ~98%

- On clean-like validation rows **without** rare packs the model is already at 99.18%.
- The remaining errors sit in rows with 1–2 rare packs. Three different attacks (Step 4
  thresholds, Step 5 cleaning, Step 5 repair) all failed to move them beyond noise. The
  evidence is consistent with part of that set being lightly corrupted rows that cannot be
  told apart from genuine rare products using CRM data, i.e. an irreducible floor for this
  approach.
- Remaining realistic headroom is small (upper bound if every 1-bit row were fixed:
  97.84% clean-like) and comes at rising cost.

---

## Step 6 — agreed, to start next session

_Status (2026-10-02): approved in principle by the team lead; work starts next session._

Sprint 3 is "Optimization **& Explainability**" (due 06/10/2026). The optimisation side has
reached diminishing returns, while explainability has not been started.

1. **Freeze the model** (Step 4 champion) as the Sprint 3 submission candidate
   (`submission_sprint3_k4_mcs5.csv`).
2. **Explainability in Python** (`scripts/explain_shap.py`):
   - SHAP for a sample of BIL columns (common, rare, and the "flipping" ones), checking
     each is driven by the CRM packs that should drive it;
   - global CRM → BIL importance map;
   - a few worked examples of wrong rows, showing *why* the model erred.
   This also serves the professor's interest in the noisy training labels.
3. **Notebook update only after explicit approval**, assembling Steps 1–6 into the Sprint 3
   notebook.
