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
| **Step 8 — final model: Step 4 without the Sprint 1 additive rules** (`notebooks/sprint3_final.ipynb`, professor's CSVs only) | **96.79% / 96.30%** | **97.21%** |
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

**Script:** `scripts/sprint3/diag1_audit.py` (trains nothing, writes nothing; ~1–2 min).
**Run:** `.venv/Scripts/python scripts/sprint3/diag1_audit.py > data/cache/diag1.txt`

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
| 6 | 2026-10-03 | `explain_shap.py` | SHAP top driver = independent co-occurrence driver for 95.7% of clearly-linked columns. Concentration on one driver predicts F1 (Spearman +0.86): errors are on items with no clear CRM cause. No reliance on injected rare packs. Rare-product misses are "right cause, p ≈ 0.2". | 96.81% clean-like (frozen) | Final notebook built (`notebooks/sprint3_final.ipynb`) |
| 7 | 2026-10-04 | `exp11_count_features.py` | Teammate's MCA lead (+0.31 in Sprint 2 set-up) tested as plain basket-size counts on the current champion: +0.03 / −0.05 mean, within noise. Not adopted. | 96.81% clean-like (unchanged) | Champion unchanged |
| 8 | 2026-10-04 | `notebooks/sprint3_final.ipynb` | Final notebook on the professor's CSVs only. Sprint 1 additive rules dropped (+0.02 on validation, within noise; mined on all of train). Test 97.21% (97.27% with rules, reported not used). | 96.79% clean-like | Final model |
| 9 | 2026-10-04 | `review_checks.py`, `exp12_review_models.py` | Mock jury review answered: configuration-level uncertainty, paired tests, 5-fold CV (96.83% ± 0.20), baselines (LR 95.97%), error detection (83–92% of known errors by reviewing 2% of customers, ~40× random), test CI [97.11, 97.31]. Rare threshold real but +0.03 — not adopted. | 96.79% clean-like (unchanged) | Final model unchanged |

---

## Step 2 — Test-like metric, dropping anomalous rows, and *why* the errors happen

Approved 2026-10-02. Scripts: `scripts/sprint3/exp6_clean_rows.py` (experiments),
`scripts/sprint3/diag2_why.py` (read-only; output `data/cache/diag2.txt`). New shared code:
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

Approved 2026-10-02. Scripts: `scripts/sprint3/exp7_drop_corrupted.py` (experiments),
`scripts/sprint3/diag3_remaining_errors.py` (read-only; output `data/cache/diag3.txt`),
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

Approved 2026-10-02. Scripts: `scripts/sprint3/exp8_retune.py` (regularisation sweep, both
seeds), `scripts/sprint3/exp9_rare_thresholds.py` (thresholds), `scripts/sprint3/diag3_remaining_errors.py
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

## Threats to validity

Raised in an external review (2026-10-02) and checked against the code. Each one is stated
with its measured or bounded impact.

| Risk | Status | Impact on the reported results |
|---|---|---|
| **Additive rules learned with validation labels.** The 42 rules (`additive_rules_v2.json`) were derived in Sprint 1 from all of `train.csv`, including rows that later fell in our validation folds. | Real leak, measured | The rules add only **+0.02 points** to the champion's clean-like validation EMR (96.79% → 96.81% on seed 42; 96.30% → 96.33% on seed 7), so the leak inflates validation by at most that. The **test score is unaffected**: the rules never saw test labels. |
| **Sprint 1 SVD and χ²+FDR selection fit on all of `train.csv`** before our split. | Real leak, **not used by the champion** | The champion uses the raw 745 CRM columns: no SVD, no FDR selection (which kept all 742 features anyway). The SVD runs (E3c, E3e) were 33 points *worse* despite the leak, so the conclusion "SVD hurts" only gets stronger. |
| **Unsupervised statistics computed on all training rows.** The rare-pack frequency (0.2%) and the test-like definition use CRM features of every training row, validation included; the filters were designed by comparing train and test *inputs*. | Minor, disclosed | Feature-only (no labels), as disclosed under Ground rules. Consensus targets are computed per CRM configuration, and configurations never cross folds, so no label information crosses the split. |
| **Survival bias from removing rows.** | Addressed | Rows are removed from **training** only. Every model is scored on all / test-like / clean-like validation rows, and the Sprint 2 champion was re-scored on the same subsets for a like-for-like comparison. The test diagnostic uses **all 97,100 test rows, unfiltered** (95.04% → 97.27%). |
| **Threshold / post-processing overfitting.** | Addressed | Thresholds were chosen on seed 42 and checked on seed 7, then rejected (within noise); the champion uses 0.5. Early Sprint 2 threshold runs (E5, single split) are not part of the champion. |
| **Repeated looks at the test set.** `solution.csv` was scored at three milestones (Sprint 2 final, Step 3, Step 4). | Disclosed | Each look came after the selection was frozen on validation, and no choice used it. Still, three looks are three looks: the test number is a diagnostic, not an untouched holdout. |
| **Classifier chain with one fixed label order** (no random orders, no ensemble of chains). | Limitation | Once regularised, the chain tied plain per-column models (87.71% vs 87.84%), so per-column was kept. An ensemble of chains (5–10× the compute) was not tried. |
| **Label powerset cannot predict unseen combinations.** | Confirmed, rejected | Measured ceiling ~23% (unseen CRM configurations produce unseen BIL configurations). Tested only because the Sprint 2 brief required all four paradigms. |
| **Engineering: numbered scripts rather than a config-driven pipeline; derived CSVs in git.** | Accepted trade-off | Every run's settings and metrics are logged centrally (`results.jsonl` → generated leaderboard). Large data, caches and submissions are gitignored; tracked derived files are small Sprint 1 summaries (≤ 1.2 MB) kept as evidence for the graded notebook. Hydra/DVC would be over-engineering for a three-sprint course project. |

---

## Step 5 — Light corruption, repair, capacity: none beats the Step 4 model

Approved 2026-10-02. Script: `scripts/sprint3/exp10_step5.py` (all three, both seeds). Every run
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

## Step 6 — Explainability (SHAP)

Approved 2026-10-03. Script: `scripts/sprint3/explain_shap.py` (read-only; output
`data/cache/explain_shap.txt`). Per-column results: `data/derived/sprint3_shap_drivers.csv`
(731 rows). Figures: `docs/sprint3/figures/`.

**Set-up.** The champion is frozen (Step 4: `min_child_samples=5`, `reg_lambda=1`, k=4
filter, rules) and retrained on the seed-42 training fold; it reproduces the reported
clean-like EMR exactly (96.81%). Predictions are explained on clean-like **validation** rows,
where the correct answer is known. SHAP values come from LightGBM's built-in TreeSHAP
(`pred_contrib=True`), identical to `shap.TreeExplainer`, in log-odds.

**Two yardsticks:**
- *Importance* of a CRM column for a BIL column = mean |SHAP| over the rows where that BIL
  column is positive (true or predicted), i.e. "why does this item get billed?". A first
  version averaged over random rows. That under-weights rare drivers, which almost never
  appear in a random sample (it gave a misleading 11.6% agreement), so it was replaced.
- *Expected driver* = the CRM column that co-occurs best with the BIL column in the
  cleaned training data. It is computed independently of the model, so agreement between
  the two is a real check.

### 6.1 Does the model rely on the CRM columns it should?

| BIL columns | Count | Top SHAP driver = expected driver | Expected driver in SHAP top 3 |
|---|---|---|---|
| with a clear expected driver (co-occurrence score ≥ 0.8) | 233 | **95.7%** | **99.1%** |
| without a clear one | 498 | 46.4% | 50.2% |
| all | 731 | 62.1% | 65.8% |

Where the data has a clear CRM → BIL link, the model has learned **that** link. Examples
(figure `shap_selected_columns.png`):
- `BIL_3748` ← `CRM_1537` (+3.5 log-odds when present).
- `BIL_3921` ← `CRM_1529`, supported by the bundle packs `1542/1533/1541` that always come
  with it.
- `BIL_SUBSCRIBER_STATUS_ORIG_Active` ← `CRM_SUBSCRIBER_STATUS_ORIG_Active` (+3.6). The
  other status and type columns likewise follow their CRM counterpart.
- The rare `BIL_3782` ← the rare `CRM_2132` (+8.5).
- `BIL_4167` is driven by three CRM packs (`1876`, `1761`, `1719`) with near-equal weight.
  That is the many-to-one CRM → BIL mapping found in Step 1, and the model handles it
  (F1 0.98).

### 6.2 Why some columns fail: no single CRM cause

Concentration = share of a column's SHAP mass on its single top driver.

| Concentration | BIL columns | Median validation F1 | Share that are "flipping" columns |
|---|---|---|---|
| ≤ 0.3 (diffuse) | 299 | **0.00** | 46% |
| 0.3 – 0.5 | 116 | 0.67 | 28% |
| 0.5 – 0.7 | 150 | 0.95 | 7% |
| > 0.7 (one clear driver) | 166 | **0.99** | 4% |

Spearman(concentration, F1) = **+0.86** over 726 columns (figure
`shap_concentration_vs_f1.png`). **The model is right where a BIL item has one clear CRM
cause and wrong where it has none.** The diffuse columns are largely the ones that flip
between customers with identical CRM configurations (`BIL_3803`: F1 0.11, mostly fired by
common packs; its best co-occurring pack `CRM_2205` has a +5.4 effect but explains only a
few of its rows). Eight rare pack columns have a *status or type* column as top driver and F1 0:
with no real CRM cause, the model falls back on a generic signal. These columns are not
predictable from CRM data, which is the explanation for the irreducible floor seen in
Steps 4–5.

### 6.3 Noise check — is the model exploiting the injected corruption?

- Median share of SHAP mass on rare CRM packs (< 0.2% of rows): **1.1%**.
- 119 columns put > 50% of their mass on rare packs, but in **100%** of them the expected
  driver is itself that rare pack: a rare billing item explained by its own rare product,
  which is legitimate.
- No column relies on rare packs that are *not* its own driver.

So there is no sign the model learned the random injected links. That is consistent with
the corrupted rows being removed before training (Step 3).

### 6.4 Worked examples (clean-like validation rows wrong by exactly one bit)

| Case | What SHAP shows | Explanation |
|---|---|---|
| `BIL_4277` missed; the row has the rare `CRM_1908` | `CRM_1908` pushes **+5.97** against a baseline of −8.21 → p = 0.22 | The model found the **right** cause, but with few training examples the regularised evidence falls just short of 0.5. This is the mechanism behind the false-negative-heavy errors (Steps 3–4). |
| `BIL_4139` missed; rare `CRM_1727` | `CRM_1727` **+7.60** vs baseline −8.44 → p = 0.23 | Same pattern, on a flipping column. |
| `BIL_3940` missed (flipping column) | `CRM_1568` +4.48, `CRM_1587` +1.44 → p = 0.40 | Near-miss on a column that is inconsistent even for identical CRM customers. |
| `BIL_3796` missed; expected driver `CRM_2099` **absent** | No contribution above +0.9 → p = 0.002 | Nothing in the CRM data explains this item. Likely label noise or a billing-only item; no model could predict it from CRM. |

**Link to Step 4.2.** The near-miss cases (p ≈ 0.2–0.4 with the right driver) are exactly
what a lower threshold for rare columns would recover. That test gave only +0.07 / +0.10:
such cases exist but are few, while lowering the threshold also adds false positives.

### Conclusions for the presentation

1. The model has learned the real CRM → BIL product mapping: 96–99% agreement with the
   independent co-occurrence check wherever a clear link exists.
2. Its errors concentrate on billing items with no clear CRM cause (Spearman +0.86
   between "one clear driver" and F1); that part is not predictable from CRM data.
3. It does not exploit the injected training noise (no column leans on rare packs other
   than its own driver).
4. Remaining misses on rare products are "right cause, not enough evidence" (p ≈ 0.2).


---

## Step 7 — Basket-size count features (a teammate's MCA lead)

Approved 2026-10-04. Script: `scripts/sprint3/exp11_count_features.py` (both seeds).

**Origin.** A teammate tested adding 2 MCA "meta-variables" to the **Sprint 2** set-up (no
corrupted-row filter, `min_child_samples=20`; his results are in
`docs/sprint3/teammate_mca_experiments_log.csv`). All-rows
validation EMR rose 87.84% → 88.15% (+0.31, one split, no clean-like metric). His own analysis
noted the MCA components mostly proxy basket size (component 2 vs number of packs: +0.61).
Before adopting MCA, with its known "double zero" problem on sparse binary data (and given
that SVD components cost ~33 points in Sprint 2), the simple version of that signal was
tested on the **current** champion:

| Clean-like validation EMR | Seed 42 | Seed 7 | Mean | vs champion |
|---|---|---|---|---|
| Champion (Step 4) | 96.81% | 96.33% | 96.57% | — |
| + number of CRM packs per customer | 96.79% | 96.41% | 96.60% | +0.03 |
| + number of CRM packs and number of rare CRM packs | 96.85% | 96.19% | 96.52% | −0.05 |

**Decision: not adopted.** Both variants are within one standard error (~0.1) with opposite
signs across splits; all-rows EMR is also unchanged (88.63% / 87.69% vs 88.65% / 87.63%).
The likely reason the teammate saw a gain: in the Sprint 2 set-up the corrupted rows were still
in training, and basket size is a strong marker of those rows (21 packs vs 12). Once they are
removed (Step 3), the signal has nothing left to add. The teammate's exact MCA features were not
re-tested (code not available); the result above suggests their effect would also vanish.

---

## Step 8 — Final notebook on the professor's CSVs only; additive rules dropped

Approved 2026-10-04. Notebook: `notebooks/sprint3_final.ipynb` (Sprints 1–3, ~10 minutes).

**Requirement (team lead).** The final notebook must run on the files the professor gave —
`train.csv`, `test.csv`, and `solution.csv` only to score the frozen model at the end — and
contain all its own code (no imports from `src/` or `scripts/`).

**Consequences.**
- **The 42 Sprint 1 additive rules are dropped from the final model.** They live in a derived
  file (`additive_rules_v2.json`). On the champion they add only +0.02 / +0.03 clean-like
  validation points (96.81% → 96.79% on seed 42, 96.33% → 96.30% on seed 7), within one
  standard error, so the simpler model is preferred. They were also mined on all of
  `train.csv`, so dropping them removes the one documented leak (Threats to validity).
  The decision rests on validation. On test, the model without rules scores **97.21%**
  against 97.27% with them; that number was reported afterwards, not used to decide.
- **SVD evidence is recomputed live** from `train.csv` (45 components for 80% of variance,
  matching Sprint 1).
- **Sprint 2–3 experiment tables are written into the notebook's markdown** (they took hours
  to run and are not recomputed). The final model, the corruption evidence and SHAP are all
  computed live.

**Result (computed in the notebook's last cell):** test EMR **97.21%** (94,391 of 97,100 rows
exactly right), 2.18% of rows one bit off, F1 micro 0.9987. Every live number matches the
script-based results (clean-like 96.79%, SHAP Spearman +0.86, noise check unchanged).

---

## Step 9 — Response to a mock jury review

Approved 2026-10-04. An AI was asked to review the repository as a professor (64/100). Each
point was checked against our outputs; the assessment and roadmap are in
`docs/sprint3/REVIEW_RESPONSE_AND_ROADMAP.md`. About 80% of the review was right. It was wrong
on one stale fact (`BIL_4167` is now F1 0.98, not "all missed") and overstated the
"40% of billing items the model cannot verify" (see 9.5).
Scripts: `scripts/sprint3/review_checks.py`, `scripts/sprint3/exp12_review_models.py`.

**Decision rule set before running (what a good engineer would do):** a rejected change is
adopted only if (1) a paired test shows it is real on both splits, (2) it is practically
meaningful (≈ 0.1 point or more), (3) it adds no leakage or fragile tuning, and (4) it holds in
5-fold cross-validation.

### 9.1 Uncertainty

Rows of one configuration share prediction and consensus target, so validation uncertainty
must be computed by resampling configurations. The review estimated SE ≈ 0.16; the real
figure is larger, because a few configurations are very frequent:

| | Seed 42 | Seed 7 |
|---|---|---|
| Final model, clean-like EMR | 96.79% | 96.30% |
| Configuration-level SE (naive row SE) | 0.38 pt (0.10) | 0.34 pt (0.11) |
| 95% CI | [95.99, 97.48] | [95.59, 96.89] |

Every test row is its own configuration, so test uncertainty is binomial: **97.21%, 95% CI
[97.11, 97.31]**. The 97.21 vs 97.27 (with/without rules) difference is inside it; the ~98%
benchmark is not (≈ 767 customers short).

### 9.2 Paired tests of the key decisions (clean-like, B − A, points, 95% CI)

| Decision | Seed 42 | Seed 7 | Verdict |
|---|---|---|---|
| Drop suspected-corrupted rows (k=4) vs keep | +0.81 [+0.44, +1.19] | +1.25 [+0.91, +1.74] | real — adopted |
| `min_child_samples` 5 vs 20 | +0.27 [+0.08, +0.62] | +0.20 [+0.08, +0.35] | real — adopted |
| Sprint 1 rules on vs off | +0.02 [+0.00, +0.04] | +0.02 [+0.00, +0.05] | real but ~6 customers — dropped (leak, derived file) |
| Rare-column threshold 0.45 vs 0.5 | +0.07 [+0.02, +0.16] | +0.10 [+0.03, +0.23] | real, see 9.3 |
| More capacity | +0.04 [−0.04, +0.15] | +0.05 [−0.12, +0.23] | not significant |
| `min_child_samples` 2 vs 5 | −0.00 [−0.05, +0.04] | +0.03 [−0.03, +0.08] | not significant |
| Basket-size count feature | −0.02 [−0.08, +0.03] | +0.09 [−0.00, +0.23] | inconsistent |
| LightGBM vs logistic regression | +0.83 [+0.55, +1.24] | +1.69 [+0.76, +3.11] | trees significantly better |

The review was right that unpaired reasoning had dismissed one real effect (the threshold).

### 9.3 5-fold GroupKFold of the final model

Clean-like EMR **96.83% ± 0.20** (std over folds; 96.64–97.12); all rows 89.07% ± 0.21.
Rare-column threshold 0.45: **+0.03**, positive in all 5 folds (+0.017 to +0.044).
**Decision: not adopted.** It is real (7 of 7 evaluations positive) but about ten customers per
fold, below the ≈ 0.1-point bar set before testing. Raising it after seeing the result would be
choosing the rule to fit the data. Recorded as an optional, low-value improvement.

### 9.4 Baselines on the same footing (clean-like validation, seed 42 / 7)

| Model | Seed 42 | Seed 7 |
|---|---|---|
| Hamming k-NN (k=15) | 56.81% | 49.64% |
| Rules floor: OR of CRM products implying the item ≥ 90% of the time | 67.94% | 74.15% |
| Logistic regression, one per billing item | 95.97% | 94.61% |
| **LightGBM, one per billing item (final)** | **96.79%** | **96.30%** |

A linear model gets within 1–2 points, which supports "billing is close to additive per
product"; the trees add the rare-product interactions. A multi-output neural net was not
built: the regularised classifier chain tied plain per-label models (Sprint 2), so shared
label structure is not where the errors are.

### 9.5 What the model cannot vouch for, segments, split realism

- Billing items with validation F1 < 0.5: 314 of 731 columns but only **0.34% of billed
  items** and **2.4% of customers**; those customers' EMR is 5.8% vs **99.0%** for everyone
  else. The review's "~40% of billing items" counted columns, not items. "Diffuse" SHAP
  columns are not the right measure: they include common bundle items explained by several
  correlated packs.
- Segments: product mix drives errors, status and type barely do. Customers with 1–2 rare
  products (7%) hold 76% of wrong customers; EMR falls to 88% at 16–20 products and 58% at 21+.
- Nearest training configuration (Hamming): distance 1 for 70% of validation vs 53% of test
  rows; test has more rows at distance 2–3, validation more at 4+ (mostly suspect rows).
  The grouped split mirrors test reasonably, slightly optimistically for the near neighbours.

### 9.6 EMR harshness

Validation bit error 1.6e-4 → 0.12 wrong bits per customer; independent errors would give EMR
88.9%, actual 96.8%. Test: bit error 5.7e-5 → independent 95.9%, actual 97.2%. About 2% of
customers carry 91% of the wrong bits: errors cluster in rare-product customers.

### 9.7 Business use — error detection

On the validation fold, known provisioning errors are customers of repeated configurations
whose actual billing differs from consensus (seed 42: 221, 0.66% of clean-like customers).
A customer is flagged if any actual billing item disagrees with the prediction (about 4% of
clean-like customers; 92% of known errors caught, precision ≥ 16%).

**Ranking by model confidence works badly.** The model's most confident disagreements are its
own mistakes on rare-product customers, so the top 2% catches only 19%. Flagged customers are
instead queued with two rules taken from earlier findings, not tuned here: rare-product
customers last (they hold 76% of model errors, 9.5), then fewer disagreeing items first (real
provisioning errors are 1–2 items, Sprint 1).

| Review the top | Seed 42: caught / precision / lift | Seed 7: caught / precision / lift |
|---|---|---|
| 1% of customers | 73% / 49% / 73× | 59% / 38% / 59× |
| 2% of customers | **92% / 31% / 46×** | **83% / 27% / 42×** |
| all flagged (~4%) | 92% / 16% / 25× | 90% / 14% / 22× |

Precision is a lower bound: errors in configurations seen once cannot be confirmed. 97.6% of
suspected-corrupted CRM records are also flagged; they form a separate CRM data-quality queue.
The notebook adds the review curve and a production design (two queues, cost trade-off,
thresholds recomputed on the live CRM base, monitoring, retraining).

### 9.8 Notebook and documents

`notebooks/sprint3_final.ipynb` gained:
- a glossary;
- the injection crosstab and partner evidence;
- EMR harshness;
- configuration-level uncertainty;
- baseline, paired-test and 5-fold tables;
- segments and coverage;
- the business section;
- corrected SHAP wording;
- the test CI;
- a **live** ablation of the corruption filter: the same final model trained without dropping the
  suspected-corrupted rows scores **94.67%** on test, against **97.21%** with the filter
  (**+2.54 points, 2,467 customers**).

It still reads only the professor's three CSVs. Also updated: `requirements.txt` pinned,
README (setup for all OSes, glossary, comparable baselines), the Sprint 2 log (forward note),
and a new executive summary (`docs/EXECUTIVE_SUMMARY.md`). The Sprint 2 notebook was executed
end to end.
