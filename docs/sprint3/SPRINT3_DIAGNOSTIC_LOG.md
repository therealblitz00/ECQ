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
| **Step 8 — final model: Step 4 without the Sprint 1 additive rules** (`notebooks/03_sprint3_final.ipynb`, professor's CSVs only) | **96.79% / 96.30%** | **97.21%** |
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
| 6 | 2026-10-03 | `explain_shap.py` | SHAP top driver = independent co-occurrence driver for 95.7% of clearly-linked columns. Concentration on one driver predicts F1 (Spearman +0.86): errors are on items with no clear CRM cause. No reliance on injected rare packs. Rare-product misses are "right cause, p ≈ 0.2". | 96.81% clean-like (frozen) | Final notebook built (`notebooks/03_sprint3_final.ipynb`) |
| 7 | 2026-10-04 | `exp11_count_features.py` | Teammate's MCA lead (+0.31 in Sprint 2 set-up) tested as plain basket-size counts on the current champion: +0.03 / −0.05 mean, within noise. Not adopted. | 96.81% clean-like (unchanged) | Champion unchanged |
| 8 | 2026-10-04 | `notebooks/03_sprint3_final.ipynb` | Final notebook on the professor's CSVs only. Sprint 1 additive rules dropped (+0.02 on validation, within noise; mined on all of train). Test 97.21% (97.27% with rules, reported not used). | 96.79% clean-like | Final model |
| 9 | 2026-10-04 | `review_checks.py`, `exp12_review_models.py` | Mock jury review answered: configuration-level uncertainty, paired tests, 5-fold CV (96.83% ± 0.20), baselines (LR 95.97%), error detection (83–92% of known errors by reviewing 2% of customers, ~40× random), test CI [97.11, 97.31]. Rare threshold real but +0.03 — not adopted. | 96.79% clean-like (unchanged) | Final model unchanged |
| 10 | 2026-10-04 | `review_checks.py detect2`, `exp12_review_models.py lr` | Second review (80/100) answered: stale text fixed; detection stress-tested (seed 7 honest split 83% [73–92%]; synthetic 1–3-item errors ~92%, unique configurations ~80%); assumed-cost operating point 1.3–2.3%; tuned logistic regression within 0.15–0.23 pt of LightGBM. | 96.79% clean-like (unchanged) | Final model unchanged |
| 11 | 2026-10-05 | `notebooks/03_sprint3_final.ipynb` | Pre-submission check: all 23 code cells ran in order, no errors, printed numbers match the text. Assumed-cost (EUR) operating point removed from the notebook and executive summary at the team's request: the costs were invented, so it adds nothing academically. | 96.79% clean-like (unchanged) | Final model unchanged; notebook ready to submit |
| 12 | 2026-10-05 | `exp13_macro_f1.py` | Macro F1 0.535 / 0.520 is driven by ~280 items never predicted; their positives sit in rows with rare CRM packs (91–92% vs 7–8%) and no CRM column predicts them (residue of the injection). Thresholds tuned for F1 on OOF: macro F1 +0.008 to +0.012 but EMR −0.10 to −0.90 → rejected. Error-minimising lower-only thresholds (`thr-net`): macro F1 +0.0021 / +0.0009 (CI > 0), EMR +0.003 / +0.023 pt → passes the pre-set rule, tiny. | 96.79% / 96.30% clean-like (unchanged) | Awaiting team-lead decision |
| 13 | 2026-10-05 | `exp14_longtail.py`, `exp14_groups.py`, `exp15_kfold_longtail.py` | Long-tail labels investigated structurally: label table + classification; Stage-2 cascade, label communities, CRM-pattern rules, targeted chain. Winner: class-weighted CRM specialists for the ~330 problem labels, add-only, only on rows with a rare CRM pack: macro F1 +0.0135 / +0.0145, EMR +0.12 / +0.19 pt (both significant). Context features (Stage-1 probs, communities) hurt; rule features tie; rule override negligible. | 96.92% / 96.49% clean-like (candidate); 5-fold 96.99% ± 0.18 vs 96.83% | Passes the adoption rule; awaiting team-lead decision |
| 14 | 2026-10-05 | `notebooks/03_sprint3_final.ipynb` | Stage 2 (long-tail specialists, rare gate) added to the final notebook after team-lead approval; executed end to end (~25 min). Validation reproduces Step 13 exactly. **Test 97.21% → 97.51%** (+290 customers); test macro F1 0.70 → 0.77. | 96.92% clean-like (seed 42) | Final model |
| 15 | 2026-10-06 | `notebooks/03_sprint3_final.ipynb` | Presentation only: markdown rewritten into 12 numbered chapters + 3 appendices (results, threats, glossary) with a linked table of contents; audit-level tables moved to Appendix A. Fixed a misplaced Business heading introduced in Step 14. Code cells, outputs and execution counts unchanged (verified). | unchanged | Submission-ready |
| 16 | 2026-10-06 | repository | Folder and file names tidied (notebooks numbered 01–03, brief moved to `docs/`, doc names made consistent), README / architecture / instructions updated, teammate's Sprint 1 re-run committed (same content, new row order). No code changed; all scripts compile, import and run. | unchanged | Repository ready for submission |
| 17 | 2026-10-07 | `diag4_unbilled_crm.py` | Professor's hint: employees added CRM packs without the matching billing. Model-free check on 293 billable CRM packs: 16,237 customers (8.7% of train) carry ≥ 1 CRM pack whose billing items are all missing; 1,333 with billing otherwise unchanged. Flag count per pack is near-constant (179 ± 32) whatever the pack size (41–48,939 rows) → the "corrupted rows" dropped in Step 3 are these customers (12,135 of 13,131 dropped rows flagged), plus 4,102 kept rows. | unchanged | Reported; next step pending |
| 18 | 2026-10-07 | `diag5_counterfactual.py` | Model-based check (Stage 1, 5-fold out-of-fold; each active CRM pack switched 1 → 0): 12,249 customers have a CRM pack whose removal explains missing billing without breaking anything; 70.5% of Step 17's customers confirmed, 11,450 found by both. Synthetic added packs that change the expected billing are found 99.4% of the time, but only 10% of uniformly random packs change it. Same high-rate packs as Step 17 (7 of top 15); no single targeted pack (`CRM_1553_PACK` outlier = Barring/Suspend status rule). | unchanged (OOF 95.18% vs raw billing on kept rows) | Reported; next step pending |
| 19 | 2026-10-07 | `diag6_crm_only_detector.py` | CRM-only detector (LightGBM on CRM, labels = Steps 17–18 agreement) so test can be scored without billing. Train out-of-fold AUC 0.995 / AP 0.977 (rare-pack count alone: AP 0.930); kept rows AP 0.82 vs 0.41; kept rows with no rare pack AP 0.05 (undetectable from CRM). Test: 77 of 97,100 customers flagged (0.08%) vs 8.27% of train → test looks clean, as the brief says. | unchanged | Reported; next step pending |
| 20 | 2026-10-07 | `fraud_report.py` | Steps 17–19 consolidated (English): 17,036 suspected train customers (9.1%): high 11,450 (both methods), medium 3,359, low 2,227 (one method, non-Active status). Column table by rate and count: `CRM_1621_PACK` 70% of its customers; counts flat (~150–180 for the most affected packs). 77 test customers flagged CRM-only. Write-up: `docs/sprint3/FRAUD_DETECTION.md`. | unchanged | Deliverable ready; notebook chapter pending approval |
| 21 | 2026-10-09 | `notebooks/03_sprint3_final.ipynb` | Final notebook restructured at the team's request: one notebook showing only the final approach (11 chapters, appendices removed, all in English). Fraud detection added as Chapter 10 (data rule + CRM-only detector). Tuning kept as reported tables in Chapter 5. Writes three CSVs: `submission.csv`, `train_flags.csv`, `fraud_columns.csv`. Executed end to end without errors (66 min on this run). Test EMR unchanged at 97.51%. | 96.92% clean-like (final model) | Submission-ready |

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

Approved 2026-10-04. Notebook: `notebooks/03_sprint3_final.ipynb` (Sprints 1–3, ~10 minutes).

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
| LightGBM vs logistic regression (untuned, C=1; see 10.3 for tuned) | +0.83 [+0.55, +1.24] | +1.69 [+0.76, +3.11] | trees better than *untuned* LR |

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
product"; the trees add the rare-product interactions. *(Correction in Step 10.3: this
logistic regression was untuned; tuned (C = 10) it gets within 0.15–0.23 points.)* A multi-output neural net was not
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

`notebooks/03_sprint3_final.ipynb` gained:
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

---

## Step 10 — Response to the second mock review (64 → 80/100)

Approved 2026-10-04. The reviewer accepted all three pushbacks from Step 9 (`BIL_4167`, the
"40% of billing items", and that the true SE is larger than it estimated). Remaining
criticisms, and what was done. Scripts: `scripts/sprint3/review_checks.py detect2`,
`scripts/sprint3/exp12_review_models.py lr <seed> <C>`.

### 10.1 Stale text

The notebook still carried pre-Step-9 reasoning. Fixed:
- "within one standard error, ~0.1 points" and "within noise" are replaced by the paired-test
  results.
- `solution.csv` is now described as read only at the end, after freezing, for scoring
  **and** for the filter ablation.
- "+2 points" vs "+2.54" is reconciled: +2.0 is the Sprint 2 → Step 3 milestone (older
  settings), +2.54 the final model with vs without the filter.

### 10.2 Error detection, stress-tested

- **How the queue rule was chosen, disclosed.** Confidence ranking was tried first on seed 42
  and failed, so seed 42 is the design split and **seed 7 the honest one; it is now quoted
  first**: 83% of known errors in the top 2% (95% configuration-bootstrap CI 73–92%);
  seed 42: 92% [86–96%].
- **Synthetic errors** (the known errors are only 1–2-item deviations in repeated
  configurations). Clean customers' actual billing was corrupted with 1, 2 or 3 errors, either
  a missing billed item or a plausible extra item drawn by prevalence; 0.65% of customers,
  5 repetitions, both seeds:

| | Seed 42 | Seed 7 |
|---|---|---|
| Injected errors flagged at all | ~100% | ~100% |
| Caught within a 2% review budget (all, 1–3 items, either kind) | 92–94% | 92–93% |
| … customers with a **unique** configuration (every test customer) | 79–82% | 77–81% |
| … customers with a repeated configuration | 97–99% | 97–98% |

  The queue is **not** limited to 1–2-item errors: recall is flat across 1–3 items. Its real
  weak spot is unique-configuration customers (~80%), which is the realistic production case.
- **Operating point from assumed costs** (€2.5 per alert = 5 analyst-minutes at €30/h;
  €20–200 per missed error): the cheapest review share is 1.3–2.3% of customers on both seeds.
  Real costs belong to the business; the method shows how to set the cut-off.

### 10.3 Logistic regression, tuned

Sweep of C on both seeds (clean-like):

| C | 0.1 | 1 | **10** | 100 |
|---|---|---|---|---|
| Seed 42 | 91.71% | 95.97% | **96.56%** | 96.55% |
| Seed 7 | 89.57% | 94.61% | **96.15%** | 96.01% |

Paired test, LightGBM vs tuned logistic regression (C = 10): **+0.23 [+0.14, +0.34]** on
seed 42, **+0.16 [−0.04, +0.33]** on seed 7 (borderline).

The earlier "+0.8 / +1.7" was against an untuned baseline, so the review was right to ask. A
tuned linear model gets within 0.15–0.23 points, which *strengthens* the additivity finding.
LightGBM is kept: it is better on both splits, but the margin is small, and a linear model
would be a reasonable choice if interpretability mattered more.

### 10.4 Validation vs test, stated plainly

- Validation customers are closer to a training customer than test customers are (70% vs 53%
  at distance 1): validation is *easier* on that.
- Validation keeps more unusual customers: it is *harder* on that.
- The filter is worth +0.8 / +1.25 on validation but +2.0 to +2.5 on test.

The biases partly cancel. The notebook now says validation is used to **rank** choices
(paired tests, two splits, 5-fold) and the test set to **measure** the result.

### 10.5 Not done, by decision

- The rare-item threshold stays unadopted (real, but +0.03: below the bar set before testing).
- No structural constraints were added to the model (argmax status repair was worth +0.01 in
  Step 1).
- Course AI-use rules: for the team to check before submitting. Step 9 and the
  review-response file disclose the AI review openly.

### 10.6 "If we had one more week"

The remaining gap to ~98% (about 767 customers) is mostly customers with 1–2 rare products,
where SHAP shows the right cause but too little training evidence (p ≈ 0.2–0.4). The next try
would pool evidence across rare products that bill the same item (the many-to-one mapping in
Step 1), for example a dedicated sub-model or shared parameters for rare-product → billing
links, judged by the same adoption rule.

---

## Step 11 — Pre-submission check; assumed-cost section removed

2026-10-05. **Plan:** read `notebooks/03_sprint3_final.ipynb` end to end before submission and
remove the EUR operating point (Step 10), which the team judged unsuitable for an academic
deliverable.

**Check.** All 23 code cells are saved with outputs, executed in order (1–23) with no
errors. The numbers in the markdown match the printed outputs (clean-like validation 96.79%,
test 97.21% [97.11, 97.31], filter ablation +2.54 pt, detection 92% [86–96%] on seed 42).
The notebook reads only `train.csv`, `test.csv` and `solution.csv` (the last only after the
model is frozen) and imports nothing from `src/` or `scripts/`.

**Change.** Removed the cost code, its output table and the two cost bullets ("Operating
point", "Cost trade-off"); "Three checks" became "Two checks". The removed code drew no
random numbers and nothing downstream used it, so every other saved output is unchanged and
no re-run was needed. One range was corrected: unique-configuration customers are caught at
about **76–82%** (the notebook's own seed-42 table shows 76–81%; the scripts gave 79–82% and
77–81%), previously written as 77–82%. The same cost bullet was removed from
`docs/EXECUTIVE_SUMMARY.md`.

**Decision.** Model and results unchanged. The review budget alone (top 2% of customers)
now sets the operating point.

---

## Step 12 — Macro F1: diagnosis, per-item thresholds, class weights

2026-10-05. Approved by the team lead as an autonomous loop: Step 1 (diagnose), then
per-item thresholds, then class weights; stop at the first success, or after 5 failed
configurations. Script: `scripts/sprint3/exp13_macro_f1.py`; every run is logged to
`data/cache/results.jsonl` as `E13-*`.

**Why.** Clean-like validation macro F1 of the final model is 0.535 (micro F1 0.996). Macro F1
weights each of the ~726 scored billing items equally, and misses outnumber false alarms
about 6 to 1. 226 items have only 1–5 validation positives.

**Set-up, fixed before any candidate was run.**
- Baseline: the final model (min_child_samples=5, reg_lambda=1, threshold 0.5, no additive
  rules) refitted on the cleaned training fold. The cache was rebuilt in this session, so the
  baseline is refitted rather than loaded.
- Thresholds are **never tuned on validation**. Each fit also produces 5-fold out-of-fold
  probabilities on the training fold's unique configurations (grouped by construction),
  weighted by each configuration's row count. Each item's threshold maximises its OOF F1.
- Metric: macro F1 over items with at least one clean-like validation positive (as in the
  notebook), against consensus targets.
- **Success rule:** on both seeds, the 95% paired configuration-bootstrap CI (1,000 draws) of
  the macro-F1 difference is above 0, **and** the clean-like EMR difference vs the baseline
  is ≥ 0. The Step 4 figures (96.81% / 96.33%) include the Sprint 1 rules, which the final
  model no longer uses, so EMR is compared with the same pipeline without rules.

**Environment.** `.venv` was rebuilt from `requirements.txt` (LightGBM 4.7.0, scikit-learn
1.9.1) and the cache from the CSVs. The baseline reproduces the notebook exactly on seed 42
(clean-like EMR 96.79%, macro F1 0.5348), and gives 96.30% / 0.5201 on seed 7.

### 12.1 Diagnosis (baseline, clean-like validation)

| | Seed 42 | Seed 7 |
|---|---|---|
| Macro F1 / micro F1 | 0.5348 / 0.9959 | 0.5201 / 0.9952 |
| Missed items (FN) vs false alarms (FP) | 3,378 vs 525 | 3,688 vs 455 |
| Items with F1 = 0 (never predicted / only false alarms) | 272 (265 / 7) | 286 (274 / 12) |
| Share of the macro-F1 shortfall due to F1 = 0 items | 81% | 82% |
| Items with 21–50 positive training configurations: count, mean F1 | 359, 0.15 | 360, 0.14 |
| Share of the shortfall from those items | 90% | 89% |
| F1 = 0 items whose true positives all score < 0.05 | 230 of 272 | 250 of 286 |
| Rows billed for an F1 = 0 item that carry a rare CRM pack (all rows) | 92% (7.1%) | 91% (7.9%) |
| Best P(item \| one CRM column) in training, median: F1 = 0 items vs F1 ≥ 0.5 | 0.12 vs 0.87 | 0.12 vs 0.87 |
| OOF macro F1: threshold 0.5 → best per-item threshold | 0.510 → 0.528 | 0.515 → 0.533 |
| Per-item threshold chosen *on validation* (upper bound, diagnostic only) | 0.566 | 0.543 |

**Reading.** The low macro F1 is not a threshold problem. About 280 billing items are never
predicted. Their positives sit almost only in rows with rare CRM packs, and no CRM column
predicts them, so they look like the residue of the random rare-pack injection (rows with 1–3
rare packs are kept in training and in the clean-like population). The model correctly gives
them near-zero probability. Even thresholds picked with hindsight on validation would reach
only 0.566 / 0.543.

### 12.2 Attempts (thresholds tuned on the training fold's OOF only, row-weighted)

Paired configuration bootstrap, 1,000 draws; Δ = candidate − baseline; EMR in points.

| # | Candidate | Rule | Seed 42: ΔF1 [CI] / ΔEMR [CI] | Seed 7: ΔF1 [CI] / ΔEMR [CI] | Verdict |
|---|---|---|---|---|---|
| 1 | `thr-free` | per item, threshold in 0.05–0.95 maximising OOF F1 (200 lowered, 166 raised) | +0.0111 [+0.0068, +0.0143] / −0.90 [−2.56, −0.05] (−299 customers) | +0.0081 [+0.0040, +0.0107] / −0.20 [−0.87, +0.21] | rejected: EMR drops |
| 2 | `thr-low` | as 1, but thresholds may only go down (0.05–0.5; ~230 lowered) | +0.0115 [+0.0074, +0.0147] / −0.84 [−2.48, +0.01] (−278) | +0.0094 [+0.0058, +0.0118] / −0.10 [−0.74, +0.31] | rejected: EMR drops |
| 3 | `thr-net` | lower-only; per item, the threshold minimising row-weighted OOF wrong bits, adopted only if it removes ≥ 2 wrong rows (48 / 50 items lowered) | +0.0021 [+0.0002, +0.0029] / +0.003 [−0.04, +0.05] (+1) | +0.0009 [+0.0001, +0.0020] / +0.023 [−0.02, +0.08] (+7) | **passes the pre-set rule** |

**Why 1 and 2 fail (seed-42 validation, diagnostic).** Maximising F1 lowers a threshold even
when it adds more false alarms than hits: for an item at F1 = 0, a single hit raises its F1
whatever the cost. Over the lowered items, new hits 131 vs new false alarms 562; one item
alone added 226 false-alarm rows. The OOF predictions anticipated this (new false alarms
correlate 0.85 between OOF and validation). Candidate 3 therefore switched to the
EMR-aligned objective (fewest wrong bits).

**Disclosure.** Candidate 3's rule was designed after looking at candidate 2's seed-42
validation errors. Seed 7 is the honest check, and it passes there too. A planned variant
(`thr-low-reg`, F1 objective with a minimum gain) was replaced by candidate 3 before it was
run. Class weights (Step 4 of the plan) were not run: the loop stops at the first success.

**Size of the effect.** The gain is statistically real but small: +0.002 / +0.001 macro F1,
EMR unchanged (+1 / +7 customers). In absolute terms clean-like EMR is 96.80% / 96.33%. That
is level with the Step 4 figures (96.81% / 96.33%), which included the Sprint 1 rules.

**Decision.** Stopped and reported to the team lead. Not adopted yet. If approved, the next
checks are 5-fold confirmation and then the notebook.

---

## Step 13 — Long-tail labels: is the low macro F1 recoverable from structure?

2026-10-05. Requested by the team lead after Step 12: thresholds are the wrong lever; test
whether the long-tail labels can be recovered through the structure of the CRM → BIL mapping
(cascade, label communities, CRM-pattern rules, targeted chains), without lowering EMR.
Scripts: `scripts/sprint3/exp14_longtail.py` (diagnosis, Stage 2, evaluation),
`exp14_groups.py` (gain by label group), `exp15_kfold_longtail.py` (5-fold). Every run is
logged to `data/cache/results.jsonl` as `E14-*`; the per-label tables are
`data/cache/e14_label_table.csv` / `-s7.csv`.

**Fixed before modelling.**
- Baseline = the Step 12 champion probabilities (threshold 0.5, no rules), the same for
  every comparison.
- Problem labels come from the training fold only: row-weighted OOF F1 < 0.5 (332 labels on
  seed 42, 335 on seed 7).
- Stage-2 features that come from a model are Stage-1 **OOF** probabilities on the training
  fold; rules, communities and neighbour lists are mined on the training fold only.
- Merge = **add-only** (Stage 2 can switch a problem label on where Stage 1 said 0; it never
  removes a Stage-1 positive and never touches other labels). Merge thresholds fixed at 0.5
  and 0.8; validation sweeps are reported, never used to choose.
- Class weights: scale_pos_weight = sqrt(neg/pos), clipped to [1, 10] (fixed in advance).

### 13.1 Label table and classification (training fold for predictors, validation for scores)

| Group (rule applied in order) | Seed 42 labels / val positives / mean val F1 | Seed 7 |
|---|---|---|
| 1 One CRM column bills it ≥ 50% (≥ 3 positive configs) | 14 / 141 / 0.41 | 21 / 134 / 0.39 |
| 2 A CRM combination (≤ 3 items) bills it ≥ 50% | 105 / 711 / 0.25 | 105 / 726 / 0.19 |
| 3 A common BIL label implies it ≥ 50% | 0 | 0 |
| 4 Diffuse | 1 / 5 / 0.00 | 1 / 3 / 0.00 |
| 5 Likely injection residue (≥ 80% of positives with a rare pack, no signal above) | 212 / 1,008 / 0.01 | 208 / 1,029 / 0.01 |

- **Combinations transfer.** The training-fold patterns of groups 1–2 fire on validation 44 +
  200 times with **93% / 92% precision** (seed 42). Group-5 patterns fire ~78,000 times with
  0% precision, as expected for noise.
- **No BIL-label structure.** No problem label is implied by a common label, and only 2% share
  a Louvain community (Jaccard ≥ 0.1, 535 communities) with any common label: the long tail
  co-occurs only with itself.
- **Identity test.** Among training configurations with a rare CRM pack, a problem label's
  best rare pack explains 13% of its positives vs 7% when the BIL rows are shuffled (95th
  percentile 7%); well-predicted rare labels reach 38%. So the long tail carries a **weak but
  real** CRM signal on top of noise. Same on both seeds.
- FP-Growth/Apriori libraries are not installed; patterns were grown per label from the CRM
  items frequent among its positive configurations (targeted conditional pattern growth, no
  global rule mining).

### 13.2 Experiments (clean-like validation; Δ vs the same Stage-1 champion; paired configuration bootstrap)

Variants: `crm-cw` = specialist on CRM only with class weights; `ctx` = + all 731 Stage-1
probabilities (cascade); `chain` = + Stage-1 probabilities of the 10 most related common
labels (targeted chain); `comm` = + community activation; `rules` = + the label's pattern
indicator and rare-pack count (`rulesoof` = indicators mined out-of-fold);
`stage1` = no Stage 2, Stage-1 probabilities re-thresholded (ablation); `rule-override` =
deterministic patterns with training precision ≥ 0.9 on ≥ 5 positive configs (32 / 37
patterns). Suffix `-t` = merge threshold; `-grare` = only rows with ≥ 1 rare CRM pack;
`-gdriver` = only rows containing one of the label's CRM predictors (P ≥ 0.3).
Seed-7 runs of `ctx` / `ctx-cw` were stopped: `ctx-cw` had already failed decisively on
seed 42 (fits take 22 min each).

| Variant | Seed 42: Δ macro F1 [CI] | Δ EMR pt [CI] | recovered / broken | Seed 7: Δ macro F1 [CI] | Δ EMR pt [CI] | recovered / broken |
|---|---|---|---|---|---|---|
| `crm-cw-t0.5` | +0.0136 [+0.0091, +0.0175] | -0.078 [-0.37, +0.11] | 63 / 89 | +0.0145 [+0.0085, +0.0173] | +0.097 [-0.07, +0.28] | 82 / 53 |
| `crm-cw-t0.8` | +0.0055 [+0.0028, +0.0078] | -0.087 [-0.39, +0.10] | 33 / 62 | +0.0089 [+0.0043, +0.0110] | +0.146 [+0.03, +0.31] | 55 / 11 |
| `crm-cw-t0.5-grare` | +0.0135 [+0.0088, +0.0175] | +0.123 [+0.07, +0.18] | 49 / 8 | +0.0145 [+0.0084, +0.0171] | +0.186 [+0.08, +0.34] | 66 / 10 |
| `crm-cw-t0.8-grare` | +0.0053 [+0.0025, +0.0076] | +0.057 [+0.03, +0.09] | 21 / 2 | +0.0083 [+0.0039, +0.0105] | +0.133 [+0.04, +0.28] | 42 / 2 |
| `crm-cw-t0.5-gdriver` | +0.0042 [+0.0019, +0.0065] | +0.054 [+0.03, +0.09] | 19 / 1 | +0.0051 [+0.0027, +0.0073] | +0.070 [+0.04, +0.11] | 23 / 2 |
| `crm-cw-t0.8-gdriver` | +0.0027 [+0.0010, +0.0044] | +0.033 [+0.01, +0.06] | 11 / 0 | +0.0023 [+0.0009, +0.0041] | +0.037 [+0.01, +0.07] | 12 / 1 |
| `stage1-t0.3-grare` | +0.0091 [+0.0051, +0.0115] | +0.081 [+0.05, +0.12] | 28 / 1 | +0.0093 [+0.0052, +0.0116] | +0.127 [+0.05, +0.27] | 43 / 5 |
| `stage1-t0.2-grare` | +0.0127 [+0.0079, +0.0156] | +0.105 [+0.06, +0.16] | 41 / 6 | +0.0106 [+0.0061, +0.0134] | -0.027 [-0.43, +0.23] | 50 / 58 |
| `ctx-t0.5` | +0.0003 [+0.0000, +0.0009] | +0.009 [+0.00, +0.02] | 3 / 0 | — | — | — |
| `ctx-t0.8` | +0.0002 [+0.0000, +0.0006] | +0.006 [+0.00, +0.02] | 2 / 0 | — | — | — |
| `ctx-cw-t0.5` | -0.0045 [-0.0053, +0.0005] | -23.126 [-38.23, -9.57] | 10 / 7702 | — | — | — |
| `ctx-cw-t0.8` | -0.0010 [-0.0022, +0.0020] | -5.297 [-9.36, -2.21] | 5 / 1767 | — | — | — |
| `ctx-cw-t0.5-grare` | +0.0001 [-0.0014, +0.0036] | -0.361 [-0.82, -0.14] | 7 / 127 | — | — | — |
| `chain-cw-t0.5` | +0.0074 [+0.0035, +0.0099] | -1.762 [-4.67, -0.11] | 37 / 623 | +0.0051 [+0.0018, +0.0069] | -0.037 [-0.19, +0.14] | 45 / 56 |
| `chain-cw-t0.5-grare` | +0.0073 [+0.0035, +0.0098] | +0.057 [+0.03, +0.10] | 24 / 5 | +0.0048 [+0.0015, +0.0065] | +0.043 [-0.05, +0.19] | 31 / 18 |
| `chain-cw-t0.8-grare` | +0.0038 [+0.0010, +0.0056] | +0.021 [+0.00, +0.04] | 9 / 2 | +0.0033 [+0.0010, +0.0047] | +0.040 [-0.05, +0.18] | 24 / 12 |
| `comm-cw-t0.5` | -0.0096 [-0.0100, -0.0046] | -26.262 [-39.97, -13.54] | 8 / 8743 | -0.0066 [-0.0081, -0.0021] | -10.857 [-16.64, -6.14] | 21 / 3282 |
| `comm-cw-t0.5-grare` | -0.0052 [-0.0060, -0.0021] | -0.631 [-1.05, -0.36] | 5 / 215 | -0.0015 [-0.0037, +0.0015] | -0.796 [-1.46, -0.39] | 8 / 247 |
| `comm-cw-t0.8-grare` | -0.0033 [-0.0043, -0.0009] | -0.418 [-0.76, -0.20] | 3 / 142 | -0.0002 [-0.0021, +0.0022] | -0.702 [-1.36, -0.31] | 6 / 217 |
| `rules-cw-t0.5` | +0.0136 [+0.0083, +0.0173] | -0.159 [-0.54, +0.18] | 100 / 153 | +0.0119 [+0.0069, +0.0149] | +0.043 [-0.12, +0.23] | 81 / 68 |
| `rules-cw-t0.5-grare` | +0.0133 [+0.0081, +0.0168] | +0.207 [+0.06, +0.46] | 83 / 14 | +0.0119 [+0.0070, +0.0150] | +0.153 [+0.05, +0.31] | 66 / 20 |
| `rules-cw-t0.8-grare` | +0.0108 [+0.0066, +0.0134] | +0.105 [+0.07, +0.15] | 37 / 2 | +0.0118 [+0.0068, +0.0140] | +0.166 [+0.07, +0.30] | 54 / 4 |
| `rulesoof-cw-t0.5-grare` | +0.0137 [+0.0086, +0.0172] | +0.222 [+0.08, +0.47] | 84 / 10 | +0.0113 [+0.0064, +0.0142] | +0.160 [+0.06, +0.31] | 64 / 16 |
| `rulesoof-cw-t0.8-grare` | +0.0097 [+0.0057, +0.0123] | +0.096 [+0.06, +0.15] | 34 / 2 | +0.0126 [+0.0074, +0.0150] | +0.180 [+0.08, +0.32] | 58 / 4 |
| `rule-override` | +0.0009 [-0.0000, +0.0016] | +0.009 [+0.00, +0.02] | 3 / 0 | +0.0006 [-0.0001, +0.0018] | +0.013 [+0.00, +0.03] | 4 / 0 |

**Reading.**
1. **Class weights are what lets a specialist recover positives** (`ctx` without weights:
   +0.0003). Without a gate, the extra positives also hit rows with no rare pack and break
   them (EMR on rare-free rows 99.17% → 98.95%).
2. **The rare gate comes straight from 13.1** (> 90% of problem-label positives are on rows
   with a rare pack). With it, rare-free rows are untouched and both metrics rise on both
   seeds. *Disclosure:* the gate was added after seeing seed-42 broken rows; seed 7 is the
   honest check and confirms it (+0.19 pt EMR, +0.0145 macro F1).
3. **Model-output features hurt.** Feeding Stage-1 probabilities (`ctx-cw`, `comm-cw`) makes
   specialists latch onto them; their OOF (train) and full-fit (validation) distributions
   differ, and thousands of rows break. The targeted chain is safer but adds less than CRM
   alone.
4. **Pattern features tie with CRM alone.** `rulesoof-cw` vs `crm-cw` (both gated, 0.5):
   macro F1 +0.0002 [−0.0036, +0.0033] / −0.0032 [−0.0055, −0.0000], EMR +0.10 [−0.03, +0.34] /
   −0.03 [−0.06, +0.00]. Simpler wins.
5. **Specialists beat re-thresholding.** `crm-cw` (gated, 0.5) vs `stage1` at 0.3 (gated):
   macro F1 +0.0044 [+0.0013, +0.0081] / +0.0052 [+0.0013, +0.0076], EMR +0.04 [+0.01, +0.08] /
   +0.06 [+0.01, +0.12]. A fixed low threshold is also fragile (0.2 breaks 58 rows on seed 7).
6. **Where the gain comes from** (`exp14_groups.py`): group 2 (CRM combinations) mean F1
   0.25 → 0.31 / 0.19 → 0.25, 12–18 labels predicted for the first time; group 5 (residue)
   0.006 → 0.025 / 0.005 → 0.021, with ~15 new hits against ~95 new false alarms that land on
   rows already wrong.

**Best combination:** `crm-cw-t0.5-grare`: clean-like EMR 96.92% / 96.49% (champion
96.79% / 96.30%), macro F1 0.548 / 0.535 (0.535 / 0.520), 49 / 66 customers recovered vs 8 / 10
broken, FP 715 / 653 (525 / 455), FN 3,319 / 3,612 (3,378 / 3,688), labels never predicted
247 / 257 (272 / 286).

### 13.3 5-fold confirmation (`exp15_kfold_longtail.py`)

GroupKFold by configuration over all training rows (as in Step 9). In every fold Stage 1,
the problem-label list (from that fold's training OOF) and the specialists are rebuilt from
scratch; scored on the held-out fold's clean-like rows (~34,400 per fold).

| Model | Clean-like EMR (mean ± sd) | Δ EMR per fold (pt) | Macro F1 (mean ± sd) | Δ macro F1 per fold | Recovered / broken (all folds) |
|---|---|---|---|---|---|
| Champion | 96.83% ± 0.18 | — | 0.524 ± 0.005 | — | — |
| **Specialists, rare gate, 0.5** | **96.99% ± 0.18** | +0.15, +0.12, +0.17, +0.15, +0.18 | **0.539 ± 0.005** | +0.014, +0.016, +0.017, +0.013, +0.014 | 346 / 82 |
| Ablation: Stage 1 at 0.3, rare gate | 96.93% ± 0.17 | +0.10, +0.08, +0.09, +0.13, +0.07 | 0.532 ± 0.004 | +0.009, +0.007, +0.010, +0.007, +0.008 | 181 / 17 |

The champion reproduces Step 9 (96.83% ± 0.20). The specialist improves both metrics in all
five folds; the simpler ablation is consistently weaker on both.

### 13.4 Decision

| Approach | Verdict |
|---|---|
| 1 Two-stage cascade, CRM-only class-weighted specialists, add-only, rare gate | **ADOPT** (pending team-lead approval): +0.15 pt EMR and +0.015 macro F1, significant on both seeds and positive in all 5 folds; no test labels; one explainable gate derived from the training-fold diagnosis |
| 1b Cascade with all Stage-1 probabilities as features | REJECT: no gain without weights; with weights the stacked probabilities break hundreds to thousands of rows |
| 2 Label communities | REJECT / insufficient signal: long-tail labels share no community with common labels; community features hurt EMR on both seeds |
| 3 CRM-pattern rules as features | REJECT (ties with the simpler CRM-only specialist, paired test) |
| 3b Deterministic rule override (precision ≥ 0.9) | REJECT: real but negligible (+0.001 macro F1, 3–4 customers) |
| 4 Targeted classifier chain | REJECT: weaker than CRM-only; EMR not significant on seed 7 |
| Hybrid | Not built: no context component added signal on top of CRM-only specialists |

**What this says about the long tail.** It is not entirely irreducible. About a third of the
problem labels (groups 1–2) have a real, transferable CRM signal (patterns with 92–93%
validation precision) that the per-label model under-uses because each has only ~20–50
positive configurations; class weights recover part of it. The remaining ~210 labels look like
injection residue: their best rare pack explains 13% of positives vs 7% by chance, and the
specialist gains little there. Macro F1 therefore stays low (0.54) for a structural reason that
the notebook can explain.

**Notebook (not changed yet).** If approved: add Stage 2 after the final fit; the problem-label
list and the specialists are derived from `train.csv` only (OOF inside the training data); add a
short section with the label table summary, the approaches tried, and this 5-fold table. The
`solution.csv` score stays a post-selection diagnostic.

---

## Step 14 — Stage 2 in the final notebook

2026-10-05. Approved by the team lead. The notebook stays self-contained (no imports from
`src/` or `scripts/`; reads only `train.csv`, `test.csv`, `solution.csv`).

**Changes.**
- New section after the Stage-1 error analysis: diagnosis and approaches table (reported from
  Steps 12–13), Stage 2 fitted live on the seed-42 training fold, paired comparison with
  Stage 1, seed-7 and 5-fold results (reported from `exp14_longtail.py` /
  `exp15_kfold_longtail.py`).
- Final fit: Stage 1, then the long-tail list (5-fold OOF inside all of the filtered
  `train.csv`), the specialists, and the add-only gated merge. The submission uses both stages.
- Test cell: reports Stage 1 alone and the final model. The filter ablation now compares
  Stage 1 with and without the filter, so only the filter differs.
- Header, glossary, storyline, threats to validity and conclusions updated. Business and SHAP
  sections still analyse Stage 1 (stated in the notebook).

**Run** (venv, pinned packages, `jupyter nbconvert --execute`, ~25 min, no errors). All
earlier outputs are unchanged.

| | Stage 1 | Stage 1 + Stage 2 |
|---|---|---|
| Clean-like validation EMR (seed 42) | 96.79% | 96.92% |
| Clean-like validation macro F1 (seed 42) | 0.5348 | 0.5483 |
| Paired difference (seed 42) | | macro F1 +0.0135 [+0.0088, +0.0175], EMR +0.12 pt [+0.07, +0.18]; 49 recovered, 8 broken |
| Long-tail items (all of `train.csv`) | | 332; 787 items added for 655 test customers |
| **Test EMR** (`solution.csv`, after freezing) | 97.21% | **97.51%** [97.41, 97.61], +290 customers |
| Test macro F1 (items with a positive) | 0.701 | 0.769 |
| Gap to the ~98% benchmark | 767 customers | 477 customers |

**Decision.** Final model = Stage 1 + Stage 2. The test score was computed after the design
was frozen and was not used to choose anything.

**Docs.** `README.md` and `docs/EXECUTIVE_SUMMARY.md` updated on 2026-10-06 to the final test
score (97.51%), with Stage 1 (97.21%) kept as a milestone.

---

## Step 15 — Final notebook restructured for submission

2026-10-06. Presentation only. The markdown of `notebooks/03_sprint3_final.ipynb` was rewritten as
a Kaggle-style deliverable: a title section with the result and a linked table of contents,
12 numbered chapters (setup, data, validation, Sprint 2 model choice, Sprint 3 data audit,
Stage 1, Stage 2, business use, explainability, final training, test results, conclusions),
and three appendices (full experiment tables, threats to validity, glossary). Detailed tables
and paired tests moved from the main text to Appendix A.

Also fixed: in Step 14 the Stage 2 cells were inserted after a markdown cell that also held the
start of the Business section, so the Business heading appeared before Stage 2.

All 25 code cells, their outputs and execution counts are byte-identical to the Step 14 run
(checked programmatically), so no re-run was needed.

**Public benchmark removed** (2026-10-06, team-lead request): the benchmark row, the
gap-to-benchmark print, the dashed benchmark line in the milestone chart and the related
sentences were removed from the notebook. Because the chart changed, the notebook was re-run
end to end: no errors, every result identical to the Step 14 run (test EMR 97.51%). Every
number in the markdown was re-checked against the new outputs; one sentence was corrected (one
test customer is not test-like, so "every test customer" became "all but one").

---

## Step 16 — Repository tidy-up

2026-10-06. Requested by the team lead: commit everything, improve folder structure and names
without breaking any code, update the README.

**Renamed (with `git mv`, history kept).**

| Before | After |
|---|---|
| `notebooks/sprint1_preprocessing_v3.ipynb` | `notebooks/01_sprint1_preprocessing.ipynb` |
| `notebooks/sprint2_modeling.ipynb` | `notebooks/02_sprint2_modeling.ipynb` |
| `notebooks/sprint3_final.ipynb` | `notebooks/03_sprint3_final.ipynb` |
| `notebooks/archive/sprint1_preprocessing.ipynb` | `notebooks/archive/sprint1_preprocessing_v1.ipynb` |
| `problem_description.md` (repo root, untracked) | `docs/PROBLEM_DESCRIPTION.md` |
| `docs/sprint1/section11_12_explainer.md` | `docs/sprint1/SECTION11_12_EXPLAINER.md` |
| `docs/sprint2/sprint2_model_candidates.md` | `docs/sprint2/MODEL_CANDIDATES.md` |

Every reference in the Markdown files was updated, including earlier entries of this log
(path text only). **Kept on purpose:** script names (imported by other scripts and by the
Sprint 2 notebook, and cited throughout this log) and `data/derived/` file names (read and
written by code inside the Sprint 1 and 2 notebooks). Notebooks resolve paths relative to
`notebooks/`, so renaming them inside that folder changes nothing.

**Checks.** All of `src/` and `scripts/` compile; every script with a main guard imports;
`exp14_groups.py` and `exp14_longtail.py compare 42` were run and reproduce their Step 13
numbers.

**Teammate's Sprint 1 re-run committed.** The re-executed Sprint 1 notebook and the ten
regenerated `data/derived/` files were compared with the committed versions: same rows and
values, only the row order differs (the additive rules have identical triggers).

**Docs.** `README.md` rewritten (final result, chaptered notebook, layout, how to reproduce
Steps 12–13); `docs/ARCHITECTURE_AND_ROUTING.md` updated (tree, Stage 2, exp13–15);
`CLAUDE.md` updated (paths, Stage 2 as the final model). Note: `scripts/make_leaderboard.py`
rebuilds the leaderboard from the local `data/cache/results.jsonl`; this machine's cache was
rebuilt in Step 12 and holds only the E13–E14 runs, so the leaderboard was not regenerated.

---

## Step 17 — "One CRM pack too many, same billing": the dropped rows as the fraud signal

2026-10-07. **New information from the professor:** the case is a Sport TV-type operator
that suspects employees changed customers' CRM packs so they get a service cheaper or free:
the CRM shows a pack (e.g. a more expensive one) but the billing was not changed. One goal is
to list those customers and the CRM columns where it happens most. The rows dropped as
"noise" in Step 3 may carry exactly this signal.

**Plan.** Model-free diagnostic on `train.csv` only (no test labels, no `solution.csv`):
1. Billing map from the kept rows (test-like, < 4 rare packs): CRM pack *p* is billed as BIL
   item *b* if P(b | p) ≥ 0.8 and lift ≥ 3 (packs with ≥ 30 kept rows). 293 of 495 such
   packs have at least one billing item ("billable").
2. Flag (customer, *p*) when *p* is active and **all** of *p*'s billing items are 0.
3. Reverse direction: "orphan" billing items (billed, but no CRM pack that is billed as them).
4. Confirmation by lookup: the configuration without *p* exists in train and has the same
   billing.

**Results.**

| | Kept rows | Dropped rows (Step 3) | Total |
|---|---|---|---|
| Customers with ≥ 1 unbilled CRM pack | 4,102 (2.4%) | 12,135 (92.4%) | **16,237 (8.7%)** |
| … and no orphan billing item ("CRM extra, billing unchanged") | 751 | 582 | **1,333** |
| … and orphan billing items too ("CRM extra + billing extra") | 3,351 | 11,553 | 14,904 |

- **Uniform across packs.** Each billable CRM pack is unbilled in 179 ± 32 customers
  (range 104–241), whether the pack has 41 or 48,939 customers (quintile means 182, 172, 171,
  174, 196). A real billing exception would scale with how common the pack is; a constant
  count means the same number of customers was altered for every pack. In rate terms the
  "most affected" columns are therefore the least common packs (e.g. `CRM_1790_PACK`,
  `CRM_2174_PACK`, `CRM_2178_PACK`: ~18% of their kept customers, ~85–95% of their dropped
  ones). Full table: `data/derived/unbilled_crm_pack_rates.csv`.
- **Additions, not swaps.** CRM and BIL pack counts per row both grow by ~1.4 per rare pack
  (11.4 / 10.8 with none → 17.6 / 17.6 with four), so packs were added on both sides; the
  added billing items are unrelated to the added CRM packs (Step 2).
- **Lookup confirmation is rare** (28 of 52,474 flags): configurations almost never repeat
  once a pack is removed, so the lookup cannot confirm or refute most flags.
- **Coverage limit.** Only the 293 billable packs can be checked; packs with < 30 clean rows
  or with no consistent billing item cannot be tested this way.

**Interpretation.** The Step 3 "corruption" is mostly the professor's fraud pattern: 92% of
the dropped rows have at least one CRM pack with no billing. Dropping them was right for
predicting the *correct* billing, but they are the answer to the detection question. A
further 4,102 kept customers show the same pattern with fewer added packs, so they stayed
in training.

**Outputs.** `data/derived/unbilled_crm_customers.csv` (one line per customer: MSISDN,
unbilled CRM packs, orphan billing items, pattern, kept/dropped),
`unbilled_crm_suspects.csv` (customer × pack), `unbilled_crm_pack_rates.csv` (per pack).

**Decision.** Reported to the team; next step pending.

---

## Step 18 — Model-based check: switch each CRM pack off and compare with the billing

2026-10-07. Approved after Step 17. Script: `scripts/sprint3/diag5_counterfactual.py`
(~19 min; `quick` argument = 20k-row smoke test).

**Plan.** Step 17 only covers packs with one clear billing item. Here the Stage 1 model
supplies the expected billing of any CRM configuration.
1. Out-of-fold predictions for every train row: 5-fold GroupKFold on the CRM configuration;
   each fold's model is trained on the kept rows (unique configs, consensus labels,
   `min_child_samples=5`, `reg_lambda=1`) of the other folds.
2. For each customer with *missing billing* (item predicted 1, billed 0) and each active CRM
   pack *p*: predict again with *p* = 0. *p* is a **culprit** if this removes at least one
   missing item and creates no new disagreement. All culprits are then removed together; if
   the prediction equals the actual billing exactly → **"billing unchanged" (model-verified)**.
3. Score = highest predicted probability among the missing items a culprit explains.
4. Synthetic check: 5,000 clean rows the model gets exactly right (1,000 per fold), one
   random CRM pack added, billing kept.

**Results.**

| | Kept rows | Dropped rows | Total |
|---|---|---|---|
| Customers with ≥ 1 culprit CRM pack | 2,475 | 9,774 | **12,249 (6.5%)** |
| … billing unchanged once culprits are removed | 290 | 22 | 312 |
| … also flagged in Step 17 | 2,011 | 9,439 | **11,450** |

- Out-of-fold Stage 1 on kept rows: 95.18% exact vs raw billing, 95.76% vs consensus.
- **Agreement with Step 17:** 70.5% of Step 17's 16,237 customers are confirmed. The rest
  mostly carry many added packs, so removing one pack also changes something else, and the
  "no new disagreement" condition fails. 799 customers are new (packs without one clear
  billing item).
- **Synthetic check:** when the added pack changes the expected billing (10.4% of random
  packs), the detector flags the customer 99.4% of the time and names the added pack 99.4%
  of the time; verdict "billing unchanged" 97.7%. When it does not (89.6%), the pack has no
  billing the model has learned (mostly very rare packs), so it cannot be detected from billing
  at all.
- **Columns.** By count, the very common packs lead (`CRM_1529_PACK` 1,507 customers, 0.3% of
  its rows) because they appear in most heavy rows. By rate, the leaders are the same less
  common packs as in Step 17: 7 of the top 15 coincide (`CRM_1667`, `1692`, `1720`, `1776`,
  `1798`, `1842`, `2174`; 4–9% of their kept customers). Full table:
  `data/derived/fraud_model_packs.csv`.
- **Checked and rejected as a target:** `CRM_1553_PACK` holds 122 of the 312 "billing
  unchanged" customers (missing `BIL_3762_PACK`, billed for 99.7% of its customers). Without
  the lift filter, common packs show hundreds to thousands of such misses (e.g.
  `CRM_2087_PACK` → `BIL_3744_PACK`: 2,503), and the 1553 cases are mostly Barring/Suspend
  customers (56% vs 23% among all 1553 customers with no rare pack) → a status-dependent
  billing rule the model misses, not an alteration.

**Interpretation.** Two independent methods (data rule, model counterfactual) agree on
11,450 customers. The altered packs are spread evenly across the catalogue, with no single
favoured pack, so the "most affected columns" are best reported as a rate: the less common
packs, where the same ~180 altered customers are a large share. Detection is limited to
packs that have a billing footprint.

**Outputs.** `data/derived/fraud_model_customers.csv` (per customer: MSISDN, score, culprit
packs, missing billing, billing-unchanged flag, Step 17 overlap), `fraud_model_packs.csv`
(per pack); out-of-fold probabilities in `data/cache/proba_fraud_oof.npy`.

**Decision.** Reported to the team; next step pending.

---

## Step 19 — CRM-only detector: learning on train, scoring test

2026-10-07. Approved. Script: `scripts/sprint3/diag6_crm_only_detector.py` (~4.5 min).
**Disclosure:** uses `test.csv` inputs only (no billing, no `solution.csv`).

**Why.** Fraud is defined by CRM vs billing, and test has no billing. The question is
whether an altered customer can be recognised from the CRM alone, so test can be scored.

**Plan.** Labels from Steps 17–18 on train: positive = flagged by both methods (11,450),
negative = flagged by neither (170,406), flagged by one only = ambiguous (5,586, not used for
training). One LightGBM classifier on the 745 CRM columns + number of rare packs + number of
packs; 5-fold GroupKFold on the CRM configuration; threshold = best F1 out-of-fold. Per
customer, the active CRM pack with the largest positive contribution (`pred_contrib`) is
reported as the suspected pack.

**Results (train, out-of-fold).**

| Rows | Positives | AUC | Average precision | Baseline: # rare packs (AP) |
|---|---|---|---|---|
| All labelled | 11,450 | 0.995 | **0.977** | 0.930 |
| Kept rows (lightly altered) | 2,011 | 0.973 | **0.822** | 0.409 |
| Kept rows with no rare pack | 173 | 0.760 | 0.053 | — |

- Threshold 0.475: precision 0.95, recall 0.94 overall; on kept rows precision 0.86, recall
  0.72; 74% of the ambiguous customers score above it.
- The suspected pack matches a pack named by Steps 17–18 for 70% of positives.
- **Limit:** an added *common* pack leaves no trace in the CRM (AP 0.05). Only unusual packs
  or combinations can be seen without billing.

**Results (test).** 77 of 97,100 customers above the threshold (**0.08%**, vs 8.27% of train
and 1.68% of kept train rows); 99th percentile score 0.085 (train 0.999). This agrees with
the brief: the test pairs are the correct ones. The flagged test customers are mostly
legitimate holders of a rare product that the detector associates with alteration: e.g.
`CRM_2205_PACK` (11 of the 77) is in 223 train rows, only 48 of them kept, so in train it
is mostly seen on altered customers.

**Outputs.** `data/derived/fraud_crm_only_test_scores.csv` (test: MSISDN, score, flag,
suspected pack), `fraud_crm_only_train_oof.csv` (train: label, out-of-fold score, suspected pack).

**Decision.** Reported to the team; next step pending.

---

## Step 20 — Consolidated fraud deliverable

2026-10-07. Approved, to be written in English. Script: `scripts/sprint3/fraud_report.py`
(~10 s; reads the outputs of Steps 17–19). Write-up: `docs/sprint3/FRAUD_DETECTION.md`.

**Plan.** One customer list with confidence levels, one column table, one test list.
- **High:** flagged by both methods (Steps 17 and 18).
- **Medium:** one method, CRM status Active.
- **Low:** one method, status not Active. The billing may depend on the status (Step 18,
  `CRM_1553_PACK`).
- **Altered packs:** the packs named by both methods when they overlap, otherwise the packs
  named by the one method.

**Results.**

| Confidence | Customers | Kept for training | Dropped in Step 3 |
|---|---|---|---|
| High | 11,450 | 2,011 | 9,439 |
| Medium | 3,359 | 1,676 | 1,683 |
| Low | 2,227 | 879 | 1,348 |
| **Total** | **17,036 (9.1%)** | 4,566 | 12,470 |

- High confidence: 5,275 customers with one altered pack, 744 with six or more; 701 with
  billing otherwise exactly unchanged.
- Low-confidence statuses: Barring 690, Block 1-way 513, Block 2-way 280, Suspend 240.
- **Columns:** 723 CRM packs have at least one suspected customer (median 15). By rate:
  `CRM_1621_PACK` 69.7%, `CRM_1773_PACK` 48.8%, `CRM_1816_PACK` 48.2%, `CRM_1738_PACK` 47.1%,
  `CRM_1700_PACK` 46.2%. By count: very common packs lead (`CRM_1529_PACK` 183 of 97,476).
- **Test:** 77 customers flagged by the CRM-only detector (Step 19), not confirmed.

**Outputs.** `data/derived/fraud_customers_train.csv`, `fraud_columns.csv`,
`fraud_test_flagged.csv`.

**Decision.** Deliverable ready. Adding a "Fraud detection" chapter to
`notebooks/03_sprint3_final.ipynb` needs explicit approval (working rules).

---

## Step 21 — One final notebook, best approach only

2026-10-09. Requested by the team lead: deliver a single notebook that shows only the final
approach (no rejected attempts), in English, writing as few CSV files as possible. The tuning
of the final model must stay in.

**Changes to `notebooks/03_sprint3_final.ipynb`**
- **Structure:** 11 chapters — Setup, Data, Validation, Cleaning the training data, The model
  (Stage 1 + Stage 2), Business use, Explainability, Final training and submission, Test
  results, Fraud detection, Conclusions.
- **Removed:**
  - the SVD check and the Sprint 2 model-family comparison;
  - Stage 1 vs Stage 2 "Does it help?" and the no-filter ablation;
  - the milestone chart and the confidence-ranking anecdote;
  - Appendices A–C.
- **Validation metrics** now describe the final model (Stage 1 + Stage 2): clean-like 96.92%,
  macro F1 0.548. 5-fold 96.99% ± 0.20 is stated in the text.
- **Tuning:** a "How the settings were chosen" section in Chapter 5 reports the *k* cut-off
  sweep, `min_child_samples` / `reg_lambda` / capacity, the rare-item threshold and Stage 2.
  These are tables from the earlier scripts, with the paired intervals, and the text says they
  were measured with the Sprint 1 rules on (+0.02).
- **Fraud detection (Chapter 10):** the data rule (Step 17) and the CRM-only detector (Step 19,
  labels = rule flags). The model counterfactual (Step 18) is not in the notebook: it needs
  five extra refits.
- **Files written:**
  - `submission.csv`
  - `train_flags.csv`: 17,233 training rows that were removed from training or are suspected
    of fraud, with the reason, the unbilled CRM products and the missing billing items
  - `fraud_columns.csv`: 293 billable CRM products with their share of suspected customers

**Run.** End to end without errors. This run took 66 min, of which the final Stage 2 refit took
50 min against ~9 min for the same work on the validation fold, so the extra time was the
machine, not the code. Results:
- test EMR 97.51%;
- 16,237 suspected customers;
- CRM-only detector out-of-fold AUC 0.992, AP 0.966;
- 172 test customers (0.18%) above the threshold, against 8.83% of train.

**Note.** The notebook's `fraud_columns.csv` (rule flags, share among all holders) overwrote the
Step 20 file of the same name written by `scripts/sprint3/fraud_report.py`, which uses a
different definition (high + medium confidence).
