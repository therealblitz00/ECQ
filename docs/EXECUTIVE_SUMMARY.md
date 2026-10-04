# Executive summary — CRM → Billing configuration prediction

Case Study I, Moodle section 8608. Final notebook: `notebooks/sprint3_final.ipynb`.

## The problem

A telecom subscription is recorded in several systems that drift out of sync. From a
customer's **CRM** record (745 yes/no product flags) we predict the **billing** record it
*should* have (731 yes/no billing items), to catch provisioning errors. The competition
metric, **exact match (EMR)**, only counts a customer as correct if all 731 items are right.

## Result

| | Test EMR |
|---|---|
| Sprint 2 model | 95.04% |
| **Final model** | **97.21%** (95% CI 97.11–97.31; 94,391 of 97,100 customers exactly right) |
| Public benchmark | ~98% (about 770 customers more) |

Per billing item the model is 99.994% accurate. EMR is lower because one wrong item fails
the whole customer, and errors cluster in a few customers with rare products.

## What we found

1. **The test set needs generalisation, not lookup.** Only 6 of 97,100 test customers have
   a CRM record identical to one in training. Validation therefore keeps identical records
   on the same side of the split, and scores against each record's most frequent billing
   (to avoid rewarding the model for copying provisioning errors).
2. **Billing is close to "one product → its billing items".** A tuned linear model (logistic
   regression) reaches ~96.2–96.6%; one LightGBM per billing item is slightly better on both
   validation splits (+0.16 to +0.23 points, paired test; borderline on one split). Copying similar customers (52%) and choosing among
   billing combinations seen before (capped at ~23%) do not work.
3. **The training data contains suspected corruption.** About 7–8% of training rows show
   impossible categories (a customer with two statuses at once, a status literally called
   `0`) and bursts of rare products injected almost uniformly at random on both sides. The
   test set has almost none. Removing these rows from training (not from scoring) is worth
   **+2.5 test points** (same model without the filter: 94.67%), confirmed by paired tests on
   two splits.
4. **Sprint 1's dimensionality reduction hurt.** PCA/SVD features cost ~33 points; the model
   uses the raw product flags.
5. **The model learned the real product mapping (SHAP).** For the 233 billing items with a
   clear CRM cause, the model's main driver matches it ~96% of the time. Errors concentrate on
   items with no clear CRM cause, and the model does not rely on the injected noise.

## Business use: catching provisioning errors

Compare the billing the model predicts with what the billing system actually contains.
Flag customers where they disagree, and queue unusual customers (rare products, many
disagreeing items) last, because that is where the model itself is weakest. (This rule was
chosen after ranking by model confidence failed on one split, so the *other* split is the
honest check and is quoted first.)
- On known provisioning errors, reviewing the **top 2% of customers catches 83%**
  (95% CI 73–92%) on the honest split, 92% on the design split: about **40× better than
  random**. Precision is ~27–31%, a lower bound.
- On **synthetic errors** (1–3 wrong items injected into clean customers) the same 2% catches
  ~92%, whatever the size of the error, but only **~80% for customers with a unique
  configuration**, which is the realistic production case.
- With **assumed** costs (€2.5 per alert, €20–200 per missed error), the cheapest review share
  is about **1.3–2.3% of customers**.

A second, separate queue flags CRM records that themselves look corrupted, so they can be
corrected.

**In production:**
- a nightly batch compares predicted with actual billing;
- the review budget sets how many alerts analysts see;
- thresholds are recomputed from the live CRM base;
- the model is retrained monthly or when new products launch (unknown products go to manual
  review);
- monitoring covers alert volume, confirmed-error rate and product-mix drift.

## How the decisions were made

- Every choice is validated on held-out data split by CRM record, and confirmed on a second
  split and by 5-fold cross-validation (final model: 96.83% ± 0.20 on clean-like rows).
- Comparisons use **paired** tests that resample whole CRM records, because customers with
  identical records are not independent.
- A change is adopted only if it is real on both splits and worth ≈ 0.1 point or more. That
  bar was set before testing. A lower threshold for rare items was real but worth only +0.03,
  so it was not adopted.
- `solution.csv` (the test answers) was never used to train or choose anything; it only
  scores frozen models.

## Limitations

- About **2% of customers** have billing items the model predicts poorly (items with no clear
  CRM cause, or that differ even between identical customers). For the other ~98%, the model
  is right ~99% of the time.
- The corruption is *suspected* from strong circumstantial evidence, not proven. The filter
  was designed by comparing training and test *inputs* (no test answers).
- Validation uncertainty is wide (about ±0.7 points on one split) because a few very
  frequent CRM records weigh heavily; the test interval (±0.10) is the reliable one.
- The ~98% benchmark is not reached; the remaining gap is mostly customers with 1–2 rare
  products, where the right cause is found but training evidence is thin.

## Where to find more

| | |
|---|---|
| Final notebook (Sprints 1–3, runs on the professor's CSVs only) | `notebooks/sprint3_final.ipynb` |
| Sprint 3 audit trail, design choices, threats to validity | `docs/sprint3/SPRINT3_DIAGNOSTIC_LOG.md` |
| Response to the mock jury review | `docs/sprint3/REVIEW_RESPONSE_AND_ROADMAP.md` |
| Every experiment | `docs/EXPERIMENT_LEADERBOARD.md` |
