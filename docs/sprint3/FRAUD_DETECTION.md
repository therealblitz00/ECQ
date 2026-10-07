# Fraud detection: CRM packs added without billing

Sprint 3, Steps 17–20 (2026-10-07). Full audit trail: `SPRINT3_DIAGNOSTIC_LOG.md`.

## The question

The operator suspects that employees changed some customers' CRM so that they receive a
service (for example a more expensive pack) cheaper or for free: the CRM shows the pack, but
the billing was never changed. The goal is to list those customers and the CRM columns
where this happens most.

## Main findings

1. **17,036 train customers (9.1%) are suspected; 11,450 (6.1%) with high confidence.**
   For these, two independent methods agree: a CRM pack is active, but the billing it always
   produces is missing.
2. **The "corrupted rows" removed before training are these customers.** 92% of the 13,131
   rows dropped in Step 3 carry at least one unbilled CRM pack. Dropping them was right for
   predicting the *correct* billing, but they are the answer to the detection question.
3. **No single pack was targeted.** Every billable pack is unbilled for about the same
   number of customers (~180), whether 41 or 48,939 customers hold it. The columns where it
   happens most are therefore the less common packs, where those ~180 customers are a large
   share. `CRM_1621_PACK` leads: 70% of its train customers are suspected.
4. **The test set looks clean.** A CRM-only detector trained on train flags 77 of 97,100 test
   customers (0.08%) against 8.3% of train. This matches the brief, which says the test pairs
   are correct.

## How customers were identified

| Step | Method | Uses | Finds |
|---|---|---|---|
| 17 | **Data rule.** On clean rows, learn which billing item(s) each CRM pack produces (P ≥ 0.8, lift ≥ 3). Flag a customer when a pack is active and all its billing items are missing. | train | 16,237 customers |
| 18 | **Model counterfactual.** Stage 1 model, out-of-fold (5-fold, grouped by CRM configuration). For a customer with missing billing, switch each active CRM pack 1 → 0. If the prediction then matches the actual billing without breaking anything, that pack is the culprit. | train | 12,249 customers |
| 19 | **CRM-only detector.** LightGBM on the CRM alone, labels = agreement of Steps 17–18, so customers without billing (test) can be scored. | train → test inputs | 77 test customers |
| 20 | **Consolidation** into confidence levels and a column table. | outputs above | this document |

**Confidence levels** (`data/derived/fraud_customers_train.csv`):

| Level | Rule | Customers | Of which kept for training |
|---|---|---|---|
| High | Found by both methods | **11,450** | 2,011 |
| Medium | One method; CRM status Active | 3,359 | 1,676 |
| Low | One method; status not Active (billing may depend on the status) | 2,227 | 879 |

For each customer the file gives the MSISDN, the altered CRM pack(s), the missing billing
item(s), the model score, the CRM-only score, the CRM status and whether the rest of the
billing is unchanged. For 701 high-confidence customers the billing is exactly what the CRM
would produce without the altered pack: the pure "one CRM pack too many, same billing" case.

## Columns where it happens most

`data/derived/fraud_columns.csv` gives one line per CRM pack: the number of suspected
customers (high + medium), the number of customers holding the pack, the rate and both ranks.

Top packs by rate (packs held by ≥ 100 train customers):

| CRM pack | Suspected customers | Customers with pack | Rate |
|---|---|---|---|
| `CRM_1621_PACK` | 177 | 254 | 69.7% |
| `CRM_1773_PACK` | 160 | 328 | 48.8% |
| `CRM_1816_PACK` | 174 | 361 | 48.2% |
| `CRM_1738_PACK` | 153 | 325 | 47.1% |
| `CRM_1700_PACK` | 139 | 301 | 46.2% |
| `CRM_1765_PACK` | 142 | 308 | 46.1% |
| `CRM_1721_PACK` | 132 | 307 | 43.0% |
| `CRM_2179_PACK` | 116 | 270 | 43.0% |
| `CRM_1663_PACK` | 138 | 322 | 42.9% |
| `CRM_1740_PACK` | 137 | 320 | 42.8% |

By count, the leaders are very common packs (`CRM_1529_PACK`: 183 suspected among 97,476
holders, 0.19%). Counts are flat (most-affected packs ~150–180 customers each), so the rate
is the meaningful ranking.

## Test customers

`data/derived/fraud_test_flagged.csv` lists the 77 test customers above the CRM-only
threshold, with score and suspected pack. Test has no billing, so these are **not confirmed
frauds**. They are mostly legitimate holders of a rare product that, in train, appears mainly
on altered customers (e.g. `CRM_2205_PACK`: 11 of the 77; 223 train rows, only 48 not
altered).

## How reliable is this?

- **Synthetic check (Step 18).** One random pack was added to 5,000 clean customers, with
  billing unchanged. When the added pack should change the billing, the detector finds the
  customer and names the right pack 99.4% of the time.
- **CRM-only detector (Step 19), out-of-fold on train.** AUC 0.995, average precision 0.977.
  On the lightly altered customers it reaches 0.82, against 0.41 for counting rare packs.
- **Clean customers rarely show the pattern.** Among customers with no rare pack, a pack
  without its billing occurs ~1–2 times per pack, against ~180 among altered customers.

## Limitations

- **No ground truth.** There is no list of confirmed frauds, so precision cannot be measured
  directly; `solution.csv` was not used.
- **Only packs with a billing footprint can be detected.** In the synthetic check, 90% of
  randomly added packs do not change the expected billing (mostly very rare packs). Such an
  addition is invisible in the billing. The true number of altered customers is therefore
  probably higher than the list.
- **An added common pack leaves no trace in the CRM alone** (Step 19, average precision 0.05),
  so without billing only unusual packs can be spotted.
- **Status-dependent billing.** Some billing items are dropped for Barring/Suspend/Blocked
  customers (e.g. `CRM_1553_PACK` → `BIL_3762_PACK`). These are kept apart as low confidence.

## Reproduce

```bash
.venv/Scripts/python scripts/sprint3/diag4_unbilled_crm.py        # Step 17, ~30 s
.venv/Scripts/python scripts/sprint3/diag5_counterfactual.py      # Step 18, ~19 min
.venv/Scripts/python scripts/sprint3/diag6_crm_only_detector.py   # Step 19, ~5 min
.venv/Scripts/python scripts/sprint3/fraud_report.py              # Step 20, ~10 s
```
