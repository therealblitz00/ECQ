# Response to the mock jury review, and roadmap

An AI was asked to review the repository as a professor would (score 64/100). This file
checks each criticism against the actual outputs, says whether we agree, and lists the fixes
in priority order. Deadline: Tuesday 06/10/2026.

## 1. Is the review right? (checked against our outputs)

| Review claim | Our check | Verdict |
|---|---|---|
| EMR harshness is never quantified (bit accuracy 99.994% vs EMR 97.2%) | True. Only one sentence in the notebook. | **Agree** — cheap to add |
| No logistic-regression baseline, although we claim billing is "additive per pack" | True. Never run. A linear model is the natural test of that claim. | **Agree** — add it |
| No multi-output neural net | True, but our evidence (Sprint 2 Iteration 7: label dependencies only compensated for overfitting) predicts little gain, at high cost. | **Disagree on priority** — state why, don't build |
| No "rules floor" baseline (OR of CRM packs with P(BIL\|CRM) ≥ 0.9) | True, and cheap. | **Agree** |
| The 52% k-NN figure is on a different population than the headline | True: it is all validation rows, the headline is clean-like / test. | **Agree** — report k-NN on clean-like too |
| Hard constraints (one status per customer) are not enforced | True, but tested in Step 1 (argmax repair +0.01). | **Partly** — mention the test |
| `BIL_4167` is "the top offender, 44 rows all missed" | **Stale.** That was the Step 3 model (`min_child_samples=20`). After Step 4 its F1 is **0.98** (`sprint3_shap_drivers.csv`). | **Disagree** |
| The primary metric ("clean-like") is self-defined using test inputs | True and already disclosed. Mitigation: show the key decisions also hold on all and test-like rows, and on test. | **Agree** — add evidence |
| The SE of ~0.1 ignores clustering (inflation 1.67) | True: rows of one configuration share prediction and target, so the effective sample is configurations. | **Agree** — compute a configuration-level bootstrap SE |
| Comparisons are unpaired; consistent small gains are rejected as noise | True. Paired tests on the same rows are much more sensitive. | **Agree** — run paired bootstrap for the key decisions and report honestly |
| Two splits are thin | Fair. | **Agree** — 5-fold GroupKFold on the final model |
| No error breakdown by customer segment | True. | **Agree** — status / type / basket size |
| No confidence interval on test EMR | True (binomial ±0.10 pt). | **Agree** |
| Business value is missing (error detection) | **The most important point.** Nothing uses the model to flag provisioning errors. | **Agree** — add an error-detection section |
| "299 columns have median F1 0 → the model cannot verify ~40% of billing items" | **Overstated.** 299 is the number of columns in the "diffuse" band, whose *median* F1 is 0; they are mostly rare. Share of columns ≠ share of billed items. | **Partly** — quantify the share of billed items and customers affected |
| SHAP "96–99% agreement" overstated | True: it is 95.7% / 99.1% on the **233** columns with a clear driver, 46% on the rest. | **Agree** — fix wording |
| "+2 points" is a hard-coded milestone in the notebook | True. | **Agree** — recompute a live ablation (final model without the corruption filter) |
| "Each injected CRM pack comes with an unrelated BIL pack" is asserted, not shown | True for the notebook (shown only in `diag2`). | **Agree** — add the crosstab + partner evidence |
| Corruption is a hypothesis, not proven | Fair: it is strong circumstantial evidence. | **Agree** — wording: "suspected" |
| Sprint 2 notebook never executed | True. | **Agree** — run it (needs approval) |
| README points to `notebooks/exports/` (gitignored) | True, minor. | **Agree** |
| Setup is Windows-only; `requirements.txt` unpinned | True. | **Agree** |
| No executive summary; private jargon | True. | **Agree** — 2-page summary + glossary |
| Sprint 2 log says multi-bit failures are "model failures", Sprint 3 says corruption | True — the understanding changed. | **Agree** — forward note in the Sprint 2 log |
| Val vs test nearest-neighbour distance never checked | True, and a good check of the split design. | **Agree** — cheap |

**Overall:** about 80% of the review is right. Its biggest point, business value, is fair.
Its errors are one stale fact (`BIL_4167`), one overstatement (the "40% of billing items"),
and one priority call (a neural net).

## 2. Roadmap — status (2026-10-04)

All items done except the two explicitly not planned. Details and numbers: Sprint 3 log,
Step 9.

| # | Item | Result | Status |
|---|---|---|---|
| 1 | Error-detection evaluation + production design | Reviewing 2% of customers catches 83–92% of known errors (~40× random) on two splits. Ranking by model confidence alone does **not** work (top 2% catches 19%): the model's most confident disagreements are its own mistakes on unusual customers, so they are queued last. | ✅ notebook + log |
| 2 | EMR harshness | Test bit error 5.7e-5: independent errors would give 95.9% EMR, actual 97.2%; ~2% of customers carry 91% of wrong bits | ✅ notebook |
| 3 | Uncertainty and paired tests | Validation SE is 0.34–0.38 pt at configuration level (larger than the review's 0.16). Test 97.21% [97.11, 97.31]. Paired tests confirm the corruption filter (+0.8 / +1.25) and `min_child_samples=5` (+0.27 / +0.20). | ✅ notebook + log |
| 4 | 5-fold GroupKFold | Final model 96.83% ± 0.20 clean-like | ✅ |
| 5 | Baselines on the same metric | Logistic regression 95.97 / 94.61 (trees significantly better: +0.83 / +1.69); rules floor 67.9 / 74.2; k-NN 56.8 / 49.6 | ✅ notebook + log |
| 6 | Errors by segment | Product mix drives errors (rare products: 76% of wrong customers); status and type barely matter | ✅ notebook |
| 7 | What the model cannot vouch for | Weak billing items = 0.34% of billed items, 2.4% of customers; others are 99.0% exact | ✅ notebook |
| 8 | Validation vs test nearest neighbours | Mostly similar; validation slightly closer (70% vs 53% at distance 1) | ✅ log + notebook (threats) |
| 9 | Live ablation + injection evidence | Without the filter 94.67% vs 97.21% (+2.54); rare CRM–BIL partner link 29% in clean rows vs 5% in suspect rows | ✅ notebook |
| 10 | Wording fixes | SHAP restated as 233 items; "suspected" corruption; k-NN on the same metric | ✅ |
| 11 | Execute the Sprint 2 notebook | See commit | ✅ |
| 12 | Pinned requirements, cross-OS setup, glossary, `exports/` mention | Done | ✅ |
| 13 | Executive summary | `docs/EXECUTIVE_SUMMARY.md` | ✅ |
| 14 | Sprint 2 forward note; Step 9 in the Sprint 3 log | Done | ✅ |
| — | Rare-column threshold (real on 7/7 evaluations but +0.03 in 5-fold) | Not adopted: below the ~0.1-point bar set before testing | ❌ by decision |
| — | Multi-output neural net / ensemble of chains | Not built: the regularised chain tied per-label models, so label structure is not where the errors are | ❌ by decision |

## 3. Second review (64 → 80/100) — response and status

The reviewer accepted our three pushbacks (`BIL_4167`, the "40% of billing items", and that
the true SE is larger than it estimated). Its remaining points, checked:

| Point | Verdict | Done |
|---|---|---|
| Stale text contradicts the new rigour ("~0.1 SE", "within noise", "`solution.csv` only in the final cell", "+2" vs "+2.54") | Agree | ✅ rewritten; "+2.0" (milestone) vs "+2.54" (final model ablation) explained |
| Detection evaluated only on easy, narrow errors | Agree — the most important point | ✅ synthetic 1–3-item errors on both seeds: ~92% caught in a 2% budget, flat across error size; **~80% for unique-configuration customers** |
| Queue rule designed on seed 42 → lead with seed 7 | Agree | ✅ disclosed; seed 7 quoted first (83%) |
| No CI on detection recall | Agree | ✅ configuration bootstrap: 83% [73–92%] (seed 7), 92% [86–96%] (seed 42) |
| Cost trade-off is prose only | Partly (costs are unknown) | ✅ illustrative operating point with assumed costs: 1.3–2.3% review share |
| Logistic regression untuned | Agree — and it mattered | ✅ C swept: tuned LR within 0.15–0.23 pt of LightGBM (was quoted as 0.8–1.7) |
| Validation-vs-test relationship undersold | Agree | ✅ stated plainly: validation ranks choices, test measures them |
| Model still enforces no structure | Disagree on value | — tested in Step 1 (+0.01) |
| Check the course's AI-use rules | Team decision | ⚠️ for the team, before submitting |
