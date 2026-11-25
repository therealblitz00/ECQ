# Sprint 2, sections 12 and 13: briefing for the presentation

This note prepares the oral explanation of sections 12 ("Putting the strategies together") and 13 ("What the segment
pipeline tells us") of `notebooks/02_modelling.ipynb`. It covers what we did, why we did it, which paths we tried and
dropped, and the questions we expect. All numbers are on the medium dataset (500 accounts), averaged over the five
validation folds, and can be checked in `data/sprint2/02_pipeline_*.csv`, `02_near_zero_choice_counts.csv` and
`02_decision_log.csv`.

---

## 1. The story in one minute

Sections 4 to 11 each solved one piece of the problem: a strategy for the accounts near zero, a model for the stable
accounts and a model for the volatile ones. Section 12 asks the obvious next question: if we put the pieces together,
do we get a better forecast for the whole bank than the simple baselines? Section 13 reads the answer honestly.

The short answer is yes and no. Inside every segment the pipeline beats the last balance. Over all accounts it cuts the
official MAPE from 736% to 464% (from 320% to 68.6% once the single worst account of each fold is left out). But it is
**not accepted** against the last balance by our own decision rule, because the typical account (median per-account
MAPE) gets slightly worse: 72.4% against 68.1%. Section 13 explains why that happens, and we tried one fix (the
per-account choice) which we measured and rejected.

---

## 2. What the pipeline is (section 12)

Each account is put in a segment at every fold, using only the 365 days before the cut-off (no leakage). Each segment
gets the strategy that won its own comparison earlier in the notebook:

| Segment | Forecast | Where it was chosen |
|---|---|---|
| Near-zero / overdraft (plus the stable accounts flagged in section 9) | Gated calendar profile, with two guards | Section 8 (profile), section 12 (guards) |
| Stable positive | ARIMA chosen automatically per account, no season | Section 11 |
| Volatile positive | Global LightGBM, MAPE-weighted loss, small forecasts set to zero | Sections 10 and 11 |
| Dormant | Last balance | No dormant accounts in the medium data, so not tested |

### The gated calendar profile, in plain words

For each day to forecast we look at what the account had on the same day of the month (plus or minus two days) during
the last year and take a weighted median (weights 1/|balance|, which is the level that minimises the MAPE). Then the
**gate**: we only use that value if at least a share q of those past balances have the same sign. Otherwise we
forecast zero. The idea is that a wrong sign is what costs the most in percentage terms, so we only take the risk where
the account has been consistent. Zero costs at most about 100% per day, which is why it is the safe fallback.

The gate q (5 values: 0.6 to 1.0) and whether to halve the profile (2 values) are tuned **at every fold**, on the two
30-day windows just before the cut-off, using only the near-zero accounts. That is 10 combinations per fold. The test
window is never used for tuning.

### The two guards (and why they exist)

The first version of the pipeline had no guards. In the December fold it scored 2 458% official, because one account
had just lost its whole balance and the profile, built on a year in which it held tens of thousands of euros, kept
forecasting that. The profile only knows the past year, so it cannot see a level shift that has just happened. The
guards fix exactly that:

1. **Recent-range guard:** a non-zero forecast is clipped to the range (min to max) of the last 30 balances.
2. **Level-shift guard:** if the median of the last 7 days is below 0.2× or above 5× the median of the last year, the
   account has just changed level, so we forecast zero.

With the guards, the December fold goes from 2 458% to 76%.

### The per-account choice (the third version)

When we looked at the version with guards, we saw that some near-zero accounts are actually perfectly flat, so their
last balance is almost exact, and forcing the profile or zero on them costs about 100% each. So we added a choice: for
every near-zero account, compare three options (guarded profile, zero, last balance) on the two windows before the
cut-off and keep the one with the lowest MAPE for that account. Again, no test data is used. Ties go to the profile.

Per fold, out of about 229 near-zero accounts, the choice gives: profile to 155 to 174 accounts, zero to 14 to 20, last
balance to 41 to 54.

---

## 3. How many versions we tried, and how we judged them

Three versions of the pipeline are scored side by side, against three references (last balance, median of the last 30
days, zero):

| Version | Official MAPE | Without the largest contributor | Typical account (median) | MAPE on balances of at least €100 | WAPE |
|---|---|---|---|---|---|
| Zero forecast | 98.7% | 98.7% | 99.8% | 99.8% | 100% |
| Last balance | 735.8% | 320.5% | 68.1% | 88.3% | 11.7% |
| Median of last 30 days | 2 075% | 244% | 66.0% | 71.2% | 13.9% |
| Pipeline, no guards, no choice | 939.5% | 72.3% | 75.4% | 57.1% | 34.2% |
| **Pipeline, guards, no choice (kept)** | **463.9%** | **68.6%** | **72.4%** | **55.5%** | **17.3%** |
| Pipeline, guards and choice | 472.4% | 77.2% | 69.0% | 56.3% | 16.8% |

All columns are over all accounts, except the second one, which leaves out the account with the largest error in each
fold.

**The decision rule** (fixed in Sprint 1, never changed after seeing results): a candidate is accepted only if it
improves the official MAPE **and** does not get worse on the two companion measures (median per-account MAPE and MAPE
on balances of at least €100). We apply it with and without the largest contributor.

Verdicts (`02_pipeline_decision_rule.csv`):

- Guards versus no guards: accepted (better on all three measures). Guards stay.
- Pipeline versus zero: accepted, but only without the largest contributor.
- Pipeline versus last balance and versus 30-day median: **not accepted**, because the typical account gets worse.
- Per-account choice versus no choice: **not accepted**. It helps the typical account (72.4% to 69.0%) but makes the
  official MAPE worse in every fold, by up to 14 points (September). Since the competition is scored by the official
  MAPE, we kept the version without it.

Counting the earlier sections, the near-zero problem alone went through: 12 constant-level candidates (section 4), two
two-step models with a sign classifier (section 5), six calendar variants (section 8), three pipeline versions
(section 12). The full list is in section 14, the decision log.

---

## 4. What we learned (section 13)

**Segment by segment the pipeline does its job.**

| Segment | Measure | Last balance | Pipeline |
|---|---|---|---|
| Near-zero | official MAPE | 602% | 96% |
| Near-zero | typical account | 147% | 96% |
| Near-zero | balances of at least €100 | 141% | 87% |
| Stable | typical account | 7.7% | 7.1% |
| Volatile | official MAPE | 243% | 89% |
| Volatile | typical account | 67% | 53% |
| Volatile | balances of at least €100 | 86% | 51% |

**One account can decide a fold.** In the November fold the official MAPE of the pipeline is 2 001%, and 68% without
one account. That account (section 9) held more than €100 000 and fell to a few euros inside the test window, with
nothing in its history that warned about it (its lowest balance of the year was 88% of its median). No method could
have predicted it. That is why we always report both versions.

**Why does the typical account get worse if every segment improves?** Because medians do not add up. The median over
all accounts is around 70%. A near-zero account that the last balance forecast with an error of 30% to 50% now gets a
forecast close to zero, with an error close to 100%, so it crosses the median and pushes it up. The accounts that
improve the most (from several hundred percent to about 100%) were already above the median, so they do not move it.
The per-account choice was our attempt to protect exactly those accounts.

**Why the per-account choice failed.** Two windows of 30 days are not enough to tell an account that is reliably flat
from one that just happened to be flat for two months. When we give the last balance to the wrong account, it costs a
lot on the official MAPE.

**The price in euros.** The pipeline misses about 17% of the euros (WAPE), against 12% for the last balance. Almost
all of the difference comes from the near-zero group, where forecasting zero is cheap in percentage terms and
expensive in euros. This is the cost of optimising the MAPE, and it is the reason we need to ask how zero balances are
treated in the final score.

**What goes to Sprint 3.** A better way to find near-zero accounts that a simple level forecasts well (more windows, or
account features instead of only recent errors); the sudden drops of stable accounts (anomaly work); dormant accounts
(none in medium); the monthly cycle, which the classical models do not include.

---

## 5. Questions we expect, with answers

**Why segments instead of one model for every account?**
Sprint 1 showed that the error comes from very different sources. Near-zero accounts are dominated by the sign and the
division by a tiny balance, stable accounts are almost a constant level, volatile accounts have big monthly cycles.
In section 11 we also saw that no single model wins everywhere: ARIMA wins on stable accounts, LightGBM on volatile
ones, and on near-zero accounts every model is roughly at the level of the zero forecast.

**Why is the near-zero strategy so close to forecasting zero?**
Because with the MAPE, zero is a strong reference: its error is |A| / (|A| + 1), which is always just under 100%.
Any value of the wrong sign, or too far, costs more than 100%. We tested 12 constant levels and none beat zero by more
than one point. The calendar gate is the first thing that beats it in all five folds, and only by 3 to 4 points.

**Isn't forecasting zero cheating?**
It is what the metric rewards, and we show the cost openly: in euros (WAPE) the pipeline is worse than the last
balance. That is why we report the WAPE and why we want to ask how zeros are treated. If the metric changes, the
strategy for that group changes.

**Why tune on the two windows before the cut-off and not on the test window?**
Tuning on the test window would be leakage, and the score would be optimistic. Using the two previous 30-day windows
mimics what we can actually do on the real forecast date.

**Why two windows and not more?**
Two windows balance freshness (level shifts are common: half of the accounts changed level in the last two months,
Sprint 1) and stability. Section 13 shows that two windows are too few for the per-account choice, so using more
windows is the first thing to test in Sprint 3.

**How did you choose the guard thresholds (30 days, 0.2 and 5)?**
Honest answer: they were chosen by reasoning, not tuned. 30 days matches the forecast horizon, and 0.2 / 5 means a
change of level by a factor of five, which is clearly a shift and not noise. The 0.2 matches the warning flag of
section 9, whose threshold we did test (5% to 50%). A sensitivity check of the guard thresholds is a fair to-do for
Sprint 3.

**Why ARIMA without season for stable accounts?**
In section 11 we fitted four models per account (ETS and ARIMA, each with and without weekly season, chosen
automatically by AICc). Without the largest contributor, ARIMA without season was the best for the stable group (7.1%
typical vs 7.7% for the last balance). The weekly season did not help: balances only move on business days, but the
level does not follow a weekly pattern.

**Why is ARIMA worse than the last balance on the official MAPE of the stable group (1 732% vs 1 703%)?**
That number is entirely the €100 000 account of the November fold. Without it ARIMA wins, which is why we chose it on
the version without the largest contributor.

**Why LightGBM for volatile accounts and not ARIMA?**
It won its comparison in sections 10 and 11 against 12 candidates and four classical models. It is one global model
trained on all accounts, so it learns calendar effects (day of month, paydays) that a per-account ARIMA with a 7-day
season cannot. We tried 5 variants of loss and tree size and kept the MAPE-weighted L1 with 15 leaves. We also tried
tsfresh features and rejected them (no gain, ten more minutes per run).

**Why remove the largest contributor? Isn't that hiding bad results?**
We always show both. We remove it only to make the comparison between models fair, because a single unpredictable
account adds about 2 000 points to one fold and decides the average regardless of the model.

**So is the pipeline better or not?**
Better than every baseline on the official MAPE and on large balances, better than zero under our rule, but not
accepted against the last balance because the typical account gets slightly worse (72.4% vs 68.1%) and because in
euros it is worse. We say so in the notebook rather than claim a win.

**How many things did you try in total?**
Section 14 (decision log) lists 12 decisions. For the pipeline itself: three versions (no guards, guards, guards plus
choice). For the near-zero group overall: about 20 forecasts across sections 4, 5 and 8, plus 10 gate settings tuned
per fold.

**Does it scale to the large dataset?**
Yes. All the near-zero logic (profiles, gate, guards, choice) is vectorised over the accounts × days matrix with no
per-account loop. ARIMA and ETS run in parallel over accounts. LightGBM is one global model.

---

## 6. Weak points to admit if asked

- The guard thresholds were not tuned.
- The per-account choice uses only two windows.
- Dormant accounts were never tested (none in medium).
- The pipeline is worse than the last balance in euros.
- The pipeline is not accepted against the last balance on the typical account.
- All results are on medium; the large dataset may behave differently.
