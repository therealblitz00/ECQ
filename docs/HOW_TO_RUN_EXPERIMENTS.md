# How to test a pre-processing idea

One harness, one protocol. You write a function that transforms the features; everything else —
the split, the targets, the de-duplication, the threshold, the models — is identical for all of
us. That is what makes your number comparable to mine.

If you change anything in the "frozen" list below, your result stops being comparable and the
comparison is worthless. Change your pre-processing function, nothing else.

---

## 1. Setup (once)

You need `train.csv` and `test.csv` in `data/`. They are gitignored — get them from the
competition, they are not in the repo.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Mac/Linux
pip install -r requirements.txt
```

**Your first run takes about 4 extra minutes.** It reads the raw CSVs once and builds
`data/cache/arrays.npz` (~574 MB, gitignored). Every run after that loads in seconds.

---

## 2. Write your idea

Copy `john_experiment.py` and change the function. It receives the training features, the
training labels and the validation features, and returns all three, transformed.

```python
import numpy as np
from testing_pipeline import run_experiment


def my_idea(X_train, Y_train, X_val):
    """One sentence saying what you think will help and why."""
    X_tr = np.copy(X_train)
    Y_tr = np.copy(Y_train)
    X_va = np.copy(X_val)

    # ... your transformation ...

    return X_tr, Y_tr, X_va


run_experiment(
    author="YourName",
    idea_name="Short_Name_For_The_Idea",
    preprocess_func=my_idea,
    models='br-lgbm',     # one model while iterating
    rules='both',
)
```

`X_train` is `(53947, 745)` uint8 — one row per unique CRM configuration, all values 0/1.
`Y_train` is `(53947, 731)` uint8. `X_val` is `(36385, 745)`.

### The two rules you must not break

**Whatever you do to a training column, do to the validation column.** If you drop feature 10
from training but not validation, the model is fed a column at prediction time it never learned
to use. The harness checks the column *counts* match, but it cannot check you dropped the
*same* columns — that one is on you.

**Never add or drop validation rows.** Scoring is against a fixed set of 36,385 validation
rows. Filtering them means you are scoring on an easier set and your number is not comparable.
The harness rejects this outright.

You *may* drop or add training rows — that is a legitimate idea (outlier removal, augmentation).
The harness prints how many you dropped.

---

## 3. Run it

```bash
python my_experiment.py
```

While iterating, pass one model. All three cost about 10 minutes.

| | cost |
|---|---|
| `br-lgbm` | ~210 s |
| `ovr-l1-lr` | ~208 s |
| `cooc-chain-lgbm` | ~190 s |
| `full_metrics=True` | +5 s per rules variant |
| `compute_auc=True` | +10 s per model |
| re-running an unchanged idea | ~15 s (cached) |

**Probabilities are cached by content.** Re-run the same idea and it loads from disk instead of
refitting. Edit your function and the cache key changes automatically, so you can never get a
stale number from a previous idea — but the cache files are 106 MB each, in `data/cache/experiments/`.
Delete that folder if disk gets tight.

---

## 4. Read the output

```
[br-lgbm] no rules
  EMR                 0.87654      EMR vs raw labels 0.87168
  bit accuracy        0.99867      Hamming loss      0.00133
  F1 micro            0.96697      F1 samples        0.97367
  F1 macro            0.37482      median per-label  0.22581
  precision micro     0.99394      recall micro      0.94142
  ROC-AUC micro       0.98572      PR-AUC micro      0.95782
            predicted 1  predicted 0
  actual 1       516562        32142
  actual 0         3151     26045580
```

**EMR is the only metric that counts.** It is the competition metric: a row scores only if all
731 BIL bits are right. The others are diagnostics.

Do not optimise F1 or Hamming loss. They actively disagree with EMR on this problem — applying
the 42 additive rules *raises* EMR by 0.19 points while making F1 micro, precision and Hamming
loss all worse, because it adds 1,358 false positives to fix 73 false negatives on rows that
were one bit from perfect. Optimise F1 and you will throw away a real gain.

`EMR vs raw labels` is EMR against the original training labels rather than the consensus ones.
Quote it when comparing to the test set, since `solution.csv` holds raw labels.

Everything is appended to `experiments_log.csv` (29 columns) so we can compare across all four
of us.

---

## 5. Baselines

Pre-processing = identity, seed 42. **Your idea must beat these or it did not help.**

| model | no rules | with rules |
|---|---|---|
| `ovr-l1-lr` | — | 86.4367 % |
| `cooc-chain-lgbm` | 87.5141 % | 87.7092 % |
| `br-lgbm` | 87.6543 % | **87.8411 %** |

Compare like with like: a `rules='both'` run gives you both columns.

**A difference under ~0.05 points is probably noise.** For scale: the whole Sprint 2 champion
decision turned on 0.022 points, and that was judged not significant (p = 0.096). If your idea
moves EMR by 0.02, you have most likely found nothing. Ask for a significance test before
claiming a win.

---

## 6. What is frozen

| | |
|---|---|
| split | `E.setup(seed=42, val_size=0.2)`, grouped on raw CRM configuration |
| targets | consensus targets from raw `train.csv` — scoring never sees your pre-processing |
| de-duplication | on the **raw** configuration, *before* your function runs |
| threshold | 0.5 |
| additive rules | applied to the **raw** validation CRM matrix, never the transformed one |

Two of these are worth understanding rather than just obeying.

**De-duplication happens before your function.** Every idea therefore starts from the same
53,947 training rows and only the features differ. If we de-duplicated *after* your transform,
a transform that merges configurations would silently change the training set size, and you
could no longer tell "better features" from "different training data". Concretely: the Sprint 1
`s10` feature set merges 358 raw configurations, and 342 of those end up carrying conflicting
labels.

**The additive rules use raw CRM columns.** The 42 rules are defined over original column names,
so they are applied to the untransformed validation matrix. If your idea masks a column that is
a rule trigger, the rule will still fire on the original value.

---

## 7. Mistakes that produce a wrong-but-plausible number

The harness catches the first four and tells you what to fix.

| mistake | what happens |
|---|---|
| dropped validation rows | rejected |
| different feature count in train vs val | rejected |
| filtered X but not Y | rejected |
| removed every training row | rejected |
| **dropped different columns in train vs val** | **not detectable — counts match, meaning does not** |
| mutating the input arrays in place | affects later runs in the same session; always `np.copy` first |
| comparing a rules run against a no-rules baseline | a free 0.19-point "gain" that is not real |
