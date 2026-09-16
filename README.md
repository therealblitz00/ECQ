<<<<<<< HEAD
<<<<<<< HEAD
# Case Study I — CRM → Billing Configuration Prediction

Team project for the case study competition (Moodle course, section id 8608).

## Problem

Predict the correct **Billing (BIL)** system configuration for a customer given their **CRM**
system configuration. A 4-Play telecom subscription (Internet, TV, mobile, fixed line) gets
provisioned across several systems (CRM, TV platform, Internet platform, Billing). Human and
automatic errors mean these systems can drift out of sync; the goal is to build a model that
predicts the *correct* Billing configuration from the CRM configuration, to help audit and
catch provisioning errors.

- Input: 745 binary CRM columns (`CRM_...`)
- Output: 731 binary Billing columns (`BIL_...`)
- Metric: **Exact Matching Ratio (EMR)** — a prediction only counts as correct if *every one*
  of the 731 labels matches for that row. Partial credit does not exist under this metric.
- Known complications: CRM→BIL is many-to-one in places, the training data contains genuine
  provisioning errors (same CRM config → different BIL config in different rows), and
  individual CRM variables don't necessarily map 1:1 to individual BIL variables.

## Data

Place the following files (from the competition, not included in this scaffold due to size)
into `data/`:

| File | Rows | Cols | Notes |
|---|---|---|---|
| `train.csv` | 187,442 | 1,477 | `MSISDN` + 745 CRM cols + 731 BIL cols |
| `test.csv` | 97,100 | 746 | `MSISDN` + 745 CRM cols |
| `solution.csv` | 97,100 | 2 | `MSISDN`, `Bill_Conf` — the 731 BIL bits concatenated into one string per row. **This looks like the answer key for the test set** — great for local validation of your own EMR score sprint-to-sprint, but double check with the instructor whether it's meant to be used this way before relying on it for anything graded. |
| `sampleSubmission.csv` | — | — | Expected submission format |

All CRM/BIL columns are strictly binary (0/1), so load them as `int8` to keep memory sane —
`train.csv` is ~560MB and will balloon under pandas' default `int64`.

## Team & process

- Work is organized in **Microsoft Planner** by the team leader (activities + assignment to
  members). This repo is where the code/notebooks actually get built.
- 3 sprints, one notebook submitted per sprint on Moodle:

| Sprint | Focus | Weight | Due |
|---|---|---|---|
| 1 | Pre-processing | 40% | **22/09/2026 23:59** |
| 2 | Modeling | 30% | **29/09/2026 23:59** |
| 3 | Optimization & Explainability | 30% | **06/10/2026 23:59** |

- End of each week: team presents completed activities + results for that sprint.

## Project layout

```
project/
├── README.md
├── requirements.txt
├── data/                       # put train.csv / test.csv / solution.csv / sampleSubmission.csv here
├── notebooks/
│   └── sprint1_preprocessing.ipynb   # starter notebook, submit-ready structure for Sprint 1
└── src/                        # shared helper code as the project grows (feature building, metrics, etc.)
```

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/sprint1_preprocessing.ipynb
```

## Sprint roadmap (suggested)

**Sprint 1 — Pre-processing** (`notebooks/sprint1_preprocessing.ipynb`)
- Load data with proper dtypes, sanity-check for missing/duplicate rows
- Quantify sparsity, drop zero-variance columns
- Quantify CRM→BIL ambiguity (provisioning errors) and decide how to handle it
- Quantify output-label cardinality (how many distinct BIL combinations actually occur)
- Export a cleaned train/test set for Sprint 2

**Sprint 2 — Modeling**
- Given 731 correlated binary labels and a large but likely-clustered label space (see Sprint 1
  §4), decide between: 731 independent binary classifiers, a classifier-chain, a
  multi-output tree model (e.g. LightGBM/XGBoost with `MultiOutputClassifier`), or reframing as
  multi-class over the ~N distinct observed label combinations.
- Optimize directly for EMR where possible (it punishes any single wrong bit), not per-label
  accuracy/F1, since those don't align with the competition metric.
- Hold out a validation split and score with EMR before submitting.

**Sprint 3 — Optimization & Explainability**
- Hyperparameter tuning / model selection against EMR on the validation split
- Explainability: e.g. SHAP to show which CRM options drive which BIL predictions — useful
  both for the report and for spotting cases where the model is exploiting quirks in the
  (error-containing) training labels rather than the true CRM→BIL logic.

## A note on the errors in the training data

The brief is explicit that `train.csv` contains provisioning errors baked into the labels
(same CRM config, inconsistent BIL config across rows), but that **test pairs are the ones
deemed correct**. That means blindly minimizing training loss can teach the model to reproduce
noise. Worth treating "which training rows look erroneous" as a first-class Sprint 1
deliverable, not just an EDA footnote — it directly affects Sprint 2's ceiling.
=======
# ECQ
>>>>>>> c0a513d5d0d417d05000b44b11174bbced71e639
=======
# ECQ
>>>>>>> c0a513d5d0d417d05000b44b11174bbced71e639
