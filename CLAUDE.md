# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Team case-study competition project (Moodle course, section 8608): predict a customer's
**Billing (BIL)** system configuration from their **CRM** system configuration. A 4-Play
telecom subscription gets provisioned across several systems (CRM, TV, Internet, Billing);
human/automatic errors let these drift out of sync, so the model's job is to predict the
*correct* BIL configuration and help catch provisioning errors.

- Input: 745 binary `CRM_*` columns. Output: 731 binary `BIL_*` columns.
- Metric: **Exact Matching Ratio (EMR)** — a row only counts as correct if all 731 BIL labels
  match. No partial credit, so per-label accuracy/F1 do not align with what's being optimized.
- `train.csv` contains real provisioning errors: identical CRM config, different BIL config
  across rows. The competition brief says test pairs are the ones considered correct — fitting
  training loss too literally teaches the model to reproduce that noise. Treat "which training
  rows look erroneous" as a first-class deliverable, not just an EDA footnote.
- CRM/BIL don't share a naming/id vocabulary, so this is a real learning problem, not a
  pack-id lookup between the two systems.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/sprint1_preprocessing_v3.ipynb
```

No test suite, lint config, or build step exists in this repo — it's a data-science notebook
project. Work happens in Jupyter notebooks under `notebooks/`, with `src/` reserved for shared
helper code (feature building, metrics) as it's extracted from notebooks.

## Data

`data/*.csv` is gitignored (files are large, not committed — `!data/.gitkeep` keeps the dir).
Get the files from the competition and place them in `data/`:

| File | Rows | Cols | Notes |
|---|---|---|---|
| `train.csv` | 187,442 | 1,477 | `MSISDN` + 745 CRM cols + 731 BIL cols (~560MB) |
| `test.csv` | 97,100 | 746 | `MSISDN` + 745 CRM cols (~148MB) |
| `solution.csv` | 97,100 | 2 | `MSISDN`, `Bill_Conf` (731 BIL bits concatenated per row) — looks like the test-set answer key, useful for local EMR validation, but confirm with the instructor before relying on it for anything graded |
| `sampleSubmission.csv` | — | — | Expected submission format |

All CRM/BIL columns are strictly binary — always load with an explicit `int8`/`uint8` dtype
map (see `notebooks/sprint1_preprocessing_v3.ipynb` cell 4), never let pandas default to `int64`,
or memory use balloons ~8x on files already in the hundreds of MB.

Two column families exist and behave differently, split via `split_taxonomy()` in the sprint-1
notebook (§4):
- One-hot categorical blocks: `BUSINESS_LINE`, `SUBSCRIBER_TYPE`, `SUBSCRIBER_STATUS` — dense,
  mutually exclusive within their group.
- `*_<id>_PACK` flags (hundreds of them) — sparse, largely independent.

## Sprint plan

One notebook submitted per sprint on Moodle; work is tracked/assigned in Microsoft Planner by
the team leader, this repo is where the code gets built.

| Sprint | Focus | Weight | Due |
|---|---|---|---|
| 1 | Pre-processing (`notebooks/sprint1_preprocessing_v3.ipynb`) | 40% | 22/09/2026 |
| 2 | Modeling | 30% | 29/09/2026 |
| 3 | Optimization & Explainability | 30% | 06/10/2026 |

Sprint 2 modeling direction (open decision, informed by Sprint 1 EDA on label cardinality/
correlation): 731 independent binary classifiers vs. a classifier chain vs. a multi-output
tree model (LightGBM/XGBoost + `MultiOutputClassifier`) vs. reframing as multi-class over the
distinct observed BIL combinations. Whatever is chosen, optimize/validate against EMR directly
(held-out split), not per-label metrics.

Sprint 3: hyperparameter tuning against EMR, plus SHAP-based explainability — both for the
report and to check the model isn't exploiting noise in the error-containing training labels.
