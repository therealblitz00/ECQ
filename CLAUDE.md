# CLAUDE.md

Guidance for AI coding agents working on this repository. Read this first, then the relevant
phase section of `ROADMAP.md` and `data_description.md`.

## Project in one paragraph

Parfois product similarity (MADSAD case study, FEP / University of Porto). For every product
colourway, return **at least 4 similar products** and **explain why** each one matches (visual
and categorical reasons), so commercial buyers can validate the result. The brief is in
`problem_description.md`; the plan and checklist are in `ROADMAP.md`; progress with evidence is
in `PROGRESS.md`.

This is the `parfois` branch. It is independent from `Projeto-I` (the default branch), which
holds a different project. Never merge between them.

## Language

All repository content is written in **English**: code, comments, docstrings, commit messages,
docs and reports. Talking to the team in Portuguese is fine; the files stay in English.

## Key facts about the data

- **Unit of analysis: the colourway, `PROD_CLR_EQUIV`** (model + colour, e.g. `167718_BU`),
  about 10,555 items. `df_product.csv` has one row per SKU (size), so always collapse sizes first.
- Join sales on `PROD_CLR_EQUIV` (left join; ~370 items have no sales).
- Link images through the **file stem of `PROG_IMAGE`**, never by building a name from
  `PROD_CLR` (some file names are irregular). About 1,100 colourways have no image, so every
  method must also work on tabular data alone.
- Read CSVs with `encoding="utf-8"`, keep codes as strings (see `STR_DTYPES` in
  `src/sprint1_preprocess.py`), replace `\xa0` and strip text.
- `UNDEFINED`, `Not Applicable`, `Without Block`, `Others` are placeholders for missing values.
- `GFA_COD` / `GFS_COD` repeat across categories: always combine them with `CAT_COD`.
- Full details: `data_description.md`.

## Repository layout

```
data/csv/          original CSVs                     READ-ONLY
data/images/       9,496 product photos              READ-ONLY
data/processed/    parquet outputs of each phase     regenerated, not in git
data/embeddings/   masks + image/text embeddings     computed once, committed
src/               pipeline scripts (see below)
outputs/           reports, review batches, contact sheets
tests/             pytest suite (toy fixtures + data contracts)
notebooks/         exploration only, never imported
```

Current scripts in `src/`:

| Script | Role |
|---|---|
| `sprint1_preprocess.py` | Shared constants (`ROOT`, `KEY`, paths, `ISSUES`, `TEAM`) and the Sprint 1 CLI: `check`, `batch`, `merge` |
| `phase1b_image_audit.py` | Masks, CLIP embeddings and image-vs-label flags. Heavy; run on one machine only |
| `review_app.py` | Browser app for the manual review (standard library + pandas only) |

As the project grows, new code goes into a package under `src/` (`features/`, `retrieval/`,
`explain/`, `eval/`), one module per roadmap phase. Reuse the constants in
`sprint1_preprocess.py` instead of redefining paths.

## Hard rules

1. **Never modify `data/csv/` or `data/images/`.** All corrections are applied by code and logged
   (`outputs/sprint1/changes_log.csv`).
2. **Never overwrite a reviewer's decisions** in `outputs/sprint1/batches/batch_*_of_05.csv`.
   Those files belong to the team members.
3. **Never silently drop or relabel items.** Flag them with an issue code and a severity, and let
   the human review decide. Items with an unreliable image use the tabular blocks only.
4. **Phases communicate only through files** in `data/processed/` and `data/embeddings/` with a
   documented schema. Do not change another phase's output schema without updating its contract
   in `ROADMAP.md`.
5. **Encode once.** Heavy models (masks, image/text embeddings) run once on one machine, and the
   results are committed to `data/embeddings/` with a sidecar `.json` (model, version, settings,
   date, row order). Everyone else only loads them: the default environment has no `torch`.
6. **Keep the core deterministic.** The similarity engine is a retrieval pipeline, not a
   multi-agent system. Explanations are computed (per-block score contributions plus shared
   attributes); an LLM may only rephrase them. See `ROADMAP.md` § Architecture decision.
7. **Restrict candidates to like-for-like items** (same `CAT_DES_EN`, or `GFA_DES_EN`) unless a
   task says otherwise.

## Environment and commands

Python 3.10+. Light environment: `requirements.txt` (pandas, pyarrow, pillow). Vision
environment for the computing machine only: `requirements-vision.txt` (torch CPU,
transformers, numpy, scipy). Add a package to `requirements.txt` only when a phase needs it.

```bash
# Setup (Windows: .venv\Scripts\python; Mac/Linux: .venv/bin/python)
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# Sprint 1
.venv/Scripts/python src/sprint1_preprocess.py check                     # ~15 s, writes data/processed/items_checked.parquet + outputs/sprint1/check_report.md
.venv/Scripts/python src/sprint1_preprocess.py batch --members 5 --id 4  # one review batch
.venv/Scripts/python src/review_app.py --id 4                            # review app on http://localhost:8765
.venv/Scripts/python src/sprint1_preprocess.py merge                     # -> data/processed/items_clean.parquet

# Phase 1b (vision environment only)
.venv/Scripts/python src/phase1b_image_audit.py --limit 300   # quick trial
.venv/Scripts/python src/phase1b_image_audit.py --reflag      # recompute flags from stored scores
```

`data/processed/` is not in git: run `check` after cloning before anything that reads it.

## Tests and CI

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt   # pytest + ruff
.venv/Scripts/python -m pytest                                # ~2 s
.venv/Scripts/python -m pytest -m "not data"                  # skip the tests on the real data files
.venv/Scripts/ruff check src tests                            # lint (config in pyproject.toml)
```

- `tests/conftest.py` builds a **toy dataset** in a temporary folder and redirects every path
  constant of `sprint1_preprocess` to it, so tests never touch the real `data/` or `outputs/`.
  Reuse the `toy_dirs`, `toy_data` and `checked_items` fixtures.
- `tests/test_data_contracts.py` (marker `data`) checks the committed files later phases rely
  on: embeddings vs their `.json` sidecar, keys of `masks`/`image_audit`, and the review batch
  CSVs (valid statuses, every colourway in exactly one batch). When you add a file to
  `data/embeddings/` or `data/processed/`, add its contract here.
- New code comes with tests that run on small fixtures and need no vision packages.
- **CI** (`.github/workflows/ci.yml`) runs `ruff` and `pytest` on Python 3.10 and 3.14 for every
  push and pull request to `parfois`. Keep it green: don't merge with a red CI.

## Code style

Match the existing scripts:
- `from __future__ import annotations`, type hints, `pathlib.Path` built from `ROOT`.
- A module docstring that says what the script does and how to run it.
- Small functions, pandas-vectorised operations, `argparse` subcommands for CLIs.
- Short comments that explain *why*, not *what*.
- Deterministic output: fixed seeds and stable sort orders, so every teammate gets identical files.
- Write CSVs for humans with `encoding="utf-8-sig"` (opens correctly in Excel).

## Docs and git

- Each task is one `ROADMAP.md` checklist item. When it is done, tick it there and add a short
  entry with evidence (numbers, report paths) to `PROGRESS.md`.
- Docs are written in plain English: short sentences, tables and lists rather than long paragraphs.
- Work on the `parfois` branch (or a branch/worktree created from it). A human reviews and merges
  every task. Commit messages are in English and start with the phase, e.g. `Phase 2a: ...`.
