# CLAUDE.md

Guidance for AI coding agents working on this repository. Read this first, then the relevant
phase section of `docs/roadmap.md` and `docs/data.md`.

## Project in one paragraph

Parfois product similarity (MADSAD case study, FEP / University of Porto). For every product
colourway, return **at least 4 similar products** and **explain why** each one matches (visual
and categorical reasons), so commercial buyers can validate the result. The brief is in
`docs/problem.md`; the plan, checklist and status are in `docs/roadmap.md`; progress with evidence
is in `docs/progress.md`.

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
  `PROD_CLR` (some file names are irregular). About 1,064 colourways have no image, so every
  method must also work on tabular data alone.
- Read CSVs with `encoding="utf-8"`, keep codes as strings (see `STR_DTYPES` in
  `src/config.py`), replace `\xa0` and strip text.
- Placeholders (`UNDEFINED`, `Not Applicable`, `Without Block`, `Sem Categoria`) are missing values
  only in the columns listed in `phase1_clean.PLACEHOLDERS`. `Others` is a real category in
  `PRINT_TYPE`, `GFA_DES_EN` and `FINISHING`.
- `GFA_COD` / `GFS_COD` repeat across categories: always combine them with `CAT_COD`.
- Full details: `docs/data.md`.

## Repository layout

```
data/csv/          original CSVs                     READ-ONLY
data/images/       9,496 product photos              READ-ONLY
data/processed/    parquet outputs of each phase     regenerated, not in git
data/embeddings/   masks + image/text embeddings     computed once, committed
src/               pipeline scripts (see below)
outputs/phase<N>/  reports, review batches, contact sheets of each phase
tests/             pytest suite (toy fixtures + data contracts)
docs/              roadmap (status), progress log, review guide, data dictionary, brief, archive/
notebooks/         self-contained notebooks (deliverables), never imported and never importing src/
```

**Naming: phases only.** Code and output folders use the roadmap phase number
(`src/phase2a_features.py`, `outputs/phase2a/`). Sprints are only the calendar in the roadmap;
never put a sprint number in a file or folder name.

Current scripts in `src/`:

| Script | Role |
|---|---|
| `config.py` | Shared settings: paths (`ROOT`, `PROCESSED_DIR`…), `KEY`, `STR_DTYPES`, issue codes (`ISSUES`), review statuses, `TEAM` |
| `phase1_checks.py` | Phase 1 CLI: `check` (automatic checks), `batch` (review batches), `merge` (apply the review) |
| `phase1b_image_audit.py` | Masks, CLIP embeddings and image-vs-label flags. Heavy; run on one machine only |
| `review_app.py` | Browser app for the manual review (standard library + pandas only) |
| `phase1_vocab.py` | Allowed values for review fixes (`build_vocabulary`, `normalise_fixes`). Used by the review app and `merge`; never accept a fix value that isn't in it |
| `phase1_clean.py` | Final cleaning (`finalise_items`), the preview table and `load_items()`. Its `CONTRACT_COLUMNS` are the columns later phases may rely on |
| `phase1b_precision.py` | Precision of the image flags from the merged review → `outputs/phase1b/flag_precision.md` (no vision packages) |
| `phase2a_features.py` | Feature blocks (`features_<block>.npy` + `feature_meta.json`, same row order): categorical, composition, numeric, text. Every block has unit-length rows and named attribute slices that add up to the score |
| `phase2a_impute.py` | Fills `MATERIAL` (apparel) and `FINISH_FAMILY` (jewellery) where the CSV is empty and the inference is reliable → `data/processed/attributes_inferred.parquet`. Run by `phase2a_features.py`; never changes the CSV values |
| `phase2a_colour.py` | Reference colour of each `CLR_DES` name, measured on the trusted photos, and the similarity of two names → `data/processed/colour_reference.parquet` + `outputs/phase2a/colour_reference.{csv,jpg}`. Run by `phase2a_features.py`; for the Phase 5 explanations, not a score block |
| `phase2a_text.py` | Sentence embedding of a description built from the attributes → `data/embeddings/text.npy` + `.json`. Heavy; vision machine only |
| `phase3_engine.py` | Similarity engine: weighted sum of per-block cosines (weights in `phase3_weights.json`), like-for-like candidates, other colours of the same model excluded → `data/processed/neighbours.parquet` (top-k with `sim_<block>` / `contrib_<block>`). CLI: `build`, `query KEY` |
| `phase4_eval.py` | Evaluation: proxy hit rate, attribute agreement, sales check, ablations, weight tuning (`run`, `--apply` writes `phase3_weights.json`) and the blind human review sheet (`sheet`; never overwrites verdicts) → `outputs/phase4/` |
| `phase4_feedback.py` | Interactive human feedback: fixed candidate pools (`pool` → `outputs/phase4/feedback_pool.json`, committed; never depends on the learned weights), the sorting app (`app --id N`, standard library + pandas; each member appends to `feedback_<n>.jsonl` only), and Bradley-Terry fitting of the block weights with held-out check (`fit`, `--apply` writes `phase3_weights.json` only if the check passes) |
| `phase2b_palettes.py` | Dominant colours per trusted photo (k-means in LAB on the product pixels) → `data/embeddings/palettes.parquet` + `.json`. Light environment, ~2.5 min |
| `phase2b_image_encoders.py` | Photos with a person: suggested re-crops (`persons` → `data/embeddings/person_crops.parquet`; only the suggestions accepted in the hand-edited `person_crops_reviewed.json` are applied, never overwritten) and image embeddings from other encoders (`embed --model fashionsiglip` → `data/embeddings/image_<model>.npy` + `.json`). Heavy; vision machine only (+ `open_clip_torch`) |
| `phase4_compare.py` | The current engine (`image_embedding` and weights of `phase3_weights.json`) against the Sprint 2 baseline (CLIP) on the team's preferences (fixed and cross-validated, bootstrap CI), photo-only label agreement, guards and the brief's example → `outputs/phase4/compare_report.md` |
| `phase5_explain.py` | Explanation of every match, an exact decomposition of the engine score (block shares, shared attributes/materials, differences, close colours) → `data/processed/explanations.parquet`, `outputs/phase5/` (brief's format, checks). CLI: `build`, `show KEY` |
| `phase5b_forecast.py` | Sales and price estimates for new products (temporal test, baselines, LightGBM + exact TreeSHAP) → `data/processed/forecast.parquet`, `outputs/phase5b/forecast_report.md` |
| `phase6_deliver.py` | Delivery: `query KEY` (similar products, why, estimates) and `report` → `outputs/phase6/report.html` |

New code goes into `src/phase<N>_<topic>.py`, one module per roadmap phase (move to a package
only when a phase outgrows one file). Import paths and constants from `config.py` instead of
redefining them.

## Hard rules

1. **Never modify `data/csv/` or `data/images/`.** All corrections are applied by code and logged
   (`outputs/phase1/changes_log.csv`).
2. **Never overwrite a reviewer's decisions** in `outputs/phase1/batches/batch_*_of_05.csv`.
   Those files belong to the team members.
3. **Never silently drop or relabel items.** Flag them with an issue code and a severity, and let
   the human review decide. Items with an unreliable image use the tabular blocks only.
4. **Phases communicate only through files** in `data/processed/` and `data/embeddings/` with a
   documented schema. Do not change another phase's output schema without updating its contract
   in `docs/roadmap.md`.
5. **Encode once.** Heavy models (masks, image/text embeddings) run once on one machine, and the
   results are committed to `data/embeddings/` with a sidecar `.json` (model, version, settings,
   date, row order). Everyone else only loads them: the default environment has no `torch`.
6. **Keep the core deterministic.** The similarity engine is a retrieval pipeline, not a
   multi-agent system. Explanations are computed (per-block score contributions plus shared
   attributes); an LLM may only rephrase them. See `docs/roadmap.md` § Architecture decision.
7. **Restrict candidates to like-for-like items** (same `CAT_DES_EN`, or `GFA_DES_EN`) unless a
   task says otherwise.
8. **A notebook never calls a project Python file.** No `import` of anything in `src/`, no `sys.path` tricks,
   no `%run`, no `subprocess`/`!python`. Every step is written in the notebook itself with pandas, NumPy,
   matplotlib, Pillow and LightGBM (`requirements-notebook.txt`), reading only data files (`data/`, `outputs/`). Steps that were not done in code
   (the manual review) or are too heavy to rerun (CLIP, embeddings) are **described**, and their saved
   results are loaded as data. Notebooks are course deliverables: a reader must be able to follow every step.

## Environment and commands

Python 3.10+. Light environment: `requirements.txt` (pandas, pyarrow, pillow). Vision
environment for the computing machine only: `requirements-vision.txt` (torch CPU,
transformers, numpy, scipy). Add a package to `requirements.txt` only when a phase needs it.

```bash
# Setup (Windows: .venv\Scripts\python; Mac/Linux: .venv/bin/python)
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt

# Phase 1
.venv/Scripts/python src/phase1_checks.py check                     # ~15 s, writes data/processed/items_checked.parquet + outputs/phase1/check_report.md
.venv/Scripts/python src/phase1_clean.py                            # -> data/processed/items_preview.parquet + outputs/phase1/clean_report.md
.venv/Scripts/python src/phase1_checks.py batch --members 5 --id 4  # one review batch
.venv/Scripts/python src/review_app.py --id 4                       # review app on http://localhost:8765
.venv/Scripts/python src/phase1_checks.py merge                     # -> data/processed/items_clean.parquet

# Phase 2a (rerun after merge)
.venv/Scripts/python src/phase2a_features.py                       # -> data/processed/features_*.npy + outputs/phase2a/features_report.md
.venv/Scripts/python src/phase2a_text.py --show 5                  # preview the descriptions (the embedding itself needs the vision environment)

# Phase 2b (computed once, results committed)
.venv/Scripts/python src/phase2b_palettes.py --limit 60            # trial + outputs/phase2b/palettes_sample.jpg; without --limit: all photos

# Phase 3 (rerun after Phase 2a)
.venv/Scripts/python src/phase3_engine.py build                  # -> data/processed/neighbours.parquet + outputs/phase3/engine_report.md, ~10 s
.venv/Scripts/python src/phase3_engine.py query 167718_BU --k 5  # one item, with the score per block

# Phase 4
.venv/Scripts/python src/phase4_eval.py run                      # -> outputs/phase4/eval_report.md, ~3 min
.venv/Scripts/python src/phase4_eval.py run --apply              # also write the chosen weights (then rerun phase3_engine.py build)
.venv/Scripts/python src/phase4_feedback.py app --id 4           # sorting app on http://localhost:8766 (pool built once with `pool`)
.venv/Scripts/python src/phase4_feedback.py fit --apply          # learn the weights from everybody's feedback_<n>.jsonl
.venv/Scripts/python src/phase4_compare.py run                   # current engine vs the Sprint 2 baseline -> outputs/phase4/compare_report.md, ~4 min

# Phase 5, 5b, 6 (rerun after Phase 3)
.venv/Scripts/python src/phase5_explain.py build                 # -> data/processed/explanations.parquet + outputs/phase5/, ~3 min
.venv/Scripts/python src/phase5_explain.py show 167718_BU        # explained top 5 of one product
.venv/Scripts/python src/phase5b_forecast.py run                 # -> data/processed/forecast.parquet + outputs/phase5b/, ~6 min
.venv/Scripts/python src/phase6_deliver.py query 167718_BU       # similar products, why, sales and price estimates
.venv/Scripts/python src/phase6_deliver.py report                # -> outputs/phase6/report.html

# Phase 1b (vision environment only)
.venv/Scripts/python src/phase1b_image_audit.py --limit 300   # quick trial
.venv/Scripts/python src/phase1b_image_audit.py --reflag      # recompute flags from stored scores
.venv/Scripts/python src/phase1b_image_audit.py --add-missing # embed only items not in data/embeddings yet
.venv/Scripts/python src/phase1b_precision.py                # flag precision from the merged review (light environment)
```

`data/processed/` is not in git: run `check` and then `python src/phase1_clean.py` after cloning
(the setup scripts do both).

**Input of Phase 2 onwards:** always `phase1_clean.load_items()`. It returns
`items_clean.parquet` (after the human review is merged) or else `items_preview.parquet` (before
it). Both have the same columns. Use `img_trusted` / `img_colour_trusted` to decide whether an
item's photo may feed the visual block; never use `has_image` alone, because the review is not done yet.

## Tests and CI

```bash
.venv/Scripts/python -m pip install -r requirements-dev.txt   # pytest + ruff
.venv/Scripts/python -m pytest                                # ~2 s
.venv/Scripts/python -m pytest -m "not data"                  # skip the tests on the real data files
.venv/Scripts/ruff check src tests                            # lint (config in pyproject.toml)
```

- `tests/conftest.py` builds a **toy dataset** in a temporary folder and redirects every path
  constant of `phase1_checks` and `phase1_clean` to it, so tests never touch the real `data/` or `outputs/`.
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

- Each task is one `docs/roadmap.md` checklist item. When it is done, tick it there (the roadmap
  is the **only** place with the status) and add a short entry with evidence (numbers, report
  paths) to `docs/progress.md`. Don't repeat the status in the README or elsewhere.
- Old documents go to `docs/archive/` with a note at the top saying what is still valid.
- Docs are written in plain English: short sentences, tables and lists rather than long paragraphs.
- Work on the `parfois` branch (or a branch/worktree created from it). A human reviews and merges
  every task. Commit messages are in English and start with the phase, e.g. `Phase 2a: ...`.
