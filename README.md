# Parfois: Similarity Detection for Fashion Retail Products

Case study for the MADSAD Master's (FEP, University of Porto). The goal is to find, for every Parfois product, at least 4 similar products and explain **why** they match, using product data, text descriptions and images.

> This is the `parfois` branch. It is independent from `Projeto-I` (the default branch), which holds a different project.

**Where we are and what's next:** [docs/roadmap.md](docs/roadmap.md) (the only place with the project status).

## Start here

1. **[docs/final_report.md](docs/final_report.md)**: the problem, the method, the results and the limitations on one page.
2. The notebooks, in reading order: [Sprint 1, preprocessing](notebooks/phase1_preprocessing.ipynb) →
   [Sprint 2, features and engine](notebooks/phase2_features.ipynb) →
   [Sprint 3, explanations and estimates](notebooks/phase3_explanations_estimates.ipynb).
3. The outputs: `outputs/phase5/similar_products.csv` (5 similar products and why, for every product),
   `outputs/phase5b/estimates.csv` (expected units and price per product), `outputs/phase6/report.html` (a buyer's view).

---

## Quick start (about 5 minutes)

### 1. Get the project

```bash
git clone -b parfois https://github.com/therealblitz00/ECQ.git parfois
```

The download is about 300 MB, because it includes the data and 9,496 images.
No git? On GitHub, switch to the **parfois** branch, then click **Code → Download ZIP** and unzip it.

### 2. Install Python (only if you don't have it)

Install **Python 3.10 or newer** from <https://www.python.org/downloads/>.
- **Windows:** during installation, tick **"Add python.exe to PATH"**.
- **Mac:** the python.org installer is the easiest option.

### 3. Double-click

| | Windows | Mac |
|---|---|---|
| **Setup (once)** | `1_setup_windows.bat` | `1_setup_mac.command` |
| **Review my batch** (opens the review app) | `2_review_windows.bat` | `2_review_mac.command` |
| **Merge all reviews** (one person, at the end) | `3_merge_windows.bat` | `3_merge_mac.command` |

The setup creates a private Python environment (`.venv/`) in the project folder, installs `requirements.txt` (pandas, pyarrow, pillow), runs the automatic data checks and builds the cleaned table (`data/processed/items_preview.parquet`). Nothing is installed system-wide.

How to review your batch: **[docs/review_guide.md](docs/review_guide.md)**.

<details>
<summary><b>Mac: "cannot be opened" or "permission denied"?</b></summary>

- Right-click the file → **Open** → **Open**. You only need to do this once.
- If it still won't run, open Terminal in the project folder and run:
  ```bash
  chmod +x *.command
  ```
</details>

<details>
<summary><b>Prefer the terminal?</b></summary>

```bash
# Windows (PowerShell): use .venv\Scripts\python   ·   Mac: use .venv/bin/python
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/phase1_checks.py check        # automatic checks
.venv/bin/python src/phase1_clean.py               # cleaned table for the next phases
.venv/bin/python src/review_app.py --id 4          # review app (your number, see the review guide)
.venv/bin/python -m pip install -r requirements-dev.txt && .venv/bin/python -m pytest   # tests
```
</details>

---

## Your task now: teach the engine what "similar" means (Phase 4)

The similarity engine is built. Now it learns from your judgement: you sort its suggestions in the browser, and
every query you sort adjusts how much the **photo**, the **category**, the **materials**, the **colours** and the
**price** count. About 20–30 minutes per person (28 queries).

1. `git pull` (or download the ZIP again).
2. Double-click **`4_feedback_windows.bat`** (Windows) or **`4_feedback_mac.command`** (Mac) and type your number:
   1 André · 2 Pedro Correia · 3 Pedro Meireles · 4 Manuel · 5 Zé.
   (Or run `python src/phase4_feedback.py app --id <n>`.)
3. For each query, put every candidate in one of two columns, by dragging it or with ✓ / ✗:
   - **Similar**: a buyer would show it next to the query. Order this column, **most similar first** (drag, or ▲ ▼).
   - **Not similar**: a buyer would not show it next to the query.
   - Judge the product, not the photo quality. Items with "no trusted photo" are judged from the text.
   - Then **Save & next**. You can go back and change an answer at any time; the latest one counts.
4. When you are done, click **Submit my feedback**. It sends only your file, `outputs/phase4/feedback_<n>.jsonl`.

The panel on the right shows what the engine has learned so far. It never changes which candidates you see: each
query's candidates were fixed in advance (three versions of the engine plus two random items), shuffled and without
scores, so the comparison stays fair. One in five queries is kept aside to check the learned weights, and ten
queries are sorted by everybody, to see how much we agree.

When everybody has submitted, one person runs:

```bash
.venv/bin/python src/phase1_checks.py check && .venv/bin/python src/phase1_checks.py merge
.venv/bin/python src/phase2a_features.py
.venv/bin/python src/phase4_feedback.py fit --apply   # learned weights, applied only if they pass the check on unseen queries
.venv/bin/python src/phase3_engine.py build           # rebuilds the similar products with them
.venv/bin/python src/phase4_eval.py run               # checks that no catalogue, visual or sales metric dropped
```

The CSV sheets `outputs/phase4/human_review_<n>.csv` from the first plan still work with `phase4_eval.py run`, but
the app replaces them.

---

## Documentation

| Read this | When you want to know |
|---|---|
| [docs/final_report.md](docs/final_report.md) | **The final report**: method, results, limitations, open questions |
| [docs/roadmap.md](docs/roadmap.md) | **Status**, the plan by phase, the open checklist, architecture decisions |
| [docs/progress.md](docs/progress.md) | What has been done, with numbers and examples |
| [docs/review_guide.md](docs/review_guide.md) | How to review your batch (who reviews what, the buttons, the flags) |
| [docs/data.md](docs/data.md) | What every column and the images mean, and the data-quality issues |
| [docs/problem.md](docs/problem.md) | The case brief from Parfois |
| [CLAUDE.md](CLAUDE.md) | Conventions for AI coding agents (and a good summary for humans) |
| [docs/archive/](docs/archive/) | Older documents kept for reference (resolved audit) |
| [notebooks/phase1_preprocessing.ipynb](notebooks/phase1_preprocessing.ipynb) | Sprint 1 preprocessing step by step, with tables and charts (needs `requirements-notebook.txt`) |
| [notebooks/phase2_features.ipynb](notebooks/phase2_features.ipynb) | Sprint 2 features and modelling step by step: the six blocks, the filled attributes, palettes and colour names; the similarity model, its evaluation and the weights learned from the team |
| [notebooks/phase3_explanations_estimates.ipynb](notebooks/phase3_explanations_estimates.ipynb) | Sprint 3 step by step: the baseline, explanations of every match and their checks, price and sales estimates with SHAP, improvements kept and rejected against the baseline (needs `requirements-notebook.txt`) |

**Naming:** the work is organised in **phases** (0, 1, 1b, 2a, …), and code and output folders use the phase number (`phase1_checks.py`, `outputs/phase1b/`). Sprints are only the calendar: the roadmap shows which phases each sprint covers.

## Repository layout

```
├── README.md                  ← you are here
├── CLAUDE.md                  conventions for AI coding agents
├── 1_/2_/3_*.bat|.command     double-click helpers (Windows / Mac)
├── requirements*.txt          packages: base · -dev (pytest, ruff) · -vision (one machine only)
├── pyproject.toml             pytest and ruff settings
├── .github/workflows/ci.yml   CI: lint + tests on every push
├── docs/                      roadmap, progress, review guide, data dictionary, brief, archive/
├── src/
│   ├── config.py              shared settings: paths, item key, issue codes, team
│   ├── phase1_checks.py       automatic checks, review batches, merge (check | batch | merge)
│   ├── phase1_clean.py        final cleaning, load_items() for the next phases
│   ├── phase1_vocab.py        allowed values for review fixes (fixed lists)
│   ├── phase1b_image_audit.py image masks, CLIP embeddings, image checks (heavy: one machine only)
│   ├── phase2a_*.py, phase2b_palettes.py   feature blocks, filled attributes, text embedding, colour palettes
│   ├── phase3_engine.py       similarity engine (build | query) + phase3_weights.json
│   ├── phase4_eval.py         evaluation, weight tuning, human review sheet (run | sheet)
│   ├── phase4_feedback.py     interactive feedback app + weights learned from it (pool | app | fit)
│   ├── phase5_explain.py      explanation of every match (build | show)
│   ├── phase5b_forecast.py    sales and price estimates with SHAP (run)
│   ├── phase6_deliver.py      command line, HTML report and the brief's example (query | report | brief)
│   └── review_app.py          browser review app
├── tests/                     pytest suite
├── data/
│   ├── csv/                   df_product.csv (17,125 SKUs), df_sales.csv (10,185 colourways)   READ-ONLY
│   ├── images/                9,496 product photos                                              READ-ONLY
│   ├── embeddings/            masks, image embeddings, image-check flags (computed once, in git)
│   └── processed/             generated tables, not in git (items_clean, features_*, neighbours)
└── outputs/
    ├── phase1/                check report, cleaning report, review_vocabulary.csv, review batches (batches/)
    ├── phase1b/               image-check report and contact sheets
    ├── phase2a/, phase2b/     feature and palette reports
    ├── phase3/                engine report + example matches
    └── phase4/                evaluation report, feedback pool + answers (feedback_<n>.jsonl, one per member)
```

## Data confidentiality

The data was provided by Parfois for this case study. Use it only for this course project.
