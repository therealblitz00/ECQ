# Parfois: Similarity Detection for Fashion Retail Products

Case study for the MADSAD Master's (FEP, University of Porto). The goal is to find, for every Parfois product, at least 4 similar products and explain **why** they match, using product data, text descriptions and images.

> This is the `parfois` branch. It is independent from `Projeto-I` (the default branch), which holds a different project.

**Where we are and what's next:** [docs/roadmap.md](docs/roadmap.md) (the only place with the project status).

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

## Your task now: score the similar products (Phase 4)

The similarity engine is built. Before we fix how much the **photo** counts against the **product attributes**,
we need your judgement on real results. About 20–30 minutes per person, nothing to install or run.

1. `git pull` (or download the ZIP again).
2. Open your contact sheet `outputs/phase4/human_review_<n>.jpg`, where `<n>` is your number:
   1 André · 2 Pedro Correia · 3 Pedro Meireles · 4 Manuel · 5 Zé.
   Each row is one product (**QUERY**, in red) followed by candidates labelled **A, B, C…**
3. Open **your** file `outputs/phase4/human_review_<n>.csv` (Excel works; one file per person, so nobody's save
   clashes with another's) and fill every row:

   | Column | What to write |
   |---|---|
   | `verdict` | `similar` · `partly` · `not` (one per row) |
   | `reviewer` | your name |
   | `notes` | optional: why, if it's a hard case |

   - **similar**: a buyer would accept it as an alternative to the query (same type of product, close style).
   - **partly**: same type, but a clear difference (style, material, colour, price).
   - **not**: a buyer would not show it next to the query.
   - Judge the product, not the photo quality. The rows with "no trusted photo" are judged from the text.
4. Save it as CSV with the same name (Excel's `;` format is fine), then commit and push only your file
   (`Phase 4: human review sheet <n>`). Don't edit other people's files.

The candidates come from three versions of the engine, mixed and without scores, so you can't tell which one
proposed what. When all 5 sheets are done, one person runs:

```bash
.venv/bin/python src/phase1_checks.py check && .venv/bin/python src/phase1_checks.py merge
.venv/bin/python src/phase2a_features.py
.venv/bin/python src/phase4_eval.py run --apply    # picks the photo weight with the best human score
.venv/bin/python src/phase3_engine.py build        # rebuilds the similar products with it
```

---

## Documentation

| Read this | When you want to know |
|---|---|
| [docs/roadmap.md](docs/roadmap.md) | **Status**, the plan by phase, the open checklist, architecture decisions |
| [docs/progress.md](docs/progress.md) | What has been done, with numbers and examples |
| [docs/review_guide.md](docs/review_guide.md) | How to review your batch (who reviews what, the buttons, the flags) |
| [docs/data.md](docs/data.md) | What every column and the images mean, and the data-quality issues |
| [docs/problem.md](docs/problem.md) | The case brief from Parfois |
| [CLAUDE.md](CLAUDE.md) | Conventions for AI coding agents (and a good summary for humans) |
| [docs/archive/](docs/archive/) | Older documents kept for reference (resolved audit) |

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
    └── phase4/                evaluation report + human review sheet (human_review_<n>.csv + .jpg, one per member)
```

## Data confidentiality

The data was provided by Parfois for this case study. Use it only for this course project.
