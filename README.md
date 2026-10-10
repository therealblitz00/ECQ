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
│   ├── phase1b_image_audit.py image masks, CLIP embeddings, image checks (heavy: one machine only)
│   └── review_app.py          browser review app
├── tests/                     pytest suite
├── data/
│   ├── csv/                   df_product.csv (17,125 SKUs), df_sales.csv (10,185 colourways)   READ-ONLY
│   ├── images/                9,496 product photos                                              READ-ONLY
│   ├── embeddings/            masks, image embeddings, image-check flags (computed once, in git)
│   └── processed/             generated tables, not in git (items_checked, items_preview, items_clean)
└── outputs/
    ├── phase1/                check report, cleaning report, review batches (batches/)
    └── phase1b/               image-check report and contact sheets
```

## Data confidentiality

The data was provided by Parfois for this case study. Use it only for this course project.
