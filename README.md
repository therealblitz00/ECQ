# Parfois: Similarity Detection for Fashion Retail Products

Case study for the MADSAD Master's (FEP, University of Porto). The goal is to find, for every Parfois product, at least 4 similar products and explain **why** they match, using product data, text descriptions and images.

> This is the `parfois` branch. It is independent from `main`, which holds a different project.

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

### 3. Run the setup, by double-clicking

| | Windows | Mac |
|---|---|---|
| **Setup (once)** | `1_setup_windows.bat` | `1_setup_mac.command` |
| **My review batch** | `2_review_windows.bat` | `2_review_mac.command` |
| **Merge all reviews** (one person, at the end) | `3_merge_windows.bat` | `3_merge_mac.command` |

The setup creates a private Python environment (`.venv/`) in the project folder, installs the packages from `requirements.txt` (pandas, pyarrow, pillow) and runs the automatic data checks. Nothing is installed system-wide.

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
# Windows (PowerShell)
py -3 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python src\sprint1_preprocess.py check
.venv\Scripts\python src\sprint1_preprocess.py batch --members 4 --id 1   # your number 1-4

# Mac (Terminal)
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/sprint1_preprocess.py check
.venv/bin/python src/sprint1_preprocess.py batch --members 4 --id 1       # your number 1-4
```
</details>

---

## Current sprint: Sprint 1 (data cleaning and validation)

We check that the **product data**, the **text description** and the **image** of each product all describe the same article. The work is split across the 4 team members:

1. Run `2_review_…` and enter your number (1–4). Your review page opens in the browser, with flagged items first.
2. Fill in your CSV, `outputs/sprint1/batches/batch_0K_of_04.csv`. The rules are in **[SPRINT1_GUIDE.md](SPRINT1_GUIDE.md)**.
3. Commit and push **only your own CSV**:
   ```bash
   git add outputs/sprint1/batches/batch_0K_of_04.csv
   git commit -m "Sprint 1 review: batch K"
   git pull --rebase && git push
   ```
4. When all 4 are in, one person runs `3_merge_…`. This produces `data/processed/items_clean.parquet`, the cleaned dataset.

Your filled-in CSV is never overwritten if you run the review script again.

**Automatic check results:** [outputs/sprint1/check_report.md](outputs/sprint1/check_report.md).

---

## Repository layout

```
├── README.md                  ← you are here
├── problem_description.md     the case brief
├── data_description.md        what every CSV column and the images mean, plus data-quality notes
├── ROADMAP.md                 project phases and architecture decisions
├── SPRINT1_GUIDE.md           how to do the Sprint 1 review
├── requirements.txt           Python packages
├── 1_/2_/3_*.bat|.command     double-click helpers (Windows / Mac)
├── src/
│   └── sprint1_preprocess.py  checks, batch split and merge (check | batch | merge)
├── data/
│   ├── csv/                   df_product.csv (17,125 SKUs), df_sales.csv (10,185 colourways)
│   ├── images/                9,496 product images (<code>_<n>.jpg)
│   └── processed/             generated, not in git
└── outputs/sprint1/
    ├── check_report.md        automatic check summary
    ├── items_checked.csv      one row per colourway with all flags
    ├── missing_values.csv
    └── batches/               review batch per member (.csv to fill in, .html to look at)
```

## Data in one minute

- **One row per SKU** (product × colour × size) in `df_product.csv`. Most work is done per **colourway** (`PROD_CLR_EQUIV`, e.g. `218792_PM`), which is the level of the sales data, the images and the expected output.
- **Join:** `df_sales.PROD_CLR_EQUIV` ↔ `df_product.PROD_CLR_EQUIV`.
- **Image:** the last part of `PROG_IMAGE` is the file name, so `/241/52/218792_PM_1` → `data/images/218792_PM_1.jpg`. 89.6% of colourways have one.
- **Known issue:** some images don't match their row (e.g. the row says necklace, the photo shows earrings). Sprint 1 exists to find these.

Full details: [data_description.md](data_description.md).

## Roadmap

See [ROADMAP.md](ROADMAP.md). The phases are: data foundation → image audit → features (product data, text, images) → similarity engine → evaluation → explanations → demo.

## Data confidentiality

The data was provided by Parfois for this case study. Use it only for this course project.
