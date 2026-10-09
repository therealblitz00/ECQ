# Parfois: Similarity Detection for Fashion Retail Products

Case study for the MADSAD Master's (FEP, University of Porto). The goal is to find, for every Parfois product, at least 4 similar products and explain **why** they match, using product data, text descriptions and images.

> This is the `parfois` branch. It is independent from `main`, which holds a different project.

**Where we are (2026-10-09):** Sprint 1 (data cleaning and validation).
- ✅ Automatic checks.
- ✅ AI image pre-screen.
- ✅ Review app.
- 🟡 Team review in progress.

What has been done, with numbers and examples: **[PROGRESS.md](PROGRESS.md)**. The plan and checklist: **[ROADMAP.md](ROADMAP.md)**.

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
| **Review my batch** (opens the review app) | `2_review_windows.bat` | `2_review_mac.command` |
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
.venv\Scripts\python src\sprint1_preprocess.py batch --members 5 --id 1   # your number, see "Who reviews what"

# Mac (Terminal)
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python src/sprint1_preprocess.py check
.venv/bin/python src/sprint1_preprocess.py batch --members 5 --id 1       # your number, see "Who reviews what"
```
</details>

---

## Current sprint: Sprint 1 (data cleaning and validation)

We check that the **product data**, the **text description** and the **image** of each product all describe the same article. The work is split across the 5 team members.

### Who reviews what

| Your number | Member | Your decisions are saved in |
|---|---|---|
| **1** | André | `outputs/sprint1/batches/batch_01_of_05.csv` |
| **2** | Pedro Correia | `outputs/sprint1/batches/batch_02_of_05.csv` |
| **3** | Pedro Meireles | `outputs/sprint1/batches/batch_03_of_05.csv` |
| **4** | Manuel | `outputs/sprint1/batches/batch_04_of_05.csv` |
| **5** | Zé | `outputs/sprint1/batches/batch_05_of_05.csv` |

Each batch has 2,111 products with the same mix of categories.

### Steps (no CSV editing needed)

1. **Open the review app.** Double-click `2_review_windows.bat` (Windows) or `2_review_mac.command` (Mac) and type **your number** from the table. The app opens in your browser. **Keep the black window open** while you review.
2. **Review.** For each product, click **Looks right**, **Wrong photo**, **Fix data**, **Not sure** or **Discard**. Every click is saved automatically, and you can stop and continue later. Start with the flagged products (the default view), then skim the rest. Details are in **[SPRINT1_GUIDE.md](SPRINT1_GUIDE.md)**.
3. **Submit.** Click **Submit my review** in the app. It sends only your file to GitHub.
4. **Merge.** When all 5 are in, one person runs `3_merge_…`. This produces `data/processed/items_clean.parquet`, the cleaned dataset.
   - Only resolved products go into the clean table.
   - Flagged products nobody reviewed (`pending`) and products marked **Not sure** are listed in `outputs/sprint1/team_review.csv` until the team decides.

<details><summary>Terminal equivalents</summary>

```bash
# Open the review app (replace 4 with your number)
.venv\Scripts\python src\review_app.py --id 4      # Windows
.venv/bin/python src/review_app.py --id 4          # Mac

# Submit by hand instead of the button (example for member 4)
git add outputs/sprint1/batches/batch_04_of_05.csv
git commit -m "Sprint 1 review: batch 4 (Manuel)"
git pull --rebase && git push
```
</details>

**Automatic check results:** [outputs/sprint1/check_report.md](outputs/sprint1/check_report.md). **AI image check results:** [outputs/phase1b/image_audit_report.md](outputs/phase1b/image_audit_report.md).

---

## Repository layout

```
├── README.md                  ← you are here
├── problem_description.md     the case brief
├── data_description.md        what every CSV column and the images mean, plus data-quality notes
├── ROADMAP.md                 project phases, checklist and architecture decisions
├── PROGRESS.md                what has been done so far, with evidence
├── SPRINT1_GUIDE.md           how to do the Sprint 1 review
├── requirements.txt           Python packages
├── 1_/2_/3_*.bat|.command     double-click helpers (Windows / Mac)
├── src/
│   ├── sprint1_preprocess.py  checks, batch split and merge (check | batch | merge)
│   ├── review_app.py          browser review app (python src/review_app.py --id K)
│   └── phase1b_image_audit.py image masks, CLIP embeddings and image checks (heavy: one machine only)
├── data/
│   ├── csv/                   df_product.csv (17,125 SKUs), df_sales.csv (10,185 colourways)
│   ├── images/                9,496 product images (<code>_<n>.jpg)
│   ├── embeddings/            image embeddings, masks and image-check flags (computed once, in git)
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
