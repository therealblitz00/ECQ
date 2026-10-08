# Sprint 1: Preprocessing and Validation Guide

Everyone runs the same script, `src/sprint1_preprocess.py`. The checks and the batch split are deterministic, so every member gets identical results without sending files around.

## Setup (once)

```bash
pip install -r requirements.txt     # or double-click 1_setup_windows.bat / 1_setup_mac.command
```

Expected layout: `data/csv/df_product.csv`, `data/csv/df_sales.csv`, `data/images/*.jpg`.

## Workflow

| Step | Who | Command | Output |
|---|---|---|---|
| 1. Automatic checks | anyone (takes about 15 s) | `python src/sprint1_preprocess.py check` | `outputs/sprint1/check_report.md`, `items_checked.csv`, `missing_values.csv` |
| 2. Get my batch | each member | `python src/sprint1_preprocess.py batch --members 5 --id K` | `outputs/sprint1/batches/batch_K_of_N.csv` and `.html` |
| 3. Review | each member | open the `.html`, fill in the `.csv` | reviewed CSV |
| 4. Merge | one person | put all reviewed CSVs in `outputs/sprint1/batches/`, run `python src/sprint1_preprocess.py merge` | `data/processed/items_clean.parquet`, `changes_log.csv`, `discarded.csv`, `team_review.csv` |

The team has **5 members**: `K` = your number (1–5). Everyone must use `--members 5`, otherwise the batches will not line up at merge.

**How the split works:** one row per colourway (model + colour, `PROD_CLR_EQUIV`), about 2,111 items each for 5 people. All colours of a model go to the same person, and each batch gets a similar mix of categories.

## How to review

The HTML sheet shows each item's image, description, category, family, colour and automatic flags. Flagged items come first and have a **red** (high) or **orange** (medium) border. Check that **image, tabular data and description describe the same article**, then fill in the CSV:

| `review_status` | When | Also fill in |
|---|---|---|
| *(empty)* | Item has no flag and looks fine. Treated as `ok` at merge | – |
| `ok` | Checked a flagged item, it is fine | – |
| `fix` | A **tabular** value or the **description** is wrong | `fixes` |
| `drop_image` | The **image** shows a different article (e.g. the row says necklace, the photo shows earrings) | `notes` |
| `discard` | Nothing matches and it can't be repaired | `notes` |
| `team_review` | Not sure | `notes` |

Always fill in `reviewer` (your name) when you set a status.

**`fixes` syntax:** `COLUMN=new value`, separated by `;`. Example: `GFA_DES_EN=Bracelets; PROD_DES_BASE=Bracelet NILE Gold`. The column must exist (see `items_checked.csv`). Every fix is logged in `changes_log.csv` with its old value and the reviewer.

**Rules:**
- Edit only the four review columns. All other columns are ignored at merge.
- Keep the file as CSV (UTF-8). If you use Excel, use *Save As → CSV UTF-8*.
- Flagged items that nobody reviewed are reported as `pending` in `team_review.csv`.
- Nothing is deleted from `data/`. `drop_image` only removes the image link from the clean table.

## What the automatic checks flag

| Issue | Severity | Meaning |
|---|---|---|
| `IMG_PATH_INVALID` | high | `PROG_IMAGE` doesn't follow the path pattern `/<season>/<category>/<file>` |
| `IMG_UNREADABLE` | high | Image file is corrupt |
| `IMG_REF_MISMATCH` | high | Image file name belongs to another model |
| `IMG_COLOUR_MISMATCH` | high | Colour code in the image file name ≠ `CLR_COD` |
| `CLR_CODE_NAME_INCONSISTENT` | high | The same colour code has a different `CLR_DES` elsewhere |
| `CLR_CONFLICT_DESC` | high | `PROD_DES` names a different colour than `CLR_DES` |
| `TYPE_CONFLICT_DESC` | high | Item type in `PROD_DES` disagrees with `GFA_DES_EN` |
| `TAB_KEY_INCONSISTENT` | high | `PROD_CLR` ≠ `PROD_REF` + `CLR_COD` |
| `IMG_CAT_MISMATCH` | medium | Image is in another category's folder (often a reclassified item) |
| `IMG_GENERIC` | medium | Image is not specific to the colour, so check that the colour matches |
| `IMG_SHARED` | medium | The same image is used by another colour or model (see `img_shared_with`) |
| `CLR_NOT_IN_DESC` | medium | Description doesn't mention the colour |
| `SALES_ZERO_QTY` | low | Zero units sold |
| `IMG_FILE_MISSING`, `SALES_MISSING` | info | No image or no sales. No action needed |

**Limitation:** these checks compare codes and text. They cannot see that a photo shows earrings when the row says necklace. That is what the human review is for. Phase 1b of the roadmap adds a vision model (CLIP) to pre-screen images, so you can review its suspects first.
