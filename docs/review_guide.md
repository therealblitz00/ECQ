# Phase 1 Review Guide

We check that the **product data**, the **text description** and the **photo** of each product all describe the same article. The work is split across the 5 team members. Setup is in the [README](../README.md#quick-start-about-5-minutes).

## Who reviews what

| Your number | Member | Your decisions are saved in |
|---|---|---|
| **1** | André | `outputs/phase1/batches/batch_01_of_05.csv` |
| **2** | Pedro Correia | `outputs/phase1/batches/batch_02_of_05.csv` |
| **3** | Pedro Meireles | `outputs/phase1/batches/batch_03_of_05.csv` |
| **4** | Manuel | `outputs/phase1/batches/batch_04_of_05.csv` |
| **5** | Zé | `outputs/phase1/batches/batch_05_of_05.csv` |

## Workflow

| Step | Who | How | Output |
|---|---|---|---|
| 1. Review | each member | double-click `2_review_…` and type your number (or `python src/review_app.py --id K`) | your batch CSV, saved on every click |
| 2. Submit | each member | **Submit my review** button in the app | your CSV on GitHub |
| 3. Merge | one person, when all 5 are in | double-click `3_merge_…` (or `python src/phase1_checks.py check` then `merge`) | `data/processed/items_clean.parquet` (decisions applied + Phase 1 cleaning), `outputs/phase1/changes_log.csv`, `discarded.csv`, `team_review.csv`, `clean_report.md` |

The batches already exist (`outputs/phase1/batches/`). If yours is missing, `2_review_…` creates it with `python src/phase1_checks.py batch --members 5 --id K`. Always use `--members 5`, otherwise the batches won't line up at merge.

**How the split works:** one row per colourway (model + colour, `PROD_CLR_EQUIV`), about 2,111 items each for 5 people. All colours of a model go to the same person, and each batch gets a similar mix of categories.

## How to review (review app, no CSV editing)

1. Double-click `2_review_windows.bat` (Windows) or `2_review_mac.command` (Mac) and type your number.
2. The review app opens in your browser. **Keep the black window open** while you review, and close it when you're done.
3. Each product is a card with its photo (click it to enlarge), description, category › family › sub-family, colour, composition and the automatic flags in plain words. Flagged products have a **red** (high) or **orange** (medium) border.
4. Ask yourself: do the **photo**, the **description** and the **data** describe the same article? Then click one of these buttons:

| Button | When | Saved as |
|---|---|---|
| **Looks right** | Everything matches | `ok` |
| **Wrong photo** | The photo shows a different article (e.g. the row says necklace, the photo shows earrings) | `drop_image` |
| **Fix data** | The description or a data field is wrong. A small form opens: type only the corrected values (suggestions appear as you type), then click **Save fix** | `fix` + `fixes` |
| **Not sure** | You can't decide. Add a note, and the team decides later | `team_review` |
| **Discard** | Nothing matches and it can't be repaired | `discard` |

   Click a selected button again to undo it. The **Note** box is optional.
5. **Work order:** the app starts on *To check*: the flagged products you haven't decided on yet. When that list is empty, switch **Show → All products** and skim the rest. Unflagged products you don't touch count as correct. The **Mark untouched on this page as "Looks right"** button speeds this up.
6. When you're done, click **Submit my review**. It sends only your file to GitHub. If that fails (e.g. git isn't installed), the app tells you which file to send to the team instead.

Your name is filled in automatically, and every click is saved immediately to `outputs/phase1/batches/batch_0K_of_05.csv`. You can close the app and continue later.

<details><summary>Editing the CSV by hand instead (advanced)</summary>

| `review_status` | When | Also fill in |
|---|---|---|
| *(empty)* | Item has no flag and looks fine. Treated as `ok` at merge | – |
| `ok` | Checked a flagged item, it is fine | – |
| `fix` | A **tabular** value or the **description** is wrong | `fixes` |
| `drop_image` | The **image** shows a different article | `notes` |
| `discard` | Nothing matches and it can't be repaired | `notes` |
| `team_review` | Not sure | `notes` |

Always fill in `reviewer` (your name) when you set a status. Don't edit the CSV while the review app is running.
</details>

**`fixes` syntax (for the CSV):** `COLUMN=new value`, separated by `;`. Example: `GFA_DES_EN=Bracelets; COMPOSITION=Pearl; Zinc`. A `;` only starts a new fix when it is followed by `COLUMN=`, so values such as `COMPOSITION` can contain `;`. The column must exist (see `outputs/phase1/items_checked.csv`). Every fix is logged in `changes_log.csv` with its old value and the reviewer.

**Rules:**
- If you edit the CSV by hand: change only the four review columns (everything else is ignored at merge), and in Excel save with *Save As → CSV UTF-8*.
- Flagged items that nobody reviewed (`pending`) and items marked `team_review` are **left out of the clean table** and listed in `team_review.csv` until the team decides. Use `merge --include-unresolved` only if you deliberately want to keep them.
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
| `SKU_ATTR_CONFLICT` | medium | Sizes of the same colourway disagree on an attribute (the `sku_conflicts` column lists which ones). The most common value is used; correct it with `fix` if it's wrong |
| `SALES_ZERO_QTY` | low | Zero units sold |
| `VIS_TYPE_MISMATCH` | high | *Image check:* the photo looks like another type of item (e.g. earrings for a necklace row), and the most similar photos agree |
| `VIS_COLOUR_MISMATCH` | medium | *Image check:* the photo looks like another colour than `CLR_DES` |
| `VIS_TARGET_NOT_FOUND` | low | *Image check:* the detector couldn't find the expected item (mostly photos of samples on cards, sketches or mannequins) |
| `IMG_NEAR_DUPLICATE` | medium | *Image check:* the same photo is used for another colour of the same model, so for at least one colour it shows the wrong colour |
| `VIS_TYPE_DOUBT` | low | *Image check:* only the image model doubts the item type (often fine: cuffs, charms, unusual shapes) |
| `IMG_NOT_PACKSHOT` | low | *Image check:* model, lifestyle or amateur photo, not a studio shot |
| `IMG_FILE_MISSING`, `SALES_MISSING` | info | No image or no sales. No action needed |

**About the image checks:** `VIS_*` and `IMG_NEAR_DUPLICATE`/`IMG_NOT_PACKSHOT` come from Phase 1b (`src/phase1b_image_audit.py`). An image model (CLIP) and an object detector (OWLv2) looked at every photo, after cutting out the target item. They were computed once and stored in `data/embeddings/`, so you don't need to run anything. They are **suggestions, not verdicts**: the model can be fooled by unusual shapes. You decide. Photos of a model wearing several items (e.g. earrings and a necklace) are fine to review by eye: judge only the item the row describes.
