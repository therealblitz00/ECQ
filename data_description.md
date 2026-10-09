# Data Description: Parfois Similarity Detection

This document describes the two CSV files in `data/csv/` and the product images in `data/images/` and how they map to the data inputs listed in `problem_description.md`. All figures come from profiling the files as delivered.

| File | Rows | Columns | Grain (one row = …) | Problem-description pillar |
|---|---|---|---|---|
| `data/csv/df_product.csv` | 17,125 | 141 | one **SKU**: product × colour × size | Master Product Data (+ image references) |
| `data/csv/df_sales.csv` | 10,185 | 3 | one **colourway**: product × colour | Accumulated Sales Data |

Product images are in `data/images/` (9,496 files, one flat folder, about 278 MB). See §4 for how they link to the CSV rows and §5 for known quality problems.

---

## 1. Identifiers and granularity

Parfois product codes are built hierarchically:

```
PROD_REF        PROD_CLR           PROD_COD
218792    →     218792_PM     →    218792_PML
(model)         (model + colour)   (model + colour + size)
```

| Level | Column | Distinct values | Meaning |
|---|---|---|---|
| Model / style | `PROD_REF` | 7,644 | Base design, independent of colour and size |
| Colourway | `PROD_CLR` | 10,571 | Model in a specific colour (`_` + 2-letter colour code, see `CLR_COD`) |
| SKU | `PROD_COD` | 17,125 (unique) | Colourway in a specific size (`U` = one size) |
| Surrogate key | `PROD_SPK` | 17,125 (unique) | Internal numeric key of the SKU record |
| Barcode | `BAR_COD` | 17,125 (unique) | EAN-13 |

**`*_EQUIV` columns.** `PROD_REF_EQUIV`, `PROD_CLR_EQUIV` and `PROD_COD_EQUIV` map an item to its *equivalent* (canonical) reference. They are identical to the non-EQUIV columns except for 28 rows, where a re-coded item points to its successor (e.g. `222563_GY` → `223376_GY`). After the mapping there are 10,555 distinct colourways.

**Join key.** `df_sales.PROD_CLR_EQUIV` ↔ `df_product.PROD_CLR_EQUIV` (many-to-one: product rows to sales rows).
- All 10,185 sales keys exist in the product table.
- 370 product colourways (≈3.5%) have no sales row.
- A colourway has 1.6 SKUs on average (median 1, max 26; e.g. the letter necklace `216741_GD` has one SKU per letter A–Z).

> **Implication for the task:** the expected output uses colourway-level IDs (e.g. `167718_BU`, `169597_FU`). Sales and images also exist per colourway, so **the colourway (`PROD_CLR_EQUIV`) is the natural unit for similarity**. Deduplicate or aggregate the SKU rows before modelling: size is the only thing that varies within a colourway.
>
> Of the four example IDs in the problem description, `167718_BU`, `169597_FU` and `169607_IG` are in both files. `117781_LB` is in neither.

---

## 2. `df_sales.csv`

| Column | Type | Description |
|---|---|---|
| `PROD_CLR_EQUIV` | string | Colourway key (unique, no nulls) |
| `SALES_QTY` | float | Accumulated units sold |
| `SALES_AMT_FX_RATE` | float | Accumulated sales value converted to one currency at FX rates (presumably EUR) |

**Summary**

| | SALES_QTY | SALES_AMT_FX_RATE |
|---|---|---|
| mean | 3,198 | 47,519 |
| median | 2,624 | 24,889 |
| max | 112,982 | 1,028,794 |
| zeros | 9 | 8 |

- The totals are accumulated over each item's life. The file has **no time dimension** (no weekly, store or country breakdown).
- Heavily right-skewed: use log-transforms or ranks when using sales as a feature or a weight.
- A few rows have `SALES_QTY = 0` but a positive amount (e.g. `213332_GD`), which gives an infinite "realised price". Guard against this when dividing.
- Median sales by category: Hand Bag ≈ 4,050 units, Jewellery ≈ 2,790, Wallet ≈ 2,720, Footwear ≈ 1,760, Apparel ≈ 1,155. Sales are only comparable within a category.

---

## 3. `df_product.csv`

### 3.1 Coverage

- **Seasons (`PROD_YEAR_SEA` / `SEA_COD`):** 24 Fall/Winter (7,468 rows, `242`), 23 Fall/Winter (4,963, `232`), 24 Spring/Summer (4,694, `241`).
- **Store dates (`STORE_DATE_FINAL`):** 2023-04-01 → 2025-01-15.
- **Categories (`CAT_DES_EN`, SKU rows):** Apparel 5,692 · Jewellery 5,583 · Footwear 2,606 · Hand Bag 1,986 · Wallet 1,258.
- **Segment (`PROD_SEG`):** Woman 14,458 · Blue Line 1,412 · Green Line 911 · Teen 333.

### 3.2 Columns by theme

Fill rate = share of non-null rows. Columns marked ★ are the most useful for the similarity task.

#### Identification and description
| Column | Fill | Description |
|---|---|---|
| `PROD_SPK`, `PROD_COD`, `PROD_CLR`, `PROD_REF`, `BAR_COD` | 100% | Keys (see §1) |
| ★ `PROD_DES` | 100% | Short name: `<Type> <THEME/line> <Colour> <Size>`, e.g. *"Earring GLDN DEL Gold U"*. 11,836 distinct values |
| `DESCRIPTION` | 51% | Same pattern as `PROD_DES` (equal in about half the rows). Adds little |
| `PROD_COD_EQUIV`, `PROD_CLR_EQUIV`, `PROD_REF_EQUIV` | 100% | Canonical keys (see §1) |
| `ITEM_REPEAT` | 75% | `With` / `Without Repetition` (re-edition of a past item) |
| `HISTORY` | 1% | Legacy reference / licence tag (e.g. `KARL`) |

#### Product hierarchy (★ core categorical features)
Several overlapping taxonomies describe the same hierarchy:

| Column(s) | Distinct | Example | Notes |
|---|---|---|---|
| ★ `CAT_COD` / `CAT_DES_EN` | 5 | `52` Jewellery, `53` Footwear, `54` Hand Bag, `64` Apparel, Wallet | Top category. `_PT` duplicates `_EN` |
| ★ `GFA_COD` / `GFA_DES_EN` | 27 / 67 | Earrings, Rings, Dress, Trousers, Ballerinas, Tote… | Product family (item type). Codes repeat across categories, so use the description or the pair (CAT, GFA) |
| ★ `GFS_COD` / `GFS_DES_EN` | 47 / 104 | Hoop Earrings, Stud, Woven, Knit, PU, Plain PU… | Sub-family: shape for jewellery, construction or material for apparel and bags |
| `L1_DES` → `L4_DES` | 4 / 5 / 20 / 60 | TEXTILE › Apparel › TOPS › Dress | Alternative merchandising tree (L3 holds lines such as *DELICATE JEWELLERY* and *COLOR COLLECTION*) |
| ★ `CATEGORY_MATRIX` / `CATEGORY_MATRIX_BUYER` | 103 / 110 | `52 - Earrings Long`, `64 - Dress_Other` | Planning category used by buyers. Good level for "comparable items" |
| `FSALDOS`, `FSALDOS_OUTLET` | 24 / 5 | `52 - EARRINGS` | Category groupings for sales and outlet reporting |
| `CAT_SIT`, `GFA_SIT`, `GFS_SIT`, `CAT_ACTIVE` | | `S`/`N`, `Y` | Status flags of the hierarchy node |

#### Colour (★)
| Column | Fill | Description |
|---|---|---|
| ★ `CLR_COD` / `CLR_DES` | 100% | 153 codes / 100 names. Top: Black, Bright Multicolor, Gold, Ecru, Silver |
| ★ `CLR_TYPE` | 100% | `Fashion`, `Basic`, `Multicolor` |
| `MATCHING` | 100% | Yes/No: belongs to a coordinated (matching) set |
| `CLR_TON_DES`, `CLR_BAS_DES`, `DISPLAY_CLR*` | ~0% | Empty |

#### Material, finish and physical attributes (★)
Category-specific attributes are only filled for the categories where they apply:

| Column | Fill | Filled for | Description |
|---|---|---|---|
| ★ `COMPOSITION` | 100% | all | Free-text material list, e.g. *"Acrylic Pearl Thermoplastic Polyurethane Zinc"* (2,847 variants). Tokenise it |
| ★ `MATERIAL` | 19% | Apparel (55%) | Main fibre: Polyester, Viscose, Cotton, Lyocell… |
| ★ `FINISHING` | 35% | Jewellery (90%), Hand Bag (45%) | Metal finish: Golden, Silver, Worn Gold, Light Golden… |
| ★ `OUTFIT` | 33% | Apparel (100%) | Top / Bottom / Dress / Overall |
| ★ `DIMENSION` | 100% | | Short / Long / Medium…. Mostly `Not Applicable` or `UNDEFINED` |
| ★ `PRINT_TYPE` | 100% | | No Print (67%), Others, Text, Floral, Animal (Leopard, Zebra), Patterns (Spots, Stripes) |
| `SHAPE` | 1% | Footwear (6%) | Slingback, Mule, Mocassins, Wedge… |
| `PRODUCT_DETAILS` | 1% | | Motif: Heart, Shell, Eye… |
| `DISTRIBUTION_BLOCK` | 100% | | Motif or theme block (Heart_Love, Seashell, Fresh Water Pearl). 96% `Without Block` |
| `NUMBER_OF_UNITS` | 100% | | Singular, Set 1 Pair, 2/3 Units… |
| `OPACITY` | 100% | | Almost always `Opaque` |
| `SZ_COD` / `SZ_DES` | 100% | | Size (U, XS/S, M/L, 36–41…). Not relevant to similarity at colourway level |
| `NET_WEIGHT` | 95% | | kg |
| `TARIFF_COD` | 100% | | Customs HS code. Encodes material and type (e.g. `7117…` imitation jewellery, `4202…` bags) |
| `MADE_IN` | 100% | | Country of origin (China 85%) |

#### Collection, theme and fashion positioning (★)
| Column | Distinct | Description |
|---|---|---|
| ★ `THEME` / `MAIN_THEME` | 816 / 701 | Design story or line (GOLDEN DELICATES, T1 ESSENTIALS, TRENDY…). Strong grouping signal. `THEME_ASIS` duplicates it |
| `THEME_CODE`, `THEME_ID` | 75 / 822 | Theme codes (`THEME_CODE` only 35% filled) |
| ★ `FASHIONTYPE` / `FASHIONTYPE_COD` | 13 | Fashion, Basic, Fashion Basic, Commercial Online, Capsule, Maxi, Plus Closet, Passerelle… |
| `COLLECTION_TYPE` / `_GROUP` / `COLLECTION_STRUCTURE` | 13 / 7 / 14 | Core Collection (77%), Exclusive Online, Set, Leather, Limited Edition, Ramadan… |
| `PROD_SEG` | 4 | Customer segment (Woman, Blue Line, Green Line, Teen) |
| `TYPOLOGY` | 4 | Store-cluster assortment (`A`, `BA`, `CBA`, `DCBA`): which store tiers receive the item |
| `PUR_BET` | 5 | Purchase bet / depth level (`M1`–`M3`, `SB`) |
| `CUSTOMIZABLE` | 3 | Non Customizable / Embroidery / Stamping |
| `SEASON_DETAIL` | 3 | Summer / Winter / Spring (4% filled) |

#### Price and cost
| Column | Fill | Description |
|---|---|---|
| ★ `PRICE_BASE_W_VAT` | 100% | Retail price incl. VAT. 40 price points from €3.99 to €250, mean ≈ €25. `PRICE_BASE_W_VAT_ASWAS` is identical |
| `PRICE_OUTLET_W_VAT` | 100% | Outlet/markdown price |
| `PRICE_LCP` | 96% | Landed cost price |
| `TRANSP_COST` | 92% | Transport cost per unit |

#### Dates and life cycle
| Column | Fill | Description |
|---|---|---|
| ★ `STORE_DATE_FINAL` | 100% | Effective store launch date (the problem's "Exposition Date"). Use this one |
| `STORE_DATE_PLANNED` / `_REAL` / `_REVISED` | 100 / 35 / 56% | Inputs to the final date (YYYYMMDD) |
| `STORE_DATE_OUTLET_REAL` | 27% | Date moved to outlet |
| `STORE_MONTH_FINAL_PC` | 100% | Launch month name |
| `ONL_ENTRY_DATE(_DT)`, `ONL_IMAGE_DATE(_DT)`, `ECI_ONLINE_DATE` | 91 / 97 / 53% | Online launch and photo dates |
| `ITEM_LIFE_CYCLE_WEEK` = `WEEKS_TARGET` | 100% | Planned selling life in weeks (mostly 8) |
| `PROD_YEAR_SEA_INI` | 100% | Season in which the item first appeared (11 values, back to older seasons for carry-overs) |
| `START_DATE`, `AUDIT_DT_INSERT` | 100% | Record validity and load timestamps (technical) |

#### Images
| Column | Description |
|---|---|
| ★ `PROG_IMAGE` | Relative image path `/<SEA_COD>/<CAT_COD>/<PROD_CLR>_<n>`, e.g. `/241/52/218792_PM_1`. 10,540 distinct paths, essentially **one image per colourway** (only 29 colourways have more than one). Some have suffixes (`_1y`, `_3y`) or irregular names (`2089811OW_1`). There is no file extension. The matching file is `data/images/<last part>.jpg` (see §4) |

#### Operational and other flags
`SUP_SFK` / `SUP_COD` (supplier, 178 distinct, a useful proxy for manufacturing style), `DISPLAY_COD` / `DISPLAY_DES` (in-store display fixture: Module, Panel, Party Panel…), `DISTRIB_CENTRAL`, `PROD_ONLINE`, `OPEN_REFERENCE`, `RECLASSIFIED` / `RECLASS_REASON` / `RECLASS_CODE` (8% reclassified, e.g. *Sales below budget*, *Season setting error*), `ECI_APPAREL`, `NIGHT_BAG`, `CFS_SFK`.

### 3.3 Columns with no information

These columns are empty or constant and can be dropped:

- **Empty / almost empty (< 0.1% filled):** `PROD_DAT_CRI`, `PHASE`, `PHASE_COD`, `DISPLAY_CLR_COD`, `DISPLAY_CLR`, `FLAG_FACT`, `PONTAMETALICA_COD`, `PONTAMETALICA_DES`, `CLR_TON_DES`, `CLR_BAS_DES`, `DIMENSION_CONSUMABLE`, `STORE_CONCEPT`, `STATUS_ARTICLE`, `PRODUCTION_TYPE`, `STORE_LOCATION`, `INFO_TXT`, `AVERAGE_STORE`, `CAPACITY`, `END_DATE`, `ITEM_GROUP`, `SECONDARY_DISPLAY`.
- **Constant:** `PROD_SIT` (S), `TAR_COD` / `TAR_DES` (Finished Product), `PERMANENT`, `PERMANENT_ASIS` (N), `OUTLET`, `OUTLET_GEN` (No), `ACTIVE_RECORD` (1), `WITH_ORDER` (1), `CAT_ACTIVE` (Y).
- **Exact duplicates of another column:** `*_DES_PT` (= `_EN`), `PRICE_BASE_W_VAT_ASWAS`, `WEEKS_TARGET` (= `ITEM_LIFE_CYCLE_WEEK`), `FASHIONTYPE_ASIS`, `THEME_ASIS`, `THEME_CODE_ASIS`, `PROD_TYPOLOGY_ORIG_ASIS`, `PURCHASE_BET_ORIG_ASIS`, `ONL_*_DATE` vs `ONL_*_DATE_DT`.

---

## 4. `data/images/`

| Property | Value |
|---|---|
| Files | 9,496 (9,489 `.jpg`, 6 `.jpeg`, 1 `.png`), flat folder, ≈278 MB |
| Format | Mostly 640×640 RGB studio packshots on a white background |
| Naming | `<last part of PROG_IMAGE>.<ext>`, e.g. `PROG_IMAGE = /241/52/218792_PM_1` → `218792_PM_1.jpg` |
| Link to items | Take the file stem (name without extension) of `PROG_IMAGE` and match it to the file name, ignoring the extension. Do **not** build file names from `PROD_CLR`: some are irregular (`2089811OW_1`, `140428CT__1`, `225874_1`) |

**Coverage (colourway level):** 9,455 of 10,555 colourways (89.6%) have an image file.

| Category | With image |
|---|---|
| Footwear | 97.0% |
| Wallet | 95.4% |
| Hand Bag | 91.9% |
| Jewellery | 89.2% |
| Apparel | 83.4% |

- 64 files are not referenced by any colourway (e.g. `192346_EC_1`, `208203_EC_1`). Most look like images of items that were later re-coded (the old code before the `*_EQUIV` mapping).
- About 1,100 colourways have no image. They need a similarity method that works on tabular data only.

## 5. Image-label mismatches

The professor warned that **some images do not match their CSV row** (e.g. the row says necklace but the image shows earrings). Checks so far:

1. **Visual spot-check:** 80 random images (16 per category) compared with their `CAT_DES_EN` / `GFA_DES_EN`.
   - 1 clear mismatch: `221631_HM` is labelled *Rings*, but the image shows a bangle or bracelet.
   - 2 photos are not studio packshots (`215514_PK`, `219039_BL`: an amateur photo on a background or table).
   - This suggests the mismatch rate is in the low single-digit %. That is still roughly 100–300 items across the dataset, enough to damage the nearest-neighbour results.
2. **Byte-identical files:** 20 groups (40 files) where different codes share exactly the same image.
   - **Same photo for different colours:** e.g. `208038_BL` and `208038_PU` (Dress) and `212067_BL` / `212067_SV` (Skirt). The image shows the wrong colour for at least one of them.
   - **Old and new code of the same item:** e.g. `2141221BK_1` / `214122_BK_1`. These are harmless duplicates.
   - **One image across different models:** e.g. `209749_WT_1` / `214038_WT_1`.

**Types of mismatch to detect:**

| Type | Example | How to detect |
|---|---|---|
| Wrong product type | Row says necklace, image shows earrings | Zero-shot CLIP classification of the image into `GFA_DES_EN`, compared with the label |
| Wrong colour | Shared photo across colourways | Dominant image colour (LAB) vs `CLR_DES`, plus duplicate image hashes |
| Wrong item entirely | Image belongs to another model | Duplicate or near-duplicate hashes across different `PROD_REF`. The item's visual neighbours mostly belong to another family |
| Not usable | Amateur photo, multiple items, missing | Background/whiteness heuristic, image size, missing file |

**Results (2026-10-09):** the automatic version of these checks is in Phase 1b (`src/phase1b_image_audit.py`). It found 24 photos of another item type, 132 of another colour and 54 photos shared with another colour of the same model. See `outputs/phase1b/image_audit_report.md` and [PROGRESS.md](PROGRESS.md).

**Handling rule:** never silently drop or relabel. Flag each item in an `image_audit.parquet` with the mismatch type and a confidence score. If an item's image is unreliable, compute its similarity **without the visual block** (tabular data only), and list the item in a data-quality report for the business.

## 6. Data-quality notes (CSV)

- **Encoding:** the file is valid UTF-8 and accents are correct (e.g. `Online + Lojas Imagem Próprias`). Read it with `encoding="utf-8"`. A Windows terminal may show `�`, but that is only the display. There are hidden non-breaking spaces, e.g. `GFA_DES_EN = 'Raincoat\xa0'`, so replace `\xa0` (non-breaking space) and strip all text columns.
- **Mixed casing / inconsistent labels:** `Golden Basics` vs `GOLDEN BASICS`, `TEXTILE` vs `Footwear` in `L1_DES`, `Transluncent` (typo). Normalise case before encoding.
- **Leading whitespace** in `COMPOSITION` (`" Zinc"`). Materials are separated by double spaces.
- **Dates come in several formats:** `YYYYMMDD` ints, `YYYYMMDD.0` floats (they have NaNs) and ISO strings.
- **`"UNDEFINED"` / `"Not Applicable"` / `"Without Block"` / `"Others"`** are placeholders for missing data and should be treated as missing, not as real categories.
- **Category-specific sparsity:** attributes such as `MATERIAL`, `FINISHING`, `OUTFIT` and `SHAPE` are only meaningful inside one category. Model per category, or encode "not applicable" explicitly.
- **Code reuse across levels:** `GFA_COD` / `GFS_COD` numbers repeat across categories. Always combine them with `CAT_COD`.

---

## 7. Suggested preparation for the similarity model

1. **Collapse to colourway level:** group `df_product` by `PROD_CLR_EQUIV` and keep the attributes that are constant within the group (everything except size, barcode and SKU keys). This gives about 10,555 items.
2. **Join sales** on `PROD_CLR_EQUIV` (left join; 370 items have no sales).
3. **Build three feature blocks**, kept separate so the model can explain its matches:
   - *Categorical/structural:* CAT › GFA › GFS, `CATEGORY_MATRIX`, `CLR_DES`, `CLR_TYPE`, `FINISHING`, `MATERIAL`, tokenised `COMPOSITION`, `PRINT_TYPE`, `DIMENSION`, `OUTFIT`, `THEME`, `FASHIONTYPE`, `PROD_SEG`, price band.
   - *Visual:* embeddings from the image file matched via `PROG_IMAGE` (only for images that pass the §5 audit).
   - *Commercial (optional, for ranking or validation):* log `SALES_QTY`, realised price, `STORE_DATE_FINAL` / season, `TYPOLOGY`, `PUR_BET`.
4. **Restrict candidates** to the same `CAT_DES_EN` (or `GFA_DES_EN`), so that matches are like-for-like substitutes. Then report which attribute blocks drove each match, as the expected output requires.
