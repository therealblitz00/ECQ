# Sprint 1 Preprocessing: Review & Actionable Improvement Roadmap

> **Status update (2026-10-09):** **BUG-001**, **BUG-002** and **BUG-003** are fixed in `src/sprint1_preprocess.py`. The batches were regenerated. The other findings are still open.

> **Document Purpose:** This document provides a complete technical audit of the current Sprint 1 preprocessing pipeline (`src/sprint1_preprocess.py`) and serves as an actionable specification for team members and subsequent AI coding agents to implement corrections.

---

## 1. Executive Summary

- **Context:** Parfois similarity detection case study (MADSAD Master's, FEP).
- **Audit Target:** `src/sprint1_preprocess.py`, supporting documentation, and output artifacts.
- **Current Pipeline Status:** **PARTIALLY COMPLETE (~55%) & CONTAINS DATA QUALITY VULNERABILITIES**.
- **Readiness for Phase 2 (Feature Engineering):** **NOT READY**. While the human-in-the-loop review workflow is functional, merging and dataset preparation suffer from aggregation bugs, leakage of unresolved flags into the clean output, unparsed dates, and zero numerical validation.

### High-Level Diagnosis
| Area | Status | Key Observation |
|---|---|---|
| **Human Review Workflow** | ✅ Strong | Deterministic 4-way balanced split, HTML contact sheets, and change logging work well. |
| **Raw Data Integrity** | ✅ Strong | `data/csv/` and `data/images/` remain strictly immutable. |
| **Aggregation Logic** | ❌ Critical Flaw (P0) | `grouped.first()` silently ignores attribute variations across SKUs within the same colourway (prices vary in 8 items, compositions in 836, images in 29). |
| **Merge & Clean Filtering** | ❌ Critical Flaw (P0) | Items with unresolved defects (`pending`) and reviewer doubts (`team_review`) are retained in `items_clean.parquet`. |
| **Review Fix Parser** | ⚠️ Bug (P1) | Semicolon delimiter in `fixes` collides with `COMPOSITION` semicolon separators, corrupting material edits. |
| **Numerical Validation** | ⚠️ Gap (P1) | Zero validation on prices, landed costs, or sales totals; known zero-division risk in realised price is unhandled. |
| **Schema Hygiene** | ⚠️ Gap (P1) | 33 empty/constant columns and 20 internal audit fields are dumped into `items_clean.parquet`. |

---

## 2. Master Comparison Matrix

| Stage | Expected Behaviour | Current Implementation | Status | Priority |
|---|---|---|---|---|
| **1. Data Inventory** | Profile files, schemas, formats, disk footprint | Loads CSVs, checks image stems, prints counts | PARTIALLY IMPLEMENTED | P2 |
| **2. Dataset Profiling** | Full summary stats for `df_product` and `df_sales` | Profiles missingness for `df_product` only | IMPLEMENTED BUT WEAK | P2 |
| **3. Granularity & Keys** | Collapse SKUs to colourway (`PROD_CLR_EQUIV`) | Strips size from `PROD_DES`; groups by colourway | COMPLETE | – |
| **4. Key Validation** | Assert uniqueness of surrogate and business keys | Validates key uniqueness across both CSVs | COMPLETE | – |
| **5. Hierarchy Mapping** | Map model → colourway → SKU; handle re-coded items | Tracks `n_skus`, sizes string; re-coded links unverified | PARTIALLY IMPLEMENTED | P2 |
| **6. Missing Values** | Impute sentinels (`UNDEFINED`, `Not Applicable`); contextualize nulls | Detects nulls in `df_product`; sentinels kept as strings | IMPLEMENTED BUT WEAK | P1 |
| **7. Duplicate Analysis** | Check row duplicates, duplicate keys, duplicate images | Exact row duplicates and MD5 image duplicates checked | COMPLETE | – |
| **8. In-Key Conflict Analysis**| Assert non-size attributes are uniform within colourway | Blind `grouped.first()` without invariance check | **MISSING** | **P0** |
| **9. Categorical Checks** | Normalize casing, typos; validate hierarchy trees | Whitespace/accent cleanup done; casing/hierarchy unhandled | IMPLEMENTED BUT INCONSISTENT | P1 |
| **10. Numerical Validation** | Validate price boundaries, positive sales, safe margins | Only checks `SALES_QTY == 0`; ignores prices/costs | **MISSING** | **P1** |
| **11. Cross-Dataset Join** | Product ↔ Sales and Product ↔ Image referential checks | Fully validated for product-to-image and sales | COMPLETE | – |
| **12. Orphan Detection** | Unreferenced sales, products without images, orphan disk files | Product-to-image/sales done; 64 orphan disk images ignored | PARTIALLY IMPLEMENTED | P2 |
| **13. Clean Master Integration**| Produce clean, typed, curated `items_clean.parquet` | Dumps raw columns; leaks `pending`/`team_review` items | **IMPLEMENTED BUT INCONSISTENT** | **P0** |
| **14. Image Inventory** | Audit file extensions, dimensions, aspect ratios, file sizes | Checks stem presence only; no dimensions audited | IMPLEMENTED BUT WEAK | P2 |
| **15. Image ↔ Item Mapping**| Extract stem from `PROG_IMAGE`; handle multi-images | Stem extraction works well; multi-image flattened | PARTIALLY IMPLEMENTED | P2 |
| **16. Image Quality** | Verify headers, corrupted files, blank/truncated images | Pillow `verify()` catches headers (1 corrupt file) | IMPLEMENTED BUT WEAK | P2 |
| **17. Exact Image Duplicates** | Cryptographic hash check for shared images | MD5 hash flags 100 items sharing image files | COMPLETE | – |
| **18. Perceptual Duplicates** | Perceptual hashes (pHash) for re-compressed images | Not in `sprint1_preprocess.py` | DEFERRED (Phase 1b) | – |
| **19. Raw Data Integrity** | Raw data strictly read-only; changes audited | Fully respected; `changes_log.csv` records edits | COMPLETE | – |
| **20. Reproducibility** | Deterministic batch split and execution order | Category-stratified greedy partition works reliably | COMPLETE | – |
| **21. Output Validation** | Assert output schemas, row counts, and null constraints | Validates review files; no check on final Parquet | IMPLEMENTED BUT WEAK | P1 |
| **22. Issue Reporting** | Structured Markdown and CSV audit logs | `check_report.md`, `items_checked.csv`, logs generated | COMPLETE | – |

---

## 3. Actionable Technical Findings

Each item below contains the diagnosis, evidence, and exact instructions for the implementation agent.

---

### Finding `BUG-001` (Priority: P0 — Critical)
#### Title: In-Key Attribute Conflict Ignored During SKU Collapsing
- **Area:** Granularity & Aggregation (`build_items`)
- **Location:** `src/sprint1_preprocess.py`, lines 123–125
- **Problem:**
  ```python
  grouped = prod.groupby(KEY, sort=True)
  items = grouped.first()
  ```
  `grouped.first()` arbitrarily chooses the first row of each colourway group. An empirical check of `df_product.csv` shows that non-size attributes are **not constant** within `PROD_CLR_EQUIV`:
  - `PRICE_BASE_W_VAT` varies in **8 colourways** (e.g. `214388_DM` has sizes at €89.99 and others at €99.99; re-coded item `216766_BG` groups rows at €15.99 and €12.99).
  - `COMPOSITION` varies in **836 colourways**.
  - `PROG_IMAGE` varies in **29 colourways**.
  - `CLR_COD` and `CLR_DES` vary in **3 re-coded colourways** (`212653_WT` combines White and Ecru; `213555_RD` combines Burgundy and Red).
- **Impact:** Non-deterministic silent corruption of product prices, material descriptions, and canonical color codes.
- **Action for Implementation Agent:**
  1. In `build_items()`, replace blind `.first()` with explicit aggregation rules:
     - For columns that must be strictly constant (`CAT_COD`, `GFA_COD`, `GFS_COD`, `CLR_COD`, `CLR_DES`), assert or check cardinality.
     - Detect variations and add issue codes:
       - `VAR_PRICE_IN_KEY` (if `PRICE_BASE_W_VAT.nunique() > 1`): record min, max, and mode; flag for human review.
       - `VAR_IMAGE_IN_KEY` (if `PROG_IMAGE.nunique() > 1`): record distinct images; flag for human review.
       - `VAR_COMPOSITION_IN_KEY` (if `COMPOSITION.nunique() > 1`): take the longest/most complete composition string.
       - `VAR_CLR_IN_KEY` (if re-coded item maps conflicting colors): preserve the canonical color matching `PROD_CLR_EQUIV`.
  2. Elevate `VAR_PRICE_IN_KEY`, `VAR_IMAGE_IN_KEY`, and `VAR_CLR_IN_KEY` to `high` priority in `ISSUES`.
- **Verification Criteria:**
  - Running `python src/sprint1_preprocess.py check` reports all in-key attribute conflicts in `check_report.md`.
  - No price or color variation is silently swallowed.

---

### Finding `BUG-002` (Priority: P0 — Critical)
#### Title: Leakage of Unreviewed (`pending`) and Uncertain (`team_review`) Records into `items_clean.parquet`
- **Area:** Batch Merging & Clean Dataset Generation (`merge_batches`)
- **Location:** `src/sprint1_preprocess.py`, lines 462–463, 484–485
- **Problem:**
  ```python
  items["review_status"] = items["review_status"].replace("", pd.NA).fillna("pending")
  ...
  clean = items[items["review_status"] != "discard"].drop(columns=["source_file"])
  clean.to_parquet(PROCESSED_DIR / "items_clean.parquet", index=False)
  ```
  Only items marked `"discard"` are excluded. Any flagged item that was left unreviewed by a team member (`pending`) or flagged with uncertainty (`team_review`) is **retained in `items_clean.parquet`**!
- **Impact:** The final clean dataset contains corrupt images, conflicting colors, and invalid paths that were never resolved, polluting downstream feature engineering.
- **Action for Implementation Agent:**
  1. Redefine the clean filter in `merge_batches()`:
     ```python
     clean = items[items["review_status"].isin(["ok", "fix"])].drop(columns=["source_file"])
     ```
  2. Output explicit separate summary logs:
     - `outputs/sprint1/team_review.csv`: contains items where `review_status == "team_review"`.
     - `outputs/sprint1/pending_review.csv`: contains items where `review_status == "pending"`.
     - `outputs/sprint1/discarded.csv`: contains items where `review_status == "discard"`.
  3. Print a clear terminal warning stating how many items remain unreviewed or pending team discussion and are therefore excluded from `items_clean.parquet`.
- **Verification Criteria:**
  - `clean["review_status"].unique()` contains **only** `['ok', 'fix']`.
  - No `pending` or `team_review` rows exist in `data/processed/items_clean.parquet`.

---

### Finding `BUG-003` (Priority: P1 — High)
#### Title: Semicolon Delimiter Collision in Batch Fix Parser
- **Area:** Batch Merging (`_parse_fixes`)
- **Location:** `src/sprint1_preprocess.py`, lines 424–431
- **Problem:**
  `_parse_fixes()` splits on `;` (`for part in str(text).split(";"):`).
  In `load_data()`, materials in `COMPOSITION` are formatted with `; ` (e.g. `Cotton; Polyester`).
  If a reviewer writes:
  `COMPOSITION=Cotton; Polyester`
  The string is split into `COMPOSITION=Cotton` and `Polyester`. Since `Polyester` has no `=`, it is silently dropped!
- **Impact:** Fixes applied to `COMPOSITION` or descriptions containing semicolons are corrupted or truncated.
- **Action for Implementation Agent:**
  1. Support `&&` or `|` as the multi-fix separator while maintaining backwards compatibility:
     ```python
     def _parse_fixes(text: str) -> list[tuple[str, str]]:
         out = []
         # Split on '&&' first if present; fallback to ';' only when '=' is present in each token
         parts = [p.strip() for p in re.split(r"\s*&&\s*", str(text)) if p.strip()]
         if len(parts) == 1 and ";" in parts[0] and "=" in parts[0]:
             # Only split by ';' if multiple assignments exist with '='
             candidate_parts = parts[0].split(";")
             if all("=" in c for c in candidate_parts if c.strip()):
                 parts = candidate_parts
         for part in parts:
             if "=" in part:
                 col, val = part.split("=", 1)
                 out.append((col.strip(), val.strip()))
         return out
     ```
  2. Update `SPRINT1_GUIDE.md` §How to review to document `&&` as the recommended delimiter for multi-column fixes.
- **Verification Criteria:**
  - Unit test `_parse_fixes("COMPOSITION=Cotton; Polyester && GFA_DES_EN=Trousers")` returns `[('COMPOSITION', 'Cotton; Polyester'), ('GFA_DES_EN', 'Trousers')]`.

---

### Finding `GAP-001` (Priority: P1 — High)
#### Title: Absence of Numerical, Pricing, and Financial Sanity Validation
- **Area:** Data Quality / Validation (`check_sales` & new `check_numerical`)
- **Location:** `src/sprint1_preprocess.py`, lines 261–264
- **Problem:**
  Only `SALES_QTY == 0` is checked.
  - No checks on negative prices (`PRICE_BASE_W_VAT < 0`), landed cost anomalies (`PRICE_LCP > PRICE_BASE_W_VAT`), or negative sales values (`SALES_AMT_FX_RATE < 0`).
  - Zero-division risk: As noted in `data_description.md` §2, items like `213332_GD` have `SALES_QTY == 0` but positive `SALES_AMT_FX_RATE`. Calculating `realised_price = SALES_AMT / SALES_QTY` yields `inf`.
  - Required baseline features from Roadmap (`log_sales_qty`, `realised_price`) are missing.
- **Impact:** Numerical defects and division-by-zero runtime exceptions in downstream modeling.
- **Action for Implementation Agent:**
  1. Implement `check_numerical(items)`:
     - Flag `PRICE_INVALID` if `PRICE_BASE_W_VAT <= 0` or `PRICE_BASE_W_VAT.isna()`.
     - Flag `PRICE_COST_ANOMALY` if `PRICE_LCP > PRICE_BASE_W_VAT`.
     - Flag `SALES_AMT_ANOMALY` if `SALES_AMT_FX_RATE < 0`.
     - Flag `ZERO_QTY_WITH_AMT` if `SALES_QTY == 0` and `SALES_AMT_FX_RATE > 0`.
  2. Compute guarded features during `build_items()`:
     ```python
     items["log_sales_qty"] = np.log1p(items["SALES_QTY"].clip(lower=0).fillna(0))
     items["realised_price"] = np.where(
         items["SALES_QTY"] > 0,
         (items["SALES_AMT_FX_RATE"] / items["SALES_QTY"]).round(2),
         np.nan
     )
     ```
- **Verification Criteria:**
  - `check_report.md` includes numerical validation tables.
  - Zero-division is prevented; `np.isinf(items["realised_price"]).any()` is `False`.

---

### Finding `GAP-002` (Priority: P1 — High)
#### Title: Missing Placeholder Sentinel Imputation and Categorical Case Normalization
- **Area:** Ingestion & Cleaning (`load_data`)
- **Location:** `src/sprint1_preprocess.py`, lines 71–83
- **Problem:**
  `ROADMAP.md` Phase 1 checklist items explicitly state:
  - `- [ ] Treat placeholders (UNDEFINED, Not Applicable, Without Block, Others) as missing values.`
  - `- [ ] Normalise casing (THEME, L1_DES…).`
  Neither is implemented. Strings like `"UNDEFINED"` and `"Not Applicable"` remain treated as genuine distinct categories, and casing variations like `"Golden Basics"` vs `"GOLDEN BASICS"` create artificial split categories.
- **Impact:** Distorted feature vectors and noisy categorical embeddings in Phase 2a.
- **Action for Implementation Agent:**
  1. In `load_data()`, replace sentinels with `pd.NA`:
     ```python
     SENTINELS = {"UNDEFINED", "Not Applicable", "NOT APPLICABLE", "Without Block", "WITHOUT BLOCK", "None", "nan"}
     # Replace for categorical columns
     ```
  2. Normalize text casing for unstructured/free-text category descriptions (`THEME`, `L1_DES`, `L2_DES`, `L3_DES`, `L4_DES`, `PRINT_TYPE`):
     - Strip whitespace, convert to consistent Title Case or Upper Case, and map known typos (e.g. `"Transluncent"` → `"Translucent"`).
- **Verification Criteria:**
  - `"UNDEFINED"` no longer appears in `items["DIMENSION"].unique()`.
  - Number of unique categories in `THEME` decreases as case-variant duplicates are merged.

---

### Finding `GAP-003` (Priority: P1 — High)
#### Title: Store and Exposition Dates Unparsed
- **Area:** Ingestion & Cleaning (`load_data`)
- **Location:** `src/sprint1_preprocess.py`
- **Problem:**
  `STORE_DATE_FINAL` (the problem's Exposition Date) and planned/real dates exist in mixed formats: integers (`20240115`), floats with NaNs (`20240115.0`), and ISO strings. They are never parsed into datetime objects.
- **Impact:** Downstream temporal filtering, recency weighting, and season-based trend validation are blocked.
- **Action for Implementation Agent:**
  1. Add date parsing helper:
     ```python
     def parse_parfois_date(s: pd.Series) -> pd.Series:
         # Convert float/int YYYYMMDD to string, then parse
         clean = s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
         return pd.to_datetime(clean, format="%Y%m%d", errors="coerce")
     ```
  2. Parse `STORE_DATE_FINAL`, `STORE_DATE_PLANNED`, and `STORE_DATE_REAL`.
  3. Validate that `STORE_DATE_FINAL` falls within plausible business years (2021–2025).
- **Verification Criteria:**
  - `pd.api.types.is_datetime64_any_dtype(items["STORE_DATE_FINAL"])` is `True`.

---

### Finding `INC-001` (Priority: P1 — High)
#### Title: Uncurated Output Schema and Technical Flag Contamination
- **Area:** Output Contract (`merge_batches`)
- **Location:** `src/sprint1_preprocess.py`, lines 485–488
- **Problem:**
  `items_clean.parquet` retains all 141 raw columns from `df_product.csv` (including 33 empty/constant columns like `INFO_TXT`, `DIMENSION_CONSUMABLE`, `FLAG_FACT`) as well as 20 internal audit fields (`desc_type_share`, `clr_expected_name`, `img_stem`, `issues`, etc.).
- **Impact:** Bloated file size, schema instability across platforms, and leakage of technical debugging flags into machine learning feature matrices.
- **Action for Implementation Agent:**
  1. Define a strict output column contract for `items_clean.parquet`:
     - **Drop 33 empty/constant columns** documented in `data_description.md` §3.3.
     - **Drop duplicate columns** (`*_DES_PT`, `PRICE_BASE_W_VAT_ASWAS`, `WEEKS_TARGET`, `THEME_ASIS`).
     - **Drop internal audit flags** (`desc_type_word`, `desc_type_share`, `clr_expected_name`, `desc_colour_status`, `issues`, `priority`).
     - **Retain core identifiers, cleaned attributes, sales metrics, and image links**: `PROD_CLR_EQUIV`, `PROD_REF`, `PROD_DES_BASE`, `CAT_COD`, `CAT_DES_EN`, `GFA_COD`, `GFA_DES_EN`, `GFS_COD`, `GFS_DES_EN`, `CATEGORY_MATRIX`, `CLR_COD`, `CLR_DES`, `CLR_TYPE`, `COMPOSITION`, `MATERIAL`, `FINISHING`, `OUTFIT`, `PRINT_TYPE`, `THEME`, `FASHIONTYPE`, `PROD_SEG`, `PRICE_BASE_W_VAT`, `PRICE_OUTLET_W_VAT`, `PRICE_LCP`, `STORE_DATE_FINAL`, `SALES_QTY`, `SALES_AMT_FX_RATE`, `log_sales_qty`, `realised_price`, `PROG_IMAGE`, `img_file`, `has_image`, `sizes`, `n_skus`, `review_status`, `reviewer`.
- **Verification Criteria:**
  - `items_clean.parquet` contains ~35 curated, typed columns instead of 160+ columns.
  - File size is optimized and strictly documented.

---

### Finding `INC-002` (Priority: P1 — High)
#### Title: In-Place Fixes Do Not Re-Validate Derived Fields or Clear Defect Codes
- **Area:** Batch Merging (`merge_batches`)
- **Location:** `src/sprint1_preprocess.py`, line 472
- **Problem:**
  When a reviewer corrects an attribute (e.g. `CLR_DES=Black` to resolve `CLR_CONFLICT_DESC`), `items.at[idx, col] = val` updates the value, but **leaves `issues` and `priority` unchanged**. The clean table retains stale flags like `CLR_CONFLICT_DESC` on fixed items!
- **Impact:** Defect reports miscount remaining errors; reviewers see items they already corrected still marked as broken.
- **Action for Implementation Agent:**
  1. In `merge_batches()`, after applying fixes, run a targeted re-validation pass:
     - Clear the resolved issue code from `issues`.
     - Recompute `priority` based on remaining issues.
     - If all issues are resolved, set `priority = "none"`.
- **Verification Criteria:**
  - An item fixed via `CLR_DES=...` no longer contains `CLR_CONFLICT_DESC` in `issues`.

---

### Finding `WEA-001` (Priority: P2 — Medium)
#### Title: Image Header Verification Misses Physical Dimensions and Color Spaces
- **Area:** Image Quality (`check_images`)
- **Location:** `src/sprint1_preprocess.py`, lines 174–176
- **Problem:**
  `PIL.Image.verify()` only reads file headers to check for corruption. It does not inspect physical image width, height, aspect ratio, or channel mode (RGB vs RGBA vs Grayscale). Non-packshot images (e.g. tiny 50x50 thumbnails or extreme aspect ratios) pass without detection.
- **Impact:** Downstream computer vision models (CLIP / ResNet) may fail or distort embeddings on irregular aspect ratios.
- **Action for Implementation Agent:**
  1. During the image file audit loop, extract `im.size` (width, height) and `im.mode`:
     ```python
     with Image.open(path) as im:
         im.verify()
         # Re-open or inspect size/format/mode
         w, h = im.size
         mode = im.mode
     ```
  2. Flag `IMG_DIM_ANOMALY` if image is not square or dimensions deviate significantly from expected standard (640×640).
- **Verification Criteria:**
  - `check_report.md` reports image dimension summary statistics (e.g. min, median, max width/height).

---

### Finding `GAP-004` (Priority: P2 — Medium)
#### Title: 64 Orphan Image Files on Disk Unaccounted For
- **Area:** Orphan Detection (`check_images`)
- **Location:** `src/sprint1_preprocess.py`, lines 146–150
- **Problem:**
  `check_images()` checks which product rows link to existing image files, but never performs the reverse check: identifying files in `data/images/` that are **not referenced by any product row**. As documented in `data_description.md` §4, 64 such files exist.
- **Impact:** Lack of catalog accounting for stranded assets; uncertainty whether these files represent obsolete items or unlinked products.
- **Action for Implementation Agent:**
  1. Add reverse orphan check:
     ```python
     disk_files = set(os.listdir(IMG_DIR))
     referenced_files = set(items["img_file"].dropna())
     orphan_images = disk_files - referenced_files
     ```
  2. Export `outputs/sprint1/orphan_images.csv` and report the count in `check_report.md`.
- **Verification Criteria:**
  - `outputs/sprint1/orphan_images.csv` is generated with exactly the unreferenced image filenames.

---

### Finding `WEA-002` (Priority: P2 — Medium)
#### Title: Brittle Singularization Logic in Item Type Consistency Check
- **Area:** Categorical Checks (`check_type_consistency`)
- **Location:** `src/sprint1_preprocess.py`, lines 224–227
- **Problem:**
  `_singular()` naively removes a trailing 's':
  `w[:-1] if len(w) > 3 and w.endswith("s") else w`
  This truncates words ending in double-s: `"Dress"` → `"dres"`, `"Glasses"` → `"glasse"`, `"Cross"` → `"cros"`.
- **Impact:** False-positive linguistic mismatches.
- **Action for Implementation Agent:**
  1. Replace with proper regex or exception guard:
     ```python
     def _singular(word: str) -> str:
         w = word.lower().strip()
         if w.endswith("ss") or len(w) <= 3:
             return w
         if w.endswith("ies"):
             return w[:-3] + "y"
         if w.endswith("s"):
             return w[:-1]
         return w
     ```
- **Verification Criteria:**
  - `_singular("Dress") == "dress"`, `_singular("Glasses") == "glasses"`, `_singular("Earrings") == "earring"`.

---

### Finding `RED-001` (Priority: P2 — Medium)
#### Title: `CLR_NOT_IN_DESC` Escalated to Medium Priority Causes Reviewer Fatigue
- **Area:** Review Queue Prioritization
- **Location:** `src/sprint1_preprocess.py`, line 51
- **Problem:**
  `CLR_NOT_IN_DESC` is assigned `medium` severity. In fashion retail, base product names often omit colour (e.g. `Earrings MELROSE`) because colour is captured in `CLR_DES`. 25 items get an orange border and priority queue placement solely for this reason.
- **Impact:** Wastes reviewer time on legitimate marketing titles rather than genuine errors.
- **Action for Implementation Agent:**
  1. Downgrade `CLR_NOT_IN_DESC` from `medium` to `info` in `ISSUES`.
- **Verification Criteria:**
  - `CLR_NOT_IN_DESC` items receive `info` priority and do not clutter the top of review contact sheets.

---

### Finding `OPP-001` (Priority: P2 — Medium)
#### Title: Asymmetric Profiling Omits Sales Dataset
- **Area:** Dataset Profiling (`missing_report`)
- **Location:** `src/sprint1_preprocess.py`, line 280
- **Problem:**
  `missing_report(prod)` is called, but `sales` is never profiled for missing values, distributions, zero counts, or summary statistics.
- **Action for Implementation Agent:**
  1. Extend `missing_report()` to generate profile summaries for both `df_product` and `df_sales`.
- **Verification Criteria:**
  - `outputs/sprint1/missing_values_sales.csv` (or consolidated profiling report) is generated.

---

### Finding `OPP-002` (Priority: P3 — Low)
#### Title: Monolithic Script Hinders Modular Testing and Maintenance
- **Area:** Architecture & Engineering
- **Location:** `src/sprint1_preprocess.py`
- **Problem:**
  All logic (loading, validation, batching, HTML rendering, merging, CLI) resides in one 531-line script without unit tests (`pytest`).
- **Action for Implementation Agent:**
  1. Refactor `src/sprint1_preprocess.py` into a package `src/sprint1_preprocess/`:
     - `loader.py`: CSV loading, sentinel handling, date parsing.
     - `builder.py`: SKU collapsing and variance assertion.
     - `validators.py`: image, color, type, and numerical checks.
     - `batcher.py`: greedy split and HTML sheet generation.
     - `merger.py`: fix parsing, validation, clean export.
     - `cli.py`: main entry point.
  2. Add unit test suite in `tests/test_sprint1_preprocess.py`.
- **Verification Criteria:**
  - `pytest` passes on all validation functions.

---

## 4. Master Actionable Roadmap Table

| ID | Priority | Category | Area | Description / Finding | Recommended Action | Phase |
|---|---|---|---|---|---|---|
| **BUG-001** | **P0** | Bug | Granularity | In-key attribute conflicts silently ignored by `grouped.first()` | Implement in-key variance validation; flag price/image/color conflicts | Phase 1 |
| **BUG-002** | **P0** | Bug | Master | Unreviewed `pending` and `team_review` items leak into `items_clean.parquet` | Exclude `pending`/`team_review` from clean Parquet; route to CSVs | Phase 1 |
| **BUG-003** | **P1** | Bug | Batching | Semicolon collision in `_parse_fixes` corrupts `COMPOSITION` | Support `&&` multi-fix delimiter and robust token parser | Phase 1 |
| **GAP-001** | **P1** | Gap | Validation | No numerical validation on prices, costs, or sales; zero division risk | Implement numerical bounds checks and guarded `realised_price` | Phase 2 |
| **GAP-002** | **P1** | Gap | Cleaning | Missing placeholder sentinels (`UNDEFINED`, etc.) and unnormalized casing | Convert sentinels to `NA`; fold categorical casing | Phase 2 |
| **GAP-003** | **P1** | Gap | Cleaning | Date columns remain unparsed in raw formats | Implement multi-format date parser for store and launch dates | Phase 2 |
| **INC-001** | **P1** | Inconsistency | Output | 33 dead columns and 20 internal audit fields in `items_clean.parquet` | Define strict output schema contract; prune dead/audit columns | Phase 2 |
| **INC-002** | **P1** | Inconsistency | Master | In-place fixes do not re-validate flags or clear defect codes | Re-run validation pass post-merge to clear resolved issue flags | Phase 3 |
| **WEA-001** | **P2** | Weakness | Images | Image verification ignores physical dimensions and channels | Capture width, height, and color channels during image loop | Phase 3 |
| **GAP-004** | **P2** | Gap | Orphans | 64 orphan image files on disk uninspected and unlogged | Add disk-to-product reverse orphan check and export CSV | Phase 3 |
| **WEA-002** | **P2** | Weakness | Logic | Brittle singularization in `_singular()` mutilates words like "Dress" | Guard words ending in "ss" against naive suffix stripping | Phase 3 |
| **RED-001** | **P2** | Redundancy | Review | `CLR_NOT_IN_DESC` flagged as medium priority causes reviewer fatigue | Downgrade `CLR_NOT_IN_DESC` severity to `info` | Phase 3 |
| **OPP-001** | **P2** | Opportunity | Profiling | Asymmetric profiling omits `df_sales.csv` | Extend profiling to sales table and numeric distributions | Phase 4 |
| **OPP-002** | **P3** | Opportunity | Structure | Monolithic script lacks modular structure and `pytest` suite | Refactor into `src/sprint1_preprocess/` package with tests | Phase 4 |

---

## 5. Phased Implementation Guide for Future Agents

Future agents assigned to implement these improvements must follow this strict dependency order:

```text
================================================================================
PHASE 1: CRITICAL DATA CORRECTNESS & PIPELINE HARDENING (P0 / P1)
================================================================================
1. [BUG-001] In build_items():
   - Check variance for PRICE_BASE_W_VAT, PROG_IMAGE, CLR_COD across SKUs.
   - Flag VAR_PRICE_IN_KEY, VAR_IMAGE_IN_KEY, VAR_CLR_IN_KEY as high severity.
2. [BUG-002] In merge_batches():
   - Set clean = items[items["review_status"].isin(["ok", "fix"])].
   - Ensure pending and team_review records never reach items_clean.parquet.
3. [BUG-003] In _parse_fixes():
   - Support '&&' delimiter so COMPOSITION fixes with semicolons are not broken.

================================================================================
PHASE 2: DATA HYGIENE & CLEAN CONTRACT (P1)
================================================================================
4. [GAP-002] In load_data():
   - Impute sentinels ('UNDEFINED', 'Not Applicable', 'Without Block') to pd.NA.
   - Normalize casing on THEME, L1_DES..L4_DES.
5. [GAP-003] In load_data() / build_items():
   - Parse STORE_DATE_FINAL, STORE_DATE_PLANNED, STORE_DATE_REAL to datetime.
6. [GAP-001] In build_items() & check_sales():
   - Add numerical validation checks for prices, landed costs, and sales.
   - Compute log_sales_qty = log1p(SALES_QTY) and guarded realised_price.
7. [INC-001] In merge_batches():
   - Prune the 33 empty/constant columns and internal audit flags from clean output.

================================================================================
PHASE 3: PIPELINE CONSISTENCY & ASSET AUDITING (P2)
================================================================================
8. [INC-002] In merge_batches():
   - Add post-merge re-validation pass to clear resolved defect flags.
9. [WEA-001] In check_images():
   - Capture image width, height, and mode; flag dimension outliers.
10. [GAP-004] In check_images():
    - Audit orphan disk image files and export outputs/sprint1/orphan_images.csv.
11. [WEA-002] In _singular():
    - Guard words ending in "ss" against incorrect stemming.
12. [RED-001] In ISSUES:
    - Downgrade CLR_NOT_IN_DESC to 'info'.

================================================================================
PHASE 4: TESTING & ARCHITECTURAL REFACTORING (P2 / P3)
================================================================================
13. [OPP-001] In missing_report():
    - Profile df_sales.csv alongside df_product.csv.
14. [OPP-002] Modularize code:
    - Split src/sprint1_preprocess.py into src/sprint1_preprocess/ package.
    - Add pytest unit tests verifying key uniqueness, null bounds, and schema.

================================================================================
PHASE 5: FUTURE STAGES (DELIBERATELY POSTPONED TO SPRINT 2)
================================================================================
- Zero-shot CLIP image-to-text semantic mismatch auditing (Phase 1b).
- Perceptual image hashing (pHash) for near-duplicate packshots (Phase 1b).
- LAB k-means color palette extraction (Phase 2b).
- Multimodal similarity retrieval engine (Phase 3).
```

---

## 6. Verification and Acceptance Criteria for Completed Work

When an implementation agent finishes the changes, the following checks **must pass**:

1. **Clean Dataset Contract Check:**
   ```bash
   python -c "
   import pandas as pd
   df = pd.read_parquet('data/processed/items_clean.parquet')
   assert df['PROD_CLR_EQUIV'].is_unique, 'Primary key not unique!'
   assert set(df['review_status'].unique()).issubset({'ok', 'fix'}), 'Unreviewed items leaked into clean dataset!'
   assert 'INFO_TXT' not in df.columns, 'Dead columns not pruned!'
   assert 'issues' not in df.columns, 'Internal audit flags leaked into clean dataset!'
   assert not df['realised_price'].isin([float('inf'), float('-inf')]).any(), 'Infinite realised price detected!'
   print('All clean dataset acceptance tests passed successfully!')
   "
   ```
2. **Review Syntax Regression Check:**
   ```bash
   python -c "
   from src.sprint1_preprocess import _parse_fixes
   res = _parse_fixes('COMPOSITION=Cotton; Polyester && GFA_DES_EN=Dress')
   assert res == [('COMPOSITION', 'Cotton; Polyester'), ('GFA_DES_EN', 'Dress')], f'Parser failed: {res}'
   print('Review parser test passed!')
   "
   ```
3. **Execution Idempotency Check:**
   Running `python src/sprint1_preprocess.py check` executes deterministically and updates all reports in `outputs/sprint1/` without altering files in `data/csv/` or `data/images/`.
