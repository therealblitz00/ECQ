# Phase 1: clean colourway table

Source: `data/processed/items_preview.parquet` · 10,555 colourways · 109 columns · review status: unreviewed 10,555

Produced by `src/phase1_clean.py` (`finalise_items`). Later phases load it with `load_items()`.

## What was cleaned

- **Dropped 61 columns**: empty or constant, exact duplicates, SKU-level (size, barcode, size-specific description) and intermediate check columns.
- **Dates** parsed to datetime (10 columns). Values that could not be parsed: none.
- **Casing unified** (most common spelling wins): `THEME` 153, `DIMENSION` 4, `GFS_DES_EN` 7, `L4_DES` 2 values.
- **Placeholders set to missing**: `DIMENSION` 7,168, `DISTRIBUTION_BLOCK` 9,896, `CATEGORY_MATRIX` 262, `CATEGORY_MATRIX_BUYER` 281. `PRINT_TYPE = Others`, `GFA_DES_EN = Others` and `FINISHING = Other` are kept as real categories.
- **Typos**: `OPACITY` Transluncent → Translucent.
- **Sales features**: `has_sales`, `log_sales_qty` (log1p), `realised_price` (empty when 0 units), `sales_pct_in_cat` (percentile within the category).

## Photos the visual block may use

- `img_trusted`: the photo shows this product (no `IMG_REF_MISMATCH`, `IMG_UNREADABLE`, `VIS_TYPE_MISMATCH` flag).
- `img_colour_trusted`: its colour can also be trusted (none of `IMG_COLOUR_MISMATCH`, `IMG_GENERIC`, `IMG_NEAR_DUPLICATE`, `IMG_SHARED`, `VIS_COLOUR_MISMATCH` either).
- Until the human review is merged, these come from the automatic flags only. After `merge`, a "Looks right" or "Fix data" decision accepts the photo and "Wrong photo" removes it.
- Items without a trusted photo use the tabular and text blocks only.

| Category | Items | With photo | Photo trusted | Colour trusted |
|---|---|---|---|---|
| Apparel | 2,259 | 1,885 | 1,879 | 1,825 |
| Footwear | 435 | 424 | 422 | 411 |
| Hand Bag | 1,971 | 1,812 | 1,812 | 1,774 |
| Jewellery | 4,639 | 4,143 | 4,128 | 3,972 |
| Wallet | 1,251 | 1,194 | 1,192 | 1,169 |
| **Total** | 10,555 | 9,458 | 9,433 | 9,151 |

## Schema

| Column | Type | Filled % |
|---|---|---|
| `PROD_CLR_EQUIV` | str | 100.0 |
| `CFS_SFK` | int64 | 100.0 |
| `CLR_COD` | str | 100.0 |
| `CLR_DES` | str | 100.0 |
| `PROD_CLR` | str | 100.0 |
| `PROD_YEAR_SEA` | str | 100.0 |
| `SEA_COD` | str | 100.0 |
| `PROD_REF` | str | 100.0 |
| `PROG_IMAGE` | str | 100.0 |
| `COMPOSITION` | str | 100.0 |
| `NET_WEIGHT` | float64 | 97.7 |
| `PROD_SEG` | str | 100.0 |
| `SUP_SFK` | int64 | 100.0 |
| `SUP_COD` | str | 100.0 |
| `THEME` | str | 100.0 |
| `MAIN_THEME` | str | 100.0 |
| `THEME_CODE` | str | 48.9 |
| `THEME_ID` | int64 | 100.0 |
| `CLR_TYPE` | str | 100.0 |
| `MATCHING` | str | 100.0 |
| `FASHIONTYPE` | str | 100.0 |
| `FASHIONTYPE_COD` | int64 | 100.0 |
| `TYPOLOGY` | str | 100.0 |
| `PROD_ONLINE` | str | 100.0 |
| `PUR_BET` | str | 99.7 |
| `TARIFF_COD` | str | 99.8 |
| `PRICE_BASE_W_VAT` | float64 | 100.0 |
| `PRICE_OUTLET_W_VAT` | float64 | 100.0 |
| `DISPLAY_COD` | int64 | 100.0 |
| `DISPLAY_DES` | str | 100.0 |
| `CATEGORY_MATRIX` | str | 97.5 |
| `DISTRIB_CENTRAL` | str | 100.0 |
| `PROD_YEAR_SEA_INI` | str | 100.0 |
| `MADE_IN` | str | 99.8 |
| `OUTFIT` | str | 21.4 |
| `STORE_DATE_PLANNED` | datetime64[ns] | 100.0 |
| `STORE_DATE_REAL` | datetime64[ns] | 36.2 |
| `STORE_DATE_REVISED` | datetime64[ns] | 56.3 |
| `STORE_DATE_OUTLET_REAL` | datetime64[ns] | 29.3 |
| `STORE_DATE_FINAL` | datetime64[ns] | 100.0 |
| `DIMENSION` | str | 32.1 |
| `NUMBER_OF_UNITS` | str | 100.0 |
| `DISTRIBUTION_BLOCK` | str | 6.2 |
| `COLLECTION_TYPE` | str | 100.0 |
| `PRINT_TYPE` | str | 100.0 |
| `OPACITY` | str | 100.0 |
| `HISTORY` | str | 2.2 |
| `OPEN_REFERENCE` | str | 100.0 |
| `PROD_REF_EQUIV` | str | 100.0 |
| `CAT_COD` | str | 100.0 |
| `CAT_DES_EN` | str | 100.0 |
| `CAT_SIT` | str | 100.0 |
| `GFA_COD` | str | 100.0 |
| `GFA_DES_EN` | str | 100.0 |
| `GFA_SIT` | str | 100.0 |
| `GFS_COD` | str | 100.0 |
| `GFS_DES_EN` | str | 100.0 |
| `GFS_SIT` | str | 100.0 |
| `FINISHING` | str | 48.5 |
| `SHAPE` | str | 0.2 |
| `PRODUCT_DETAILS` | str | 1.7 |
| `PROD_TYPOLOGY_ORIG` | str | 1.4 |
| `PURCHASE_BET_ORIG` | str | 2.4 |
| `MATERIAL` | str | 12.9 |
| `START_DATE` | datetime64[ns] | 100.0 |
| `COLLECTION_TYPE_GROUP` | str | 100.0 |
| `L1_DES` | str | 100.0 |
| `L2_DES` | str | 100.0 |
| `L3_DES` | str | 100.0 |
| `L4_DES` | str | 100.0 |
| `FSALDOS` | str | 100.0 |
| `FSALDOS_OUTLET` | str | 100.0 |
| `PRICE_LCP` | float64 | 98.4 |
| `TRANSP_COST` | float64 | 94.6 |
| `SEASON_DETAIL` | str | 3.6 |
| `ONL_ENTRY_DATE_DT` | datetime64[ns] | 94.9 |
| `ITEM_LIFE_CYCLE_WEEK` | int64 | 100.0 |
| `ONL_IMAGE_DATE_DT` | datetime64[ns] | 97.9 |
| `RECLASSIFIED` | int64 | 100.0 |
| `ITEM_REPEAT` | str | 77.5 |
| `ECI_ONLINE_DATE` | datetime64[ns] | 59.0 |
| `AUDIT_DT_INSERT` | datetime64[ns] | 100.0 |
| `RECLASS_REASON` | str | 11.4 |
| `RECLASS_CODE` | str | 11.4 |
| `STORE_MONTH_FINAL_PC` | str | 100.0 |
| `CUSTOMIZABLE` | str | 78.2 |
| `CATEGORY_MATRIX_BUYER` | str | 97.3 |
| `COLLECTION_STRUCTURE` | str | 100.0 |
| `PROD_DES_BASE` | str | 100.0 |
| `sku_conflicts` | str | 100.0 |
| `refs_all` | str | 100.0 |
| `n_skus` | int64 | 100.0 |
| `sizes` | str | 100.0 |
| `n_images_listed` | int64 | 100.0 |
| `SALES_QTY` | float64 | 96.5 |
| `SALES_AMT_FX_RATE` | float64 | 96.5 |
| `issues` | str | 100.0 |
| `img_file` | str | 89.6 |
| `has_image` | bool | 100.0 |
| `img_shared_with` | str | 100.0 |
| `vis_note` | str | 100.0 |
| `priority` | str | 100.0 |
| `review_status` | str | 100.0 |
| `has_sales` | bool | 100.0 |
| `log_sales_qty` | float64 | 96.5 |
| `realised_price` | float64 | 96.4 |
| `sales_pct_in_cat` | float64 | 96.5 |
| `img_trusted` | bool | 100.0 |
| `img_colour_trusted` | bool | 100.0 |
