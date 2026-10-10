"""Phase 1: final cleaning of the colourway table (the input of every later phase).

    python src/phase1_clean.py            # preview: clean data/processed/items_checked.parquet now

`finalise_items()` is the single cleaning step. It runs in two places:
- `merge` (src/phase1_checks.py) applies it after the review decisions
  -> data/processed/items_clean.parquet, the official Phase 1 output;
- this script applies it to the automatic-check output before the human review
  -> data/processed/items_preview.parquet, so Phase 2 can start now.
Both files have the same columns; only the rows and the review columns differ.
Later phases call `load_items()`, which returns the official table when it exists.

Steps: drop empty/constant/duplicate/SKU-level columns, parse dates, unify casing, fix
known typos, turn placeholders into missing values, add sales features, and decide which
photos the visual block may use (`img_trusted`, `img_colour_trusted`).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import KEY, OUT_DIR, PROCESSED_DIR  # noqa: E402

REPORT_DIR = OUT_DIR
CLEAN_FILE = "items_clean.parquet"
PREVIEW_FILE = "items_preview.parquet"

# Columns later phases may rely on (data contract, checked by tests/test_data_contracts.py).
CONTRACT_COLUMNS = [
    KEY, "PROD_REF", "PROD_DES_BASE", "CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN", "CATEGORY_MATRIX",
    "CLR_COD", "CLR_DES", "CLR_TYPE", "COMPOSITION", "MATERIAL", "FINISHING", "OUTFIT", "DIMENSION",
    "PRINT_TYPE", "THEME", "FASHIONTYPE", "PROD_SEG", "PRICE_BASE_W_VAT", "STORE_DATE_FINAL", "SEA_COD",
    "sizes", "n_skus", "SALES_QTY", "SALES_AMT_FX_RATE", "has_sales", "log_sales_qty", "realised_price",
    "sales_pct_in_cat", "img_file", "has_image", "img_trusted", "img_colour_trusted", "issues", "priority",
    "review_status",
]
# Empty (< 0.1% filled) or constant in the delivered data (docs/data.md §3.3). A fixed list,
# not recomputed, so the output schema does not change from one run to the next.
EMPTY_OR_CONSTANT = [
    "PROD_DAT_CRI", "PHASE", "PHASE_COD", "DISPLAY_CLR_COD", "DISPLAY_CLR", "FLAG_FACT", "PONTAMETALICA_COD",
    "PONTAMETALICA_DES", "CLR_TON_DES", "CLR_BAS_DES", "DIMENSION_CONSUMABLE", "STORE_CONCEPT", "STATUS_ARTICLE",
    "PRODUCTION_TYPE", "STORE_LOCATION", "INFO_TXT", "AVERAGE_STORE", "CAPACITY", "END_DATE", "ITEM_GROUP",
    "SECONDARY_DISPLAY", "NIGHT_BAG", "ECI_APPAREL", "PROD_SIT", "TAR_COD", "TAR_DES", "PERMANENT",
    "PERMANENT_ASIS", "OUTLET", "OUTLET_GEN", "ACTIVE_RECORD", "WITH_ORDER", "CAT_ACTIVE",
]
# Exact copies of another column (the copy is dropped, the original kept).
DUPLICATES = {
    "CAT_DES_PT": "CAT_DES_EN", "GFA_DES_PT": "GFA_DES_EN", "GFS_DES_PT": "GFS_DES_EN",
    "PRICE_BASE_W_VAT_ASWAS": "PRICE_BASE_W_VAT", "WEEKS_TARGET": "ITEM_LIFE_CYCLE_WEEK",
    "FASHIONTYPE_ASIS": "FASHIONTYPE", "THEME_ASIS": "THEME", "THEME_CODE_ASIS": "THEME_CODE",
    "PROD_TYPOLOGY_ORIG_ASIS": "PROD_TYPOLOGY_ORIG", "PURCHASE_BET_ORIG_ASIS": "PURCHASE_BET_ORIG",
    "ONL_ENTRY_DATE": "ONL_ENTRY_DATE_DT", "ONL_IMAGE_DATE": "ONL_IMAGE_DATE_DT",
}
# Describe one size, not the colourway (sizes are listed in `sizes` / `n_skus`).
SKU_LEVEL = ["PROD_SPK", "PROD_COD", "PROD_COD_EQUIV", "BAR_COD", "SZ_COD", "SZ_DES", "PROD_DES", "DESCRIPTION"]
# Intermediate columns of the automatic checks (their result is in `issues`).
CHECK_INTERNALS = ["img_stem", "img_colour_code", "img_md5", "img_readable", "clr_expected_name",
                   "desc_colour_status", "desc_type_word", "desc_type_share"]

DATE_COLS = ["STORE_DATE_FINAL", "STORE_DATE_PLANNED", "STORE_DATE_REAL", "STORE_DATE_REVISED",
             "STORE_DATE_OUTLET_REAL", "START_DATE", "ONL_ENTRY_DATE_DT", "ONL_IMAGE_DATE_DT", "ECI_ONLINE_DATE",
             "AUDIT_DT_INSERT"]

# Values that mean "no value" in that column (compared case-insensitively). Kept on purpose as real
# categories: PRINT_TYPE "Others" (a print outside the list, unlike "No Print"), GFA/GFS "Others"
# and FINISHING "Other" (real families and finishes), L3/L4 "OTHERS" (merchandising buckets).
PLACEHOLDERS = {
    "DIMENSION": {"not applicable", "undefined"},
    "DISTRIBUTION_BLOCK": {"without block"},
    "CATEGORY_MATRIX": {"sem categoria"},
    "CATEGORY_MATRIX_BUYER": {"sem categoria"},
}
TYPOS = {"OPACITY": {"Transluncent": "Translucent"}}
# Free text, keys and paths: never re-cased.
NO_RECASE = {KEY, "PROD_CLR", "PROD_REF", "PROD_REF_EQUIV", "PROD_CLR_EQUIV", "PROD_DES_BASE", "PROG_IMAGE",
             "img_file", "issues", "sku_conflicts", "vis_note", "img_shared_with", "refs_all", "sizes",
             "fixes", "notes", "reviewer"}

# Image flags that make the photo unusable for visual similarity (it shows another product or is broken).
IMG_BLOCKING = {"IMG_UNREADABLE", "IMG_REF_MISMATCH", "VIS_TYPE_MISMATCH"}
# Flags that make the photo's colour unreliable (the shape is still fine).
IMG_COLOUR_BLOCKING = IMG_BLOCKING | {"IMG_COLOUR_MISMATCH", "IMG_NEAR_DUPLICATE", "VIS_COLOUR_MISMATCH",
                                      "IMG_SHARED", "IMG_GENERIC"}
# Review decisions that accept the photo: "Looks right" and "Fix data" (the data was wrong, not the photo).
PHOTO_ACCEPTED = {"ok", "fix"}


# --------------------------------------------------------------------------- steps
def parse_dates(s: pd.Series) -> pd.Series:
    """20230726 (int), 20230726.0 (float with NaN) or ISO '2023-07-26[ 21:40:36]' -> datetime64."""
    text = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    compact = text.str.fullmatch(r"\d{8}").fillna(False)
    out = pd.to_datetime(text.where(compact), format="%Y%m%d", errors="coerce")
    iso = pd.to_datetime(text.where(~compact), format="ISO8601", errors="coerce")
    return out.fillna(iso).astype("datetime64[ns]")


def unify_casing(s: pd.Series) -> tuple[pd.Series, int]:
    """Spellings that differ only in case ('Golden Basics' / 'GOLDEN BASICS') -> the most common one."""
    v = s.dropna()
    if v.empty or not v.map(type).eq(str).all():
        return s, 0
    canon: dict[str, str] = {}
    for value, _ in sorted(v.value_counts().items(), key=lambda kv: (-kv[1], kv[0])):
        canon.setdefault(value.casefold(), value)
    new = v.str.casefold().map(canon)
    changed = new != v
    if not changed.any():
        return s, 0
    out = s.copy()
    out.loc[new.index[changed]] = new[changed]
    return out, int(changed.sum())


def image_trust(items: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Can the visual block use this photo (shape / colour)? A human decision overrides the automatic flags."""
    codes = items["issues"].fillna("").str.split(";").map(set)
    accepted = items.get("review_status", pd.Series("", index=items.index)).isin(PHOTO_ACCEPTED)
    has = items["has_image"].fillna(False).astype(bool)
    trusted = has & (accepted | codes.map(IMG_BLOCKING.isdisjoint))
    colour = has & (accepted | codes.map(IMG_COLOUR_BLOCKING.isdisjoint))
    return trusted, colour


def finalise_items(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Clean the colourway table. Returns the table and a log of what changed (for the report).

    Columns that are absent are skipped, so the function also works on partial tables.
    """
    items = items.copy()
    log: dict = {"rows": len(items)}

    drop = [c for c in EMPTY_OR_CONSTANT + list(DUPLICATES) + SKU_LEVEL + CHECK_INTERNALS if c in items.columns]
    items = items.drop(columns=drop)
    log["dropped_columns"] = drop

    log["dates"] = {}
    for c in [c for c in DATE_COLS if c in items.columns]:
        before = items[c].notna().sum()
        items[c] = parse_dates(items[c])
        log["dates"][c] = int(before - items[c].notna().sum())  # values that could not be parsed

    for col, fixes in TYPOS.items():
        if col in items.columns:
            items[col] = items[col].replace(fixes)

    log["recased"] = {}
    for c in [c for c in items.columns if c not in NO_RECASE and pd.api.types.is_string_dtype(items[c])]:
        items[c], n = unify_casing(items[c])
        if n:
            log["recased"][c] = n

    log["placeholders"] = {}
    for col, values in PLACEHOLDERS.items():
        if col in items.columns:
            is_ph = items[col].astype("string").str.casefold().isin(values).fillna(False)
            items.loc[is_ph, col] = None
            log["placeholders"][col] = int(is_ph.sum())

    if "SALES_QTY" in items.columns:
        qty, amt = items["SALES_QTY"], items["SALES_AMT_FX_RATE"]
        items["has_sales"] = qty.notna()
        items["log_sales_qty"] = np.log1p(qty)
        items["realised_price"] = (amt / qty).where(qty > 0)  # 0 units with a positive amount -> unknown
        # Sales are only comparable inside a category: percentile rank 0..1 within CAT_DES_EN.
        items["sales_pct_in_cat"] = items.groupby("CAT_DES_EN")["SALES_QTY"].rank(pct=True)

    if {"issues", "has_image"} <= set(items.columns):
        items["img_trusted"], items["img_colour_trusted"] = image_trust(items)
        log["img_trusted"] = int(items["img_trusted"].sum())
        log["img_colour_trusted"] = int(items["img_colour_trusted"].sum())

    items = items.sort_values(KEY).reset_index(drop=True)
    return items, log


# --------------------------------------------------------------------------- I/O
def load_items() -> pd.DataFrame:
    """The Phase 1 table for later phases: the reviewed one if merged, otherwise the preview."""
    for name in (CLEAN_FILE, PREVIEW_FILE):
        path = PROCESSED_DIR / name
        if path.exists():
            return pd.read_parquet(path)
    raise SystemExit("No Phase 1 table. Run: python src/phase1_checks.py check && python src/phase1_clean.py")


def write_report(items: pd.DataFrame, log: dict, source: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(items)
    schema = pd.DataFrame({"column": items.columns, "dtype": items.dtypes.astype(str).values,
                           "filled_pct": (items.notna().mean() * 100).round(1).values})
    rows = "\n".join(f"| `{r.column}` | {r.dtype} | {r.filled_pct} |" for r in schema.itertuples())
    by_cat = items.groupby("CAT_DES_EN").agg(items=(KEY, "size"), with_photo=("has_image", "sum"),
                                             photo_trusted=("img_trusted", "sum"),
                                             colour_trusted=("img_colour_trusted", "sum"))
    cat_rows = "\n".join(f"| {c} | {r['items']:,} | {r.with_photo:,} | {r.photo_trusted:,} | {r.colour_trusted:,} |"
                         for c, r in by_cat.iterrows())
    statuses = items["review_status"].value_counts() if "review_status" in items.columns else pd.Series(dtype=int)
    dates = ", ".join(f"`{c}` {v}" for c, v in log["dates"].items() if v) or "none"
    lines = [
        "# Phase 1: clean colourway table",
        "",
        f"Source: `{source}` · {n:,} colourways · {len(items.columns)} columns · "
        f"review status: {', '.join(f'{k} {v:,}' for k, v in statuses.items()) or 'not reviewed yet'}",
        "",
        "Produced by `src/phase1_clean.py` (`finalise_items`). Later phases load it with `load_items()`.",
        "",
        "## What was cleaned",
        "",
        f"- **Dropped {len(log['dropped_columns'])} columns**: empty or constant, exact duplicates, SKU-level "
        "(size, barcode, size-specific description) and intermediate check columns.",
        f"- **Dates** parsed to datetime ({len(log['dates'])} columns). Values that could not be parsed: {dates}.",
        f"- **Casing unified** (most common spelling wins): "
        f"{', '.join(f'`{c}` {v}' for c, v in log['recased'].items()) or 'none'} values.",
        f"- **Placeholders set to missing**: {', '.join(f'`{c}` {v:,}' for c, v in log['placeholders'].items())}. "
        "`PRINT_TYPE = Others`, `GFA_DES_EN = Others` and `FINISHING = Other` are kept as real categories.",
        "- **Typos**: `OPACITY` Transluncent → Translucent.",
        "- **Sales features**: `has_sales`, `log_sales_qty` (log1p), `realised_price` (empty when 0 units), "
        "`sales_pct_in_cat` (percentile within the category).",
        "",
        "## Photos the visual block may use",
        "",
        "- `img_trusted`: the photo shows this product (no `" + "`, `".join(sorted(IMG_BLOCKING)) + "` flag).",
        "- `img_colour_trusted`: its colour can also be trusted (none of `"
        + "`, `".join(sorted(IMG_COLOUR_BLOCKING - IMG_BLOCKING)) + "` either).",
        "- Until the human review is merged, these come from the automatic flags only. After `merge`, a "
        "\"Looks right\" or \"Fix data\" decision accepts the photo and \"Wrong photo\" removes it.",
        "- Items without a trusted photo use the tabular and text blocks only.",
        "",
        "| Category | Items | With photo | Photo trusted | Colour trusted |",
        "|---|---|---|---|---|",
        cat_rows,
        f"| **Total** | {n:,} | {int(items['has_image'].sum()):,} | {log['img_trusted']:,} | "
        f"{log['img_colour_trusted']:,} |",
        "",
        "## Schema",
        "",
        "| Column | Type | Filled % |",
        "|---|---|---|",
        rows,
        "",
    ]
    (REPORT_DIR / "clean_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    path = PROCESSED_DIR / "items_checked.parquet"
    if not path.exists():
        raise SystemExit("Run the automatic checks first: python src/phase1_checks.py check")
    items, log = finalise_items(pd.read_parquet(path).assign(review_status="unreviewed"))
    items.to_parquet(PROCESSED_DIR / PREVIEW_FILE, index=False)
    write_report(items, log, f"data/processed/{PREVIEW_FILE}")
    print(f"Preview (before the human review): {len(items):,} colourways, {len(items.columns)} columns "
          f"-> {PROCESSED_DIR / PREVIEW_FILE}")
    print(f"Photos trusted: {log['img_trusted']:,} (colour also trusted: {log['img_colour_trusted']:,}). "
          f"Report: {REPORT_DIR / 'clean_report.md'}")


if __name__ == "__main__":
    main()
