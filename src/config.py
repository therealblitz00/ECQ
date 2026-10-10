"""Shared settings for every pipeline script: paths, the item key, issue codes and the team.

Import from here instead of redefining paths, e.g. `from config import KEY, PROCESSED_DIR`.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv"
IMG_DIR = ROOT / "data" / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
EMB_DIR = ROOT / "data" / "embeddings"  # masks and embeddings (committed, computed on one machine)
OUT_DIR = ROOT / "outputs" / "phase1"  # reports and review batches of Phase 1
BATCH_DIR = OUT_DIR / "batches"

KEY = "PROD_CLR_EQUIV"
# Phase 1 review: batch number -> team member (used with --members 5).
TEAM = {1: "André", 2: "Pedro Correia", 3: "Pedro Meireles", 4: "Manuel", 5: "Zé"}
# Codes must stay strings (leading zeros, no float conversion).
STR_DTYPES = {c: str for c in ["PROD_REF", "PROD_REF_EQUIV", "BAR_COD", "SEA_COD", "CAT_COD",
                               "GFA_COD", "GFS_COD", "SUP_COD", "TARIFF_COD"]}

# Issue code -> severity. "high" = sources contradict each other, must be reviewed.
ISSUES = {
    "IMG_PATH_INVALID": "high",        # PROG_IMAGE does not follow /<season>/<cat>/<file>
    "IMG_FILE_MISSING": "info",        # no image file -> item will use tabular data only
    "IMG_UNREADABLE": "high",          # file exists but is corrupt
    "IMG_REF_MISMATCH": "high",        # image file name belongs to another model
    "IMG_CAT_MISMATCH": "medium",      # category folder in path != CAT_COD (often a reclassified item)
    "IMG_COLOUR_MISMATCH": "high",     # colour code in file name != CLR_COD
    "IMG_GENERIC": "medium",           # file name has no colour -> colour cannot be confirmed
    "IMG_SHARED": "medium",            # same image used by another colour or model
    "TAB_KEY_INCONSISTENT": "high",    # PROD_CLR != PROD_REF + CLR_COD
    "CLR_CODE_NAME_INCONSISTENT": "high",  # CLR_COD letters map to a different CLR_DES elsewhere
    "CLR_CONFLICT_DESC": "high",       # PROD_DES names a different colour than CLR_DES
    "CLR_NOT_IN_DESC": "medium",       # PROD_DES does not mention the colour at all
    "TYPE_CONFLICT_DESC": "high",      # PROD_DES item type disagrees with GFA_DES_EN
    "SALES_MISSING": "info",           # no row in df_sales
    "SALES_ZERO_QTY": "low",           # SALES_QTY == 0
    "SKU_ATTR_CONFLICT": "medium",     # sizes of the same colourway disagree on an attribute (see sku_conflicts)
    # Image checks from Phase 1b (src/phase1b_image_audit.py), added when data/embeddings/image_audit.parquet exists
    "VIS_TYPE_MISMATCH": "high",       # CLIP and the most similar photos say another item type
    "VIS_TYPE_DOUBT": "low",           # CLIP alone says another item type
    "VIS_COLOUR_MISMATCH": "medium",   # CLIP says another colour than CLR_DES
    "VIS_TARGET_NOT_FOUND": "low",     # detector cannot find the expected item (mostly sample/prototype photos)
    "IMG_NEAR_DUPLICATE": "medium",    # same photo as another colour of the same model (shows the wrong colour)
    "IMG_NOT_PACKSHOT": "low",         # model, lifestyle or amateur photo
}
SEVERITY_RANK = {"high": 3, "medium": 2, "low": 1, "info": 0}

REVIEW_STATUSES = {"ok", "fix", "drop_image", "discard", "team_review"}
REVIEW_COLS = ["review_status", "fixes", "reviewer", "notes"]
