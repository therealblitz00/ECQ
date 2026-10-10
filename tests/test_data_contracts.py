"""Contracts of the committed files that later phases rely on (data/csv, data/embeddings, review batches).

These tests read the real repository files, not the toy dataset. They are skipped when the
files are missing (e.g. a partial checkout).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import phase1_clean as p1
import sprint1_preprocess as sp

pytestmark = pytest.mark.data

EMB_DIR = sp.ROOT / "data" / "embeddings"
BATCH_DIR = sp.ROOT / "outputs" / "sprint1" / "batches"


def require(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(sp.ROOT)} not found")
    return path


@pytest.fixture(scope="module")
def colourways() -> set[str]:
    path = require(sp.ROOT / "data" / "csv" / "df_product.csv")
    return set(pd.read_csv(path, usecols=[sp.KEY], encoding="utf-8")[sp.KEY].str.strip())


def test_sales_keys_are_unique_and_join_to_products(colourways):
    sales = pd.read_csv(require(sp.ROOT / "data" / "csv" / "df_sales.csv"), encoding="utf-8")
    assert sales[sp.KEY].is_unique
    assert set(sales[sp.KEY].str.strip()) <= colourways


def test_image_embeddings_match_their_sidecar(colourways):
    meta = json.loads(require(EMB_DIR / "image_clip.json").read_text(encoding="utf-8"))
    emb = np.load(require(EMB_DIR / "image_clip.npy"))
    assert emb.shape == (meta["n"], meta["dim"]) == (len(meta["keys"]), meta["dim"])
    assert emb.dtype == np.dtype(meta["dtype"])
    assert len(set(meta["keys"])) == len(meta["keys"])
    assert set(meta["keys"]) <= colourways
    assert np.isfinite(emb).all()
    if meta.get("normalised"):
        assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=1e-2)


@pytest.mark.parametrize("name, columns", [
    ("masks.parquet", [sp.KEY]),
    ("image_audit.parquet", [sp.KEY, "vis_issues", "vis_note"]),  # read by sprint1_preprocess and review_app
])
def test_embedding_tables_have_one_row_per_known_colourway(colourways, name, columns):
    df = pd.read_parquet(require(EMB_DIR / name))
    assert set(columns) <= set(df.columns)
    assert df[sp.KEY].is_unique
    assert set(df[sp.KEY]) <= colourways


def test_image_audit_uses_known_issue_codes():
    audit = pd.read_parquet(require(EMB_DIR / "image_audit.parquet"), columns=["vis_issues"])
    codes = set(audit["vis_issues"].fillna("").str.split(";").explode()) - {""}
    assert codes <= set(sp.ISSUES)


def test_review_batches_are_valid_and_cover_every_colourway(colourways):
    files = sorted(BATCH_DIR.glob("batch_*_of_*.csv")) if BATCH_DIR.exists() else []
    if not files:
        pytest.skip("no review batches")
    batches = [pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False) for f in files]
    for f, df in zip(files, batches):
        assert list(df.columns) == sp.BATCH_COLS + sp.REVIEW_COLS, f.name
        statuses = set(df["review_status"].str.strip().str.lower())
        assert statuses <= sp.REVIEW_STATUSES | {""}, f"{f.name}: invalid review_status {statuses}"
    keys = pd.concat([df[sp.KEY] for df in batches])
    assert keys.is_unique
    assert set(keys) == {k for k in colourways if pd.notna(k)}


@pytest.fixture(scope="module")
def phase1_preview():
    require(sp.ROOT / "data" / "csv" / "df_product.csv")
    items, ctx = sp.run_checks(verify_files=False)
    preview, _ = p1.finalise_items(items.assign(review_status="unreviewed"))
    return preview, ctx


def test_phase1_table_has_one_row_per_colourway_and_every_sale(phase1_preview, colourways):
    """Phase 1 'done when': unique key, all sales rows joined, stable row count."""
    items, ctx = phase1_preview
    assert items[sp.KEY].is_unique
    assert set(items[sp.KEY]) == colourways
    assert int(items["has_sales"].sum()) == len(ctx["sales"])
    missing = [c for c in p1.CONTRACT_COLUMNS if c not in items.columns]
    assert missing == []
    assert items["STORE_DATE_FINAL"].notna().all()
    assert items["img_trusted"].le(items["has_image"]).all()
    assert items["img_colour_trusted"].le(items["img_trusted"]).all()
