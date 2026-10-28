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
import config
import phase1_checks as checks
import phase2a_features as f2a

pytestmark = pytest.mark.data

EMB_DIR = config.EMB_DIR
BATCH_DIR = config.BATCH_DIR


def require(path):
    if not path.exists():
        pytest.skip(f"{path.relative_to(config.ROOT)} not found")
    return path


@pytest.fixture(scope="module")
def colourways() -> set[str]:
    path = require(config.ROOT / "data" / "csv" / "df_product.csv")
    return set(pd.read_csv(path, usecols=[config.KEY], encoding="utf-8")[config.KEY].str.strip())


def test_sales_keys_are_unique_and_join_to_products(colourways):
    sales = pd.read_csv(require(config.ROOT / "data" / "csv" / "df_sales.csv"), encoding="utf-8")
    assert sales[config.KEY].is_unique
    assert set(sales[config.KEY].str.strip()) <= colourways


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
    ("masks.parquet", [config.KEY]),
    ("image_audit.parquet", [config.KEY, "vis_issues", "vis_note"]),  # read by phase1_checks and review_app
])
def test_embedding_tables_have_one_row_per_known_colourway(colourways, name, columns):
    df = pd.read_parquet(require(EMB_DIR / name))
    assert set(columns) <= set(df.columns)
    assert df[config.KEY].is_unique
    assert set(df[config.KEY]) <= colourways


def test_image_audit_uses_known_issue_codes():
    audit = pd.read_parquet(require(EMB_DIR / "image_audit.parquet"), columns=["vis_issues"])
    codes = set(audit["vis_issues"].fillna("").str.split(";").explode()) - {""}
    assert codes <= set(config.ISSUES)


def test_review_batches_are_valid_and_cover_every_colourway(colourways):
    files = sorted(BATCH_DIR.glob("batch_*_of_*.csv")) if BATCH_DIR.exists() else []
    if not files:
        pytest.skip("no review batches")
    batches = [pd.read_csv(f, dtype=str, encoding="utf-8-sig", keep_default_na=False) for f in files]
    for f, df in zip(files, batches):
        assert list(df.columns) == checks.BATCH_COLS + config.REVIEW_COLS, f.name
        statuses = set(df["review_status"].str.strip().str.lower())
        assert statuses <= config.REVIEW_STATUSES | {""}, f"{f.name}: invalid review_status {statuses}"
    keys = pd.concat([df[config.KEY] for df in batches])
    assert keys.is_unique
    assert set(keys) == {k for k in colourways if pd.notna(k)}


@pytest.fixture(scope="module")
def phase1_preview():
    require(config.ROOT / "data" / "csv" / "df_product.csv")
    items, ctx = checks.run_checks(verify_files=False)
    preview, _ = p1.finalise_items(items.assign(review_status="unreviewed"))
    return preview, ctx


def test_phase1_table_has_one_row_per_colourway_and_every_sale(phase1_preview, colourways):
    """Phase 1 'done when': unique key, all sales rows joined, stable row count."""
    items, ctx = phase1_preview
    assert items[config.KEY].is_unique
    assert set(items[config.KEY]) == colourways
    assert int(items["has_sales"].sum()) == len(ctx["sales"])
    missing = [c for c in p1.CONTRACT_COLUMNS if c not in items.columns]
    assert missing == []
    assert items["STORE_DATE_FINAL"].notna().all()
    assert items["img_trusted"].le(items["has_image"]).all()
    assert items["img_colour_trusted"].le(items["img_trusted"]).all()


def test_phase2a_blocks_follow_the_contract(phase1_preview):
    """Every feature block: one row per colourway, unit-length rows, attribute slices that tile the columns."""
    items = f2a.phase2a_impute.add_finish_family(phase1_preview[0].sort_values(config.KEY, kind="stable").reset_index(drop=True))
    for name, build in f2a.BLOCKS.items():
        m, columns, attrs = build(items)
        assert m.shape == (len(items), len(columns)) and m.dtype == np.float32, name
        norms = np.linalg.norm(m, axis=1)
        assert np.allclose(norms[norms > 0], 1, atol=1e-5), name
        assert sorted((a["start"], a["stop"]) for a in attrs.values())[-1][1] == m.shape[1], name
        assert sum(a["stop"] - a["start"] for a in attrs.values()) == m.shape[1], name


def test_text_embeddings_match_their_sidecar_and_the_phase1_table(phase1_preview):
    """data/embeddings/text.npy: one unit vector per colourway, row order in text.json."""
    require(config.EMB_DIR / "text.npy")
    emb = np.load(config.EMB_DIR / "text.npy")
    meta = json.loads((config.EMB_DIR / "text.json").read_text(encoding="utf-8"))
    assert emb.shape == (meta["n"], meta["dim"]) == (len(meta["keys"]), meta["dim"])
    assert {"model", "revision", "template", "created"} <= set(meta)
    assert np.allclose(np.linalg.norm(emb.astype(np.float32), axis=1), 1, atol=1e-2)
    assert set(meta["keys"]) == set(phase1_preview[0][config.KEY])


def test_palettes_match_their_sidecar():
    """data/embeddings/palettes.parquet: one row per trusted photo, shares sum to 1, unit palette vectors."""
    require(config.EMB_DIR / "palettes.parquet")
    pal = pd.read_parquet(config.EMB_DIR / "palettes.parquet")
    meta = json.loads((config.EMB_DIR / "palettes.json").read_text(encoding="utf-8"))
    assert pal[config.KEY].tolist() == meta["keys"] and len(pal) == meta["n"]
    assert pal[config.KEY].is_unique
    assert np.allclose(pal["shares"].map(sum), 1, atol=1e-3)
    vec = np.stack(pal["palette_vector"].map(np.asarray))
    assert vec.shape[1] == len(meta["names"]) and np.allclose(np.linalg.norm(vec, axis=1), 1, atol=1e-5)


def test_inferred_attributes_only_fill_gaps_of_items_with_a_photo(phase1_preview):
    """phase2a_impute: never replaces a CSV value, only fills items with a trusted photo, only known values."""
    require(config.EMB_DIR / "palettes.parquet")
    items = phase1_preview[0].sort_values(config.KEY, kind="stable").reset_index(drop=True)
    inferred, _ = f2a.phase2a_impute.infer(items)
    full = f2a.phase2a_impute.add_finish_family(items).set_index(config.KEY)
    for col, sub in inferred.groupby("column"):
        rows = full.loc[sub[config.KEY]]
        assert rows[col].isna().all() and rows["img_trusted"].all(), col
        assert set(sub["value"]) <= set(full[col].dropna()), col
        assert set(rows["CAT_DES_EN"]) == {f2a.phase2a_impute.TARGETS[col]}, col


def test_neighbours_follow_the_phase3_contract():
    """data/processed/neighbours.parquet: >= 4 like-for-like neighbours per colourway, never the same model."""
    path = require(config.PROCESSED_DIR / "neighbours.parquet")
    nb = pd.read_parquet(path)
    meta = json.loads((config.PROCESSED_DIR / "neighbours.json").read_text(encoding="utf-8"))
    items = p1.load_items().set_index(config.KEY)
    assert list(nb.columns) == meta["columns"] and len(nb) == meta["rows"]
    assert nb.groupby("query").size().reindex(items.index, fill_value=0).min() >= 4
    assert not nb.duplicated(["query", "candidate"]).any()
    np.testing.assert_allclose(nb.filter(like="contrib_").sum(axis=1), nb["score"], atol=1e-4)
    for col in ("CAT_DES_EN",) + (() if meta["include_siblings"] else ("PROD_REF",)):
        same = items.loc[nb["query"], col].to_numpy() == items.loc[nb["candidate"], col].to_numpy()
        assert same.all() if col == "CAT_DES_EN" else not same.any(), col
    untrusted = ~items.loc[nb["candidate"], "img_trusted"].to_numpy(bool)
    assert (nb.loc[untrusted, "contrib_image"] == 0).all()


def test_human_review_sheet_is_valid():
    """outputs/phase4/human_review_<n>.csv: together the queries of the sidecar, one row per pair, known verdicts."""
    import phase4_eval as ev
    meta = json.loads(require(ev.SHEET_META).read_text(encoding="utf-8"))
    s = pd.concat([ev.read_sheet(p) for p in ev.sheet_paths()], ignore_index=True)
    assert list(dict.fromkeys(s["query"])) == meta["queries"]
    assert not s.duplicated(["query", "candidate"]).any()
    verdicts = s["verdict"].str.strip().str.lower()
    assert verdicts[verdicts != ""].isin(ev.VERDICTS).all(), "verdict must be similar, partly or not"


def test_feedback_pool_and_answers_are_valid():
    """outputs/phase4/feedback_pool.json (fixed pools) and feedback_<n>.jsonl (each member's answers)."""
    import phase4_feedback as fb
    pool = json.loads(require(fb.POOL_FILE).read_text(encoding="utf-8"))
    # The pool and the answers are committed; the Phase 1 table is not (CI never builds it),
    # so the like-for-like check runs only where data/processed/ exists.
    has_table = any((config.PROCESSED_DIR / f).exists() for f in (p1.CLEAN_FILE, p1.PREVIEW_FILE))
    items = p1.load_items().set_index(config.KEY) if has_table else None
    assert set(pool["weights"]) == set(pool["blocks"])
    by_q = {q["query"]: q for q in pool["queries"]}
    assert len(by_q) == len(pool["queries"])
    for q in pool["queries"]:
        keys = [c["key"] for c in q["candidates"]]
        assert len(keys) == len(set(keys)) and q["query"] not in keys, q["query"]
        if items is not None:
            assert (items.loc[keys, "CAT_DES_EN"] == items.at[q["query"], "CAT_DES_EN"]).all(), q["query"]
        assert set(q["members"]) <= set(config.TEAM)
    for r in fb.read_feedback():
        assert r["member"] in by_q[r["query"]]["members"], (r["member"], r["query"])
        fb.check_answer(by_q[r["query"]], r["similar"], r["not"])


def test_colour_reference_covers_every_colour_name():
    """data/processed/colour_reference.parquet (phase2a_colour): one row per CLR_DES name, a colour or a reason."""
    ref = pd.read_parquet(require(config.PROCESSED_DIR / "colour_reference.parquet"))
    items = p1.load_items()
    assert ref["CLR_DES"].is_unique and set(ref["CLR_DES"]) == set(items["CLR_DES"].dropna())
    measured = ref["no_reference"].isna()
    assert ref.loc[measured, ["L", "a", "b"]].notna().all().all() and ref.loc[measured, "hex"].str.match(r"^#[0-9a-f]{6}$").all()
    assert ref.loc[~measured, ["L", "a", "b"]].isna().all().all()


def test_fashionsiglip_embeddings_match_their_sidecar_and_the_clip_rows():
    meta = json.loads((config.EMB_DIR / "image_fashionsiglip.json").read_text(encoding="utf-8"))
    clip = json.loads((config.EMB_DIR / "image_clip.json").read_text(encoding="utf-8"))
    emb = np.load(config.EMB_DIR / "image_fashionsiglip.npy")
    assert emb.shape == (meta["n"], meta["dim"]) and emb.dtype == np.float16
    assert meta["keys"] == clip["keys"]                     # same photos, same row order as the baseline
    norms = np.linalg.norm(emb.astype(np.float32), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-2)


def test_person_recrops_are_confirmed_suggestions_of_the_automatic_rule():
    pc = pd.read_parquet(config.EMB_DIR / "person_crops.parquet")
    review = json.loads((config.EMB_DIR / "person_crops_reviewed.json").read_text(encoding="utf-8"))
    assert pc[config.KEY].is_unique
    assert not (pc["recrop"] & ~pc["auto_rule"]).any()        # nothing re-cropped that the rule did not suggest
    assert sorted(pc.loc[pc["recrop"], config.KEY]) == sorted(review["accepted"])
    assert set(pc.loc[pc["auto_rule"], config.KEY]) == set(review["accepted"]) | set(review["rejected"])  # all reviewed


def test_the_recrop_rule_in_the_code_reproduces_the_committed_suggestions():
    import phase2b_image_encoders as ie
    pc = pd.read_parquet(config.EMB_DIR / "person_crops.parquet")
    masks = pd.read_parquet(config.EMB_DIR / "masks.parquet").set_index(config.KEY)
    again = ie.decide(pc.drop(columns=["auto_rule", "recrop"]), masks)
    assert (again["auto_rule"] == pc["auto_rule"]).all() and (again["recrop"] == pc["recrop"]).all()
    boxes = pc.loc[pc["recrop"], "item_box"]
    assert all(len(b) == 4 and 0 <= b[0] < b[2] <= 1 and 0 <= b[1] < b[3] <= 1 for b in boxes)


def test_explanations_follow_their_contract():
    path = config.PROCESSED_DIR / "explanations.parquet"
    if not path.exists():
        pytest.skip("run src/phase5_explain.py build")
    ex = pd.read_parquet(path)
    shares = [c for c in ex.columns if c.startswith("share_")]
    assert len(shares) == 6 and set(ex["rank"]) <= set(range(1, 6))
    assert ex.groupby("query").size().min() >= 4                       # the brief: at least 4 similar products
    assert np.allclose(ex[shares].sum(axis=1), 1.0, atol=1e-4)           # the block shares add up to the score
    assert ex["completeness_error"].max() < 1e-5
    assert not (ex["query"] == ex["candidate"]).any()


def test_forecast_follows_its_contract():
    path = config.PROCESSED_DIR / "forecast.parquet"
    if not path.exists():
        pytest.skip("run src/phase5b_forecast.py run")
    fc = pd.read_parquet(path)
    assert fc[config.KEY].is_unique
    for t in ("sales", "price"):
        assert fc[f"{t}_estimate"].notna().all() and (fc[f"{t}_estimate"] >= 0).all()
        assert all(k.startswith(("cross-validated (target not seen)", "forecast")) for k in fc[f"{t}_estimate_kind"])
        assert fc[f"{t}_typical_error"].nunique() == 1
    assert (fc["price_point_estimate"] > 0).all() and fc["price_point_estimate"].nunique() <= 40   # the list prices
    for t in ("sales", "price"):                                       # the 80% range brackets the estimate
        assert (fc[f"{t}_low"] <= fc[f"{t}_estimate"]).all() and (fc[f"{t}_estimate"] <= fc[f"{t}_high"]).all()
        sizes = fc[f"{t}_reasons"].map(lambda s: [float(p.rsplit(" ", 1)[1].replace("−", "").replace("+", ""))
                                                  for p in s.split("; ") if p])
        assert sizes.map(lambda v: len(v) <= 3 and v == sorted(v, reverse=True)).all()   # biggest effect first


def test_estimates_csv_matches_forecast():
    path, fpath = config.ROOT / "outputs" / "phase5b" / "estimates.csv", config.PROCESSED_DIR / "forecast.parquet"
    if not (path.exists() and fpath.exists()):
        pytest.skip("run src/phase5b_forecast.py run")
    est = pd.read_csv(path, dtype={config.KEY: str})
    fc = pd.read_parquet(fpath).set_index(config.KEY)
    assert est[config.KEY].is_unique and len(est) == len(fc)            # every product, once
    cols = ["sales_estimate", "sales_low", "sales_high", "price_estimate", "price_point_estimate", "price_low",
            "price_high"]
    assert np.allclose(est.set_index(config.KEY).loc[fc.index, cols].to_numpy(float), fc[cols].to_numpy(float),
                       equal_nan=True)                                 # the same values the query prints
