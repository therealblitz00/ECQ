"""Phase 1b `--add-missing`: new items are appended without touching the stored embeddings.

The CLIP/detector step is replaced by a fake, so the test needs no vision packages.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import phase1_checks as checks
import phase1b_image_audit as pb

AUDIT_COLS = ["group_pred", "p_group_pred", "p_group_expected", "colour_pred", "p_colour_expected"]


def fake_embed_and_score(items, clip, batch_size, t0):
    rng = np.random.default_rng(len(items))
    emb = rng.normal(size=(len(items), 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    masks = pd.DataFrame({checks.KEY: items[checks.KEY], "img_file": items["img_file"], "white_border": 1.0,
                          "packshot": True, "mask_method": "threshold", "box": None, "target_found": True,
                          "det_score": np.nan, "det_strongest": None, "det_strongest_score": np.nan})
    audit = items[[checks.KEY, "PROD_REF", "CAT_DES_EN", "GFA_DES_EN", "CLR_DES", "group_expected",
                   "colour_expected"]].copy()
    audit["group_pred"], audit["p_group_pred"] = audit["group_expected"], 0.9
    audit["p_group_expected"], audit["colour_pred"], audit["p_colour_expected"] = 0.9, audit["colour_expected"], 0.9
    return emb, masks.reset_index(drop=True), audit.reset_index(drop=True)


@pytest.fixture
def stored(toy_data, monkeypatch):
    """Embeddings already computed for all toy items with a photo except 100005_GN (the repaired path)."""
    monkeypatch.setattr(pb, "EMB_DIR", checks.EMB_DIR)
    monkeypatch.setattr(pb, "IMG_DIR", checks.IMG_DIR)
    monkeypatch.setattr(pb, "REPORT_DIR", toy_data / "outputs" / "phase1b")
    monkeypatch.setattr(pb, "embed_and_score", fake_embed_and_score)
    monkeypatch.setattr(pb, "Clip", lambda: None)
    items = pb._items_with_image()
    old = items[items[checks.KEY] != "100005_GN"].reset_index(drop=True)
    emb, masks, audit = fake_embed_and_score(old, None, 64, 0)
    pb.save_outputs(emb, masks, pb.compare_all(audit, masks, emb), "", None, created="2026-10-09T00:00:00+00:00")
    return emb


def test_add_missing_appends_only_the_new_item(stored):
    pb.add_missing(batch_size=64, t0=0)
    meta = json.loads((checks.EMB_DIR / "image_clip.json").read_text(encoding="utf-8"))
    emb = np.load(checks.EMB_DIR / "image_clip.npy")
    audit = pd.read_parquet(checks.EMB_DIR / "image_audit.parquet")
    masks = pd.read_parquet(checks.EMB_DIR / "masks.parquet")

    assert meta["keys"][-1] == "100005_GN" and meta["n"] == len(meta["keys"]) == len(emb)
    assert meta["keys"] == audit[checks.KEY].tolist() == masks[checks.KEY].tolist()
    assert meta["created"] == "2026-10-09T00:00:00+00:00" and "updated" in meta
    np.testing.assert_array_equal(emb[:-1], stored.astype(np.float16))  # stored vectors untouched
    assert {"vis_issues", "vis_note", "nn_same_type_share"} <= set(audit.columns)


def test_add_missing_does_nothing_when_everything_is_embedded(stored, capsys):
    pb.add_missing(batch_size=64, t0=0)
    pb.add_missing(batch_size=64, t0=0)
    assert "Nothing to add" in capsys.readouterr().out
    assert len(np.load(checks.EMB_DIR / "image_clip.npy")) == len(stored) + 1
