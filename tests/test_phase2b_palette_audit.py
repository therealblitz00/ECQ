"""Phase 2b palette audit (src/phase2b_palette_audit.py) on toy tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase2b_palette_audit as au
from config import KEY


def test_accepted_names_cover_shades_and_skip_metal_and_multicolour():
    assert "burgundy" in au.accepted_names("Burgundy") and "red" in au.accepted_names("Burgundy")
    assert "light blue" in au.accepted_names("Light Blue")
    assert au.accepted_names("Gold") is None and au.accepted_names("Bright Multicolor") is None
    assert au.accepted_names(np.nan) is None and au.accepted_names("Zzz") is None


def _toy():
    items = pd.DataFrame({KEY: ["a", "b", "c", "d"], "CLR_DES": ["Black", "Gold", "Red", "Bright Multicolor"],
                          "img_trusted": True, "img_colour_trusted": True})
    masks = pd.DataFrame({KEY: ["a", "b", "c", "d"], "packshot": [True, True, False, True],
                          "target_found": [True, True, True, True]})
    persons = pd.DataFrame({KEY: ["a"], "person_score": [0.3], "person_full": [0.9], "item_box": [None]})
    audit = pd.DataFrame({KEY: ["a", "b", "c", "d"], "group_expected": ["bag"] * 4,
                          "det_strongest": [None, None, None, None]})
    return items, masks, persons, audit


def test_photo_types_follow_the_rule_order():
    t = au.photo_types(*_toy()).set_index(KEY)
    assert t.loc["a", "ptype"] == "person"
    assert t.loc["b", "ptype"] == "metal name"
    assert t.loc["c", "ptype"] == "not a packshot"
    assert t.loc["d", "ptype"] == "multicolour name"


def test_agreement_counts_only_checked_names():
    t = pd.DataFrame({KEY: list("abc"), "CLR_DES": ["Black", "Black", "Gold"], "ptype": ["plain packshot"] * 3,
                      "dom_name": ["black", "white", "beige"], "top2": [{"black"}, {"white", "black"}, {"beige"}],
                      "dist_ref": [1.0, 30.0, np.nan]})
    out = au.agreement(t).loc["plain packshot"]
    assert out["checked"] == 2 and out["dominant fits CLR_DES"] == pytest.approx(0.5)
    assert out["top 2 fit CLR_DES"] == pytest.approx(1.0)


def test_sample_pairs_stay_inside_the_group_and_auc():
    rng = np.random.default_rng(0)
    groups = np.array(["x"] * 5 + ["y"] * 5 + ["z"])
    i, j = au.sample_pairs(groups, 200, rng)
    assert (i != j).all() and (groups[i] == groups[j]).all() and "z" not in set(groups[i])
    assert au.auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert au.auc(np.array([1.0]), np.array([1.0])) == 0.5
