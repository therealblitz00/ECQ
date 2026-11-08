"""Toy tests of the multi-item rules of phase2b_multi_item (no data files, no models)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase2b_multi_item as mi  # noqa: E402

K = mi.KEY


def test_units_case():
    assert mi.units_case("Singular") == "single"
    assert mi.units_case(None) == "single"
    assert mi.units_case(np.nan) == "single"
    assert mi.units_case("Set 1 Pair") == "pair"
    assert mi.units_case("3 Units (Different)") == "multi_different"
    assert mi.units_case("Set 3 Pairs") == "multi_same"
    assert mi.units_case("2 Units") == "multi_same"


def test_garment_box():
    box = [0.2, 0.0, 0.8, 1.0]
    assert mi.garment_box(box, "Top", "Apparel") == [0.2, 0.0, 0.8, 0.55]
    assert mi.garment_box(box, "Bottom", "Apparel") == [0.2, 0.45, 0.8, 1.0]
    assert mi.garment_box(box, "Dress", "Apparel") is None
    assert mi.garment_box(box, None, "Hand Bag") is None
    assert mi.garment_box(None, "Top", "Apparel")[3] == 0.55


def _items():
    keys = ["A_1", "B_1", "C_1", "D_1", "E_1", "F_1"]
    return pd.DataFrame({
        K: keys,
        "img_trusted": [True] * 6,
        "CAT_DES_EN": ["Jewellery", "Apparel", "Apparel", "Jewellery", "Jewellery", "Hand Bag"],
        "GFA_DES_EN": ["Earrings", "Shirt", "Shirt", "Rings", "Earrings", "Tote"],
        "OUTFIT": [None, "Top", "Top", None, None, None],
        "NUMBER_OF_UNITS": ["Set 1 Pair", "Singular", "Singular", "3 Units (Different)", "Singular", "Singular"],
    })


def test_build_suggestions_rules():
    items = _items()
    full = [0.1, 0.0, 0.9, 1.0]
    masks = pd.DataFrame({K: items[K], "box": [full] * 6, "packshot": [True, True, True, True, False, True],
                          "target_found": [True] * 6})
    person = pd.DataFrame({
        K: ["B_1", "C_1", "F_1"],
        "person_full": [0.9, 0.2, 0.9], "person_score": [0.3, 0.3, 0.3],
        "item_score": [0.6, 0.0, 0.0], "item_box": [[0.3, 0.1, 0.6, 0.4], None, None]})
    out = mi.build_suggestions(items, masks, person, {"accepted": []}, {"accepted": [], "rejected": {}}).set_index(K)
    assert out.at["A_1", "action"] == "keep"                       # pair of earrings is the normal case
    assert out.at["B_1", "action"] == "crop_suggested_reviewed_rule"   # small item box on a person
    assert out.at["B_1", "box_source"] == "owlv2_item_box"
    assert out.at["C_1", "action"] == "check_possible_person"      # OWLv2 only, packshot, CLIP says no
    assert out.at["D_1", "action"] == "flag_multi_piece" and out.at["D_1", "several_pieces"]
    assert out.at["E_1", "action"] == "keep_lifestyle"
    assert out.at["F_1", "action"] == "exclude_image" and out.at["F_1", "exclude_image"]
    # a reviewer's accept turns a suggestion into an applied crop, a reject keeps the photo as it is
    acc = mi.build_suggestions(items, masks, person, {"accepted": ["B_1"]}, {"accepted": [], "rejected": {}}).set_index(K)
    assert acc.at["B_1", "action"] == "crop_applied"
    rej = mi.build_suggestions(items, masks, person, {"accepted": []}, {"accepted": [], "rejected": {"B_1": "print"}}).set_index(K)
    assert rej.at["B_1", "action"] == "keep_rejected" and rej.at["B_1", "suggested_box"] is None


def test_counts_table_totals():
    items = _items()
    masks = pd.DataFrame({K: items[K], "box": [None] * 6, "packshot": [True] * 6, "target_found": [True] * 6})
    person = pd.DataFrame({K: [], "person_full": [], "person_score": [], "item_score": [], "item_box": []})
    df = mi.build_suggestions(items, masks, person, {}, {})
    t = mi.counts_table(df).set_index("category")
    assert t.at["All", "trusted photos"] == 6
    assert t.at["All", "several pieces"] == 1
    assert t.at["All", "pair (Set 1 Pair)"] == 1
