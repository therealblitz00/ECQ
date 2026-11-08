"""Landed-cost flags (toy table, no model is trained) and the WAPE helper of the exposure script."""
from __future__ import annotations

import numpy as np
import pandas as pd
import phase5b_cost_check as cc
import phase5b_exposure as expo
from config import KEY


def toy_items() -> pd.DataFrame:
    rows = []
    # 25 ordinary products of one sub-family: cost 4, price 12 (markup 3), one model per product
    for i in range(25):
        rows.append((f"A{i}_BK", f"A{i}", 4.0, 12.0))
    rows += [("B1_BK", "B1", 20.0, 10.0),      # cost above the price
             ("C1_BK", "C1", 0.0, 10.0),       # cost not positive
             ("D1_BK", "D1", 1.0, 12.0), ("D1_WT", "D1", 8.0, 12.0),     # 8x apart from the other colour
             ("E1_BK", "E1", 4.0, 60.0)]       # real but unusual markup (15x): suspect, not certain
    df = pd.DataFrame(rows, columns=[KEY, "PROD_REF", "PRICE_LCP", "PRICE_BASE_W_VAT"])
    return df.assign(CAT_DES_EN="Hand Bag", GFA_DES_EN="Bag", GFS_DES_EN="Shopper", SEA_COD="232")


def flags() -> pd.Series:
    t = cc.flag_costs(toy_items())
    return t.set_index(KEY)["cost_flag"]


def test_certain_flags():
    f = flags()
    assert f["B1_BK"] == "certain" and f["C1_BK"] == "certain"
    assert f["D1_BK"] == "certain" and f["D1_WT"] == "certain"      # each colour is far from the other


def test_ordinary_products_are_not_flagged():
    f = flags()
    assert (f[[f"A{i}_BK" for i in range(25)]] == "none").all()


def test_unusual_markup_is_only_suspect():
    assert flags()["E1_BK"] == "suspect"


def test_flags_never_change_the_input():
    items = toy_items()
    before = items.copy()
    cc.flag_costs(items)
    pd.testing.assert_frame_equal(items, before)


def test_wape_is_units_missed_over_real_units():
    y = np.log1p(np.array([100.0, 300.0]))
    p = np.log1p(np.array([150.0, 250.0]))
    assert abs(expo.wape(y, p) - (50 + 50) / 400) < 1e-9


def test_with_weeks_adds_one_column():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    out = expo.with_weeks(X, np.array([10.0, 20.0]))
    assert list(out.columns) == ["a", "weeks_on_sale"] and list(X.columns) == ["a"]
