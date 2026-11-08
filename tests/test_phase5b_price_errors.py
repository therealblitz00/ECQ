"""Tables of phase5b_price_errors on hand-made data (no data files, no LightGBM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase5b_price_errors as pe


def test_top_share_takes_the_largest_and_is_stable_on_ties():
    v = np.array([1.0, 5.0, 5.0, 2.0, 9.0, 0.0, 3.0, 4.0, 6.0, 7.0])
    assert pe.top_share(v, 0.2).tolist() == [False] * 4 + [True] + [False] * 4 + [True]
    ties = pe.top_share(np.array([1.0, 1.0, 1.0]), 0.34)
    assert ties.sum() == 2 and ties[:2].all()


def test_price_band_and_markup_class():
    bands = pe.price_band(pd.Series([3.99, 10.0, 19.99, 45.0, 250.0])).tolist()
    assert bands == ["< 10 €", "10-20 €", "10-20 €", "40-80 €", ">= 80 €"]
    cls = pe.markup_class(pd.Series([np.nan, 0.5, 1.0, 1.25, 2.0])).tolist()
    assert cls == ["no cost", "below 0.8x", "0.8-1.25x", "0.8-1.25x", "above 1.25x"]


def _toy_errors() -> pd.DataFrame:
    n = 40
    cat = ["a"] * 30 + ["b"] * 6 + ["c"] * 4
    err = np.r_[np.full(30, 1.0), np.full(6, 10.0), np.full(4, 2.0)]
    price = np.r_[np.full(30, 20.0), np.full(6, 50.0), np.full(4, 4.0)]
    t = pd.DataFrame({pe.KEY: [f"k{i}" for i in range(n)], "cat": cat, "abs_error_eur": err,
                      "list_price": price, "abs_error_pct": err / price,
                      "direction": ["too high"] * 20 + ["too low"] * 20})
    t["worst_eur"] = pe.top_share(t["abs_error_eur"].to_numpy())
    t["worst_pct"] = pe.top_share(t["abs_error_pct"].to_numpy())
    return t


def test_group_summary_pools_small_groups_and_shares_add_up():
    s = pe.group_summary(_toy_errors(), "cat", min_group=5)
    assert list(s.index) == ["a", "b", pe.OTHER]       # c has 4 products: pooled
    assert s["n"].sum() == 40
    assert s["share of worst €"].sum() == pytest.approx(1.0)
    assert s["share of worst %"].sum() == pytest.approx(1.0)
    # the 4 worst in euros are all in b (error 10): composition 100%, rate 4 / 6
    assert s.loc["b", "share of worst €"] == pytest.approx(1.0)
    assert s.loc["b", "in worst € (rate)"] == pytest.approx(4 / 6)
    # the 4 worst in % are the cheap c products (2 / 4 = 50%), pooled into the "other" row
    assert s.loc[pe.OTHER, "share of worst %"] == pytest.approx(1.0)
    assert s.loc["a", "MAE €"] == pytest.approx(1.0)


def test_direction_table_counts():
    t = _toy_errors().assign(error_eur=lambda d: np.where(d["direction"] == "too high", 1, -1) * d["abs_error_eur"])
    d = pe.direction_table(t)
    assert d.loc["all test products", "n"] == 40
    assert d.loc["all test products", "estimated too high"] == pytest.approx(0.5)
    assert d.loc["worst 10% in €", "estimated too high"] == pytest.approx(0.0)   # b rows are 30-35: too low


def test_no_cost_comparison_picks_the_smaller_error():
    t = pd.DataFrame({"cost_status": ["no cost (or <= 0)"] * 4 + ["has cost"] * 2,
                      "list_price": [10.0, 10.0, 20.0, 20.0, 10.0, 10.0],
                      "estimate": [12.0, 8.0, 24.0, 16.0, 10.0, 10.0],
                      "estimate_analogue": [11.0, 9.0, 21.0, 19.0, 15.0, 5.0]})
    res = pe.no_cost_comparison(t, np.array(["m1", "m2", "m3", "m4", "m5", "m6"]), boot=200)
    tab = res["table"]
    assert tab.loc["without cost", "n"] == 4
    assert tab.loc["without cost", "MAE € chosen model"] == pytest.approx(3.0)
    assert tab.loc["without cost", "MAE € analogue"] == pytest.approx(1.0)
    assert res["ci"][0] > 0          # the analogue has the smaller error in every resample


def test_realised_vs_list_ignores_products_without_units():
    items = pd.DataFrame({"CAT_DES_EN": ["a", "a", "a", "b"], "SEA_COD": ["232", "232", "232", "241"],
                          "SALES_QTY": [10.0, 30.0, 0.0, np.nan], "SALES_AMT_FX_RATE": [80.0, 300.0, 50.0, np.nan],
                          "PRICE_BASE_W_VAT": [10.0, 10.0, 10.0, 10.0]})
    r = pe.realised_vs_list(items)
    row = r.loc[("a", "232")]
    assert row["n"] == 2                                   # 0 units with an amount: realised price unknown
    assert row["median"] == pytest.approx(0.9)
    assert row["unit-weighted mean"] == pytest.approx((0.8 * 10 + 1.0 * 30) / 40)
    assert row["above list price"] == pytest.approx(0.0)
    assert ("b", "241") not in r.index
    assert r.loc[("All", "all"), "n"] == 2
