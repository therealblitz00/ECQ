"""Aggregations and metrics of phase5b_metrics on hand-made data (no data files)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase5b_metrics as mt


def test_weighted_quantile_and_mode():
    v, w = np.array([1.0, 2.0, 10.0]), np.array([1.0, 1.0, 5.0])
    assert mt.wquantile(v, w, 0.5) == 10.0
    assert mt.wquantile(v, np.ones(3), 0.5) == 2.0
    assert np.isnan(mt.wquantile(v, np.zeros(3)))
    assert mt.mode_value(np.array([3.0, 4.0, 4.0])) == 4.0
    assert mt.mode_value(np.array([3.0, 4.0, 4.0]), np.array([9.0, 1.0, 1.0])) == 3.0


def test_trimmed_mean_drops_the_extremes():
    assert mt.trimmed(np.array([0.0, 5.0, 5.0, 5.0, 100.0])) == 5.0


def test_group_aggregate_median_matches_the_baseline_of_phase5b():
    import phase5b_forecast as fc
    items = pd.DataFrame({"CAT_DES_EN": ["a", "a", "a", "b", "b", "c"], "g": ["x", "x", "y", "x", "x", "x"]})
    y = np.array([1.0, 3.0, 10.0, 4.0, 6.0, np.nan])
    fit = np.array([True, True, True, True, True, False])
    mine = mt.group_aggregate(items, y, fit, ["CAT_DES_EN", "g"], mt.GROUP_AGGS["median (current)"])
    assert np.allclose(mine, fc.group_baseline(items, y, fit, ["CAT_DES_EN", "g"]))


def test_units_metrics_split_over_and_under_order():
    u = np.array([100.0, 100.0, 100.0])
    y = np.log1p(u)
    p = np.log1p(np.array([150.0, 50.0, 100.0]))
    m = mt.units_metrics(y, p)
    assert m["wape"] == pytest.approx(100 / 300)
    assert m["over_units"] == pytest.approx(50 / 300) and m["under_units"] == pytest.approx(50 / 300)
    assert m["bias_units"] == pytest.approx(0.0)
    assert m["share_over"] == pytest.approx(1 / 3)


def test_price_metrics_points():
    pts = np.array([10.0, 15.0, 20.0])
    y = np.log(np.array([10.0, 15.0, 20.0]))
    m = mt.price_metrics_more(pts)(y, np.log(np.array([10.0, 20.0, 20.0])))
    assert m["exact_price_point"] == pytest.approx(2 / 3) and m["within_1_point"] == pytest.approx(1.0)
    assert m["over_eur"] == pytest.approx(5 / 3)


def test_bootstrap_table_interval_contains_the_value_and_ref_has_no_diff():
    rng = np.random.default_rng(0)
    y = rng.uniform(2, 8, 200)
    preds = {"a": y + rng.normal(0, 0.5, 200), "b": y + rng.normal(0, 1.0, 200)}
    t = mt.bootstrap_table(mt.aggregation_metrics, y, preds, np.arange(200) // 2, "a", n=40)
    r = t[(t.model == "b") & (t.metric == "mae_log")].iloc[0]
    assert r.ci_lo <= r.value <= r.ci_hi and r.diff_vs_ref > 0
    assert t[(t.model == "a")]["diff_vs_ref"].isna().all()
