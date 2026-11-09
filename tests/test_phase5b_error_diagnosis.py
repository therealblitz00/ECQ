"""Building blocks of phase5b_error_diagnosis on hand-made data (no data files, no LightGBM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase5b_error_diagnosis as ed


def test_halves_keep_a_model_together():
    g = np.array(["a", "a", "b", "c", "c", "c", "d"])
    h = ed.halves(g)
    assert set(h) <= {0, 1}
    assert h[0] == h[1] and h[3] == h[4] == h[5]


def test_cross_fitted_range_covers_about_80_percent():
    rng = np.random.default_rng(1)
    resid = rng.normal(size=4000)
    side = np.arange(4000) % 2
    cov = ed.cross_fitted_range(resid, side)
    assert cov.mean() == pytest.approx(0.8, abs=0.03)


def test_range_per_segment_fixes_a_wider_segment():
    rng = np.random.default_rng(2)
    seg = np.r_[np.zeros(3000, int), np.ones(1000, int)]
    resid = np.r_[rng.normal(0, 1, 3000), rng.normal(0, 3, 1000)]     # segment 1 is three times noisier
    side = np.arange(4000) % 2
    one = ed.cross_fitted_range(resid, side)
    per = ed.cross_fitted_range(resid, side, seg)
    assert one[seg == 1].mean() < 0.6                                  # one range under-covers the noisy segment
    assert per[seg == 1].mean() == pytest.approx(0.8, abs=0.05)
    w = ed.width(resid, side, seg)
    assert np.nanmedian(w[seg == 1]) > np.nanmedian(w[seg == 0])


def test_leave_one_out_median():
    price = pd.Series([10.0, 12.0, 20.0, 5.0])
    group = pd.Series(["g", "g", "g", "solo"])
    loo = ed.leave_one_out_median(price, group)
    assert loo.tolist()[:3] == [16.0, 15.0, 11.0]
    assert np.isnan(loo.iloc[3])


def _items():
    return pd.DataFrame({ed.KEY: ["m1_A", "m1_B", "m2_A", "m3_A", "m4_A"],
                         "PROD_REF": ["m1", "m1", "m2", "m3", "m4"],
                         "SEA_COD": ["232", "232", "232", "241", "242"],
                         "CAT_DES_EN": "Bags", "GFA_DES_EN": "Tote", "GFS_DES_EN": ["s", "s", "s", "t", "s"],
                         "MATERIAL": "PU", "FINISHING": "Other",
                         "log_cost": np.log([10.0, 10.0, 10.3, 10.0, 10.0]),
                         "PRICE_BASE_W_VAT": [29.99, 29.99, 35.99, 25.99, 29.99]})


def test_noise_groups_one_colour_per_model_training_seasons_only():
    d = ed.noise_groups(_items(), ["SEA_COD", *ed.SUBFAMILY], 0.05)
    assert d["PROD_REF"].tolist() == ["m1", "m2", "m3"]                 # m1 once, m4 (242) out
    assert d.loc[d["PROD_REF"] == "m1", "group"].iloc[0] == d.loc[d["PROD_REF"] == "m2", "group"].iloc[0]


def test_noise_groups_accept_missing_text():
    items = _items()
    items["MATERIAL"] = pd.array(["PU", "PU", None, None, "PU"], dtype="string")
    d = ed.noise_groups(items, ed.NOISE_KEYS["strict"], 0.05)
    assert d["group"].str.contains("(missing)", regex=False).sum() == 2


def test_seen_subfamily():
    items = _items()
    pool = items["SEA_COD"].eq("232").to_numpy()
    assert ed.seen_subfamily(items, pool).tolist() == [True, True, True, False, True]


def test_boot_ci_contains_the_mean():
    rng = np.random.default_rng(3)
    x = rng.normal(5, 1, 400)
    lo, hi = ed.boot_ci(lambda r: float(np.mean(x[r])), np.arange(400).astype(str), n=200)
    assert lo < x.mean() < hi
