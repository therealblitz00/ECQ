"""Tables of phase5b_validation_errors on hand-made data (no data files, no LightGBM)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase5b_validation_errors as ve


def _toy() -> pd.DataFrame:
    n = 40
    t = pd.DataFrame({ve.KEY: [f"k{i}" for i in range(n)], "cat": ["a"] * 30 + ["b"] * 6 + ["c"] * 4,
                      "abs_error_units": np.r_[np.full(30, 1.0), np.full(6, 10.0), np.full(4, 2.0)],
                      "abs_error_log": np.r_[np.full(30, 0.1), np.full(6, 0.2), np.full(4, 0.9)],
                      "too_high": [True] * 20 + [False] * 20, "covered": [True] * 32 + [False] * 8})
    t["worst_units"] = ve.pe.top_share(t["abs_error_units"].to_numpy())
    t["worst_factor"] = ve.pe.top_share(t["abs_error_log"].to_numpy())
    return t


def test_summary_shares_rates_and_coverage():
    s = ve.summary(_toy(), "cat", ve.SALES_ERRS, min_group=5)
    assert list(s.index) == ["a", "b", ve.OTHER]
    assert s["share of worst units"].sum() == pytest.approx(1.0)
    assert s.loc["b", "share of worst units"] == pytest.approx(1.0)        # the 4 largest unit errors are in b
    assert s.loc["b", "in worst units (rate)"] == pytest.approx(4 / 6)
    assert s.loc[ve.OTHER, "share of worst factor"] == pytest.approx(1.0)  # the 4 largest factors are the c rows
    assert s.loc["a", "coverage of the 80% range"] == pytest.approx(1.0)
    assert s.loc[ve.OTHER, "coverage of the 80% range"] == pytest.approx(0.0)


def test_direction_counts_each_set():
    d = ve.direction(_toy(), ve.SALES_ERRS)
    assert d.loc["all products", "n"] == 40
    assert d.loc["worst 10% (units)", "n"] == 4
    assert d.loc["rest (in neither)", "n"] == 32
    assert d.loc["all products", "estimated too high"] == pytest.approx(0.5)


def test_subsample_keeps_models_together_and_is_reproducible():
    groups = np.repeat(np.arange(20).astype(str), 2)
    train = np.ones(40, bool)
    a = ve.subsample(train, groups, 0.5, seed=0)
    assert a.sum() == 20
    assert all(a[i] == a[i + 1] for i in range(0, 40, 2))
    assert (a == ve.subsample(train, groups, 0.5, seed=0)).all()
    assert ve.subsample(train, groups, 1.0, seed=3).all()


def test_season_comparison_joins_both_seasons():
    def t(n_b):
        d = pd.DataFrame({ve.KEY: [f"k{i}" for i in range(30 + n_b)], "cat": ["a"] * 30 + ["b"] * n_b,
                          "abs_error_eur": np.r_[np.ones(30), np.full(n_b, 5.0)],
                          "abs_error_pct": np.r_[np.full(30, 0.05), np.full(n_b, 0.3)]})
        d["worst_eur"] = ve.pe.top_share(d["abs_error_eur"].to_numpy())
        d["worst_pct"] = ve.pe.top_share(d["abs_error_pct"].to_numpy())
        return d
    c = ve.season_comparison(t(30), t(40), {"category": "cat"})
    assert list(c.index) == ["category: a", "category: b"]
    assert c.loc["category: b", "241: share of worst €"] == pytest.approx(1.0)
    assert c.loc["category: b", "242: share of products"] == pytest.approx(40 / 70)
    assert ve.season_comparison(t(30), None, {"category": "cat"}) is None
