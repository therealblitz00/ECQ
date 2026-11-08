"""Comparison and decision rule of phase5b_nocost_rule on hand-made data (no data files, no LightGBM)."""
from __future__ import annotations

import numpy as np
import pytest

import phase5b_nocost_rule as nr


def _toy():
    y = np.log(np.array([10.0, 20.0, 10.0, 20.0, 10.0, 10.0]))
    fallback = np.log(np.array([13.0, 26.0, 7.0, 14.0, 10.0, 10.0]))
    analogue = np.log(np.array([11.0, 21.0, 9.0, 19.0, 50.0, 50.0]))
    chosen = np.log(np.array([13.0, 26.0, 7.0, 14.0, 11.0, 9.0]))     # = fallback where there is no cost
    rows = np.ones(6, bool)
    no_cost = np.array([True, True, True, True, False, False])
    groups = np.array(["m1", "m2", "m3", "m4", "m5", "m6"])
    return y, fallback, analogue, chosen, rows, no_cost, groups


def test_compare_only_scores_products_without_cost():
    r = nr.compare(*_toy(), boot=200)
    assert r["n_no_cost"] == 4 and r["n_eval"] == 6
    assert r["fallback"]["mae_eur"] == pytest.approx((3 + 6 + 3 + 6) / 4)
    assert r["analogue"]["mae_eur"] == pytest.approx((1 + 1 + 1 + 1) / 4)
    assert r["fallback"]["too_high"] == pytest.approx(0.5)


def test_rule_changes_only_the_products_without_cost():
    r = nr.compare(*_toy(), boot=200)
    cur, new = r["overall"]["current"]["mae_eur"], r["overall"]["with the rule"]["mae_eur"]
    assert cur == pytest.approx((3 + 6 + 3 + 6 + 1 + 1) / 6)
    assert new == pytest.approx((1 + 1 + 1 + 1 + 1 + 1) / 6)   # products with a cost keep the chosen model


def test_decision_needs_the_whole_interval_above_zero():
    r = nr.compare(*_toy(), boot=200)
    assert r["ci_eur"][0] > 0 and nr.decide(r)
    y, fb, an, ch, rows, nc, g = _toy()
    worse = nr.compare(y, an, fb, ch, rows, nc, g, boot=200)          # estimates swapped: the analogue loses
    assert not nr.decide(worse)
    few = nr.compare(y, fb, an, ch, rows, np.array([True, False, False, False, False, False]), g, boot=50)
    assert few["ci_eur"] is None and not nr.decide(few)
