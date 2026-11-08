"""CatBoost comparison on a toy table (a few trees, no data files)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import phase5b_catboost as cbm


def toy() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    n = 120
    cat = pd.Series(rng.choice(["a", "b", None], n), dtype="string")
    X = pd.DataFrame({"CAT": pd.Categorical(cat), "x": rng.normal(size=n)})
    y = (X["CAT"] == "a").to_numpy() * 1.0 + X["x"].to_numpy() * 0.5 + rng.normal(scale=0.1, size=n)
    return X, y


def test_missing_category_becomes_a_level_and_numbers_stay():
    X, _ = toy()
    out, cats = cbm.cb_frame(X)
    assert cats == ["CAT"]
    assert set(out["CAT"]) <= {"a", "b", "NA"} and "NA" in set(out["CAT"])
    assert out["x"].equals(X["x"])


def test_fit_is_deterministic_and_explanations_add_up():
    X, y = toy()
    rows = np.arange(len(y)) < 90
    p = {"iterations": 25, "depth": 3}
    m1, a = cbm.fit_cb(X, y, rows, p)
    _, b = cbm.fit_cb(X, y, rows, p)
    assert np.allclose(a, b)
    assert np.corrcoef(a[~rows], y[~rows])[0, 1] > 0.5
    assert cbm.shap_check(m1, X, ~rows)["max_additivity_error"] < 1e-4


def test_tune_picks_the_lowest_error():
    best, table = cbm.tune(lambda p: abs(p["depth"] - 6) + p["learning_rate"],
                           {"depth": (4, 6), "learning_rate": (0.03, 0.08)})
    assert best == {"depth": 6, "learning_rate": 0.03}
    assert len(table) == 4
