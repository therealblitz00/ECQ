"""Phase 5 explanations and Phase 5b estimates on hand-made data (no data files, no vision packages)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase3_engine as eng
import phase5_explain as ex
import phase5b_forecast as fc

KEY = eng.KEY


def unit(*rows) -> np.ndarray:
    m = np.array(rows, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


# --------------------------------------------------------------------------- explanations
class FakeContext:
    """Just what slice_parts needs: one block with two named slices."""

    def __init__(self):
        self.blocks = {"categorical": unit([1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0])}
        self.attrs = {"categorical": {"family": (0, 2), "colour": (2, 4)}}


def test_attribute_parts_add_up_to_the_block_similarity():
    ctx = FakeContext()
    m = ctx.blocks["categorical"]
    for q, c in [(0, 1), (0, 2), (1, 2)]:
        parts = ex.slice_parts(ctx, "categorical", q, c)
        assert sum(parts.values()) == pytest.approx(float(m[q] @ m[c]), abs=1e-6)


def test_a_slice_without_overlap_is_not_listed():
    ctx = FakeContext()
    parts = ex.slice_parts(ctx, "categorical", 0, 1)   # rows 0 and 1 share family[0] and colour[1] only
    assert set(parts) == {"family", "colour"}
    parts = ex.slice_parts(ctx, "categorical", 0, 2)   # share family[1] only
    assert set(parts) == {"family"}


# --------------------------------------------------------------------------- estimates
def test_constant_estimate_ranks_nothing():
    assert fc.spearman(np.array([1.0, 2, 3]), np.array([5.0, 5, 5])) == 0.0
    assert fc.spearman(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) == pytest.approx(1.0)


def test_sales_metrics_on_a_perfect_and_a_constant_estimate():
    y = np.arange(40, dtype=float)
    groups = np.array(["Rings|5|242"] * 20 + ["Rings|6|242"] * 20)
    perfect = fc.sales_metrics(y, y, groups)
    assert perfect["spearman_within_cat"] == pytest.approx(1.0)
    assert perfect["spearman_same_month"] == pytest.approx(1.0)
    assert perfect["top20_precision"] == pytest.approx(1.0)
    const = fc.sales_metrics(y, np.zeros(40), groups)
    assert const["spearman_within_cat"] == 0.0
    assert 0.0 <= const["top20_precision"] <= 0.5          # a random pick, not a perfect one


def test_price_metrics_snap_to_the_list_prices():
    points = np.array([5.99, 7.99, 9.99, 12.99])
    y = np.log(np.array([7.99, 9.99]))
    m = fc.price_metrics(y, np.log(np.array([7.80, 12.50])), points)
    assert m["exact_price_point"] == 0.5                    # 7.80 -> 7.99 right; 12.50 -> 12.99, one point off 9.99
    assert m["within_1_point"] == 1.0


def test_group_baseline_uses_only_the_training_rows():
    items = pd.DataFrame({"CAT_DES_EN": ["A", "A", "A", "B"], "GFA_DES_EN": ["x", "x", "y", "z"]})
    y = np.array([1.0, 3.0, 100.0, 7.0])
    fit = np.array([True, True, False, True])
    p = fc.group_baseline(items, y, fit, ["CAT_DES_EN", "GFA_DES_EN"])
    assert p[0] == p[1] == 2.0
    assert p[2] == 2.0                                      # unseen family: category median, never its own 100
    assert p[3] == 7.0


def test_pca_is_fitted_on_training_rows_and_keeps_missing_rows_missing():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(30, 6)).astype(np.float32)
    m[5] = 0                                                # no embedding
    out = fc.pca(m[:20], m, k=3)
    assert out.shape == (30, 3)
    assert np.isnan(out[5]).all()
    assert np.isfinite(out[np.arange(30) != 5]).all()


def test_analogues_never_use_the_products_own_model():
    items = pd.DataFrame({KEY: ["A_BK", "A_RD", "B_BK", "C_BK"], "PROD_REF": ["A", "A", "B", "C"],
                          "PROD_REF_EQUIV": ["A", "A", "B", "C"], "CAT_DES_EN": ["Rings"] * 4,
                          "target": [10.0, 20.0, 1.0, 3.0]})
    blocks = {"categorical": unit([1, 0], [1, 0], [0.6, 0.8], [0.8, 0.6])}
    pool = np.ones(4, bool)
    est, keys = fc.analogues(items, blocks, {"categorical": 1.0}, "target", pool, k=2)
    assert "A_RD" not in keys[0] and "A_BK" not in keys[0]   # the other colour of A is never an analogue of A_BK
    assert set(keys[0]) == {"B_BK", "C_BK"}
    assert 1.0 <= est[0] <= 3.0


def test_constant_estimate_with_float_noise_in_its_std_still_ranks_nothing():
    p = np.full(30, 7.3)                                    # np.std of this is ~1e-15, not 0
    assert fc.spearman(np.arange(30.0), p) == 0.0
