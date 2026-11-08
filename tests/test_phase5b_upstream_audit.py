"""Toy-fixture tests of the pure checks of the upstream audit (no real data, no model fits)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import phase5b_upstream_audit as ua  # noqa: E402


def toy() -> pd.DataFrame:
    n = 8
    return pd.DataFrame({
        "SEA_COD": ["232"] * 4 + ["241"] * 2 + ["242"] * 2,
        "CAT_DES_EN": ["A", "A", "B", "B"] * 2,
        "SALES_QTY": [100.0, np.nan, 0.0, 50.0, 10.0, np.nan, np.nan, 5.0],
        "SALES_AMT_FX_RATE": [900.0, np.nan, 0.0, 400.0, 100.0, np.nan, np.nan, 40.0],
        "PRICE_BASE_W_VAT": [10.0, 10.0, 20.0, 20.0, 10.0, 10.0, 12.0, 12.0],
        "STORE_DATE_FINAL": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01", "2024-05-01", "2024-12-01",
                             "2024-06-01", "2024-07-01"],
        "STORE_DATE_PLANNED": ["2024-01-01", "2024-02-20", "2024-03-01", "2025-06-01", "2024-05-01", "2024-12-01",
                               "2024-06-01", "2024-07-01"],
        "img_trusted": [True, True, False, True, True, True, False, False],
        "PROD_REF": list("abcdefgh"), "PROD_REF_EQUIV": list("abcdefgh")[:n],
    }).assign(log_sales=lambda d: np.log1p(d["SALES_QTY"].where(d["SALES_QTY"] > 0)))


def test_unknown_sales_counts_only_finished_seasons():
    u = ua.unknown_sales(toy())
    assert u["whole"] == 6 and u["no_row_whole"] == 2 and u["zero_whole"] == 1
    assert u["no_row_whole_launched"] == 1       # the 241 colourway launched in December is after the extraction
    assert u["no_row_by_season"]["242"] == 1


def test_planned_vs_final_and_images():
    p = ua.planned_vs_final(toy())
    assert p["differs"] == 2 / 8 and p["over_1_year"] == 1
    img = ua.image_availability(toy())
    assert img["242"] == 1.0 and img["232"] == 0.25


def test_realised_ratio_and_exposure():
    r = ua.realised_ratio(toy())
    assert abs(r["232"] - 0.65) < 1e-9            # (900/100)/10 and (400/50)/20 give 0.9 and 0.4
    e = ua.exposure_corr(toy().assign(CAT_DES_EN="A"), "232")
    assert e["weeks_max"] > e["weeks_min"]


def test_eta2_is_one_for_a_perfect_split_and_collapse_rare():
    d = pd.DataFrame({"CAT_DES_EN": ["A"] * 4, "log_sales": [1.0, 1.0, 3.0, 3.0], "FT": ["x", "x", "y", "y"]})
    assert abs(ua.eta2_within_category(d, "FT") - 1.0) < 1e-9
    X = pd.DataFrame({"c": pd.Categorical(["a", "a", "a", "b"])})
    out = ua.collapse_rare(X, ["c"], np.ones(4, bool), 2)
    assert list(out["c"].astype(str)) == ["a", "a", "a", "OTHER"]
