"""Phase 1 cleaning (src/phase1_clean.py): finalise_items and the preview / clean tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase1_clean as p1
import sprint1_preprocess as sp


# --------------------------------------------------------------------------- single steps
def test_parse_dates_handles_the_three_formats():
    s = pd.Series([20230726, 20231206.0, "2023-12-06", "2023-09-10 21:40:36.310", np.nan, "garbage"], dtype=object)
    out = p1.parse_dates(s)
    assert out.dtype == "datetime64[ns]"
    assert out[:3].dt.strftime("%Y-%m-%d").tolist() == ["2023-07-26", "2023-12-06", "2023-12-06"]
    assert out[3] == pd.Timestamp("2023-09-10 21:40:36.310")
    assert out[4:].isna().all()


def test_unify_casing_keeps_the_most_common_spelling():
    s = pd.Series(["Golden Basics", "GOLDEN BASICS", "Golden Basics", None, "Neptune"])
    out, n = p1.unify_casing(s)
    assert out.tolist()[:3] == ["Golden Basics"] * 3
    assert out.isna().tolist() == [False, False, False, True, False]
    assert n == 1


def test_unify_casing_ignores_non_text_columns():
    s = pd.Series([1, 2, None], dtype=object)
    assert p1.unify_casing(s) == (s, 0)


def frame(**cols) -> pd.DataFrame:
    n = len(next(iter(cols.values())))
    base = {sp.KEY: [f"{100000 + i}_BK" for i in range(n)], "CAT_DES_EN": ["Jewellery"] * n}
    return pd.DataFrame({**base, **cols})


def test_placeholders_are_missing_only_where_they_mean_no_value():
    items = frame(DIMENSION=["Not Applicable", "UNDEFINED", "Undefined", "Long"],
                  PRINT_TYPE=["Others", "No Print", "Others", "Floral - Flowers"],
                  GFA_DES_EN=["Others", "Rings", "Rings", "Rings"],
                  DISTRIBUTION_BLOCK=["Without Block", "Heart_Love", "Without Block", "Seashell"])
    out, log = p1.finalise_items(items)
    assert out["DIMENSION"].isna().tolist() == [True, True, True, False]
    assert out["PRINT_TYPE"].tolist() == ["Others", "No Print", "Others", "Floral - Flowers"]
    assert out.at[0, "GFA_DES_EN"] == "Others"
    assert log["placeholders"] == {"DIMENSION": 3, "DISTRIBUTION_BLOCK": 2}


def test_sales_features_guard_against_zero_and_missing_sales():
    items = frame(SALES_QTY=[0.0, 99.0, np.nan, 9.0], SALES_AMT_FX_RATE=[10.0, 990.0, np.nan, 90.0])
    out, _ = p1.finalise_items(items)
    assert out["has_sales"].tolist() == [True, True, False, True]
    assert out["log_sales_qty"].round(3).tolist()[:2] == [0.0, 4.605]
    assert out["realised_price"].isna().tolist() == [True, False, True, False]
    assert out.at[1, "realised_price"] == 10.0
    assert out["sales_pct_in_cat"].tolist()[:2] == [1 / 3, 1.0]


@pytest.mark.parametrize("issues, has_image, status, trusted, colour", [
    ("", True, "unreviewed", True, True),
    ("SALES_MISSING", True, "unreviewed", True, True),
    ("IMG_SHARED", True, "unreviewed", True, False),  # right product, colour doubtful
    ("VIS_COLOUR_MISMATCH", True, "unreviewed", True, False),
    ("VIS_TYPE_MISMATCH", True, "unreviewed", False, False),  # photo of another product
    ("IMG_REF_MISMATCH;IMG_SHARED", True, "unreviewed", False, False),
    ("VIS_TYPE_MISMATCH", True, "ok", True, True),  # a reviewer said "Looks right"
    ("IMG_COLOUR_MISMATCH", True, "fix", True, True),  # "Fix data": the photo was fine
    ("VIS_TYPE_MISMATCH", True, "team_review", False, False),
    ("", False, "ok", False, False),  # no photo (or "Wrong photo" removed it)
])
def test_photo_trust(issues, has_image, status, trusted, colour):
    out, _ = p1.finalise_items(frame(issues=[issues], has_image=[has_image], review_status=[status]))
    assert (bool(out.at[0, "img_trusted"]), bool(out.at[0, "img_colour_trusted"])) == (trusted, colour)


# --------------------------------------------------------------------------- whole table
@pytest.fixture
def preview(checked_items) -> pd.DataFrame:
    items, _ = p1.finalise_items(checked_items.reset_index().assign(review_status="unreviewed"))
    return items


def test_finalise_keeps_one_row_per_colourway_and_drops_noise_columns(preview, checked_items):
    assert preview[sp.KEY].tolist() == sorted(checked_items.index)
    for col in ["PROD_COD", "SZ_DES", "BAR_COD", "PROD_DES", "img_md5", "desc_type_word"]:
        assert col not in preview.columns
    assert {"PROD_DES_BASE", "sizes", "n_skus", "img_trusted", "log_sales_qty"} <= set(preview.columns)


def test_finalise_is_idempotent(preview):
    again, _ = p1.finalise_items(preview)
    pd.testing.assert_frame_equal(again, preview)


def test_preview_cli_writes_table_and_report(checked_items, monkeypatch):
    items, ctx = sp.run_checks(verify_files=False)
    sp.write_check_outputs(items, ctx)
    monkeypatch.setattr("sys.argv", ["phase1_clean.py"])
    p1.main()
    out = pd.read_parquet(sp.PROCESSED_DIR / p1.PREVIEW_FILE)
    assert len(out) == 4 and set(out["review_status"]) == {"unreviewed"}
    assert "Photos the visual block may use" in (p1.REPORT_DIR / "clean_report.md").read_text(encoding="utf-8")


def test_load_items_prefers_the_reviewed_table(toy_dirs):
    with pytest.raises(SystemExit):
        p1.load_items()
    pd.DataFrame({sp.KEY: ["A"]}).to_parquet(sp.PROCESSED_DIR / p1.PREVIEW_FILE)
    assert p1.load_items()[sp.KEY].tolist() == ["A"]
    pd.DataFrame({sp.KEY: ["B"]}).to_parquet(sp.PROCESSED_DIR / p1.CLEAN_FILE)
    assert p1.load_items()[sp.KEY].tolist() == ["B"]
