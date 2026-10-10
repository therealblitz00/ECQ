"""Phase 2a (src/phase2a_features.py): the categorical block and its files."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import phase1_clean as p1
import phase1_checks as checks
import phase2a_features as f2a

SPEC = {
    "CAT": (["CAT_DES_EN"], 1.0),
    "GFA": (["CAT_DES_EN", "GFA_DES_EN"], 3.0),
    "CLR_DES": (["CLR_DES"], 2.0),
    "FINISHING": (["FINISHING"], 1.5),
}


@pytest.fixture
def items() -> pd.DataFrame:
    return pd.DataFrame({
        checks.KEY: ["1_BK", "2_BK", "3_GD", "4_BK"],
        "CAT_DES_EN": ["Jewellery", "Jewellery", "Jewellery", "Apparel"],
        "GFA_DES_EN": ["Others", "Others", "Earrings", "Others"],
        "CLR_DES": ["Black", "Black", "Gold", "Black"],
        "FINISHING": ["Golden", None, "Golden", None],
    })


def test_rows_are_unit_length_and_dot_product_is_the_weighted_share(items):
    m, columns, attrs = f2a.categorical_block(items, SPEC)
    np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1, rtol=1e-6)
    # 1 and 2 share CAT (1), GFA (3) and colour (2); 1 also has a finish (1.5) that 2 lacks.
    expected = 6 / np.sqrt(7.5 * 6)
    assert m[0] @ m[1] == pytest.approx(expected, rel=1e-6)
    assert len(columns) == attrs["FINISHING"]["stop"] == m.shape[1]


def test_family_only_matches_inside_its_category(items):
    m, _, attrs = f2a.categorical_block(items, SPEC)
    # "Others" in Jewellery and "Others" in Apparel are different families.
    assert "GFA" not in f2a.shared_attributes(m, attrs, 0, 3)
    assert set(f2a.shared_attributes(m, attrs, 0, 1)) == {"CAT", "GFA", "CLR_DES"}


def test_missing_value_never_counts_as_a_match(items):
    m, _, attrs = f2a.categorical_block(items, SPEC)
    sl = slice(attrs["FINISHING"]["start"], attrs["FINISHING"]["stop"])
    assert not m[[1, 3], sl].any()
    assert attrs["FINISHING"]["filled"] == 2


def test_shared_attributes_sum_to_the_score(items):
    m, _, attrs = f2a.categorical_block(items, SPEC)
    assert sum(f2a.shared_attributes(m, attrs, 0, 2).values()) == pytest.approx(float(m[0] @ m[2]), rel=1e-6)


def test_columns_do_not_depend_on_row_order(items):
    _, cols_a, _ = f2a.categorical_block(items, SPEC)
    _, cols_b, _ = f2a.categorical_block(items.iloc[::-1], SPEC)
    assert cols_a == cols_b


def test_default_spec_uses_only_columns_of_the_phase1_contract():
    used = {c for cols, _ in f2a.CATEGORICAL.values() for c in cols}
    assert used <= set(p1.CONTRACT_COLUMNS)


def test_main_writes_matrix_meta_and_report(checked_items, toy_dirs, monkeypatch):
    preview, _ = p1.finalise_items(checked_items.reset_index().assign(review_status="unreviewed"))
    for col in {c for cols, _ in f2a.CATEGORICAL.values() for c in cols} - set(preview.columns):
        preview[col] = None  # the toy data has only a few attribute columns
    preview.to_parquet(checks.PROCESSED_DIR / p1.PREVIEW_FILE, index=False)
    monkeypatch.setattr(f2a, "PROCESSED_DIR", checks.PROCESSED_DIR)
    monkeypatch.setattr(f2a, "REPORT_DIR", toy_dirs / "outputs" / "phase2a")
    monkeypatch.setattr("sys.argv", ["phase2a_features.py"])
    f2a.main()

    m = np.load(checks.PROCESSED_DIR / "features_categorical.npy")
    meta = json.loads((checks.PROCESSED_DIR / f2a.META_FILE).read_text(encoding="utf-8"))
    assert meta["source"] == "items_preview" and meta["keys"] == sorted(preview[checks.KEY])
    assert meta["blocks"]["categorical"]["shape"] == list(m.shape) == [len(preview), len(meta["blocks"]["categorical"]["columns"])]
    assert (toy_dirs / "outputs" / "phase2a" / "features_report.md").exists()
