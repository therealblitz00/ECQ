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
    assert used <= set(p1.CONTRACT_COLUMNS) | {"FINISH_FAMILY"}  # FINISH_FAMILY is derived in Phase 2a


def test_main_writes_matrix_meta_and_report(checked_items, toy_dirs, monkeypatch):
    preview, _ = p1.finalise_items(checked_items.reset_index().assign(review_status="unreviewed"))
    for col in {c for cols, _ in f2a.CATEGORICAL.values() for c in cols} - set(preview.columns):
        preview[col] = None  # the toy data has only a few attribute columns
    preview["STORE_DATE_FINAL"], preview["SEA_COD"] = pd.Timestamp("2024-01-01"), "241"
    preview.to_parquet(checks.PROCESSED_DIR / p1.PREVIEW_FILE, index=False)
    monkeypatch.setattr(f2a, "PROCESSED_DIR", checks.PROCESSED_DIR)
    monkeypatch.setattr(f2a, "REPORT_DIR", toy_dirs / "outputs" / "phase2a")
    monkeypatch.setattr(f2a, "NEEDS_FILE", {"text": checks.EMB_DIR / "text.npy"})  # not in the toy data: skipped
    monkeypatch.setattr(f2a.phase2a_impute, "run", lambda: pd.DataFrame(columns=[checks.KEY, "column", "value"]))
    monkeypatch.setattr(f2a.phase2a_colour, "PALETTES_FILE", checks.EMB_DIR / "palettes.parquet")  # not in the toy data
    monkeypatch.setattr(f2a.phase2a_colour, "REFERENCE_FILE", checks.PROCESSED_DIR / "colour_reference.parquet")
    monkeypatch.setattr(f2a.phase2a_colour, "REPORT_DIR", toy_dirs / "outputs" / "phase2a")
    monkeypatch.setattr("sys.argv", ["phase2a_features.py"])
    f2a.main()

    m = np.load(checks.PROCESSED_DIR / "features_categorical.npy")
    assert set(json.loads((checks.PROCESSED_DIR / f2a.META_FILE).read_text(encoding="utf-8"))["blocks"]) == set(f2a.BLOCKS) - {"text"}
    meta = json.loads((checks.PROCESSED_DIR / f2a.META_FILE).read_text(encoding="utf-8"))
    assert meta["source"] == "items_preview" and meta["keys"] == sorted(preview[checks.KEY])
    assert meta["blocks"]["categorical"]["shape"] == list(m.shape) == [len(preview), len(meta["blocks"]["categorical"]["columns"])]
    assert (toy_dirs / "outputs" / "phase2a" / "features_report.md").exists()
    ref = pd.read_parquet(checks.PROCESSED_DIR / "colour_reference.parquet")
    assert set(ref["CLR_DES"]) == set(preview["CLR_DES"].dropna()) and ref["hex"].eq("").all()  # no palettes: no colour


# --------------------------------------------------------------------------- composition and numeric blocks
@pytest.fixture
def products() -> pd.DataFrame:
    return pd.DataFrame({
        checks.KEY: ["1_GD", "2_GD", "3_GD", "4_BK"],
        "CAT_DES_EN": ["Jewellery", "Jewellery", "Jewellery", "Wallet"],
        "COMPOSITION": ["Zinc; Pearl", " Pearl ;Zinc", "Zinc", None],
        "PRICE_BASE_W_VAT": [5.99, 6.99, 29.99, 29.99],
        "STORE_DATE_FINAL": pd.to_datetime(["2023-04-01", "2023-04-20", "2024-10-01", "2023-04-01"]),
        "SEA_COD": ["232", "232", "242", "232"],
    })


def test_composition_ignores_order_and_spaces_and_weights_rare_materials_more(products):
    m, columns, attrs = f2a.composition_block(products)
    assert columns == ["COMPOSITION=Pearl", "COMPOSITION=Zinc"]
    assert m[0] @ m[1] == pytest.approx(1.0, rel=1e-6)
    assert attrs["Pearl"]["weight"] > attrs["Zinc"]["weight"]
    shared = f2a.shared_attributes(m, attrs, 0, 2)
    assert set(shared) == {"Zinc"} and shared["Zinc"] < 1  # sharing only one of two materials is a partial match
    assert not m[3].any()  # no composition: never a match


def test_soft_bands_give_closer_values_a_higher_score():
    x = np.array([0.0, 0.1, 0.5, np.nan])
    m = f2a.soft_bands(x, np.arange(-0.3, 1.31, 0.05), 0.1)
    assert m[0] @ m[1] == pytest.approx(np.exp(-0.01 / 0.04), abs=0.02)
    assert m[0] @ m[2] < 0.01 and not m[3].any()


def test_numeric_compares_price_within_the_category(products):
    m, _, attrs = f2a.numeric_block(products)
    np.testing.assert_allclose(np.linalg.norm(m, axis=1), 1, rtol=1e-6)
    v = f2a.numeric_values(products)
    assert v.loc[3, "price_pct"] == 1.0  # the only wallet is the top of its own category
    near, far = f2a.shared_attributes(m, attrs, 0, 1), f2a.shared_attributes(m, attrs, 0, 2)
    assert near["launch"] > far.get("launch", 0) and "season" in near and "season" not in far
    assert set(attrs) == {"price", "launch", "season"}


# --------------------------------------------------------------------------- text block
def test_text_block_follows_this_table_row_order(tmp_path, monkeypatch):
    emb = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float16)  # stored for keys B, A
    np.save(tmp_path / "text.npy", emb)
    (tmp_path / "text.json").write_text(json.dumps({"keys": ["B", "A"]}), encoding="utf-8")
    monkeypatch.setattr(f2a, "EMB_DIR", tmp_path)
    m, columns, attrs = f2a.text_block(pd.DataFrame({checks.KEY: ["A", "B", "C"]}))
    assert m.tolist() == [[0, 1, 0], [1, 0, 0], [0, 0, 0]]  # C was not embedded: never a match
    assert len(columns) == 3 and attrs["text"]["filled"] == 2
