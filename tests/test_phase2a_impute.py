"""Phase 2a inferred attributes (src/phase2a_impute.py) on hand-made data."""
from __future__ import annotations

import numpy as np
import pandas as pd

import phase2a_impute as imp

KEY = imp.KEY


def test_finish_family_is_a_fixed_mapping():
    items = pd.DataFrame({"FINISHING": ["Worn Gold", "Rodhium", "Black Nikel", None]})
    assert imp.add_finish_family(items)["FINISH_FAMILY"].tolist()[:3] == ["gold", "silver", "black"]
    assert set(imp.FAMILY_OF) == {v for vals in imp.FINISH_FAMILY.values() for v in vals}


def test_sibling_value_only_when_the_other_colourways_agree():
    items = pd.DataFrame({"PROD_REF": ["1", "1", "1", "2", "2", "2"],
                          "MATERIAL": ["Cotton", None, "Cotton", "Linen", "Viscose", None]})
    sib = imp.sibling_values(items, "MATERIAL")
    assert sib[1] == "Cotton" and pd.isna(sib[5])


def test_single_material_composition_rule():
    items = pd.DataFrame({"COMPOSITION": ["Cotton", " Viscose ", "Cotton; Elastane", "Ramie", None]})
    out = imp.single_material(items, {"Cotton", "Viscose"})
    assert out.tolist()[:2] == ["Cotton", "Viscose"] and out[2:].isna().all()


def test_photo_must_agree_with_a_predicted_metal():
    ok = imp.photo_agrees(np.array(["gold", "gold", "silver", "silver", "other"]),
                          np.array([12.0, 1.0, 1.0, 15.0, np.nan]))
    assert ok.tolist() == [True, False, True, False, True]


def test_knn_vote_and_grouped_cross_validation():
    X = np.array([[1, 0], [0.99, 0.1], [0, 1], [0.1, 0.99]], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.array(["a", "a", "b", "b"], dtype=object)
    pred, conf = imp.knn_vote(X[:1], X[1:], y[1:], k=1)
    assert pred[0] == "a" and conf[0] == 1.0
    cv = imp.cross_validate(X, y, np.array(["m1", "m2", "m3", "m4"]), folds=2)
    assert len(cv) == 4 and set(cv.columns) == {"truth", "pred", "conf"}


def test_inferred_values_fill_gaps_and_never_replace_the_csv():
    items = pd.DataFrame({KEY: ["A", "B"], "MATERIAL": ["Linen", None], "FINISHING": [None, None]})
    inferred = pd.DataFrame({KEY: ["A", "B"], "column": ["MATERIAL", "MATERIAL"], "value": ["Cotton", "Cotton"]})
    out = imp.apply_inferred(items, inferred)
    assert out["MATERIAL"].tolist() == ["Linen", "Cotton"] and "FINISH_FAMILY" in out


def test_the_vote_never_sees_the_attributes_it_guesses():
    """Two items that differ only in FINISHING / FINISH_FAMILY / MATERIAL get the same neighbour features."""
    import phase2a_features as f2a
    cols = {c for cols, _ in f2a.CATEGORICAL.values() for c in cols}
    items = pd.DataFrame({c: ["x", "x"] for c in cols} | {
        KEY: ["NOT_EMBEDDED_1", "NOT_EMBEDDED_2"], "COMPOSITION": ["Zinc", "Zinc"], "img_trusted": [False, False],
        "FINISHING": ["Golden", "Rodhium"], "MATERIAL": ["Cotton", "Linen"]})
    X = imp.neighbour_features(imp.add_finish_family(items), imp.WEIGHTS["FINISH_FAMILY"])
    np.testing.assert_allclose(X[0], X[1])
