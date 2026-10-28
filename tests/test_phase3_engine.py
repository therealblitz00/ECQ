"""Phase 3 similarity engine (src/phase3_engine.py) on hand-made blocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase3_engine as eng

KEY = eng.KEY


def unit(*rows) -> np.ndarray:
    m = np.array(rows, dtype=np.float32)
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return np.divide(m, n, out=np.zeros_like(m), where=n > 0)


@pytest.fixture
def toy() -> tuple[pd.DataFrame, dict]:
    items = pd.DataFrame({
        KEY: ["A_BK", "A_RD", "B_BK", "C_BK", "D_BK", "E_BK", "F_BK"],
        "PROD_REF": ["A", "A", "B", "C", "D", "E", "F"],
        "PROD_REF_EQUIV": ["A", "A", "B", "C", "D", "E", "A"],      # F is a re-coded A
        "CAT_DES_EN": ["Jewellery"] * 6 + ["Jewellery"],
        "GFA_DES_EN": ["Rings", "Rings", "Rings", "Rings", "Earrings", "Rings", "Rings"],
    })
    blocks = {
        "categorical": unit([1, 0], [1, 0], [1, 0], [0.8, 0.6], [0, 1], [1, 0], [1, 0]),
        "image": unit([1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 0], [1, 0]),  # E has no trusted photo
    }
    return items, blocks


W = {"categorical": 0.5, "image": 0.5}


def test_weights_are_rescaled_over_the_blocks_both_items_have(toy):
    items, blocks = toy
    has = eng.has_block(blocks)
    s = eng.score_matrix(blocks, has, W, np.array([0]), np.arange(len(items)))[0]
    assert s[2] == pytest.approx(0.5)            # B: same categories, different photo
    assert s[3] == pytest.approx((0.8 + 1) / 2)  # C: close categories, same photo
    assert s[5] == pytest.approx(1.0)            # E: no photo -> categorical only, not penalised


def test_candidates_exclude_the_item_its_model_and_re_coded_models(toy):
    items, blocks = toy
    nb = eng.neighbours(items, blocks, W, k=10)
    a = nb[nb["query"] == "A_BK"]
    assert set(a["candidate"]) == {"B_BK", "C_BK", "D_BK", "E_BK"}  # not itself, A_RD (sibling) or F (re-coded A)
    assert a["rank"].tolist() == list(range(1, len(a) + 1))
    assert a["score"].is_monotonic_decreasing
    with_siblings = eng.neighbours(items, blocks, W, k=10, include_siblings=True, queries=["A_BK"])
    assert {"A_RD", "F_BK"} <= set(with_siblings["candidate"]) and "A_BK" not in set(with_siblings["candidate"])


def test_family_scope_keeps_candidates_in_the_same_family(toy):
    items, blocks = toy
    nb = eng.neighbours(items, blocks, W, k=10, scope="family")
    fam = items.set_index(KEY)["GFA_DES_EN"]
    assert (fam.loc[nb["query"]].to_numpy() == fam.loc[nb["candidate"]].to_numpy()).all()
    assert "D_BK" not in set(nb["query"])  # alone in its family: no candidates


def test_contributions_add_up_and_a_missing_photo_never_counts(toy):
    items, blocks = toy
    nb = eng.neighbours(items, blocks, W, k=10)
    contrib = nb.filter(like="contrib_").sum(axis=1)
    np.testing.assert_allclose(contrib, nb["score"], atol=1e-5)
    e = nb[(nb["query"] == "E_BK") | (nb["candidate"] == "E_BK")]
    assert e["sim_image"].isna().all() and (e["contrib_image"] == 0).all()


def test_ties_go_to_the_lower_key(toy):
    items, blocks = toy
    blocks = {"categorical": np.tile(unit([1, 0]), (len(items), 1))}  # every pair scores 1
    nb = eng.neighbours(items, blocks, {"categorical": 1.0}, k=3, queries=["A_BK"])
    assert nb["candidate"].tolist() == ["B_BK", "C_BK", "D_BK"]


def test_aligned_puts_rows_in_order_and_zeroes_untrusted_photos():
    emb = unit([1, 0], [0, 1], [1, 1])
    out = eng.aligned(["Z", "X", "Y", "W"], emb, ["X", "Y", "Z"], np.array([True, True, False, True]))
    np.testing.assert_allclose(out, unit([1, 1], [1, 0], [0, 0], [0, 0]), atol=1e-6)


def test_the_committed_weights_cover_known_blocks():
    w = eng.load_weights()
    assert set(w) <= set(eng.TABULAR) | {"image", "palette"}
    assert all(v >= 0 for v in w.values()) and sum(w.values()) > 0


def test_diversity_filter_limits_colours_per_model():
    items = pd.DataFrame({
        KEY: ["Q_1", "M1_A", "M1_B", "M1_C", "M2_A"],
        "PROD_REF": ["Q", "M1", "M1", "M1", "M2"],
        "PROD_REF_EQUIV": ["Q", "M1", "M1", "M1", "M2"],
        "CAT_DES_EN": ["Jewellery"] * 5,
        "GFA_DES_EN": ["Rings"] * 5,
    })
    blocks = {"categorical": np.tile(unit([1, 0]), (5, 1))}
    nb = eng.neighbours(items, blocks, {"categorical": 1.0}, k=2, queries=["Q_1"], max_per_model=1)
    assert nb["candidate"].tolist() == ["M1_A", "M2_A"]
    nb_unlim = eng.neighbours(items, blocks, {"categorical": 1.0}, k=2, queries=["Q_1"], max_per_model=0)
    assert nb_unlim["candidate"].tolist() == ["M1_A", "M1_B"]
