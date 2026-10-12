"""Phase 4 evaluation (src/phase4_eval.py) on a hand-made catalogue."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase4_eval as ev

KEY = ev.KEY


@pytest.fixture
def items() -> pd.DataFrame:
    return pd.DataFrame({
        KEY: ["A_BK", "A_RD", "B_BK", "C_BK", "D_GD", "E_BK"],
        "PROD_REF": ["A", "A", "B", "C", "D", "E"],
        "PROD_REF_EQUIV": ["A", "A", "B", "C", "D", "E"],
        "CAT_DES_EN": ["Jewellery"] * 6,
        "GFA_DES_EN": ["Rings"] * 4 + ["Earrings", "Rings"],
        "GFS_DES_EN": ["Cocktail"] * 4 + ["Hoop", "Cocktail"],
        "THEME": ["Nile", "Nile", "Nile", "Nile", "Nile", None],
        "CLR_DES": ["Black", "Red", "Black", "Black", "Gold", "Black"],
        "FINISHING": ["Golden", "Golden", "Golden", "Silver", None, "Golden"],
        "MATERIAL": [None] * 6,
        "sales_pct_in_cat": [0.9, 0.5, 0.8, 0.1, 0.2, 0.7],
        "PRICE_BASE_W_VAT": [10.0, 10.0, 10.0, 20.0, 5.0, 10.0],
        "img_trusted": [True] * 6,
    })


def test_the_split_keeps_every_colour_of_a_model_together_and_is_fixed(items):
    tune = ev.split_models(items)
    assert tune[0] == tune[1]
    assert (tune == ev.split_models(items)).all()
    assert 0 < tune.sum() < len(items)


def test_known_groups_need_another_model_and_every_attribute(items):
    g = ev.known_groups(items)
    assert g[0] == g[2] == g[3] >= 0      # A_BK, B_BK, C_BK: same path, theme and colour, different models
    assert g[1] == -1 and g[4] == -1      # alone in their group (A_RD: only colour red)
    assert g[5] == -1                     # theme missing


def test_metrics_on_a_known_ranking(items):
    nb = pd.DataFrame({"query": ["A_BK"] * 4, "rank": [1, 2, 3, 4],
                       "candidate": ["E_BK", "B_BK", "D_GD", "C_BK"]})
    pools = {0: {2, 5}, 2: set(), 3: set(), 4: set(), 5: set()}
    m = ev.metrics(items, nb, ev.known_groups(items), pools)
    assert m["hit@4"] == 1.0                             # B_BK is a known similar item
    assert m["same_family"] == 0.75                      # D_GD is earrings
    assert m["same_finishing"] == pytest.approx(2 / 3)   # D_GD has no finishing; C_BK is silver
    assert np.isnan(m["same_material"])                  # nobody has a material
    assert m["visual@4"] == 0.5                          # E_BK and B_BK are among A_BK's nearest photos


def test_a_change_that_hurts_one_metric_is_refused():
    old = {m: 0.5 for m in ev.GUARDED} | {"objective": 0.5}
    up = {m: 0.6 for m in ev.GUARDED} | {"objective": 0.6}
    assert ev.better(up, old)
    traded = dict(up, same_material=0.45)
    assert not ev.better(traded, old)


def test_with_image_keeps_the_proportions_of_the_other_blocks():
    w = ev.with_image({"image": 0.3, "categorical": 0.4, "text": 0.2, "numeric": 0.1}, 0.6)
    assert sum(w.values()) == pytest.approx(1)
    assert w["image"] == 0.6 and w["categorical"] / w["text"] == pytest.approx(2)


def test_the_sheet_is_blind_deduplicated_and_never_overwritten(items, tmp_path):
    pooled = pd.DataFrame({"query": ["C_BK", "C_BK", "C_BK", "A_BK"], "rank": [1, 2, 1, 1],
                           "candidate": ["E_BK", "B_BK", "E_BK", "B_BK"]})
    s = ev.make_sheet(items, pooled, ["C_BK", "A_BK"])
    assert s["query"].tolist() == ["C_BK", "C_BK", "A_BK"]          # sheet order of the queries
    assert s.loc[s["query"] == "C_BK", "candidate"].tolist() == ["B_BK", "E_BK"]  # key order, not rank
    assert s["label"].tolist() == ["A", "B", "A"] and (s["verdict"] == "").all()
    path = tmp_path / "human_review.csv"
    s.to_csv(path, index=False, encoding="utf-8-sig")
    assert not ev.has_verdicts(path)
    s.loc[0, "verdict"] = "Similar "
    s.to_csv(path, index=False, encoding="utf-8-sig")
    assert ev.has_verdicts(path)
    v = ev.read_verdicts(path)
    assert v["value"].tolist() == [1.0]


def test_a_sheet_saved_by_excel_in_portuguese_is_read(tmp_path):
    path = tmp_path / "human_review_1.csv"
    path.write_bytes("sheet;query;candidate;verdict;notes\n1;A_BK;B_BK;partly;cor diferente, preço\n".encode("cp1252"))
    s = ev.read_sheet(path)
    assert s.loc[0, "verdict"] == "partly" and s.loc[0, "notes"] == "cor diferente, preço"
