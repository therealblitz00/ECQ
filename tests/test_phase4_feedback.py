"""Phase 4 interactive feedback (src/phase4_feedback.py) on hand-made pools and catalogues."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import phase3_engine as engine
import phase4_feedback as fb

KEY = fb.KEY
BLOCKS = ["image", "categorical"]


def toy_pool(holdout: set[str] = frozenset()) -> dict:
    """Two queries; candidate x is close on the image, y on the category, z on neither."""
    def cand(key, image, categorical):
        return {"key": key, "random": False, "sim": {"image": image, "categorical": categorical}}
    queries = [{"query": q, "holdout": q in holdout, "members": [1],
                "candidates": [cand("x", 0.9, 0.2), cand("y", 0.2, 0.9), cand("z", 0.1, 0.1), cand("n", None, 0.5)]}
               for q in ("q1", "q2")]
    return {"blocks": BLOCKS, "weights": {"image": 0.5, "categorical": 0.5}, "queries": queries}


def test_preferences_follow_the_order_and_the_not_column():
    index, _, _ = fb.sim_matrix(toy_pool())
    win, lose = fb.preference_pairs([{"query": "q1", "similar": ["y", "x"], "not": ["z", "n"]}], index)
    pairs = {(int(a), int(b)) for a, b in zip(win, lose)}
    i = {k: index[("q1", k)] for k in "xyzn"}
    assert pairs == {(i["y"], i["x"]), (i["y"], i["z"]), (i["y"], i["n"]), (i["x"], i["z"]), (i["x"], i["n"])}


def test_scores_are_the_engine_score_with_missing_blocks_left_out():
    _, s, h = fb.sim_matrix(toy_pool())
    sc = fb.scores(np.array([0.75, 0.25]), s, h)
    assert sc[0] == pytest.approx(0.75 * 0.9 + 0.25 * 0.2)
    assert sc[3] == pytest.approx(0.5)  # no image similarity: the category alone, weight rescaled
    # same formula as the engine on real blocks
    blocks = {"image": np.array([[1, 0], [0.6, 0.8], [0, 0]], dtype=np.float32),  # item 2 has no photo
              "categorical": np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)}
    has = engine.has_block(blocks)
    w = {"image": 0.75, "categorical": 0.25}
    eng = engine.score_matrix(blocks, has, w, np.array([0]), np.array([1, 2]))[0]
    parts = engine.pair_breakdown(blocks, has, {"image": 1.0, "categorical": 1.0}, np.array([0, 0]), np.array([1, 2]))
    s2 = np.nan_to_num(np.stack([parts["sim_image"], parts["sim_categorical"]], axis=1))
    h2 = np.isfinite(np.stack([parts["sim_image"], parts["sim_categorical"]], axis=1)).astype(float)
    np.testing.assert_allclose(fb.scores(np.array([0.75, 0.25]), s2, h2), eng, atol=1e-6)


def test_the_gradient_matches_finite_differences():
    index, s, h = fb.sim_matrix(toy_pool())
    win, lose = fb.preference_pairs([{"query": "q1", "similar": ["x", "n"], "not": ["y", "z"]}], index)
    theta0 = np.log([0.5, 0.5])
    p = np.array([0.3, -0.2, np.log(15.0)])
    _, g = fb.loss_and_grad(p, s, h, win, lose, theta0)
    eps = 1e-6
    num = [(fb.loss_and_grad(p + eps * e, s, h, win, lose, theta0)[0]
            - fb.loss_and_grad(p - eps * e, s, h, win, lose, theta0)[0]) / (2 * eps) for e in np.eye(3)]
    np.testing.assert_allclose(g, num, rtol=1e-4, atol=1e-6)


def test_the_weights_move_towards_what_the_reviewers_prefer():
    pool = toy_pool()
    photo_lovers = [{"member": 1, "query": q, "similar": ["x"], "not": ["y", "z", "n"]} for q in ("q1", "q2")]
    res = fb.learn(pool, photo_lovers)
    assert res["learned"]["image"] > res["start"]["image"]
    assert sum(res["learned"].values()) == pytest.approx(1, abs=1e-3)
    assert fb.learn(pool, [])["learned"] == res["start"]  # no answers: the start weights


def test_held_out_queries_are_measured_but_never_fitted():
    pool = toy_pool(holdout={"q2"})
    recs = [{"member": 1, "query": q, "similar": ["x"], "not": ["y", "z", "n"]} for q in ("q1", "q2")]
    res = fb.learn(pool, recs)
    assert res["queries"] == 1 and res["holdout"]["learned"]["pairs"] == 3
    assert res["holdout"]["learned"]["pair_accuracy"] >= res["holdout"]["start"]["pair_accuracy"]
    assert fb.learn(pool, [])["holdout"]["start"]["pair_accuracy"] is None  # JSON-safe when empty


def test_the_latest_answer_of_a_member_counts(tmp_path):
    f = tmp_path / "feedback_1.jsonl"
    lines = [{"member": 1, "query": "q1", "similar": ["x"], "not": ["y"]},
             {"member": 1, "query": "q1", "similar": ["y"], "not": ["x"]},
             {"member": 1, "query": "q2", "similar": [], "not": ["x"]}]
    f.write_text("\n".join(json.dumps(r) for r in lines) + "\n", encoding="utf-8")
    recs = fb.read_feedback([f])
    assert [(r["query"], r["similar"]) for r in recs] == [("q1", ["y"]), ("q2", [])]


def test_an_answer_must_place_every_candidate_once():
    q = toy_pool()["queries"][0]
    fb.check_answer(q, ["x", "y"], ["z", "n"])
    for similar, nots in ((["x"], ["z", "n"]), (["x", "y", "x"], ["z", "n"]), (["x", "y", "w"], ["z", "n"])):
        with pytest.raises(ValueError):
            fb.check_answer(q, similar, nots)


def test_reviewer_agreement_counts_pairs_both_members_order():
    a = {"member": 1, "query": "q1", "similar": ["x", "y"], "not": ["z"]}
    b = {"member": 2, "query": "q1", "similar": ["y", "x"], "not": ["z"]}
    res = fb.agreement([a, b])
    assert res == {"pairs": 3, "agreement": pytest.approx(2 / 3)}  # x-y disagree; x-z and y-z agree


@pytest.fixture
def catalogue() -> tuple[pd.DataFrame, dict]:
    n = 12
    items = pd.DataFrame({KEY: [f"{i:03d}_BK" for i in range(n)], "PROD_REF": [f"{i // 2:03d}" for i in range(n)],
                          "CAT_DES_EN": ["Jewellery"] * 8 + ["Wallet"] * 4})
    items["PROD_REF_EQUIV"] = items["PROD_REF"]
    rng = np.random.default_rng(0)
    blocks = {}
    for b in ("image", "categorical"):
        m = rng.normal(size=(n, 4)).astype(np.float32)
        blocks[b] = m / np.linalg.norm(m, axis=1, keepdims=True)
    return items, blocks


def test_the_pool_is_fixed_like_for_like_and_without_siblings(catalogue, monkeypatch):
    items, blocks = catalogue
    monkeypatch.setattr(fb, "SHARED", 1)
    monkeypatch.setattr(fb, "POOL_TOP", 2)
    w = {"image": 0.5, "categorical": 0.5}
    pool = fb.build_pool(items, blocks, w, ["000_BK", "009_BK"], [1, 2])
    again = fb.build_pool(items, blocks, w, ["000_BK", "009_BK"], [1, 2])
    assert [q["candidates"] for q in pool["queries"]] == [q["candidates"] for q in again["queries"]]
    it = items.set_index(KEY)
    for q in pool["queries"]:
        keys = [c["key"] for c in q["candidates"]]
        assert len(keys) == len(set(keys)) and q["query"] not in keys
        assert (it.loc[keys, "CAT_DES_EN"] == it.at[q["query"], "CAT_DES_EN"]).all()
        assert (it.loc[keys, "PROD_REF"] != it.at[q["query"], "PROD_REF"]).all()
        assert sum(c["random"] for c in q["candidates"]) <= fb.RANDOM_PER_QUERY
    assert sorted(len(q["members"]) for q in pool["queries"]) == [1, 2]  # one shared query
