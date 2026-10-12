"""Phase 1b flag precision (src/phase1b_precision.py) on a hand-made review."""
from __future__ import annotations

import pandas as pd
import pytest

import phase1b_precision as fp

KEY = fp.KEY


@pytest.fixture
def review() -> tuple[pd.DataFrame, pd.DataFrame]:
    items = pd.DataFrame({
        KEY: ["A", "B", "C", "D", "E"],
        "review_status": ["fix", "ok", "drop_image", "fix", "ok"],
        "fixes": ["CLR_DES=Teal", "", "", "GFA_DES_EN=Rings", ""],
    })
    audit = pd.DataFrame({
        KEY: ["A", "B", "C", "D", "E"],
        "vis_issues": ["VIS_COLOUR_MISMATCH", "VIS_COLOUR_MISMATCH", "VIS_TYPE_MISMATCH",
                       "VIS_TYPE_MISMATCH;VIS_COLOUR_MISMATCH", ""],
    })
    return items, audit


def test_outcomes_read_the_fixed_columns(review):
    items, _ = review
    o = fp.outcomes(items, {"A", "B"}).set_index(KEY)
    assert o.loc["A", "colour_fixed"] and not o.loc["A", "type_fixed"]
    assert o.loc["D", "type_fixed"] and not o.loc["D", "colour_fixed"]
    assert o.loc["C", "dropped"]
    assert o["full_review"].tolist() == [True, True, False, False, False]


def test_a_flag_counts_only_when_the_reviewer_acted_on_it(review):
    items, audit = review
    t = fp.precision_table(audit, fp.outcomes(items, set(items[KEY]))).set_index("flag")
    # colour: A fixed the colour (yes), B ok (no), D fixed the type, not the colour (no)
    assert (t.loc["VIS_COLOUR_MISMATCH", "flagged"], t.loc["VIS_COLOUR_MISMATCH", "confirmed"]) == (3, 1)
    # type: C dropped (yes), D fixed the family (yes)
    assert t.loc["VIS_TYPE_MISMATCH", "precision"] == 1.0
    assert t.loc["IMG_NEAR_DUPLICATE", "flagged"] == 0


def test_recall_uses_only_fully_reviewed_items(review):
    items, audit = review
    audit.loc[audit[KEY] == "A", "vis_issues"] = ""  # the colour problem of A was found by eye
    t = fp.precision_table(audit, fp.outcomes(items, {"A", "B", "E"}))
    assert t.attrs["problems"] == 1 and t.attrs["any_recall"] == 0.0
