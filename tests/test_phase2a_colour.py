"""Phase 2a colour tones (src/phase2a_colour.py) on hand-made palettes."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import phase2a_colour as pc

KEY = pc.KEY
NAVY, MIDNIGHT, RED, WHITE = [22.0, 2.0, -9.0], [29.0, 3.0, -8.0], [45.0, 47.0, 26.0], [91.0, 0.0, 3.0]


@pytest.fixture
def items() -> pd.DataFrame:
    names = ["Navy"] * 5 + ["Midnight Blue"] * 5 + ["Red"] * 5 + ["Gold"] * 5 + ["Bright Multicolor"] * 5 + ["Lime"] * 2
    return pd.DataFrame({KEY: [f"{i}_X" for i in range(len(names))], "CLR_DES": names,
                         "img_colour_trusted": [True] * len(names)})


@pytest.fixture
def palettes(items) -> pd.DataFrame:
    centre = {"Navy": NAVY, "Midnight Blue": MIDNIGHT, "Red": RED, "Gold": [89.0, 1.0, 10.0],
              "Bright Multicolor": [84.0, 1.0, 5.0], "Lime": [82.0, -15.0, 48.0]}
    rows = []
    for k, name in zip(items[KEY], items["CLR_DES"]):
        # the product colour is the largest share; a small white part (background or lining) must be ignored
        rows.append({KEY: k, "reliable": True, "lab": [np.array(centre[name]), np.array(WHITE)],
                     "shares": np.array([0.7, 0.3])})
    return pd.DataFrame(rows)


def test_the_reference_is_the_dominant_colour_and_some_names_have_none(items, palettes):
    ref = pc.reference_colours(items, palettes).set_index("CLR_DES")
    assert ref.loc["Navy", ["L", "a", "b"]].tolist() == NAVY
    assert ref.loc["Navy", "photos"] == 5 and ref.loc["Navy", "hex"].startswith("#")
    assert ref.loc["Gold", "no_reference"] == "metallic"
    assert ref.loc["Bright Multicolor", "no_reference"] == "multicolour"
    assert ref.loc["Lime", "no_reference"] == f"under {pc.MIN_PHOTOS} photos"
    assert ref.loc[ref["no_reference"].notna(), ["L", "a", "b"]].isna().all().all()


def test_untrusted_photos_are_not_measured(items, palettes):
    items = items.assign(img_colour_trusted=items["CLR_DES"] != "Navy")
    assert pc.reference_colours(items, palettes).set_index("CLR_DES").at["Navy", "photos"] == 0


def test_close_shades_score_high_and_names_without_a_colour_only_match_themselves(items, palettes):
    ref = pc.reference_colours(items, palettes)
    sim = pd.DataFrame(pc.similarity(ref), index=ref["CLR_DES"], columns=ref["CLR_DES"])
    assert np.allclose(np.diag(sim), 1)
    assert sim.at["Navy", "Midnight Blue"] > 0.8 and sim.at["Navy", "Red"] < 0.01
    assert sim.at["Gold", "Bright Multicolor"] == 0 and sim.at["Gold", "Navy"] == 0


def test_item_rows_reproduce_the_similarity_exactly(items, palettes):
    ref = pc.reference_colours(items, palettes)
    names = pd.Series(["Navy", "Midnight Blue", "Gold", None, "Navy"])
    rows, cols = pc.tone_rows(names, ref)
    assert rows.shape[1] == len(cols)
    sim = pd.DataFrame(pc.similarity(ref), index=ref["CLR_DES"], columns=ref["CLR_DES"])
    assert rows[0] @ rows[4] == pytest.approx(1, abs=1e-5)
    assert rows[0] @ rows[1] == pytest.approx(sim.at["Navy", "Midnight Blue"], abs=1e-5)
    assert rows[0] @ rows[2] == pytest.approx(0, abs=1e-5)
    assert not rows[3].any()  # missing colour name: never a match
    assert np.array_equal(rows, pc.tone_rows(names, ref)[0])  # deterministic


def test_without_palettes_every_name_is_listed_without_a_colour(items):
    ref = pc.reference_colours(items, None)
    assert len(ref) == items["CLR_DES"].nunique() and ref["hex"].eq("").all()


def test_lab_to_hex_round_trips_known_colours():
    assert pc.lab_to_hex(np.array([100.0, 0.0, 0.0])) == "#ffffff"
    assert pc.lab_to_hex(np.array([0.0, 0.0, 0.0])) == "#000000"
    assert pc.lab_to_hex(np.array([53.24, 80.09, 67.2])) == "#ff0000"


def test_write_outputs(items, palettes, tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "REFERENCE_FILE", tmp_path / "colour_reference.parquet")
    monkeypatch.setattr(pc, "REPORT_DIR", tmp_path / "out")
    pc.write_outputs(pc.reference_colours(items, palettes))
    human = pd.read_csv(tmp_path / "out" / "colour_reference.csv", encoding="utf-8-sig").set_index("CLR_DES")
    assert human.at["Navy", "closest_names"].startswith("Midnight Blue")
    assert (tmp_path / "out" / "colour_reference.jpg").exists()
    assert (tmp_path / "colour_reference.json").exists()
