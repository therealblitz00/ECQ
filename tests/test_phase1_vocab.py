"""Allowed values for review fixes (src/phase1_vocab.py)."""
from __future__ import annotations

import pandas as pd
import pytest

import phase1_vocab as pv

ITEMS = pd.DataFrame({  # "Padded Pu" is the majority spelling, as in the real data (62 vs 7)
    "CLR_DES": ["Black", "Black", "Gold", "Silver", "Green", "Silver"],
    "CAT_DES_EN": ["Jewellery", "Jewellery", "Jewellery", "Hand Bag", "Hand Bag", "Hand Bag"],
    "GFA_DES_EN": ["Earrings", "Rings", "Rings", "Tote", "Tote", "Tote"],
    "GFS_DES_EN": ["Hoop", "Plain", "Plain", "Padded Pu", "Padded PU", "Padded Pu"],
    "FINISHING": ["Golden", "Silver", "Golden", None, "Golden", None],
    "MATERIAL": [None] * 6,
    "COMPOSITION": ["Zinc", "Pearl; Zinc", "Brass", "Polyurethane", "polyurethane", "Polyurethane"],
})
VOCAB = pv.build_vocabulary(ITEMS)
EARRING = {"CAT_DES_EN": "Jewellery", "GFA_DES_EN": "Earrings", "GFS_DES_EN": "Hoop", "CLR_DES": "Black"}


def fix(text: str, item=EARRING):
    return pv.normalise_fixes(pv.parse_fixes(text), item, VOCAB)


@pytest.mark.parametrize("text, expected", [
    ("CLR_DES=Black", [("CLR_DES", "Black")]),
    ("CLR_DES=Black; COMPOSITION=Pearl; Zinc", [("CLR_DES", "Black"), ("COMPOSITION", "Pearl; Zinc")]),
    (" GFA_DES_EN = Rings ;FINISHING=Golden", [("GFA_DES_EN", "Rings"), ("FINISHING", "Golden")]),
    ("", []),
])
def test_parse_fixes(text, expected):
    assert pv.parse_fixes(text) == expected


def test_vocabulary_uses_one_spelling_per_value():
    assert VOCAB["CLR_DES"] == ["Black", "Gold", "Green", "Silver"]
    assert ["Hand Bag", "Tote", "Padded Pu"] in VOCAB["hierarchy"]
    assert ["Hand Bag", "Tote", "Padded PU"] not in VOCAB["hierarchy"]
    assert VOCAB["FINISHING"] == {"Hand Bag": ["Golden"], "Jewellery": ["Golden", "Silver"]}
    assert VOCAB["COMPOSITION"].count("Polyurethane") == 1 and "polyurethane" not in VOCAB["COMPOSITION"]


def test_existing_colour_is_accepted_in_its_canonical_spelling():
    assert fix("CLR_DES=  gold ") == ({"CLR_DES": "Gold"}, [])


@pytest.mark.parametrize("text, message", [
    ("CLR_DES=Mustard", "not an existing value"),
    ("PROD_DES_BASE=Ring Gold", "cannot be corrected"),
    ("CLR_DES=", "empty value"),
    ("COMPOSITION=Zinc; Unobtainium", "unknown material"),
])
def test_values_outside_the_vocabulary_are_rejected(text, message):
    fixes, errors = fix(text)
    assert any(message in e for e in errors), errors


def test_family_must_belong_to_the_category_and_sub_family_to_the_family():
    assert fix("GFA_DES_EN=Rings; GFS_DES_EN=Plain") == ({"GFA_DES_EN": "Rings", "GFS_DES_EN": "Plain"}, [])
    _, errors = fix("GFA_DES_EN=Rings")  # "Hoop" is not a sub-family of Rings
    assert errors and "combination" in errors[0]
    _, errors = fix("GFA_DES_EN=Tote; GFS_DES_EN=Padded Pu")  # a bag family for a jewellery item
    assert errors and "combination" in errors[0]
    ok, errors = fix("CAT_DES_EN=hand bag; GFA_DES_EN=Tote; GFS_DES_EN=PADDED PU")
    assert errors == [] and ok == {"CAT_DES_EN": "Hand Bag", "GFA_DES_EN": "Tote", "GFS_DES_EN": "Padded Pu"}


def test_finishing_must_be_used_in_the_items_category():
    assert fix("FINISHING=silver") == ({"FINISHING": "Silver"}, [])
    _, errors = fix("FINISHING=Silver; CAT_DES_EN=Hand Bag; GFA_DES_EN=Tote; GFS_DES_EN=Padded Pu")
    assert errors == ["FINISHING: 'Silver' is not used in category Hand Bag"]


def test_composition_is_stored_as_a_sorted_list_of_known_materials():
    assert fix("COMPOSITION=zinc; Pearl;Zinc") == ({"COMPOSITION": "Pearl; Zinc"}, [])


def test_vocabulary_table_lists_every_allowed_value():
    table = pv.vocabulary_table(VOCAB)
    assert len(table[table["column"] == "CLR_DES"]) == 4
    assert {"category", "family"} <= set(table.columns)
