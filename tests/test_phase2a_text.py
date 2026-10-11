"""Phase 2a text descriptions (src/phase2a_text.py). The embedding itself needs torch and is not tested here."""
from __future__ import annotations

import pandas as pd

import phase2a_text as txt


def test_description_uses_the_attributes_and_skips_empty_ones():
    items = pd.DataFrame({
        "GFS_DES_EN": ["Hoop Earrings", "Plain PU"], "GFA_DES_EN": ["Earrings", "Wallet"],
        "CAT_DES_EN": ["Jewellery", "Wallet"], "CLR_DES": ["Gold", "Black"], "FINISHING": ["Golden", None],
        "COMPOSITION": ["Zinc; Pearl", "Polyurethane"], "PRINT_TYPE": ["No Print", "Animal - Leopard"],
        "THEME": ["GOLDEN DELICATES", "Basics"], "FASHIONTYPE": ["Fashion", "Basic"], "PROD_SEG": ["Woman", "Woman"],
    })
    a, b = txt.describe(items)
    assert a == ("Hoop Earrings (Earrings, Jewellery). Colour: Gold. Finish: Golden. Made of Zinc, Pearl. "
                 "Theme: Golden Delicates. Fashion, Woman.")
    assert "Finish" not in b and "Print: Animal - Leopard." in b and "Theme: Basics." in b
