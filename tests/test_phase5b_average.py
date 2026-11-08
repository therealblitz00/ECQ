"""Phase 5b average: the family names and the three-reasons text (no model is trained)."""
from __future__ import annotations

import pandas as pd
import phase5b_average as avg


def test_weeks_on_sale_has_its_own_family():
    assert avg.family("weeks_on_sale") == "weeks on sale"
    assert avg.family("img_pc3") == "photo (image components)"


def test_reasons_are_three_at_most_and_biggest_first():
    fam = pd.DataFrame({"a": [0.1, 0.5], "b": [-0.4, 0.05], "c": [0.2, -0.3], "d": [0.0, 0.02]})
    out = avg.reasons(fam)
    for text in out:
        sizes = [float(p.rsplit(" ", 1)[1].replace("−", "").replace("+", "")) for p in text.split("; ")]
        assert len(sizes) <= 3 and sizes == sorted(sizes, reverse=True)
    assert out[0].startswith("b −0.40")
