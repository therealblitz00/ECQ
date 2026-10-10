"""Phase 1: the allowed values for review fixes, so every reviewer corrects data the same way.

A fix may only use values that already exist in the data (after unifying their casing):
- colour: one of the existing `CLR_DES` names;
- category › family › sub-family: an existing combination (a family must belong to the category,
  a sub-family to the family);
- finishing / material: a value already used in the item's category;
- composition: a list of existing materials.
The description is not editable: fix the structured field it contradicts instead. If the right
value is not in the list, the reviewer marks the item "Not sure" and writes it in the note.

The review app offers only these values, and `merge` rejects anything else (CSV edited by hand).
`check` also writes the full list to outputs/phase1/review_vocabulary.csv.
"""
from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_DIR  # noqa: E402
from phase1_clean import unify_casing  # noqa: E402

# Column -> label in the review app. Order = order of the form.
EDITABLE = {
    "CLR_DES": "Colour",
    "CAT_DES_EN": "Category",
    "GFA_DES_EN": "Family",
    "GFS_DES_EN": "Sub-family",
    "FINISHING": "Finishing",
    "MATERIAL": "Material",
    "COMPOSITION": "Composition",
}
HIERARCHY = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"]
BY_CATEGORY = ["FINISHING", "MATERIAL"]  # only meaningful inside some categories

_FIX_SEPARATOR = re.compile(r";\s*(?=[A-Za-z_][A-Za-z0-9_]*\s*=)")


def parse_fixes(text: str) -> list[tuple[str, str]]:
    """'CLR_DES=Black; COMPOSITION=Pearl; Zinc' -> [('CLR_DES', 'Black'), ('COMPOSITION', 'Pearl; Zinc')]

    A ';' only starts a new fix when it is followed by 'COLUMN=', so values may contain ';'.
    """
    out = []
    for part in _FIX_SEPARATOR.split(str(text)):
        if "=" in part:
            col, val = part.split("=", 1)
            out.append((col.strip(), val.strip()))
    return out


def format_fixes(fixes: Mapping[str, str]) -> str:
    return "; ".join(f"{c}={v}" for c, v in fixes.items())


def _materials(s: pd.Series) -> pd.Series:
    return s.dropna().str.split(";").explode().str.strip().loc[lambda x: x != ""]


def build_vocabulary(items: pd.DataFrame) -> dict:
    """Allowed values per editable column, taken from the colourway table (canonical casing)."""
    items = items.copy()
    for c in [c for c in EDITABLE if c in items.columns and c != "COMPOSITION"]:
        items[c], _ = unify_casing(items[c])
    vocab: dict = {}
    if "CLR_DES" in items.columns:
        vocab["CLR_DES"] = sorted(items["CLR_DES"].dropna().unique())
    if set(HIERARCHY) <= set(items.columns):
        tri = items[HIERARCHY].fillna("").drop_duplicates().sort_values(HIERARCHY)
        vocab["hierarchy"] = tri.values.tolist()
        vocab["CAT_DES_EN"] = sorted(tri["CAT_DES_EN"].unique())
    for col in BY_CATEGORY:
        if {col, "CAT_DES_EN"} <= set(items.columns):
            vocab[col] = {cat: sorted(g.dropna().unique()) for cat, g in items.groupby("CAT_DES_EN")[col]}
    if "COMPOSITION" in items.columns:
        mats, _ = unify_casing(_materials(items["COMPOSITION"]))
        vocab["COMPOSITION"] = sorted(mats.unique())
    return vocab


def load_vocabulary() -> dict:
    path = PROCESSED_DIR / "items_checked.parquet"
    if not path.exists():
        raise SystemExit("Run the automatic checks first: python src/phase1_checks.py check")
    return build_vocabulary(pd.read_parquet(path, columns=list(EDITABLE)))


def _canonical(value: str, allowed: list[str]) -> str | None:
    """The allowed spelling of `value`, ignoring case and extra spaces; None if not allowed."""
    key = " ".join(value.split()).casefold()
    return next((a for a in allowed if a.casefold() == key), None)


def normalise_fixes(fixes: list[tuple[str, str]], item: Mapping, vocab: dict) -> tuple[dict[str, str], list[str]]:
    """Check one item's fixes against the vocabulary. Returns (canonical fixes, error messages)."""
    out: dict[str, str] = {}
    errors: list[str] = []
    for col, value in fixes:
        if col not in EDITABLE:
            errors.append(f"'{col}' cannot be corrected in the review (allowed: {', '.join(EDITABLE)})")
        elif not value:
            errors.append(f"{col}: empty value")
        elif col == "COMPOSITION":
            mats = [m for m in (v.strip() for v in value.split(";")) if m]
            canon = [_canonical(m, vocab.get("COMPOSITION", [])) for m in mats]
            unknown = [m for m, c in zip(mats, canon) if c is None]
            if unknown:
                errors.append(f"COMPOSITION: unknown material(s) {unknown}")
            else:
                out[col] = "; ".join(sorted(set(canon)))
        elif col in BY_CATEGORY:
            out[col] = value  # checked below, once the category is known
        elif col in HIERARCHY:
            out[col] = value  # checked below as a combination
        else:
            canon = _canonical(value, vocab.get(col, []))
            if canon is None:
                errors.append(f"{col}: '{value}' is not an existing value")
            else:
                out[col] = canon

    def current(col: str) -> str:
        v = item.get(col)
        return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)

    if any(c in out for c in HIERARCHY):
        if "hierarchy" not in vocab:
            errors.append("category › family › sub-family cannot be corrected here")
        else:
            tri = vocab["hierarchy"]
            cat = _canonical(out.get("CAT_DES_EN", current("CAT_DES_EN")), sorted({t[0] for t in tri}))
            fam = _canonical(out.get("GFA_DES_EN", current("GFA_DES_EN")),
                             sorted({t[1] for t in tri if t[0] == cat}))
            sub = _canonical(out.get("GFS_DES_EN", current("GFS_DES_EN")),
                             sorted({t[2] for t in tri if t[0] == cat and t[1] == fam}))
            if cat is None or fam is None or sub is None:
                combo = " › ".join(out.get(c, current(c)) for c in HIERARCHY)
                errors.append(f"'{combo}' is not an existing category › family › sub-family combination")
            else:
                for c, v in zip(HIERARCHY, [cat, fam, sub]):
                    if c in out:
                        out[c] = v

    category = out.get("CAT_DES_EN", current("CAT_DES_EN"))
    for col in [c for c in BY_CATEGORY if c in out]:
        canon = _canonical(out[col], vocab.get(col, {}).get(category, []))
        if canon is None:
            errors.append(f"{col}: '{out[col]}' is not used in category {category}")
            del out[col]
        else:
            out[col] = canon
    return out, errors


def vocabulary_table(vocab: dict) -> pd.DataFrame:
    """Flat list for people editing a batch CSV by hand (outputs/phase1/review_vocabulary.csv)."""
    rows = [("CLR_DES", v, "", "") for v in vocab.get("CLR_DES", [])]
    rows += [("CAT_DES_EN › GFA_DES_EN › GFS_DES_EN", sub, cat, fam) for cat, fam, sub in vocab.get("hierarchy", [])]
    for col in BY_CATEGORY:
        rows += [(col, v, cat, "") for cat, vals in vocab.get(col, {}).items() for v in vals]
    rows += [("COMPOSITION (one material)", v, "", "") for v in vocab.get("COMPOSITION", [])]
    return pd.DataFrame(rows, columns=["column", "value", "category", "family"])
