"""Phase 2a: tabular and text features, one matrix per block, all in the same row order.

    python src/phase2a_features.py        # -> data/processed/features_<block>.npy + feature_meta.json

Input: `phase1_clean.load_items()` (the reviewed table once merged, the preview before that).
Rerun after the review merge: the fixes change some attributes.

Blocks so far:
- categorical: weighted one-hot of the product attributes (hierarchy, colour, finish, theme…).
  Each attribute value is stored as sqrt(weight), so the dot product of two raw rows is the sum
  of the weights of the attributes they share. Rows are then L2-normalised, so the dot product
  of two stored rows is their cosine similarity, and the slice of one attribute gives its share
  of that score (used for the explanations in Phase 5). A missing value is a row of zeros: it
  never counts as a match.

Weights are starting values; Phase 4 tunes them.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
from config import KEY, PROCESSED_DIR, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase2a"
META_FILE = "feature_meta.json"
SEP = " › "

# attribute -> (source columns, weight). Several columns form one path, because GFA and GFS
# names repeat across categories (e.g. "Others") and only mean something below their parent.
CATEGORICAL = {
    "CAT": (["CAT_DES_EN"], 1.0),
    "GFA": (["CAT_DES_EN", "GFA_DES_EN"], 3.0),
    "GFS": (["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"], 3.0),
    "CATEGORY_MATRIX": (["CATEGORY_MATRIX"], 1.5),
    "CLR_DES": (["CLR_DES"], 2.0),
    "CLR_TYPE": (["CLR_TYPE"], 0.5),
    "FINISHING": (["FINISHING"], 1.5),
    "MATERIAL": (["MATERIAL"], 1.0),
    "PRINT_TYPE": (["PRINT_TYPE"], 1.0),
    "DIMENSION": (["DIMENSION"], 1.0),
    "OUTFIT": (["OUTFIT"], 1.0),
    "THEME": (["THEME"], 1.5),
    "FASHIONTYPE": (["FASHIONTYPE"], 0.5),
    "PROD_SEG": (["PROD_SEG"], 0.5),
    "NUMBER_OF_UNITS": (["NUMBER_OF_UNITS"], 0.5),
}


def attribute_values(items: pd.DataFrame, cols: list[str]) -> pd.Series:
    """One label per item for an attribute; missing if any part of the path is missing."""
    parts = items[cols].astype("string")
    values = parts.iloc[:, 0]
    for col in cols[1:]:
        values = values + SEP + parts[col]
    return values


def categorical_block(items: pd.DataFrame, spec: dict = CATEGORICAL) -> tuple[np.ndarray, list[str], dict]:
    """Weighted one-hot matrix (rows L2-normalised), its column names and per-attribute metadata."""
    blocks, columns, attrs, start = [], [], {}, 0
    for name, (cols, weight) in spec.items():
        values = attribute_values(items, cols)
        levels = sorted(values.dropna().unique())  # sorted: identical columns on every machine
        codes = pd.Categorical(values, categories=levels).codes
        onehot = np.zeros((len(items), len(levels)), dtype=np.float32)
        filled = codes >= 0
        onehot[np.flatnonzero(filled), codes[filled]] = np.sqrt(weight)
        blocks.append(onehot)
        columns += [f"{name}={v}" for v in levels]
        attrs[name] = {"source": cols, "weight": weight, "start": start, "stop": start + len(levels),
                       "n_values": len(levels), "filled": int(filled.sum())}
        start += len(levels)
    matrix = np.hstack(blocks)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    return matrix, columns, attrs


def shared_attributes(matrix: np.ndarray, attrs: dict, i: int, j: int) -> dict[str, float]:
    """Contribution of each attribute to the cosine similarity of rows i and j (sums to the score)."""
    out = {}
    for name, a in attrs.items():
        c = float(matrix[i, a["start"]:a["stop"]] @ matrix[j, a["start"]:a["stop"]])
        if c > 0:
            out[name] = c
    return out


def items_source() -> str:
    return "items_clean" if (phase1_clean.PROCESSED_DIR / phase1_clean.CLEAN_FILE).exists() else "items_preview"


def save_block(name: str, matrix: np.ndarray, columns: list[str], extra: dict, keys: list[str], source: str) -> None:
    """Write features_<name>.npy and update its entry in feature_meta.json (same row order for all blocks)."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / f"features_{name}.npy", matrix)
    path = PROCESSED_DIR / META_FILE
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if meta.get("keys") != keys:  # other rows: the stored blocks no longer line up, start again
        meta = {"keys": keys, "blocks": {}}
    meta.update(source=source, n=len(keys), updated=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    meta["blocks"][name] = {"file": f"features_{name}.npy", "shape": list(matrix.shape),
                            "dtype": str(matrix.dtype), "columns": columns, **extra}
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def example_neighbours(items: pd.DataFrame, matrix: np.ndarray, attrs: dict, queries: list[str], k: int = 4) -> str:
    """Markdown with the top-k categorical neighbours of a few items (a sanity check, not the engine)."""
    keys = items[KEY].tolist()
    lines = []
    for q in queries:
        if q not in keys:
            continue
        i = keys.index(q)
        scores = matrix @ matrix[i]
        same_ref = (items["PROD_REF"] == items["PROD_REF"].iloc[i]).to_numpy()
        scores[same_ref] = -1  # other colours of the same model are trivially similar
        top = np.argsort(-scores, kind="stable")[:k]
        r = items.iloc[i]
        lines += [f"**`{q}`**: {r.GFS_DES_EN}, {r.CLR_DES}, {r.THEME}", "",
                  "| Neighbour | Score | Sub-family | Colour | Theme | Shared attributes |", "|---|---|---|---|---|---|"]
        for j in top:
            c = items.iloc[j]
            shared = ", ".join(shared_attributes(matrix, attrs, i, j))
            lines.append(f"| `{c[KEY]}` | {scores[j]:.2f} | {c.GFS_DES_EN} | {c.CLR_DES} | {c.THEME} | {shared} |")
        lines.append("")
    return "\n".join(lines)


def write_report(items: pd.DataFrame, matrix: np.ndarray, attrs: dict, source: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(items)
    rows = "\n".join(f"| `{name}` | {' › '.join(a['source'])} | {a['weight']} | {a['n_values']} | "
                     f"{a['filled'] / n:.1%} |" for name, a in attrs.items())
    rng = np.random.default_rng(0)
    queries = items[KEY].iloc[np.sort(rng.choice(n, size=min(3, n), replace=False))].tolist()
    text = f"""# Phase 2a: feature report

Generated by `src/phase2a_features.py` from `{source}.parquet` ({n:,} colourways).

## Categorical block

`data/processed/features_categorical.npy`: {matrix.shape[0]:,} × {matrix.shape[1]:,} (float32, rows L2-normalised).
The dot product of two rows is their cosine similarity; each attribute's slice gives its share of it.

| Attribute | Source columns | Weight | Values | Filled |
|---|---|---|---|---|
{rows}

## Example neighbours (categorical block only)

Three random items (seed 0); other colours of the same model are excluded.

{example_neighbours(items, matrix, attrs, queries)}"""
    (REPORT_DIR / "features_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    items = phase1_clean.load_items().sort_values(KEY, kind="stable").reset_index(drop=True)
    source = items_source()
    matrix, columns, attrs = categorical_block(items)
    save_block("categorical", matrix, columns, {"attributes": attrs}, items[KEY].tolist(), source)
    write_report(items, matrix, attrs, source)
    print(f"categorical: {matrix.shape[0]:,} x {matrix.shape[1]:,} from {source} "
          f"-> {PROCESSED_DIR / 'features_categorical.npy'}")


if __name__ == "__main__":
    main()
