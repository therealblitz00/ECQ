"""Phase 2a: tabular and text features, one matrix per block, all in the same row order.

    python src/phase2a_features.py        # -> data/processed/features_<block>.npy + feature_meta.json

Input: `phase1_clean.load_items()` (the reviewed table once merged, the preview before that).
Rerun after the review merge: the fixes change some attributes.

Every block follows the same contract, so Phase 3 can add them up and Phase 5 can explain them:
rows are L2-normalised (the dot product of two rows is their cosine similarity) and the block is
split into named *attributes* (column slices); the dot product of one slice is that attribute's
share of the score. A missing value is a row of zeros: it never counts as a match.

Blocks:
- categorical: weighted one-hot of the product attributes (hierarchy, colour, finish, theme…).
  Each value is stored as sqrt(weight), so two items score the sum of the weights they share.
- composition: multi-hot of the materials in COMPOSITION, weighted by rarity (IDF), so sharing
  "Pearl" counts more than sharing "Polyester". One attribute per material.
- numeric: price position within the category, launch date and season. Price and date are
  encoded as soft bands (Gaussian bumps), so close values score close to 1 and far values 0.
- text: sentence embedding of a description built from the attributes, computed once by
  phase2a_text.py (vision machine) and loaded from data/embeddings/text.npy. One attribute.

Before the blocks are built, phase2a_impute fills MATERIAL (apparel) and FINISH_FAMILY (jewellery) where the
CSV is empty and the inference is reliable (see outputs/phase2a/impute_report.md). CSV values are never replaced.

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
import phase2a_impute  # noqa: E402
from config import EMB_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402

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
    "FINISH_FAMILY": (["FINISH_FAMILY"], 1.0),  # gold / silver / …: partial credit when the detailed finishing differs
    "MATERIAL": (["MATERIAL"], 1.0),
    "PRINT_TYPE": (["PRINT_TYPE"], 1.0),
    "DIMENSION": (["DIMENSION"], 1.0),
    "OUTFIT": (["OUTFIT"], 1.0),
    "THEME": (["THEME"], 1.5),
    "FASHIONTYPE": (["FASHIONTYPE"], 0.5),
    "PROD_SEG": (["PROD_SEG"], 0.5),
    "NUMBER_OF_UNITS": (["NUMBER_OF_UNITS"], 0.5),
}

# numeric part -> weight. Price is compared within the category (a 30 € bag is cheap, a 30 € ring is not).
NUMERIC = {"price": 2.0, "launch": 1.0, "season": 0.5}
PRICE_SIGMA = 0.10    # in price percentile within the category: 10 points apart -> similarity ≈ 0.78
LAUNCH_SIGMA = 1.5    # in months: 1 month apart -> ≈ 0.89, 3 months -> ≈ 0.37


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0).astype(np.float32, copy=False)


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
    return _normalise_rows(np.hstack(blocks)), columns, attrs


def material_tokens(composition: pd.Series) -> pd.Series:
    """Set of materials per item ("Brass; Glass; Zinc" -> {"Brass", "Glass", "Zinc"})."""
    return composition.fillna("").map(lambda s: sorted({t.strip() for t in str(s).split(";") if t.strip()}))


def composition_block(items: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    """IDF-weighted multi-hot of materials. One attribute (one column) per material."""
    tokens = material_tokens(items["COMPOSITION"])
    vocab = sorted({t for ts in tokens for t in ts})
    col = {t: j for j, t in enumerate(vocab)}
    df = pd.Series([t for ts in tokens for t in ts]).value_counts()
    idf = {t: float(np.log((1 + len(items)) / (1 + df[t])) + 1) for t in vocab}
    matrix = np.zeros((len(items), len(vocab)), dtype=np.float32)
    for i, ts in enumerate(tokens):
        for t in ts:
            matrix[i, col[t]] = idf[t]
    attrs = {t: {"source": ["COMPOSITION"], "weight": round(idf[t], 3), "start": j, "stop": j + 1,
                 "n_values": 1, "filled": int(df[t])} for t, j in col.items()}
    return _normalise_rows(matrix), [f"COMPOSITION={t}" for t in vocab], attrs


def soft_bands(x: np.ndarray, centres: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian bumps at `centres`, unit length per row: dot(a, b) ≈ exp(-(xa - xb)² / 4σ²)."""
    m = np.exp(-((x[:, None] - centres[None, :]) ** 2) / (2 * sigma**2)).astype(np.float32)
    m[np.isnan(x)] = 0
    return _normalise_rows(m)


def numeric_values(items: pd.DataFrame) -> pd.DataFrame:
    """Price percentile within the category and launch month (months since the first launch)."""
    price_pct = items.groupby("CAT_DES_EN")["PRICE_BASE_W_VAT"].rank(pct=True, method="average")
    date = pd.to_datetime(items["STORE_DATE_FINAL"])
    months = (date - date.min()).dt.days / 30.44
    return pd.DataFrame({"price_pct": price_pct.astype(float), "launch_month": months.astype(float)})


def numeric_block(items: pd.DataFrame, weights: dict = NUMERIC) -> tuple[np.ndarray, list[str], dict]:
    v = numeric_values(items)
    # centres run 3σ past both ends, so values at the edges are compared like the others
    price_c = np.arange(-3 * PRICE_SIGMA, 1 + 3 * PRICE_SIGMA + 1e-9, PRICE_SIGMA / 2)
    lo, hi = np.nanmin(v.launch_month), np.nanmax(v.launch_month)
    launch_c = np.arange(np.floor(lo - 3 * LAUNCH_SIGMA), np.ceil(hi + 3 * LAUNCH_SIGMA) + 1e-9, LAUNCH_SIGMA / 2)
    seasons = sorted(items["SEA_COD"].dropna().astype(str).unique())
    season = np.zeros((len(items), len(seasons)), dtype=np.float32)
    codes = pd.Categorical(items["SEA_COD"].astype("string"), categories=seasons).codes
    season[np.flatnonzero(codes >= 0), codes[codes >= 0]] = 1
    parts = {
        "price": (soft_bands(v.price_pct.to_numpy(), price_c, PRICE_SIGMA), [f"price_pct≈{c:.3f}" for c in price_c],
                  ["PRICE_BASE_W_VAT", "CAT_DES_EN"]),
        "launch": (soft_bands(v.launch_month.to_numpy(), launch_c, LAUNCH_SIGMA),
                   [f"launch_month≈{c:g}" for c in launch_c], ["STORE_DATE_FINAL"]),
        "season": (season, [f"SEA_COD={s}" for s in seasons], ["SEA_COD"]),
    }
    blocks, columns, attrs, start = [], [], {}, 0
    for name, (m, cols, source) in parts.items():
        blocks.append(m * np.float32(np.sqrt(weights[name])))
        columns += cols
        attrs[name] = {"source": source, "weight": weights[name], "start": start, "stop": start + m.shape[1],
                       "n_values": m.shape[1], "filled": int((m.sum(axis=1) > 0).sum())}
        start += m.shape[1]
    return _normalise_rows(np.hstack(blocks)), columns, attrs


def text_block(items: pd.DataFrame) -> tuple[np.ndarray, list[str], dict]:
    """The committed text embeddings, put in this table's row order (items not embedded: zeros)."""
    meta = json.loads((EMB_DIR / "text.json").read_text(encoding="utf-8"))
    emb = np.load(EMB_DIR / "text.npy").astype(np.float32)
    pos = pd.Series(range(len(meta["keys"])), index=meta["keys"])
    idx = items[KEY].map(pos)
    matrix = np.zeros((len(items), emb.shape[1]), dtype=np.float32)
    found = idx.notna().to_numpy()
    matrix[found] = emb[idx[found].astype(int).to_numpy()]
    attrs = {"text": {"source": ["description from the attributes"], "weight": 1.0, "start": 0,
                      "stop": emb.shape[1], "n_values": emb.shape[1], "filled": int(found.sum())}}
    return _normalise_rows(matrix), [f"text_{d}" for d in range(emb.shape[1])], attrs


BLOCKS = {"categorical": categorical_block, "composition": composition_block, "numeric": numeric_block,
          "text": text_block}
# Blocks computed once on the vision machine: skipped (with a message) when their file is missing.
NEEDS_FILE = {"text": EMB_DIR / "text.npy"}


def shared_attributes(matrix: np.ndarray, attrs: dict, i: int, j: int) -> dict[str, float]:
    """Contribution of each attribute to the cosine similarity of rows i and j (sums to the score)."""
    out = {}
    for name, a in attrs.items():
        c = float(matrix[i, a["start"]:a["stop"]] @ matrix[j, a["start"]:a["stop"]])
        if c > 1e-6:
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


def example_neighbours(items: pd.DataFrame, matrix: np.ndarray, attrs: dict, queries: list[str], k: int = 4,
                       top_attrs: int = 6) -> str:
    """Markdown with the top-k neighbours of a few items in one block (a sanity check, not the engine)."""
    keys = items[KEY].tolist()
    lines = []
    for q in queries:
        if q not in keys:
            continue
        i = keys.index(q)
        scores = matrix @ matrix[i]
        # same category only, as Phase 3 will do; other colours of the same model are trivially similar
        scores[(items["CAT_DES_EN"] != items["CAT_DES_EN"].iloc[i]).to_numpy()] = -1
        scores[(items["PROD_REF"] == items["PROD_REF"].iloc[i]).to_numpy()] = -1
        top = np.argsort(-scores, kind="stable")[:k]
        r = items.iloc[i]
        lines += [f"**`{q}`**: {r.GFS_DES_EN}, {r.CLR_DES}, {r.PRICE_BASE_W_VAT:.2f} €, {r.COMPOSITION}", "",
                  "| Neighbour | Score | Sub-family | Colour | Price | Composition | Main shared attributes |",
                  "|---|---|---|---|---|---|---|"]
        for j in top:
            c = items.iloc[j]
            shared = sorted(shared_attributes(matrix, attrs, i, j).items(), key=lambda kv: -kv[1])[:top_attrs]
            text = ", ".join(f"{n} {v:.2f}" for n, v in shared)
            lines.append(f"| `{c[KEY]}` | {scores[j]:.2f} | {c.GFS_DES_EN} | {c.CLR_DES} | {c.PRICE_BASE_W_VAT:.2f} | "
                         f"{c.COMPOSITION} | {text} |")
        lines.append("")
    return "\n".join(lines)


def write_report(items: pd.DataFrame, built: dict, source: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    n = len(items)
    rng = np.random.default_rng(0)
    queries = items[KEY].iloc[np.sort(rng.choice(n, size=min(3, n), replace=False))].tolist()
    cat_m, _, cat_a = built["categorical"]
    comp_m, _, comp_a = built["composition"]
    num_m, _, num_a = built["numeric"]
    cat_rows = "\n".join(f"| `{k}` | {' › '.join(a['source'])} | {a['weight']} | {a['n_values']} | "
                         f"{a['filled'] / n:.1%} |" for k, a in cat_a.items())
    comp = sorted(comp_a.items(), key=lambda kv: -kv[1]["filled"])
    comp_rows = "\n".join(f"| {k} | {a['filled']:,} | {a['weight']} |" for k, a in comp[:8] + comp[-4:])
    num_rows = "\n".join(f"| `{k}` | {', '.join(a['source'])} | {a['weight']} | {a['n_values']} |" for k, a in num_a.items())
    text_section = ""
    if "text" in built:
        txt_m, _, txt_a = built["text"]
        text_section = (f"\n## Text block\n\nSentence embedding (`data/embeddings/text.npy`, model in `text.json`) of a "
                        f"description built from the attributes; {txt_a['text']['filled']:,} of {n:,} items embedded.\n\n"
                        f"### Text\n\n{example_neighbours(items, txt_m, txt_a, queries)}")
    shapes = "\n".join(f"| `features_{b}.npy` | {m.shape[0]:,} × {m.shape[1]:,} |" for b, (m, _, _) in built.items())
    text = f"""# Phase 2a: feature report

Generated by `src/phase2a_features.py` from `{source}.parquet` ({n:,} colourways).
Every block: float32, rows L2-normalised (dot product = cosine similarity), split into named attributes whose
slices add up to the score. Row order and attribute slices: `data/processed/feature_meta.json`.

| File | Shape |
|---|---|
{shapes}

## Categorical block

| Attribute | Source columns | Weight | Values | Filled |
|---|---|---|---|---|
{cat_rows}

## Composition block

{len(comp_a)} materials from `COMPOSITION`, weighted by rarity (IDF = log((1 + N) / (1 + items with it)) + 1).
Most and least common:

| Material | Items | Weight |
|---|---|---|
{comp_rows}

## Numeric block

| Part | Source | Weight | Columns |
|---|---|---|---|
{num_rows}

- `price`: percentile of the price within the category, as soft bands (σ = {PRICE_SIGMA}): 10 percentile points
  apart → similarity ≈ 0.78, 30 apart → ≈ 0.1.
- `launch`: months since the first launch date, as soft bands (σ = {LAUNCH_SIGMA} months): 1 month apart → ≈ 0.89,
  3 months → ≈ 0.37.
- `season`: one-hot of `SEA_COD`.

## Example neighbours, one block at a time

Three random items (seed 0); same category only, other colours of the same model excluded.

### Categorical

{example_neighbours(items, cat_m, cat_a, queries)}
### Composition

{example_neighbours(items, comp_m, comp_a, queries)}
### Numeric

{example_neighbours(items, num_m, num_a, queries)}{text_section}"""
    (REPORT_DIR / "features_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    items = phase1_clean.load_items().sort_values(KEY, kind="stable").reset_index(drop=True)
    source = items_source()
    inferred = phase2a_impute.run()
    items = phase2a_impute.apply_inferred(items, inferred)
    print(f"inferred attributes: {len(inferred):,} values filled (outputs/phase2a/impute_report.md)")
    built = {}
    for name, fn in BLOCKS.items():
        if name in NEEDS_FILE and not NEEDS_FILE[name].exists():
            print(f"{name}: skipped, {NEEDS_FILE[name]} not found (run src/phase2a_text.py on the vision machine)")
            continue
        matrix, columns, attrs = fn(items)
        save_block(name, matrix, columns, {"attributes": attrs}, items[KEY].tolist(), source)
        built[name] = (matrix, columns, attrs)
        print(f"{name}: {matrix.shape[0]:,} x {matrix.shape[1]:,} from {source}")
    write_report(items, built, source)


if __name__ == "__main__":
    main()
