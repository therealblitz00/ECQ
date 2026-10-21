"""Phase 2a: fill missing MATERIAL (apparel) and the finishing family (jewellery), only where it is reliable.

    python src/phase2a_impute.py     # -> data/processed/attributes_inferred.parquet + outputs/phase2a/impute_report.md
                                     #    (phase2a_features.py also runs it before building the blocks)

Nothing in data/csv or items_clean.parquet is changed: inferred values live in their own file, with the method
and a confidence, and are used by the feature blocks only. Only items with a trusted photo are filled.

What is filled, and how (accuracy measured by 5-fold cross-validation grouped by model, see the report):
- MATERIAL, apparel only (the field only exists for apparel):
  1. sibling: another colourway of the same model has it (they agree 99.7% of the time);
  2. composition: the composition is a single material that is also a MATERIAL value (98%);
  3. neighbours: vote of the 15 most similar apparel items, kept when the vote share >= MIN_CONF.
- FINISH_FAMILY, jewellery only: gold / rose gold / silver / black / other, from FINISHING when it is filled,
  otherwise from the neighbours' vote, checked against the photo: "gold" only if the palette is yellowish
  (mean b* >= GOLD_MIN_B), "silver" only if it is not (mean b* <= SILVER_MAX_B). Gold is 79% of the
  jewellery, so without this check silver items are too often filled as gold. The detailed FINISHING (golden vs worn gold vs brushed golden…) is
  not filled: neighbours get it right too rarely. Hand bag finishing is not filled either (≈ 77%).

Neighbours use the categorical block without the target attributes, the composition block, the CLIP image
embedding and the colour palette. The text block is left out: its description contains the finishing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
from config import EMB_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase2a"
OUT_FILE = "attributes_inferred.parquet"
K = 15
MIN_CONF = 0.8
FOLDS = 5
SEED = 0
# Neighbour weights per target, chosen by cross-validation: material is mostly told by the composition,
# the metal mostly by the photo.
WEIGHTS = {"MATERIAL": {"categorical": 1.0, "composition": 3.0, "image": 1.0, "palette": 0.5},
           "FINISH_FAMILY": {"categorical": 1.0, "composition": 1.0, "image": 2.0, "palette": 2.0}}
MIN_PRECISION = 0.90  # a value is only filled if its cross-validated precision reaches this
MIN_SUPPORT = 10      # ... measured on at least this many kept predictions
GOLD_MIN_B = 3.0     # mean b* (yellowness) of the palette: gold items median 12, silver items median 1
SILVER_MAX_B = 6.0
TARGETS = {"MATERIAL": "Apparel", "FINISH_FAMILY": "Jewellery"}

FINISH_FAMILY = {
    "gold": ["Golden", "Worn Gold", "Brushed Golden", "Antik Golden", "Gold Plated", "Matte Golden", "Burnish Golden",
             "Light Golden", "Worn Yellow Gold", "Matt Brushed Gold", "Scratched Gold"],
    "rose gold": ["Rose Gold", "Brushed Rose Gold", "Worn Rose God"],
    "silver": ["Rodhium", "Silver", "Worn Rhodium", "Worn Silver", "Brushed Silver", "Antik Silver", "Brushed Rhodium",
               "Matte Silver", "Silver Plated"],
    "black": ["Black Nikel", "Black Plating"],
    "other": ["Other", "Multifinishing", "Cooper"],
}
FAMILY_OF = {v: fam for fam, vals in FINISH_FAMILY.items() for v in vals}


def add_finish_family(items: pd.DataFrame) -> pd.DataFrame:
    """FINISH_FAMILY from FINISHING (a fixed mapping, not an inference)."""
    return items.assign(FINISH_FAMILY=items["FINISHING"].map(FAMILY_OF))


def neighbour_features(items: pd.DataFrame, weights: dict) -> np.ndarray:
    """Unit rows: categorical (without the targets) + composition + CLIP image + palette, weighted."""
    import phase2a_features as f2a
    # every attribute being guessed is left out, or cross-validation would see the answer
    spec = {k: v for k, v in f2a.CATEGORICAL.items() if k not in ("FINISHING", "MATERIAL", "FINISH_FAMILY")}
    parts = {"categorical": f2a.categorical_block(items, spec)[0], "composition": f2a.composition_block(items)[0],
             "image": _aligned(items, "image_clip.npy", "image_clip.json"), "palette": _palettes(items)}
    trusted = items["img_trusted"].fillna(False).to_numpy() if "img_trusted" in items else np.ones(len(items), bool)
    parts["image"][~trusted] = 0  # a photo the review dropped says nothing about this item
    parts["palette"][~trusted] = 0
    return f2a._normalise_rows(np.hstack([m * np.float32(np.sqrt(weights[n])) for n, m in parts.items()]))


def _aligned(items: pd.DataFrame, npy: str, meta: str) -> np.ndarray:
    path = EMB_DIR / npy
    if not path.exists():
        return np.zeros((len(items), 1), np.float32)
    keys = json.loads((EMB_DIR / meta).read_text(encoding="utf-8"))["keys"]
    emb = np.load(path).astype(np.float32)
    idx = items[KEY].map(pd.Series(range(len(keys)), index=keys))
    out = np.zeros((len(items), emb.shape[1]), np.float32)
    ok = idx.notna().to_numpy()
    out[ok] = emb[idx[ok].astype(int).to_numpy()]
    return out


def _palettes(items: pd.DataFrame) -> np.ndarray:
    path = EMB_DIR / "palettes.parquet"
    if not path.exists():
        return np.zeros((len(items), 1), np.float32)
    pal = pd.read_parquet(path).set_index(KEY)["palette_vector"]
    vec = items[KEY].map(pal)
    width = len(next(v for v in pal))
    return np.stack([np.asarray(v, np.float32) if isinstance(v, (list, np.ndarray)) else np.zeros(width, np.float32)
                     for v in vec])


def palette_yellowness(items: pd.DataFrame) -> pd.Series:
    """Share-weighted mean b* of the photo's palette (NaN without a palette)."""
    path = EMB_DIR / "palettes.parquet"
    if not path.exists():
        return pd.Series(np.nan, index=items.index)
    pal = pd.read_parquet(path).set_index(KEY)
    b = pd.Series([sum(sh * c[2] for sh, c in zip(shares, labs)) for shares, labs in zip(pal["shares"], pal["lab"])],
                  index=pal.index)
    return items[KEY].map(b)


def photo_agrees(pred: np.ndarray, yellow: np.ndarray) -> np.ndarray:
    """False where the photo contradicts a predicted metal (gold that isn't yellowish, silver that is)."""
    pred, yellow = np.asarray(pred, dtype=object), np.asarray(yellow, dtype=float)
    bad = ((pred == "gold") & ~(yellow >= GOLD_MIN_B)) | ((pred == "silver") & (yellow > SILVER_MAX_B))
    return ~bad


def knn_vote(X_query: np.ndarray, X_train: np.ndarray, y_train: np.ndarray, k: int = K) -> tuple[np.ndarray, np.ndarray]:
    """Similarity-weighted vote of the k nearest training rows: predicted value and its vote share."""
    pred, conf = np.empty(len(X_query), dtype=object), np.zeros(len(X_query))
    for s in range(0, len(X_query), 512):
        sim = X_query[s:s + 512] @ X_train.T
        top = np.argsort(-sim, axis=1, kind="stable")[:, :k]
        for i, row in enumerate(top):
            w = np.clip(sim[i, row], 0, None)
            votes = pd.Series(w).groupby(y_train[row]).sum()
            if votes.sum() <= 0:
                continue
            best = votes.sort_index().idxmax()  # sorted: ties break the same way everywhere
            pred[s + i], conf[s + i] = best, votes[best] / votes.sum()
    return pred, conf


def cross_validate(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = FOLDS) -> pd.DataFrame:
    """Out-of-fold prediction for every labelled row; folds by model, so siblings never help each other."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(groups)
    fold = pd.Series(rng.integers(0, folds, len(uniq)), index=uniq).loc[groups].to_numpy()
    pred, conf = np.empty(len(y), dtype=object), np.zeros(len(y))
    for f in range(folds):
        te, tr = fold == f, fold != f
        pred[te], conf[te] = knn_vote(X[te], X[tr], y[tr])
    return pd.DataFrame({"truth": y, "pred": pred, "conf": conf})


def sibling_values(items: pd.DataFrame, col: str) -> pd.Series:
    """The value shared by all other filled colourways of the same model (NaN if none or if they disagree)."""
    filled = items[items[col].notna()]
    one = filled.groupby("PROD_REF")[col].agg(lambda s: s.iloc[0] if s.nunique() == 1 else np.nan)
    return items["PROD_REF"].map(one)


def single_material(items: pd.DataFrame, vocab: set[str]) -> pd.Series:
    toks = items["COMPOSITION"].fillna("").map(lambda s: [t.strip() for t in s.split(";") if t.strip()])
    return toks.map(lambda t: t[0] if len(t) == 1 and t[0] in vocab else np.nan)


def infer(items: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Inferred values for the items with a trusted photo, and the cross-validation evidence."""
    items = add_finish_family(items.reset_index(drop=True))
    photo = items["img_trusted"].fillna(False)
    yellow = palette_yellowness(items).to_numpy()
    rows, evidence = [], {}
    for col, cat in TARGETS.items():
        X = neighbour_features(items, WEIGHTS[col])
        in_cat = items["CAT_DES_EN"].eq(cat)
        labelled = in_cat & items[col].notna()
        y, g = items.loc[labelled, col].to_numpy(), items.loc[labelled, "PROD_REF"].to_numpy()
        cv = cross_validate(X[labelled.to_numpy()], y, g)
        cv["photo_ok"] = photo_agrees(cv["pred"], yellow[labelled.to_numpy()]) if col == "FINISH_FAMILY" else True
        kept = (cv["conf"] >= MIN_CONF) & cv["photo_ok"]
        prec = (cv["pred"] == cv["truth"])[kept].groupby(cv["pred"][kept]).agg(["mean", "size"])
        allowed = set(prec.index[(prec["mean"] >= MIN_PRECISION) & (prec["size"] >= MIN_SUPPORT)])
        cv.attrs.update(precision=prec, allowed=sorted(allowed))
        evidence[col] = cv
        todo = in_cat & items[col].isna() & photo
        guess = pd.DataFrame(index=items.index[todo], columns=["value", "method", "confidence"], dtype=object)
        if col == "MATERIAL":
            vocab = set(items.loc[in_cat, col].dropna())
            for method, values in (("sibling", sibling_values(items, col)), ("composition", single_material(items, vocab))):
                free = guess["value"].isna() & values.loc[guess.index].notna()
                guess.loc[free, ["value", "method", "confidence"]] = list(zip(
                    values.loc[guess.index[free]], [method] * free.sum(), [1.0] * free.sum()))
        free = guess.index[guess["value"].isna()]
        if len(free):
            pred, conf = knn_vote(X[free.to_numpy()], X[labelled.to_numpy()], y)
            keep = conf >= MIN_CONF
            if col == "FINISH_FAMILY":
                keep &= photo_agrees(pred, yellow[free.to_numpy()])
            keep &= np.isin(pred, list(allowed))
            guess.loc[free[keep], ["value", "method", "confidence"]] = list(zip(
                pred[keep], ["neighbours"] * keep.sum(), conf[keep].round(3)))
        guess = guess.dropna(subset=["value"])
        rows.append(pd.DataFrame({KEY: items.loc[guess.index, KEY], "column": col, "value": guess["value"],
                                  "method": guess["method"], "confidence": guess["confidence"].astype(float)}))
        evidence[col].attrs.update(todo=int(todo.sum()), missing_no_photo=int((in_cat & items[col].isna() & ~photo).sum()))
    out = pd.concat(rows, ignore_index=True).sort_values(["column", KEY], kind="stable").reset_index(drop=True)
    return out, evidence


def apply_inferred(items: pd.DataFrame, inferred: pd.DataFrame) -> pd.DataFrame:
    """Items with FINISH_FAMILY added and the inferred values filled in (CSV values are never replaced)."""
    items = add_finish_family(items)
    for col, sub in inferred.groupby("column"):
        values = items[KEY].map(sub.set_index(KEY)["value"])
        items[col] = items[col].where(items[col].notna(), values)
    return items


def _pct(x: float) -> str:
    return "–" if pd.isna(x) else f"{x:.0%}"


def write_report(inferred: pd.DataFrame, evidence: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = []
    for col, cv in evidence.items():
        ok = cv["pred"] == cv["truth"]
        kept = [(cv.conf >= t) & cv.photo_ok for t in (0.5, 0.6, 0.7, 0.8, 0.9)]
        rows = "\n".join(f"| ≥ {t:.1f} | {_pct(k.mean())} | {_pct(ok[k].mean())} |"
                         for t, k in zip((0.5, 0.6, 0.7, 0.8, 0.9), kept))
        prec = cv.attrs["precision"].sort_values("size", ascending=False)
        per_class = ", ".join(f"{v} {_pct(r['mean'])} (n={int(r['size'])})" for v, r in prec.iterrows())
        photo_note = "; gold / silver kept only when the photo agrees" if col == "FINISH_FAMILY" else ""
        filled = inferred[inferred["column"] == col]
        methods = "\n".join(f"| {m} | {n:,} |" for m, n in filled["method"].value_counts().items())
        values = ", ".join(f"{v} {n:,}" for v, n in filled["value"].value_counts().items())
        blocks.append(f"""## {col} ({TARGETS[col]})

Cross-validation of the neighbours on the {len(cv):,} items that have the value (folds by model{photo_note}):
overall {_pct(ok.mean())}, majority class {_pct(pd.Series(cv.truth).value_counts(normalize=True).iloc[0])}.

| Vote share | Items kept | Accuracy |
|---|---|---|
{rows}

Precision per predicted value at a vote share ≥ {MIN_CONF}: {per_class}.
Only values with precision ≥ {MIN_PRECISION:.0%} on ≥ {MIN_SUPPORT} predictions are filled by the neighbours:
{", ".join(cv.attrs["allowed"]) or "none"}.

Filled: **{len(filled):,} of {cv.attrs['todo']:,}** items with a photo and no value
({cv.attrs['missing_no_photo']:,} more have no photo and are not filled). Neighbours are kept at a vote share ≥ {MIN_CONF}.

| Method | Items |
|---|---|
{methods}

Values: {values}.
""")
    text = f"""# Phase 2a: inferred attributes

Generated by `src/phase2a_impute.py` → `data/processed/{OUT_FILE}` (one row per filled value, with method and
confidence). The CSV and `items_clean.parquet` are not changed; the feature blocks use these values where the
CSV is empty.

Not filled on purpose: detailed `FINISHING` (golden / worn gold / brushed golden…) and hand bag finishing,
because the neighbours get them right too rarely (about 88% on 30% of the jewellery, and 77% for bags).

{chr(10).join(blocks)}"""
    (REPORT_DIR / "impute_report.md").write_text(text, encoding="utf-8")


def run() -> pd.DataFrame:
    items = phase1_clean.load_items().sort_values(KEY, kind="stable").reset_index(drop=True)
    inferred, evidence = infer(items)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    inferred.to_parquet(PROCESSED_DIR / OUT_FILE, index=False)
    write_report(inferred, evidence)
    return inferred


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    inferred = run()
    print(inferred.groupby(["column", "method"]).size().to_string())
    print(f"-> {PROCESSED_DIR / OUT_FILE}")


if __name__ == "__main__":
    main()
