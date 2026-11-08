"""Phase 2b: read-only audit of the colour palettes and of the way text interacts with them.

    python src/phase2b_palette_audit.py            # -> outputs/phase2b/palette_audit.md + palette_audit_sheet.jpg (~2 min)

Needs `data/processed/` (run phase1_checks.py check, phase1_clean.py, phase2a_features.py once). Light environment
only. Nothing in data/embeddings/ or in the existing scripts is changed; the audit reuses committed files.

Questions answered:
1. How reliable is a palette by photo type (person / other item in the frame / not a packshot / metal / plain)?
   Reliability = how often the dominant palette colour is an acceptable name for the CSV colour `CLR_DES`, and how far
   (CIELAB) it lies from the reference colour of that name. A re-crop test (person photos redone on the item box of
   `person_crops.parquet`) shows what a bad box costs.
2. Text and palette:
   a. is the reference colour of `phase2a_colour` (measured on the photos) polluted by wrong photos?
   b. do the text block and the categorical block already carry colour, so that the palette counts it twice?
      (similarities of the same random pairs; pairs of the same model, where only the colour differs)
   c. what does the palette weight do to the team's preferences (pool of `phase4_feedback.py`, reused) and to the
      Phase 4 metrics (`eval_metrics.csv`, reused)?
3. A recommendation written from those numbers.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
import phase2a_colour as pc  # noqa: E402
import phase2b_palettes as pal_mod  # noqa: E402
import phase3_engine as engine  # noqa: E402
import phase4_feedback as fb  # noqa: E402
from config import EMB_DIR, IMG_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402

OUT_DIR = ROOT / "outputs" / "phase2b"
SEED = 0
PERSON_MIN = 0.10   # OWLv2 score for a person / face / hand / legs, as in phase2b_image_encoders (person_full, the CLIP
                    # filter, is not used: its "hand wearing jewellery" prompt also fires on plain ring packshots)
N_PAIRS = 150_000
SWEEP = [0.0, 0.017, 0.034, 0.068, 0.10, 0.15, 0.20]
CURRENT_PALETTE_WEIGHT = 0.034
TYPES = ["person", "other item in frame", "not a packshot", "metal name", "multicolour name", "plain packshot"]

# Acceptable palette colour names for the colour words of CLR_DES. Wide on purpose: the question is "is this palette
# plausible for this name", not "is it the same word". The first word found in the name decides.
ACCEPT = {
    "black": {"black", "grey", "navy"}, "white": {"white", "silver", "grey", "beige"},
    "ecru": {"white", "beige", "silver", "grey", "camel"}, "ivory": {"white", "beige", "silver", "grey"},
    "cream": {"white", "beige", "silver"}, "off": {"white", "beige", "silver", "grey"},
    "grey": {"grey", "silver", "black", "white"}, "taupe": {"beige", "camel", "brown", "grey"},
    "beige": {"beige", "camel", "white", "gold", "brown"}, "camel": {"camel", "beige", "brown", "gold", "orange"},
    "sand": {"beige", "camel", "white", "gold"}, "nude": {"beige", "camel", "pink", "white"},
    "brown": {"brown", "camel", "burgundy", "black"}, "chocolate": {"brown", "black", "burgundy"},
    "khaki": {"khaki", "green", "brown", "camel", "grey", "beige"},
    "green": {"green", "khaki", "teal", "mint", "yellow"}, "olive": {"khaki", "green", "brown"},
    "mint": {"mint", "green", "light blue", "turquoise", "white"},
    "navy": {"navy", "blue", "black"}, "blue": {"blue", "navy", "light blue", "turquoise", "teal", "lilac", "purple"},
    "turquoise": {"turquoise", "teal", "light blue", "mint", "green", "blue"},
    "teal": {"teal", "turquoise", "green", "blue", "navy"},
    "burgundy": {"burgundy", "red", "brown", "purple"}, "wine": {"burgundy", "red", "brown", "purple"},
    "red": {"red", "burgundy", "orange", "pink", "fuchsia"}, "orange": {"orange", "camel", "red", "yellow"},
    "yellow": {"yellow", "gold", "beige", "orange"}, "pink": {"pink", "fuchsia", "lilac", "beige", "red"},
    "fuchsia": {"fuchsia", "pink", "purple", "red"}, "purple": {"purple", "lilac", "fuchsia", "blue", "pink"},
    "lilac": {"lilac", "purple", "pink", "blue", "light blue"}, "lavender": {"lilac", "purple", "light blue"},
}
METAL_WORDS = {"gold", "silver", "nickel", "copper", "bronze"}
METAL_NAMES = {"gold", "silver", "beige", "camel", "yellow", "white", "grey", "brown", "orange"}


# --------------------------------------------------------------------------- photo types
def accepted_names(clr: str | None) -> set[str] | None:
    """Palette names that fit a CLR_DES, or None for names the check does not cover (metal, multicolour, unknown)."""
    if clr is None or (isinstance(clr, float) and np.isnan(clr)):
        return None
    words = re.findall(r"[a-z]+", str(clr).lower())
    if "multicolor" in words or "multicolour" in words or any(w in METAL_WORDS for w in words):
        return None
    for w in words:
        if w in ACCEPT:
            return ACCEPT[w]
    return None


def photo_types(items: pd.DataFrame, masks: pd.DataFrame, persons: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """One row per photo with its type (first matching rule wins, see TYPES) and the engine flag."""
    t = items[[KEY, "CLR_DES", "img_trusted", "img_colour_trusted"]].merge(
        masks[[KEY, "packshot", "target_found"]], on=KEY, how="inner")
    t = t.merge(persons[[KEY, "person_score", "person_full", "item_box"]], on=KEY, how="left")
    t = t.merge(audit[[KEY, "group_expected", "det_strongest"]], on=KEY, how="left")
    person = t["person_score"] >= PERSON_MIN
    other = (t["det_strongest"].notna() & (t["det_strongest"] != t["group_expected"])) | (t["target_found"] == False)  # noqa: E712
    multi = t["CLR_DES"].astype(str).str.lower().str.contains("multicolor")
    metal = t["CLR_DES"].astype(str).str.lower().map(lambda s: any(w in METAL_WORDS for w in re.findall(r"[a-z]+", s)))
    t["ptype"] = np.select([person, other, ~t["packshot"].fillna(False).astype(bool), metal, multi],
                           TYPES[:5], default=TYPES[5])
    t["person"] = person
    return t


def dominant(pal: pd.DataFrame) -> pd.DataFrame:
    """Dominant palette colour per photo: name, LAB, share."""
    j = pal["shares"].map(lambda s: int(np.argmax(s)))
    return pd.DataFrame({KEY: pal[KEY].to_numpy(),
                         "dom_name": [n[i] for n, i in zip(pal["names"], j)],
                         "dom_share": [s[i] for s, i in zip(pal["shares"], j)],
                         "L": [np.asarray(lab[i])[0] for lab, i in zip(pal["lab"], j)],
                         "a": [np.asarray(lab[i])[1] for lab, i in zip(pal["lab"], j)],
                         "b": [np.asarray(lab[i])[2] for lab, i in zip(pal["lab"], j)],
                         "top2": [set(np.asarray(n)[np.argsort(-np.asarray(s), kind="stable")[:2]])
                                  for n, s in zip(pal["names"], pal["shares"])]})


def _median(x: pd.Series) -> float:
    return float(x.median()) if x.notna().any() else np.nan


def agreement(t: pd.DataFrame) -> pd.DataFrame:
    """Per photo type: photos, share whose dominant colour fits CLR_DES, share whose top-2 contain a fitting colour."""
    acc = t["CLR_DES"].map(accepted_names)
    t = t.assign(covered=acc.notna(),
                 ok_dom=[(d in a) if a else np.nan for d, a in zip(t["dom_name"], acc)],
                 ok_top2=[bool(s & a) if a else np.nan for s, a in zip(t["top2"], acc)])
    rows = []
    for name, g in t.groupby("ptype"):
        c = g[g["covered"]]
        rows.append({"photo type": name, "photos": len(g), "checked": len(c),
                     "dominant fits CLR_DES": c["ok_dom"].astype(float).mean() if len(c) else np.nan,
                     "top 2 fit CLR_DES": c["ok_top2"].astype(float).mean() if len(c) else np.nan,
                     "median distance to reference": _median(g["dist_ref"])})
    out = pd.DataFrame(rows).set_index("photo type").reindex(TYPES).dropna(how="all")
    return out


def reference_distance(t: pd.DataFrame, ref: pd.DataFrame) -> pd.Series:
    """Distance (phase2a_colour convention: lightness counted half) of each photo's dominant colour to the reference
    colour of its CLR_DES; NaN for names without a reference."""
    r = ref.set_index("CLR_DES")[["L", "a", "b"]]
    lab = r.reindex(t["CLR_DES"].astype(str)).to_numpy(float)
    own = t[["L", "a", "b"]].to_numpy(float)
    w = np.array([pc.L_WEIGHT, 1.0, 1.0])
    return pd.Series(np.sqrt((((own - lab) * w) ** 2).sum(1)), index=t.index)


# --------------------------------------------------------------------------- metals
def metal_table(t: pd.DataFrame) -> pd.DataFrame:
    """How a gold or a silver item looks on the palette: dominant name shares and the gold-vs-silver separation."""
    rows = []
    for word in ("gold", "silver"):
        g = t[t["CLR_DES"].astype(str).str.lower().str.contains(word)]
        rows.append({"CLR_DES contains": word, "photos": len(g),
                     "dominant = gold": (g["dom_name"] == "gold").mean(), "dominant = silver": (g["dom_name"] == "silver").mean(),
                     "dominant beige/camel/white": g["dom_name"].isin(["beige", "camel", "white"]).mean(),
                     "median b* (yellowness)": g["b"].median(), "median L*": g["L"].median()})
    return pd.DataFrame(rows).set_index("CLR_DES contains")


def auc(x: np.ndarray, y: np.ndarray) -> float:
    """P(x > y) with ties at one half (rank-sum form of the Mann-Whitney U)."""
    r = pd.Series(np.concatenate([x, y])).rank().to_numpy()
    return float((r[: len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


# --------------------------------------------------------------------------- re-crop test
def recrop_test(t: pd.DataFrame, limit: int = 600) -> pd.DataFrame:
    """Redo the palette of person photos on the item box suggested in person_crops (read-only, in memory)."""
    cand = t[t["person"] & t["item_box"].notna() & t["img_colour_trusted"].fillna(False).astype(bool)]
    cand = cand.sort_values(KEY).head(limit)
    files = pd.read_parquet(EMB_DIR / "palettes.parquet", columns=[KEY, "img_file"]).set_index(KEY)["img_file"]
    rows = []
    for r in cand.itertuples():
        try:
            with Image.open(IMG_DIR / files[getattr(r, KEY)]) as img:
                e = pal_mod.palette(pal_mod.product_pixels(img, list(r.item_box)))
        except (OSError, KeyError):
            continue
        k = int(np.argmax([x["share"] for x in e]))
        rows.append({KEY: getattr(r, KEY), "new_name": e[k]["name"], "new_lab": e[k]["lab"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reference pollution
def pollution(items: pd.DataFrame, pal: pd.DataFrame, t: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Reference colours with all photos vs. without person / other-item photos: shift per name and flipped close pairs."""
    ref_all = pc.reference_colours(items, pal)
    bad = set(t.loc[t["ptype"].isin(["person", "other item in frame"]), KEY])
    ref_clean = pc.reference_colours(items, pal[~pal[KEY].isin(bad)])
    both = ref_all.merge(ref_clean, on="CLR_DES", suffixes=("", "_clean"))
    w = np.array([pc.L_WEIGHT, 1.0, 1.0])
    d = np.sqrt((((both[["L", "a", "b"]].to_numpy(float) - both[["L_clean", "a_clean", "b_clean"]].to_numpy(float)) * w) ** 2).sum(1))
    both["shift"] = d
    both["lost_reference"] = both["no_reference"].isna() & both["no_reference_clean"].notna()
    s_all, s_clean = pc.similarity(ref_all), pc.similarity(ref_clean)
    iu = np.triu_indices(len(ref_all), 1)
    close_a, close_c = s_all[iu] >= 0.5, s_clean[iu] >= 0.5
    stats = {"names_with_reference": int(ref_all["no_reference"].isna().sum()),
             "photos_used": int(ref_all.loc[ref_all["no_reference"].isna(), "photos"].sum()),
             "photos_flagged_in_reference": int(t[t["ptype"].isin(["person", "other item in frame"])
                                                  & t["img_colour_trusted"].fillna(False).astype(bool)
                                                  & t["packshot"].fillna(False).astype(bool)][KEY].nunique()),
             "close_pairs": int(close_a.sum()), "close_pairs_clean": int(close_c.sum()),
             "pairs_changed": int((close_a != close_c).sum()),
             "names_shift_gt_3": int((both["shift"] > 3).sum()), "median_shift": float(both["shift"].median()),
             "max_shift": float(both["shift"].max()), "lost_reference": int(both["lost_reference"].sum())}
    return both, stats


# --------------------------------------------------------------------------- overlap with text and categorical
def sample_pairs(groups: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """n random pairs (i, j), i != j, with the same group label."""
    codes = pd.factorize(groups)[0]
    order = np.argsort(codes, kind="stable")
    count = np.bincount(codes)
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    i = rng.integers(len(codes), size=n)
    j = order[start[codes[i]] + (rng.random(n) * count[codes[i]]).astype(int)]
    keep = (i != j) & (count[codes[i]] > 1)
    return i[keep], j[keep]


def sibling_pairs(model: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Pairs of different colourways of one model (only the colour, and what comes with it, differs)."""
    i, j = sample_pairs(model, n, rng)
    return i, j


def pair_sims(blocks: dict, has: dict, i: np.ndarray, j: np.ndarray, colour_slice: np.ndarray) -> pd.DataFrame:
    out = {}
    for b in ["image", "categorical", "text", "composition", "palette", "numeric"]:
        s = np.einsum("ij,ij->i", blocks[b][i], blocks[b][j])
        out[f"sim_{b}"] = np.where((has[b][i] > 0) & (has[b][j] > 0), s, np.nan)
    # the part of the categorical similarity that comes from the colour attributes (CLR_DES, CLR_TYPE)
    cat = blocks["categorical"]
    out["sim_cat_colour"] = np.einsum("ij,ij->i", cat[i][:, colour_slice], cat[j][:, colour_slice])
    return pd.DataFrame(out)


def colour_columns(meta_path: Path = PROCESSED_DIR / "feature_meta.json") -> np.ndarray:
    attrs = json.loads(meta_path.read_text(encoding="utf-8"))["blocks"]["categorical"]["attributes"]
    cols = [np.arange(a["start"], a["stop"]) for k, a in attrs.items() if k in ("CLR_DES", "CLR_TYPE")]
    return np.concatenate(cols)


def corr_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df[cols].corr(method="spearman").round(2)


def colour_effect(items: pd.DataFrame, blocks: dict, has: dict, rng: np.random.Generator, colour_slice: np.ndarray):
    """Random pairs of the same sub-family: mean block similarity when the colour name is the same vs. different."""
    i, j = sample_pairs(items["GFS_DES_EN"].astype(str).to_numpy(), N_PAIRS, rng)
    s = pair_sims(blocks, has, i, j, colour_slice)
    same = (items["CLR_DES"].to_numpy()[i] == items["CLR_DES"].to_numpy()[j])
    rows = []
    for c in ["sim_text", "sim_palette", "sim_categorical", "sim_cat_colour", "sim_image", "sim_composition"]:
        a, b = s.loc[same, c].mean(), s.loc[~same, c].mean()
        rows.append({"block": c.replace("sim_", ""), "same colour name": a, "different colour name": b, "gap": a - b,
                     "gap in SDs": (a - b) / s[c].std()})
    out = pd.DataFrame(rows).set_index("block")
    w = engine.load_weights()
    wsum = sum(w.values())
    # what a colour match is worth in the final score: gap x weight (cat_colour sits inside the categorical weight)
    key = {"text": "text", "palette": "palette", "categorical": "categorical", "cat_colour": "categorical",
           "image": "image", "composition": "composition"}
    out["score points of the gap"] = [out.loc[b, "gap"] * w[key[b]] / wsum for b in out.index]
    return out, int(same.sum()), int((~same).sum())


# --------------------------------------------------------------------------- the team's preferences
def pool_sweep(flagged: set[str]) -> tuple[pd.DataFrame, dict]:
    """Pair accuracy and similar@3 of the team's sorting for different palette weights, on the committed pool."""
    pool, records = fb.load_pool(), fb.read_feedback()
    index, s, h = fb.sim_matrix(pool)
    blocks = pool["blocks"]
    w0 = np.array([engine.load_weights()[b] for b in blocks])
    k = blocks.index("palette")
    qs = sorted({r["query"] for r in records})
    win, lose = fb.preference_pairs(records, index)
    # which pool rows involve a flagged photo (query or candidate)
    rowkey = {v: kk for kk, v in index.items()}
    flag_row = np.array([(rowkey[r][0] in flagged) or (rowkey[r][1] in flagged) for r in range(len(s))])
    h_drop = h.copy()
    h_drop[flag_row, k] = 0.0

    def per_query(w, hh):
        sc = fb.scores(w, s, hh)
        out = {}
        for r in records:
            cands = r["similar"] + r["not"]
            pairs = [(a, b) for ii, a in enumerate(r["similar"]) for b in r["similar"][ii + 1:] + r["not"]]
            ok = [sc[index[(r["query"], a)]] > sc[index[(r["query"], b)]] for a, b in pairs]
            order = sorted(cands, key=lambda c: (-round(sc[index[(r["query"], c)]], 6), c))
            out[(r["member"], r["query"])] = (np.sum(ok), len(ok), np.mean([c in r["similar"] for c in order[:3]]))
        return out

    def summary(res):
        v = np.array(list(res.values()), dtype=float)
        return v[:, 0].sum() / max(v[:, 1].sum(), 1), v[:, 2].mean()

    def boot(res_a, res_b, n=500):
        keys = list(res_a)
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(n):
            pick = rng.integers(len(keys), size=len(keys))
            a = np.array([res_a[keys[p]] for p in pick], dtype=float)
            b = np.array([res_b[keys[p]] for p in pick], dtype=float)
            diffs.append(a[:, 0].sum() / max(a[:, 1].sum(), 1) - b[:, 0].sum() / max(b[:, 1].sum(), 1))
        return np.percentile(diffs, [2.5, 97.5])

    base = per_query(w0, h)
    rows = []
    for v in SWEEP:
        w = w0.copy()
        w[k] = v
        r = per_query(w, h)
        acc, top3 = summary(r)
        lo, hi = boot(r, base)
        rows.append({"palette weight": v, "pair accuracy": acc, "similar@3": top3, "vs current (low)": lo, "vs current (high)": hi})
    r = per_query(w0, h_drop)
    acc, top3 = summary(r)
    lo, hi = boot(r, base)
    rows.append({"palette weight": "0.034, dropped on other-item / non-packshot photos", "pair accuracy": acc, "similar@3": top3,
                 "vs current (low)": lo, "vs current (high)": hi})
    info = {"pairs": int(len(win)), "queries": len(qs), "members": len({r["member"] for r in records}),
            "flagged_rows_share": float(flag_row.mean())}
    return pd.DataFrame(rows), info


# --------------------------------------------------------------------------- contact sheet
def contact_sheet(worst: pd.DataFrame, files: pd.Series, path: Path, w: int = 190, cols: int = 6) -> None:
    rows = (len(worst) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * (w + 78)), "white")
    d = ImageDraw.Draw(sheet)
    for n, r in enumerate(worst.itertuples()):
        x, y = (n % cols) * w, (n // cols) * (w + 78)
        try:
            with Image.open(IMG_DIR / files[getattr(r, KEY)]) as im:
                im = im.convert("RGB")
                im.thumbnail((w - 8, w - 8))
                sheet.paste(im, (x + 4, y + 4))
        except (OSError, KeyError):
            pass
        sx = x + 4
        for hx, sh in zip(r.hex, r.shares):
            ww = max(1, int((w - 8) * sh))
            d.rectangle([sx, y + w, sx + ww, y + w + 16], fill=hx)
            sx += ww
        d.text((x + 4, y + w + 20), f"{getattr(r, KEY)} | {r.ptype}", fill="black")
        d.text((x + 4, y + w + 33), f"CLR_DES: {str(r.CLR_DES)[:26]}", fill=(180, 0, 0))
        d.text((x + 4, y + w + 46), f"palette: {', '.join(list(r.names)[:3])}", fill="black")
        d.text((x + 4, y + w + 59), f"distance to reference: {r.dist_ref:.0f}", fill=(90, 90, 90))
    sheet.save(path, quality=85)


def worst_cases(t: pd.DataFrame, pal: pd.DataFrame, n_all: int = 18, n_person: int = 12) -> pd.DataFrame:
    """Photos feeding the engine whose palette is furthest from the reference of their colour name."""
    eng = t[t["img_colour_trusted"].fillna(False).astype(bool) & t["dist_ref"].notna() & (t["fits"] == False)]  # noqa: E712
    a = eng.sort_values(["dist_ref", KEY], ascending=[False, True]).head(n_all)
    b = eng[eng["ptype"] == "person"].sort_values(["dist_ref", KEY], ascending=[False, True])
    b = b[~b[KEY].isin(a[KEY])].head(n_person)
    pick = pd.concat([b, a])
    return pick.merge(pal[[KEY, "names", "hex", "shares"]], on=KEY)


# --------------------------------------------------------------------------- report
INT_COLUMNS = {"photos", "checked", "photos (all)", "photos (without person/other)"}


def md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 2, index: bool = True) -> str:
    def f(c, v):
        if c in INT_COLUMNS and not pd.isna(v):
            return f"{int(v):,}"
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return ""
        if c in pct:
            return f"{v:.0%}"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else f"{v:,.{digits}f}"
    cols = ([df.index.name or ""] if index else []) + list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        cells = ([str(idx)] if index else []) + [f(c, r[c]) for c in df.columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    items = phase1_clean.load_items()
    masks = pd.read_parquet(EMB_DIR / "masks.parquet")
    persons = pd.read_parquet(EMB_DIR / "person_crops.parquet")
    audit = pd.read_parquet(EMB_DIR / "image_audit.parquet")
    pal = pd.read_parquet(EMB_DIR / "palettes.parquet")

    # --- 1. quality by photo type
    t = photo_types(items, masks, persons, audit)
    t = t[t[KEY].isin(pal[KEY])].merge(dominant(pal), on=KEY)
    ref = pc.reference_colours(items, pal)
    t["dist_ref"] = reference_distance(t, ref)
    acc_names = t["CLR_DES"].map(accepted_names)
    t["fits"] = [(d in a) if a else np.nan for d, a in zip(t["dom_name"], acc_names)]
    t["in_engine"] = t["img_colour_trusted"].fillna(False).astype(bool)
    pal_reliable = pal.set_index(KEY)["reliable"]
    t["reliable_flag"] = t[KEY].map(pal_reliable)
    engine_t = t[t["in_engine"]]
    q_all = agreement(t)
    q_engine = agreement(engine_t)
    person_marked_reliable = float(t.loc[t["ptype"] == "person", "reliable_flag"].mean())

    metals = metal_table(t)
    g = t[t["CLR_DES"].astype(str).str.lower().str.contains("gold")]
    s = t[t["CLR_DES"].astype(str).str.lower().str.contains("silver")]
    auc_b = auc(g["b"].to_numpy(), s["b"].to_numpy())
    auc_l = auc(s["L"].to_numpy(), g["L"].to_numpy())

    rc = recrop_test(t)
    if len(rc):
        rc = rc.merge(t[[KEY, "CLR_DES", "dom_name", "L", "a", "b"]], on=KEY)
        acc = rc["CLR_DES"].map(accepted_names)
        cov = acc.notna()
        old_ok = np.mean([d in a for d, a in zip(rc.loc[cov, "dom_name"], acc[cov])]) if cov.any() else np.nan
        new_ok = np.mean([d in a for d, a in zip(rc.loc[cov, "new_name"], acc[cov])]) if cov.any() else np.nan
        same_name = float((rc["dom_name"] == rc["new_name"]).mean())
        old_lab = rc[["L", "a", "b"]].to_numpy(float)
        new_lab = np.stack(rc["new_lab"].map(np.asarray))
        shift = np.sqrt(((((old_lab - new_lab) * np.array([pc.L_WEIGHT, 1, 1]))) ** 2).sum(1))
        rc_info = {"n": len(rc), "covered": int(cov.sum()), "old_ok": old_ok, "new_ok": new_ok, "same_name": same_name,
                   "median_shift": float(np.median(shift)), "p75": float(np.percentile(shift, 75))}
    else:
        rc_info = None

    # --- 2a. reference pollution
    both, pstats = pollution(items, pal, t)
    top_shift = both[both["shift"] > 0].sort_values("shift", ascending=False).head(8)[
        ["CLR_DES", "photos", "photos_clean", "shift"]].set_index("CLR_DES")

    # --- 2b. overlap
    blocks = engine.load_blocks(items)
    has = engine.has_block(blocks)
    cs = colour_columns()
    cat = items["CAT_DES_EN"].astype(str).to_numpy()
    i, j = sample_pairs(cat, N_PAIRS, rng)
    rand = pair_sims(blocks, has, i, j, cs)
    cols = ["sim_palette", "sim_text", "sim_categorical", "sim_cat_colour", "sim_image"]
    c_rand = corr_table(rand, cols)
    model = items["PROD_REF"].astype(str).to_numpy()
    si, sj = sibling_pairs(model, N_PAIRS // 3, rng)
    different = items["CLR_DES"].to_numpy()[si] != items["CLR_DES"].to_numpy()[sj]
    sib = pair_sims(blocks, has, si[different], sj[different], cs)
    c_sib = corr_table(sib, ["sim_palette", "sim_text", "sim_cat_colour", "sim_image"])
    sib_mean = sib[["sim_text", "sim_palette", "sim_cat_colour", "sim_image"]].mean()
    eff, n_same, n_diff = colour_effect(items, blocks, has, rng, cs)
    nb = pd.read_parquet(PROCESSED_DIR / "neighbours.parquet")
    c_nb = corr_table(nb, ["sim_palette", "sim_text", "sim_categorical"])
    w = engine.load_weights()
    wsum = sum(w.values())
    # share of the score colour reaches through text and categorical (CLR_DES slice) vs. the palette

    # --- 2c. weights
    flagged = set(t.loc[t["ptype"].isin(["other item in frame", "not a packshot"]), KEY])
    sweep, sinfo = pool_sweep(flagged)
    em = pd.read_csv(ROOT / "outputs" / "phase4" / "eval_metrics.csv", encoding="utf-8-sig").set_index("configuration")
    ab = em.loc[["current weights", "without palette", "only palette"],
                ["hit@4", "same_colour", "same_finishing", "visual@4", "sales_corr"]]

    # --- sheet
    worst = worst_cases(t, pal)
    contact_sheet(worst, pal.set_index(KEY)["img_file"], OUT_DIR / "palette_audit_sheet.jpg")

    write_report(locals())
    print(f"Report: {OUT_DIR / 'palette_audit.md'}\nSheet: {OUT_DIR / 'palette_audit_sheet.jpg'}")


def write_report(v: dict) -> None:
    q_all, q_engine, metals, rc_info, pstats = v["q_all"], v["q_engine"], v["metals"], v["rc_info"], v["pstats"]
    sweep, sinfo, ab, eff = v["sweep"], v["sinfo"], v["ab"], v["eff"]
    engine_t = v["engine_t"]
    person_marked_reliable = v["person_marked_reliable"]
    n_eng = len(engine_t)
    plain = q_engine.loc["plain packshot"] if "plain packshot" in q_engine.index else None
    pers = q_engine.loc["person"] if "person" in q_engine.index else None
    other = q_engine.loc["other item in frame"]
    nop = q_engine.loc["not a packshot"]
    cur = sweep.iloc[SWEEP.index(CURRENT_PALETTE_WEIGHT)]
    drop = sweep.iloc[-1]
    zero = sweep.iloc[0]
    best = sweep.iloc[:-1].sort_values(["pair accuracy", "similar@3"], ascending=False).iloc[0]
    r_rand = v["c_rand"].loc["sim_palette", "sim_text"]
    r_sib_text = v["c_sib"].loc["sim_palette", "sim_text"]

    def verdict(row):
        lo, hi = row["vs current (low)"], row["vs current (high)"]
        return "no clear difference" if lo <= 0 <= hi else ("better" if lo > 0 else "worse")

    rc_text = "No person photo with a suggested item box could be redone."
    if rc_info:
        rc_text = (f"{rc_info['n']} person photos used by the engine were redone on the item box of `person_crops.parquet` "
                   f"(in memory only). The dominant colour name stays the same for {rc_info['same_name']:.0%} of them; "
                   f"the median move of the dominant colour is {rc_info['median_shift']:.1f} units (75th percentile "
                   f"{rc_info['p75']:.1f}). Share of the checked photos whose dominant colour fits `CLR_DES`: "
                   f"{rc_info['old_ok']:.0%} with the current box, {rc_info['new_ok']:.0%} on the item box "
                   f"({rc_info['covered']} photos checked).")

    gap_pp = (plain["dominant fits CLR_DES"] - pers["dominant fits CLR_DES"]) if (plain is not None and pers is not None) else np.nan

    text = f"""# Phase 2b: audit of the colour palette and of the text effect on it

Generated by `src/phase2b_palette_audit.py` (read-only: no embedding, no existing script and no data file was changed).
Reused files: `data/embeddings/{{masks,palettes,person_crops,image_audit}}.parquet`, `data/processed/neighbours.parquet`,
`outputs/phase4/feedback_pool.json` + `feedback_*.jsonl`, `outputs/phase4/eval_metrics.csv`.

How the palette is used today: 4 k-means colours per photo, in the Phase 3 engine as block `palette` (weight
{CURRENT_PALETTE_WEIGHT}, {CURRENT_PALETTE_WEIGHT / sum(engine.load_weights().values()):.1%} of the score) for the {n_eng:,}
photos with `img_colour_trusted`. In Phase 5 the shared palette colour names are one fact of the explanation, and
`phase2a_colour` builds the reference colour of each `CLR_DES` from the same palettes.

## Summary

- The palette is reliable on plain packshots: the dominant colour fits `CLR_DES` for
  {plain['dominant fits CLR_DES']:.0%} of them (median distance to the reference colour {plain['median distance to reference']:.1f}).
- Photos with a person (detector) are only a little worse: {pers['dominant fits CLR_DES']:.0%} ({gap_pp * 100:.0f} points
  less than plain packshots, {int(pers['checked'])} photos checked), and redoing them on the item box changes nothing.
  The weak spots are photos where the detector finds another item or not the target
  ({other['dominant fits CLR_DES']:.0%}, {int(other['photos'])} in the engine) and photos that are not packshots
  ({nop['dominant fits CLR_DES']:.0%}, {int(nop['photos'])} in the engine). `reliable` in `palettes.parquet` only means
  "white background", so {person_marked_reliable:.0%} of the person photos are marked reliable.
- Gold and silver cannot be told apart by name, but the palette still separates them a little (AUC of yellowness
  {v['auc_b']:.2f}); gold items look beige or white, as the Phase 2b report said.
- Text and the categorical block carry colour too, but the palette does not repeat them: on random pairs the palette
  correlates {r_rand:.2f} with the text. A colour match is worth {eff.loc['cat_colour', 'score points of the gap']:.3f}
  score points through the categorical colour attributes, {eff.loc['palette', 'score points of the gap']:.3f} through the
  palette and {eff.loc['text', 'score points of the gap']:.3f} through the text (same-colour pairs of one sub-family, gap x weight).
- Palette weight on the team's preferences ({sinfo['pairs']:,} pairs, {sinfo['queries']} queries): current
  {cur['pair accuracy']:.1%} pair accuracy; weight 0 gives {zero['pair accuracy']:.1%} ({verdict(zero)}); the best weight
  tried ({best['palette weight']}) gives {best['pair accuracy']:.1%}. Dropping the palette on other-item / non-packshot photos gives
  {drop['pair accuracy']:.1%} ({verdict(drop)}).

## 1. Palette quality by photo type

Photo types (the first rule that matches): **person** = `person_score >= {PERSON_MIN}` in
`person_crops.parquet` (the detector found a person, face, hand or legs; the Phase 1b box then holds skin, hair and
other garments). The CLIP score `person_full` is not used: a first version with it flagged plain ring packshots;
**other item in frame** = the detector's strongest item differs from the expected group, or the target was not found;
**not a packshot** = `packshot = False` in `masks.parquet`; **metal name** = `CLR_DES` with gold, silver, nickel, copper
or bronze; **multicolour name**; **plain packshot** = the rest. "Several items in one photo" can only be approximated:
the detector gives the strongest item, not a count, so "other item in frame" is a lower bound.

"Fits `CLR_DES`" means the palette colour is in a list of acceptable names for the colour word of `CLR_DES` (for
example `Burgundy` accepts burgundy, red, brown, purple; the list is `ACCEPT` in the script). Metal and multicolour
names are not checked. The distance is the one of `phase2a_colour` (CIELAB, lightness counted half) between the
dominant colour and the reference colour of the name.

**Photos that feed the engine (`img_colour_trusted`):**

{md(q_engine, pct={"dominant fits CLR_DES", "top 2 fit CLR_DES"}, digits=1)}

**All photos with a palette:**

{md(q_all, pct={"dominant fits CLR_DES", "top 2 fit CLR_DES"}, digits=1)}

**Re-crop test (person photos).** {rc_text}

**Metals.**

{md(metals, pct={"dominant = gold", "dominant = silver", "dominant beige/camel/white"}, digits=1)}

The yellowness (b*) of gold items is higher than that of silver items in {v['auc_b']:.0%} of random pairs, and silver is
lighter than gold in {v['auc_l']:.0%}. The palette therefore keeps a weak gold-versus-silver signal, but it names both
mostly as beige, white or silver. The metal itself should keep coming from `FINISHING` / `FINISH_FAMILY`.

Worst cases (photo, swatches, `CLR_DES`, palette names, distance to the reference): `palette_audit_sheet.jpg`. Only photos whose dominant colour does not fit the name. The first
12 are person photos, the next 18 the worst of the others.

## 2. Impact of text on the palette

### 2a. Is the reference colour of each name polluted?

`phase2a_colour` takes the median dominant colour of the trusted packshots with each `CLR_DES`. It does not remove
person photos (they are `packshot = True` when the background is white). Here it is recomputed without the person and
other-item photos and compared:

- {pstats['names_with_reference']} names have a reference, built from {pstats['photos_used']:,} photos;
  {pstats['photos_flagged_in_reference']:,} photos that feed the engine are person or other-item photos.
- Median shift of a reference colour when they are removed: {pstats['median_shift']:.2f} units (maximum
  {pstats['max_shift']:.1f}); {pstats['names_shift_gt_3']} names move by more than 3 units;
  {pstats['lost_reference']} names would lose the reference (under 5 photos left).
- Pairs of names counted as "close colours" (similarity at least 0.5): {pstats['close_pairs']} with all photos,
  {pstats['close_pairs_clean']} without; {pstats['pairs_changed']} pairs change.

Largest shifts:

{md(top_shift_fmt(v['top_shift']), digits=2)}

The median is robust, so the reference is only lightly polluted. The reference is for the Phase 5 wording only, not
for the score.

### 2b. Does text carry colour, so that colour is counted twice?

The text embedding describes each item with "Colour: {{CLR_DES}}"; the categorical block has a one-hot `CLR_DES`
(attribute weight 2) and `CLR_TYPE`. Both therefore react to the colour name, in addition to the palette.

Spearman correlation of the block similarities for {len(v['rand']):,} random pairs of one category:

{md(v['c_rand'])}

Pairs of different colourways of the same model ({len(v['sib']):,} pairs): everything else is equal, so a correlation
with the palette can only come from colour.

{md(v['c_sib'])}

Same colour name against a different colour name, random pairs of one sub-family
({v['n_same']:,} same, {v['n_diff']:,} different). A block that carries colour has a clear gap:

{md(eff, digits=3)}

Top-10 pairs of the engine (`neighbours.parquet`), where pairs are already selected for being similar:

{md(v['c_nb'])}

Reading:

- On random pairs the palette correlates with the text ({r_rand:.2f}) and with the colour part of the categorical block
  ({v['c_rand'].loc['sim_palette', 'sim_cat_colour']:.2f}) hardly at all, so the three do not repeat each other
  pair by pair. The palette measures the look of the photo; the other two measure the name.
- Among colourways of one model, text and palette similarity are almost unrelated ({r_sib_text:.2f}); the text
  similarity of such siblings is {v['sib_mean']['sim_text']:.3f} on average, so the colour words move the text
  embedding by at most {1 - v['sib_mean']['sim_text']:.3f} (everything else in the sentence is equal).
- What a colour match is worth in the score (last column of the gap table, gap x weight): the categorical colour
  attributes {eff.loc['cat_colour', 'score points of the gap']:.3f}, the palette {eff.loc['palette', 'score points of the gap']:.3f},
  the text {eff.loc['text', 'score points of the gap']:.3f}. The name is counted mainly by the categorical block;
  text is the smallest of the three. (Same-colour pairs also look alike in other ways, e.g. image and composition,
  so the text gap is an upper bound of the effect of the colour words.)

### 2c. What does the palette weight do?

Team preferences: the pool of `phase4_feedback.py` ({sinfo['members']} members, {sinfo['queries']} sorted queries,
{sinfo['pairs']:,} preference pairs). Only the palette weight changes; the other weights are the current ones. Bootstrap
over the sorted queries, 95% interval of the difference in pair accuracy against the current weight.

{md(sweep.set_index("palette weight"), pct={"pair accuracy", "similar@3"}, digits=3)}

{sinfo['flagged_rows_share']:.0%} of the pool pairs involve a flagged photo (other item in frame or not a packshot).

Phase 4 ablation (`eval_metrics.csv`, proxy metrics on all queries):

{md(ab, pct={"hit@4", "same_colour", "same_finishing"}, digits=3)}

## 3. Recommendation

{recommend(v)}
"""
    (OUT_DIR / "palette_audit.md").write_text(text, encoding="utf-8")


def recommend(v: dict) -> str:
    q, sw, eff, ab = v["q_engine"], v["sweep"], v["eff"], v["ab"]
    plain, pers, other, nop = q.loc["plain packshot"], q.loc["person"], q.loc["other item in frame"], q.loc["not a packshot"]
    n_flag = int(other["photos"] + nop["photos"])
    n_eng = len(v["engine_t"])
    cur, zero, drop = sw.iloc[SWEEP.index(CURRENT_PALETTE_WEIGHT)], sw.iloc[0], sw.iloc[-1]
    high = sw[sw["palette weight"].isin([0.068, 0.10])]
    rc = v["rc_info"]
    rc_line = (f"Person photos redone on the item box: {rc['old_ok']:.0%} fit with the current box, {rc['new_ok']:.0%} on the item "
               f"box, so a re-crop for the palette gains nothing.") if rc else ""
    return f"""**1. Photos that should not feed the palette.** Leave out of the palette block the photos where the detector finds
another item or not the target ({int(other['photos'])} in the engine) and the photos that are not packshots
({int(nop['photos'])}): {n_flag} of {n_eng:,} ({n_flag / n_eng:.0%}). Their dominant colour fits `CLR_DES` for
{other['dominant fits CLR_DES']:.0%} and {nop['dominant fits CLR_DES']:.0%} against {plain['dominant fits CLR_DES']:.0%} for plain
packshots, and their colour lies {other['median distance to reference']:.0f} and {nop['median distance to reference']:.0f}
units from the reference of the name against {plain['median distance to reference']:.0f}. Person photos are only
{(plain['dominant fits CLR_DES'] - pers['dominant fits CLR_DES']) * 100:.0f} points worse ({int(pers['checked'])} checked,
{int(pers['photos'])} in the engine), so the data does not justify excluding them. {rc_line}
Dropping the palette on the other-item and non-packshot photos costs nothing on the team's preferences (pair accuracy
{drop['pair accuracy']:.1%} against {cur['pair accuracy']:.1%}, the interval includes 0), so the change is safe.
Concretely: build the palette mask of the engine (and `reliable`) from the photo type and not from `packshot` alone;
the photos stay in the `image` block. Same filter for the reference colours of `phase2a_colour`
(small effect: median shift {v['pstats']['median_shift']:.2f}, {v['pstats']['pairs_changed']} of
{v['pstats']['close_pairs']} close-colour pairs change, but it is free).

**What the worst cases are** (`palette_audit_sheet.jpg`): mostly not palette errors. Many are names that describe a
detail of the item (a gold ring with a black stone is `Black`) or a vivid colour whose reference is a dull median (a
lime `Green`). The palette is then right and the name is the odd one, so the distance to the reference alone must not be
used to discard photos.

**2. Palette weight.** Keep {CURRENT_PALETTE_WEIGHT}; do not raise it. At 0.068 and 0.10 the pair accuracy of the team falls by
{(cur['pair accuracy'] - high['pair accuracy'].max()) * 100:.1f} to {(cur['pair accuracy'] - high['pair accuracy'].min()) * 100:.1f} points (the 95% interval excludes 0),
while 0 and 0.017 are statistically the same as the current value ({zero['pair accuracy']:.1%} / {cur['pair accuracy']:.1%}).
The Phase 4 proxies say the palette helps a little (without it: same colour
{ab.loc['without palette', 'same_colour']:.0%} against {ab.loc['current weights', 'same_colour']:.0%}, visual@4
{ab.loc['without palette', 'visual@4']:.3f} against {ab.loc['current weights', 'visual@4']:.3f}), so there is no reason to
remove it, and the Phase 5 explanation uses it.

**3. Text.** The colour words of the description do not need to go. A colour match is worth
{eff.loc['text', 'score points of the gap']:.3f} score points through the text, against
{eff.loc['palette', 'score points of the gap']:.3f} through the palette and
{eff.loc['cat_colour', 'score points of the gap']:.3f} through the categorical colour attributes. The block that counts the
colour name most is the categorical one (`CLR_DES`, attribute weight 2.0); that is the one to look at if buyers say
that colour dominates the result. If the text is ever re-embedded (vision machine), the "Colour:" sentence can be
dropped, but this audit cannot test that and the expected gain is below 0.01 of the score.

**4. Metals.** Palette names do not separate gold and silver (gold items: {v['metals'].loc['gold', 'dominant = gold']:.0%} named
gold, {v['metals'].loc['gold', 'dominant beige/camel/white']:.0%} beige, camel or white). Keep using `FINISHING` /
`FINISH_FAMILY` for the metal and never write "gold" in an explanation from the palette alone.
"""


def top_shift_fmt(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"photos": "photos (all)", "photos_clean": "photos (without person/other)", "shift": "shift"})


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    run()


if __name__ == "__main__":
    main()
