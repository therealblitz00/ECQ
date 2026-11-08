"""Phase 2b: photos where the image shows more than the product being matched (a model, several pieces).

Light environment only (pandas, numpy, Pillow); no model is run. Everything is a SUGGESTION for a human to check.

    python src/phase2b_multi_item.py run     # count + suggest crops + test with the saved embeddings
                                             # -> data/embeddings/multi_item_crops.parquet (+ .json)
                                             #    data/embeddings/multi_item_reviewed.json (template, only if missing)
                                             #    outputs/phase2b/multi_item_crops_sheet.jpg, multi_item_report.md

The problem (the professor's question): the image vector and the palette of a photo describe everything in the crop.
- A model wearing several items: the crop is the person, so the vector describes the outfit.
- A set of pieces ("3 Units (Different)", a set of rings): the vector describes the group, not one product.
- A pair of earrings ("Set 1 Pair") is the normal case for earrings and is left alone.

Cases and rules (see `RULES`): person found by OWLv2 -> crop to the item box when it is clearly smaller than the current
crop; apparel with no usable box -> garment region from `OUTFIT`; no usable crop -> fall back to the tabular blocks
(flag, never drop); several pieces -> flag. A crop is only applied when its key is in
`data/embeddings/multi_item_reviewed.json` (`accepted`), filled by a person after the before/after sheet; this script
never fills it and never touches the other files in data/embeddings/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMB_DIR, IMG_DIR, KEY, ROOT  # noqa: E402

CROPS_FILE = EMB_DIR / "multi_item_crops.parquet"
REVIEWED_FILE = EMB_DIR / "multi_item_reviewed.json"
PERSON_REVIEWED_FILE = EMB_DIR / "person_crops_reviewed.json"   # read only
OUT_DIR = ROOT / "outputs" / "phase2b"
SHEET_FILE = OUT_DIR / "multi_item_crops_sheet.jpg"
REPORT_FILE = OUT_DIR / "multi_item_report.md"

PERSON_MIN = 0.10          # same OWLv2 thresholds as phase2b_image_encoders
ITEM_MIN = 0.10
CLIP_PERSON_MIN = 0.50     # CLIP-only evidence: noisy on jewellery packshots ("a hand wearing jewellery"), never acted on
MAX_SHARE_STRICT = 0.70    # the existing auto_rule
MAX_SHARE_RELAXED = 0.85   # new, looser suggestions (always sent to the sheet)
# Share of the current box height that holds the garment, when the detector gave no usable box (OUTFIT column).
GARMENT_REGION = {"Top": (0.0, 0.55), "Bottom": (0.45, 1.0)}
FOOTWEAR_REGION = (0.65, 1.0)

RULES = [
    ("person, item box found, box < 70% of the crop", "crop to the item box", "the existing rule; already reviewed"),
    ("person, item box found, box 70-85% of the crop", "crop to the item box", "new suggestion; check on the sheet"),
    ("person, apparel top / bottom or footwear, no usable box", "crop to the garment region", "heuristic by OUTFIT; check"),
    ("person confirmed (lifestyle photo, or CLIP agrees), nothing usable", "no crop: image block and palette off", "tabular blocks only"),
    ("OWLv2 sees a person on a packshot, CLIP does not", "no action; check by eye", "OWLv2 fires on flat coats and jewellery"),
    ("lifestyle photo (not a packshot), already cropped by the detector", "keep the crop; flag", "target not found: off"),
    ("several pieces (units > 1, not a pair)", "no crop: flag", "test below decides if the image block stays"),
    ("pair of earrings (Set 1 Pair)", "keep", "normal case of the category"),
]


# --------------------------------------------------------------------------- pure rules (tested on toy data)
def units_case(value) -> str:
    """single / pair / multi_same / multi_different from NUMBER_OF_UNITS."""
    s = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value).strip()
    if s in ("", "Singular"):
        return "single"
    if s == "Set 1 Pair":
        return "pair"
    m = re.search(r"(\d+)", s)
    if m is None or int(m.group(1)) <= 1:
        return "single"
    return "multi_different" if "Different" in s else "multi_same"


def box_area(b) -> float:
    return 1.0 if b is None or len(b) != 4 else float((b[2] - b[0]) * (b[3] - b[1]))


def garment_box(box, outfit, cat):
    """Garment region inside the current box when the detector gave nothing usable; None if no rule applies."""
    base = [0.0, 0.0, 1.0, 1.0] if box is None or len(box) != 4 else [float(v) for v in box]
    if cat == "Footwear":
        lo, hi = FOOTWEAR_REGION
    elif cat == "Apparel" and outfit in GARMENT_REGION:
        lo, hi = GARMENT_REGION[outfit]
    else:
        return None
    h = base[3] - base[1]
    return [round(base[0], 4), round(base[1] + lo * h, 4), round(base[2], 4), round(base[1] + hi * h, 4)]


def decide_row(r: dict, accepted: set, rejected: set) -> tuple[str, str, list | None, str]:
    """(action, reason, suggested box, source) of one trusted photo."""
    cur = r["box"]
    cur_area = box_area(cur)
    if r["person_detected"]:
        ib, score = r.get("item_box"), r.get("item_score")
        ratio = box_area(ib) / cur_area if ib is not None and len(ib) == 4 else np.nan
        usable = ib is not None and len(ib) == 4 and pd.notna(score) and score >= ITEM_MIN
        if r[KEY] in rejected:
            return "keep_rejected", "person found but a reviewer rejected the re-crop (packshot / print / pair)", None, ""
        if usable and ratio < MAX_SHARE_STRICT:
            act = "crop_applied" if r[KEY] in accepted else "crop_suggested_reviewed_rule"
            return act, "person in the photo; item box clearly smaller than the crop", list(ib), "owlv2_item_box"
        if usable and ratio < MAX_SHARE_RELAXED:
            return "crop_suggested", "person in the photo; item box 70-85% of the crop", list(ib), "owlv2_item_box"
        # OWLv2 alone also fires on flat garments and jewellery packshots, so the heuristic crop and the exclusion need a
        # second witness: a lifestyle photo, or CLIP also seeing a person
        confirmed = (not r["packshot"]) or (pd.notna(r.get("person_full")) and r["person_full"] >= CLIP_PERSON_MIN)
        g = garment_box(cur, r.get("OUTFIT"), r["CAT_DES_EN"])
        if confirmed and g is not None and box_area(g) / cur_area < MAX_SHARE_STRICT:
            return "crop_suggested_heuristic", "person in the photo; garment region by OUTFIT (no usable box)", g, "garment_heuristic"
        if usable:
            return "keep_person_whole_item", "person found but the item fills the crop (dress / overall / worn all over)", None, ""
        if confirmed:
            return "exclude_image", "person in the photo and no usable item box", None, ""
        return "check_possible_person", "OWLv2 sees a person but CLIP and the packshot say otherwise; look before acting", None, ""
    if not r["packshot"]:
        if r["target_found"] is False:
            return "exclude_image", "lifestyle photo and the expected item was not found", None, ""
        return "keep_lifestyle", "lifestyle photo; Phase 1b already cropped it to the item", None, ""
    if r["person_possible"]:
        return "check_possible_person", "only CLIP suggests a person (noisy on jewellery); look before acting", None, ""
    if r["unit_case"] in ("multi_same", "multi_different"):
        return "flag_multi_piece", "several pieces in one photo", None, ""
    return "keep", "", None, ""


def build_suggestions(items: pd.DataFrame, masks: pd.DataFrame, person: pd.DataFrame, reviewed: dict,
                      person_reviewed: dict) -> pd.DataFrame:
    """One row per trusted photo with the person / pieces flags, the action and the suggested box."""
    it = items.set_index(KEY)
    trusted = it.index[it["img_trusted"].fillna(False).astype(bool)]
    df = pd.DataFrame(index=trusted)
    df["CAT_DES_EN"] = it.loc[trusted, "CAT_DES_EN"]
    df["GFA_DES_EN"] = it.loc[trusted, "GFA_DES_EN"]
    df["OUTFIT"] = it.loc[trusted, "OUTFIT"]
    df["NUMBER_OF_UNITS"] = it.loc[trusted, "NUMBER_OF_UNITS"]
    df["unit_case"] = df["NUMBER_OF_UNITS"].map(units_case)
    m = masks.set_index(KEY)
    df["box"] = [m.at[k, "box"] if k in m.index else None for k in df.index]
    df["packshot"] = [bool(m.at[k, "packshot"]) if k in m.index else True for k in df.index]
    df["target_found"] = [m.at[k, "target_found"] if k in m.index else None for k in df.index]
    p = person.set_index(KEY)
    for c in ["person_full", "person_score", "item_score", "item_box"]:
        df[c] = [p.at[k, c] if k in p.index else (None if c == "item_box" else np.nan) for k in df.index]
    df["person_detected"] = df["person_score"].fillna(0) >= PERSON_MIN
    df["person_possible"] = (df["person_full"].fillna(0) >= CLIP_PERSON_MIN) & ~df["person_detected"]
    accepted = set(reviewed.get("accepted", [])) | set(person_reviewed.get("accepted", []))
    rejected = set(person_reviewed.get("rejected", {})) | set(reviewed.get("rejected", {}))
    rows = []
    for k, r in df.iterrows():
        d = r.to_dict()
        d[KEY] = k
        rows.append(decide_row(d, accepted, rejected))
    df["action"], df["reason"], df["suggested_box"], df["box_source"] = zip(*rows) if rows else ([], [], [], [])
    df["has_person"] = df["person_detected"] | ~df["packshot"]
    df["several_pieces"] = df["unit_case"].isin(["multi_same", "multi_different"])
    df["exclude_image"] = df["action"].eq("exclude_image")
    df["accepted"] = df.index.isin(accepted)
    return df.reset_index().rename(columns={"index": KEY})


# --------------------------------------------------------------------------- evidence with the saved embeddings
ENCODERS = ["image_clip", "image_fashionclip_nopersoncrops", "image_dinov2_nopersoncrops", "image_fashionsiglip"]
LABELS = {"sub-family": "GFS_DES_EN", "material": "MATERIAL", "finish": "FINISHING", "colour": "CLR_DES"}


def label_agreement(items: pd.DataFrame, image: np.ndarray, queries: list[str]) -> dict[str, float]:
    """Share of each query's 4 nearest photos (image block alone) that share a label the image never saw."""
    import phase3_engine as engine
    pos = pd.Series(np.arange(len(items)), index=items[KEY])
    if not queries:
        return {f"same {n}": np.nan for n in LABELS}
    nb = engine.neighbours(items, {"image": image}, {"image": 1.0}, k=4, queries=queries)
    q, c = pos[nb["query"]].to_numpy(), pos[nb["candidate"]].to_numpy()
    out = {}
    for name, col in LABELS.items():
        v = items[col].astype("string").to_numpy(object, na_value=None)
        both = pd.notna(v[q]) & pd.notna(v[c])
        out[f"same {name}"] = float((v[q][both] == v[c][both]).mean()) if both.sum() >= 50 else np.nan   # <50 compared pairs: blank
    return out


def team_pairs(items: pd.DataFrame, images: dict[str, np.ndarray], weights: dict, groups: dict[str, set],
               drop_image_for: set | None = None) -> pd.DataFrame:
    """Team preference accuracy on the pairs that involve a photo of each group (query or either candidate).

    `drop_image_for`: simulate the recommendation (image block off whenever the query or the candidate is in the set)."""
    import phase4_feedback as fb
    pool, recs = fb.load_pool(), fb.read_feedback()
    index, s0, h0 = fb.sim_matrix(pool)
    blocks = pool["blocks"]
    j = blocks.index("image")
    pos = pd.Series(np.arange(len(items)), index=items[KEY])
    keys = sorted(index.items(), key=lambda kv: kv[1])
    qk = [q for (q, _), _ in keys]
    ck = [c for (_, c), _ in keys]
    qi, ci = pos[qk].to_numpy(), pos[ck].to_numpy()
    win, lose = fb.preference_pairs(recs, index)
    w = np.array([weights[b] for b in blocks])
    inv = {name: np.array([(qk[i] in g) or (ck[i] in g) for i in range(len(qk))]) for name, g in groups.items()}
    rows = []
    for stem, img in images.items():
        s, h = s0.copy(), h0.copy()
        s[:, j] = np.where(h[:, j] > 0, np.einsum("ij,ij->i", img[qi], img[ci]), 0)
        if drop_image_for:
            off = np.array([(qk[i] in drop_image_for) or (ck[i] in drop_image_for) for i in range(len(qk))])
            h[off, j] = 0
        sc = fb.scores(w, s, h)
        ok = sc[win] > sc[lose]
        for name, flag in inv.items():
            sel = flag[win] | flag[lose]
            rows.append({"encoder": stem, "group": name, "pairs": int(sel.sum()),
                         "queries": len({qk[i] for i in np.unique(np.r_[win[sel], lose[sel]])}),
                         "accuracy": float(ok[sel].mean()) if sel.any() else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- contact sheet
def square_crop(img: Image.Image, box, size: int = 170, pad: float = 0.04) -> Image.Image:
    """Same crop as phase1b_image_audit.square_crop, without importing its heavy dependencies."""
    from PIL import ImageOps
    img = img.convert("RGB")
    if box is not None and len(box) == 4:
        w, h = img.size
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        img = img.crop((max(0, (x0 - px) * w), max(0, (y0 - py) * h), min(w, (x1 + px) * w), min(h, (y1 + py) * h)))
    side = max(img.size)
    return ImageOps.pad(img, (side, side), color=(255, 255, 255)).resize((size, size))


def contact_sheet(df: pd.DataFrame, files: dict[str, str], path: Path, max_pairs: int = 48, flagged_samples: int = 8) -> int:
    """Before | after for every suggested crop, then a few 'flag only' photos (before only)."""
    sug = df[df["suggested_box"].notna() & df["action"].str.startswith("crop_suggested")].copy()
    order = {"crop_suggested": 0, "crop_suggested_heuristic": 1, "crop_suggested_reviewed_rule": 2}
    sug = sug.sort_values(by="action", key=lambda s: s.map(order)).head(max_pairs)
    multi = df[df["action"].isin(["flag_multi_piece", "exclude_image"])].sort_values(KEY).head(flagged_samples)
    size, cols, label_h = 170, 3, 34
    cells = [(r, True) for _, r in sug.iterrows()] + [(r, False) for _, r in multi.iterrows()]
    cw = size * 2 + 12
    rows_n = max(1, -(-len(cells) // cols))
    sheet = Image.new("RGB", (cols * cw, rows_n * (size + label_h)), "white")
    d = ImageDraw.Draw(sheet)
    for n, (r, pair) in enumerate(cells):
        x, y = (n % cols) * cw, (n // cols) * (size + label_h)
        try:
            img = Image.open(IMG_DIR / files[r[KEY]])
        except (OSError, KeyError):
            continue
        sheet.paste(square_crop(img, r["box"], size), (x, y + label_h))
        if pair:
            sheet.paste(square_crop(img, r["suggested_box"], size), (x + size + 6, y + label_h))
        d.text((x + 2, y + 2), f"{r[KEY]} {r['CAT_DES_EN']}", fill=(0, 0, 0))
        d.text((x + 2, y + 16), (r["action"] if pair else f"FLAG: {r['action']} / {r['NUMBER_OF_UNITS']}")[:44], fill=(200, 0, 0))
    sheet.save(path, quality=88)
    return len(cells)


# --------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame, floatfmt: str = "{:.1%}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append("" if pd.isna(v) else floatfmt.format(v) if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def counts_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("CAT_DES_EN")
    t = pd.DataFrame({
        "trusted photos": g.size(),
        "person (OWLv2)": g["person_detected"].sum(),
        "not a packshot": g["packshot"].apply(lambda s: int((~s).sum())),
        "person or lifestyle": g["has_person"].sum(),
        "CLIP-only possible person": g["person_possible"].sum(),
        "several pieces": g["several_pieces"].sum(),
        "pair (Set 1 Pair)": g["unit_case"].apply(lambda s: int((s == "pair").sum())),
        "OUTFIT set (top/bottom/dress/overall)": g["OUTFIT"].apply(lambda s: int(s.isin(["Top", "Bottom", "Dress", "Overall"]).sum())),
    })
    t.loc["All"] = t.sum()
    return t.reset_index().rename(columns={"CAT_DES_EN": "category"}).astype({c: int for c in t.columns})


def write_report(df, evidence, team, team_drop, accepted_eval, meta, n_sheet) -> None:
    n = len(df)
    act = df["action"].value_counts().rename_axis("action").reset_index(name="photos")
    act_cat = pd.crosstab(df["CAT_DES_EN"], df["action"])
    act_cat = act_cat.loc[:, [c for c in act_cat.columns if c != "keep"]].reset_index().rename(columns={"CAT_DES_EN": "category"})
    pal = meta["palette"]
    L = [
        "# Phase 2b: photos with a model or several pieces",
        "",
        f"Generated by `src/phase2b_multi_item.py run` ({meta['date']}). Light environment, no model run: the counts use the "
        "CSV columns, `masks.parquet` and the saved `person_crops.parquet`; the tests use the saved embeddings.",
        "",
        "## 1. How big is the problem",
        "",
        f"{n:,} trusted photos. `person (OWLv2)` = the Phase 2b detector found a person (only the 2,454 photos that CLIP "
        "flagged were checked). `not a packshot` = lifestyle photo, already cropped to the item by the Phase 1b detector. "
        "`CLIP-only possible person` is noisy (it fires on jewellery packshots, `a hand wearing jewellery`) and is only "
        "listed, never acted on. `OUTFIT` is the garment part (top, bottom, dress), not a set of pieces, so it is used only "
        "to place the garment on a model.",
        "",
        md_table(counts_table(df)),
        "",
        f"Photos with a person or lifestyle photos: **{int(df['has_person'].sum()):,}** ({df['has_person'].mean():.1%}). "
        f"Photos with several pieces: **{int(df['several_pieces'].sum()):,}** "
        f"({int((df['unit_case'] == 'multi_different').sum())} different pieces, "
        f"{int((df['unit_case'] == 'multi_same').sum())} identical units). "
        f"Pairs (normal for earrings): {int((df['unit_case'] == 'pair').sum()):,}. `masks.parquet` keeps ONE box per photo, "
        "so several boxes per photo cannot be counted from it; the pieces come from `NUMBER_OF_UNITS`.",
        "",
        "## 2. Crop rule per case",
        "",
        "| case | action | note |",
        "|---|---|---|",
        *[f"| {a} | {b} | {c} |" for a, b, c in RULES],
        "",
        "Applied to the trusted photos:",
        "",
        md_table(act),
        "",
        md_table(act_cat),
        "",
        f"Suggested boxes: `data/embeddings/multi_item_crops.parquet` (one row per photo with an action other than `keep`, "
        f"with `reason`, `box`, `suggested_box`, `box_source`). Before/after sheet: `outputs/phase2b/multi_item_crops_sheet.jpg` "
        f"({n_sheet} cells; a pair per suggested crop, then a few flag-only photos). The suggested boxes are NOT applied "
        "until a person writes the key in `data/embeddings/multi_item_reviewed.json` (`accepted`; `rejected` with a reason). "
        "The template is empty and is not filled by the script.",
        "",
        "## 3. Effect with the saved embeddings",
        "",
        "Only the 6 accepted photos were embedded with a person re-crop, and no FashionSigLIP vector without crops is saved, "
        "so the effect of the crop itself cannot be isolated here: the new crops need the vision machine (section 5). What the "
        "saved files do show is how well each encoder handles the affected groups.",
        "",
        "### 3a. Photo alone: do the 4 nearest photos share a label the image never saw?",
        "",
        "Queries are the photos of the group (all trusted photos of the group, not only the test models); the candidates are "
        "like-for-like. `image_clip` is the Sprint 2 baseline; `image_fashionsiglip` is the current encoder (6 person re-crops).",
        "",
        md_table(evidence),
        "",
        "### 3b. Team preferences on the pairs that involve a photo of the group",
        "",
        "Same fixed weights for every encoder (`phase3_weights.json`); only the image block changes. Accuracy = share of the "
        "team's preference pairs the engine orders the same way. Small groups: read the `pairs` column before the percentage.",
        "",
        md_table(team),
        "",
        "### 3c. What if the recommendation is applied (image block off for the excluded photos)?",
        "",
        "Image block switched off for every pair whose query or candidate is in the `exclude` set below (the engine already "
        "averages over the blocks both items have), current encoder.",
        "",
        md_table(team_drop),
        "",
        accepted_eval,
        "",
        "## 4. Recommendation",
        "",
        meta["recommendation"],
        "",
        "Palette: " + f"{pal['n_unreliable']} trusted photos already have `reliable = False`; "
        f"of the {pal['n_person']} photos with a person or lifestyle crop, {pal['person_unreliable']} are unreliable already, "
        f"and of the {pal['n_excl']} photos in the exclude set, {pal['excl_unreliable']} are. "
        "The palette is measured on the product pixels of the same box, so a model's skin and clothes leak in. The rest of "
        "the exclude set (reliable = True today) should leave the palette block too (the engine then uses the other blocks). "
        "Several pieces: the palette of a set describes all its pieces, which is also what a buyer sees, so it stays.",
        "",
        "## 5. What to run on the vision machine",
        "",
        "1. A person opens `outputs/phase2b/multi_item_crops_sheet.jpg`, and writes the keys to keep in "
        "`data/embeddings/multi_item_reviewed.json` (`accepted`), the others in `rejected` with a reason.",
        "2. Optional, only to widen the search: OWLv2 on the `check_possible_person` photos (needs the vision environment; "
        "the whole `persons` run took 16.6 h, `wall_seconds` in `person_crops.json`, and it rewrites `person_crops.parquet`, so do "
        "not rerun it as it is; a `--keys` option on the flagged photos only would be a small extension).",
        "3. Embed the accepted crops. `phase2b_image_encoders.py embed` reads `person_crops.parquet`, not "
        "`multi_item_crops.parquet`, and always writes `image_<model>.npy`, so a small extension is needed first (not done here, "
        "existing scripts are not edited in this task): a `--extra-crops data/embeddings/multi_item_crops.parquet` option that "
        "applies `suggested_box` for the keys in `multi_item_reviewed.json` (`accepted`), and a `--suffix _multicrops` option "
        "so the output is `image_fashionsiglip_multicrops.npy` and never replaces `image_fashionsiglip.npy`. Then:",
        "",
        "```",
        "pip install -r requirements-vision.txt open_clip_torch",
        "# control: Phase 1b crops only. Safe as it is: writes a NEW file image_fashionsiglip_nopersoncrops.npy",
        "python src/phase2b_image_encoders.py embed --model fashionsiglip --no-person-crops",
        "# treatment: person crops + the accepted multi-item crops (needs the extension above)",
        "python src/phase2b_image_encoders.py embed --model fashionsiglip --extra-crops data/embeddings/multi_item_crops.parquet --suffix _multicrops",
        "# light environment again: add the new stems to ENCODERS in this script and to the call below",
        "python src/phase4_compare.py encoders image_fashionsiglip image_fashionsiglip_nopersoncrops image_fashionsiglip_multicrops",
        "python src/phase2b_multi_item.py run",
        "python src/phase2b_palettes.py        # palette on the same crops (light environment; check its output path first)",
        "```",
        "",
        "Compare the three FashionSigLIP files on the affected groups (section 3) and on the global team metric; adopt the "
        "crops only if the affected groups improve and the guards in `compare_report.md` hold. The control matters: it is the "
        "only way to isolate the effect of the person crops, which the saved files cannot show.",
        "",
    ]
    REPORT_FILE.write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- run
def run() -> None:
    import phase1_clean
    import phase3_engine as engine
    items = phase1_clean.load_items()
    masks = pd.read_parquet(EMB_DIR / "masks.parquet")
    person = pd.read_parquet(EMB_DIR / "person_crops.parquet")
    reviewed = json.loads(REVIEWED_FILE.read_text(encoding="utf-8")) if REVIEWED_FILE.exists() else {"accepted": [], "rejected": {}}
    if not REVIEWED_FILE.exists():
        REVIEWED_FILE.write_text(json.dumps({
            "note": "Human review of the suggested crops in multi_item_crops.parquet, checked on "
                    "outputs/phase2b/multi_item_crops_sheet.jpg. Edit by hand only; never filled or regenerated by the script.",
            "accepted": [], "rejected": {}}, indent=1), encoding="utf-8")
    pr = json.loads(PERSON_REVIEWED_FILE.read_text(encoding="utf-8"))
    df = build_suggestions(items, masks, person, reviewed, pr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    out = df[df["action"] != "keep"].copy()
    out["box"] = out["box"].map(lambda b: None if b is None else [float(v) for v in b])
    out["status"] = "suggestion"
    out[[KEY, "CAT_DES_EN", "GFA_DES_EN", "NUMBER_OF_UNITS", "unit_case", "has_person", "person_detected", "action", "reason",
         "box", "suggested_box", "box_source", "exclude_image", "accepted", "status"]].to_parquet(CROPS_FILE, index=False)
    files = items.set_index(KEY)["img_file"].to_dict()
    n_sheet = contact_sheet(df, files, SHEET_FILE)

    # evidence with the saved embeddings
    weights = engine.load_weights()
    trusted = set(df[KEY])
    groups = {
        "packshot, single piece (reference)": set(df.loc[df["packshot"] & ~df["has_person"] & (df["unit_case"] == "single"), KEY]),
        "pair of earrings": set(df.loc[df["unit_case"] == "pair", KEY]),
        "several pieces, identical": set(df.loc[df["unit_case"] == "multi_same", KEY]),
        "several pieces, different": set(df.loc[df["unit_case"] == "multi_different", KEY]),
        "person (OWLv2)": set(df.loc[df["person_detected"], KEY]),
        "lifestyle (not a packshot)": set(df.loc[~df["packshot"], KEY]),
        "person, accepted re-crop (5 trusted of 6)": set(df.loc[df["action"] == "crop_applied", KEY]),
    }
    images = {s: engine.load_blocks(items, image=s)["image"] for s in ENCODERS}
    rng = np.random.default_rng(0)
    ev_rows = []
    for gname, g in groups.items():
        qs = sorted(g & trusted)
        if gname.startswith("packshot, single") and len(qs) > 1500:     # the reference only needs a sample
            qs = sorted(rng.choice(qs, 1500, replace=False).tolist())
        for stem, img in images.items():
            ev_rows.append({"group": gname, "n photos": len(qs), "encoder": stem.replace("_nopersoncrops", ""),
                            **label_agreement(items, img, qs)})
    evidence = pd.DataFrame(ev_rows)
    team = team_pairs(items, images, weights, groups)
    team["encoder"] = team["encoder"].str.replace("_nopersoncrops", "")
    team = team.rename(columns={"accuracy": "team accuracy"})

    excl = set(df.loc[df["exclude_image"], KEY])
    cur = {"image_fashionsiglip": images["image_fashionsiglip"]}
    keep = team_pairs(items, cur, weights, {"exclude set": excl, "several pieces, different": groups["several pieces, different"],
                                            "several pieces, identical": groups["several pieces, identical"]})
    off_excl = team_pairs(items, cur, weights, {"exclude set": excl}, drop_image_for=excl)
    off_multi = team_pairs(items, cur, weights, {"several pieces, different": groups["several pieces, different"],
                                                 "several pieces, identical": groups["several pieces, identical"]},
                           drop_image_for=set(df.loc[df["several_pieces"], KEY]))
    keep["image block"] = "on"
    off = pd.concat([off_excl, off_multi])
    off["image block"] = "off for the group"
    team_drop = pd.concat([keep, off]).drop(columns="encoder").rename(columns={"accuracy": "team accuracy"})
    team_drop = team_drop.sort_values(["group", "image block"], ascending=[True, False])

    # the 6 accepted re-crops: current encoder vs the others (no crop-free FashionSigLIP exists)
    acc = sorted(groups["person, accepted re-crop (5 trusted of 6)"] & trusted)
    accepted_eval = (f"The {len(acc)} accepted person re-crops ({', '.join(acc)}; the sixth, 211451_BM, has an untrusted photo) "
                     "are embedded with the crop in "
                     "`image_fashionsiglip` only; every other saved encoder saw the whole person. Their rows in 3a/3b "
                     "are the only direct hint, with far too few photos to conclude anything.")

    pal = pd.read_parquet(EMB_DIR / "palettes.parquet").set_index(KEY)
    hp = df.loc[df["has_person"], KEY]
    unrel = ~pal["reliable"].astype(bool)
    meta = {"date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "palette": {"n_unreliable": int(unrel.sum()), "n_person": len(hp),
                        "person_unreliable": int(unrel.reindex(hp).fillna(True).sum()),
                        "n_excl": len(excl), "excl_unreliable": int(unrel.reindex(sorted(excl)).fillna(True).sum())},
            "recommendation": recommend(df, evidence, team, team_drop)}
    write_report(df, evidence, team, team_drop, accepted_eval, meta, n_sheet)
    CROPS_FILE.with_suffix(".json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rows": len(out),
        "thresholds": {"person_min": PERSON_MIN, "item_min": ITEM_MIN, "max_share_strict": MAX_SHARE_STRICT,
                       "max_share_relaxed": MAX_SHARE_RELAXED},
        "sources": ["masks.parquet", "person_crops.parquet", "person_crops_reviewed.json", "NUMBER_OF_UNITS", "OUTFIT"],
        "note": "Suggestions only. Applied only for keys in multi_item_reviewed.json (accepted).",
        "actions": df["action"].value_counts().to_dict()}, indent=1), encoding="utf-8")
    print(df["action"].value_counts().to_string())
    print(f"-> {CROPS_FILE}\n-> {SHEET_FILE}\n-> {REPORT_FILE}")


def recommend(df: pd.DataFrame, evidence: pd.DataFrame, team: pd.DataFrame, team_drop: pd.DataFrame) -> str:
    """Recommendation text written from the numbers above (kept in code so it follows a rerun)."""
    c = df["action"].value_counts()
    sib = evidence[evidence["encoder"] == "image_fashionsiglip"].set_index("group")["same sub-family"]
    ref = sib.get("packshot, single piece (reference)", np.nan)
    lines = [
        "| photos | count | what to do |", "|---|---|---|",
        f"| re-crop (already accepted) | {int(c.get('crop_applied', 0)) + int(c.get('crop_suggested_reviewed_rule', 0))} "
        "| keep the accepted crops of `person_crops_reviewed.json` (6, one on an untrusted photo) |",
        f"| re-crop (new, to be checked on the sheet) | {int(c.get('crop_suggested', 0)) + int(c.get('crop_suggested_heuristic', 0))} "
        "| accept or reject by eye in `multi_item_reviewed.json`, then embed on the vision machine |",
        f"| exclude from the image block and the palette | {int(c.get('exclude_image', 0))} "
        "| fall back to the tabular blocks; keep the row and the `exclude_image` flag (hard rule 3) |",
        f"| flag only: several pieces | {int(c.get('flag_multi_piece', 0))} | keep the image block (see the test below) |",
        f"| check by eye: possible person (one witness only) | {int(c.get('check_possible_person', 0))} | no action until looked at |",
        f"| leave alone | {int(c.get('keep', 0)) + int(c.get('keep_lifestyle', 0)) + int(c.get('keep_rejected', 0)) + int(c.get('keep_person_whole_item', 0))} "
        "| packshots, pairs, lifestyle photos already cropped, rejected re-crops, worn-all-over garments |",
        "",
    ]
    for g in ["several pieces, different", "several pieces, identical", "person (OWLv2)", "lifestyle (not a packshot)"]:
        if g in sib and pd.notna(ref):
            lines.append(f"- {g}: nearest photos share the sub-family {sib[g]:.1%} of the time, against {ref:.1%} for single-piece packshots "
                         f"({'lower by ' + format(ref - sib[g], '.1%') if sib[g] < ref else 'not lower'}).")
    td = team_drop.set_index(["group", "image block"])["team accuracy"]
    for g in ["several pieces, different", "several pieces, identical", "exclude set"]:
        if (g, "on") in td.index and (g, "off for the group") in td.index:
            on, off = td[(g, "on")], td[(g, "off for the group")]
            lines.append(f"- {g}: team accuracy {on:.1%} with the image block, {off:.1%} without it "
                         f"({'the image helps: keep it' if on > off else 'no loss without it'}; small samples, see the pairs column).")
    lines += ["", "Summary: re-crop only what a person accepts on the sheet; switch the image block and the palette off only for "
              "the exclude set (a confirmed person and no usable item box); for several pieces, flag and keep the image "
              "block, because the photo of the set is what the buyer compares and the encoders match sets of different pieces as well as single pieces (identical units a little worse, 125 photos)."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_subparsers(dest="cmd", required=True).add_parser("run")
    ap.parse_args()
    run()


if __name__ == "__main__":
    main()
