"""Phase 1b: item masks, image embeddings and image <-> label checks.

Heavy step: run it on ONE machine only (needs requirements-vision.txt). The results are
committed to data/embeddings/ so teammates only load them (see ROADMAP "Compute strategy").

    python src/phase1b_image_audit.py                 # full run (~20-40 min on a laptop CPU)
    python src/phase1b_image_audit.py --limit 300     # quick trial on a sample

Steps:
1. Mask: white-background packshots -> crop to the non-white pixels (threshold).
   Other photos (model / lifestyle / amateur) -> text-prompted detector (OWLv2) with the
   CSV item type as prompt ("earrings"), crop to the best box.
2. Embed every masked crop once with CLIP (ViT-B/32) -> data/embeddings/image_clip.npy.
3. Checks on the embeddings:
   - type: zero-shot CLIP over 15 broad item types vs the CSV family;
   - neighbours: do the item's nearest images belong to the same broad type?
   - colour: zero-shot CLIP over basic colours vs CLR_DES;
   - near-duplicate photos across colours/models (perceptual hash);
   - not a packshot (model / lifestyle / amateur photo).
4. Write data/embeddings/{masks,image_audit}.parquet and outputs/phase1b/ report + contact sheets.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import EMB_DIR, IMG_DIR, KEY, ROOT  # noqa: E402
from phase1_checks import run_checks  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase1b"
CLIP_MODEL = "openai/clip-vit-base-patch32"
DETECTOR_MODEL = "google/owlv2-base-patch16-ensemble"

# ----------------------------------------------------------------------------- vocabularies
# Broad item type per family (GFA_DES_EN). Families not listed are too vague to check.
GROUPS = {
    "earrings": ["Earrings"],
    "necklace": ["Necklaces"],
    "ring": ["Rings"],
    "bracelet": ["Bracelets"],
    "keychain": ["Key chains"],
    "phone case": ["Mobile Accessories", "Phone Holder"],
    "bag": ["Backpack", "Belt Bag", "Briefcase", "Bucket", "Computer Backpack", "Computer Hand Bag", "Cross",
            "Envelope", "Gym Bag", "Hand", "Lunch Bag", "Mini Bag", "Sac", "Shopper", "Shoulder Bag", "Tote"],
    "wallet": ["Wallet", "Card Holder", "Coin Purse", "Multipurpose Purse", "Pencil Case"],
    "shoes": ["Ankle Boots", "Ballerinas", "Boots", "Flat Sandals", "Flat Shoes", "High Heel Sandals",
              "High Heel Shoes", "Trainers"],
    "top": ["Shirt", "Blouse", "T-shirt", "Top", "Sweater", "Sweatshirt", "Cardigan", "Vest", "Bodys"],
    "coat": ["Coat", "Blazer", "Raincoat", "Cape", "Poncho", "Kimono", "Kaftan"],
    "dress": ["Dress", "Jumpsuit"],
    "trousers": ["Trousers", "Jeans", "Shorts"],
    "skirt": ["Skirt"],
    "swimwear": ["Bath Suit", "Bikini"],
}
FAMILY_TO_GROUP = {f: g for g, fams in GROUPS.items() for f in fams}
GROUP_PHRASE = {
    "earrings": "a pair of earrings", "necklace": "a necklace", "ring": "a ring", "bracelet": "a bracelet",
    "keychain": "a keychain", "phone case": "a phone case", "bag": "a handbag", "wallet": "a wallet",
    "shoes": "a pair of shoes", "top": "a top", "coat": "a coat", "dress": "a dress",
    "trousers": "a pair of trousers", "skirt": "a skirt", "swimwear": "a swimsuit",
}
DETECT_QUERY = {"earrings": "earring", "necklace": "necklace", "ring": "ring", "bracelet": "bracelet",
                "keychain": "keychain", "phone case": "phone case", "bag": "bag", "wallet": "wallet",
                "shoes": "shoe", "top": "top", "coat": "coat", "dress": "dress", "trousers": "trousers",
                "skirt": "skirt", "swimwear": "swimsuit"}
# Pairs CLIP can't reliably tell apart (or that genuinely overlap): never flagged as a type mismatch.
COMPATIBLE = {frozenset(p) for p in [("bag", "wallet"), ("phone case", "bag"), ("top", "coat"), ("top", "dress"), ("dress", "skirt"),
                                     ("dress", "coat"), ("phone case", "wallet"), ("keychain", "bracelet"),
                                     ("keychain", "necklace"), ("top", "swimwear"), ("dress", "swimwear"),
                                     ("trousers", "skirt")]}
TEMPLATES = ["a photo of {}.", "a product photo of {}.", "a close-up photo of {}."]

# Basic colour per CLR_DES. Multicolours and vague names are not checked.
COLOURS = {
    "black": ["Black", "Black Nickel"],
    "white": ["White", "Snow", "Ice", "Off White", "Ivory", "Ecru", "Pearl"],
    "beige": ["Beige", "Nude", "Skin", "Natural Color", "Dark Natural Color", "Straw", "Champagne", "Taupe"],
    "brown": ["Brown", "Dark Brown", "Chocolate", "Coffee", "Cognac", "Camel", "Bronze", "Copper"],
    "grey": ["Grey", "Dark Grey", "Light Grey", "Pastel Grey", "Anthracite"],
    "blue": ["Blue", "Bright Blue", "Light Blue", "Pastel Blue", "Blue Jeans", "Indigo", "Indigo Dye", "Navy",
             "Midnight Blue", "Oxford Blue"],
    "green": ["Green", "Dark Green", "Forest Green", "Light Green", "Pastel Green", "Mint"],
    "khaki": ["Khaki", "Olive"],
    "turquoise": ["Turquoise", "Aquamarine", "Teal"],
    "red": ["Red", "Brick Red", "Cherry", "Burgundy", "Wine", "Maroon", "Pomegranate"],
    "pink": ["Pink", "Light Pink", "Dark Pink", "Pastel Pink", "Old Rose", "Fuchsia", "Magenta", "Raspberry"],
    "purple": ["Purple", "Lilac", "Violet", "Amethyst", "Aubergine", "Mauve"],
    "yellow": ["Yellow", "Light Yellow", "Dark Yellow", "Mustard", "Lime"],
    "orange": ["Orange", "Pastel Orange", "Papaya", "Coral", "Peach", "Paprika", "Salmon"],
    "gold": ["Gold", "Light Gold", "Old Gold", "Rose Gold"],
    "silver": ["Silver", "Old Silver", "Nickel"],
}
NAME_TO_COLOUR = {n: c for c, names in COLOURS.items() for n in names}
COLOUR_NEIGHBOURS = {frozenset(p) for p in [
    ("white", "beige"), ("white", "grey"), ("white", "silver"), ("beige", "brown"), ("beige", "gold"),
    ("beige", "pink"), ("beige", "yellow"), ("beige", "orange"), ("brown", "red"), ("brown", "orange"),
    ("brown", "gold"), ("red", "pink"), ("red", "orange"), ("pink", "purple"), ("pink", "orange"),
    ("blue", "turquoise"), ("green", "turquoise"), ("green", "yellow"), ("green", "grey"), ("yellow", "orange"),
    ("yellow", "gold"), ("grey", "black"), ("grey", "silver"), ("grey", "blue"), ("blue", "black"),
    ("gold", "orange"), ("silver", "white"), ("gold", "white"), ("khaki", "green"), ("khaki", "beige"),
    ("khaki", "brown"), ("khaki", "grey")]}


# ----------------------------------------------------------------------------- masking
def white_border_share(img: Image.Image) -> float:
    a = np.asarray(img.convert("RGB").resize((64, 64)), dtype=np.int16)
    border = np.concatenate([a[:3].reshape(-1, 3), a[-3:].reshape(-1, 3), a[:, :3].reshape(-1, 3), a[:, -3:].reshape(-1, 3)])
    return float((border.min(1) > 225).mean())


def threshold_box(img: Image.Image, thr: int = 235) -> tuple[float, float, float, float] | None:
    """Bounding box (relative) of the non-white pixels of a packshot."""
    a = np.asarray(img.convert("RGB"), dtype=np.int16)
    mask = a.min(2) < thr
    if mask.mean() < 0.002:
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape
    return xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h


def square_crop(img: Image.Image, box, pad: float = 0.04) -> Image.Image:
    """Crop to a relative box (with padding) and pad to a white square."""
    img = img.convert("RGB")
    if box is not None:
        w, h = img.size
        x0, y0, x1, y1 = box
        px, py = (x1 - x0) * pad, (y1 - y0) * pad
        img = img.crop((max(0, (x0 - px) * w), max(0, (y0 - py) * h), min(w, (x1 + px) * w), min(h, (y1 + py) * h)))
    side = max(img.size)
    return ImageOps.pad(img, (side, side), color=(255, 255, 255))


class Detector:
    """OWLv2 text-prompted detector, loaded lazily (only needed for non-packshots)."""

    def __init__(self):
        from transformers import Owlv2ForObjectDetection, Owlv2Processor
        import torch
        self.torch = torch
        self.proc = Owlv2Processor.from_pretrained(DETECTOR_MODEL)
        self.model = Owlv2ForObjectDetection.from_pretrained(DETECTOR_MODEL).eval()
        self.queries = list(DETECT_QUERY.values())

    def detect(self, img: Image.Image, group: str | None):
        """Best box for the expected type, plus the type with the strongest detection overall."""
        img = img.convert("RGB")
        inputs = self.proc(text=[[f"a photo of a {q}" for q in self.queries]], images=img, return_tensors="pt")
        with self.torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[0].sigmoid()            # (n_boxes, n_queries)
        boxes = out.pred_boxes[0]                   # (cx, cy, w, h) relative to the padded square
        side = max(img.size)
        sx, sy = side / img.size[0], side / img.size[1]

        def to_rel(b):
            cx, cy, w, h = b.tolist()
            return (max(0, (cx - w / 2) * sx), max(0, (cy - h / 2) * sy), min(1, (cx + w / 2) * sx), min(1, (cy + h / 2) * sy))

        best_q = int(logits.max(0).values.argmax())
        strongest = (list(DETECT_QUERY)[best_q], float(logits[:, best_q].max()))
        if group is None:
            return None, 0.0, strongest
        qi = list(DETECT_QUERY).index(group)
        bi = int(logits[:, qi].argmax())
        return to_rel(boxes[bi]), float(logits[bi, qi]), strongest


# ----------------------------------------------------------------------------- embeddings
class Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
        self.model = CLIPModel.from_pretrained(CLIP_MODEL).eval()

    @staticmethod
    def _as_tensor(out):
        return out if hasattr(out, "norm") else out.pooler_output  # transformers 4.x vs 5.x return types

    def images(self, imgs: list[Image.Image]) -> np.ndarray:
        with self.torch.no_grad():
            f = self._as_tensor(self.model.get_image_features(**self.proc(images=imgs, return_tensors="pt")))
        return (f / f.norm(dim=-1, keepdim=True)).numpy()

    def texts(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            f = self._as_tensor(self.model.get_text_features(**self.proc(text=texts, return_tensors="pt", padding=True)))
        return (f / f.norm(dim=-1, keepdim=True)).numpy()

    def prompt_bank(self, phrases: list[str]) -> np.ndarray:
        """One averaged, normalised text vector per phrase over the TEMPLATES."""
        vecs = np.stack([self.texts([t.format(p) for t in TEMPLATES]).mean(0) for p in phrases])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def softmax(x: np.ndarray, scale: float = 100.0) -> np.ndarray:
    z = x * scale
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def dhash(img: Image.Image) -> np.uint64:
    g = np.asarray(img.convert("L").resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


# ----------------------------------------------------------------------------- main
def main() -> None:
    import os
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="only process a random sample of N items (trial run)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--reflag", action="store_true", help="only recompute flags/report from stored scores")
    args = ap.parse_args()
    if args.reflag:
        return reflag()
    t0 = time.time()

    items, _ = run_checks(verify_files=False)
    items = items[items["has_image"]].reset_index(drop=True)

    def readable(f: str) -> bool:  # damaged files (already flagged IMG_UNREADABLE) are skipped
        try:
            with Image.open(IMG_DIR / f) as im:
                im.verify()
            return True
        except Exception:
            return False

    ok = items["img_file"].map(readable)
    if (~ok).any():
        print(f"Skipping {int((~ok).sum())} unreadable image(s): {', '.join(items.loc[~ok, 'img_file'])}", flush=True)
    items = items[ok].reset_index(drop=True)
    if args.limit:
        items = items.sample(args.limit, random_state=0).reset_index(drop=True)
    items["group_expected"] = items["GFA_DES_EN"].map(FAMILY_TO_GROUP)
    items["colour_expected"] = items["CLR_DES"].map(NAME_TO_COLOUR)
    print(f"{len(items)} items with an image. Loading CLIP...", flush=True)

    clip = Clip()
    groups = list(GROUPS)
    group_vecs = clip.prompt_bank([GROUP_PHRASE[g] for g in groups])

    # 1-2. masks + embeddings (streamed in batches to keep memory low)
    detector = None
    mask_rows, embs, hashes = [], [], []
    batch_imgs: list[Image.Image] = []
    for i, r in enumerate(items.itertuples(index=False)):
        img = Image.open(IMG_DIR / r.img_file).convert("RGB")
        hashes.append(dhash(img))
        wb = white_border_share(img)
        row = {KEY: r.PROD_CLR_EQUIV, "img_file": r.img_file, "white_border": round(wb, 3), "packshot": wb > 0.9,
               "mask_method": "threshold", "box": None, "target_found": True, "det_score": np.nan,
               "det_strongest": None, "det_strongest_score": np.nan}
        if wb > 0.9:
            row["box"] = threshold_box(img)
        else:
            if detector is None:
                print("Loading detector for non-packshot photos...", flush=True)
                detector = Detector()
            box, score, strongest = detector.detect(img, r.group_expected if isinstance(r.group_expected, str) else None)
            row.update(mask_method="detector", det_score=round(score, 3), det_strongest=strongest[0],
                       det_strongest_score=round(strongest[1], 3))
            found = box is not None and score >= 0.15
            row["target_found"] = found if isinstance(r.group_expected, str) else None
            row["box"] = box if found else None
        mask_rows.append(row)
        batch_imgs.append(square_crop(img, row["box"]))
        if len(batch_imgs) == args.batch_size or i == len(items) - 1:
            embs.append(clip.images(batch_imgs))
            batch_imgs = []
            print(f"  {i + 1}/{len(items)} images  ({time.time() - t0:.0f}s)", flush=True)

    emb = np.concatenate(embs).astype(np.float32)
    masks = pd.DataFrame(mask_rows)
    masks["box"] = masks["box"].map(lambda b: None if b is None else [round(float(v), 4) for v in b])

    # 3a. type: zero-shot over broad item types
    gi = {g: k for k, g in enumerate(groups)}
    p_group = softmax(emb @ group_vecs.T)

    # Refine: a packshot whose type disagrees may be a model on a white background (e.g. a bag
    # carried by a person). Detect the expected item, crop to it and embed again.
    exp_g = items["group_expected"].to_numpy()
    refine = [n for n in range(len(items)) if masks.at[n, "packshot"] and isinstance(exp_g[n], str)
              and groups[p_group[n].argmax()] != exp_g[n]
              and frozenset((exp_g[n], groups[p_group[n].argmax()])) not in COMPATIBLE
              and p_group[n, gi[exp_g[n]]] < 0.10]
    print(f"Refining {len(refine)} packshots where the type disagrees...", flush=True)
    boxes = masks["box"].tolist()  # list cells are awkward to assign with .at
    if refine and detector is None:
        detector = Detector()
    for j, n in enumerate(refine):
        img = Image.open(IMG_DIR / items.at[n, "img_file"]).convert("RGB")
        box, score, strongest = detector.detect(img, exp_g[n])
        masks.at[n, "det_score"], masks.at[n, "det_strongest"] = round(score, 3), strongest[0]
        masks.at[n, "det_strongest_score"] = round(strongest[1], 3)
        if box is not None and score >= 0.15 and (box[2] - box[0]) * (box[3] - box[1]) < 0.7:
            masks.at[n, "mask_method"] = "detector_refine"
            boxes[n] = [round(float(v), 4) for v in box]
            emb[n] = clip.images([square_crop(img, box)])[0]
        if (j + 1) % 25 == 0:
            print(f"  refined {j + 1}/{len(refine)}  ({time.time() - t0:.0f}s)", flush=True)
    masks["box"] = boxes
    p_group = softmax(emb @ group_vecs.T)
    audit = items[[KEY, "PROD_REF", "CAT_DES_EN", "GFA_DES_EN", "CLR_DES", "group_expected", "colour_expected"]].copy()
    audit["group_pred"] = [groups[k] for k in p_group.argmax(1)]
    audit["p_group_pred"] = p_group.max(1).round(3)
    audit["p_group_expected"] = [round(float(p_group[n, gi[g]]), 3) if isinstance(g, str) else np.nan
                                 for n, g in enumerate(audit["group_expected"])]

    # 3b. neighbours: share of the 10 nearest images (other models) with the same broad type
    sims = emb @ emb.T
    same_ref = audit["PROD_REF"].to_numpy()[:, None] == audit["PROD_REF"].to_numpy()[None, :]
    sims[same_ref] = -1
    nn = np.argsort(-sims, axis=1)[:, :10]
    exp = audit["group_expected"].to_numpy()
    audit["nn_same_type_share"] = [round(float(np.mean(exp[nn[n]] == exp[n])), 2) if isinstance(exp[n], str) else np.nan
                                   for n in range(len(audit))]

    # 3c. colour: zero-shot over basic colours, phrased with the item type ("a black handbag")
    colours = list(COLOURS)
    ci = {c: k for k, c in enumerate(colours)}
    colour_bank = {g: clip.prompt_bank([f"a {c} {GROUP_PHRASE[g].split(' ', 1)[1] if GROUP_PHRASE[g].startswith('a ') else GROUP_PHRASE[g]}"
                                        for c in colours]) for g in groups}
    generic_bank = clip.prompt_bank([f"a {c} object" for c in colours])
    p_col = np.stack([softmax(emb[n] @ (colour_bank[g] if isinstance(g, str) else generic_bank).T)
                      for n, g in enumerate(audit["group_expected"])])
    audit["colour_pred"] = [colours[k] for k in p_col.argmax(1)]
    audit["p_colour_expected"] = [round(float(p_col[n, ci[c]]), 3) if isinstance(c, str) else np.nan
                                  for n, c in enumerate(audit["colour_expected"])]

    # 3d. near-duplicate photos (dHash, Hamming distance <= 4) across different colours or models
    h = np.array(hashes, dtype=np.uint64)
    clr = audit["CLR_DES"].to_numpy()
    ref = audit["PROD_REF"].to_numpy()
    keys = audit[KEY].to_numpy()
    near = [[] for _ in range(len(h))]
    full_sims = emb @ emb.T  # same photo => same hash AND almost identical CLIP vector
    for s in range(0, len(h), 512):
        d = np.bitwise_count(h[s:s + 512, None] ^ h[None, :])
        for a, b in zip(*np.where(d <= 2)):
            a += s
            if a != b and full_sims[a, b] >= 0.95 and (clr[a] != clr[b] or ref[a] != ref[b]):
                near[a].append(keys[b])
    audit["near_duplicate_of"] = ["|".join(v[:10]) for v in near]

    audit = audit.merge(masks[[KEY, "packshot", "mask_method", "target_found", "det_strongest"]], on=KEY)

    audit["near_duplicate_of"] = confirm_near_duplicates(audit, masks)
    audit[["vis_issues", "vis_note"]] = compute_flags(audit)

    # outputs
    suffix = "_sample" if args.limit else ""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / f"image_clip{suffix}.npy", emb.astype(np.float16))
    meta = {"model": CLIP_MODEL, "detector": DETECTOR_MODEL, "dim": int(emb.shape[1]), "dtype": "float16",
            "normalised": True, "input": "masked crop, padded to a white square, CLIP processor (224 px)",
            "mask": {"packshot_rule": "white border share > 0.9", "threshold": 235, "detector_min_score": 0.15,
                     "pad": 0.04},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": int(len(audit)),
            "keys": audit[KEY].tolist()}
    (EMB_DIR / f"image_clip{suffix}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    masks.to_parquet(EMB_DIR / f"masks{suffix}.parquet", index=False)
    audit.to_parquet(EMB_DIR / f"image_audit{suffix}.parquet", index=False)
    write_report(audit, masks, suffix, time.time() - t0)
    print(f"Done in {time.time() - t0:.0f}s -> {EMB_DIR}", flush=True)


SMALL_JEWELLERY = {"ring", "bracelet", "earrings", "keychain"}  # no sense of scale on a white background


def _thumb(f: str) -> np.ndarray:
    return np.asarray(Image.open(IMG_DIR / f).convert("RGB").resize((16, 16)), dtype=np.int16)


def confirm_near_duplicates(audit: pd.DataFrame, masks: pd.DataFrame, max_diff: float = 8.0) -> list[str]:
    """Keep a near-duplicate pair only when it means "this photo shows another colour":
    same model (PROD_REF), different colour (CLR_DES) AND almost identical product pixels.

    Compared on product pixels only (the white background would hide a silver vs gold difference).
    Same colour on another model code is a re-edition, not an error, so it is not kept.
    """
    files = dict(zip(masks[KEY], masks["img_file"]))
    colour = dict(zip(audit[KEY], audit["CLR_DES"]))
    model = dict(zip(audit[KEY], audit["PROD_REF"]))
    cache: dict[str, np.ndarray] = {}

    def thumb(k):
        if k not in cache:
            cache[k] = np.asarray(Image.open(IMG_DIR / files[k]).convert("RGB").resize((32, 32)), dtype=np.int16)
        return cache[k]

    def same_pixels(a, b) -> bool:
        ta, tb = thumb(a), thumb(b)
        fg = (ta.min(2) < 235) | (tb.min(2) < 235)
        if fg.mean() < 0.01:
            return False
        return float(np.abs(ta - tb)[fg].mean()) <= max_diff

    out = []
    for k, others in zip(audit[KEY], audit["near_duplicate_of"]):
        keep = [o for o in str(others or "").split("|")
                if o and o in files and model.get(o) == model.get(k) and colour.get(o) != colour.get(k)
                and same_pixels(k, o)]
        out.append("|".join(keep))
    return out


def compute_flags(audit: pd.DataFrame) -> pd.DataFrame:
    """Turn the stored scores into flags + a plain-language note (cheap: no model needed)."""
    def name(g):
        return GROUP_PHRASE[g].replace("a pair of ", "").replace("a ", "")

    def one(r) -> tuple[str, str]:
        out, notes = [], []
        g, gp = r.group_expected, r.group_pred
        if isinstance(g, str) and gp != g and frozenset((g, gp)) not in COMPATIBLE and r.p_group_expected < 0.10:
            # Strong only when CLIP is confident AND the most similar photos are of other types too;
            # small jewellery confused with other small jewellery is only a doubt.
            strong = (r.p_group_expected < 0.05 and r.nn_same_type_share <= 0.2
                      and not {g, gp} <= SMALL_JEWELLERY)
            out.append("VIS_TYPE_MISMATCH" if strong else "VIS_TYPE_DOUBT")
            notes.append(f"photo looks like {name(gp)} (CSV family: {r.GFA_DES_EN})")
        c, cp = r.colour_expected, r.colour_pred
        if isinstance(c, str) and cp != c and frozenset((c, cp)) not in COLOUR_NEIGHBOURS:
            # Printed / multi-colour items fool the colour check, so require either a very confident
            # model or a photo that is shared with another colour.
            if r.p_colour_expected < 0.005 or (r.p_colour_expected < 0.02 and r.near_duplicate_of):
                out.append("VIS_COLOUR_MISMATCH")
                notes.append(f"photo looks {cp} (CSV colour: {r.CLR_DES})")
        if r.target_found is False:
            out.append("VIS_TARGET_NOT_FOUND")
            notes.append(f"no {name(g)} found in the photo")
        if r.near_duplicate_of:
            out.append("IMG_NEAR_DUPLICATE")
            notes.append("same photo as the other colour(s): " + r.near_duplicate_of.replace("|", ", "))
        if not r.packshot:
            out.append("IMG_NOT_PACKSHOT")
            notes.append("not a studio photo (model, lifestyle or amateur)")
        return ";".join(out), " · ".join(notes)

    return audit.apply(lambda r: pd.Series(one(r), index=["vis_issues", "vis_note"]), axis=1)


def reflag() -> None:
    """Recompute flags and the report from the stored scores, without rerunning the models."""
    audit = pd.read_parquet(EMB_DIR / "image_audit.parquet")
    masks = pd.read_parquet(EMB_DIR / "masks.parquet")
    audit["near_duplicate_of"] = confirm_near_duplicates(audit, masks)
    audit[["vis_issues", "vis_note"]] = compute_flags(audit)
    audit.to_parquet(EMB_DIR / "image_audit.parquet", index=False)
    write_report(audit, masks, "", None)
    print(audit["vis_issues"].str.split(";").explode().loc[lambda s: s != ""].value_counts().to_string())


def contact_sheet(rows: pd.DataFrame, masks: pd.DataFrame, path: Path, cols: int = 6, w: int = 230) -> None:
    rows = rows.head(60)
    if rows.empty:
        return
    m = masks.set_index(KEY)
    h = w + 46
    sheet = Image.new("RGB", (cols * w, h * ((len(rows) + cols - 1) // cols)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(rows.itertuples(index=False)):
        img = Image.open(IMG_DIR / m.at[r.PROD_CLR_EQUIV, "img_file"]).convert("RGB")
        box = m.at[r.PROD_CLR_EQUIV, "box"]
        img.thumbnail((w - 6, w - 6))
        x, y = (i % cols) * w, (i // cols) * h
        sheet.paste(img, (x + 3, y + 3))
        if box is not None and m.at[r.PROD_CLR_EQUIV, "mask_method"] == "detector":
            bw, bh = img.size
            d.rectangle([x + 3 + box[0] * bw, y + 3 + box[1] * bh, x + 3 + box[2] * bw, y + 3 + box[3] * bh],
                        outline=(0, 160, 0), width=2)
        d.text((x + 3, y + w - 2), f"{r.PROD_CLR_EQUIV} | {r.GFA_DES_EN} | {r.CLR_DES}"[:38], fill=(0, 0, 0))
        d.text((x + 3, y + w + 12), str(r.vis_note)[:38], fill=(200, 0, 0))
        d.text((x + 3, y + w + 25), str(r.vis_note)[38:76], fill=(200, 0, 0))
    sheet.save(path, quality=88)


def write_report(audit: pd.DataFrame, masks: pd.DataFrame, suffix: str, secs: float | None) -> None:
    codes = audit["vis_issues"].str.split(";").explode().loc[lambda s: s != ""].value_counts()
    lines = [f"# Phase 1b: image audit{' (sample)' if suffix else ''}", "",
             f"- Items with an image: {len(audit):,}" + (f" · runtime {secs / 60:.1f} min on CPU" if secs else ""),
             f"- Masks: {int((masks['mask_method'] == 'threshold').sum()):,} by white-background threshold, "
             f"{int((masks['mask_method'] == 'detector').sum()):,} by the detector (OWLv2)",
             f"- Embeddings: `data/embeddings/image_clip{suffix}.npy` ({CLIP_MODEL}, 512 numbers per image)", "",
             "| Flag | Items |", "|---|---|"] + [f"| `{c}` | {n} |" for c, n in codes.items()] + [""]
    for code in ["VIS_TYPE_MISMATCH", "VIS_TYPE_DOUBT", "VIS_COLOUR_MISMATCH", "VIS_TARGET_NOT_FOUND",
                 "IMG_NEAR_DUPLICATE", "IMG_NOT_PACKSHOT"]:
        rows = audit[audit["vis_issues"].str.contains(code)].sort_values("p_group_expected")
        if len(rows):
            name = f"{code.lower()}{suffix}.jpg"
            contact_sheet(rows, masks, REPORT_DIR / name)
            lines.append(f"- {code}: contact sheet `outputs/phase1b/{name}` (first 60)")
    (REPORT_DIR / f"image_audit_report{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
