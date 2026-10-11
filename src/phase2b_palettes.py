"""Phase 2b: dominant colour palette of each product photo (k-means in LAB on the product pixels).

    python src/phase2b_palettes.py             # all trusted photos (~3-5 min, light environment)
    python src/phase2b_palettes.py --limit 60  # quick trial + swatch sheet

Output (committed, computed once): data/embeddings/palettes.parquet + palettes.json, and
outputs/phase2b/palettes_report.md + palettes_sample.jpg (photos next to their swatches).

Per item: up to K colours (LAB centre, hex, share of the product pixels, nearest colour name) and a
`palette_vector`: the shares summed per colour name and L2-normalised, so the dot product of two items
is a palette similarity whose parts read as "both mostly gold and white" (used by Phases 3 and 5).

Product pixels: inside the item box from data/embeddings/masks.parquet (Phase 1b), without the
near-white background. On packshots that is the product; on model / lifestyle photos the box still
holds some background, so their palette is marked `reliable = False`. Only photos the review kept
(`img_trusted`) are used.
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
from config import EMB_DIR, IMG_DIR, KEY, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase2b"
K = 4                 # colours per palette
MIN_SHARE = 0.05      # smaller clusters are dropped (and the shares renormalised)
WHITE = 235           # a pixel with all channels >= WHITE is background
SIDE = 96             # the crop is resized to at most SIDE px before clustering
SEED = 0

# Reference colours for naming palette entries (sRGB). Metals are separate from yellow/grey on purpose.
NAMED = {
    "black": (25, 25, 25), "white": (245, 245, 242), "grey": (128, 128, 128), "silver": (190, 190, 195),
    "gold": (205, 170, 90), "beige": (222, 203, 170), "camel": (190, 140, 85), "brown": (110, 70, 40),
    "red": (200, 35, 40), "burgundy": (115, 25, 45), "pink": (240, 150, 175), "fuchsia": (215, 45, 140),
    "orange": (240, 130, 45), "yellow": (245, 215, 55), "green": (45, 150, 75), "khaki": (120, 120, 65),
    "mint": (165, 220, 190), "teal": (10, 120, 120), "turquoise": (60, 195, 200), "light blue": (155, 195, 230),
    "blue": (45, 85, 190), "navy": (28, 38, 80), "purple": (110, 55, 150), "lilac": (190, 165, 220),
}
NAMES = list(NAMED)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0-255, any shape (..., 3)) -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


NAMED_LAB = rgb_to_lab(np.array(list(NAMED.values())))


def colour_name(lab: np.ndarray) -> np.ndarray:
    """Nearest reference colour (plain Euclidean ΔE76) for each LAB row."""
    d = ((np.atleast_2d(lab)[:, None, :] - NAMED_LAB[None, :, :]) ** 2).sum(-1)
    return np.array(NAMES)[d.argmin(1)]


def kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means with k-means++ start and a fixed seed (identical result on every machine)."""
    rng = np.random.default_rng(seed)
    k = min(k, len(np.unique(x, axis=0)))
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(centres)
    for _ in range(iters):
        labels = ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[labels == j].mean(0) if (labels == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c, ((x[:, None] - c[None]) ** 2).sum(-1).argmin(1)


def product_pixels(img: Image.Image, box) -> np.ndarray:
    """RGB pixels of the product: inside the box, background (near-white) removed."""
    img = img.convert("RGB")
    if box is not None and len(box) == 4:
        w, h = img.size
        x0, y0, x1, y1 = box
        img = img.crop((int(x0 * w), int(y0 * h), max(int(x0 * w) + 1, int(x1 * w)), max(int(y0 * h) + 1, int(y1 * h))))
    img.thumbnail((SIDE, SIDE))
    a = np.asarray(img, dtype=np.uint8).reshape(-1, 3)
    keep = a.min(1) < WHITE
    # an all-white product (white shirt, pearl) would vanish: then keep the box as it is
    return a[keep] if keep.mean() > 0.02 else a


def palette(pixels: np.ndarray, k: int = K) -> list[dict]:
    """Dominant colours, largest first: lab, hex, share, name."""
    lab = rgb_to_lab(pixels)
    centres, labels = kmeans(lab, k)
    share = np.bincount(labels, minlength=len(centres)) / len(labels)
    keep = share >= MIN_SHARE
    centres, share = centres[keep], share[keep] / share[keep].sum()
    rgb = np.array([pixels[labels == j].mean(0) for j in np.flatnonzero(keep)])
    order = np.argsort(-share, kind="stable")
    names = colour_name(centres)
    return [{"lab": centres[j].round(2).tolist(), "hex": "#%02x%02x%02x" % tuple(int(v) for v in rgb[j].round()),
             "share": round(float(share[j]), 4), "name": str(names[j])} for j in order]


def palette_vector(entries: list[dict]) -> np.ndarray:
    """Shares summed per colour name, unit length (dot product of two items = palette similarity)."""
    v = np.zeros(len(NAMES), dtype=np.float32)
    for e in entries:
        v[NAMES.index(e["name"])] += e["share"]
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def shared_colours(a: np.ndarray, b: np.ndarray, min_part: float = 0.05) -> dict[str, float]:
    """Contribution of each colour name to the palette similarity of two items."""
    return {NAMES[j]: float(a[j] * b[j]) for j in np.flatnonzero(a * b >= min_part)}


def items_to_do(limit: int | None) -> pd.DataFrame:
    items = phase1_clean.load_items()
    items = items[items["img_trusted"].fillna(False) & items["img_file"].notna()]
    masks = pd.read_parquet(EMB_DIR / "masks.parquet")[[KEY, "box", "packshot", "mask_method"]]
    items = items[[KEY, "img_file", "CLR_DES", "GFA_DES_EN"]].merge(masks, on=KEY, how="left")
    items = items.sort_values(KEY, kind="stable").reset_index(drop=True)
    return items.sample(limit, random_state=SEED).sort_values(KEY).reset_index(drop=True) if limit else items


def compute(items: pd.DataFrame) -> pd.DataFrame:
    rows, t0 = [], time.time()
    for n, r in enumerate(items.itertuples(), 1):
        try:
            with Image.open(IMG_DIR / r.img_file) as img:
                box = None if r.box is None or (isinstance(r.box, float) and np.isnan(r.box)) else list(r.box)
                entries = palette(product_pixels(img, box))
        except OSError:
            continue  # damaged file: already flagged IMG_UNREADABLE in Phase 1
        rows.append({KEY: getattr(r, KEY), "img_file": r.img_file, "reliable": bool(r.packshot is True),
                     "n_colours": len(entries), "names": [e["name"] for e in entries],
                     "hex": [e["hex"] for e in entries], "shares": [e["share"] for e in entries],
                     "lab": [e["lab"] for e in entries], "palette_vector": palette_vector(entries).tolist()})
        if n % 500 == 0:
            print(f"  {n:,}/{len(items):,} ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def swatch_sheet(pal: pd.DataFrame, path: Path, n: int = 24, w: int = 180) -> None:
    sample = pal.sample(min(n, len(pal)), random_state=SEED).sort_values(KEY)
    cols = 6
    sheet = Image.new("RGB", (cols * w, ((len(sample) + cols - 1) // cols) * (w + 46)), "white")
    d = ImageDraw.Draw(sheet)
    for i, r in enumerate(sample.itertuples()):
        x, y = (i % cols) * w, (i // cols) * (w + 46)
        with Image.open(IMG_DIR / r.img_file) as im:
            im = im.convert("RGB")
            im.thumbnail((w - 8, w - 8))
            sheet.paste(im, (x + 4, y + 4))
        sx = x + 4
        for hx, sh in zip(r.hex, r.shares):
            ww = max(1, int((w - 8) * sh))
            d.rectangle([sx, y + w, sx + ww, y + w + 18], fill=hx)
            sx += ww
        d.text((x + 4, y + w + 22), f"{getattr(r, KEY)}: {', '.join(r.names[:3])}", fill="black")
    sheet.save(path, quality=85)


def write_outputs(pal: pd.DataFrame, items: pd.DataFrame, secs: float, trial: bool) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_sample" if trial else ""
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    pal.to_parquet(EMB_DIR / f"palettes{suffix}.parquet", index=False)
    meta = {"method": "k-means in CIELAB on product pixels", "k": K, "min_share": MIN_SHARE, "white": WHITE,
            "side": SIDE, "seed": SEED, "names": NAMES, "vector": "shares summed per name, L2-normalised",
            "source": "img_trusted photos of load_items(), box from masks.parquet",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n": len(pal),
            "keys": pal[KEY].tolist()}
    (EMB_DIR / f"palettes{suffix}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    swatch_sheet(pal, REPORT_DIR / "palettes_sample.jpg")
    top = pal.explode("names").groupby("names").size().sort_values(ascending=False)
    main = pal["names"].str[0].value_counts()
    # how often the main palette colour agrees with the CSV colour name (rough check)
    m = pal.merge(items[[KEY, "CLR_DES"]], on=KEY)
    agree = m.apply(lambda r: any(n in str(r.CLR_DES).lower() for n in r.names), axis=1).mean()
    rows = "\n".join(f"| {name} | {int(main.get(name, 0)):,} | {int(top[name]):,} |" for name in top.index)
    text = f"""# Phase 2b: colour palettes

Generated by `src/phase2b_palettes.py` in {secs:.0f} s: {len(pal):,} photos (only photos kept by the review),
{int(pal['reliable'].sum()):,} of them packshots (`reliable`). Up to {K} colours per photo, clusters under
{MIN_SHARE:.0%} dropped. Swatches next to their photos: `palettes_sample.jpg`.

Rough check: for {agree:.0%} of the items, one of the palette colour names appears in the CSV colour name
(e.g. "gold" in "Light Gold"). Names don't map one to one (e.g. "Ecru" vs white/beige, multicolour items), so
this is a sanity check, not an accuracy.

**Metals look like light neutrals.** On a packshot, the largest cluster of a gold item is its highlights:
light and barely saturated (median LAB of the CSV "Gold" items: L 86, b 15), the same as beige or ecru; silver
looks white or grey. So colour names don't tell metal from beige or white: use `FINISHING` from the CSV
for the metal. Palette *similarity* is not affected, because two gold items get the same palette.

| Colour name | Main colour of | In the palette of |
|---|---|---|
{rows}
"""
    (REPORT_DIR / f"palettes_report{suffix}.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, help="random sample of N photos (trial)")
    args = ap.parse_args()
    t0 = time.time()
    items = items_to_do(args.limit)
    print(f"{len(items):,} photos")
    pal = compute(items)
    write_outputs(pal, items, time.time() - t0, bool(args.limit))
    print(f"Done in {time.time() - t0:.0f}s -> {EMB_DIR / 'palettes.parquet'}")


if __name__ == "__main__":
    main()
