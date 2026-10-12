"""Phase 2a: colour tones measured on the photos, so close colour names score as close.

    python src/phase2a_colour.py          # -> data/processed/colour_reference.parquet + outputs/phase2a/colour_reference.{csv,jpg}

Also run by phase2a_features.py. Phase 5 uses the table to say that two different names are close colours
("Navy ≈ Midnight Blue"). The engine does not use it: a soft colour match in the categorical block (`tone_rows`)
changed none of the Phase 4 numbers at the same colour weight (docs/progress.md, entry 22).

`CLR_DES` has about 100 names and many are shades of one colour ("Navy", "Midnight Blue", "Blue"). As a one-hot
they are as different as "Blue" and "Red". Here every name gets a reference colour measured on the photos:
the median, in CIELAB, of the dominant palette colour (Phase 2b) of the trusted packshots with that name.
Two names then score exp(-d² / 2τ²), where d is their distance in LAB with lightness counted half (the
textile convention of CMC 2:1: a lighter or darker shade of a colour is still that colour).

Names without a measured colour only match themselves:
- multicolour names (one reference colour means nothing for a print);
- metallic names (Gold, Silver…): on a photo a gold finish measures as light beige, and gold vs silver is a
  key distinction for buyers that FINISHING already encodes;
- names with fewer than MIN_PHOTOS photos.

The similarity matrix of the names is factorised (eigen-decomposition, it is positive semi-definite), so each
item gets a row whose dot product with another item's row is exactly the similarity of their two names.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
from config import EMB_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase2a"
REFERENCE_FILE = PROCESSED_DIR / "colour_reference.parquet"
PALETTES_FILE = EMB_DIR / "palettes.parquet"
TAU = 8.0             # 8 LAB units apart -> similarity 0.61; 16 apart -> 0.14
L_WEIGHT = 0.5        # lightness counts half (CMC 2:1)
MIN_PHOTOS = 5
METALLIC = {"Gold", "Light Gold", "Old Gold", "Rose Gold", "Silver", "Old Silver", "Nickel", "Black Nickel",
            "Copper", "Bronze"}


def lab_to_hex(lab: np.ndarray) -> str:
    """CIELAB (D65) -> sRGB hex, clipped to the gamut."""
    L, a, b = lab
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f > 6 / 29, f**3, 3 * (6 / 29) ** 2 * (f - 4 / 29)) * np.array([0.95047, 1.0, 1.08883])
    m = np.array([[3.2404542, -1.5371385, -0.4985314], [-0.9692660, 1.8760108, 0.0415560],
                  [0.0556434, -0.2040259, 1.0572252]])
    c = np.clip(m @ xyz, 0, 1)
    c = np.where(c > 0.0031308, 1.055 * c ** (1 / 2.4) - 0.055, 12.92 * c)
    return "#" + "".join(f"{round(v * 255):02x}" for v in c)


def exclusion(name: str) -> str | None:
    if "multicolor" in name.lower():
        return "multicolour"
    if name in METALLIC:
        return "metallic"
    return None


def reference_colours(items: pd.DataFrame, palettes: pd.DataFrame | None, min_photos: int = MIN_PHOTOS) -> pd.DataFrame:
    """One row per CLR_DES name: photos used, reference LAB colour (or why there is none)."""
    names = sorted(items["CLR_DES"].dropna().astype(str).unique())
    ref = pd.DataFrame({"CLR_DES": names, "items": items["CLR_DES"].value_counts().reindex(names).to_numpy()})
    ref[["photos", "L", "a", "b"]] = np.nan
    if palettes is not None and len(palettes):
        trusted = items.loc[items["img_colour_trusted"].fillna(False).astype(bool), [KEY, "CLR_DES"]]
        pal = palettes[palettes["reliable"]].merge(trusted, on=KEY)
        # the dominant colour of each photo (largest share of the product pixels)
        dom = np.stack([np.stack(lab)[int(np.argmax(s))] for lab, s in zip(pal["lab"], pal["shares"])]) \
            if len(pal) else np.zeros((0, 3))
        med = pd.DataFrame(dom, columns=["L", "a", "b"]).groupby(pal["CLR_DES"].to_numpy()).median()
        med["photos"] = pal["CLR_DES"].value_counts()
        ref = ref.set_index("CLR_DES")
        ref.update(med)
        ref = ref.reset_index()
    ref["photos"] = ref["photos"].fillna(0).astype(int)
    reason = ref["CLR_DES"].map(exclusion)
    reason = reason.where(reason.notna(), np.where(ref["photos"] < min_photos, f"under {min_photos} photos", None))
    ref["no_reference"] = reason
    ref.loc[ref["no_reference"].notna(), ["L", "a", "b"]] = np.nan
    ref[["L", "a", "b"]] = ref[["L", "a", "b"]].round(2)
    ref["hex"] = [lab_to_hex(r) if not np.isnan(r).any() else "" for r in ref[["L", "a", "b"]].to_numpy()]
    return ref


def similarity(ref: pd.DataFrame, tau: float = TAU, l_weight: float = L_WEIGHT) -> np.ndarray:
    """Name × name similarity (1 on the diagonal; names without a reference only match themselves)."""
    lab = ref[["L", "a", "b"]].to_numpy(float) * np.array([l_weight, 1.0, 1.0])
    d2 = ((lab[:, None, :] - lab[None, :, :]) ** 2).sum(axis=2)
    sim = np.exp(-d2 / (2 * tau**2))
    sim[np.isnan(sim)] = 0
    np.fill_diagonal(sim, 1.0)
    return sim


def tone_rows(names: pd.Series, ref: pd.DataFrame, tau: float = TAU) -> tuple[np.ndarray, list[str]]:
    """One row per item; row_i · row_j = similarity of their names (zeros when CLR_DES is missing)."""
    vals, vecs = np.linalg.eigh(similarity(ref, tau))
    vals = np.clip(vals, 0, None)  # rounding can leave tiny negative eigenvalues
    keep = vals > 1e-9
    factor = vecs[:, keep] * np.sqrt(vals[keep])
    # the sign of an eigenvector is arbitrary: fix it so every machine writes the same file
    factor *= np.where(factor[np.abs(factor).argmax(axis=0), np.arange(factor.shape[1])] < 0, -1, 1)
    codes = pd.Categorical(names.astype("string"), categories=ref["CLR_DES"]).codes
    rows = np.zeros((len(names), factor.shape[1]), dtype=np.float32)
    rows[codes >= 0] = factor[codes[codes >= 0]]
    return rows, [f"CLR_TONE_{k}" for k in range(factor.shape[1])]


def load_palettes() -> pd.DataFrame | None:
    return pd.read_parquet(PALETTES_FILE) if PALETTES_FILE.exists() else None


def closest(ref: pd.DataFrame, sim: np.ndarray, n: int = 3, floor: float = 0.3) -> list[str]:
    out = []
    for i in range(len(ref)):
        order = [j for j in np.argsort(-sim[i], kind="stable") if j != i and sim[i, j] >= floor][:n]
        out.append(", ".join(f"{ref['CLR_DES'].iat[j]} {sim[i, j]:.2f}" for j in order))
    return out


def write_outputs(ref: pd.DataFrame, tau: float = TAU) -> None:
    """The reference table (for Phases 3-5), a CSV for humans and a swatch sheet to check it by eye."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(REFERENCE_FILE, index=False)
    meta = {"method": "median CIELAB of the dominant palette colour of trusted packshots per CLR_DES",
            "tau": tau, "l_weight": L_WEIGHT, "min_photos": MIN_PHOTOS, "metallic": sorted(METALLIC),
            "similarity": "exp(-d^2 / (2 tau^2)), d = LAB distance with L scaled by l_weight; no reference: 1 only "
                          "with itself", "source": "data/embeddings/palettes.parquet"}
    REFERENCE_FILE.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    sim = similarity(ref, tau)
    human = ref.assign(closest_names=closest(ref, sim)).sort_values(["items", "CLR_DES"], ascending=[False, True])
    human.to_csv(REPORT_DIR / "colour_reference.csv", index=False, encoding="utf-8-sig")
    swatch_sheet(human[human["hex"] != ""], REPORT_DIR / "colour_reference.jpg")


def swatch_sheet(ref: pd.DataFrame, path: Path, cols: int = 6, w: int = 200, h: int = 86) -> None:
    rows = max(1, -(-len(ref) // cols))  # an empty sheet still has one row (no palettes: nothing measured)
    img = Image.new("RGB", (cols * w, rows * h), "white")
    d = ImageDraw.Draw(img)
    for n, r in enumerate(ref.itertuples()):
        x, y = (n % cols) * w, (n // cols) * h
        d.rectangle([x + 6, y + 6, x + w - 6, y + 50], fill=r.hex, outline=(160, 160, 160))
        d.text((x + 8, y + 54), f"{r.CLR_DES}", fill="black")
        d.text((x + 8, y + 68), f"{r.photos} photos · {r.hex}", fill=(90, 90, 90))
    img.save(path, quality=92)


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    items = phase1_clean.load_items()
    ref = reference_colours(items, load_palettes())
    write_outputs(ref)
    used = ref["no_reference"].isna()
    print(f"{used.sum()} of {len(ref)} colour names have a measured colour, covering "
          f"{ref.loc[used, 'items'].sum() / ref['items'].sum():.0%} of the items with a colour name")


if __name__ == "__main__":
    main()
