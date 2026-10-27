"""Phase 2b: photos that show a person, re-cropped to the item, and image embeddings from other encoders.

Heavy: run on ONE machine with requirements-vision.txt (+ open_clip_torch for FashionSigLIP). Results are committed
to data/embeddings/ with a sidecar .json, so teammates only load them.

    python src/phase2b_image_encoders.py persons                  # -> data/embeddings/person_crops.parquet (+ .json)
    python src/phase2b_image_encoders.py embed --model fashionsiglip   # -> data/embeddings/image_fashionsiglip.npy (+ .json)

Why (measured in outputs/phase2b/image_encoders_report.md):
- Phase 1b crops a photo on a white background to its non-white pixels. When a model wears the item, that crop is
  the whole person, so the image vector describes the outfit (e.g. 211451_BM, a swimsuit, embedded as trousers).
  `persons` finds those photos in two stages: CLIP on the whole photo ("a person", "a face", "a hand"...) as a
  cheap filter, then the OWLv2 detector for a person AND for the expected item. The rule only SUGGESTS a re-crop
  (person and item found, item box < 70% of the Phase 1b crop); a person checks each suggestion before/after and the
  decision is kept in data/embeddings/person_crops_reviewed.json, which this script never overwrites.
- `embed` encodes the item crops (with those re-crops) with a fashion-specific encoder. FashionSigLIP was chosen
  over CLIP ViT-B/32, FashionCLIP and DINOv2 on labels the encoders never saw (sub-family, material and finish of
  the nearest photos) and on the team's sorted preferences.
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
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
from config import EMB_DIR, IMG_DIR, KEY  # noqa: E402

PERSON_FILE = EMB_DIR / "person_crops.parquet"
FILTER_MIN = 0.15            # CLIP person probability on the whole photo above which OWLv2 checks it
PERSON_MIN = 0.10            # OWLv2 score for "a person / face / hand / legs" (calibrated on contact sheets)
ITEM_MIN = 0.10              # OWLv2 score of the expected item
MAX_SHARE_OF_CROP = 0.70     # the item box must be clearly smaller than the Phase 1b crop, or nothing is gained
REVIEWED_FILE = EMB_DIR / "person_crops_reviewed.json"   # human decisions on the suggestions: never regenerated
PERSON_PROMPTS = ["a person", "a fashion model wearing the item", "a woman's face", "a hand wearing jewellery",
                  "legs and feet of a person"]
PRODUCT_PROMPTS = ["a product photo on a white background", "a packshot of a fashion accessory",
                   "a garment laid flat", "shoes on a white background"]
OWL_PERSON = ["a photo of a person", "a photo of a human face", "a photo of a human hand", "a photo of human legs"]
ENCODERS = {
    "clip": ("openai/clip-vit-base-patch32", "transformers"),
    "fashionclip": ("patrickjohncyh/fashion-clip", "transformers"),
    "fashionsiglip": ("hf-hub:Marqo/marqo-fashionSigLIP", "open_clip"),
    "dinov2": ("facebook/dinov2-small", "dinov2"),
}


def item_crop(img: Image.Image, box) -> Image.Image:
    import phase1b_image_audit as audit
    return audit.square_crop(img, None if box is None or len(box) != 4 else list(box))


def box_area(b) -> float:
    return (b[2] - b[0]) * (b[3] - b[1]) if b is not None and len(b) == 4 else 1.0


def auto_rule(df: pd.DataFrame, masks: pd.DataFrame) -> pd.Series:
    """Suggested re-crops: a person, the expected item found, and an item box clearly smaller than the Phase 1b crop
    (photos already cropped to the item by the detector in Phase 1b have the same box and are left alone)."""
    current = df[KEY].map(lambda k: box_area(masks.at[k, "box"]))
    ratio = df["item_box"].map(lambda b: box_area(b) if b is not None else np.nan) / current
    return ((df["person_score"] >= PERSON_MIN) & (df["item_score"] >= ITEM_MIN) & (ratio < MAX_SHARE_OF_CROP)).fillna(False)


def reviewed() -> dict:
    """The human review of the suggestions (keys accepted / rejected after a before-after check by eye)."""
    if REVIEWED_FILE.exists():
        return json.loads(REVIEWED_FILE.read_text(encoding="utf-8"))
    return {"accepted": [], "rejected": []}


def decide(df: pd.DataFrame, masks: pd.DataFrame) -> pd.DataFrame:
    """auto_rule = the machine's suggestion; recrop = suggested AND accepted by a person (never one without the other)."""
    out = df.copy()
    out["auto_rule"] = auto_rule(out, masks)
    out["recrop"] = out["auto_rule"] & out[KEY].isin(set(reviewed()["accepted"]))
    pending = out.loc[out["auto_rule"] & ~out[KEY].isin(set(reviewed()["accepted"]) | set(reviewed()["rejected"])), KEY]
    if len(pending):
        print(f"{len(pending)} suggested re-crops not reviewed yet (kept as they are): {', '.join(pending)}")
    return out


# --------------------------------------------------------------------------- persons
def cmd_persons(args: argparse.Namespace) -> None:
    import torch
    from transformers import Owlv2ForObjectDetection, Owlv2Processor

    import phase1b_image_audit as audit
    torch.set_num_threads(args.threads)
    items = phase1_clean.load_items().set_index(KEY)
    trusted = items.index[items["img_trusted"].fillna(False).astype(bool)].tolist()
    t0 = time.time()
    clip = audit.Clip()
    pos, neg = clip.prompt_bank(PERSON_PROMPTS), clip.prompt_bank(PRODUCT_PROMPTS)
    scores = {}
    for start in range(0, len(trusted), 32):
        keys = trusted[start:start + 32]
        f = clip.images([Image.open(IMG_DIR / items.at[k, "img_file"]).convert("RGB") for k in keys])
        p = audit.softmax(np.c_[f @ pos.T, f @ neg.T])[:, :len(PERSON_PROMPTS)].sum(1)
        scores.update(dict(zip(keys, p.round(4))))
    cand = [k for k in trusted if scores[k] >= FILTER_MIN]
    print(f"CLIP filter: {len(cand):,} of {len(trusted):,} trusted photos may show a person ({time.time() - t0:.0f}s)",
          flush=True)
    proc = Owlv2Processor.from_pretrained(audit.DETECTOR_MODEL)
    model = Owlv2ForObjectDetection.from_pretrained(audit.DETECTOR_MODEL).eval()
    items_q = list(audit.DETECT_QUERY)
    rows = []
    for n, k in enumerate(cand):
        img = Image.open(IMG_DIR / items.at[k, "img_file"]).convert("RGB")
        group = audit.FAMILY_TO_GROUP.get(items.at[k, "GFA_DES_EN"])
        texts = OWL_PERSON + [f"a photo of a {audit.DETECT_QUERY[g]}" for g in items_q]
        with torch.no_grad():
            out = model(**proc(text=[texts], images=img, return_tensors="pt"))
        lg, boxes = out.logits[0].sigmoid().numpy(), out.pred_boxes[0].numpy()
        side = max(img.size)
        sx, sy = side / img.size[0], side / img.size[1]

        def rel(b, sx=sx, sy=sy):
            cx, cy, w, h = b
            return [round(float(v), 4) for v in (max(0, (cx - w / 2) * sx), max(0, (cy - h / 2) * sy),
                                                  min(1, (cx + w / 2) * sx), min(1, (cy + h / 2) * sy))]
        np_ = len(OWL_PERSON)
        row = {KEY: k, "person_full": float(scores[k]), "person_score": float(lg[:, :np_].max()),
               "item_group": group, "item_score": np.nan, "item_box": None}
        if group is not None:
            qi = np_ + items_q.index(group)
            bi = int(lg[:, qi].argmax())
            row["item_score"], row["item_box"] = float(lg[bi, qi]), rel(boxes[bi])
        rows.append(row)
        if (n + 1) % 100 == 0:
            print(f"  OWLv2 {n + 1}/{len(cand)} ({time.time() - t0:.0f}s)", flush=True)
    masks = pd.read_parquet(EMB_DIR / "masks.parquet").set_index(KEY)
    df = decide(pd.DataFrame(rows), masks)
    df.to_parquet(PERSON_FILE, index=False)
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "clip": audit.CLIP_MODEL,
            "detector": audit.DETECTOR_MODEL, "filter_min": FILTER_MIN, "person_min": PERSON_MIN, "item_min": ITEM_MIN,
            "max_share_of_crop": MAX_SHARE_OF_CROP, "reviewed_file": REVIEWED_FILE.name, "person_prompts": PERSON_PROMPTS, "product_prompts": PRODUCT_PROMPTS,
            "owl_person": OWL_PERSON, "checked": len(df), "auto": int(df["auto_rule"].sum()), "recropped": int(df["recrop"].sum()),
            "seconds": round(time.time() - t0)}
    PERSON_FILE.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"{int(df['recrop'].sum()):,} photos re-cropped to the item -> {PERSON_FILE}")


# --------------------------------------------------------------------------- embeddings
def encoder(name: str):
    import torch
    model_id, lib = ENCODERS[name]
    if lib == "open_clip":
        import open_clip
        model, _, pre = open_clip.create_model_and_transforms(model_id)
        model.eval()

        def enc(imgs):
            with torch.no_grad():
                return model.encode_image(torch.stack([pre(i) for i in imgs])).numpy()
        return enc
    if lib == "dinov2":
        from transformers import AutoImageProcessor, AutoModel
        dp, dm = AutoImageProcessor.from_pretrained(model_id), AutoModel.from_pretrained(model_id).eval()

        def enc_dino(imgs):
            with torch.no_grad():
                h = dm(**dp(images=imgs, return_tensors="pt")).last_hidden_state
            return torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1).numpy()   # CLS token + mean of the patch tokens
        return enc_dino
    from transformers import CLIPModel, CLIPProcessor
    proc, model = CLIPProcessor.from_pretrained(model_id), CLIPModel.from_pretrained(model_id).eval()

    def enc(imgs):
        with torch.no_grad():
            o = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
        return (o if hasattr(o, "norm") else o.pooler_output).numpy()
    return enc


def crop_boxes(use_person_crops: bool = True) -> tuple[list[str], dict[str, list | None], dict[str, str]]:
    """Keys in the order of the Phase 1b embeddings, their crop box (re-crop where decided) and image file."""
    masks = pd.read_parquet(EMB_DIR / "masks.parquet")
    keys = json.loads((EMB_DIR / "image_clip.json").read_text(encoding="utf-8"))["keys"]
    m = masks.set_index(KEY)
    boxes = {k: (None if m.at[k, "box"] is None else list(m.at[k, "box"])) for k in keys}
    if use_person_crops and PERSON_FILE.exists():
        pc = pd.read_parquet(PERSON_FILE)
        for k, b in zip(pc.loc[pc["recrop"], KEY], pc.loc[pc["recrop"], "item_box"]):
            if k in boxes:
                boxes[k] = list(b)
    return keys, boxes, dict(zip(m.index, m["img_file"]))


def cmd_embed(args: argparse.Namespace) -> None:
    import torch
    torch.set_num_threads(args.threads)
    keys, boxes, files = crop_boxes(not args.no_person_crops)
    enc = encoder(args.model)
    t0, vecs, batch = time.time(), [], []
    for i, k in enumerate(keys):
        batch.append(item_crop(Image.open(IMG_DIR / files[k]).convert("RGB"), boxes[k]))
        if len(batch) == args.batch_size or i == len(keys) - 1:
            vecs.append(enc(batch))
            batch = []
            if (i + 1) % 1024 < args.batch_size:
                print(f"  {i + 1}/{len(keys)} ({time.time() - t0:.0f}s)", flush=True)
    m = np.concatenate(vecs).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    stem = f"image_{args.model}" + ("" if not args.no_person_crops else "_nopersoncrops")
    np.save(EMB_DIR / f"{stem}.npy", m.astype(np.float16))
    n_re = sum(1 for k in keys if PERSON_FILE.exists() and not args.no_person_crops
               and k in set(pd.read_parquet(PERSON_FILE).query("recrop")[KEY]))
    import importlib.metadata as md
    lib = {"open_clip": "open_clip_torch", "transformers": "transformers", "dinov2": "transformers"}[ENCODERS[args.model][1]]
    meta = {"model": ENCODERS[args.model][0], "library": ENCODERS[args.model][1], "library_version": md.version(lib),
            "torch_version": md.version("torch"), "dim": int(m.shape[1]),
            "dtype": "float16", "normalised": True,
            "input": "Phase 1b item crop padded to a white square; photos with a person re-cropped to the item "
                     "(person_crops.parquet)" if not args.no_person_crops else "Phase 1b item crop",
            "recropped": n_re, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0), "n": len(keys), "keys": keys}
    (EMB_DIR / f"{stem}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{m.shape} -> {EMB_DIR / stem}.npy ({time.time() - t0:.0f}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("persons")
    p.add_argument("--threads", type=int, default=8)
    e = sub.add_parser("embed")
    e.add_argument("--model", choices=list(ENCODERS), default="fashionsiglip")
    e.add_argument("--no-person-crops", action="store_true", help="use the Phase 1b crops unchanged")
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    {"persons": cmd_persons, "embed": cmd_embed}[args.cmd](args)


if __name__ == "__main__":
    main()
