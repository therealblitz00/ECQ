"""Phase 2a: text embedding of a description built from the product attributes.

Heavy step: run it on ONE machine with requirements-vision.txt (torch + transformers). The result is
committed to data/embeddings/text.npy + text.json, and phase2a_features.py loads it as the `text` block.

    python src/phase2a_text.py            # ~1 min on a laptop CPU
    python src/phase2a_text.py --show 5   # print 5 descriptions and stop (no model needed)

Why a text block on top of the categorical one: the categorical block only matches identical labels, while a
sentence model also sees that "Golden Basics" and "GOLDEN DELICATES", or "Hoop Earrings" and "Huggie", are
close. The description uses the reviewed attributes, never the photo.
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
from config import EMB_DIR, KEY  # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEMPLATE = "{type}. Colour: {colour}. {finish}{material}{print}Theme: {theme}. {style}"


def _clean(v) -> str:
    return "" if pd.isna(v) else str(v).strip()


def describe(items: pd.DataFrame) -> pd.Series:
    """One English sentence per item, e.g. "Hoop Earrings (Earrings, Jewellery). Colour: Gold. ..."."""
    def one(r) -> str:
        theme = _clean(r.get("THEME"))
        theme = theme.title() if theme.isupper() else theme  # many themes are in capitals
        parts = {
            "type": f"{_clean(r.get('GFS_DES_EN'))} ({_clean(r.get('GFA_DES_EN'))}, {_clean(r.get('CAT_DES_EN'))})",
            "colour": _clean(r.get("CLR_DES")) or "unknown",
            "finish": f"Finish: {_clean(r.get('FINISHING'))}. " if _clean(r.get("FINISHING")) else "",
            "material": f"Made of {_clean(r.get('COMPOSITION')).replace(';', ',')}. " if _clean(r.get("COMPOSITION")) else "",
            "print": f"Print: {_clean(r.get('PRINT_TYPE'))}. " if _clean(r.get("PRINT_TYPE")) not in ("", "No Print") else "",
            "theme": theme or "none",
            "style": f"{_clean(r.get('FASHIONTYPE'))}, {_clean(r.get('PROD_SEG'))}.",
        }
        return TEMPLATE.format(**parts).strip()
    return items.apply(one, axis=1)


def embed(texts: list[str], batch_size: int = 128) -> tuple[np.ndarray, str]:
    """Mean-pooled, L2-normalised sentence embeddings (the standard all-MiniLM-L6-v2 recipe)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(MODEL), AutoModel.from_pretrained(MODEL).eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch_size):
            enc = tok(texts[s:s + batch_size], padding=True, truncation=True, max_length=128, return_tensors="pt")
            hidden = model(**enc).last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            out.append(torch.nn.functional.normalize(vec, dim=1).numpy())
            print(f"  {min(s + batch_size, len(texts)):,}/{len(texts):,}", end="\r", flush=True)
    from huggingface_hub import snapshot_download  # the local snapshot folder is named after the model commit
    return np.vstack(out).astype(np.float16), Path(snapshot_download(MODEL, local_files_only=True)).name


def save(emb: np.ndarray, revision: str, keys: list[str], texts: list[str], source: str) -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / "text.npy", emb)
    meta = {"model": MODEL, "revision": revision, "dim": int(emb.shape[1]), "dtype": str(emb.dtype), "normalised": True,
            "pooling": "mean over tokens", "input": "description built from the attributes (describe())",
            "template": TEMPLATE, "source": source, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n": len(keys), "keys": keys, "examples": texts[:3]}
    (EMB_DIR / "text.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, help="print N descriptions and stop")
    args = ap.parse_args()
    items = phase1_clean.load_items().sort_values(KEY, kind="stable").reset_index(drop=True)
    texts = describe(items).tolist()
    if args.show:
        print("\n".join(texts[:args.show]))
        return
    source = "items_clean" if (phase1_clean.PROCESSED_DIR / phase1_clean.CLEAN_FILE).exists() else "items_preview"
    print(f"Embedding {len(texts):,} descriptions with {MODEL}...")
    emb, revision = embed(texts)
    save(emb, revision, items[KEY].tolist(), texts, source)
    print(f"\n-> {EMB_DIR / 'text.npy'}")


if __name__ == "__main__":
    main()
