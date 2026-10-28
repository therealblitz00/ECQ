"""Phase 3: similarity engine. Top-k similar colourways for every colourway, with the score split by block.

    python src/phase3_engine.py build                   # -> data/processed/neighbours.parquet + .json, outputs/phase3/
    python src/phase3_engine.py build --scope family    # candidates from the same family only
    python src/phase3_engine.py query 167718_BU --k 5   # one item, printed

Inputs (light environment, no torch): the Phase 2a blocks (`features_<block>.npy`, `feature_meta.json`), the
image embeddings (`data/embeddings/<image_embedding>.npy`, CLIP by default, set in phase3_weights.json)
and the colour palettes (`palettes.parquet`).
Rerun after `phase2a_features.py`.

Score of a candidate c for a query q:

    score = sum_b w_b * sim_b(q, c) / sum_b w_b      over the blocks b that BOTH items have

- Every block has unit-length rows, so sim_b is a dot product (cosine similarity). An item "has" a block when
  its row is not all zeros.
- A photo feeds the `image` block only if `img_trusted`, and the `palette` block only if `img_colour_trusted`.
  Otherwise the item is scored on its other blocks with the weights rescaled (the fallback of the roadmap), so a
  flagged photo can never make an item a match.
- `contrib_<block>` = w_b * sim_b / sum_b w_b; the contributions add up to the score (Phase 5 explains with them).

Candidates: same `CAT_DES_EN` (or same family with `--scope family`), never the query itself, and never another
colour of the same model (`PROD_REF` or `PROD_REF_EQUIV`) unless `--include-siblings`: those would make the
results trivially easy. Ties are broken by the candidate key, so every machine gets the same file.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
from config import EMB_DIR, IMG_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402

WEIGHTS_FILE = Path(__file__).resolve().parent / "phase3_weights.json"
OUT_FILE = "neighbours.parquet"
REPORT_DIR = ROOT / "outputs" / "phase3"
TABULAR = ["categorical", "composition", "numeric", "text"]  # built by phase2a_features.py
SCOPES = {"category": ["CAT_DES_EN"], "family": ["CAT_DES_EN", "GFA_DES_EN"]}
MIN_K = 4             # the brief: at least 4 similar products
CHUNK = 1024          # queries scored at once (memory: CHUNK x largest category x float32 per block)
DECIMALS = 6          # scores are rounded before ranking, so BLAS rounding never reorders ties


# --------------------------------------------------------------------------- inputs
def load_weights(path: Path = WEIGHTS_FILE) -> dict[str, float]:
    return {b: float(w) for b, w in json.loads(path.read_text(encoding="utf-8"))["weights"].items()}


def image_embedding(path: Path = WEIGHTS_FILE) -> str:
    """Stem of the image embedding the engine uses (`data/embeddings/<stem>.npy` + `.json`); CLIP by default."""
    return json.loads(path.read_text(encoding="utf-8")).get("image_embedding", "image_clip")


def aligned(keys: list[str], emb: np.ndarray, emb_keys: list[str], use: np.ndarray) -> np.ndarray:
    """Rows of `emb` in the order of `keys`; zeros for keys not embedded or not allowed by `use`."""
    pos = pd.Series(range(len(emb_keys)), index=emb_keys)
    idx = pd.Series(keys).map(pos)
    out = np.zeros((len(keys), emb.shape[1]), dtype=np.float32)
    ok = idx.notna().to_numpy() & use
    out[ok] = emb[idx[ok].astype(int).to_numpy()]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)  # float16 files: renormalise


def load_blocks(items: pd.DataFrame, image: str | None = None) -> dict[str, np.ndarray]:
    """Every block as a float32 matrix in the row order of `items` (sorted by KEY, like the Phase 2a files).

    `image`: stem of the image embedding (default: `image_embedding` in phase3_weights.json, else image_clip)."""
    keys = items[KEY].tolist()
    meta = json.loads((PROCESSED_DIR / "feature_meta.json").read_text(encoding="utf-8"))
    if meta["keys"] != keys:
        raise SystemExit("feature_meta.json does not match the Phase 1 table. Run: python src/phase2a_features.py")
    blocks = {b: np.load(PROCESSED_DIR / meta["blocks"][b]["file"]) for b in TABULAR if b in meta["blocks"]}
    stem = image or image_embedding()
    clip = json.loads((EMB_DIR / f"{stem}.json").read_text(encoding="utf-8"))
    blocks["image"] = aligned(keys, np.load(EMB_DIR / f"{stem}.npy").astype(np.float32), clip["keys"],
                              items["img_trusted"].fillna(False).to_numpy(bool))
    pal = pd.read_parquet(EMB_DIR / "palettes.parquet")
    blocks["palette"] = aligned(keys, np.stack(pal["palette_vector"].map(np.asarray)).astype(np.float32),
                                pal[KEY].tolist(), items["img_colour_trusted"].fillna(False).to_numpy(bool))
    return blocks


# --------------------------------------------------------------------------- scoring
def has_block(blocks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {b: (np.abs(m).sum(axis=1) > 0).astype(np.float32) for b, m in blocks.items()}


def score_matrix(blocks: dict, has: dict, weights: dict, q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Scores of queries q (rows) against candidates c (columns), weights rescaled per pair."""
    num = np.zeros((len(q), len(c)), dtype=np.float32)
    den = np.zeros_like(num)
    for b, w in weights.items():
        if w <= 0 or b not in blocks:
            continue
        both = np.outer(has[b][q], has[b][c]) * np.float32(w)
        num += both * (blocks[b][q] @ blocks[b][c].T)
        den += both
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def pair_breakdown(blocks: dict, has: dict, weights: dict, q: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    """sim_<b> (NaN when one item lacks the block) and contrib_<b> for the pairs (q[i], c[i])."""
    used = [b for b, w in weights.items() if w > 0 and b in blocks]
    den = sum(np.float32(weights[b]) * has[b][q] * has[b][c] for b in used)
    out = {}
    for b in used:
        both = has[b][q] * has[b][c]
        sim = np.einsum("ij,ij->i", blocks[b][q], blocks[b][c])
        out[f"sim_{b}"] = np.where(both > 0, sim, np.nan)
        out[f"contrib_{b}"] = np.divide(weights[b] * both * sim, den, out=np.zeros_like(sim), where=den > 0)
    return out


def groups(items: pd.DataFrame, scope: str) -> list[np.ndarray]:
    """Row indices of each candidate pool (category, or category + family)."""
    keys = items[SCOPES[scope]].astype("string").fillna("<missing>")
    return [np.asarray(ix) for _, ix in sorted(keys.groupby(list(keys.columns), sort=True).indices.items())]


def excluded(items: pd.DataFrame, q: np.ndarray, c: np.ndarray, include_siblings: bool) -> np.ndarray:
    """True where candidate c may not be returned for query q (itself, or the same model)."""
    out = q[:, None] == c[None, :]
    if not include_siblings:
        for col in ("PROD_REF", "PROD_REF_EQUIV"):
            v = items[col].astype("string").fillna("").to_numpy(dtype=object)
            out |= (v[q][:, None] == v[c][None, :]) & (v[q][:, None] != "")
    return out


def neighbours(items: pd.DataFrame, blocks: dict, weights: dict, k: int = 10, scope: str = "category",
               include_siblings: bool = False, queries: list[str] | None = None,
               two_stage: bool = True, stage1_pool: int = 50, max_per_model: int = 1) -> pd.DataFrame:
    """Top-k candidates of every query (or only of `queries`): one row per (query, rank).

    When two_stage is True and the query has a trusted photo, Stage 1 selects the most visually
    similar candidates (top stage1_pool), and Stage 2 re-ranks them by composite score.
    When max_per_model is >= 1, at most max_per_model colourways of the same PROD_REF are returned.
    """
    has = has_block(blocks)
    wanted = None if queries is None else items[KEY].isin(queries).to_numpy()
    models = items["PROD_REF"].astype("string").fillna("").to_numpy(dtype=object)
    has_image = "image" in blocks and np.any(has["image"] > 0)
    frames = []

    for pool in groups(items, scope):
        qs = pool if wanted is None else pool[wanted[pool]]
        if len(qs) == 0:
            continue
        pool_has_img = (has["image"][pool] > 0) if has_image else np.zeros(len(pool), bool)
        pool_models = models[pool]

        for start in range(0, len(qs), CHUNK):
            q = qs[start:start + CHUNK]
            s = np.round(score_matrix(blocks, has, weights, q, pool).astype(np.float64), DECIMALS)
            ex = excluded(items, q, pool, include_siblings)
            s[ex] = -np.inf

            # Stage 1 filter: for queries with a trusted photo, restrict to top stage1_pool visual candidates
            if two_stage and has_image:
                q_has_img = has["image"][q] > 0
                if np.any(q_has_img):
                    vis = blocks["image"][q] @ blocks["image"][pool].T
                    vis_masked = np.where(pool_has_img[None, :] & ~ex, vis, -np.inf)
                    for row_i in np.flatnonzero(q_has_img):
                        finite_vis = np.flatnonzero(np.isfinite(vis_masked[row_i]))
                        if len(finite_vis) > stage1_pool:
                            top_vis_local = finite_vis[np.argsort(-vis_masked[row_i, finite_vis])[:stage1_pool]]
                            mask = np.ones(len(pool), bool)
                            mask[top_vis_local] = False
                            s[row_i, mask] = -np.inf

            # Selection with diversity
            order = np.argsort(-s, axis=1, kind="stable")
            q_out, rank_out, c_out, score_out = [], [], [], []

            for row_i in range(len(q)):
                row_q = q[row_i]
                row_order = order[row_i]
                seen_models: dict[str, int] = {}
                count = 0
                for col_j in row_order:
                    val = s[row_i, col_j]
                    if not np.isfinite(val):
                        break
                    cand_idx = pool[col_j]
                    m = pool_models[col_j]
                    if max_per_model and max_per_model > 0 and m != "":
                        c_count = seen_models.get(m, 0)
                        if c_count >= max_per_model:
                            continue
                        seen_models[m] = c_count + 1
                    count += 1
                    q_out.append(row_q)
                    rank_out.append(count)
                    c_out.append(cand_idx)
                    score_out.append(val)
                    if count >= k:
                        break

                # Fallback pass if diversity was too restrictive to reach MIN_K
                if count < min(k, len(pool) - 1):
                    for col_j in row_order:
                        val = s[row_i, col_j]
                        if not np.isfinite(val):
                            break
                        cand_idx = pool[col_j]
                        if count > 0 and cand_idx in set(c_out[-count:]):
                            continue
                        count += 1
                        q_out.append(row_q)
                        rank_out.append(count)
                        c_out.append(cand_idx)
                        score_out.append(val)
                        if count >= k:
                            break

            if q_out:
                frames.append(pd.DataFrame({
                    "query": items[KEY].to_numpy()[q_out],
                    "rank": rank_out,
                    "candidate": items[KEY].to_numpy()[c_out],
                    "score": score_out,
                    "_q": q_out,
                    "_c": c_out
                }))

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["query", "rank", "candidate",
                                                                                    "score", "_q", "_c"])
    parts = pair_breakdown(blocks, has, weights, out["_q"].to_numpy(int), out["_c"].to_numpy(int))
    out = out.assign(**{n: v.astype(np.float32) for n, v in parts.items()})
    return out.drop(columns=["_q", "_c"]).sort_values(["query", "rank"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------- outputs
def save(nb: pd.DataFrame, settings: dict, source: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    nb.to_parquet(PROCESSED_DIR / OUT_FILE, index=False)
    meta = {**settings, "source": source, "rows": len(nb), "queries": int(nb["query"].nunique()),
            "columns": list(nb.columns), "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (PROCESSED_DIR / "neighbours.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def summary_tables(items: pd.DataFrame, nb: pd.DataFrame, blocks: dict) -> dict[str, pd.DataFrame]:
    """Coverage, block shares and attribute agreement of the top 4, per category."""
    it = items.set_index(KEY)
    top = nb[nb["rank"] <= MIN_K].copy()
    top["CAT"] = it.loc[top["query"], "CAT_DES_EN"].to_numpy()
    # sub-family names repeat across families ("Others"): compare the family › sub-family path
    paths = {"GFA_DES_EN": it["GFA_DES_EN"].astype("string"),
             "GFS_DES_EN": it["GFA_DES_EN"].astype("string") + " › " + it["GFS_DES_EN"].astype("string"),
             "CLR_DES": it["CLR_DES"].astype("string")}
    for col, v in paths.items():
        a, b = v.loc[top["query"]].to_numpy(object, na_value=None), v.loc[top["candidate"]].to_numpy(object, na_value=None)
        top[f"same_{col}"] = pd.Series(a == b).where(pd.notna(a) & pd.notna(b)).astype(float).to_numpy()
    pa, pb = it.loc[top["query"], "PRICE_BASE_W_VAT"].to_numpy(), it.loc[top["candidate"], "PRICE_BASE_W_VAT"].to_numpy()
    top["price_ratio"] = np.maximum(pa, pb) / np.minimum(pa, pb)
    n_found = nb.groupby("query").size().reindex(items[KEY], fill_value=0)
    cov = pd.DataFrame({"CAT": items["CAT_DES_EN"].to_numpy(), "found": n_found.to_numpy(),
                        "photo": items["img_trusted"].fillna(False).to_numpy(bool)})
    per_cat = cov.groupby("CAT").agg(items=("found", "size"), with_photo=("photo", "sum"),
                                     fewer_than_4=("found", lambda s: int((s < MIN_K).sum())))
    agree = top.groupby("CAT").agg(score_rank1=("score", lambda s: s[top.loc[s.index, "rank"] == 1].mean()),
                                   score_rank4=("score", lambda s: s[top.loc[s.index, "rank"] == MIN_K].mean()),
                                   same_family=("same_GFA_DES_EN", "mean"), same_subfamily=("same_GFS_DES_EN", "mean"),
                                   same_colour=("same_CLR_DES", "mean"), price_ratio=("price_ratio", "median"))
    contrib = [c for c in top.columns if c.startswith("contrib_")]
    shares = top.groupby("CAT")[contrib].sum()
    shares = shares.div(shares.sum(axis=1), axis=0).rename(columns=lambda c: c.removeprefix("contrib_"))
    photo = top.assign(photo=it.loc[top["query"], "img_trusted"].fillna(False).to_numpy(bool))
    by_photo = photo.groupby("photo").agg(queries=("query", "nunique"), mean_score=("score", "mean"),
                                          same_subfamily=("same_GFS_DES_EN", "mean"), same_colour=("same_CLR_DES", "mean"))
    return {"coverage": per_cat, "agreement": agree, "shares": shares, "by_photo": by_photo}


def contact_sheet(items: pd.DataFrame, nb: pd.DataFrame, queries: list[str], path: Path, k: int = MIN_K,
                  cell: int = 200, text: int = 58) -> None:
    """One row per query: the query photo, then its top-k with score, sub-family and colour."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        bold = ImageFont.truetype("arialbd.ttf", 12)
    except OSError:
        font = bold = ImageFont.load_default()
    it = items.set_index(KEY)
    sheet = Image.new("RGB", ((k + 1) * cell + 12, len(queries) * (cell + text)), "white")
    d = ImageDraw.Draw(sheet)
    for r, q in enumerate(queries):
        row = [(q, None)] + list(nb.loc[nb["query"] == q, ["candidate", "score"]].head(k).itertuples(index=False))
        for c, (key, score) in enumerate(row):
            x, y = c * cell + (12 if c else 0), r * (cell + text)
            item = it.loc[key]
            if bool(item["img_trusted"]):
                try:
                    im = Image.open(IMG_DIR / item["img_file"]).convert("RGB")
                    im.thumbnail((cell - 8, cell - 8))
                    sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
                except OSError:
                    d.text((x + 8, y + 90), "photo unreadable", fill="red", font=font)
            else:
                d.text((x + 8, y + 90), "no trusted photo", fill="#999", font=font)
            head = f"QUERY {key}" if score is None else f"{c}. {key}  {score:.2f}"
            d.text((x + 4, y + cell), head, fill="#b00" if score is None else "black", font=bold)
            d.text((x + 4, y + cell + 15), textwrap.shorten(f"{item['GFS_DES_EN']}", 30), fill="black", font=font)
            d.text((x + 4, y + cell + 30), textwrap.shorten(f"{item['CLR_DES']} · {item['PRICE_BASE_W_VAT']:.2f} €", 30),
                   fill="#333", font=font)
            d.rectangle([x, y, x + cell - 1, y + cell + text - 1], outline="#bbb")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def example_queries(items: pd.DataFrame, n_photo: int = 10, n_tabular: int = 2, seed: int = 0) -> list[str]:
    """Random queries (fixed seed): one or two per category with a photo, plus a few without one."""
    rng = np.random.default_rng(seed)
    photo = items.loc[items["img_trusted"].fillna(False).astype(bool), KEY].to_numpy()
    none = items.loc[~items["img_trusted"].fillna(False).astype(bool), KEY].to_numpy()
    picked = list(rng.choice(photo, size=min(n_photo, len(photo)), replace=False))
    picked += list(rng.choice(none, size=min(n_tabular, len(none)), replace=False))
    return sorted(picked, key=lambda q: (items.set_index(KEY).loc[q, "CAT_DES_EN"], q))


def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 2) -> str:
    def fmt(col, v):
        if pd.isna(v):
            return "–"
        if col in pct:
            return f"{v:.0%}"
        return f"{v:,}" if isinstance(v, (int, np.integer)) else f"{v:.{digits}f}"
    head = "| " + " | ".join([df.index.name or ""] + list(df.columns)) + " |"
    rows = ["| " + " | ".join([str(i)] + [fmt(c, df.at[i, c]) for c in df.columns]) + " |" for i in df.index]
    return "\n".join([head, "|" + "---|" * (len(df.columns) + 1)] + rows)


def write_report(items: pd.DataFrame, nb: pd.DataFrame, blocks: dict, settings: dict, source: str,
                 queries: list[str]) -> None:
    t = summary_tables(items, nb, blocks)
    has = {b: int(v.sum()) for b, v in has_block(blocks).items()}
    wrows = "\n".join(f"| `{b}` | {w:.2f} | {has.get(b, 0):,} |" for b, w in settings["weights"].items())
    agree = t["agreement"].rename(columns={"price_ratio": "median price ratio"})
    by_photo = t["by_photo"].rename(index={True: "trusted photo", False: "tabular only"})
    by_photo.index.name = "query"
    text = f"""# Phase 3: similarity engine report

Generated by `src/phase3_engine.py build` from `{source}.parquet` ({len(items):,} colourways).
Output: `data/processed/neighbours.parquet` ({len(nb):,} rows: query, rank, candidate, score, `sim_<block>`,
`contrib_<block>`) and its settings in `neighbours.json`.

## Settings

- Top-k: **{settings['k']}** per query (the brief asks for at least {MIN_K}).
- Candidates: same {' + '.join(SCOPES[settings['scope']])}; the query itself and
  {'other colours of the same model are allowed' if settings['include_siblings'] else 'other colours of the same model (`PROD_REF` / `PROD_REF_EQUIV`) are excluded'}.
- Weights (`src/phase3_weights.json`, starting values for Phase 4). A pair is scored on the blocks both items
  have, and the weights are rescaled to sum to 1 over those blocks.

| Block | Weight | Items with the block |
|---|---|---|
{wrows}

The `image` block only uses photos with `img_trusted`, the `palette` block only photos with `img_colour_trusted`.

## Coverage

{_md(t['coverage'])}

{int((nb.loc[nb['rank'] == 1, 'score'] >= 0.99).sum())} queries have a best match scoring ≥ 0.99: almost always the
same design under another model code (e.g. a size or pack variant). They are kept, because they are different
models, but Phase 4 should check whether buyers see them as duplicates.

## Top {MIN_K}: scores and agreement with the query

Share of the top {MIN_K} with the same family, sub-family and colour (`CLR_DES`) as the query, and the median ratio
of the higher to the lower price (1.00 = same price). A first sanity check; Phase 4 evaluates properly.

{_md(agree, pct={'same_family', 'same_subfamily', 'same_colour'})}

{_md(by_photo, pct={'same_subfamily', 'same_colour'})}

## What drives the score (share of the top-{MIN_K} scores per block)

{_md(t['shares'], pct=set(t['shares'].columns))}

## Examples

`outputs/phase3/examples.jpg`: {len(queries)} random queries (seed 0), each followed by its top {MIN_K}.
The last ones have no trusted photo and are matched on the tabular blocks only.
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "engine_report.md").write_text(text, encoding="utf-8")


def items_source() -> str:
    return "items_clean" if (phase1_clean.PROCESSED_DIR / phase1_clean.CLEAN_FILE).exists() else "items_preview"


# --------------------------------------------------------------------------- CLI
def run(args: argparse.Namespace, queries: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    items = phase1_clean.load_items().sort_values(KEY, kind="stable").reset_index(drop=True)
    weights = load_weights(Path(args.weights))
    blocks = load_blocks(items)
    unknown = set(weights) - set(blocks)
    if unknown:
        raise SystemExit(f"weights for unknown blocks: {sorted(unknown)} (available: {sorted(blocks)})")
    two_stage = getattr(args, "two_stage", True)
    stage1_pool = getattr(args, "stage1_pool", 50)
    max_per_model = getattr(args, "max_per_model", 1)
    settings = {
        "k": args.k, "scope": args.scope, "include_siblings": args.include_siblings,
        "two_stage": bool(two_stage), "stage1_pool": int(stage1_pool), "max_per_model": int(max_per_model),
        "weights": weights
    }
    nb = neighbours(items, blocks, weights, args.k, args.scope, args.include_siblings, queries,
                    two_stage=two_stage, stage1_pool=stage1_pool, max_per_model=max_per_model)
    return items, nb, blocks, settings


def cmd_build(args: argparse.Namespace) -> None:
    items, nb, blocks, settings = run(args)
    source = items_source()
    save(nb, settings, source)
    queries = example_queries(items)
    contact_sheet(items, nb, queries, REPORT_DIR / "examples.jpg")
    write_report(items, nb, blocks, settings, source, queries)
    few = (nb.groupby("query").size().reindex(items[KEY], fill_value=0) < MIN_K).sum()
    print(f"{nb['query'].nunique():,} queries, {len(nb):,} rows -> {PROCESSED_DIR / OUT_FILE}; "
          f"{few} items with fewer than {MIN_K} neighbours; report outputs/phase3/engine_report.md")


def cmd_query(args: argparse.Namespace) -> None:
    items, nb, _, _ = run(args, [args.key])
    if nb.empty:
        raise SystemExit(f"{args.key}: unknown colourway or no candidates")
    it = items.set_index(KEY)
    show = nb.assign(subfamily=it.loc[nb["candidate"], "GFS_DES_EN"].to_numpy(),
                     colour=it.loc[nb["candidate"], "CLR_DES"].to_numpy(),
                     price=it.loc[nb["candidate"], "PRICE_BASE_W_VAT"].to_numpy())
    q = it.loc[args.key]
    print(f"{args.key}: {q['GFS_DES_EN']}, {q['CLR_DES']}, {q['PRICE_BASE_W_VAT']:.2f} €, photo: {bool(q['img_trusted'])}")
    cols = ["rank", "candidate", "score", "subfamily", "colour", "price"] + [c for c in nb if c.startswith("contrib_")]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(show[cols].rename(columns=lambda c: c.removeprefix("contrib_")).round(3).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scope", choices=list(SCOPES), default="category", help="candidate pool")
    common.add_argument("--include-siblings", action="store_true", help="allow other colours of the same model")
    common.add_argument("--weights", default=str(WEIGHTS_FILE), help="JSON file with the block weights")
    common.add_argument("--two-stage", action=argparse.BooleanOptionalAction, default=True,
                        help="two-stage retrieval with visual candidate filtering (default: True)")
    common.add_argument("--stage1-pool", type=int, default=50,
                        help="stage 1 pool size for visual candidate filtering (default: 50)")
    common.add_argument("--max-per-model", type=int, default=1,
                        help="maximum colours of one model in top-k (default: 1; 0 for unlimited)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", parents=[common], help="top-k for every colourway")
    b.add_argument("--k", type=int, default=10)
    b.set_defaults(fn=cmd_build)
    q = sub.add_parser("query", parents=[common], help="top-k for one colourway")
    q.add_argument("key")
    q.add_argument("--k", type=int, default=5)
    q.set_defaults(fn=cmd_query)
    args = p.parse_args()
    if args.k < MIN_K:
        p.error(f"--k must be at least {MIN_K} (the brief)")
    args.fn(args)


if __name__ == "__main__":
    main()
