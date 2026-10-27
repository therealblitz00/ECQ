"""Phase 5: an explanation for every similar product, computed from the engine's own score (no model, no LLM).

    python src/phase5_explain.py build            # -> data/processed/explanations.parquet, outputs/phase5/
    python src/phase5_explain.py show 167718_BU   # the explained top 5 of one product, printed

The engine score is a weighted mean of block similarities, and every block is a dot product of unit-length rows
split into named attribute slices. So the explanation is an exact decomposition, not an approximation:

    score = sum over blocks of contrib_b                      (stored by Phase 3)
    sim_categorical = sum over attributes of  q[slice] . c[slice]   (family, sub-family, finish, colour, ...)
    sim_composition = sum over materials  of  q[m] * c[m]           (shared materials, rare ones weigh more)
    sim_numeric     = price part + launch part + season part

Each pair gets: the share of the score from each block, the shared attributes with their exact contribution, the
main differences (price, colour with the measured closeness of the two colour names, finishing, sub-family), the
shared dominant colours of the photos and a one-line summary. The output follows the brief: for each product, at
least 4 similar products and the most relevant characteristics.

Quality checks written to the report: completeness (the parts add up to the score), faithfulness (removing the
block the explanation names first moves the match more than removing another block), agreement with the team's
judgements (what separates the candidates they called similar), and the cost per explanation.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
import phase3_engine as engine  # noqa: E402
from config import EMB_DIR, KEY, PROCESSED_DIR, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase5"
OUT_FILE = "explanations.parquet"
TOP = 5                       # similar products explained per query (the brief asks for at least 4)
BLOCK_LABEL = {"image": "photo", "categorical": "catalogue attributes", "text": "description",
               "composition": "materials", "palette": "photo colours", "numeric": "price and launch"}
ATTR_LABEL = {"CAT": "category", "GFA": "family", "GFS": "sub-family", "CATEGORY_MATRIX": "planning category",
              "CLR_DES": "colour", "CLR_TYPE": "colour type", "FINISHING": "finish", "FINISH_FAMILY": "metal",
              "MATERIAL": "main fabric", "PRINT_TYPE": "print", "DIMENSION": "size type", "OUTFIT": "outfit",
              "THEME": "theme", "FASHIONTYPE": "fashion type", "PROD_SEG": "segment", "NUMBER_OF_UNITS": "units"}
SKIP_SHARED = {"CAT"}         # always shared (candidates come from the same category): not worth saying
CLOSE_COLOUR = 0.5            # two colour names at least this similar (measured on photos) are called "close"
SAME_PRICE = 0.05             # within 5% counts as the same price


# --------------------------------------------------------------------------- inputs
class Context:
    """Everything needed to explain pairs, loaded once."""

    def __init__(self, items: pd.DataFrame | None = None, blocks: dict | None = None):
        self.items = phase1_clean.load_items() if items is None else items
        self.blocks = engine.load_blocks(self.items) if blocks is None else blocks
        self.has = engine.has_block(self.blocks)
        self.weights = engine.load_weights()
        self.meta = json.loads((PROCESSED_DIR / "feature_meta.json").read_text(encoding="utf-8"))
        self.pos = pd.Series(np.arange(len(self.items)), index=self.items[KEY])
        self.attrs = {b: self._slices(b) for b in ("categorical", "composition", "numeric")}
        self.colour_sim = self._colour_similarity()
        self.palette_names = self._palette_names()
        self.it = self.items.set_index(KEY)

    def _slices(self, block: str) -> dict[str, tuple[int, int]]:
        meta = self.meta["blocks"].get(block, {})
        attrs = meta.get("attributes", {})
        if isinstance(attrs, dict) and attrs and isinstance(next(iter(attrs.values())), dict):
            return {a: (v["start"], v["stop"]) for a, v in attrs.items()}
        cols = meta.get("columns", [])
        if block == "composition":
            return {c.split("=", 1)[1]: (i, i + 1) for i, c in enumerate(cols)}
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(cols):
            groups.setdefault(c.split("≈")[0].split("=")[0].replace("_pct", ""), []).append(i)
        return {g: (min(ix), max(ix) + 1) for g, ix in groups.items()}

    def _colour_similarity(self) -> dict[tuple[str, str], float]:
        path = PROCESSED_DIR / "colour_reference.parquet"
        if not path.exists():
            return {}
        ref = pd.read_parquet(path).dropna(subset=["L"])
        cfg = json.loads((PROCESSED_DIR / "colour_reference.json").read_text(encoding="utf-8"))
        tau, lw = float(cfg.get("tau", 8.0)), float(cfg.get("l_weight", 0.5))
        lab = ref[["L", "a", "b"]].to_numpy(float) * np.array([lw, 1, 1])
        sim = np.exp(-((lab[:, None] - lab[None]) ** 2).sum(-1) / (2 * tau ** 2))
        names = ref["CLR_DES"].tolist()
        return {(a, b): float(sim[i, j]) for i, a in enumerate(names) for j, b in enumerate(names) if i != j}

    def _palette_names(self) -> dict[str, list[str]]:
        path = EMB_DIR / "palettes.parquet"
        if not path.exists():
            return {}
        pal = pd.read_parquet(path)
        trusted = set(self.items.loc[self.items["img_colour_trusted"].fillna(False).astype(bool), KEY])
        out = {}
        for k, names, shares in zip(pal[KEY], pal["names"], pal["shares"]):
            if k in trusted:
                out[k] = list(dict.fromkeys(n for n, s in zip(names, shares) if s >= 0.15))
        return out


# --------------------------------------------------------------------------- one pair
def slice_parts(ctx: Context, block: str, qi: int, ci: int) -> dict[str, float]:
    """Exact split of sim_block(q, c) over the block's attribute slices (they add up to the similarity)."""
    m = ctx.blocks[block]
    out = {}
    for a, (s, e) in ctx.attrs[block].items():
        v = float(m[qi, s:e] @ m[ci, s:e])
        if abs(v) > 1e-9:
            out[a] = v
    return out


def _value(ctx: Context, key: str, attr: str):
    src = ctx.meta["blocks"]["categorical"]["attributes"].get(attr, {}).get("source", [attr])
    v = ctx.it.at[key, src[-1]] if src[-1] in ctx.it.columns else None
    return None if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v) else v


def explain_pair(ctx: Context, row) -> dict:
    """Explanation of one neighbour row (query, candidate, score, sim_<b>, contrib_<b>)."""
    q, c = row["query"], row["candidate"]
    qi, ci = int(ctx.pos[q]), int(ctx.pos[c])
    score = float(row["score"])
    contrib = {b: float(row.get(f"contrib_{b}", 0.0) or 0.0) for b in ctx.weights}
    contrib = {b: v for b, v in contrib.items() if v > 0}
    shares = {b: v / score for b, v in contrib.items()} if score > 0 else {}

    # categorical: exact contribution of every shared attribute to the TOTAL score
    shared = []
    if "categorical" in contrib and row.get("sim_categorical", 0) > 0:
        parts = slice_parts(ctx, "categorical", qi, ci)
        scale = contrib["categorical"] / float(row["sim_categorical"])
        for a, v in sorted(parts.items(), key=lambda kv: -kv[1]):
            if a not in SKIP_SHARED:
                shared.append({"attribute": ATTR_LABEL.get(a, a), "value": _value(ctx, q, a), "points": v * scale})
    materials = []
    if "composition" in contrib and row.get("sim_composition", 0) > 0:
        parts = slice_parts(ctx, "composition", qi, ci)
        scale = contrib["composition"] / float(row["sim_composition"])
        materials = [{"material": m, "points": v * scale} for m, v in sorted(parts.items(), key=lambda kv: -kv[1])]
    numeric = {}
    if "numeric" in contrib and row.get("sim_numeric", 0) > 0:
        parts = slice_parts(ctx, "numeric", qi, ci)
        scale = contrib["numeric"] / float(row["sim_numeric"])
        numeric = {a: v * scale for a, v in parts.items()}

    # differences a buyer checks first
    a, b = ctx.it.loc[q], ctx.it.loc[c]
    diffs = []
    pq, pc = float(a["PRICE_BASE_W_VAT"]), float(b["PRICE_BASE_W_VAT"])
    if abs(pc - pq) > SAME_PRICE * pq:
        diffs.append(f"price {pc:.2f} € vs {pq:.2f} €")
    cq, cc = a["CLR_DES"], b["CLR_DES"]
    colour_note = "same colour"
    if cq != cc:
        s = ctx.colour_sim.get((cq, cc))
        colour_note = f"close colours: {cc} ≈ {cq} ({s:.2f})" if s is not None and s >= CLOSE_COLOUR \
            else f"different colour: {cc} vs {cq}"
        if not colour_note.startswith("close"):
            diffs.append(colour_note)
    for col, label in [("GFS_DES_EN", "sub-family"), ("FINISHING", "finish"), ("MATERIAL", "main fabric")]:
        vq, vc = a.get(col), b.get(col)
        if pd.notna(vq) and pd.notna(vc) and vq != vc:
            diffs.append(f"{label} {vc} vs {vq}")
    pal_q, pal_c = ctx.palette_names.get(q, []), ctx.palette_names.get(c, [])
    shared_colours = [n for n in pal_c if n in pal_q]

    top_blocks = sorted(shares.items(), key=lambda kv: -kv[1])
    lead = ", ".join(f"{BLOCK_LABEL[b]} {s:.0%}" for b, s in top_blocks[:3])
    facts = [f"same {s['attribute']}: {s['value']}" for s in shared[:3] if s["value"] is not None]
    if materials:
        facts.append("shared materials: " + ", ".join(m["material"] for m in materials[:3]))
    if colour_note.startswith("close") or (colour_note == "same colour" and not any(
            s["attribute"] == "colour" for s in shared[:3])):
        facts.append(colour_note if colour_note.startswith("close") else f"same colour: {cq}")
    if shared_colours:
        facts.append("photo colours: " + ", ".join(shared_colours[:3]))
    summary = f"{lead}. " + "; ".join(facts) + (". Differs in " + "; ".join(diffs[:2]) + "." if diffs else ".")
    return {"query": q, "rank": int(row["rank"]), "candidate": c, "score": score,
            **{f"share_{b}": shares.get(b, 0.0) for b in ctx.weights},
            "shared_attributes": json.dumps(shared, default=str), "shared_materials": json.dumps(materials),
            "numeric_parts": json.dumps(numeric), "differences": json.dumps(diffs, ensure_ascii=False),
            "shared_photo_colours": "|".join(shared_colours), "summary": summary,
            "completeness_error": abs(sum(contrib.values()) - score)}


def explain(ctx: Context, nb: pd.DataFrame, top: int = TOP) -> pd.DataFrame:
    rows = nb[nb["rank"] <= top]
    return pd.DataFrame([explain_pair(ctx, r) for r in rows.to_dict("records")])


# --------------------------------------------------------------------------- quality checks
def completeness(ctx: Context, nb: pd.DataFrame, n: int = 2000, seed: int = 0) -> dict:
    """Do the parts add up? Blocks to the score; categorical slices to sim_categorical (sampled pairs)."""
    s = nb.sample(min(n, len(nb)), random_state=seed)
    contrib_cols = [f"contrib_{b}" for b in ctx.weights if f"contrib_{b}" in nb]
    block_err = float((s[contrib_cols].fillna(0).sum(axis=1) - s["score"]).abs().max())
    cat_err = 0.0
    for r in s.to_dict("records"):
        if pd.notna(r.get("sim_categorical")):
            parts = slice_parts(ctx, "categorical", int(ctx.pos[r["query"]]), int(ctx.pos[r["candidate"]]))
            cat_err = max(cat_err, abs(sum(parts.values()) - r["sim_categorical"]))
    return {"max_error_blocks": block_err, "max_error_attributes": cat_err, "pairs": len(s)}


def faithfulness(ctx: Context, nb: pd.DataFrame, n_queries: int = 300, seed: int = 0) -> pd.DataFrame:
    """Deletion test: drop one block (weight 0) and see how far the explained match falls in its query's ranking.

    If the explanation is faithful, dropping the block it names first moves the match more than dropping the block
    it names last, or a random block."""
    rng = np.random.default_rng(seed)
    queries = rng.choice(nb["query"].unique(), size=min(n_queries, nb["query"].nunique()), replace=False)
    first = nb[(nb["rank"] == 1) & nb["query"].isin(queries)]
    blocks = [b for b, w in ctx.weights.items() if w > 0]
    rows = []
    groups = {tuple(sorted(k)) if isinstance(k, tuple) else k: v for k, v in
              ctx.items.groupby("CAT_DES_EN", sort=True).indices.items()}
    cat_of = ctx.items["CAT_DES_EN"].to_numpy()
    for r in first.to_dict("records"):
        qi, ci = int(ctx.pos[r["query"]]), int(ctx.pos[r["candidate"]])
        pool = np.asarray(groups[cat_of[qi]])
        pool = pool[~engine.excluded(ctx.items, np.array([qi]), pool, include_siblings=False)[0]]
        present = {b: r.get(f"contrib_{b}", 0) or 0 for b in blocks}
        ranked = [b for b, v in sorted(present.items(), key=lambda kv: -kv[1]) if v > 0]
        if len(ranked) < 2:
            continue
        picks = {"named first": ranked[0], "named last": ranked[-1], "random": ranked[rng.integers(len(ranked))]}
        for label, b in picks.items():
            w = {**ctx.weights, b: 0.0}
            s = engine.score_matrix(ctx.blocks, ctx.has, w, np.array([qi]), pool)[0]
            new_rank = int((s > s[pool == ci][0]).sum()) + 1
            rows.append({"removed": label, "block": b, "new_rank": new_rank})
    df = pd.DataFrame(rows)
    return df.groupby("removed")["new_rank"].agg(["mean", "median", lambda x: (x > 1).mean()]).rename(
        columns={"<lambda_0>": "share no longer first"})


def human_alignment(ctx: Context) -> pd.DataFrame | None:
    """Mean block shares of the candidates the team called similar vs not similar (feedback pools)."""
    pool_path = ROOT / "outputs" / "phase4" / "feedback_pool.json"
    if not pool_path.exists():
        return None
    import phase4_feedback as fb
    pool, recs = fb.load_pool(), fb.read_feedback()
    rows = []
    for r in recs:
        for label, keys in (("similar", r["similar"]), ("not similar", r["not"])):
            for c in keys:
                qi, ci = int(ctx.pos[r["query"]]), int(ctx.pos[c])
                br = engine.pair_breakdown(ctx.blocks, ctx.has, ctx.weights, np.array([qi]), np.array([ci]))
                total = sum(float(br[f"contrib_{b}"][0]) for b in ctx.weights if f"contrib_{b}" in br)
                rows.append({"verdict": label, "score": total,
                             **{BLOCK_LABEL[b]: float(br[f"sim_{b}"][0]) for b in ctx.weights
                                if f"sim_{b}" in br and np.isfinite(br[f"sim_{b}"][0])}})
    del pool
    df = pd.DataFrame(rows)
    out = df.groupby("verdict").mean().T
    out["difference"] = out["similar"] - out["not similar"]
    sd = df.drop(columns="verdict").std()
    out["difference / sd"] = out["difference"] / sd
    return out.sort_values("difference / sd", ascending=False)


# --------------------------------------------------------------------------- outputs
def brief_export(ctx: Context, ex: pd.DataFrame) -> pd.DataFrame:
    """The brief's format: query, similar product, rank, score and the most relevant characteristics."""
    it = ctx.it
    out = ex[["query", "rank", "candidate", "score", "summary"]].copy()
    out["query_description"] = it.loc[out["query"], "PROD_DES_BASE"].to_numpy()
    out["candidate_description"] = it.loc[out["candidate"], "PROD_DES_BASE"].to_numpy()
    out["score"] = out["score"].round(3)
    return out[["query", "query_description", "rank", "candidate", "candidate_description", "score", "summary"]]


def write_report(comp: dict, faith: pd.DataFrame, human: pd.DataFrame | None, timing: dict, ex: pd.DataFrame,
                 examples: list[str]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    share_cols = [c for c in ex.columns if c.startswith("share_")]
    mean_shares = ex.groupby("rank")[share_cols].mean().round(3)
    mean_shares.columns = [BLOCK_LABEL[c.replace("share_", "")] for c in share_cols]

    def md(df: pd.DataFrame, digits: int = 3) -> str:
        cols = list(df.columns)
        lines = ["| | " + " | ".join(map(str, cols)) + " |", "|" + "---|" * (len(cols) + 1)]
        for i, r in df.iterrows():
            lines.append(f"| {i} | " + " | ".join(f"{v:.{digits}f}" if isinstance(v, float) else str(v)
                                                     for v in r.to_numpy()) + " |")
        return "\n".join(lines)

    ex_lines = []
    for q in examples:
        ex_lines.append(f"**{q}**")
        for r in ex[ex["query"] == q].itertuples():
            ex_lines.append(f"{r.rank}. `{r.candidate}` ({r.score:.3f}): {r.summary}")
        ex_lines.append("")
    lines = [
        "# Phase 5: explanations of the similar products",
        "",
        f"Generated by `src/phase5_explain.py build` ({datetime.now(timezone.utc):%Y-%m-%d}): "
        f"{ex['query'].nunique():,} products × top {TOP} = {len(ex):,} explained matches.",
        "",
        "## How an explanation is computed",
        "",
        "The engine score is a weighted mean of block similarities, and each block is a dot product of unit-length "
        "rows split into named slices. So every explanation is an **exact decomposition** of the score (no "
        "surrogate model, no sampling, no LLM): the share of each block, then inside the blocks the shared "
        "attributes, the shared materials (rare materials weigh more) and the price / launch parts. Differences "
        "(price, colour with the measured closeness of the two colour names, finish, sub-family) and the shared "
        "dominant colours of the photos are added from the data.",
        "",
        "## Checks",
        "",
        f"- **Completeness** ({comp['pairs']:,} sampled pairs): the block contributions add up to the score up to "
        f"{comp['max_error_blocks']:.1e}; the attribute parts add up to the categorical similarity up to "
        f"{comp['max_error_attributes']:.1e} (float32 rounding).",
        f"- **Cost**: {timing['ms_per_pair']:.2f} ms per explained match on a laptop CPU "
        f"({timing['pairs']:,} matches in {timing['seconds']:.0f} s), so every product is explained in the batch run.",
        "- **Faithfulness (deletion test)**: for the best match of random queries, one block is removed (weight 0) "
        "and the match's new rank is measured. Removing the block the explanation names **first** should move the "
        "match most:",
        "",
        md(faith, 2),
        "",
    ]
    if human is not None:
        lines += ["- **Agreement with the team** (candidates sorted in the Phase 4 feedback app): mean block similarity "
                  "of the candidates the team called *similar* vs *not similar*. A large standardised difference "
                  "means the block separates what people see as similar, so it deserves its place in the "
                  "explanation:", "", md(human, 3), ""]
    lines += ["## Mean share of the score per block, by rank", "", md(mean_shares, 3), "", "## Examples", "",
              *ex_lines,
              "## Files", "",
              "- `data/processed/explanations.parquet`: every explained match (shares, shared attributes with points, "
              "materials, numeric parts, differences, summary).",
              "- `outputs/phase5/similar_products.csv`: the brief's format for every product (top 5, with the summary).",
              "- `outputs/phase5/brief_example.json`: the brief's example product, fully explained.", ""]
    (REPORT_DIR / "explain_report.md").write_text("\n".join(lines), encoding="utf-8")


def cmd_build(args: argparse.Namespace) -> None:
    ctx = Context()
    nb = pd.read_parquet(PROCESSED_DIR / "neighbours.parquet")
    t0 = time.time()
    ex = explain(ctx, nb, TOP)
    secs = time.time() - t0
    ex.to_parquet(PROCESSED_DIR / OUT_FILE, index=False)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    brief_export(ctx, ex).to_csv(REPORT_DIR / "similar_products.csv", index=False, encoding="utf-8-sig")
    q = "167718_BU"
    if q in set(ex["query"]):
        rec = ex[ex["query"] == q].to_dict("records")
        for r in rec:
            for k in ("shared_attributes", "shared_materials", "numeric_parts", "differences"):
                r[k] = json.loads(r[k])
        (REPORT_DIR / "brief_example.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False, default=float),
                                                        encoding="utf-8")
    comp = completeness(ctx, nb)
    faith = faithfulness(ctx, nb)
    human = human_alignment(ctx)
    examples = [q] + list(np.random.default_rng(0).choice(sorted(ex["query"].unique()), 3, replace=False))
    write_report(comp, faith, human, {"ms_per_pair": 1000 * secs / len(ex), "pairs": len(ex), "seconds": secs},
                 ex, examples)
    print(f"{len(ex):,} matches explained in {secs:.0f} s -> {PROCESSED_DIR / OUT_FILE}; report {REPORT_DIR}")
    print(faith.round(2).to_string())


def cmd_show(args: argparse.Namespace) -> None:
    ctx = Context()
    nb = pd.read_parquet(PROCESSED_DIR / "neighbours.parquet")
    ex = explain(ctx, nb[nb["query"] == args.key], args.k)
    for r in ex.itertuples():
        print(f"{r.rank}. {r.candidate}  {r.score:.3f}  {r.summary}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("show")
    s.add_argument("key")
    s.add_argument("--k", type=int, default=TOP)
    args = ap.parse_args()
    {"build": cmd_build, "show": cmd_show}[args.cmd](args)


if __name__ == "__main__":
    main()
