"""Phase 4: evaluation of the similarity engine. One fixed script, so every change reports the same numbers.

    python src/phase4_eval.py run            # -> outputs/phase4/eval_report.md + eval_metrics.csv + weight_search.csv
    python src/phase4_eval.py run --apply    # also writes the chosen weights to src/phase3_weights.json
    python src/phase4_eval.py sheet          # -> outputs/phase4/human_review.csv + contact sheets for the team

There is no ground truth, so three families of signals are combined:

- catalogue: `hit@k` (share of queries with a "known similar" item in the top k: same family › sub-family, `THEME`
  and `CLR_DES`, another model) and the share of the top 4 that agree with the query on sub-family, colour,
  finishing and material.
- visual: the share of the top 4 among the query's 20 nearest photos (same category, other models).
- commercial: Spearman correlation between the query's sales percentile in its category and the mean of its top 4
  (sales are not used by the engine; launch date is, and it drives sales).

Each family is circular for one block (catalogue rewards the categorical block, visual the image block), so the
proxies cannot say how much the photo should weigh against the attributes. Therefore:

1. Automatic tuning only keeps a change that raises the mean of the three families without lowering any single
   metric by more than TOLERANCE (no trade-off is made blindly, not even inside a family).
2. The trade-off itself (the image weight) is shown as a sweep, and decided by the human review: the `sheet` pools
   the top 5 of three image weights, blind and in key order, and `run` scores every configuration on the verdicts.

Weights are tuned on half of the models (fixed split by `PROD_REF`); every number in the report is measured on the
other half. `sheet` never overwrites a sheet that already has verdicts.
"""
from __future__ import annotations

import argparse
import json
import string
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
import phase3_engine as engine  # noqa: E402
from config import IMG_DIR, KEY, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase4"
SHEET_FILE = REPORT_DIR / "human_review.csv"
SHEET_META = REPORT_DIR / "human_review.json"
K = engine.MIN_K
VISUAL_POOL = 20         # a candidate is a "visual neighbour" if it is among the query's 20 nearest photos
SEED = 0
VERDICTS = {"similar": 1.0, "partly": 0.5, "not": 0.0}
SHEET_PER_CAT = (8, 2)   # queries per category with / without a trusted photo
SHEET_TOP = 5
IMAGE_SWEEP = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
SHEET_IMAGE = (0.0, 0.3, 0.6)  # image weights whose top 5 go into the human sheet
SEARCH_FACTORS = (0.0, 0.5, 1.5, 2.0)  # coordinate search: tried multipliers of each block weight
MIN_GAIN = 0.005         # the mean of the families must rise at least this much
TOLERANCE = 0.01         # and no guarded metric may drop more than this
MIN_COVERAGE = 0.9       # share of a configuration's top 5 judged by the team before its human score counts
FAMILIES = ["catalogue", "visual", "commercial"]
GUARDED = ["hit@4", "same_subfamily", "same_colour", "same_finishing", "same_material", "visual@4", "sales_corr"]


# --------------------------------------------------------------------------- ground signals
def split_models(items: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    """True for the tuning half: models (PROD_REF) shuffled with a fixed seed, the first half."""
    refs = np.sort(items["PROD_REF"].astype(str).unique())
    tune = set(np.random.default_rng(seed).permutation(refs)[: len(refs) // 2])
    return items["PROD_REF"].astype(str).isin(tune).to_numpy()


def known_groups(items: pd.DataFrame) -> pd.Series:
    """Group id of the "known similar" items: same category › family › sub-family, theme and colour.

    -1 when an attribute is missing or no other model shares the group."""
    cols = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN", "THEME", "CLR_DES"]
    full = items[cols].notna().all(axis=1)
    gid = pd.Series(-1, index=items.index)
    gid[full] = items.loc[full].groupby(cols, sort=True).ngroup()
    models = items.loc[full].groupby(gid[full])["PROD_REF"].transform("nunique")
    gid[full & models.reindex(items.index).lt(2)] = -1
    return gid


def visual_pools(items: pd.DataFrame, image: np.ndarray, n: int = VISUAL_POOL) -> dict[int, set[int]]:
    """For each item with a photo: its n nearest photos in the same category, other models only."""
    has = np.abs(image).sum(axis=1) > 0
    out: dict[int, set[int]] = {}
    for _, ix in items.groupby("CAT_DES_EN", sort=True).indices.items():
        ix = np.asarray(ix)
        ix = ix[has[ix]]
        for start in range(0, len(ix), engine.CHUNK):
            q = ix[start:start + engine.CHUNK]
            s = np.round((image[q] @ image[ix].T).astype(np.float64), engine.DECIMALS)
            s[engine.excluded(items, q, ix, include_siblings=False)] = -np.inf
            top = np.argsort(-s, axis=1, kind="stable")[:, :n]
            out.update({int(qi): set(ix[t].tolist()) for qi, t in zip(q, top)})
    return out


# --------------------------------------------------------------------------- metrics
AGREEMENT = ["same_subfamily", "same_colour", "same_finishing", "same_material"]


def metrics(items: pd.DataFrame, nb: pd.DataFrame, gid: pd.Series, pools: dict[int, set[int]]) -> dict[str, float]:
    """All the evaluation numbers for one set of neighbours (rows: query, rank, candidate)."""
    pos = pd.Series(np.arange(len(items)), index=items[KEY])
    q, c = pos.loc[nb["query"]].to_numpy(), pos.loc[nb["candidate"]].to_numpy()
    rank = nb["rank"].to_numpy()
    top = rank <= K
    qkey = nb["query"].to_numpy()
    out: dict[str, float] = {}

    g = gid.to_numpy()
    hit = (g[q] >= 0) & (g[q] == g[c])
    eligible = pd.Series(g[q] >= 0).groupby(qkey).first()
    for k in (K, 10):
        per_q = pd.Series(hit & (rank <= k)).groupby(qkey).any()
        out[f"hit@{k}"] = float(per_q[eligible].mean()) if eligible.any() else np.nan

    gfs = items["GFA_DES_EN"].astype("string") + " › " + items["GFS_DES_EN"].astype("string")
    for name, v in {"same_family": items["GFA_DES_EN"], "same_subfamily": gfs, "same_colour": items["CLR_DES"],
                    "same_finishing": items["FINISHING"], "same_material": items["MATERIAL"]}.items():
        v = v.astype("string").to_numpy(object, na_value=None)
        both = top & pd.notna(v[q]) & pd.notna(v[c])
        out[name] = float((v[q][both] == v[c][both]).mean()) if both.any() else np.nan

    pairs = [(qi, ci) for qi, ci in zip(q[top], c[top]) if qi in pools and ci in pools]
    out["visual@4"] = float(np.mean([ci in pools[qi] for qi, ci in pairs])) if pairs else np.nan

    sales = items["sales_pct_in_cat"].to_numpy(float)
    tq = pd.DataFrame({"q": q[top], "own": sales[q[top]], "nb": sales[c[top]]}).groupby("q").agg(
        own=("own", "first"), nb=("nb", "mean")).dropna()
    # Spearman = Pearson on ranks (pandas' method="spearman" needs scipy, not in the light environment)
    out["sales_corr"] = float(tq["own"].rank().corr(tq["nb"].rank())) if len(tq) > 2 else np.nan
    price = items["PRICE_BASE_W_VAT"].to_numpy(float)
    out["price_ratio"] = float(np.nanmedian(np.maximum(price[q[top]], price[c[top]])
                                            / np.minimum(price[q[top]], price[c[top]])))
    return out


def families(m: dict[str, float]) -> dict[str, float]:
    """The three families and their mean (the objective)."""
    f = {"catalogue": float(np.nanmean([m["hit@4"]] + [m[a] for a in AGREEMENT])),
         "visual": m["visual@4"], "commercial": m["sales_corr"]}
    return {**f, "objective": float(np.nanmean(list(f.values())))}


def random_baseline(items: pd.DataFrame, queries: np.ndarray, k: int = 10, seed: int = SEED) -> pd.DataFrame:
    """k random candidates per query from the same category (other models): the floor of every metric."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, ix in items.groupby("CAT_DES_EN", sort=True).indices.items():
        ix = np.asarray(ix)
        for qi in queries[np.isin(queries, ix)]:
            pool = ix[~engine.excluded(items, np.array([qi]), ix, include_siblings=False)[0]]
            pick = rng.choice(pool, size=min(k, len(pool)), replace=False)
            rows += [(items[KEY].iat[qi], r + 1, items[KEY].iat[ci]) for r, ci in enumerate(pick)]
    return pd.DataFrame(rows, columns=["query", "rank", "candidate"])


# --------------------------------------------------------------------------- configurations
def normalised(w: dict[str, float]) -> dict[str, float]:
    total = sum(w.values())
    return {b: round(v / total, 4) for b, v in w.items()}


def with_image(base: dict[str, float], w_image: float) -> dict[str, float]:
    """Image weight set to w_image; the other blocks keep their proportions and share the rest."""
    rest = {b: v for b, v in base.items() if b != "image"}
    total = sum(rest.values())
    return {"image": w_image, **{b: (1 - w_image) * v / total for b, v in rest.items()}}


def ablations(base: dict[str, float]) -> dict[str, dict[str, float]]:
    """Tabular only, visual only, each block alone and each block left out."""
    visual = {"image", "palette"}
    configs = {"tabular only": {b: (0.0 if b in visual else w) for b, w in base.items()},
               "visual only": {b: (w if b in visual else 0.0) for b, w in base.items()}}
    configs.update({f"only {b}": {x: float(x == b) for x in base} for b in base})
    configs.update({f"without {b}": {x: (0.0 if x == b else w) for x, w in base.items()} for b in base})
    return configs


class Evaluator:
    """Runs the engine for a set of weights on some queries and scores the result (cached)."""

    def __init__(self, items: pd.DataFrame, blocks: dict, gid: pd.Series, pools: dict):
        self.items, self.blocks, self.gid, self.pools = items, blocks, gid, pools
        self.cache: dict[tuple, dict] = {}

    def __call__(self, weights: dict[str, float], queries: list[str]) -> dict[str, float]:
        key = (tuple(sorted(normalised(weights).items())), len(queries), queries[0] if queries else "")
        if key not in self.cache:
            nb = engine.neighbours(self.items, self.blocks, weights, k=10, queries=queries)
            m = metrics(self.items, nb, self.gid, self.pools)
            self.cache[key] = {**m, **families(m)}
        return self.cache[key]


def better(new: dict, old: dict) -> bool:
    """A change is kept if the objective rises and no guarded metric drops more than TOLERANCE."""
    return (new["objective"] >= old["objective"] + MIN_GAIN
            and all(new[m] >= old[m] - TOLERANCE for m in GUARDED))


def coordinate_search(evaluate, start: dict[str, float], queries: list[str], passes: int = 2) -> pd.DataFrame:
    """Greedy search: multiply one block weight at a time, keep the change when `better`."""
    best, best_m = dict(start), evaluate(start, queries)
    log = [{"step": 0, "change": "start", **normalised(best), **best_m}]
    for _ in range(passes):
        improved = False
        for b in sorted(start):
            for f in SEARCH_FACTORS:
                trial = {**best, b: best[b] * f if best[b] > 0 else (0.1 if f > 1 else 0.0)}
                if trial == best or sum(trial.values()) <= 0:
                    continue
                m = evaluate(trial, queries)
                if better(m, best_m):
                    best, best_m, improved = trial, m, True
                    log.append({"step": len(log), "change": f"{b} ×{f:g}", **normalised(best), **best_m})
        if not improved:
            break
    return pd.DataFrame(log)


# --------------------------------------------------------------------------- human review
def sheet_queries(items: pd.DataFrame, seed: int = SEED) -> list[str]:
    """Per category: 8 random queries with a trusted photo and 2 without (fixed seed)."""
    rng = np.random.default_rng(seed)
    picked = []
    for _, grp in items.groupby("CAT_DES_EN", sort=True):
        photo = grp["img_trusted"].fillna(False).astype(bool)
        for mask, n in zip((photo, ~photo), SHEET_PER_CAT):
            keys = np.sort(grp.loc[mask, KEY].to_numpy())
            picked += sorted(rng.choice(keys, size=min(n, len(keys)), replace=False))
    return picked


def describe(items: pd.DataFrame, keys) -> list[str]:
    it = items.set_index(KEY).loc[list(keys)]
    return [f"{r.CAT_DES_EN} › {r.GFA_DES_EN} › {r.GFS_DES_EN} | {r.CLR_DES} | "
            f"{r.FINISHING if pd.notna(r.FINISHING) else '-'} | {r.PRICE_BASE_W_VAT:.2f} €" for r in it.itertuples()]


def has_verdicts(path: Path = SHEET_FILE) -> bool:
    if not path.exists():
        return False
    s = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    return s["verdict"].str.strip().ne("").any()


def make_sheet(items: pd.DataFrame, pooled: pd.DataFrame, queries: list[str], per_sheet: int = 10) -> pd.DataFrame:
    """One row per (query, candidate) pooled from several configurations; candidates in key order (blind)."""
    s = pooled[["query", "candidate"]].drop_duplicates().copy()
    s["q_order"] = s["query"].map({q: i for i, q in enumerate(queries)})
    s = s.sort_values(["q_order", "candidate"], kind="stable").reset_index(drop=True)
    s["sheet"] = s["q_order"] // per_sheet + 1
    s["label"] = s.groupby("query").cumcount().map(lambda i: string.ascii_uppercase[i])
    s["query_item"] = describe(items, s["query"])
    s["candidate_item"] = describe(items, s["candidate"])
    return s[["sheet", "query", "query_item", "label", "candidate", "candidate_item"]].assign(
        verdict="", reviewer="", notes="")


def review_contact_sheet(items: pd.DataFrame, rows: pd.DataFrame, path: Path, cell: int = 170, text: int = 46) -> None:
    """One row per query: the query photo, then its pooled candidates labelled A, B, C… (no scores: blind)."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font, bold = ImageFont.truetype("arial.ttf", 11), ImageFont.truetype("arialbd.ttf", 12)
    except OSError:
        font = bold = ImageFont.load_default()
    it = items.set_index(KEY)
    queries = list(dict.fromkeys(rows["query"]))
    width = 1 + int(rows.groupby("query").size().max())
    sheet = Image.new("RGB", (width * cell + 12, len(queries) * (cell + text)), "white")
    d = ImageDraw.Draw(sheet)
    for r, q in enumerate(queries):
        cells = [(q, "QUERY")] + list(rows.loc[rows["query"] == q, ["candidate", "label"]].itertuples(index=False))
        for c, (key, label) in enumerate(cells):
            x, y = c * cell + (12 if c else 0), r * (cell + text)
            item = it.loc[key]
            if bool(item["img_trusted"]):
                try:
                    im = Image.open(IMG_DIR / item["img_file"]).convert("RGB")
                    im.thumbnail((cell - 8, cell - 8))
                    sheet.paste(im, (x + (cell - im.width) // 2, y + (cell - im.height) // 2))
                except OSError:
                    d.text((x + 8, y + 80), "photo unreadable", fill="red", font=font)
            else:
                d.text((x + 8, y + 80), "no trusted photo", fill="#999", font=font)
            d.text((x + 4, y + cell), f"{label}  {key}", fill="#b00" if c == 0 else "black", font=bold)
            d.text((x + 4, y + cell + 15), textwrap.shorten(str(item["GFS_DES_EN"]), 26), fill="black", font=font)
            d.text((x + 4, y + cell + 29), textwrap.shorten(f"{item['CLR_DES']} · {item['PRICE_BASE_W_VAT']:.2f} €", 26),
                   fill="#333", font=font)
            d.rectangle([x, y, x + cell - 1, y + cell + text - 1], outline="#bbb")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=85)


def read_verdicts(path: Path = SHEET_FILE) -> pd.DataFrame | None:
    if not has_verdicts(path):
        return None
    s = pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=False)
    s["value"] = s["verdict"].str.strip().str.lower().map(VERDICTS)
    return s.dropna(subset=["value"])[["query", "candidate", "value"]]


def human_score(items: pd.DataFrame, blocks: dict, weights: dict, verdicts: pd.DataFrame) -> dict[str, float]:
    """Mean verdict of a configuration's top 5 on the sheet queries, and the share of those pairs judged."""
    queries = sorted(verdicts["query"].unique())
    nb = engine.neighbours(items, blocks, weights, k=SHEET_TOP, queries=queries)
    j = nb.merge(verdicts, on=["query", "candidate"], how="left")
    return {"human_precision@5": float(j["value"].mean()), "judged": float(j["value"].notna().mean()),
            "human_precision@1": float(j.loc[j["rank"] == 1, "value"].mean())}


# --------------------------------------------------------------------------- report
METRIC_COLS = ["hit@4", "hit@10", "same_family", "same_subfamily", "same_colour", "same_finishing", "same_material",
               "visual@4", "sales_corr", "price_ratio"]
FAMILY_COLS = FAMILIES + ["objective"]
PCT = {"hit@4", "hit@10", "same_family", "same_subfamily", "same_colour", "same_finishing", "same_material", "visual@4",
       "catalogue", "visual", "judged", "human_precision@5", "human_precision@1"}


def table(df: pd.DataFrame, cols: list[str]) -> str:
    def fmt(c, v):
        if isinstance(v, str):
            return v
        if pd.isna(v):
            return "–"
        return f"{v:.0%}" if c in PCT else f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)
    head = "| " + " | ".join([df.index.name or ""] + cols) + " |"
    rows = ["| " + " | ".join([str(i)] + [fmt(c, df.at[i, c]) for c in cols]) + " |" for i in df.index]
    return "\n".join([head, "|" + "---|" * (len(cols) + 1)] + rows)


def write_report(res: pd.DataFrame, sweep: pd.DataFrame, search: pd.DataFrame, current: dict, chosen: dict,
                 how: str, applied: bool, n: dict, human: pd.DataFrame | None) -> None:
    blocks = list(current)
    wtab = pd.DataFrame({"current": normalised(current), "chosen": normalised(chosen)}).T[blocks]
    wtab.index.name = "weights"
    steps = search[["step", "change"] + FAMILY_COLS].set_index("step")
    main = res.loc[["random (same category)", "current weights", "chosen weights"]]
    abl = res.drop(index=["random (same category)", "chosen weights"])
    human_text = (f"No verdicts yet. The team fills `outputs/phase4/human_review.csv` (`similar` / `partly` / `not`) "
                  f"using the contact sheets `human_review_<n>.jpg`, then reruns `run`. The sheet pools the top "
                  f"{SHEET_TOP} of image weights {', '.join(f'{w:g}' for w in SHEET_IMAGE)}, so the verdicts score each "
                  f"of them." if human is None else table(human, list(human.columns)))
    text = f"""# Phase 4: evaluation report

Generated by `src/phase4_eval.py run` ({datetime.now(timezone.utc):%Y-%m-%d}). Every number is measured on the
**test half** of the models ({n['test']:,} queries); weights are tuned on the other half ({n['tune']:,} queries).
The split is fixed (models shuffled with seed {SEED}), so runs are comparable.

## Metrics

| Metric | Family | Meaning |
|---|---|---|
| `hit@4`, `hit@10` | catalogue | share of queries with a "known similar" item (same family › sub-family, theme and colour, another model) in the top 4 / 10; {n['known']:,} test queries have one |
| `same_*` | catalogue | share of the top 4 with the same family, sub-family, colour (`CLR_DES`), finishing, material (pairs where both have a value) |
| `visual@4` | visual | share of the top 4 among the query's {VISUAL_POOL} nearest photos (queries and candidates with a trusted photo) |
| `sales_corr` | commercial | Spearman correlation of the query's sales percentile with the mean of its top 4 (sales are not used by the engine) |
| `price_ratio` | – | median of higher / lower price in the top 4 (1.00 = same price) |
| `catalogue` | | mean of `hit@4`, `same_subfamily`, `same_colour`, `same_finishing`, `same_material` |
| `objective` | | mean of `catalogue`, `visual` (= `visual@4`) and `commercial` (= `sales_corr`) |

Each family is circular for one block: `catalogue` rewards the categorical and text blocks, `visual` rewards the
image block (with only the image block, `visual@4` is 100% by definition). A search on the objective alone moves
almost all the weight to the image and loses a third of the material agreement, so the tuning below never trades
one family for another, and the image weight is left to the human review.

## Result

{table(main, METRIC_COLS)}

{table(main, FAMILY_COLS)}

## Image weight: the trade-off the proxies cannot decide

The other blocks keep their proportions. More image weight buys visual agreement and costs catalogue agreement.

{table(sweep, METRIC_COLS[:8] + FAMILY_COLS)}

## Ablations (current weights, blocks switched off or used alone)

{table(abl, METRIC_COLS)}

{table(abl, FAMILY_COLS)}

## Weight tuning

Greedy search on the tuning half: each block weight in turn is multiplied by {', '.join(f'{f:g}' for f in SEARCH_FACTORS)}.
A change is kept only if the objective rises by at least {MIN_GAIN} **and** none of {', '.join(f'`{m}`' for m in GUARDED)}
drops by more than {TOLERANCE}.

{table(steps, ['change'] + FAMILY_COLS)}

{table(wtab, blocks)}

{how} {'Written to `src/phase3_weights.json`; rerun `phase3_engine.py build`.' if applied else 'The weights file was not changed (`--apply` writes it).'}

## Human review

{human_text}
"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "eval_report.md").write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- CLI
def load() -> tuple[pd.DataFrame, dict]:
    items = phase1_clean.load_items().sort_values(KEY, kind="stable").reset_index(drop=True)
    return items, engine.load_blocks(items)


def human_table(items: pd.DataFrame, blocks: dict, configs: dict, verdicts: pd.DataFrame) -> pd.DataFrame:
    rows = {name: human_score(items, blocks, w, verdicts) for name, w in configs.items()}
    out = pd.DataFrame(rows).T[["judged", "human_precision@1", "human_precision@5"]]
    out.index.name = "configuration"
    return out


def cmd_run(args: argparse.Namespace) -> None:
    items, blocks = load()
    current = engine.load_weights()
    tune_mask = split_models(items)
    tune_q, test_q = items.loc[tune_mask, KEY].tolist(), items.loc[~tune_mask, KEY].tolist()
    gid = known_groups(items)
    evaluate = Evaluator(items, blocks, gid, visual_pools(items, blocks["image"]))

    search = coordinate_search(evaluate, current, tune_q)
    chosen = {b: float(search.iloc[-1][b]) for b in current}
    how = (f"The search kept {len(search) - 1} change(s)." if len(search) > 1
           else "No change passed the rule, so the current weights are kept.")
    tuned = json.loads(engine.WEIGHTS_FILE.read_text(encoding="utf-8")).get("tuned")
    if tuned:
        how = f"The current weights already come from an earlier tuning ({tuned}). " + how
    sweep_cfg = {f"image {w:g}": with_image(chosen, w) for w in IMAGE_SWEEP}

    verdicts = read_verdicts()
    human = None
    if verdicts is not None:
        human = human_table(items, blocks, {"current weights": current, **sweep_cfg}, verdicts)
        ok = human[human["judged"] >= MIN_COVERAGE]
        best = ok.drop(index="current weights", errors="ignore")["human_precision@5"]
        if not best.empty:
            chosen = sweep_cfg[best.idxmax()]
            how = f"Image weight from the human review: `{best.idxmax()}` has the best human precision@5."

    configs = {"current weights": current, "chosen weights": chosen, **ablations(current)}
    res = pd.DataFrame({name: evaluate(w, test_q) for name, w in configs.items()}).T
    base = random_baseline(items, np.flatnonzero(~tune_mask))
    m = metrics(items, base, gid, evaluate.pools)
    res = pd.concat([pd.DataFrame({"random (same category)": {**m, **families(m)}}).T, res])
    res.index.name = "configuration"
    sweep = pd.DataFrame({name: evaluate(w, test_q) for name, w in sweep_cfg.items()}).T
    sweep.index.name = "configuration"

    applied = bool(args.apply) and normalised(chosen) != normalised(current)
    if applied:
        meta = json.loads(engine.WEIGHTS_FILE.read_text(encoding="utf-8"))
        meta["weights"] = normalised(chosen)
        meta["note"] = ("Block weights of the Phase 3 score (score = sum of w_b * sim_b over the blocks both items have, "
                        "divided by the sum of those w_b). Tuned by src/phase4_eval.py (outputs/phase4/eval_report.md); "
                        "the image weight is to be confirmed by the human review.")
        meta["tuned"] = f"phase4_eval.py run --apply, {datetime.now(timezone.utc):%Y-%m-%d}"
        engine.WEIGHTS_FILE.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([res, sweep]).to_csv(REPORT_DIR / "eval_metrics.csv", encoding="utf-8-sig")
    search.to_csv(REPORT_DIR / "weight_search.csv", index=False, encoding="utf-8-sig")
    n = {"test": len(test_q), "tune": len(tune_q), "known": int((gid.to_numpy()[~tune_mask] >= 0).sum())}
    write_report(res, sweep, search, current, chosen, how, applied, n, human)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(pd.concat([res, sweep])[METRIC_COLS + FAMILY_COLS].round(3).to_string())
    print("chosen:", normalised(chosen), "(applied)" if applied else "")


def cmd_sheet(args: argparse.Namespace) -> None:
    if has_verdicts():
        raise SystemExit(f"{SHEET_FILE} already has verdicts: not overwritten (the team's work).")
    items, blocks = load()
    current = engine.load_weights()
    queries = sheet_queries(items)
    configs = {f"image {w:g}": normalised(with_image(current, w)) for w in SHEET_IMAGE}
    pooled = pd.concat([engine.neighbours(items, blocks, w, k=SHEET_TOP, queries=queries) for w in configs.values()])
    s = make_sheet(items, pooled, queries)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    s.to_csv(SHEET_FILE, index=False, encoding="utf-8-sig")
    SHEET_META.write_text(json.dumps({"queries": queries, "top": SHEET_TOP, "configs": configs, "seed": SEED,
                                      "created": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                                     indent=1), encoding="utf-8")
    for n, grp in s.groupby("sheet"):
        review_contact_sheet(items, grp, REPORT_DIR / f"human_review_{n}.jpg")
    print(f"{len(queries)} queries, {len(s)} pairs to judge -> {SHEET_FILE} + {s['sheet'].nunique()} contact sheets")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="metrics, ablations, weight tuning -> outputs/phase4/eval_report.md")
    r.add_argument("--apply", action="store_true", help="write the chosen weights to src/phase3_weights.json")
    r.set_defaults(fn=cmd_run)
    s = sub.add_parser("sheet", help="blind human review sheet (top 5 of several image weights, pooled)")
    s.set_defaults(fn=cmd_sheet)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
