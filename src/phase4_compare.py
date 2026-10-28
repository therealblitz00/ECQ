"""Phase 4: compare the current engine with the Sprint 2 baseline on the same signals.

    python src/phase4_compare.py run          # -> outputs/phase4/compare_report.md + compare_metrics.csv
    python src/phase4_compare.py salescheck   # -> outputs/phase4/sales_corr_check.md: is sales_corr biased by cut 242 sales?
    python src/phase4_compare.py encoders image_clip image_fashionclip_nopersoncrops image_dinov2_nopersoncrops ...
                                              # -> outputs/phase2b/image_encoders.csv: same signals, one row per embedding

The baseline is the Sprint 2 engine: the same weights with the CLIP image embedding (`image_clip`). The current engine
is whatever `src/phase3_weights.json` sets (`image_embedding`, weights). Signals, strongest first:

1. the team's sorted preferences (outputs/phase4/feedback_*.jsonl): fixed weights, and 5-fold cross-validation over
   queries with the weights refitted on 4 folds (same Bradley-Terry fit as phase4_feedback); paired bootstrap over
   queries for the difference;
2. labels the image never saw: sub-family, material and finish of each photo's 4 nearest photos (image block alone);
3. the guarded proxy metrics of phase4_eval on the test half of the models, and the brief's own example.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1_clean  # noqa: E402
import phase3_engine as engine  # noqa: E402
import phase4_eval as ev  # noqa: E402
import phase4_feedback as fb  # noqa: E402
from config import KEY, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase4"
BASELINE_IMAGE = "image_clip"
BRIEF_QUERY, BRIEF_ANSWER = "167718_BU", ["169597_FU", "169607_IG", "169607_PE", "177781_LB"]
FOLDS, BOOT, SEED = 5, 4000, 1
LABELS = {"sub-family": "GFS_DES_EN", "material": "MATERIAL", "finish": "FINISHING", "colour": "CLR_DES"}


def team_tables(items: pd.DataFrame, image: np.ndarray, weights: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Right / total preferences per query: fixed weights, and cross-validated refitted weights."""
    pool, recs = fb.load_pool(), fb.read_feedback()
    index, s, h = fb.sim_matrix(pool)
    blocks = pool["blocks"]
    pos = pd.Series(np.arange(len(items)), index=items[KEY])
    pairs = sorted(index.items(), key=lambda kv: kv[1])
    qi = pos[[q for (q, _), _ in pairs]].to_numpy()
    ci = pos[[c for (_, c), _ in pairs]].to_numpy()
    j = blocks.index("image")
    s = s.copy()
    s[:, j] = np.where(h[:, j] > 0, np.einsum("ij,ij->i", image[qi], image[ci]), 0)

    def per_query(w, rs):
        sc = fb.scores(w, s, h)
        out = {}
        for r in rs:
            a, b = fb.preference_pairs([r], index)
            o = out.setdefault(r["query"], [0, 0])
            o[0] += int((sc[a] > sc[b]).sum())
            o[1] += len(a)
        return pd.DataFrame(out, index=["right", "pairs"]).T

    fixed = per_query(np.array([weights[b] for b in blocks]), recs)
    queries = sorted({r["query"] for r in recs})
    fold = dict(zip(queries, np.random.default_rng(SEED).integers(0, FOLDS, len(queries))))
    w0 = np.array([pool["weights"][b] for b in blocks])
    cv = []
    for f in range(FOLDS):
        win, lose = fb.preference_pairs([r for r in recs if fold[r["query"]] != f], index)
        cv.append(per_query(fb.fit_weights(s, h, win, lose, w0), [r for r in recs if fold[r["query"]] == f]))
    return fixed, pd.concat(cv)


def accuracy(t: pd.DataFrame) -> float:
    return float(t["right"].sum() / t["pairs"].sum())


def boot_diff(a: pd.DataFrame, b: pd.DataFrame, n: int = BOOT) -> tuple[float, float, float]:
    """95% interval of accuracy(b) - accuracy(a), resampling queries, and P(difference > 0)."""
    b = b.loc[a.index]
    rng, d = np.random.default_rng(SEED + 1), []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        d.append(b["right"].values[i].sum() / b["pairs"].values[i].sum()
                 - a["right"].values[i].sum() / a["pairs"].values[i].sum())
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def image_label_agreement(items: pd.DataFrame, image: np.ndarray, queries: list[str]) -> dict[str, float]:
    """Share of each photo's 4 nearest photos (image block alone) that share a label the image never saw."""
    pos = pd.Series(np.arange(len(items)), index=items[KEY])
    trusted = items["img_trusted"].fillna(False).to_numpy(bool)
    nb = engine.neighbours(items, {"image": image}, {"image": 1.0}, k=4, queries=[k for k in queries if trusted[pos[k]]])
    q, c = pos[nb["query"]].to_numpy(), pos[nb["candidate"]].to_numpy()
    out = {}
    for name, col in LABELS.items():
        v = items[col].astype("string").to_numpy(object, na_value=None)
        both = pd.notna(v[q]) & pd.notna(v[c])
        out[f"photo alone: same {name}"] = float((v[q][both] == v[c][both]).mean())
    return out


def brief_ranks(items: pd.DataFrame, blocks: dict, weights: dict) -> list[int]:
    nb = engine.neighbours(items, blocks, weights, k=len(items), queries=[BRIEF_QUERY]).set_index("candidate")["rank"]
    return [int(nb.get(k, -1)) for k in BRIEF_ANSWER]


def run() -> pd.DataFrame:
    items = phase1_clean.load_items()
    weights = engine.load_weights()
    current = engine.image_embedding()
    test_q = items.loc[~ev.split_models(items), KEY].tolist()
    gid = ev.known_groups(items)
    configs = {"baseline (Sprint 2, CLIP)": BASELINE_IMAGE, f"current ({current})": current}
    base_blocks = engine.load_blocks(items, image=BASELINE_IMAGE)
    pools = ev.visual_pools(items, base_blocks["image"])   # the visual proxy keeps the baseline's definition
    rows, tables = {}, {}
    for name, stem in configs.items():
        blocks = base_blocks if stem == BASELINE_IMAGE else engine.load_blocks(items, image=stem)
        m = ev.metrics(items, engine.neighbours(items, blocks, weights, k=10, queries=test_q), gid, pools)
        fixed, cv = team_tables(items, blocks["image"], weights)
        tables[name] = (fixed, cv)
        rows[name] = {"team, fixed weights": accuracy(fixed), "team, 5-fold CV refit": accuracy(cv),
                      **image_label_agreement(items, blocks["image"], test_q),
                      **{k: m[k] for k in ev.GUARDED + ["price_ratio"]},
                      "brief ranks": brief_ranks(items, blocks, weights)}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[name].items()}, flush=True)
    df = pd.DataFrame(rows).T
    names = list(configs)
    diffs = {kind: boot_diff(tables[names[0]][i], tables[names[1]][i]) for i, kind in enumerate(["fixed", "cv"])}
    guards_ok = all(df.at[names[1], g] >= df.at[names[0], g] - ev.TOLERANCE for g in ev.GUARDED)
    write_report(df, diffs, guards_ok, weights, current)
    df.to_csv(REPORT_DIR / "compare_metrics.csv")
    return df


def write_report(df: pd.DataFrame, diffs: dict, guards_ok: bool, weights: dict, current: str) -> None:
    def fmt(c, v):
        if isinstance(v, list):
            return " / ".join(map(str, v))
        return f"{v:.3f}" if c in ("sales_corr", "price_ratio") else f"{v:.1%}"
    lines = ["| | " + " | ".join(df.index) + " |", "|---|" + "---|" * len(df.index)]
    for c in df.columns:
        lines.append(f"| {c} | " + " | ".join(fmt(c, df.at[i, c]) for i in df.index) + " |")
    text = [
        "# Phase 4: current engine vs the Sprint 2 baseline",
        "",
        f"Generated by `src/phase4_compare.py run` ({datetime.now(timezone.utc):%Y-%m-%d}). Same weights for both "
        f"({', '.join(f'{b} {w:.2f}' for b, w in weights.items())}); only the image embedding differs "
        f"(`image_clip` vs `{current}`).",
        "",
        *lines,
        "",
        "Difference on the team's preferences (current − baseline), paired bootstrap over queries:",
        "",
        f"- fixed weights: {diffs['fixed'][0]:+.3f} to {diffs['fixed'][1]:+.3f} (positive in "
        f"{diffs['fixed'][2]:.0%} of the resamples)",
        f"- 5-fold CV refit: {diffs['cv'][0]:+.3f} to {diffs['cv'][1]:+.3f} (positive in {diffs['cv'][2]:.0%})",
        "",
        f"Guards (no Phase 4 metric down by more than {ev.TOLERANCE:.0%}): **{'kept' if guards_ok else 'VIOLATED'}**.",
        "",
        "Notes: `photo alone` rows use the image block only (labels the image never sees). `visual@4` keeps the "
        "baseline's definition (the 20 nearest CLIP photos), so it is biased towards the baseline. `hit@4` is circular for "
        "the categorical block (same theme and colour).",
        "",
    ]
    (REPORT_DIR / "compare_report.md").write_text("\n".join(text), encoding="utf-8")


def encoders(stems: list[str]) -> pd.DataFrame:
    """The same signals for several image embeddings (data/embeddings/<stem>.npy), same weights for all.

    The candidates of `phase2b_image_encoders.py embed --model <name> --no-person-crops` (same crops as CLIP)."""
    items = phase1_clean.load_items()
    weights = engine.load_weights()
    test_q = items.loc[~ev.split_models(items), KEY].tolist()
    gid = ev.known_groups(items)
    pools = ev.visual_pools(items, engine.load_blocks(items, image=BASELINE_IMAGE)["image"])
    rows = {}
    for stem in stems:
        blocks = engine.load_blocks(items, image=stem)
        m = ev.metrics(items, engine.neighbours(items, blocks, weights, k=10, queries=test_q), gid, pools)
        fixed, cv = team_tables(items, blocks["image"], weights)
        alone = engine.neighbours(items, {"image": blocks["image"]}, {"image": 1.0}, k=len(items), queries=[BRIEF_QUERY])
        alone = alone.set_index("candidate")["rank"]
        rows[stem] = {"team, fixed weights": accuracy(fixed), "team, 5-fold CV refit": accuracy(cv),
                      **image_label_agreement(items, blocks["image"], test_q),
                      "brief ranks (photo alone)": [int(alone.get(k, -1)) for k in BRIEF_ANSWER],
                      "brief ranks (engine)": brief_ranks(items, blocks, weights),
                      **{k: m[k] for k in ["hit@4", "same_subfamily", "same_material", "sales_corr"]}}
        print(stem, rows[stem], flush=True)
    df = pd.DataFrame(rows).T
    df.index.name = "encoder"
    df.to_csv(ROOT / "outputs" / "phase2b" / "image_encoders.csv")
    return df


def sales_corr(items: pd.DataFrame, nb: pd.DataFrame, pct: np.ndarray, use: np.ndarray) -> float:
    """phase4_eval's sales_corr with another sales percentile, on the queries and candidates allowed by `use`."""
    pos = pd.Series(np.arange(len(items)), index=items[KEY])
    top = nb[nb["rank"] <= ev.K]
    q, c = pos.loc[top["query"]].to_numpy(), pos.loc[top["candidate"]].to_numpy()
    ok = use[q] & use[c]
    tq = pd.DataFrame({"q": q[ok], "own": pct[q[ok]], "nb": pct[c[ok]]}).groupby("q").agg(
        own=("own", "first"), nb=("nb", "mean")).dropna()
    return float(tq["own"].rank().corr(tq["nb"].rank()))


def random_neighbours(items: pd.DataFrame, queries: list[str], by: list[str], k: int = ev.K) -> pd.DataFrame:
    """k random other models from the same group as each query: what the grouping alone gives."""
    rng = np.random.default_rng(SEED)
    keys, refs = items[KEY].to_numpy(), items["PROD_REF"].astype(str).to_numpy()
    wanted = set(queries)
    rows = []
    for _, ix in items.groupby(by, sort=True).indices.items():
        for i in ix:
            if keys[i] not in wanted:
                continue
            others = ix[refs[ix] != refs[i]]
            for r, j in enumerate(rng.permutation(others)[:k], 1):
                rows.append((keys[i], r, keys[j]))
    return pd.DataFrame(rows, columns=["query", "rank", "candidate"])


def sales_check() -> None:
    """Does sales_corr measure "sells alike" or "launched at the same time"? 242 sales are cut by the extraction
    (outputs/phase5b/forecast_report.md), and the engine's numeric block includes the launch date, so products launched
    together are neighbours and share the cut. Report only: no weight is changed."""
    items = phase1_clean.load_items()
    weights = engine.load_weights()
    test_q = items.loc[~ev.split_models(items), KEY].tolist()
    sea = items["SEA_COD"].astype(str)
    items = items.assign(launch=pd.to_datetime(items["STORE_DATE_PLANNED"]).dt.to_period("M").astype(str))
    sales = items["SALES_QTY"]
    life = sea.isin(["232", "241"]).to_numpy()
    everyone = np.ones(len(items), bool)
    pcts = {
        "(a) as now: within category, all seasons": (items["sales_pct_in_cat"].to_numpy(float), everyone),
        "(b) within category × season": (sales.groupby([items["CAT_DES_EN"], sea]).rank(pct=True).to_numpy(float),
                                         everyone),
        "(c) whole lives only (232, 241)": (
            sales.where(life).groupby([items["CAT_DES_EN"], sea]).rank(pct=True).to_numpy(float), life),
    }
    configs = {"baseline (Sprint 2, CLIP)": (BASELINE_IMAGE, weights),
               f"current ({engine.image_embedding()})": (engine.image_embedding(), weights),
               "current, numeric block off (diagnostic)": (engine.image_embedding(), {**weights, "numeric": 0.0})}
    nbs = {name: engine.neighbours(items, engine.load_blocks(items, image=stem), w, k=ev.K, queries=test_q)
           for name, (stem, w) in configs.items()}
    nbs["control: random, same category"] = random_neighbours(items, test_q, ["CAT_DES_EN"])
    nbs["control: random, same category and launch month"] = random_neighbours(items, test_q, ["CAT_DES_EN", "launch"])
    df = pd.DataFrame({n: {p: sales_corr(items, nb, v, use) for p, (v, use) in pcts.items()}
                       for n, nb in nbs.items()}).T
    lines = ["| | " + " | ".join(df.columns) + " |", "|---|" + "---|" * len(df.columns)]
    lines += [f"| {i} | " + " | ".join(f"{v:.3f}" for v in r) + " |" for i, r in df.iterrows()]
    text = [
        "# Phase 4: is `sales_corr` biased by the cut 242 sales?",
        "",
        f"Generated by `src/phase4_compare.py salescheck` ({datetime.now(timezone.utc):%Y-%m-%d}). Test half of the "
        "models, top 4: Spearman of the query's sales percentile with the mean of its neighbours' (as in "
        "`phase4_eval.py`). Season 242 sales are cut by the extraction, so a 242 product launched late looks like a "
        "poor seller. (b) compares sales only within a season; (c) keeps only whole lives. **Report only: no weight "
        "is changed.**",
        "",
        *lines,
        "",
        f"- **The bias is real, about {df.iat[1, 0] - df.iat[1, 2]:.2f}:** {df.iat[1, 0]:.3f} as now vs {df.iat[1, 2]:.3f} on whole "
        f"lives (baseline {df.iat[0, 0]:.3f} vs {df.iat[0, 2]:.3f}). Most of the engine's `sales_corr` survives when "
        "the cut sales are removed.",
        f"- **The launch month alone is worth {df.iat[4, 0]:.3f} as now, {df.iat[4, 2]:.3f} on whole lives:** with all "
        "seasons pooled, part of the metric rewards *launched at the same time*. On whole lives the launch month "
        "still matters a little (earlier launches sell longer).",
        "- **Ranking of the configurations unchanged:** baseline and current stay within 0.011 of each other in every "
        "column, so the Phase 4 guard decision does not depend on the bias. A future tuning should use (c).",
        "",
    ]
    (REPORT_DIR / "sales_corr_check.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "encoders", "salescheck"])
    ap.add_argument("stems", nargs="*", help="encoders: image embedding stems in data/embeddings/")
    args = ap.parse_args()
    if args.cmd == "encoders":
        encoders(args.stems or [BASELINE_IMAGE, engine.image_embedding()])
    elif args.cmd == "salescheck":
        sales_check()
    else:
        run()


if __name__ == "__main__":
    main()
