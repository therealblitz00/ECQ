"""Phase 5b: the sales model against the buyer's own forecast, and how much it leans on the buyer's planning classes.

    python src/phase5b_buyer.py run        # -> outputs/phase5b/buyer_report.md + buyer_benchmark.csv, ~5 min

The buyer's purchase depth (`PUR_BET`: SB, M1, M2, M3) and store tiers (`TYPOLOGY`: A, BA, CBA, DCBA) are decided
when the order is placed, the same moment the model is meant to help. They are the buyer's own estimate of how well
a product will sell, so they are the human reference for the model. They are never a feature of the delivered model
(`phase5b_forecast.py`); here they are only a benchmark.

Same protocol as `phase5b_forecast.py`: whole-life units, train 232 -> the new 241 models (both halves shown), and
the same-month ranking on 242 (cut sales, ranking only). What is measured:
1. The buyer alone: median sales of the training products with the same category and decision(s). Within a
   category this is just the buyer's decision, with the order of its levels learned from the training season (no
   order assumed).
2. The model against the buyer, and model + buyer against the buyer alone (does the model add information?).
3. Pairs of products of the same category and launch month: when the model and the buyer disagree on which one
   sells more, who is right; and how well the model orders the pairs the buyer left tied (same decisions).
4. Ablation of the planning classes the buyers assign (`FASHIONTYPE`, `COLLECTION_TYPE`, `CATEGORY_MATRIX`).

Caveat (stated in the report): purchase depth and store tiers also *cause* sales (more stock, more stores), so the
buyer's reference is optimistic; the model can only see the product.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5b_forecast as fc  # noqa: E402

PLANNING_CLASSES = ["FASHIONTYPE", "COLLECTION_TYPE", "CATEGORY_MATRIX"]
BUYER = {"buyer: purchase depth (PUR_BET)": ["PUR_BET"],
         "buyer: store tiers (TYPOLOGY)": ["TYPOLOGY"],
         "buyer: both decisions": ["PUR_BET", "TYPOLOGY"]}
BUYER_REF = "buyer: both decisions"
MODEL = "model (delivered, no buyer decisions)"
COMBINED = "model + buyer decisions"
N_BOOT = 300
SPLITS = [("Validation half: 232 → new 241 models (A)", fc.SALES_VALIDATION, True),
          ("Test half: 232 → new 241 models (B)", fc.SALES_TEST, True),
          (f"Ranking check: 232 + 241 → {fc.TEST_SEASON} (cut sales: ranking only)", fc.TEST, False)]
RANK_ONLY = ["spearman_within_cat", "spearman_same_month", "top20_precision"]


def month_groups(items: pd.DataFrame) -> np.ndarray:
    return (items["CAT_DES_EN"].astype(str) + "|" + items["launch_month"].astype(int).astype(str) + "|"
            + items["SEA_COD"].astype(str)).to_numpy()


def predictions(items, blocks, train) -> tuple[dict, dict, np.ndarray]:
    """Buyer references, the delivered model, model + buyer, and the planning-class ablations."""
    y = items["log_sales"].to_numpy(float)
    fit_rows = train & np.isfinite(y)
    preds = {name: fc.group_baseline(items, y, fit_rows, ["CAT_DES_EN"] + cols) for name, cols in BUYER.items()}
    X = fc.feature_table(items, blocks, train, "log_sales", None)
    _, preds[MODEL] = fc.fit_predict(X, y, train, fit_rows)
    Xb = fc.feature_table(items, blocks, train, "log_sales", None, extra=fc.BUYER_DECISIONS)
    _, preds[COMBINED] = fc.fit_predict(Xb, y, train, fit_rows)
    ablations = {}
    for name, cols in [("without CATEGORY_MATRIX", ["CATEGORY_MATRIX"]),
                       ("without FASHIONTYPE and COLLECTION_TYPE", ["FASHIONTYPE", "COLLECTION_TYPE"]),
                       ("without the three planning classes", PLANNING_CLASSES)]:
        _, ablations[name] = fc.fit_predict(X.drop(columns=cols), y, train, fit_rows)
    return preds, ablations, y


def pair_table(y, model, buyer, groups, models, n_boot: int = N_BOOT) -> dict:
    """Pairs of different models in the same category x launch month (groups of more than 10, as in the metric) with
    different sales. Splits them by what the buyer says and counts who orders them right; 95% intervals resample
    models (a pair counts once per draw of each of its two models)."""
    uniq, inv = np.unique(models, return_inverse=True)
    a_parts, b_parts = [], []
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        if len(ix) <= 10:
            continue
        a, b = np.triu_indices(len(ix), 1)
        a_parts.append(ix[a])
        b_parts.append(ix[b])
    a, b = np.concatenate(a_parts), np.concatenate(b_parts)
    keep = (y[a] != y[b]) & (inv[a] != inv[b])
    a, b = a[keep], b[keep]
    truth = np.sign(y[a] - y[b])
    sm, sb = np.sign(model[a] - model[b]), np.sign(buyer[a] - buyer[b])
    masks = {"all pairs": np.ones(len(a), bool), "pairs the buyer ranks": sb != 0,
             "buyer ranks the pair, model agrees": (sb != 0) & (sm == sb),
             "buyer ranks the pair, model disagrees": (sb != 0) & (sm == -sb),
             "buyer ties the pair (same decisions)": sb == 0}
    hits = {"model": (sm == truth).astype(float), "buyer": (sb == truth).astype(float)}
    rng = np.random.default_rng(fc.SEED)
    boot = {(m, h): [] for m in masks for h in hits}
    for _ in range(n_boot):   # one weight vector at a time: pairs x resamples would not fit in memory
        d = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq)).astype(float)
        w = d[inv[a]] * d[inv[b]]
        for m, mask in masks.items():
            tot = w[mask].sum()
            for h, hit in hits.items():
                boot[(m, h)].append((w[mask] * hit[mask]).sum() / tot if tot else np.nan)
    out = {}
    for m, mask in masks.items():
        out[m] = {"pairs": int(mask.sum()), "share_of_pairs": float(mask.mean())}
        for h, hit in hits.items():
            bs = np.array(boot[(m, h)])
            out[m][f"{h}_right"] = (float(hit[mask].mean()) if mask.any() else np.nan,
                                    float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))
    return out


def fmt_ci(t) -> str:
    return f"{t[0]:.1%} [{t[1]:.1%}, {t[2]:.1%}]"


def run_split(items, blocks, label, seasons, units: bool) -> dict:
    train, test = fc.split(items, *seasons)
    preds, ablations, y = predictions(items, blocks, train)
    ev = test & np.isfinite(y)
    groups, models = month_groups(items)[ev], fc.model_groups(items)[ev]
    table = {n: fc.sales_metrics(y[ev], p[ev], groups) for n, p in {**preds, **ablations}.items()}
    month = lambda yy, pp, cc: fc.sales_metrics(yy, pp, cc)["spearman_same_month"]  # noqa: E731
    err = lambda yy, pp, _: -np.mean(np.abs(yy - pp))  # noqa: E731  (units error saved, log units)
    metrics = [("same-month ranking", month)] + ([("units error saved (log)", err)] if units else [])
    ci = {}
    for metric_name, m in metrics:
        for a, b in [(BUYER_REF, MODEL), (BUYER_REF, COMBINED), (MODEL, COMBINED)] + [(MODEL, n) for n in ablations]:
            p_a, p_b = preds[a][ev], (preds.get(b) if b in preds else ablations[b])[ev]
            ci[(b, a, metric_name)] = fc.bootstrap_diff(m, y[ev], p_a, p_b, models, groups, n=N_BOOT)
    pairs = pair_table(y[ev], preds[MODEL][ev], preds[BUYER_REF][ev], groups, models)
    fit = train & np.isfinite(y)
    order = {c: items[fit].groupby(c)["log_sales"].median().sort_values(ascending=False).round(2).to_dict()
             for c in fc.BUYER_DECISIONS}
    return {"label": label, "units": units, "table": table, "ci": ci, "pairs": pairs, "order": order,
            "n_fit": int(fit.sum()), "n_eval": int(ev.sum())}


def _pairs_md(pairs: dict) -> list[str]:
    lines = ["| pairs | n | share | model right | buyer right |", "|---|---|---|---|---|"]
    for name, p in pairs.items():
        lines.append(f"| {name} | {p['pairs']:,} | {p['share_of_pairs']:.1%} | {fmt_ci(p['model_right'])} | "
                     f"{fmt_ci(p['buyer_right'])} |")
    return lines


def write_report(res: list[dict]) -> str:
    L = ["# Phase 5b: the sales model against the buyer\n",
         f"Generated by `src/phase5b_buyer.py run` ({date.today().isoformat()}). Same protocol and metrics as "
         "`forecast_report.md` (whole-life units on 241; 242 only for the same-month ranking); 95% intervals resample "
         f"models ({N_BOOT} resamples). Buyer decisions are a benchmark here, never a feature of the delivered model.\n",
         "**The buyer's reference** = median sales of the training products with the same category and the same "
         "decision. Within a category it is the buyer's decision itself, with its levels ordered by the training "
         "season (median log units, 232):\n"]
    for c, o in res[0]["order"].items():
        L.append(f"- `{c}`: " + ", ".join(f"{k} {v}" for k, v in o.items()))
    L.append("\n**Caveat.** Purchase depth and store tiers also *cause* sales (more stock and more stores sell more), "
             "so the buyer's numbers are an optimistic reference; the model only sees the product.\n")
    order = list(BUYER) + [MODEL, COMBINED]
    for r in res:
        L.append(f"## {r['label']}: {r['n_fit']:,} training / {r['n_eval']:,} evaluated products\n")
        df = pd.DataFrame(r["table"]).T.loc[order]
        if not r["units"]:
            df = df[RANK_ONLY]
        L.append(fc._md(df, pct={"top20_precision", "within_2x"}))
        L.append("\nDifferences (95% CI):\n")
        for (b, a, metric), (lo, hi) in r["ci"].items():
            if "without" not in b:
                L.append(f"- {metric}: {b} − {a}: {lo:+.3f} to {hi:+.3f}")
        L.append("\nPairs of two different models of the same category launched in the same month, with different "
                 "sales. *Right* = orders the pair like the real sales; a tie counts as wrong (the buyer ties every "
                 "pair with the same decisions; then only the model can order it).\n")
        L += _pairs_md(r["pairs"])
        L.append("")
    L.append("## Planning classes: is the model re-reading the buyer's classification?\n")
    L.append("`FASHIONTYPE`, `COLLECTION_TYPE` and `CATEGORY_MATRIX` are planning classes set by the buying team; when "
             "they are assigned is to confirm with Parfois. Dropping them from the delivered model:\n")
    for r in res:
        names = [MODEL] + [k for k in r["table"] if k.startswith("without")]
        df = pd.DataFrame(r["table"]).T.loc[names]
        L.append(f"**{r['label']}**\n")
        L.append(fc._md(df if r["units"] else df[RANK_ONLY], pct={"top20_precision", "within_2x"}))
        L.append("")
        for (b, a, metric), (lo, hi) in r["ci"].items():
            if "without" in b:
                L.append(f"- {metric}: {b} − model: {lo:+.3f} to {hi:+.3f}")
        L.append("")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    items, blocks = fc.load()
    res = [run_split(items, blocks, label, seasons, units) for label, seasons, units in SPLITS]
    rows = [{"split": r["label"], "estimate": n, **m} for r in res for n, m in r["table"].items()]
    pd.DataFrame(rows).to_csv(fc.REPORT_DIR / "buyer_benchmark.csv", index=False)
    report = write_report(res)
    (fc.REPORT_DIR / "buyer_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
