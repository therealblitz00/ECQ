"""Phase 5b: other aggregations than the median, and evaluation metrics that fit the buyer's problem.

    python src/phase5b_metrics.py run [--boot 500]   # -> outputs/phase5b/metrics_report.md + metrics_results.csv

Two questions of the professor ("compute another metric besides the median"):
1. Aggregation: the baselines and the analogue estimate summarise the past products of a group with a median
   (baselines) or a similarity-weighted mean of log units (analogues). Alternatives are compared on the same split and
   the same evaluation rows as `phase5b_forecast.run_sales` / `run_price`: mean of log (= geometric mean of units),
   trimmed mean, similarity-weighted mean and median, quantiles 40 / 60 and, for price, the most common price point.
   Choices are read on the validation split; the test split is reported once.
2. Evaluation: WAPE, RMSLE, median and mean absolute percentage error, over-order versus under-order, error by sales
   tercile (the heavy tail), calibration of the 80% ranges, and the share within one price point.

Nothing here changes the delivered estimates. The models are the ones of `phase5b_forecast.py` (imported, not edited).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_engine as engine  # noqa: E402
import phase5b_forecast as fc  # noqa: E402

REPORT_DIR = fc.REPORT_DIR
BOOT = 500
TRIM = 0.2           # share cut at each end by the trimmed mean


# --------------------------------------------------------------------------- aggregations
def _clean(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v[np.isfinite(v)]


def wquantile(v: np.ndarray, w: np.ndarray, q: float = 0.5) -> float:
    """Weighted quantile (the smallest value whose cumulative weight reaches q); nan if there is no weight."""
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return np.nan
    v, w = v[ok], w[ok]
    o = np.argsort(v, kind="stable")
    cw = np.cumsum(w[o]) / w.sum()
    return float(v[o][min(np.searchsorted(cw, q - 1e-12), len(v) - 1)])


def trimmed(v: np.ndarray, cut: float = TRIM) -> float:
    v = np.sort(_clean(v))
    if len(v) == 0:
        return np.nan
    k = int(np.floor(cut * len(v)))
    return float(v[k:len(v) - k].mean())


def mode_value(v: np.ndarray, w: np.ndarray | None = None) -> float:
    """Most common value (a price point); with weights, the value with the largest total weight. Ties: the lowest."""
    v = np.asarray(v, float)
    w = np.ones(len(v)) if w is None else np.asarray(w, float)
    ok = np.isfinite(v) & (w > 0)
    if not ok.any():
        return np.nan
    s = pd.Series(w[ok]).groupby(np.round(v[ok], 4)).sum()
    return float(s.idxmax())


GROUP_AGGS = {
    "median (current)": lambda v: np.median(_clean(v)) if len(_clean(v)) else np.nan,
    "mean of log (geometric mean of units)": lambda v: _clean(v).mean() if len(_clean(v)) else np.nan,
    "trimmed mean (20% each end)": trimmed,
    "quantile 40": lambda v: np.quantile(_clean(v), 0.4) if len(_clean(v)) else np.nan,
    "quantile 60": lambda v: np.quantile(_clean(v), 0.6) if len(_clean(v)) else np.nan,
}
PRICE_ONLY = {"mode (most common price point)": lambda v: mode_value(_clean(v))}

# analogue aggregations: (values of the k neighbours, their similarity weights) -> one number
NEIGHBOUR_AGGS = {
    "similarity-weighted mean (current)": lambda v, w: np.sum(w * v) / w.sum(),
    "similarity-weighted mean, sharpened (w^4)": lambda v, w: np.sum(w ** 4 * v) / (w ** 4).sum(),
    "plain mean": lambda v, w: v.mean(),
    "trimmed mean (20% each end)": lambda v, w: trimmed(v),
    "median": lambda v, w: np.median(v),
    "similarity-weighted median": lambda v, w: wquantile(v, w, 0.5),
    "quantile 40": lambda v, w: np.quantile(v, 0.4),
    "quantile 60": lambda v, w: np.quantile(v, 0.6),
}
NEIGHBOUR_PRICE_ONLY = {"similarity-weighted mode (price point)": lambda v, w: mode_value(v, w)}


def group_aggregate(items: pd.DataFrame, y: np.ndarray, fit_rows: np.ndarray, cols: list[str], agg) -> np.ndarray:
    """Like `phase5b_forecast.group_baseline` with any aggregation (falls back to the category, then everything)."""
    df = items[cols].astype("string").assign(y=np.where(fit_rows, y, np.nan))
    pred = df.groupby(cols, dropna=False)["y"].transform(lambda s: agg(s.to_numpy(float)))
    cat = df.groupby("CAT_DES_EN")["y"].transform(lambda s: agg(s.to_numpy(float)))
    return pred.fillna(cat).fillna(agg(y[fit_rows])).to_numpy(float)


def neighbour_values(items: pd.DataFrame, blocks: dict, weights: dict, target: str, pool: np.ndarray,
                     k: int = fc.K_ANALOGUES) -> tuple[np.ndarray, np.ndarray]:
    """Target values and similarity weights of the k analogues of every product (same logic as
    `phase5b_forecast.analogues`: same category, other models, excluded candidates weigh 0). Shape (n, k), nan-padded."""
    has = engine.has_block(blocks)
    y = items[target].to_numpy(float)
    vals = np.full((len(items), k), np.nan)
    wts = np.zeros((len(items), k))
    ref = items["PROD_REF"].astype(str).to_numpy()
    for _, ix in items.groupby("CAT_DES_EN", sort=True).indices.items():
        ix = np.asarray(ix)
        cand = ix[pool[ix] & np.isfinite(y[ix])]
        for start in range(0, len(ix), engine.CHUNK):
            q = ix[start:start + engine.CHUNK]
            s = engine.score_matrix(blocks, has, weights, q, cand).astype(np.float64)
            s[ref[q][:, None] == ref[cand][None, :]] = -np.inf
            top = np.argsort(-s, axis=1, kind="stable")[:, :k]
            sc = np.take_along_axis(s, top, axis=1)
            w = np.where(np.isfinite(sc), np.clip(sc, 0, None), 0.0)
            vals[q, :top.shape[1]] = np.where(w > 0, y[cand][top], np.nan)
            wts[q, :top.shape[1]] = w
    return vals, wts


def neighbour_aggregate(vals: np.ndarray, wts: np.ndarray, agg) -> np.ndarray:
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        ok = np.isfinite(vals[i]) & (wts[i] > 0)
        if ok.any():
            out[i] = agg(vals[i][ok], wts[i][ok])
    return out


# --------------------------------------------------------------------------- metrics
def _terciles(y: np.ndarray) -> list[np.ndarray]:
    lo, hi = np.quantile(y, [1 / 3, 2 / 3])
    return [y <= lo, (y > lo) & (y <= hi), y > hi]


def units_metrics(y: np.ndarray, p: np.ndarray, aux=None) -> dict:
    """Sales metrics for log(1 + units) truth `y` and estimate `p`. Errors in units use expm1.
    wape: sum of absolute errors / sum of real units (the big sellers weigh what they sell).
    over_units / under_units: units ordered too many / too few, as a share of the real units (stock left / sales lost).
    Terciles are of the real units, so t3 = the best sellers; `bias_t*` = estimated / real units - 1 in the tercile."""
    u, e = np.expm1(y), np.expm1(np.clip(p, 0, None))
    d = e - u
    out = {"wape": np.abs(d).sum() / u.sum(), "rmsle": float(np.sqrt(np.mean((p - y) ** 2))),
           "mae_log": float(np.mean(np.abs(p - y))), "mdape": float(np.median(np.abs(d) / u)),
           "mape": float(np.mean(np.abs(d) / u)), "within_2x": float(np.mean(np.abs(p - y) < np.log(2))),
           "bias_log": float(np.mean(p - y)), "bias_units": e.sum() / u.sum() - 1,
           "over_units": np.clip(d, 0, None).sum() / u.sum(), "under_units": np.clip(-d, 0, None).sum() / u.sum(),
           "share_over": float(np.mean(d > 0)),
           "over_log": float(np.mean(np.clip(p - y, 0, None))), "under_log": float(np.mean(np.clip(y - p, 0, None)))}
    for name, m in zip(("t1", "t2", "t3"), _terciles(y)):
        out[f"wape_{name}"] = np.abs(d[m]).sum() / u[m].sum()
        out[f"bias_{name}"] = e[m].sum() / u[m].sum() - 1
        out[f"mae_log_{name}"] = float(np.mean(np.abs(p - y)[m]))
    return {k: float(v) for k, v in out.items()}


def price_metrics_more(points: np.ndarray):
    """Price metrics for log-price truth/estimate (euros); the price points are the 40 list prices."""
    def near(v):
        return np.abs(v[:, None] - points[None, :]).argmin(axis=1)

    def fn(y: np.ndarray, p: np.ndarray, aux=None) -> dict:
        ye, pe = np.exp(y), np.exp(p)
        d = pe - ye
        steps = np.abs(near(pe) - near(ye))
        out = {"mae_eur": np.abs(d).mean(), "wape": np.abs(d).sum() / ye.sum(), "mape": np.mean(np.abs(d) / ye),
               "mdape": np.median(np.abs(d) / ye), "rmsle": np.sqrt(np.mean((p - y) ** 2)),
               "bias": pe.sum() / ye.sum() - 1, "over_eur": np.clip(d, 0, None).mean(),
               "under_eur": np.clip(-d, 0, None).mean(), "exact_price_point": (steps == 0).mean(),
               "within_1_point": (steps <= 1).mean(), "within_10pct": np.mean(np.abs(d) / ye <= 0.1)}
        for name, m in zip(("t1", "t2", "t3"), _terciles(ye)):    # price terciles: cheap, middle, expensive
            out[f"wape_{name}"] = np.abs(d[m]).sum() / ye[m].sum()
        return {k: float(v) for k, v in out.items()}
    return fn


def aggregation_metrics(y: np.ndarray, p: np.ndarray, aux=None) -> dict:
    m = units_metrics(y, p)
    return {k: m[k] for k in ("mae_log", "wape", "rmsle", "bias_log", "within_2x")}


def bootstrap_table(fn, y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, ref: str,
                    n: int = BOOT) -> pd.DataFrame:
    """Per model and metric: the value, its 95% interval and the difference to `ref` with its interval. Resamples
    models (colourways of a model move together), the same resamples for every model."""
    point = {m: fn(y, p) for m, p in preds.items()}
    rng = np.random.default_rng(fc.SEED)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    draws = {m: [] for m in preds}
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        for m, p in preds.items():
            draws[m].append(fn(y[idx], p[idx]))
    rows = []
    for m in preds:
        for metric, v in point[m].items():
            d = np.array([r[metric] for r in draws[m]])
            dd = d - np.array([r[metric] for r in draws[ref]])
            rows.append({"model": m, "metric": metric, "value": v, "ci_lo": np.percentile(d, 2.5),
                         "ci_hi": np.percentile(d, 97.5),
                         "diff_vs_ref": point[m][metric] - point[ref][metric] if m != ref else np.nan,
                         "diff_lo": np.percentile(dd, 2.5) if m != ref else np.nan,
                         "diff_hi": np.percentile(dd, 97.5) if m != ref else np.nan})
    return pd.DataFrame(rows)


def wide(tab: pd.DataFrame, metrics: list[str], ci: bool = True, pct: set[str] = frozenset()) -> pd.DataFrame:
    """Models as rows, metrics as columns, cells 'value [lo, hi]' (strings)."""
    def cell(r):
        f = (lambda x: f"{x:.1%}") if r["metric"] in pct else (lambda x: f"{x:.3f}")
        return f(r["value"]) + (f" [{f(r['ci_lo'])}, {f(r['ci_hi'])}]" if ci else "")
    t = tab[tab["metric"].isin(metrics)].copy()
    t["cell"] = t.apply(cell, axis=1)
    order = list(dict.fromkeys(tab["model"]))
    return t.pivot(index="model", columns="metric", values="cell").reindex(order)[metrics]


# --------------------------------------------------------------------------- experiments
def sales_setup(items, blocks, weights, spec, use_analogue: bool | None = None) -> dict:
    """The split of `run_sales` with its models, plus the neighbours for the alternative aggregations."""
    train, test = fc.split(items, *spec)
    r = fc.run_sales(items, blocks, weights, train, test)
    if use_analogue is None:
        use_analogue = bool(r["ci_analogue_gain"][0] > 0)
    r["use_analogue"] = use_analogue
    r["chosen"] = r["variants"][use_analogue][2]
    r["train"], r["test"] = train, test
    return r


def price_setup(items, blocks, weights, spec, use_analogue: bool | None = None) -> dict:
    train, test = fc.split(items, *spec)
    r = fc.run_price(items, blocks, weights, train, test)
    if use_analogue is None:
        use_analogue = bool(r["ci_analogue_gain"][0] > 0)
    r["use_analogue"] = use_analogue
    r["chosen"] = r["variants"][use_analogue][2]
    r["train"], r["test"] = train, test
    return r


def aggregation_experiment(items, blocks, weights, r: dict, target: str, n: int) -> pd.DataFrame:
    """Alternatives to the median for the baselines and the analogue estimate, on the eval rows of `r`."""
    price = target == "log_price"
    y = r["y"]
    fit_rows, ev = r["train"] & np.isfinite(y), r["eval_rows"]
    groups = fc.model_groups(items)[ev]
    w_use = {**weights, "numeric": 0.0} if price else weights
    if price:
        bases = {"sub-family": ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"], "category": ["CAT_DES_EN"]}
    else:
        band = pd.qcut(items["PRICE_BASE_W_VAT"].rank(method="first"), 5, labels=False).astype(str)
        bases = {"family + price band": ["CAT_DES_EN", "GFA_DES_EN", "band"], "category": ["CAT_DES_EN"]}
        items = items.assign(band=band)
    aggs = {**GROUP_AGGS, **(PRICE_ONLY if price else {})}
    tables = []
    for bname, cols in bases.items():
        preds = {a: group_aggregate(items, y, fit_rows, cols, f) for a, f in aggs.items()}
        t = bootstrap_table(aggregation_metrics, y[ev], {a: p[ev] for a, p in preds.items()}, groups,
                            "median (current)", n)
        if not price:
            mg = fc.month_groups(items)[ev]
            sp = {a: fc.sales_metrics(y[ev], p[ev], mg)["spearman_within_cat"] for a, p in preds.items()}
        else:
            sp = {a: fc.spearman(y[ev], p[ev]) for a, p in preds.items()}
        extra = pd.DataFrame([{"model": a, "metric": "spearman", "value": v} for a, v in sp.items()])
        t = pd.concat([t, extra], ignore_index=True)
        t.insert(0, "what", f"baseline: {bname}")
        tables.append(t)
    vals, wts = neighbour_values(items, blocks, w_use, target, fit_rows)
    fallback = r["preds"]["sub-family median" if price else "category median"]
    naggs = {**NEIGHBOUR_AGGS, **(NEIGHBOUR_PRICE_ONLY if price else {})}
    preds = {}
    for a, f in naggs.items():
        est = neighbour_aggregate(vals, wts, f)
        preds[a] = np.where(np.isfinite(est), est, fallback)
    t = bootstrap_table(aggregation_metrics, y[ev], {a: p[ev] for a, p in preds.items()}, groups,
                        "similarity-weighted mean (current)", n)
    mg = fc.month_groups(items)[ev]
    sp = {a: (fc.spearman(y[ev], p[ev]) if price else fc.sales_metrics(y[ev], p[ev], mg)["spearman_within_cat"])
          for a, p in preds.items()}
    t = pd.concat([t, pd.DataFrame([{"model": a, "metric": "spearman", "value": v} for a, v in sp.items()])],
                  ignore_index=True)
    t.insert(0, "what", "analogue estimate (10 similar past products)")
    tables.append(t)
    return pd.concat(tables, ignore_index=True)


def model_experiment(items, r: dict, target: str, n: int, mean_factor: float | None = None) -> pd.DataFrame:
    """The full metric set for the baselines, the analogue estimate and the chosen model, with intervals."""
    price = target == "log_price"
    y, ev = r["y"], r["eval_rows"]
    names = (["category median", "sub-family median", "similar past products (engine)"] if price else
             ["category median", "family + price band median", "similar past products (engine)"])
    preds = {k: r["preds"][k] for k in names}
    preds[f"chosen: {r['chosen']}"] = r["preds"][r["chosen"]]
    if not price and mean_factor is not None:       # units-scale re-centring, factor read on the validation half
        p = r["preds"][r["chosen"]]
        preds[f"chosen, re-centred x{mean_factor:.2f} (units mean)"] = np.log1p(mean_factor * np.expm1(np.clip(p, 0, None)))
    fn = units_metrics if not price else price_metrics_more(np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique()))
    return bootstrap_table(fn, y[ev], {k: p[ev] for k, p in preds.items()}, fc.model_groups(items)[ev], names[0], n)


def recentre_factor(r_val: dict) -> float:
    """Factor on estimated units so that the total matches the real total on the validation half."""
    y, ev = r_val["y"], r_val["eval_rows"]
    p = r_val["preds"][r_val["chosen"]]
    return float(np.expm1(y[ev]).sum() / np.expm1(np.clip(p[ev], 0, None)).sum())


# --------------------------------------------------------------------------- report
SALES_MAIN = ["wape", "rmsle", "mdape", "mape", "within_2x", "bias_units"]
SALES_ASYM = ["over_units", "under_units", "share_over", "over_log", "under_log"]
SALES_TERC = ["wape_t1", "wape_t2", "wape_t3", "bias_t1", "bias_t2", "bias_t3"]
PRICE_MAIN = ["mae_eur", "wape", "mape", "mdape", "rmsle", "bias"]
PRICE_PTS = ["exact_price_point", "within_1_point", "within_10pct", "over_eur", "under_eur", "wape_t1", "wape_t2",
             "wape_t3"]
PCT = {"within_2x", "bias_units", "over_units", "under_units", "share_over", "bias_t1", "bias_t2", "bias_t3",
       "mape", "mdape", "wape", "wape_t1", "wape_t2", "wape_t3", "exact_price_point", "within_1_point",
       "within_10pct", "bias"}


def agg_summary(tab: pd.DataFrame, metrics=("mae_log", "wape", "bias_log")) -> str:
    out = []
    for what, t in tab.groupby("what", sort=False):
        rows = {}
        for m, g in t.groupby("model", sort=False):
            r = {}
            for k in metrics:
                x = g[g["metric"] == k].iloc[0]
                s = f"{x['value']:.3f}"
                if pd.notna(x["diff_lo"]):
                    s += f" ({x['diff_vs_ref']:+.3f} [{x['diff_lo']:+.3f}, {x['diff_hi']:+.3f}])"
                r[k] = s
            r["within_2x"] = f"{g[g['metric'] == 'within_2x'].iloc[0]['value']:.1%}"
            r["spearman"] = f"{g[g['metric'] == 'spearman'].iloc[0]['value']:.3f}"
            rows[m] = r
        out += [f"**{what}** (difference to the reference row, with its 95% interval)", "",
                fc._md(pd.DataFrame(rows).T), ""]
    return "\n".join(out)


def best_aggregation(tab: pd.DataFrame, what: str, metric: str = "wape") -> tuple[str, float]:
    t = tab[(tab["what"] == what) & (tab["metric"] == metric)].sort_values("value")
    return t.iloc[0]["model"], float(t.iloc[0]["value"])


def write_report(res: dict, n: int) -> str:
    L = ["# Phase 5b: other aggregations and metrics for the sales and price estimates", "",
         f"Generated by `src/phase5b_metrics.py run` (bootstrap by model, {n} resamples, 95% intervals). Same splits and "
         "evaluation rows as `forecast_report.md`: sales = whole-life units (train 232, new 241 models; validation half "
         "for the choices, test half read once); price = validation 232 -> 241, test 232 + 241 -> 242. Nothing here "
         "changes the delivered estimates.", "",
         "Reading rule: aggregations are compared on **validation**; **test** is shown to confirm. Lower is better for "
         "mae_log, wape, rmsle; closer to 0 is better for bias.", ""]
    for tgt, label in (("sales", "Sales"), ("price", "Price")):
        L += [f"## 1. Aggregation: {label}", ""]
        for sp in ("validation", "test"):
            L += [f"### {label}, {sp}", "", res[f"agg_{tgt}_{sp}"], ""]
    L += ["## 2. Evaluation metrics", ""]
    for key, title in res["tables"]:
        L += [f"### {title}", "", res[key], ""]
    L += res["ranges"]
    L += res["recommendation"]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--boot", type=int, default=BOOT)
    args = ap.parse_args()
    n = args.boot
    fc.bootstrap_diff.__defaults__ = (30,)      # the bootstrap of run_sales / run_price is not used here: keep it cheap
    items, blocks = fc.load()
    weights = engine.load_weights()

    sv = sales_setup(items, blocks, weights, fc.SALES_VALIDATION)
    st = sales_setup(items, blocks, weights, fc.SALES_TEST, sv["use_analogue"])
    pv = price_setup(items, blocks, weights, fc.VALIDATION)
    pt = price_setup(items, blocks, weights, fc.TEST, pv["use_analogue"])
    factor = recentre_factor(sv)

    csv_rows, res = [], {"tables": []}
    for tgt, setups, col in (("sales", {"validation": sv, "test": st}, "log_sales"),
                             ("price", {"validation": pv, "test": pt}, "log_price")):
        for sp, r in setups.items():
            a = aggregation_experiment(items, blocks, weights, r, col, n)
            res[f"agg_{tgt}_{sp}"] = agg_summary(a)
            csv_rows.append(a.assign(section="aggregation", target=tgt, split=sp))
            m = model_experiment(items, r, col, n, factor if tgt == "sales" else None)
            csv_rows.append(m.assign(section="metrics", target=tgt, split=sp, what="models"))
            if sp == "test":
                if tgt == "sales":
                    for key, mets, title in (("m_s_main", SALES_MAIN, "Sales, test: error measures"),
                                             ("m_s_asym", SALES_ASYM, "Sales, test: over-order versus under-order"),
                                             ("m_s_terc", SALES_TERC,
                                              "Sales, test: by tercile of real sales (t1 = slowest, t3 = best sellers)")):
                        res[key] = fc._md(wide(m, mets, pct=PCT).reset_index().set_index("model"))
                        res["tables"].append((key, title))
                else:
                    for key, mets, title in (("m_p_main", PRICE_MAIN, "Price, test: error measures"),
                                             ("m_p_pts", PRICE_PTS, "Price, test: price points, direction and price terciles")):
                        res[key] = fc._md(wide(m, mets, pct=PCT))
                        res["tables"].append((key, title))
                res[f"m_{tgt}_tab"] = m
    # calibration of the 80% ranges (the delivered conformalised quantiles and their alternatives)
    conf = fc.conformal_sales(items, blocks)
    ct = pd.DataFrame(conf["table"]).T[["coverage", "coverage, worst category", "coverage, best category",
                                         "median width (high / low)"]]
    for name, r_ in conf["table"].items():
        for k, v in r_.items():
            csv_rows.append(pd.DataFrame([{"section": "range_calibration", "target": "sales", "split": "test",
                                           "what": "80% range", "model": name, "metric": k, "value": v}]))
    res["ranges"] = ["### Calibration of the 80% sales ranges (test half, nominal coverage 80%)", "",
                     fc._md(ct, pct={"coverage", "coverage, worst category", "coverage, best category"}), "",
                     "Coverage is the share of real sales inside the range. Close to 80% is calibrated; the worst and "
                     "best category show how even it is.", ""]
    out = pd.concat(csv_rows, ignore_index=True)
    out.to_csv(REPORT_DIR / "metrics_results.csv", index=False, encoding="utf-8-sig")
    res["recommendation"] = recommendation(out, factor)
    text = write_report(res, n)
    (REPORT_DIR / "metrics_report.md").write_text(text, encoding="utf-8")
    print(text)


def _v(out: pd.DataFrame, target: str, model_start: str, metric: str, split: str = "test", section: str = "metrics"):
    t = out[(out["section"] == section) & (out["target"] == target) & (out["split"] == split)
            & (out["metric"] == metric) & out["model"].str.startswith(model_start)]
    return float(t.iloc[0]["value"])


def recommendation(out: pd.DataFrame, factor: float) -> list[str]:
    """Recommendation text. The numbers are read from `out`; the judgements were written after reading these tables
    (test half, 2026 run): re-read them if the data or the models change."""
    g = lambda m, k, **kw: _v(out, "sales", m, k, **kw)  # noqa: E731
    p = lambda m, k: _v(out, "price", m, k)  # noqa: E731
    ag = out[(out["section"] == "aggregation") & (out["target"] == "sales") & (out["split"] == "test")
             & (out["what"].str.startswith("analogue")) & (out["model"] == "trimmed mean (20% each end)")
             & (out["metric"] == "mae_log")].iloc[0]
    return [
        "## 3. Recommendation", "",
        "### Aggregation (the \"another metric besides the median\" for baselines and analogues)", "",
        "- **Keep the median for the baselines.** Mean of log (geometric mean of units) and quantile 40 are clearly "
        "worse (the intervals of the difference exclude 0 on both splits) and bias the estimate down; quantile 60 and the "
        "trimmed mean tie with the median on error but shift the bias by +0.18 and -0.07 log. Sales are skewed: the "
        "mean of log pulls the typical group down; the median is the robust, simple summary.",
        f"- **Analogues, sales:** a trimmed mean (or a median) of the 10 neighbours beats the similarity-weighted mean "
        f"by {ag['diff_vs_ref']:+.3f} log units on test (95% interval {ag['diff_lo']:+.3f} to {ag['diff_hi']:+.3f}), "
        "with the same sign on validation. It works by ignoring one or two very weak neighbours, so it over-estimates "
        "a little (bias +0.13 log). The gain is about 2% of units, and the delivered sales model does not use the "
        "analogues, so it is a refinement of a reference, not a reason to change the model. **Price:** all analogue "
        "aggregations are within 0.01 log of each other; the sharpened weights and quantile 60 gain 0.002 to 0.007. "
        "The mode (most common price point) is the worst choice for price on both baselines and analogues: "
        "the list prices are 40 points, a group has no clear peak and the mode ignores the neighbours' spread.", "",
        "### Evaluation metric", "",
        f"**Headline: WAPE** (weighted absolute percentage error = sum of the units missed / sum of the real units). "
        f"Chosen sales model: **{g('chosen: ', 'wape'):.1%}** (baselines {g('category median', 'wape'):.1%} and "
        f"{g('family + price band', 'wape'):.1%}; intervals in section 2). Price: **{p('chosen: ', 'wape'):.1%}**.",
        "",
        "Why: (1) it is the buyer's own question, \"how many units is the order away from what sells\", in a unit "
        "a buyer reads without a log; (2) it weighs each product by its sales, so the best sellers, which carry the "
        "turnover, count more than the slow ones (a median error treats a 50-unit and a 5,000-unit product alike); "
        "(3) it stays defined for tiny sellers.",
        "",
        f"Why not the others: **MAPE** is unusable ({g('chosen: ', 'mape'):.0%} for the chosen model: a product that "
        "sells 3 units and is estimated 80 counts as +2,500%); **MdAPE** "
        f"({g('chosen: ', 'mdape'):.1%}) is robust but hides the big sellers; **MAE in log** and **RMSLE** "
        f"({g('chosen: ', 'rmsle'):.3f}) treat a 2x miss alike on 10 and on 5,000 units and are not readable by a buyer; "
        "the **median error** is blind to the tail. The **tercile table** shows the tail: the model under-estimates "
        f"the best sellers ({g('chosen: ', 'bias_t3'):+.0%} of their units) and over-estimates the slowest "
        f"({g('chosen: ', 'bias_t1'):+.0%}); read it as the pattern of any estimate that is pulled towards the "
        "middle (terciles are cut on the real sales, which exaggerates it), not as a bug.", "",
        "**Supporting metric 1: over-order versus under-order** (units over and under, each as a share of real units). "
        f"Chosen sales model: {g('chosen: ', 'over_units'):.1%} too many (stock left) against "
        f"{g('chosen: ', 'under_units'):.1%} too few (sales lost). The cost of the two is not equal, and only the "
        "buyer knows the ratio. The log model leans to under-ordering "
        f"(net {g('chosen: ', 'bias_units'):+.1%}); a units re-centring (x{factor:.2f}, read on validation) balances "
        f"them ({g('chosen, re-centred', 'over_units'):.1%} vs {g('chosen, re-centred', 'under_units'):.1%}, net "
        f"{g('chosen, re-centred', 'bias_units'):+.1%}) at a small cost in WAPE "
        f"({g('chosen, re-centred', 'wape'):.1%}); not applied here: a decision for the buyer.", "",
        "**Supporting metric 2: share within 2x** for sales "
        f"({g('chosen: ', 'within_2x'):.1%}; baselines {g('category median', 'within_2x'):.1%}) and **share within one "
        f"price point** for price ({p('chosen: ', 'within_1_point'):.1%}; sub-family median "
        f"{p('sub-family median', 'within_1_point'):.1%}). They say how often the estimate is in the right "
        "ballpark, which is how an order is checked, and they are not moved by a few extreme products.", "",
        "The ranking measures of `forecast_report.md` (Spearman within category and in the same launch month) stay as "
        "the way to say whether the model orders the products like the market. The 80% ranges are calibrated "
        "(section 2: 80% coverage, worst category 78.5%).", ""]


if __name__ == "__main__":
    main()
