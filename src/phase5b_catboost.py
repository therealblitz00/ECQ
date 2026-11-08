"""Phase 5b: CatBoost against LightGBM for the sales and price estimates (read-only comparison).

    python src/phase5b_catboost.py run          # -> outputs/phase5b/catboost_report.md + catboost_results.csv

Like for like with `phase5b_forecast.py`: same data, same splits (sales: 232 -> halves of the new 241 models, the
first half for the choices and the second reported once; price: 232 -> 241 for the choices, 232 + 241 -> 242
reported), same features (`feature_table`), same targets (whole-life log units; price as markup over the landed cost)
and the same metrics and model-level bootstrap. The categorical attributes go to CatBoost as native `cat_features`
(a missing value is its own level). The small grid of depth / learning rate / L2 is chosen on the choice split only
(no early stopping on any evaluation row); the reported split is scored once with the chosen setting.
Nothing existing is overwritten (no `forecast.parquet`).
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_engine as engine  # noqa: E402
import phase5b_forecast as fc  # noqa: E402

REPORT_DIR = fc.REPORT_DIR
SEED = fc.SEED
ITERATIONS = 800
GRID = {"depth": (4, 6, 8), "learning_rate": (0.03, 0.08), "l2_leaf_reg": (3.0, 10.0)}
BASE = dict(loss_function="RMSE", iterations=ITERATIONS, random_seed=SEED, thread_count=4, verbose=0,
            allow_writing_files=False)


# --------------------------------------------------------------------------- model
def cb_frame(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Categorical columns as plain strings (missing = the level "NA"), numbers untouched."""
    out = X.copy()
    cats = [c for c in out if isinstance(out[c].dtype, pd.CategoricalDtype)]
    for c in cats:
        out[c] = out[c].astype(object).where(out[c].notna(), "NA").astype(str)
    return out, cats


def fit_cb(X: pd.DataFrame, y: np.ndarray, fit_rows: np.ndarray, params: dict | None = None):
    """Fit on `fit_rows`; returns (model, prediction for every row)."""
    from catboost import CatBoostRegressor
    Xc, cats = cb_frame(X)
    model = CatBoostRegressor(**{**BASE, **(params or {})}, cat_features=cats)
    model.fit(Xc[fit_rows], y[fit_rows])
    return model, model.predict(Xc)


def shap_check(model, X: pd.DataFrame, rows: np.ndarray, n: int = 300) -> dict:
    """Per-prediction explanations: CatBoost ShapValues (one column per feature + the expected value)."""
    from catboost import Pool
    Xc, cats = cb_frame(X)
    idx = np.flatnonzero(rows)[:n]
    t0 = time.perf_counter()
    sv = model.get_feature_importance(Pool(Xc.iloc[idx], cat_features=cats), type="ShapValues")
    dt = time.perf_counter() - t0
    err = float(np.abs(sv.sum(axis=1) - model.predict(Xc.iloc[idx])).max())
    return {"n": len(idx), "ms_per_product": 1000 * dt / max(len(idx), 1), "max_additivity_error": err}


# --------------------------------------------------------------------------- tuning
def tune(score, grid: dict = GRID) -> tuple[dict, pd.DataFrame]:
    """`score(params)` -> validation error (lower is better). Returns the best params and the whole grid."""
    keys = list(grid)
    rows = []
    for combo in itertools.product(*grid.values()):
        p = dict(zip(keys, combo))
        rows.append({**p, "val_error": float(score(p))})
    table = pd.DataFrame(rows)
    best = table.loc[table["val_error"].idxmin()]
    return {k: (int(best[k]) if k == "depth" else float(best[k])) for k in keys}, table


# --------------------------------------------------------------------------- sales
def sales_setup(items, blocks, weights, split_args):
    y = items["log_sales"].to_numpy(float)
    known = np.isfinite(y)
    train, ev = fc.split(items, *split_args)
    fit_rows, eval_rows = train & known, ev & known
    ana, _ = fc.analogues(items, blocks, weights, "log_sales", fit_rows)
    X0 = fc.feature_table(items, blocks, train, "log_sales", None)
    X1 = fc.feature_table(items, blocks, train, "log_sales", ana)
    return y, fit_rows, eval_rows, {"no analogues": X0, "+ analogues": X1}


def tune_sales(items, blocks, weights) -> tuple[dict, pd.DataFrame]:
    y, fit_rows, ev, Xs = sales_setup(items, blocks, weights, fc.SALES_VALIDATION)
    X = Xs["no analogues"]

    def score(p):
        _, pred = fit_cb(X, y, fit_rows, p)
        return np.mean(np.abs(y[ev] - pred[ev]))
    return tune(score)


def tune_price(items, blocks, weights) -> tuple[dict, pd.DataFrame]:
    y, fit_rows, ev, X0, X1, cost, has = price_setup(items, blocks, weights, fc.VALIDATION)
    points = np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique())
    X = X0
    _, no_cost = fit_cb(X.drop(columns="log_cost"), y, fit_rows)    # fallback fixed; the grid tunes the markup model

    def score(p):
        _, pm = fit_cb(X, y - cost, fit_rows & has, p)
        pred = np.where(has, cost + pm, no_cost)
        return fc.price_metrics(y[ev], pred[ev], points)["mae_eur"]
    return tune(score)


def price_setup(items, blocks, weights, split_args):
    y = items["log_price"].to_numpy(float)
    train, ev = fc.split(items, *split_args)
    cost = items["log_cost"].to_numpy(float)
    has = np.isfinite(cost)
    ana, _ = fc.analogues(items, blocks, {**weights, "numeric": 0.0}, "log_price", train)
    X0 = fc.feature_table(items, blocks, train, "log_price", None).assign(log_cost=cost)
    X1 = fc.feature_table(items, blocks, train, "log_price", ana).assign(log_cost=cost)
    return y, train, ev, X0, X1, cost, has


# --------------------------------------------------------------------------- comparisons
def compare_sales(items, blocks, weights, split_args, params, explain: bool = False) -> dict:
    """LightGBM (from the forecast module) and CatBoost on one sales split; paired bootstrap of CatBoost - LightGBM."""
    lgb_run = fc.run_sales(items, blocks, weights, *fc.split(items, *split_args))
    y, fit_rows, ev, Xs = sales_setup(items, blocks, weights, split_args)
    cats = fc.month_groups(items)
    preds = {k: v for k, v in lgb_run["preds"].items()
             if k in ("category median", "family + price band median", "similar past products (engine)",
                      "gradient boosting, no analogues", "gradient boosting + analogues")}
    preds = {("LightGBM " + k.split(", ")[-1].replace("gradient boosting ", "") if "boosting" in k else k): v
             for k, v in preds.items()}
    models = {}
    for name, X in Xs.items():
        models[name], preds[f"CatBoost {name}"] = fit_cb(X, y, fit_rows, params)
    preds["average LightGBM + CatBoost (no analogues)"] = (preds["LightGBM no analogues"]
                                                           + preds["CatBoost no analogues"]) / 2
    table = {n: fc.sales_metrics(y[ev], p[ev], cats[ev]) for n, p in preds.items()}
    groups = fc.model_groups(items)[ev]
    metrics = {"same-month ranking": lambda a, b, c: fc.sales_metrics(a, b, c)["spearman_same_month"],
               "ranking in category": lambda a, b, c: fc.sales_metrics(a, b, c)["spearman_within_cat"],
               "units error saved (log)": lambda a, b, c: -fc.sales_metrics(a, b, c)["mae_log"]}
    ci = {}
    for cb_name, lg_name in [("CatBoost no analogues", "LightGBM no analogues"),
                             ("CatBoost + analogues", "LightGBM + analogues"),
                             ("average LightGBM + CatBoost (no analogues)", "LightGBM no analogues")]:
        ci[cb_name] = {m: fc.bootstrap_diff(f, y[ev], preds[lg_name][ev], preds[cb_name][ev], groups, cats[ev])
                       for m, f in metrics.items()}
    shap = shap_check(models["no analogues"], Xs["no analogues"], ev) if explain else None
    return {"table": table, "ci": ci, "n_fit": int(fit_rows.sum()), "n_eval": int(ev.sum()), "shap": shap}


def compare_price(items, blocks, weights, split_args, params, explain: bool = False) -> dict:
    lgb_run = fc.run_price(items, blocks, weights, *fc.split(items, *split_args))
    y, fit_rows, ev, X0, X1, cost, has = price_setup(items, blocks, weights, split_args)
    points = np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique())
    preds = {k: v for k, v in lgb_run["preds"].items()
             if k in ("sub-family median", "cost × sub-family median markup", "similar past products (engine)",
                      "gradient boosting, no analogues", "gradient boosting + analogues")}
    preds = {("LightGBM " + k.split(", ")[-1].replace("gradient boosting ", "") if "boosting" in k else k): v
             for k, v in preds.items()}
    _, no_cost = fit_cb(X0.drop(columns="log_cost"), y, fit_rows, params)   # products without a landed cost
    models = {}
    for name, X in {"no analogues": X0, "+ analogues": X1}.items():
        models[name], pm = fit_cb(X, y - cost, fit_rows & has, params)
        preds[f"CatBoost {name}"] = np.where(has, cost + pm, no_cost)
    preds["average LightGBM + CatBoost (no analogues)"] = (preds["LightGBM no analogues"]
                                                           + preds["CatBoost no analogues"]) / 2
    table = {n: fc.price_metrics(y[ev], p[ev], points) for n, p in preds.items()}
    groups = fc.model_groups(items)[ev]
    metrics = {"€ of error saved": lambda a, b, _: -fc.price_metrics(a, b, points)["mae_eur"],
               "within 1 price point": lambda a, b, _: fc.price_metrics(a, b, points)["within_1_point"]}
    ci = {}
    for cb_name, lg_name in [("CatBoost no analogues", "LightGBM no analogues"),
                             ("CatBoost + analogues", "LightGBM + analogues"),
                             ("average LightGBM + CatBoost (no analogues)", "LightGBM no analogues")]:
        ci[cb_name] = {m: fc.bootstrap_diff(f, y[ev], preds[lg_name][ev], preds[cb_name][ev], groups, groups)
                       for m, f in metrics.items()}
    shap = shap_check(models["no analogues"], X0, ev & has) if explain else None
    return {"table": table, "ci": ci, "n_fit": int(fit_rows.sum()), "n_eval": int(ev.sum()), "shap": shap}


# --------------------------------------------------------------------------- output
def results_csv(res: dict) -> pd.DataFrame:
    rows = []
    for (target, split_name), r in res.items():
        for model, m in r["table"].items():
            rows.append({"target": target, "split": split_name, "model": model, "n_eval": r["n_eval"], **m})
    return pd.DataFrame(rows)


def _ci_lines(r: dict, digits: int = 3) -> str:
    out = []
    for name, d in r["ci"].items():
        out.append(f"- {name} minus LightGBM: " + "; ".join(f"{m} {lo:+.{digits}f} to {hi:+.{digits}f}"
                                                          for m, (lo, hi) in d.items()))
    return "\n".join(out)


def write_report(res: dict, tuned: dict, grids: dict) -> None:
    sp = {"top20_precision", "within_2x"}
    pp = {"mape", "exact_price_point", "within_1_point", "bias"}
    L = ["# Phase 5b: CatBoost against LightGBM", "",
         f"Generated by `src/phase5b_catboost.py run` ({datetime.now(timezone.utc):%Y-%m-%d}). Same data, splits, "
         "features, targets and metrics as `forecast_report.md`; intervals are 95% bootstrap over models "
         f"({fc.BOOT} resamples). Nothing of the delivered estimates is changed.", "",
         "## Protocol", "",
         f"- CatBoost `RMSE`, {ITERATIONS} trees, fixed seed, 4 threads, no early stopping. The categorical attributes "
         "are native `cat_features` (missing = its own level).",
         "- Small grid (depth 4/6/8 x learning rate 0.03/0.08 x L2 3/10) chosen on the **choice split only** "
         "(sales: first half of the new 241 models, mean absolute log error; price: 232 -> 241, mean absolute error in "
         "euros). The reported split is scored once with the chosen setting. LightGBM keeps the parameters of the "
         "forecast module (not tuned here), so the grid slightly favours CatBoost.",
         f"- Chosen: sales {tuned['sales']}; price {tuned['price']}.",
         "- Same variants as the forecast: with and without the similar past products as a feature; the price model "
         "estimates the markup over the landed cost and falls back to a model without cost.", ""]
    for key, title, pct, d in [(("sales", "test"), "Sales, test (232 -> second half of the new 241 models)", sp, 3),
                               (("sales", "242 ranking"), "Sales, ranking check on 242 (232 + 241 -> 242, cut sales)", sp, 3),
                               (("price", "test"), "Price, test (232 + 241 -> 242)", pp, 3)]:
        r = res[key]
        L += [f"## {title}: {r['n_fit']:,} training / {r['n_eval']:,} test products", "",
              fc._md(pd.DataFrame(r["table"]).T, pct=pct, digits=d), "",
              "Paired difference, CatBoost - LightGBM (95% CI; > 0 = CatBoost better):", "", _ci_lines(r), ""]
    for t in ("sales", "price"):
        g = grids[t].sort_values("val_error")
        L += [f"## Grid on the choice split: {t}", "", fc._md(g.set_index(pd.RangeIndex(len(g))), digits=4), ""]
    s, p = res[("sales", "test")]["shap"], res[("price", "test")]["shap"]
    L += ["## Per-prediction explanations", "",
          f"CatBoost gives exact per-prediction SHAP (`get_feature_importance(type='ShapValues')`): {s['ms_per_product']:.1f} "
          f"ms per sales product, {p['ms_per_product']:.1f} ms per price product, rows add up to the prediction "
          f"(max error {max(s['max_additivity_error'], p['max_additivity_error']):.1e}), so the Phase 5b family "
          "explanations could be rebuilt on it; `phase5b_forecast.py` still uses LightGBM TreeSHAP.", ""]
    (REPORT_DIR / "catboost_report.md").write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    items, blocks = fc.load()
    weights = engine.load_weights()
    t0 = time.perf_counter()
    tuned, grids = {}, {}
    tuned["sales"], grids["sales"] = tune_sales(items, blocks, weights)
    print("sales grid done", tuned["sales"], f"{time.perf_counter() - t0:.0f}s", flush=True)
    tuned["price"], grids["price"] = tune_price(items, blocks, weights)
    print("price grid done", tuned["price"], f"{time.perf_counter() - t0:.0f}s", flush=True)
    res = {("sales", "test"): compare_sales(items, blocks, weights, fc.SALES_TEST, tuned["sales"], explain=True),
           ("sales", "242 ranking"): compare_sales(items, blocks, weights, fc.TEST, tuned["sales"]),
           ("price", "test"): compare_price(items, blocks, weights, fc.TEST, tuned["price"], explain=True)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_csv(res).to_csv(REPORT_DIR / "catboost_results.csv", index=False, encoding="utf-8-sig")
    write_report(res, tuned, grids)
    for k, r in res.items():
        print(k, "\n", pd.DataFrame(r["table"]).T.round(3).to_string(), "\n", _ci_lines(r), flush=True)
    print(f"-> {REPORT_DIR / 'catboost_report.md'} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
