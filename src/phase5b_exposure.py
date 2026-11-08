"""Phase 5b: does the time on sale explain the sales error, and is the LightGBM + CatBoost average better out of fold?

    python src/phase5b_exposure.py run [--boot 500]   # -> outputs/phase5b/exposure_report.md + exposure_results.csv

The upstream audit found that season 241 had only 15-53 weeks on sale at the extraction (232: 39-83) and that the
residual of the sales model grows with the weeks on sale. Here `weeks_on_sale` (extraction date minus the final store
date, `phase5b_extraction.extraction_date`) is added as a feature and measured:

- Split A: train 232, test the new 241 models (choice half 0, reported half 1), as in the forecast module.
- Split B: train 232 + the choice half of 241, test the reported half. 232 never has fewer than 39 weeks, so a tree
  trained on it alone cannot learn the 15-39 week range; split B shows the 241 weeks to the model.
- Out of fold: 5-fold cross-fitting grouped by model on every whole-life product (232 + 241), the way the delivered
  estimates are made (`phase5b_forecast.out_of_fold`), for LightGBM, CatBoost and their average.

Two ways to score a model with the feature: at the real weeks on sale (what the 241 sales actually are), and at a
reference of 64 weeks (the 232 median: the whole-life estimate a buyer wants). Nothing here changes the delivered
model; the CatBoost settings are those chosen in `phase5b_catboost.py`.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_engine as engine  # noqa: E402
import phase5b_catboost as cb  # noqa: E402
import phase5b_extraction as ext_mod  # noqa: E402
import phase5b_forecast as fc  # noqa: E402

CB_SALES = {"depth": 6, "learning_rate": 0.03, "l2_leaf_reg": 10.0}   # chosen in phase5b_catboost.py
REFERENCE_WEEKS = 64.0                                                # median weeks on sale of season 232
FOLDS = 5


# --------------------------------------------------------------------------- data and metrics
def weeks_on_sale(items: pd.DataFrame) -> np.ndarray:
    ext, _ = ext_mod.extraction_date(items)
    return ((ext - pd.to_datetime(items["STORE_DATE_FINAL"])).dt.days / 7).to_numpy(float)


def wape(y: np.ndarray, p: np.ndarray) -> float:
    """Units missed over real units (inputs are log1p units)."""
    yu, pu = np.expm1(y), np.expm1(p)
    return float(np.abs(yu - pu).sum() / yu.sum())


def score(y, p, groups) -> dict:
    return {"same-month ranking": fc.sales_metrics(y, p, groups)["spearman_same_month"],
            "mae_log": float(np.mean(np.abs(y - p))), "wape": wape(y, p),
            "bias_log": float(np.mean(p - y))}


def fit_both(X: pd.DataFrame, y: np.ndarray, fit_rows: np.ndarray, X_pred: pd.DataFrame | None = None):
    """LightGBM, CatBoost and their average; predictions on `X_pred` (default: X)."""
    Xp = X if X_pred is None else X_pred
    lgb_model, _ = fc.fit_predict(X, y, np.ones(len(X), bool), fit_rows)
    cb_model, _ = cb.fit_cb(X, y, fit_rows, CB_SALES)
    p_l = lgb_model.predict(Xp)
    Xc, _ = cb.cb_frame(Xp)
    p_c = cb_model.predict(Xc)
    return {"LightGBM": p_l, "CatBoost": p_c, "average": (p_l + p_c) / 2}


def with_weeks(X: pd.DataFrame, weeks: np.ndarray) -> pd.DataFrame:
    return X.assign(weeks_on_sale=weeks)


# --------------------------------------------------------------------------- splits A and B
def run_split(items, blocks, weights, weeks, name: str, train_seasons, extra_rows_half: int | None, boot: int) -> dict:
    y = items["log_sales"].to_numpy(float)
    known = np.isfinite(y)
    train, test = fc.split(items, train_seasons, "241", 1)
    fit_rows = train & known
    if extra_rows_half is not None:        # split B: the choice half of the new 241 models joins the training
        _, half0 = fc.split(items, train_seasons, "241", extra_rows_half)
        fit_rows = fit_rows | (half0 & known)
    ev = test & known
    X0 = fc.feature_table(items, blocks, train, "log_sales", None)
    Xw = with_weeks(X0, weeks)
    Xref = with_weeks(X0, np.full(len(items), REFERENCE_WEEKS))
    cats = fc.month_groups(items)
    groups = fc.model_groups(items)[ev]
    preds = {}
    for label, p in fit_both(X0, y, fit_rows).items():
        preds[f"{label}, no exposure"] = p
    for label, p in fit_both(Xw, y, fit_rows).items():
        preds[f"{label}, + weeks on sale (real weeks)"] = p
    for label, p in fit_both(Xw, y, fit_rows, Xref).items():
        preds[f"{label}, + weeks on sale (64-week reference)"] = p
    rows = [{"split": name, "model": k, "n_eval": int(ev.sum()), **score(y[ev], p[ev], cats[ev])}
            for k, p in preds.items()]
    ci = {}
    for label in ("LightGBM", "CatBoost", "average"):
        a, b = preds[f"{label}, no exposure"], preds[f"{label}, + weeks on sale (real weeks)"]
        ci[label] = {
            "mae_log saved": fc.bootstrap_diff(lambda yy, pp, cc: -np.mean(np.abs(yy - pp)), y[ev], a[ev], b[ev],
                                               groups, cats[ev], n=boot),
            "same-month ranking": fc.bootstrap_diff(lambda yy, pp, cc: fc.sales_metrics(yy, pp, cc)["spearman_same_month"],
                                                    y[ev], a[ev], b[ev], groups, cats[ev], n=boot)}
    resid_corr = float(pd.Series(y[ev] - preds["LightGBM, no exposure"][ev]).corr(pd.Series(weeks[ev])))
    return {"rows": rows, "ci": ci, "resid_corr_no_exposure": resid_corr}


# --------------------------------------------------------------------------- out of fold
def oof_sales(items, blocks, weeks, use_weeks: bool) -> tuple[dict, np.ndarray]:
    """Cross-fitted estimates (folds grouped by model, as `phase5b_forecast.out_of_fold`) of every whole-life product."""
    y = items["log_sales"].to_numpy(float)
    known = np.isfinite(y) & fc.whole_life(items)
    allrows = np.ones(len(items), bool)
    X = fc.feature_table(items, blocks, allrows, "log_sales", None)
    if use_weeks:
        X = with_weeks(X, weeks)
    refs = fc.model_groups(items)
    uniq = np.unique(refs[known])
    fold_of = dict(zip(uniq, np.random.default_rng(fc.SEED).integers(0, FOLDS, len(uniq))))
    fold = np.array([fold_of.get(r, -1) for r in refs])
    est = {k: np.full(len(items), np.nan) for k in ("LightGBM", "CatBoost", "average")}
    for f in range(FOLDS):
        fit_rows, pred_rows = known & (fold != f), known & (fold == f)
        for k, p in fit_both(X, y, fit_rows).items():
            est[k][pred_rows] = p[pred_rows]
    return est, known


def run_oof(items, blocks, weeks, boot: int) -> dict:
    y = items["log_sales"].to_numpy(float)
    cats = fc.month_groups(items)
    sea = items["SEA_COD"].astype(str).to_numpy()
    rows, ci, store = [], {}, {}
    for use_weeks in (False, True):
        est, known = oof_sales(items, blocks, weeks, use_weeks)
        tag = "+ weeks on sale" if use_weeks else "no exposure"
        store[tag] = (est, known)
        for k, p in est.items():
            for part, mask in [("232 + 241", known), ("232", known & (sea == "232")), ("241", known & (sea == "241"))]:
                rows.append({"split": f"out of fold, {part}", "model": f"{k}, {tag}", "n_eval": int(mask.sum()),
                             **score(y[mask], p[mask], cats[mask])})
    groups = fc.model_groups(items)
    for tag in store:
        est, known = store[tag]
        g = groups[known]
        for k in ("CatBoost", "average"):
            ci[(tag, k)] = {
                "mae_log saved vs LightGBM": fc.bootstrap_diff(
                    lambda yy, pp, cc: -np.mean(np.abs(yy - pp)), y[known], est["LightGBM"][known], est[k][known],
                    g, cats[known], n=boot),
                "same-month ranking vs LightGBM": fc.bootstrap_diff(
                    lambda yy, pp, cc: fc.sales_metrics(yy, pp, cc)["spearman_same_month"], y[known],
                    est["LightGBM"][known], est[k][known], g, cats[known], n=boot)}
    return {"rows": rows, "ci": ci}


# --------------------------------------------------------------------------- report
def interval(t: tuple[float, float]) -> str:
    return f"{t[0]:+.3f} to {t[1]:+.3f}"


def write_report(weeks: np.ndarray, items, a: dict, b: dict, oof: dict) -> pd.DataFrame:
    table = pd.DataFrame(a["rows"] + b["rows"] + oof["rows"])
    table.to_csv(fc.REPORT_DIR / "exposure_results.csv", index=False, encoding="utf-8-sig")
    sea = items["SEA_COD"].astype(str)
    wk = pd.Series(weeks).groupby(sea.to_numpy()).describe()[["min", "50%", "max"]].round(0)

    def block(split_rows):
        d = pd.DataFrame(split_rows).drop(columns="split").set_index("model")
        return fc._md(d, digits=3)

    ci_lines = lambda r: "\n".join(f"- {k}: " + "; ".join(f"{m} {interval(v)}" for m, v in d.items())   # noqa: E731
                                   for k, d in r["ci"].items())
    oof_ci = "\n".join(f"- {tag}, {k}: " + "; ".join(f"{m} {interval(v)}" for m, v in d.items())
                       for (tag, k), d in oof["ci"].items())
    oof_tables = "\n\n".join(
        f"**{part}**\n\n" + block([r for r in oof["rows"] if r["split"] == f"out of fold, {part}"])
        for part in ("232 + 241", "232", "241"))
    text = f"""# Time on sale and the LightGBM + CatBoost average

Generated by `src/phase5b_exposure.py` on {datetime.now(timezone.utc):%Y-%m-%d}. Nothing here changes the delivered model.

`weeks_on_sale` = weeks between the final store date and the extraction (early November 2024). Weeks on sale at the
extraction by season:

{fc._md(wk, digits=0)}

Correlation of the LightGBM residual (no exposure) with the weeks on sale, new 241 models, reported half:
**{a['resid_corr_no_exposure']:+.2f}**.

Scores: same-month ranking (higher is better), `mae_log` (error in log units), `wape` (units missed over real units),
`bias_log` (positive: over-estimates). Intervals are 95%, resampling models; "saved" means the error went down.

## Split A: train 232, test the new 241 models (reported half)

{block(a['rows'])}

Effect of adding the weeks on sale (real weeks):

{ci_lines(a)}

## Split B: train 232 + the choice half of 241, test the reported half

{block(b['rows'])}

Effect of adding the weeks on sale (real weeks):

{ci_lines(b)}

## Out of fold (5 folds grouped by model, 232 + 241)

{oof_tables}

CatBoost and the average against LightGBM, same folds:

{oof_ci}

## How to read it

- "real weeks" scores the model against the 241 sales as they are (a partial life). "64-week reference" predicts every
  product as if it had sold for the median 232 time, the whole-life estimate a buyer wants; it is scored against the same
  partial-life sales, so it is not an error against a true whole life (which the data does not contain for 241).
- Split A trains on 232 only, whose weeks on sale never go below 39, so a tree cannot learn the 15-39 week range of 241.
"""
    (fc.REPORT_DIR / "exposure_report.md").write_text(text, encoding="utf-8")
    return table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    items, blocks = fc.load()
    engine.load_weights()
    weeks = weeks_on_sale(items)
    a = run_split(items, blocks, None, weeks, "A: train 232", ("232",), None, args.boot)
    b = run_split(items, blocks, None, weeks, "B: train 232 + 241 choice half", ("232",), 0, args.boot)
    oof = run_oof(items, blocks, weeks, args.boot)
    write_report(weeks, items, a, b, oof)
    print("wrote", fc.REPORT_DIR / "exposure_report.md")


if __name__ == "__main__":
    main()
