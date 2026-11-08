"""Phase 5b: sales estimate = average of LightGBM and CatBoost, trained with the weeks on sale.

    python src/phase5b_average.py run   # rewrites the sales columns of data/processed/forecast.parquet and
                                                  # outputs/phase5b/estimates.csv; report: outputs/phase5b/average_report.md

Run after `phase5b_forecast.py run` (it reads that forecast and replaces only the sales point estimate, its 80% range,
its typical error and its reasons; price, the buyer's second opinion and the similar products are untouched).

Why (progress.md §25.14): out of fold the average ranks products better than LightGBM (same-month ranking +0.010 to
+0.023) and, with the weeks on sale as a feature, matches its error; CatBoost alone is worse. Season 241 had only 15-53
weeks on sale at the extraction (232: 39-83), so the model sees `weeks_on_sale` when it learns and the delivered estimate
is made at a reference of 64 weeks (the median of 232): "units over a standard whole life" for every product, also
for the season 242 forecasts.

Delivered estimates are never in-sample: 5 folds grouped by model (as `phase5b_forecast.out_of_fold`) on every
whole-life product; the others use the model fitted on all of them. The 80% range is conformalised quantile regression
on the same folds (LightGBM 10th / 90th percentiles widened by one constant fitted on the out-of-fold products of
season 241), the typical error is measured out of fold on season 241 at the real weeks on sale.
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
import phase5b_catboost as cb  # noqa: E402
import phase5b_exposure as expo  # noqa: E402
import phase5b_forecast as fc  # noqa: E402
from config import KEY, PROCESSED_DIR  # noqa: E402

FOLDS = 5
REFERENCE_WEEKS = expo.REFERENCE_WEEKS
CALIBRATION_SEASON = "241"     # the closest season to a new product that is a whole life


def family(col: str) -> str:
    return "weeks on sale" if col == "weeks_on_sale" else fc.feature_family(col)


def cb_shap(model, X: pd.DataFrame) -> pd.DataFrame:
    from catboost import Pool
    Xc, cats = cb.cb_frame(X)
    sv = model.get_feature_importance(Pool(Xc, cat_features=cats), type="ShapValues")
    return pd.DataFrame(sv[:, :-1], columns=list(X.columns), index=X.index)


def out_of_fold(items, blocks, weeks) -> dict:
    """Out-of-fold average estimate (real weeks and reference weeks), quantiles and per-family SHAP, log scale."""
    y = items["log_sales"].to_numpy(float)
    known = np.isfinite(y) & fc.whole_life(items)
    allrows = np.ones(len(items), bool)
    X = expo.with_weeks(fc.feature_table(items, blocks, allrows, "log_sales", None), weeks)
    Xref = X.assign(weeks_on_sale=REFERENCE_WEEKS)
    Xc_real, _ = cb.cb_frame(X)
    Xc_ref, _ = cb.cb_frame(Xref)
    refs = fc.model_groups(items)
    uniq = np.unique(refs[known])
    fold_of = dict(zip(uniq, np.random.default_rng(fc.SEED).integers(0, FOLDS, len(uniq))))
    fold = np.array([fold_of.get(r, -1) for r in refs])
    n = len(items)
    out = {k: np.full(n, np.nan) for k in ("real", "ref", "lgb_ref", "lo_real", "hi_real", "lo_ref", "hi_ref")}
    fam = pd.DataFrame(index=items.index, dtype=float)
    parts = [(known & (fold != f), known & (fold == f)) for f in range(FOLDS)] + [(known, ~known)]
    for fit_rows, pred_rows in parts:
        if not pred_rows.any():
            continue
        lgb_m, _ = fc.fit_predict(X, y, allrows, fit_rows)
        cb_m, _ = cb.fit_cb(X, y, fit_rows, expo.CB_SALES)
        out["real"][pred_rows] = ((lgb_m.predict(X) + cb_m.predict(Xc_real)) / 2)[pred_rows]
        p_lgb_ref, p_cb_ref = lgb_m.predict(Xref), cb_m.predict(Xc_ref)
        out["ref"][pred_rows] = ((p_lgb_ref + p_cb_ref) / 2)[pred_rows]
        out["lgb_ref"][pred_rows] = p_lgb_ref[pred_rows]
        for alpha, tag in ((fc.RANGE_Q[0], "lo"), (fc.RANGE_Q[1], "hi")):
            q_m, _ = fc.fit_predict(X, y, allrows, fit_rows, alpha=alpha)
            out[f"{tag}_real"][pred_rows] = q_m.predict(X)[pred_rows]
            out[f"{tag}_ref"][pred_rows] = q_m.predict(Xref)[pred_rows]
        sv = (fc.shap_values(lgb_m, Xref[pred_rows]).drop(columns="expected_value")
              + cb_shap(cb_m, Xref[pred_rows])) / 2          # both additive in log units: exact for the average
        f_ = sv.T.groupby(family).sum().T
        fam = f_ if fam.empty else pd.concat([fam, f_])
    out["known"] = known
    out["fam"] = fam.reindex(items.index)
    return out


def reasons(fam: pd.DataFrame) -> list[str]:
    return ["; ".join(f"{f} {'+' if v > 0 else '−'}{abs(v):.2f}"
                      for f, v in r.dropna().reindex(r.dropna().abs().sort_values(ascending=False).index)[:3].items())
            for _, r in fam.iterrows()]


def deliver(items, blocks) -> tuple[pd.DataFrame, dict]:
    forecast = pd.read_parquet(PROCESSED_DIR / fc.OUT_FILE)
    assert (forecast[KEY].to_numpy() == items[KEY].to_numpy()).all(), "forecast and items are not in the same order"
    y = items["log_sales"].to_numpy(float)
    weeks = expo.weeks_on_sale(items)
    o = out_of_fold(items, blocks, weeks)
    known = o["known"]
    cal = known & items["SEA_COD"].astype(str).eq(CALIBRATION_SEASON).to_numpy()
    e = np.maximum(o["lo_real"] - y, y - o["hi_real"])[cal]
    q = float(np.quantile(e, min(1.0, (fc.RANGE_Q[1] - fc.RANGE_Q[0]) * (1 + 1 / len(e)))))
    lo = np.minimum(o["lo_ref"] - q, o["ref"])
    hi = np.maximum(o["hi_ref"] + q, o["ref"])
    cats = fc.month_groups(items)
    m = fc.sales_metrics(y[cal], o["real"][cal], cats[cal])
    old = forecast.copy()
    new = forecast.copy()
    new["sales_estimate_lightgbm"] = old["sales_estimate"]
    new["sales_estimate"] = np.expm1(o["ref"]).round(0)
    new["sales_low"] = np.expm1(np.maximum(lo, 0)).round(0)
    new["sales_high"] = np.expm1(hi).round(0)
    new["sales_typical_error"] = float(np.exp(m["mae_log"]))
    new["sales_reasons"] = reasons(o["fam"])
    # the range must bracket the rounded estimate (contract)
    new["sales_low"] = np.minimum(new["sales_low"], new["sales_estimate"])
    new["sales_high"] = np.maximum(new["sales_high"], new["sales_estimate"])
    info = coverage_and_comparison(items, y, o, old, new, cal, q, m)
    return new, info


def coverage_and_comparison(items, y, o, old, new, cal, q, m) -> dict:
    cats_c = items["CAT_DES_EN"].astype(str).to_numpy()
    sea = items["SEA_COD"].astype(str).to_numpy()
    known = o["known"]

    def cover(low, high, mask):
        inside = (y >= np.log1p(low)) & (y <= np.log1p(high))
        return float(inside[mask].mean())
    real_lo, real_hi = np.minimum(o["lo_real"] - q, o["real"]), np.maximum(o["hi_real"] + q, o["real"])
    rows = []
    for s in ("232", "241"):
        mask = known & (sea == s)
        by_cat = pd.Series(((y >= real_lo) & (y <= real_hi))[mask]).groupby(cats_c[mask]).mean()
        rows.append({"season": s, "n": int(mask.sum()),
                     "coverage (real weeks)": float(((y >= real_lo) & (y <= real_hi))[mask].mean()),
                     "coverage, worst category": float(by_cat.min()),
                     "mae_log average (real weeks)": float(np.mean(np.abs(y - o["real"])[mask])),
                     "mae_log LightGBM delivered before": float(np.mean(np.abs(y - np.log1p(old["sales_estimate"]
                                                                                             .to_numpy(float)))[mask])),
                     "bias_log average (real weeks)": float(np.mean((o["real"] - y)[mask]))})
    return {"table": pd.DataFrame(rows).set_index("season"), "q": q, "n_cal": int(cal.sum()),
            "typical": float(np.exp(m["mae_log"])), "within_2x": float(m["within_2x"]),
            "same_month": float(m["spearman_same_month"]),
            "reference_vs_real_241": float(np.mean((o["ref"] - o["real"])[known & (sea == "241")])),
            "median_change": float(np.median(new["sales_estimate"] / old["sales_estimate"].clip(lower=1)))}


def write_report(info: dict) -> None:
    t = info["table"]
    text = f"""# Sales estimate: LightGBM + CatBoost average with the weeks on sale

Generated by `src/phase5b_average.py` on {datetime.now(timezone.utc):%Y-%m-%d}. Replaces the sales point estimate, range,
typical error and reasons of `forecast.parquet`; the previous LightGBM estimate stays in `sales_estimate_lightgbm`.

- Estimate at a reference of {REFERENCE_WEEKS:.0f} weeks on sale (the median of season 232): units over a standard whole life.
- Typical error: x{info['typical']:.2f} (out of fold, season {CALIBRATION_SEASON}, real weeks on sale); within 2x: {info['within_2x']:.1%};
  same-month ranking {info['same_month']:.3f}. The earlier figure (x1.71) was a single split trained on 232 only: not the
  same experiment, do not compare the two numbers directly.
- 80% range: conformalised quantiles, constant widening {info['q']:.3f} (log) fitted on {info['n_cal']:,} out-of-fold products.
- Median change of the estimate against the previous LightGBM estimate: x{info['median_change']:.2f}.
- Mean difference between the reference-weeks and the real-weeks estimate in season 241 (log units): {info['reference_vs_real_241']:+.3f}.

{fc._md(t, digits=3)}

Out of fold, the estimates never saw the product's own sales. Season 241 is not a whole life (15-53 weeks on sale at the
extraction), so its errors are measured against partial lives; the estimate itself is made at the reference weeks.
"""
    (fc.REPORT_DIR / "average_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    items, blocks = fc.load()
    new, info = deliver(items, blocks)
    new.to_parquet(PROCESSED_DIR / fc.OUT_FILE, index=False)
    fc.estimates_csv(items, new).to_csv(fc.REPORT_DIR / "estimates.csv", index=False, encoding="utf-8")
    meta_path = PROCESSED_DIR / "forecast.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["sales_average"] = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                             "models": "LightGBM + CatBoost (mean of the two)", "catboost_params": expo.CB_SALES,
                             "reference_weeks": REFERENCE_WEEKS, "conformal_q": info["q"],
                             "typical_error": info["typical"], "within_2x": info["within_2x"]}
    meta_path.write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    write_report(info)
    print(info["table"].round(3).to_string())
    print(f"typical x{info['typical']:.2f}, q {info['q']:.3f} -> {PROCESSED_DIR / fc.OUT_FILE}")


if __name__ == "__main__":
    main()
