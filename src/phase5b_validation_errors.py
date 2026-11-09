"""Phase 5b: largest errors and 80% range coverage on the validation splits, and learning curves. No test product.

    python src/phase5b_validation_errors.py worst   # -> outputs/phase5b/validation_errors.md + validation_errors.csv, ~10 min
    python src/phase5b_validation_errors.py curve   # -> outputs/phase5b/learning_curve.md + learning_curve.csv, ~10 min

`worst`: the analysis of `phase5b_price_errors.py` (made on the 242 test) repeated on the validation splits, and the
first one for sales. Price: validation 232 -> new 241 models. Sales: the validation half of the new 241 models (the
other half is the sales test and is not used). Models: the validation versions of the chosen ones (LightGBM, no
analogues); the delivered sales estimate (LightGBM + CatBoost average with the weeks on sale) is not refitted here.
For each target the 10% largest errors are taken twice (price: in euros and in percent; sales: in units and as a
factor), described by group, and the 80% range coverage is given per group. Ranges are cross-fitted (two halves of
the models; percentiles of real / estimate from one half applied to the other), as in `phase5b_error_diagnosis.py`.
The 242 shares from `price_errors.csv` are put next to the 241 ones when that file exists (description only).

`curve`: is the error limited by the amount of training data or by the information in it? The chosen models are
refitted on 25%, 50%, 75% and 100% of the training models (three random draws by model below 100%) and scored on the
same validation products, together with the error on their own training rows.

Decisions fixed before the run: worst = top 10%; groups under 25 products pooled; seeds 0, 1, 2.
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
import phase5b_cost_check as cc  # noqa: E402
import phase5b_error_diagnosis as ed  # noqa: E402
import phase5b_exposure as expo  # noqa: E402
import phase5b_forecast as fc  # noqa: E402
import phase5b_price_errors as pe  # noqa: E402
from config import KEY  # noqa: E402

CHOSEN = "gradient boosting, no analogues"
MIN_GROUP = pe.MIN_GROUP
OTHER = pe.OTHER
FRACTIONS = (0.25, 0.5, 0.75, 1.0)
SEEDS = (0, 1, 2)
WEEK_BANDS = [0, 26, 39, np.inf]
WEEK_LABELS = ["< 26 weeks", "26-39 weeks", ">= 39 weeks"]
PRICE_GROUPS = {"category": "CAT_DES_EN", "family": "GFA_DES_EN", "collection type": "COLLECTION_TYPE",
                "fashion type": "FASHIONTYPE", "list price band": "price_band", "markup vs category": "markup_class",
                "cost flag": "cost_flag"}
SALES_GROUPS = {"category": "CAT_DES_EN", "family": "GFA_DES_EN", "collection type": "COLLECTION_TYPE",
                "fashion type": "FASHIONTYPE", "launch month": "launch_month", "weeks on sale at the extraction":
                "weeks_band", "real units (quintile)": "units_band", "list price band": "price_band"}
PRICE_ERRS = {"€": ("abs_error_eur", "worst_eur"), "%": ("abs_error_pct", "worst_pct")}
SALES_ERRS = {"units": ("abs_error_units", "worst_units"), "factor": ("abs_error_log", "worst_factor")}


# --------------------------------------------------------------------------- tables
def pooled(t: pd.DataFrame, col: str, min_group: int = MIN_GROUP) -> pd.Series:
    size = t[col].value_counts()
    return t[col].where(t[col].map(size) >= min_group, OTHER)


def summary(t: pd.DataFrame, col: str, errs: dict, min_group: int = MIN_GROUP) -> pd.DataFrame:
    """Per group: size, share of the products, for each error the share of the worst (composition) and the share of
    the group in the worst (rate), the mean error, the share estimated too high and the 80% range coverage."""
    g = pooled(t, col, min_group)
    out = pd.DataFrame({"n": t.groupby(g).size()})
    out["share of products"] = out["n"] / len(t)
    for label, (err, worst) in errs.items():
        w = t.groupby(g)[worst].sum()
        out[f"share of worst {label}"] = w / t[worst].sum()
        out[f"in worst {label} (rate)"] = w / out["n"]
        out[f"mean error {label}"] = t.groupby(g)[err].mean()
    out["estimated too high"] = t.groupby(g)["too_high"].mean()
    out["coverage of the 80% range"] = t.groupby(g)["covered"].mean()
    order = out.drop(index=OTHER, errors="ignore").sort_values("n", ascending=False, kind="stable")
    out = pd.concat([order, out.loc[out.index == OTHER]])
    out.index.name = None
    return out


def direction(t: pd.DataFrame, errs: dict) -> pd.DataFrame:
    sets = {f"worst 10% ({label})": t[worst] for label, (_, worst) in errs.items()}
    rest = ~np.logical_or.reduce([t[w] for _, w in errs.values()])
    sets |= {"rest (in neither)": rest, "all products": pd.Series(True, index=t.index)}
    rows = {}
    for name, m in sets.items():
        rows[name] = {"n": int(m.sum()), "estimated too high": float(t.loc[m, "too_high"].mean()),
                      "coverage of the 80% range": float(t.loc[m, "covered"].mean())}
        for label, (err, _) in errs.items():
            rows[name][f"median error {label}"] = float(t.loc[m, err].median())
    return pd.DataFrame(rows).T.astype({"n": int})


def add_coverage(t: pd.DataFrame, resid: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    return t.assign(covered=ed.cross_fitted_range(resid, ed.halves(groups)))


def price_frame(items, r, train, flags) -> pd.DataFrame:
    t = pe.error_table(items, r["y"], r["preds"], r["eval_rows"], train, flags)
    t["too_high"] = t["direction"].eq("too high")
    resid = np.log(t["list_price"]) - np.log(t["estimate"])
    return add_coverage(t, resid.to_numpy(), fc.model_groups(items)[r["eval_rows"]])


def sales_frame(items, r, weeks: np.ndarray) -> pd.DataFrame:
    rows = r["eval_rows"]
    y, p = r["y"], r["preds"][CHOSEN]
    t = pd.DataFrame({KEY: items[KEY], "description": items["PROD_DES_BASE"], "CAT_DES_EN": items["CAT_DES_EN"],
                      "GFA_DES_EN": items["GFA_DES_EN"], "COLLECTION_TYPE": items["COLLECTION_TYPE"],
                      "FASHIONTYPE": items["FASHIONTYPE"], "launch_month": items["launch_month"],
                      "weeks_on_sale": weeks, "list_price": items["PRICE_BASE_W_VAT"].astype(float),
                      "units": np.expm1(y), "estimate": np.expm1(p), "resid": y - p})[rows].reset_index(drop=True)
    for c in ("COLLECTION_TYPE", "FASHIONTYPE"):
        t[c] = t[c].astype("string").fillna("(missing)")
    t["launch_month"] = t["launch_month"].astype("Int64").astype("string").fillna("(missing)")
    t["weeks_band"] = pd.cut(t["weeks_on_sale"], WEEK_BANDS, labels=WEEK_LABELS, right=False).astype(str)
    t["units_band"] = pd.qcut(t["units"].rank(method="first"), 5,
                              labels=["1 (lowest)", "2", "3", "4", "5 (highest)"]).astype(str)
    t["price_band"] = pe.price_band(t["list_price"])
    t["abs_error_units"] = (t["estimate"] - t["units"]).abs()
    t["abs_error_log"] = t["resid"].abs()
    t["too_high"] = t["resid"] < 0
    t["worst_units"] = pe.top_share(t["abs_error_units"].to_numpy())
    t["worst_factor"] = pe.top_share(t["abs_error_log"].to_numpy())
    return add_coverage(t, t["resid"].to_numpy(), fc.model_groups(items)[rows])


def season_comparison(t241: pd.DataFrame, t242: pd.DataFrame | None, cols: dict) -> pd.DataFrame | None:
    """Share of the products and of the worst (€ and %) per group, validation 241 next to the test 242."""
    if t242 is None:
        return None
    keep = ["share of products", "share of worst €", "share of worst %"]
    blocks = []
    for name, col in cols.items():
        sides = [summary(t.assign(too_high=False, covered=np.nan), col, PRICE_ERRS)[keep].add_prefix(f"{season}: ")
                 for season, t in (("241", t241), ("242", t242))]
        b = sides[0].join(sides[1], how="outer")
        b.index = [f"{name}: {v}" for v in b.index]
        blocks.append(b)
    return pd.concat(blocks)


# --------------------------------------------------------------------------- learning curve
def subsample(train: np.ndarray, groups: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Training rows of a random `frac` of the training models (all colours of a model together)."""
    if frac >= 1:
        return train.copy()
    uniq = np.unique(groups[train])
    keep = set(np.random.default_rng(seed).permutation(uniq)[: int(round(frac * len(uniq)))])
    return train & np.array([g in keep for g in groups])


def fit_price(items, blocks, sub: np.ndarray) -> np.ndarray:
    """The chosen price model fitted on `sub` (markup over the cost; no cost -> model without cost)."""
    y, cost = items["log_price"].to_numpy(float), items["log_cost"].to_numpy(float)
    has = np.isfinite(cost)
    X0 = fc.feature_table(items, blocks, sub, "log_price", None)
    _, no_cost = fc.fit_predict(X0, y, sub, sub)
    _, pm = fc.fit_predict(X0.assign(log_cost=cost), y - cost, sub, sub & has)
    return np.where(has, cost + pm, no_cost)


def fit_sales(items, blocks, sub: np.ndarray) -> np.ndarray:
    y = items["log_sales"].to_numpy(float)
    X = fc.feature_table(items, blocks, sub, "log_sales", None)
    _, p = fc.fit_predict(X, y, sub, sub & np.isfinite(y))
    return p


def price_err(y, p, rows) -> dict:
    ye, pe_ = np.exp(y[rows]), np.exp(p[rows])
    return {"MAE €": float(np.mean(np.abs(ye - pe_))), "MAPE": float(np.mean(np.abs(ye - pe_) / ye))}


def sales_err(y, p, rows) -> dict:
    d = np.abs(y[rows] - p[rows])
    return {"mae_log": float(np.mean(d)), "within 2x": float(np.mean(d < np.log(2)))}


def learning_curve(items, blocks, target: str) -> pd.DataFrame:
    if target == "price":
        train, test = fc.split(items, *fc.VALIDATION)
        y, fit, err = items["log_price"].to_numpy(float), fit_price, price_err
    else:
        train, test = fc.split(items, *fc.SALES_VALIDATION)
        y, fit, err = items["log_sales"].to_numpy(float), fit_sales, sales_err
        known = np.isfinite(y)
        train, test = train & known, test & known
    groups = fc.model_groups(items)
    rows = []
    for frac in FRACTIONS:
        for seed in (SEEDS if frac < 1 else SEEDS[:1]):
            sub = subsample(train, groups, frac, seed)
            p = fit(items, blocks, sub)
            row = {"target": target, "fraction": frac, "seed": seed, "training models": len(np.unique(groups[sub])),
                   "training rows": int(sub.sum())}
            row |= {f"train {k}": v for k, v in err(y, p, sub).items()}
            row |= {f"validation {k}": v for k, v in err(y, p, test).items()}
            rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- reports
def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 2) -> str:
    ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    return fc._md(df.astype({c: str for c in ints}), pct=pct, digits=digits)


PCT = {"share of products", "share of worst €", "in worst € (rate)", "share of worst %", "in worst % (rate)",
       "mean error %", "share of worst units", "in worst units (rate)", "share of worst factor",
       "in worst factor (rate)", "estimated too high", "coverage of the 80% range", "median error %"}


def over_line(s: pd.DataFrame, label: str, top: int = 3) -> str:
    s = s.drop(index=OTHER, errors="ignore")
    gap = (s[f"share of worst {label}"] - s["share of products"]).sort_values(ascending=False, kind="stable")
    keep = gap[gap > 0].head(top).index
    if not len(keep):
        return f"no group of {MIN_GROUP} products or more is over-represented"
    return "; ".join(f"{g}: {s.loc[g, f'share of worst {label}']:.0%} of the worst against "
                     f"{s.loc[g, 'share of products']:.0%} of the products" for g in keep)


def coverage_line(s: pd.DataFrame) -> str:
    c = s.drop(index=OTHER, errors="ignore")["coverage of the 80% range"]
    return f"lowest {c.idxmin()} ({c.min():.0%}), highest {c.idxmax()} ({c.max():.0%})"


def section(title: str, t: pd.DataFrame, groups: dict, errs: dict) -> str:
    parts = [f"## {title}\n\n{len(t)} products.\n\n### Direction and coverage\n\n{_md(direction(t, errs), PCT)}\n"]
    for name, col in groups.items():
        s = summary(t, col, errs)
        lines = "  \n".join(f"Over-represented among the worst {label}: {over_line(s, label)}." for label in errs)
        parts.append(f"### By {name}\n\n{_md(s, PCT)}\n\n{lines}  \nCoverage of the 80% range: {coverage_line(s)}.\n")
    return "\n".join(parts)


def worst_report(tp: pd.DataFrame, ts: pd.DataFrame, comp: pd.DataFrame | None) -> str:
    comp_text = ("`price_errors.csv` (242 test) not found: no comparison." if comp is None else
                 _md(comp, pct=set(comp.columns)))
    return f"""# Largest errors and range coverage on the validation splits

Generated by `src/phase5b_validation_errors.py worst` on {datetime.now(timezone.utc):%Y-%m-%d}. Read-only; no test
product. Price: validation 232 -> new 241 models. Sales: validation half of the new 241 models. Per product:
`validation_errors.csv`.

Reading: `share of worst X` = of the worst products by X, the share in this group; `in worst X (rate)` = of the group's
products, the share in the worst (10% if errors were spread evenly). Price errors in euros (`€`) and in percent of the
list price (`%`); sales errors in units and as a factor (absolute log error: 0.69 = a factor of 2). Coverage = share of
products inside their cross-fitted 80% range (80% expected). Season 241 had 15 to 53 weeks on sale at the extraction,
so its sales are partial lives; the sales model here does not use the weeks on sale.

{section("Price (validation 232 -> 241)", tp, PRICE_GROUPS, PRICE_ERRS)}
### Same groups in the 242 test (`price_errors.md`)

Shares of the products and of the worst in each season (description only; nothing is chosen from it):

{comp_text}

{section("Sales (validation half of 241)", ts, SALES_GROUPS, SALES_ERRS)}"""


def curve_report(c: pd.DataFrame) -> str:
    parts = []
    for target, cols in (("price", ["MAE €", "MAPE"]), ("sales", ["mae_log", "within 2x"])):
        d = c[c["target"] == target]
        agg = {"training models": ("training models", "mean")}
        for k in cols:
            agg |= {f"train {k}": (f"train {k}", "mean"), f"validation {k}": (f"validation {k}", "mean"),
                    f"validation {k} (min)": (f"validation {k}", "min"), f"validation {k} (max)": (f"validation {k}", "max")}
        s = d.groupby("fraction").agg(**agg)
        s.index = [f"{f:.0%}" for f in s.index]
        pct = {k for k in s.columns if "MAPE" in k or "within" in k}
        first, last = s.iloc[-2], s.iloc[-1]
        k = cols[0]
        change = last[f"validation {k}"] - first[f"validation {k}"]
        gap = last[f"validation {k}"] - last[f"train {k}"]
        split = "232 -> 241" if target == "price" else "232 -> validation half of 241"
        parts.append(f"""## {target.capitalize()} ({split})

{_md(s.astype({'training models': int}), pct=pct, digits=3)}

From 75% to 100% of the training models the validation {k} changes by {change:+.3f}. At 100%, the validation {k} is
{gap:+.3f} above the error on the training rows.""")
    return f"""# Learning curves of the chosen price and sales models

Generated by `src/phase5b_validation_errors.py curve` on {datetime.now(timezone.utc):%Y-%m-%d}. Read-only; no test
product. The chosen models (LightGBM, no analogues; price = cost x markup, no cost -> model without cost) refitted on a
share of the training models (three random draws below 100%, all colours of a model together) and scored on the same
validation products. Per run: `learning_curve.csv`.

Reading: if the validation error still falls between 75% and 100%, more training seasons would likely help; if it is
flat, the limit is the information in the data. Tree models always fit their training rows more closely, so the gap
between training and validation error is expected; a gap that grows while the validation error stops falling points
to a model that adapts to its training rows more than the data supports.

{(chr(10) * 2).join(parts)}
"""


# --------------------------------------------------------------------------- main
def run_worst() -> None:
    items, blocks = fc.load()
    weights = engine.load_weights()
    flags = cc.flag_costs(items)["cost_flag"]
    train, test = fc.split(items, *fc.VALIDATION)
    tp = price_frame(items, fc.run_price(items, blocks, weights, train, test), train, flags)
    train, test = fc.split(items, *fc.SALES_VALIDATION)
    ts = sales_frame(items, fc.run_sales(items, blocks, weights, train, test), expo.weeks_on_sale(items))
    f242 = fc.REPORT_DIR / "price_errors.csv"
    t242 = pd.read_csv(f242, encoding="utf-8-sig", dtype={KEY: str}) if f242.exists() else None
    comp = season_comparison(tp, t242, {"category": "CAT_DES_EN", "collection type": "COLLECTION_TYPE",
                                        "list price band": "price_band", "markup vs category": "markup_class"})
    fc.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([tp.assign(target="price"), ts.assign(target="sales")], ignore_index=True).to_csv(
        fc.REPORT_DIR / "validation_errors.csv", index=False, encoding="utf-8-sig")
    (fc.REPORT_DIR / "validation_errors.md").write_text(worst_report(tp, ts, comp), encoding="utf-8")
    print("price", len(tp), "sales", len(ts), "-> wrote", fc.REPORT_DIR / "validation_errors.md")


def run_curve() -> None:
    items, blocks = fc.load()
    c = pd.concat([learning_curve(items, blocks, t) for t in ("price", "sales")], ignore_index=True)
    fc.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    c.to_csv(fc.REPORT_DIR / "learning_curve.csv", index=False, encoding="utf-8-sig")
    (fc.REPORT_DIR / "learning_curve.md").write_text(curve_report(c), encoding="utf-8")
    print(c.groupby(["target", "fraction"]).mean(numeric_only=True).round(3))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["worst", "curve"])
    args = ap.parse_args()
    if args.command == "worst":
        run_worst()
    else:
        run_curve()


if __name__ == "__main__":
    main()
