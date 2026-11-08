"""Phase 5b: where the largest price errors are, on the official price test (232 + 241 -> new models of 242).

    python src/phase5b_price_errors.py run      # -> outputs/phase5b/price_errors.md + price_errors.csv + price_realised_vs_list.csv, ~15 min

The professor asked about the size of the price errors. This script describes the products with the largest errors of
the chosen price model (`phase5b_forecast`, "gradient boosting, no analogues": landed cost x markup, products without a
cost use the model without cost). It does not change the model, its inputs or any other output; it reruns the same
code on the same split, so the overall numbers match `forecast_report.md`.

Decisions (each one is stated again in the report):
- Split: the official price test, not `estimates.csv`. `estimates.csv` is cross-validated across seasons (a product
  is scored by a model trained on the other folds, which include later seasons), so its errors are lower than on a
  forward test.
- "Worst": the 10% of test products with the largest error, measured twice: in euros (where most money is at stake)
  and in percent of the list price (where cheap products weigh as much as expensive ones). The two lists differ.
- Groups with fewer than 25 test products are pooled into one row: a share computed on a handful of products moves
  by tens of points with one product.
- Unusual markup: list price / landed cost outside 0.8-1.25 x the median markup of the product's category (median on
  the training seasons). This uses the actual price, so it describes an error after the fact; it is not a feature.
- Cost flags: the `certain` / `suspect` flags of `phase5b_cost_check.py`, recomputed with the same function.
- Products without a cost: the chosen model's fallback (model without cost) is compared with the similar-products
  (analogue) estimate on those products only.
- Example: the shopper `216630_BK` (season 241) is not in the test (241 trains the test model), so it is shown on the
  validation split 232 -> 241, where it is a new model, with the SHAP contributions of the markup model.
- List vs realised price (sales amount / units) is descriptive only. Whether `SALES_AMT_FX_RATE` includes VAT, and
  in which currencies the sales were made, is not documented: the gap is a question for Parfois, not a correction.
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
import phase5b_forecast as fc  # noqa: E402
from config import KEY  # noqa: E402

WORST_SHARE = 0.10          # share of test products called "the worst"
MIN_GROUP = 25              # smaller groups are pooled into one row
MARKUP_BAND = (0.8, 1.25)   # markup / category median markup inside this band = usual markup
PRICE_BANDS = [0, 10, 20, 40, 80, np.inf]
PRICE_LABELS = ["< 10 €", "10-20 €", "20-40 €", "40-80 €", ">= 80 €"]
EXAMPLE = "216630_BK"       # Shopper Bag MAXI DOTTS Black, real fur, markup 3.88x (see cost_check.md)
CHOSEN = "gradient boosting, no analogues"
ANALOGUE = "similar past products (engine)"
NO_COST = "gradient boosting without cost (previous model)"
GROUPS = {"category": "CAT_DES_EN", "family": "GFA_DES_EN", "collection type": "COLLECTION_TYPE",
          "fashion type": "FASHIONTYPE", "list price band": "price_band", "markup vs category": "markup_class",
          "landed cost": "cost_status", "cost flag": "cost_flag"}
OTHER = f"other (groups under {MIN_GROUP})"


# --------------------------------------------------------------------------- per-product table
def top_share(values: np.ndarray, share: float = WORST_SHARE) -> np.ndarray:
    """True for the `share` of rows with the largest values (ties broken by row order, so the result is stable)."""
    n = int(np.ceil(share * len(values)))
    out = np.zeros(len(values), bool)
    out[np.argsort(-np.asarray(values, float), kind="stable")[:n]] = True
    return out


def price_band(price: pd.Series) -> pd.Series:
    return pd.cut(price.astype(float), PRICE_BANDS, labels=PRICE_LABELS, right=False).astype(str)


def markup_class(ratio: pd.Series) -> pd.Series:
    """Markup over the category median: below / inside / above `MARKUP_BAND`; no cost when the ratio is missing."""
    lo, hi = MARKUP_BAND
    return pd.Series(np.select([ratio.isna(), ratio < lo, ratio > hi],
                               ["no cost", f"below {lo}x", f"above {hi}x"], f"{lo}-{hi}x"), index=ratio.index)


def error_table(items: pd.DataFrame, y: np.ndarray, preds: dict, eval_rows: np.ndarray, train: np.ndarray,
                flags: pd.Series) -> pd.DataFrame:
    """One row per test product: attributes, list price, cost, estimates, errors and the two "worst" flags."""
    price = items["PRICE_BASE_W_VAT"].astype(float)
    cost = np.exp(items["log_cost"])                       # NaN when the cost is missing or <= 0
    markup = price / cost
    cat_med = markup[train].groupby(items.loc[train, "CAT_DES_EN"]).median()
    ratio = markup / items["CAT_DES_EN"].map(cat_med)
    t = pd.DataFrame({KEY: items[KEY], "description": items["PROD_DES_BASE"], "season": items["SEA_COD"],
                      "CAT_DES_EN": items["CAT_DES_EN"], "GFA_DES_EN": items["GFA_DES_EN"],
                      "GFS_DES_EN": items["GFS_DES_EN"], "COLLECTION_TYPE": items["COLLECTION_TYPE"],
                      "FASHIONTYPE": items["FASHIONTYPE"], "list_price": price,
                      "landed_cost": pd.to_numeric(items["PRICE_LCP"], errors="coerce"),
                      "cost_status": np.where(cost.notna(), "has cost", "no cost (or <= 0)"),
                      "markup": markup, "markup_vs_category": ratio, "markup_class": markup_class(ratio),
                      "price_band": price_band(price), "cost_flag": flags.to_numpy(),
                      "estimate": np.exp(preds[CHOSEN]), "estimate_analogue": np.exp(preds[ANALOGUE]),
                      "estimate_without_cost": np.exp(preds[NO_COST])})
    t = t[eval_rows].reset_index(drop=True)
    for c in ("COLLECTION_TYPE", "FASHIONTYPE"):
        t[c] = t[c].astype("string").fillna("(missing)")
    t["error_eur"] = t["estimate"] - t["list_price"]
    t["abs_error_eur"] = t["error_eur"].abs()
    t["abs_error_pct"] = t["abs_error_eur"] / t["list_price"]
    t["direction"] = np.where(t["error_eur"] > 0, "too high", np.where(t["error_eur"] < 0, "too low", "exact"))
    t["worst_eur"] = top_share(t["abs_error_eur"].to_numpy())
    t["worst_pct"] = top_share(t["abs_error_pct"].to_numpy())
    return t


# --------------------------------------------------------------------------- summaries
def group_summary(t: pd.DataFrame, col: str, min_group: int = MIN_GROUP) -> pd.DataFrame:
    """Per group: size, share of the test, share of the worst (composition), share of the group that is in the worst
    (rate), mean errors and share estimated too high. Groups under `min_group` products are pooled."""
    size = t[col].value_counts()
    g = t[col].where(t[col].map(size) >= min_group, OTHER)
    n_we, n_wp = t["worst_eur"].sum(), t["worst_pct"].sum()
    out = t.assign(_g=g).groupby("_g").agg(
        n=(KEY, "size"), worst_eur=("worst_eur", "sum"), worst_pct=("worst_pct", "sum"),
        mae_eur=("abs_error_eur", "mean"), mape=("abs_error_pct", "mean"),
        too_high=("direction", lambda s: (s == "too high").mean()))
    out.insert(1, "share of test", out["n"] / len(t))
    out.insert(2, "share of worst €", out.pop("worst_eur") / n_we)
    out.insert(3, "in worst € (rate)", out["share of worst €"] * n_we / out["n"])
    out.insert(4, "share of worst %", out.pop("worst_pct") / n_wp)
    out.insert(5, "in worst % (rate)", out["share of worst %"] * n_wp / out["n"])
    out = out.rename(columns={"mae_eur": "MAE €", "mape": "MAPE", "too_high": "estimated too high"})
    order = out.drop(index=OTHER, errors="ignore").sort_values("n", ascending=False, kind="stable")
    out = pd.concat([order, out.loc[out.index == OTHER]])
    out.index.name = None
    return out


PCT_COLS = {"share of test", "share of worst €", "in worst € (rate)", "share of worst %", "in worst % (rate)", "MAPE",
            "estimated too high"}


def direction_table(t: pd.DataFrame) -> pd.DataFrame:
    sets = {"worst 10% in €": t["worst_eur"], "worst 10% in %": t["worst_pct"],
            "rest (not in either)": ~(t["worst_eur"] | t["worst_pct"]), "all test products": pd.Series(True, t.index)}
    rows = {name: {"n": int(m.sum()), "estimated too high": float((t.loc[m, "direction"] == "too high").mean()),
                   "median error €": float(t.loc[m, "error_eur"].median()),
                   "median abs error €": float(t.loc[m, "abs_error_eur"].median()),
                   "median abs error %": float(t.loc[m, "abs_error_pct"].median())} for name, m in sets.items()}
    return pd.DataFrame(rows).T.astype({"n": int})


def no_cost_comparison(t: pd.DataFrame, groups: np.ndarray, boot: int = fc.BOOT) -> dict:
    """On products without a cost: the chosen model's fallback against the analogue estimate (paired bootstrap by
    model, positive = the analogue has fewer euros of error)."""
    rows = {}
    for name, m in [("without cost", t["cost_status"] != "has cost"), ("with cost", t["cost_status"] == "has cost")]:
        d = t[m]
        rows[name] = {"n": len(d)}
        for label, col in [("chosen model", "estimate"), ("analogue", "estimate_analogue")]:
            e = (d[col] - d["list_price"]).abs()
            rows[name][f"MAE € {label}"] = float(e.mean()) if len(d) else np.nan
            rows[name][f"MAPE {label}"] = float((e / d["list_price"]).mean()) if len(d) else np.nan
    m = (t["cost_status"] != "has cost").to_numpy()
    ci = None
    if m.sum() >= 2:
        y = np.log(t.loc[m, "list_price"].to_numpy())
        a, b = np.log(t.loc[m, "estimate"].to_numpy()), np.log(t.loc[m, "estimate_analogue"].to_numpy())
        metric = lambda yy, pp, _: -float(np.mean(np.abs(np.exp(yy) - np.exp(pp))))  # noqa: E731
        ci = fc.bootstrap_diff(metric, y, a, b, groups[m], groups[m], n=boot)
    return {"table": pd.DataFrame(rows).T.astype({"n": int}), "ci": ci}


def realised_vs_list(items: pd.DataFrame) -> pd.DataFrame:
    """Realised average price (sales amount / units) over the list price, by category and season, products with
    sales only. Descriptive: VAT and currency of the amount are not documented."""
    qty = pd.to_numeric(items["SALES_QTY"], errors="coerce")
    amt = pd.to_numeric(items["SALES_AMT_FX_RATE"], errors="coerce")
    ratio = ((amt / qty).where(qty > 0) / items["PRICE_BASE_W_VAT"].astype(float)).rename("ratio")
    d = pd.DataFrame({"category": items["CAT_DES_EN"], "season": items["SEA_COD"].astype(str), "ratio": ratio,
                      "qty": qty}).dropna(subset=["ratio"])

    def summary(g: pd.DataFrame) -> pd.Series:
        return pd.Series({"n": len(g), "median": g["ratio"].median(), "p25": g["ratio"].quantile(0.25),
                          "p75": g["ratio"].quantile(0.75), "above list price": (g["ratio"] > 1.0).mean(),
                          "unit-weighted mean": np.average(g["ratio"], weights=g["qty"])})
    parts = [d.groupby(["category", "season"]).apply(summary, include_groups=False),
             d.assign(category="All").groupby(["category", "season"]).apply(summary, include_groups=False),
             d.assign(category="All", season="all").groupby(["category", "season"]).apply(summary, include_groups=False)]
    return pd.concat(parts).astype({"n": int})


def example(items, blocks, weights, key: str = EXAMPLE) -> dict | None:
    """The example product on the validation split (232 -> 241): estimate, analogue and the SHAP contributions of the
    markup model, summed by feature family (in log points; exp(sum) multiplies the markup)."""
    i = np.flatnonzero(items[KEY].to_numpy() == key)
    if not len(i):
        return None
    i = int(i[0])
    train, test = fc.split(items, *fc.VALIDATION)
    if not test[i]:
        return {"key": key, "in_test": False}
    r = fc.run_price(items, blocks, weights, train, test)
    model, X, _ = r["variants"][False]
    sv = fc.shap_values(model, X.iloc[[i]]).iloc[0]
    base = float(sv.pop("expected_value"))
    fam = sv.groupby(sv.index.map(fc.feature_family)).sum().sort_values(key=np.abs, ascending=False)
    cost = float(np.exp(items["log_cost"].iloc[i]))
    price = float(items["PRICE_BASE_W_VAT"].iloc[i])
    sub = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"]
    log_markup = r["y"] - items["log_cost"].to_numpy()

    def median_markup(cols: list[str]) -> tuple[float, int]:
        same = (items[cols].astype(str) == items.loc[i, cols].astype(str)).all(axis=1).to_numpy() & train
        m = log_markup[same & np.isfinite(log_markup)]
        return (float(np.exp(np.median(m))) if len(m) else np.nan), int(len(m))
    sub_markup, n_sub = median_markup(sub)
    cat_markup, n_cat = median_markup(["CAT_DES_EN"])
    return {"key": key, "in_test": True, "description": items["PROD_DES_BASE"].iloc[i],
            "sub_family": items["GFS_DES_EN"].iloc[i], "list_price": price, "cost": cost, "markup": price / cost,
            "estimate": float(np.exp(r["preds"][CHOSEN][i])), "analogue": float(np.exp(r["preds"][ANALOGUE][i])),
            "analogue_keys": r["analogue_keys"][i][:5], "sub_family_markup": sub_markup, "n_sub": n_sub,
            "category_markup": cat_markup, "n_cat": n_cat,
            "base_markup": float(np.exp(base)), "estimated_markup": float(np.exp(base + sv.sum())),
            "contributions": fam.head(6)}


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 2) -> str:
    """`phase5b_forecast._md` with integer columns kept as integers (a row of mixed types is read as floats)."""
    ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    return fc._md(df.astype({c: str for c in ints}), pct=pct, digits=digits)


def _share_line(s: pd.DataFrame, label: str, top: int = 3) -> str:
    """Plain sentences for the groups most over-represented among the worst (share of the worst above the share of
    the test), largest difference first."""
    s = s.drop(index=OTHER, errors="ignore")
    gap = (s[f"share of worst {label}"] - s["share of test"]).sort_values(ascending=False, kind="stable")
    keep = gap[gap > 0].head(top).index
    if not len(keep):
        return f"no group of {MIN_GROUP} test products or more is over-represented"
    return "; ".join(f"{g}: {s.loc[g, f'share of worst {label}']:.0%} of the worst against "
                     f"{s.loc[g, 'share of test']:.0%} of the test" for g in keep)


def worst_list(t: pd.DataFrame, flag: str, sort: str, n: int = 10) -> pd.DataFrame:
    cols = [KEY, "description", "CAT_DES_EN", "COLLECTION_TYPE", "list_price", "landed_cost", "estimate",
            "abs_error_eur", "abs_error_pct", "markup_vs_category", "cost_flag"]
    w = t[t[flag]].sort_values([sort, KEY], ascending=[False, True]).head(n)[cols]
    return w.set_index(KEY).round({"list_price": 2, "landed_cost": 2, "estimate": 2, "abs_error_eur": 2,
                                   "markup_vs_category": 2})


def write_report(t: pd.DataFrame, summaries: dict, direction: pd.DataFrame, nocost: dict, realised: pd.DataFrame,
                 ex: dict | None, overall: dict) -> str:
    md = _md
    n, nw = len(t), int(t["worst_eur"].sum())
    both = int((t["worst_eur"] & t["worst_pct"]).sum())
    sec = []
    for name, s in summaries.items():
        sec.append(f"### By {name}\n\n{md(s, pct=PCT_COLS, digits=2)}\n\n"
                   f"Over-represented among the worst in €: {_share_line(s, '€')}.  \n"
                   f"Over-represented among the worst in %: {_share_line(s, '%')}.")
    ci = nocost["ci"]
    ci_line = ("not computed (fewer than 2 products)" if ci is None else
               f"{ci[0]:+.2f} to {ci[1]:+.2f} € per product (95% interval, resampling models; positive = the analogue "
               "estimate has the smaller error)")
    if ex is None:
        ex_text = f"`{EXAMPLE}` is not in the data."
    elif not ex["in_test"]:
        ex_text = f"`{EXAMPLE}` is not a new model of 241 on the validation split; no example shown."
    else:
        sub_line = ("no product of this sub-family with a cost in 232" if not ex["n_sub"] else
                    f"{ex['sub_family_markup']:.2f}x ({ex['n_sub']} products)")
        contrib = pd.DataFrame({"log points": ex["contributions"], "x markup": np.exp(ex["contributions"])})
        ex_text = f"""`{ex['key']}` {ex['description']} (sub-family {ex['sub_family']}), validation split 232 -> 241 (it is a new model of 241).

| | value |
|---|---|
| list price | {ex['list_price']:.2f} € |
| landed cost | {ex['cost']:.2f} € |
| actual markup (price / cost) | {ex['markup']:.2f}x |
| median markup of its sub-family in 232 | {sub_line} |
| median markup of its category in 232 | {ex['category_markup']:.2f}x ({ex['n_cat']} products) |
| markup estimated by the model | {ex['estimated_markup']:.2f}x (starting value {ex['base_markup']:.2f}x) |
| estimated price (chosen model) | {ex['estimate']:.2f} € |
| analogue estimate | {ex['analogue']:.2f} € (first analogues: {', '.join(ex['analogue_keys'])}) |

Largest SHAP contributions to the estimated markup, summed by feature family (negative = lowers the markup):

{md(contrib, digits=2)}

Reading: the model starts from the average markup of its training rows and each family multiplies it by `x markup`. A large negative
contribution of `landed cost` means the model learned that products with a high cost tend to have a lower markup."""
    r_season = realised.xs("All", level="category")
    text = f"""# Largest price errors on the official price test

Generated by `src/phase5b_price_errors.py` on {datetime.now(timezone.utc):%Y-%m-%d}. Read-only: it reruns the price
model of `phase5b_forecast.py` on the same split and changes no other file. Per-product table: `price_errors.csv`.

## Setup and decisions

| Decision | Choice | Reason |
|---|---|---|
| Split | train 232 + 241, test the new models of 242 ({n} products) | the forward test reported in `forecast_report.md`; `estimates.csv` is cross-validated across seasons and gives lower errors |
| Model | `{CHOSEN}` | the chosen price model (landed cost x markup; no cost -> model without cost) |
| Worst | top {WORST_SHARE:.0%} by absolute error, in € and in % ({nw} products each; {both} are in both) | errors in € weigh expensive products, errors in % weigh cheap ones |
| Small groups | groups under {MIN_GROUP} test products pooled into `{OTHER}` | a share on a few products changes by tens of points with one product |
| Unusual markup | price / cost outside {MARKUP_BAND[0]}-{MARKUP_BAND[1]}x the median markup of the category (training seasons) | describes the product after the fact (uses the actual price); not a model feature |
| Cost flags | `certain` / `suspect` from `phase5b_cost_check.flag_costs` | same rules as `cost_check.md` |

Overall on this split: MAE {overall['mae_eur']:.3f} €, MAPE {overall['mape']:.1%}, exact price point
{overall['exact_price_point']:.1%}, within one price point {overall['within_1_point']:.1%}, bias {overall['bias']:+.1%}
(same numbers as `forecast_report.md`).

Column meaning in the tables below: `share of worst €` = of the worst products in €, the share in this group
(composition); `in worst € (rate)` = of this group's products, the share in the worst in € (expected
{WORST_SHARE:.0%} if errors were spread evenly). The same for %.

## Direction of the errors

{md(direction, pct={'estimated too high', 'median abs error %'}, digits=2)}

## Where the worst errors are

{chr(10).join(s + chr(10) for s in sec)}
## The ten largest errors in €

{md(worst_list(t, 'worst_eur', 'abs_error_eur'), pct={'abs_error_pct'}, digits=2)}

## The ten largest errors in %

{md(worst_list(t, 'worst_pct', 'abs_error_pct'), pct={'abs_error_pct'}, digits=2)}

## Products without a landed cost: fallback model or analogue estimate

{md(nocost['table'], pct={'MAPE chosen model', 'MAPE analogue'}, digits=2)}

Paired difference on the products without a cost: {ci_line}.

## Example: a product with a high cost and a high markup

{ex_text}

## List price and realised average price (descriptive)

Realised average price = `SALES_AMT_FX_RATE` / `SALES_QTY`, divided by the list price (`PRICE_BASE_W_VAT`), products
with at least one unit sold. All categories, by season:

{md(r_season, pct={'above list price'}, digits=2)}

By category and season: `price_realised_vs_list.csv`.

What is not known, and is a question for Parfois before reading the gap as discounts:

- whether `SALES_AMT_FX_RATE` includes VAT (the list price does);
- whether sales in other countries were made at other list prices before the conversion to euros;
- how much of the gap comes from outlet and promotions (season 242 had less time in the outlet at the extraction).

The price model estimates the list price, not the realised price. This section does not change that choice.
"""
    return text


# --------------------------------------------------------------------------- main
def run(boot: int) -> None:
    items, blocks = fc.load()
    weights = engine.load_weights()
    flags = cc.flag_costs(items)["cost_flag"]
    train, test = fc.split(items, *fc.TEST)
    r = fc.run_price(items, blocks, weights, train, test)
    t = error_table(items, r["y"], r["preds"], r["eval_rows"], train, flags)
    summaries = {name: group_summary(t, col) for name, col in GROUPS.items()}
    nocost = no_cost_comparison(t, fc.model_groups(items)[r["eval_rows"]], boot)
    realised = realised_vs_list(items)
    ex = example(items, blocks, weights)
    overall = r["table"][CHOSEN]
    fc.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(fc.REPORT_DIR / "price_errors.csv", index=False, encoding="utf-8-sig")
    realised.round(4).to_csv(fc.REPORT_DIR / "price_realised_vs_list.csv", encoding="utf-8-sig")
    (fc.REPORT_DIR / "price_errors.md").write_text(
        write_report(t, summaries, direction_table(t), nocost, realised, ex, overall), encoding="utf-8")
    print(f"test products {len(t)}, MAE {overall['mae_eur']:.3f} €; wrote", fc.REPORT_DIR / "price_errors.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--boot", type=int, default=fc.BOOT, help="bootstrap resamples for the no-cost comparison")
    args = ap.parse_args()
    run(args.boot)


if __name__ == "__main__":
    main()
