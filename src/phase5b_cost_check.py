"""Phase 5b: check the landed cost (`PRICE_LCP`) before the price model uses it.

    python src/phase5b_cost_check.py run [--boot 500]   # -> outputs/phase5b/cost_check.md + cost_flags.csv + cost_check_results.csv

The price model estimates price = landed cost x markup, so a wrong cost goes straight into the estimate. The team's
review of the largest errors found costs above the price (210243_GN: 63.65 EUR for a 12.99 EUR wallet) and costs 10x
away from the other colours of the same model. This script flags them (never deletes or edits the CSV) and measures
what happens to the price estimate when the model ignores a flagged cost.

Flags (`cost_flag`):
- `certain`: cost <= 0, cost above the price, or more than 5x away (either way) from the median cost of the other
  colours of the same model. The three tests do not need the price model.
- `suspect`: not `certain`, but the markup is far from its sub-family (robust z-score > 3.5). Reported, never acted on:
  some products really have another margin (216630_BK, a fur bag at 3.88x).

Variants scored on the same splits as the forecast module (validation 232 -> 241, test 232 + 241 -> 242):
current model; `certain` costs treated as missing (the model without cost); `certain` + `suspect` treated as missing.
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
import phase5b_forecast as fc  # noqa: E402
from config import KEY  # noqa: E402

RATIO_LIMIT = 5.0          # a colour whose cost is this many times above/below the other colours of its model
Z_LIMIT = 3.5              # robust z-score of the markup inside the sub-family
MIN_GROUP = 20             # smallest sub-family for the z-score
SUB = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"]
CHOSEN = "gradient boosting, no analogues"


# --------------------------------------------------------------------------- flags
def other_colours_cost(items: pd.DataFrame, cost: pd.Series) -> pd.Series:
    """Median cost of the other colourways of the same model (NaN when there are none with a cost)."""
    ref = items["PROD_REF"].astype(str)
    out = pd.Series(np.nan, index=items.index)
    for _, ix in items.groupby(ref).groups.items():
        if len(ix) < 2:
            continue
        c = cost[ix]
        for i in ix:
            rest = c.drop(i).dropna()
            if len(rest):
                out[i] = rest.median()
    return out


def flag_costs(items: pd.DataFrame) -> pd.DataFrame:
    """One row per colourway: cost, price, the three tests and the final `cost_flag` (none / suspect / certain)."""
    raw = pd.to_numeric(items["PRICE_LCP"], errors="coerce")
    price = items["PRICE_BASE_W_VAT"].astype(float)
    others = other_colours_cost(items, raw.where(raw > 0))
    ratio = raw / others
    markup = np.log(price / raw.where(raw > 0))
    group = items[SUB].astype(str).agg("|".join, axis=1)
    med = markup.groupby(group).transform("median")
    mad = (markup - med).abs().groupby(group).transform("median")
    size = markup.groupby(group).transform("count")
    z = (markup - med) / (1.4826 * mad.clip(lower=0.05))
    t = pd.DataFrame({KEY: items[KEY], "CAT_DES_EN": items["CAT_DES_EN"], "SEA_COD": items["SEA_COD"],
                      "cost": raw, "price": price, "other_colours_median_cost": others,
                      "cost_vs_other_colours": ratio, "markup": price / raw.where(raw > 0), "markup_z": z})
    t["test_cost_not_positive"] = raw.notna() & (raw <= 0)
    t["test_cost_above_price"] = raw > price
    t["test_far_from_other_colours"] = (ratio > RATIO_LIMIT) | (ratio < 1 / RATIO_LIMIT)
    certain = t[["test_cost_not_positive", "test_cost_above_price", "test_far_from_other_colours"]].any(axis=1)
    suspect = ~certain & (z.abs() > Z_LIMIT) & (size >= MIN_GROUP)
    t["cost_flag"] = np.where(certain, "certain", np.where(suspect, "suspect", "none"))
    return t


# --------------------------------------------------------------------------- scoring
def with_costs_removed(items: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    out = items.copy()
    out.loc[rows, "log_cost"] = np.nan
    return out


def run_variants(items, blocks, weights, flags: pd.DataFrame, boot: int) -> dict:
    variants = {"current (cost used as given)": np.zeros(len(items), bool),
                "certain costs ignored": (flags["cost_flag"] == "certain").to_numpy(),
                "certain + suspect costs ignored": (flags["cost_flag"] != "none").to_numpy()}
    results, rows = {}, []
    for split_name, args in [("validation 232 -> 241", fc.VALIDATION), ("test 232 + 241 -> 242", fc.TEST)]:
        train, test = fc.split(items, *args)
        runs = {}
        for v, drop in variants.items():
            runs[v] = fc.run_price(with_costs_removed(items, drop), blocks, weights, train, test)
        base = runs["current (cost used as given)"]
        ev = base["eval_rows"]
        y = base["y"]
        groups = fc.model_groups(items)[ev]
        flag_ev = flags["cost_flag"].to_numpy()[ev]
        for v, r in runs.items():
            for model in (CHOSEN, "gradient boosting + analogues"):
                p = r["preds"][model]
                m = fc.price_metrics(y[ev], p[ev], np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique()))
                row = {"split": split_name, "variant": v, "model": model, "n_eval": int(ev.sum()),
                       "mae_eur": m["mae_eur"], "mape": m["mape"], "within_1_point": m["within_1_point"],
                       "bias": m["bias"]}
                for sub in ("certain", "suspect"):
                    s = flag_ev == sub
                    row[f"n {sub}"] = int(s.sum())
                    row[f"mae_eur {sub}"] = (
                        float(np.mean(np.abs(np.exp(y[ev][s]) - np.exp(p[ev][s])))) if s.any() else np.nan)
                rows.append(row)
        metric = lambda yy, pp, _: -fc.price_metrics(yy, pp, np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique()))["mae_eur"]  # noqa: E731
        ci = {v: fc.bootstrap_diff(metric, y[ev], base["preds"][CHOSEN][ev], r["preds"][CHOSEN][ev], groups, groups, n=boot)
              for v, r in runs.items() if v != "current (cost used as given)"}
        results[split_name] = {"runs": runs, "ci": ci, "ev": ev, "y": y}
    return {"rows": pd.DataFrame(rows), "by_split": results}


def error_by_category(items, res: dict) -> pd.DataFrame:
    r = res["by_split"]["test 232 + 241 -> 242"]
    ev, y = r["ev"], r["y"]
    out = {}
    for v, run in r["runs"].items():
        p = run["preds"][CHOSEN]
        d = pd.DataFrame({"cat": items["CAT_DES_EN"].to_numpy()[ev], "err": np.abs(np.exp(y[ev]) - np.exp(p[ev])),
                          "pct": np.abs(np.exp(y[ev]) - np.exp(p[ev])) / np.exp(y[ev])})
        out[(v, "mae_eur")] = d.groupby("cat")["err"].mean()
        out[(v, "mape")] = d.groupby("cat")["pct"].mean()
    t = pd.DataFrame(out)
    t["n"] = pd.Series(items["CAT_DES_EN"].to_numpy()[ev]).value_counts()
    return t.sort_values("n", ascending=False)


# --------------------------------------------------------------------------- report
def interval(t) -> str:
    return f"{t[0]:+.3f} to {t[1]:+.3f}"


def write_report(items, flags: pd.DataFrame, res: dict) -> None:
    flags.to_csv(fc.REPORT_DIR / "cost_flags.csv", index=False, encoding="utf-8-sig")
    res["rows"].to_csv(fc.REPORT_DIR / "cost_check_results.csv", index=False, encoding="utf-8-sig")
    counts = flags.groupby(["cost_flag", "SEA_COD"]).size().unstack(fill_value=0)
    tests = flags[["test_cost_not_positive", "test_cost_above_price", "test_far_from_other_colours"]].sum()
    # costs <= 0 are already treated as missing by the model: the cases to ask about are the others
    clear = flags[(flags["cost_flag"] == "certain") & ~flags["test_cost_not_positive"]].assign(
        severity=lambda d: np.maximum(d["cost"] / d["price"], d["cost_vs_other_colours"].fillna(1).clip(lower=1e-6)
                                      .rpow(1).pipe(lambda r: np.maximum(r, 1 / r)))
    ).sort_values("severity", ascending=False).head(10)
    clear_t = clear[[KEY, "CAT_DES_EN", "cost", "price", "other_colours_median_cost"]].round(2)
    cat = error_by_category(items, res)
    cat.columns = [" / ".join(c) if isinstance(c, tuple) and c[1] else (c[0] if isinstance(c, tuple) else c)
                   for c in cat.columns]
    tbl = res["rows"][res["rows"]["model"] == CHOSEN].drop(columns="model").set_index(["split", "variant"])
    ci_lines = "\n".join(f"- {s}, {v}: euros saved per product {interval(c)}"
                         for s, r in res["by_split"].items() for v, c in r["ci"].items())
    text = f"""# Landed-cost check before the price model

Generated by `src/phase5b_cost_check.py` on {datetime.now(timezone.utc):%Y-%m-%d}. The CSV values are never changed; flags are in `cost_flags.csv`.
Model scored: `{CHOSEN}` (cost x markup); the same table for the model with analogues is in `cost_check_results.csv`.

## Flags

{fc._md(counts.reset_index().set_index("cost_flag"), digits=0)}

Tests behind `certain` (a product can fail more than one): {', '.join(f"{k.replace('test_', '').replace('_', ' ')} = {int(v)}" for k, v in tests.items())}.

The ten clearest cases to ask Parfois about:

{fc._md(clear_t.set_index(KEY), digits=2)}

## Price error with and without the flagged costs

{fc._md(tbl, pct={"mape", "within_1_point", "bias"}, digits=3)}

`n certain` / `n suspect`: evaluation products with that flag (costs <= 0 are missing for the model in every variant).

Paired change against the current model (95% interval, resampling models; positive = fewer euros of error):

{ci_lines}

## Error by category, test 232 + 241 -> 242 (euros and relative error)

{fc._md(cat, digits=3)}
"""
    (fc.REPORT_DIR / "cost_check.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    items, blocks = fc.load()
    weights = engine.load_weights()
    flags = flag_costs(items)
    res = run_variants(items, blocks, weights, flags, args.boot)
    write_report(items, flags, res)
    print(flags["cost_flag"].value_counts().to_dict())
    print("wrote", fc.REPORT_DIR / "cost_check.md")


if __name__ == "__main__":
    main()
