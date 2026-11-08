"""Phase 5b: for products without a landed cost, the similar-products estimate against the model without cost.

    python src/phase5b_nocost_rule.py run      # -> outputs/phase5b/nocost_rule.md + nocost_rule.csv, ~10 min

The price model estimates price = landed cost x markup; products without a cost (missing or <= 0) fall back to the
gradient-boosting model without cost. On the official test (232 + 241 -> 242) the similar-products (analogue)
estimate had a smaller error on those 129 products (`price_errors.md`, +0.73 to +2.72 EUR per product). That finding
came from the test, so it is checked on the validation split before any change. This script changes no model and no
other output.

Decision rule, fixed before the run: the similar-products estimate replaces the fallback only if, on the validation
split (232 -> new 241 models), the 95% interval of the paired difference in euros of error is above 0 (the
similar-products estimate has the smaller error). The test split is reported, not used to decide.
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

FALLBACK = "gradient boosting without cost (previous model)"
ANALOGUE = "similar past products (engine)"
CHOSEN = "gradient boosting, no analogues"
SPLITS = {"validation 232 -> 241 (decides)": fc.VALIDATION, "test 232 + 241 -> 242 (reported)": fc.TEST}


def errors(y: np.ndarray, p: np.ndarray) -> dict:
    """Mean absolute error in euros and in percent of the list price (log inputs)."""
    ye, pe = np.exp(y), np.exp(p)
    return {"mae_eur": float(np.mean(np.abs(ye - pe))), "mape": float(np.mean(np.abs(ye - pe) / ye)),
            "too_high": float(np.mean(pe > ye))}


def compare(y: np.ndarray, fallback: np.ndarray, analogue: np.ndarray, chosen: np.ndarray, rows: np.ndarray,
            no_cost: np.ndarray, groups: np.ndarray, boot: int = fc.BOOT) -> dict:
    """On the evaluation rows: both estimates on the products without a cost, the paired intervals (positive = the
    similar-products estimate has the smaller error) and the overall error of the chosen model with and without
    the rule."""
    m = rows & no_cost
    out = {"n_eval": int(rows.sum()), "n_no_cost": int(m.sum())}
    if m.sum() < 2:
        return {**out, "fallback": {}, "analogue": {}, "ci_eur": None, "ci_pct": None, "overall": {}}
    out["fallback"] = errors(y[m], fallback[m])
    out["analogue"] = errors(y[m], analogue[m])
    eur = lambda yy, pp, _: -float(np.mean(np.abs(np.exp(yy) - np.exp(pp))))  # noqa: E731  (higher is better)
    pct = lambda yy, pp, _: -float(np.mean(np.abs(np.exp(yy) - np.exp(pp)) / np.exp(yy)))  # noqa: E731
    g = groups[m]
    out["ci_eur"] = fc.bootstrap_diff(eur, y[m], fallback[m], analogue[m], g, g, n=boot)
    out["ci_pct"] = fc.bootstrap_diff(pct, y[m], fallback[m], analogue[m], g, g, n=boot)
    with_rule = np.where(no_cost, analogue, chosen)
    out["overall"] = {"current": errors(y[rows], chosen[rows]), "with the rule": errors(y[rows], with_rule[rows])}
    return out


def decide(res: dict) -> bool:
    """The rule passes when the validation interval in euros is entirely above 0."""
    ci = res.get("ci_eur")
    return ci is not None and ci[0] > 0


def run_split(items, blocks, weights, args, boot: int) -> dict:
    train, test = fc.split(items, *args)
    r = fc.run_price(items, blocks, weights, train, test)
    no_cost = ~np.isfinite(items["log_cost"].to_numpy(float))
    p = r["preds"]
    return compare(r["y"], p[FALLBACK], p[ANALOGUE], p[CHOSEN], r["eval_rows"], no_cost,
                   fc.model_groups(items), boot)


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 2) -> str:
    """`phase5b_forecast._md` with integer columns kept as integers (a row of mixed types is read as floats)."""
    ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    return fc._md(df.astype({c: str for c in ints}), pct=pct, digits=digits)


def table(results: dict) -> pd.DataFrame:
    rows = []
    for split, r in results.items():
        for est in ("fallback", "analogue"):
            if r[est]:
                rows.append({"split": split, "products without cost": r["n_no_cost"],
                             "estimate": "model without cost (current)" if est == "fallback" else "similar products",
                             "MAE €": r[est]["mae_eur"], "MAPE": r[est]["mape"], "estimated too high": r[est]["too_high"]})
    return pd.DataFrame(rows)


def overall_table(results: dict) -> pd.DataFrame:
    rows = [{"split": s, "all test products": r["n_eval"], "variant": v, "MAE €": e["mae_eur"], "MAPE": e["mape"]}
            for s, r in results.items() for v, e in r["overall"].items()]
    return pd.DataFrame(rows)


def interval(ci, pct: bool = False) -> str:
    if ci is None:
        return "not computed (fewer than 2 products)"
    return f"{ci[0]:+.1%} to {ci[1]:+.1%}" if pct else f"{ci[0]:+.2f} to {ci[1]:+.2f} €"


def write_report(results: dict, passed: bool) -> str:
    md = _md
    t = table(results)
    o = overall_table(results)
    ci_lines = "\n".join(f"- {s}: {interval(r['ci_eur'])} per product; in percent of the price "
                         f"{interval(r['ci_pct'], pct=True)}" for s, r in results.items())
    verdict = ("passes: the similar-products estimate can replace the model without cost for products without a cost"
               if passed else "does not pass: the model without cost stays for products without a cost")
    return f"""# Products without a landed cost: similar-products estimate against the model without cost

Generated by `src/phase5b_nocost_rule.py` on {datetime.now(timezone.utc):%Y-%m-%d}. Read-only: no model or other output
changes. Per split: `nocost_rule.csv`.

## Why and how

The price model estimates price = landed cost x markup. Products without a cost (missing or <= 0) use the gradient-boosting
model without cost. In `price_errors.md` (official test 232 + 241 -> 242) the similar-products estimate had the smaller error
on those products. That observation came from the test, so measuring the change on the same test would give an optimistic
result. The decision is taken on the validation split, where the other choices of the price model were made.

Decision rule, fixed before the run: adopt the similar-products estimate for products without a cost only if the 95%
interval of the paired difference in euros on the validation split is above 0. The test split is reported only.

## Products without a cost

{md(t.set_index(["split", "estimate"]), pct={"MAPE", "estimated too high"}, digits=2)}

Paired difference (95% interval, resampling models; positive = the similar-products estimate has the smaller error):

{ci_lines}

## Effect on all evaluation products

{md(o.set_index(["split", "variant"]), pct={"MAPE"}, digits=3)}

## Decision

The rule {verdict}.

The idea came from the 242 test; the test numbers above are therefore not a blind check of the rule.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--boot", type=int, default=fc.BOOT)
    args = ap.parse_args()
    items, blocks = fc.load()
    weights = engine.load_weights()
    results = {name: run_split(items, blocks, weights, split, args.boot) for name, split in SPLITS.items()}
    passed = decide(results["validation 232 -> 241 (decides)"])
    fc.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    flat = [{"split": s, "n_eval": r["n_eval"], "n_no_cost": r["n_no_cost"],
             **{f"{e}_{k}": v for e in ("fallback", "analogue") for k, v in r[e].items()},
             "ci_eur_low": (r["ci_eur"] or (np.nan,) * 2)[0], "ci_eur_high": (r["ci_eur"] or (np.nan,) * 2)[1],
             "ci_pct_low": (r["ci_pct"] or (np.nan,) * 2)[0], "ci_pct_high": (r["ci_pct"] or (np.nan,) * 2)[1]}
            for s, r in results.items()]
    pd.DataFrame(flat).to_csv(fc.REPORT_DIR / "nocost_rule.csv", index=False, encoding="utf-8-sig")
    (fc.REPORT_DIR / "nocost_rule.md").write_text(write_report(results, passed), encoding="utf-8")
    for s, r in results.items():
        print(s, "products without cost:", r["n_no_cost"], "interval €:", r["ci_eur"])
    print("rule", "passes" if passed else "does not pass", "-> wrote", fc.REPORT_DIR / "nocost_rule.md")


if __name__ == "__main__":
    main()
