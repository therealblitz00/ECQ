"""Phase 5b: when were the sales extracted, and does the 232 -> 241 sales result hold on the 241 products that had
really finished selling?

    python src/phase5b_extraction.py   # -> outputs/phase5b/extraction_check.md, ~2 min

The extraction date is not given. `SALES_QTY` is accumulated until it, so products launched after it have no sales:
the date is read from the share of products without a sale by launch week. The outlet dates (`STORE_DATE_OUTLET_REAL`)
that fall after it are planned, not real. Then the delivered sales model (trained on 232, `FEATURES`) and the similar
past products are scored on the new 241 models (both halves) split by how long they had been on sale at the extraction,
and on the ones already in the outlet. If the 241 totals were cut, the late launches would show a positive bias (the
estimate above the cut truth) and a weaker ranking. Read-only: nothing delivered changes.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_engine as engine  # noqa: E402
import phase5b_forecast as fc  # noqa: E402

LAUNCH_WINDOW = ("2024-09-01", "2025-02-01")


def extraction_date(items: pd.DataFrame) -> tuple[pd.Timestamp, pd.DataFrame]:
    """First launch week in which most products (>= 50%, 20 or more) have no sale: the extraction falls in it."""
    start = pd.to_datetime(items["STORE_DATE_FINAL"])
    late = items[(start >= LAUNCH_WINDOW[0]) & (start < LAUNCH_WINDOW[1])]
    week = pd.to_datetime(late["STORE_DATE_FINAL"]).dt.to_period("W").dt.start_time
    units = late["SALES_QTY"]
    t = pd.DataFrame({"products": units.groupby(week).size(), "median units": units.groupby(week).median(),
                      "share without a sale": (units.fillna(0) == 0).groupby(week).mean()})
    t["products"] = t["products"].astype(int)
    first = t.index[(t["share without a sale"] >= 0.5) & (t["products"] >= 20)][0]
    return first, t


def counts_as_text(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Counts and medians printed as whole numbers (a table row of mixed types is read as floats)."""
    return df.assign(**{c: df[c].map(lambda v: "" if pd.isna(v) else f"{v:,.0f}") for c in cols})


def run() -> str:
    items, blocks = fc.load()
    weights = engine.load_weights()
    ext, weeks = extraction_date(items)
    sea = items["SEA_COD"].astype(str)
    start = pd.to_datetime(items["STORE_DATE_FINAL"])
    outlet = pd.to_datetime(items["STORE_DATE_OUTLET_REAL"], errors="coerce")
    s241 = sea.eq("241")
    outlet_share = {"with an outlet date": float(outlet[s241].notna().mean()),
                    "in the outlet before the extraction": float((outlet[s241] < ext).mean())}

    train, test0 = fc.split(items, *fc.SALES_VALIDATION)
    _, test1 = fc.split(items, *fc.SALES_TEST)
    y = items["log_sales"].to_numpy(float)
    fit_rows, ev = train & np.isfinite(y), (test0 | test1) & np.isfinite(y)
    _, gb = fc.fit_predict(fc.feature_table(items, blocks, train, "log_sales", None), y, train, fit_rows)
    ana, _ = fc.analogues(items, blocks, weights, "log_sales", fit_rows)
    cat_med = fc.group_baseline(items, y, fit_rows, ["CAT_DES_EN"])
    sim = np.where(np.isfinite(ana), ana, cat_med)
    groups = (items["CAT_DES_EN"].astype(str) + "|" + items["launch_month"].astype(int).astype(str) + "|"
              + sea).to_numpy()
    models = fc.model_groups(items)
    months_on_sale = ((ext - start).dt.days / 30.44).to_numpy()
    subsets = {
        "all new 241 models (both halves)": ev,
        "launched Nov 2023 - Jan 2024 (>= 9 months on sale)": ev & (months_on_sale >= 9),
        "launched Feb - Jul 2024 (< 9 months on sale)": ev & (months_on_sale < 9),
        "in the outlet before the extraction": ev & (outlet < ext).to_numpy(),
        "not yet in the outlet at the extraction": ev & ~(outlet < ext).to_numpy(),
    }
    m_month = lambda yy, pp, cc: fc.sales_metrics(yy, pp, cc)["spearman_same_month"]  # noqa: E731
    rows = []
    for name, r in subsets.items():
        g, s = fc.sales_metrics(y[r], gb[r], groups[r]), fc.sales_metrics(y[r], sim[r], groups[r])
        lo, hi = fc.bootstrap_diff(m_month, y[r], sim[r], gb[r], models[r], groups[r], n=300)
        rows.append({"subset": name, "products": int(r.sum()), "median units": float(np.median(np.expm1(y[r]))),
                     "same-month ranking": g["spearman_same_month"], "similar products": s["spearman_same_month"],
                     "gain, 95% CI": f"{lo:+.3f} to {hi:+.3f}", "error factor": float(np.exp(g["mae_log"])),
                     "within 2x": g["within_2x"], "bias": float(np.exp(g["bias_log"]) - 1),
                     "top-20% precision": g["top20_precision"]})
    df = pd.DataFrame(rows).set_index("subset")
    by_month = pd.DataFrame({"month": start[ev].dt.to_period("M").astype(str).to_numpy(),
                             "bias": gb[ev] - y[ev]}).groupby("month")["bias"].agg(["size", "mean"])
    by_month["bias"] = np.exp(by_month.pop("mean")) - 1
    text = f"""# When were the sales extracted, and does the 241 result hold?

Generated by `src/phase5b_extraction.py` ({datetime.now(timezone.utc):%Y-%m-%d}). Read-only.

## Extraction date

Products launched after the extraction have no sales. Share without a sale by launch week (`STORE_DATE_FINAL`):

{fc._md(counts_as_text(weeks, ["products", "median units"]).reset_index(names="launch week").assign(**{"launch week": lambda d: d["launch week"].dt.strftime("%Y-%m-%d")}).set_index("launch week"), pct={"share without a sale"})}

**The sales were extracted in the week of {ext:%d %B %Y}**, not about January 2025 (the last store date, a planned
one). Season 241: {100 * outlet_share['with an outlet date']:.0f}% of the products have an outlet date, but only
{100 * outlet_share['in the outlet before the extraction']:.0f}% had gone to the outlet before the extraction; the later
dates are planned.

## The sales model on the new 241 models, by time on sale

Model trained on 232 (`FEATURES`, as delivered), scored on both halves of the new 241 models. Gain: same-month ranking
of the model minus the similar past products, 95% interval (paired bootstrap by model, 300 resamples).

{fc._md(counts_as_text(df, ["products", "median units"]), pct={"within 2x", "bias", "top-20% precision"})}

Bias by launch month (estimate / truth − 1; a cut truth would show as a positive bias for the late months):

{fc._md(counts_as_text(by_month, ["size"]), pct={"bias"})}
"""
    (fc.REPORT_DIR / "extraction_check.md").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    print(run())
