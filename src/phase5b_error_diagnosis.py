"""Phase 5b: why the estimates miss. Two read-only diagnoses that use no test product.

    python src/phase5b_error_diagnosis.py run   # -> outputs/phase5b/error_diagnosis.md + error_novelty.csv + error_noise.csv, ~10 min

1. Novelty (validation only). Do the errors grow when a new product resembles nothing in the training season?
   Price: validation split 232 -> new 241 models (where the price choices are made). Sales: the validation half of the
   new 241 models (`SALES_VALIDATION`; the other half is the sales test and is not used). Novelty is measured three ways
   on the training products only: the engine's similarity to the closest product and the mean over the 10 closest
   (same category, as in the engine), and whether the product's sub-family existed in the training season. The models
   are the validation versions of the chosen ones (`phase5b_forecast.run_price` / `run_sales`, LightGBM, no analogues);
   the delivered sales estimate (LightGBM + CatBoost average with the weeks on sale) is not refitted here.
   80% ranges are cross-fitted: the validation models are split in two halves by model, the 10th and 90th percentiles
   of real / estimate are taken on one half and applied to the other. The "new product" warning is tested the same
   way: one range for the products in the lowest similarity quintile and another for the rest.

2. Noise floor (price, training seasons 232 + 241 only). Products that are identical on the attributes that set a
   price (season, category, family, sub-family, material, finishing, landed cost within a band) can still have
   different prices. Each product is estimated by the median price of the other models of its group (leave one out):
   what is left is variation these attributes cannot explain. The price model's cross-validated estimate
   (`estimates.csv`) is scored on the same products for comparison. This is a floor only for a model that sees these
   attributes; the model sees more (colour, print, supplier, origin, segment, collection, weight, sizes, the exact
   cost), so it can do better on some products.

Decisions fixed before the run: quintiles of the mean similarity to the 10 closest products; "new product" = lowest
quintile; bootstrap by model (1,000 resamples); one colourway per model in the noise groups (colours of one model
usually share a price, which would make the groups look less noisy); groups need at least 2 models.
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

CHOSEN = "gradient boosting, no analogues"
SUBFAMILY = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"]
RANGE_Q = (0.1, 0.9)
N_BINS = 5
COST_BANDS = {"strict": 0.05, "loose": 0.10}     # landed cost band: +-5% / +-10% (log width 0.1 / 0.2)
NOISE_KEYS = {"strict": ["SEA_COD", *SUBFAMILY, "MATERIAL", "FINISHING"], "loose": ["SEA_COD", *SUBFAMILY]}
QUINTILE_LABELS = ["1 (least similar)", "2", "3", "4", "5 (most similar)"]


def joined(df: pd.DataFrame) -> pd.Series:
    """Columns joined into one key; a missing value is its own level (string columns keep NA through astype(str))."""
    return df.astype(object).where(df.notna(), "(missing)").astype(str).agg("|".join, axis=1)


# --------------------------------------------------------------------------- 1. novelty
def nearest_similarity(items: pd.DataFrame, blocks: dict, weights: dict, rows: np.ndarray, pool: np.ndarray,
                       k: int = fc.K_ANALOGUES) -> pd.DataFrame:
    """Engine similarity of each product in `rows` to the closest product of `pool` and the mean over the k closest
    (same category, other models), as the engine scores them."""
    has = engine.has_block(blocks)
    top1 = np.full(len(items), np.nan)
    topk = np.full(len(items), np.nan)
    ref = items["PROD_REF"].astype(str).to_numpy()
    for _, ix in items.groupby("CAT_DES_EN", sort=True).indices.items():
        ix = np.asarray(ix)
        cand, qs = ix[pool[ix]], ix[rows[ix]]
        if not len(cand) or not len(qs):
            continue
        for start in range(0, len(qs), engine.CHUNK):
            q = qs[start:start + engine.CHUNK]
            s = engine.score_matrix(blocks, has, weights, q, cand).astype(np.float64)
            s[ref[q][:, None] == ref[cand][None, :]] = -np.inf
            s = -np.sort(-s, axis=1)[:, :k]
            top1[q] = s[:, 0]
            topk[q] = np.where(np.isfinite(s), s, np.nan).mean(axis=1)
    return pd.DataFrame({"sim_closest": top1, "sim_top10": topk}, index=items.index)


def seen_subfamily(items: pd.DataFrame, pool: np.ndarray, require_cost: bool = False) -> np.ndarray:
    key = joined(items[SUBFAMILY])
    ok = pool & (np.isfinite(items["log_cost"].to_numpy(float)) if require_cost else pool)
    return key.isin(set(key[ok])).to_numpy()


def quintile(x: pd.Series) -> pd.Series:
    return pd.qcut(x.rank(method="first"), N_BINS, labels=QUINTILE_LABELS).astype(str)


def halves(groups: np.ndarray, seed: int = fc.SEED) -> np.ndarray:
    """0/1 per row, the same for every colourway of a model."""
    uniq = np.unique(groups)
    side = dict(zip(uniq, np.random.default_rng(seed).permutation(len(uniq)) % 2))
    return np.array([side[g] for g in groups])


def cross_fitted_range(resid: np.ndarray, side: np.ndarray, segment: np.ndarray | None = None) -> np.ndarray:
    """Covered or not, per row: the 80% range of a row comes from the residual quantiles of the other half (and of
    the same segment when one is given). `resid` = log(real) - log(estimate)."""
    seg = np.zeros(len(resid), int) if segment is None else np.asarray(segment)
    covered = np.zeros(len(resid), bool)
    for s in np.unique(seg):
        for h in (0, 1):
            fit, use = (seg == s) & (side != h), (seg == s) & (side == h)
            if fit.sum() < 10 or not use.any():
                covered[use] = False
                continue
            lo, hi = np.quantile(resid[fit], RANGE_Q)
            covered[use] = (resid[use] >= lo) & (resid[use] <= hi)
    return covered


def width(resid: np.ndarray, side: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Width (high / low, as a factor) of the cross-fitted range applied to each row."""
    out = np.full(len(resid), np.nan)
    for s in np.unique(segment):
        for h in (0, 1):
            fit, use = (segment == s) & (side != h), (segment == s) & (side == h)
            if fit.sum() >= 10:
                lo, hi = np.quantile(resid[fit], RANGE_Q)
                out[use] = np.exp(hi - lo)
    return out


def novelty_table(t: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    """Errors and range coverage by group of `col`."""
    agg = {"n": (KEY, "size"), "similarity (mean of 10)": ("sim_top10", "mean"),
           "coverage of the 80% range": ("covered", "mean")}
    if target == "price":
        agg |= {"MAE €": ("abs_eur", "mean"), "MAPE": ("abs_pct", "mean"), "estimated too high": ("too_high", "mean")}
    else:
        agg |= {"mae_log": ("abs_log", "mean"), "within 2x": ("within_2x", "mean"),
                "estimated too high": ("too_high", "mean")}
    out = t.groupby(col, sort=True).agg(**agg)
    out.index.name = None
    return out


def boot_ci(stat, groups: np.ndarray, n: int = fc.BOOT, seed: int = fc.SEED) -> tuple[float, float]:
    """95% interval of `stat(rows)` resampling models with replacement."""
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(stat(np.concatenate([members[i] for i in pick])))
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def novelty_frame(items, y, pred, rows, sims, seen, target: str) -> pd.DataFrame:
    t = pd.DataFrame({KEY: items[KEY], "model": fc.model_groups(items), "category": items["CAT_DES_EN"],
                      "sim_closest": sims["sim_closest"], "sim_top10": sims["sim_top10"],
                      "sub-family in training": np.where(seen, "seen", "not seen"),
                      "log_real": y, "log_est": pred})[rows].reset_index(drop=True)
    t["resid"] = t["log_real"] - t["log_est"]
    t["too_high"] = t["resid"] < 0
    if target == "price":
        real, est = np.exp(t["log_real"]), np.exp(t["log_est"])
        t["abs_eur"] = (real - est).abs()
        t["abs_pct"] = t["abs_eur"] / real
        t["err"] = t["abs_pct"]          # the price error compared across quintiles: percent of the price
    else:
        t["abs_log"] = t["resid"].abs()
        t["within_2x"] = t["abs_log"] < np.log(2)
        t["err"] = t["abs_log"]
    t["similarity quintile"] = quintile(t["sim_top10"])
    side = halves(t["model"].to_numpy())
    t["covered"] = cross_fitted_range(t["resid"].to_numpy(), side)
    t["new_product"] = t["similarity quintile"] == QUINTILE_LABELS[0]
    seg = t["new_product"].to_numpy().astype(int)
    t["covered_with_warning"] = cross_fitted_range(t["resid"].to_numpy(), side, seg)
    t["width_one_range"] = width(t["resid"].to_numpy(), side, np.zeros(len(t), int))
    t["width_with_warning"] = width(t["resid"].to_numpy(), side, seg)
    t["target"] = target
    return t


def novelty_summary(t: pd.DataFrame, boot: int) -> dict:
    g = t["model"].to_numpy()
    err, sim = t["err"].to_numpy(), t["sim_top10"].to_numpy()
    low = t["similarity quintile"].eq(QUINTILE_LABELS[0]).to_numpy()
    high = t["similarity quintile"].eq(QUINTILE_LABELS[-1]).to_numpy()
    rho = lambda r: fc.spearman(sim[r], err[r])  # noqa: E731
    gap = lambda r: float(np.mean(err[r][low[r]]) - np.mean(err[r][high[r]]))  # noqa: E731
    warn = t["new_product"].to_numpy()
    cov = {"one range": (float(t.loc[warn, "covered"].mean()), float(t.loc[~warn, "covered"].mean())),
           "with the warning": (float(t.loc[warn, "covered_with_warning"].mean()),
                                float(t.loc[~warn, "covered_with_warning"].mean()))}
    wid = {"one range": (float(t.loc[warn, "width_one_range"].median()), float(t.loc[~warn, "width_one_range"].median())),
           "with the warning": (float(t.loc[warn, "width_with_warning"].median()),
                                float(t.loc[~warn, "width_with_warning"].median()))}
    return {"n": len(t), "spearman": rho(np.arange(len(t))), "spearman_ci": boot_ci(rho, g, boot),
            "gap": gap(np.arange(len(t))), "gap_ci": boot_ci(gap, g, boot), "coverage": cov, "width": wid,
            "overall_coverage": float(t["covered"].mean())}


# --------------------------------------------------------------------------- 2. noise floor
def noise_groups(items: pd.DataFrame, keys: list[str], band: float) -> pd.DataFrame:
    """One colourway per model (the first by key) in the training seasons, with a cost, grouped by `keys` and the
    landed cost rounded to bands of +-`band`."""
    d = items[items["SEA_COD"].astype(str).isin(fc.COMPLETE_SEASONS) & np.isfinite(items["log_cost"])].copy()
    d = d.sort_values(KEY, kind="stable").drop_duplicates("PROD_REF")
    d["cost_band"] = np.floor(d["log_cost"] / (2 * band)).astype(int)
    d["group"] = joined(d[keys]) + "|" + d["cost_band"].astype(str)
    return d


def leave_one_out_median(price: pd.Series, group: pd.Series) -> pd.Series:
    """Median price of the other members of the group (NaN for a group of one)."""
    out = pd.Series(np.nan, index=price.index)
    for _, ix in price.groupby(group).groups.items():
        if len(ix) < 2:
            continue
        v = price.loc[ix].to_numpy(float)
        out.loc[ix] = [np.median(np.delete(v, i)) for i in range(len(v))]
    return out


def noise_floor(items: pd.DataFrame, oof: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per definition: products in groups of 2+ models, error of the leave-one-out group median, and the error of
    the model's cross-validated estimate on the same products."""
    rows, per_product = [], []
    for name, keys in NOISE_KEYS.items():
        d = noise_groups(items, keys, COST_BANDS[name])
        price = d["PRICE_BASE_W_VAT"].astype(float)
        d["loo_median"] = leave_one_out_median(price, d["group"])
        d["model_oof"] = d[KEY].map(oof)
        d = d[d["loo_median"].notna() & d["model_oof"].notna()]
        e_g = (d["loo_median"] - d["PRICE_BASE_W_VAT"]).abs()
        e_m = (d["model_oof"] - d["PRICE_BASE_W_VAT"]).abs()
        rows.append({"definition": name, "attributes": " + ".join(keys) + f" + cost within ±{COST_BANDS[name]:.0%}",
                     "groups": d["group"].nunique(), "products (models)": len(d),
                     "share of training models": len(d) / items.loc[items["SEA_COD"].astype(str).isin(
                         fc.COMPLETE_SEASONS), "PROD_REF"].nunique(),
                     "same price as the group median": float((e_g < 0.005).mean()),
                     "MAE € group median": float(e_g.mean()), "MAPE group median": float((e_g / d["PRICE_BASE_W_VAT"]).mean()),
                     "MAE € model (cross-validated)": float(e_m.mean()),
                     "MAPE model (cross-validated)": float((e_m / d["PRICE_BASE_W_VAT"]).mean())})
        per_product.append(d[[KEY, "SEA_COD", *SUBFAMILY, "PRICE_BASE_W_VAT", "PRICE_LCP", "group", "loo_median",
                              "model_oof"]].assign(definition=name))
    return pd.DataFrame(rows).set_index("definition"), pd.concat(per_product, ignore_index=True)


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 2) -> str:
    ints = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
    return fc._md(df.astype({c: str for c in ints}), pct=pct, digits=digits)


def _ci(ci, pct: bool = False) -> str:
    return f"{ci[0]:+.1%} to {ci[1]:+.1%}" if pct else f"{ci[0]:+.3f} to {ci[1]:+.3f}"


def novelty_section(name: str, t: pd.DataFrame, s: dict, target: str) -> str:
    pct = {"coverage of the 80% range", "MAPE", "estimated too high", "within 2x"}
    unit = "percent of the price" if target == "price" else "log units"
    cov, wid = s["coverage"], s["width"]
    gap = f"{s['gap']:+.1%}" if target == "price" else f"{s['gap']:+.3f}"
    gap_unit = "percentage points of the price" if target == "price" else "log units"
    warn = pd.DataFrame({"products": ["lowest quintile (warning)", "the other quintiles"],
                         "coverage, one range": [cov["one range"][0], cov["one range"][1]],
                         "coverage, range per group": [cov["with the warning"][0], cov["with the warning"][1]],
                         "median width, one range": [wid["one range"][0], wid["one range"][1]],
                         "median width, range per group": [wid["with the warning"][0], wid["with the warning"][1]]}
                        ).set_index("products")
    return f"""### {name}

{s['n']} products. Rank correlation between the similarity (mean of the 10 closest) and the error ({unit}):
{s['spearman']:+.3f} (95% interval {_ci(s['spearman_ci'])}; negative = less similar products have larger errors).
Error of the least similar quintile minus the most similar one: {gap} ({gap_unit}; 95% interval
{_ci(s['gap_ci'], pct=target == 'price')}).

By similarity quintile:

{_md(novelty_table(t, "similarity quintile", target), pct=pct, digits=3)}

By sub-family seen in the training season{' (with a landed cost)' if target == 'price' else ''}:

{_md(novelty_table(t, "sub-family in training", target), pct=pct, digits=3)}

"New product" warning (lowest quintile gets its own 80% range; width = high / low of the range):

{_md(warn, pct={'coverage, one range', 'coverage, range per group'}, digits=2)}
"""


def write_report(nov: dict, noise: pd.DataFrame) -> str:
    sections = "\n".join(novelty_section(n, t, s, t["target"].iloc[0]) for n, (t, s) in nov.items())
    return f"""# Why the estimates miss: novelty and noise floor

Generated by `src/phase5b_error_diagnosis.py` on {datetime.now(timezone.utc):%Y-%m-%d}. Read-only. Uses no test product:
price on the validation split 232 -> new 241 models, sales on the validation half of the new 241 models, the noise floor
on the training seasons 232 + 241. Per product: `error_novelty.csv`, `error_noise.csv`.

## 1. Are the errors larger for products that resemble nothing in the training season?

Similarity = the engine's score (same category, other models) against the products of the training season (232):
the closest one and the mean of the 10 closest. Quintiles of the mean of the 10 closest. The 80% ranges are
cross-fitted on two halves of the models (see the script's docstring), so coverage is measured on products that did
not set the range.

{sections}
## 2. How much price variation do the attributes leave? (training seasons 232 + 241)

One colourway per model, products with a landed cost. Each product is estimated by the median price of the other
models of its group (leave one out). The model column is the price model's cross-validated estimate (`estimates.csv`,
5 folds grouped by model) on the same products.

{_md(noise, pct={'share of training models', 'same price as the group median', 'MAPE group median', 'MAPE model (cross-validated)'}, digits=2)}

Reading: the group median uses only these attributes, so its error is what they leave unexplained. It is a floor for a
model with these attributes only; the price model also sees colour, print, supplier, origin, segment, collection type,
weight, number of sizes and the exact cost (no photo or description), so it can be below it. The cross-validated estimates of `estimates.csv` are optimistic (folds mix seasons), and the
groups cover only the products that share these attributes with another model.
"""


# --------------------------------------------------------------------------- main
def run(boot: int) -> None:
    items, blocks = fc.load()
    weights = engine.load_weights()
    nov = {}
    # price: validation split, all new 241 models
    train, test = fc.split(items, *fc.VALIDATION)
    r = fc.run_price(items, blocks, weights, train, test)
    rows = r["eval_rows"]
    sims = nearest_similarity(items, blocks, weights, rows, train)
    seen = seen_subfamily(items, train, require_cost=True)
    t = novelty_frame(items, r["y"], r["preds"][CHOSEN], rows, sims, seen, "price")
    nov["Price (validation 232 -> 241)"] = (t, novelty_summary(t, boot))
    # sales: the validation half only (the other half is the sales test)
    train, test = fc.split(items, *fc.SALES_VALIDATION)
    r = fc.run_sales(items, blocks, weights, train, test)
    rows = r["eval_rows"]
    sims = nearest_similarity(items, blocks, weights, rows, train)
    seen = seen_subfamily(items, train)
    t = novelty_frame(items, r["y"], r["preds"][CHOSEN], rows, sims, seen, "sales")
    nov["Sales (validation half of 241)"] = (t, novelty_summary(t, boot))
    # noise floor: training seasons, model's cross-validated price estimate from estimates.csv
    est = pd.read_csv(fc.REPORT_DIR / "estimates.csv", encoding="utf-8-sig", dtype={KEY: str})
    noise, per_product = noise_floor(items, est.set_index(KEY)["price_estimate"])
    fc.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([t for t, _ in nov.values()], ignore_index=True).to_csv(
        fc.REPORT_DIR / "error_novelty.csv", index=False, encoding="utf-8-sig")
    per_product.to_csv(fc.REPORT_DIR / "error_noise.csv", index=False, encoding="utf-8-sig")
    (fc.REPORT_DIR / "error_diagnosis.md").write_text(write_report(nov, noise), encoding="utf-8")
    for n, (_, s) in nov.items():
        print(n, "spearman", round(s["spearman"], 3), s["spearman_ci"])
    print(noise[["products (models)", "MAE € group median", "MAE € model (cross-validated)"]])
    print("wrote", fc.REPORT_DIR / "error_diagnosis.md")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--boot", type=int, default=fc.BOOT)
    run(ap.parse_args().boot)


if __name__ == "__main__":
    main()
