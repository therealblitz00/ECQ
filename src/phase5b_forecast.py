"""Phase 5b: explainable sales and price estimates for a product, from its attributes, photo and similar products.

    python src/phase5b_forecast.py run         # -> data/processed/forecast.parquet + outputs/phase5b/forecast_report.md

Business use (brief, slide 4): buyers must size the order of a NEW product. The engine already finds similar
products; here their sales (and prices) become an estimate, next to a model that learns from all attributes.

Data decisions (each measured in the report):
- Unit: the colourway. Sales targets: log(1 + units). Products without a sales row (370) or with 0 units (9) are
  unknown, not zero: they are left out of training and evaluation, and still get an estimate.
- Price: every colourway has a list price (no 0 and no missing value: checked in the report), so all are used.
- Time: SALES_QTY is accumulated until the extraction, so later launches had less time to sell (season 242 sells
  about a third of the older seasons). The model gets the days on sale; evaluation is temporal: trained on the two
  older seasons, tested on the newest one, on models never seen in training (truly new products).
- Buyer decisions (PUR_BET purchase depth, TYPOLOGY store tiers) are left out: they encode the buyer's own
  expectation of the sales we want to estimate. A reference model with them is reported, not used.
- The price model never sees the price (nor the numeric similarity block, which contains it); landed cost is left
  out too, since price = cost x markup would make the estimate trivial (reference reported).

Explanations: the gradient-boosting model gives exact per-prediction SHAP values (LightGBM `pred_contrib`, TreeSHAP:
no sampling, milliseconds per product); the analogue estimate is explained by the similar past products it averages.
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
import phase1_clean  # noqa: E402
import phase3_engine as engine  # noqa: E402
from config import KEY, PROCESSED_DIR, ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase5b"
OUT_FILE = "forecast.parquet"
SEED = 0
TEST_SEASON = "242"             # the newest season: forecast it from the two older ones
K_ANALOGUES = 10                # similar past products averaged by the analogue estimate
PCS = 16                        # principal components kept from the image and text embeddings
BOOT = 1000                     # bootstrap resamples (by model) for the confidence intervals
CATEGORICAL = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN", "CATEGORY_MATRIX", "CLR_DES", "CLR_TYPE", "FINISHING",
               "MATERIAL", "PRINT_TYPE", "DIMENSION", "OUTFIT", "FASHIONTYPE", "PROD_SEG", "NUMBER_OF_UNITS",
               "COLLECTION_TYPE", "MADE_IN", "SUP_COD"]
BUYER_DECISIONS = ["PUR_BET", "TYPOLOGY"]
LGB_PARAMS = dict(objective="regression_l2", learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=600,
                  random_state=SEED, deterministic=True, force_col_wise=True, num_threads=4, verbose=-1)


# --------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    items = phase1_clean.load_items().sort_values(KEY).reset_index(drop=True)
    blocks = engine.load_blocks(items)
    inferred = PROCESSED_DIR / "attributes_inferred.parquet"
    if inferred.exists():  # the reliable filled values of Phase 2a (material of apparel), as the engine uses them
        inf = pd.read_parquet(inferred)
        for col in ["MATERIAL"]:
            fill = inf[inf["column"] == col].set_index(KEY)["value"]
            items[col] = items[col].where(items[col].notna(), items[KEY].map(fill))
    d = pd.to_datetime(items["STORE_DATE_FINAL"])
    items["days_on_sale"] = (d.max() - d).dt.days.astype(float)
    items["launch_month"] = d.dt.month.astype(float)
    items["log_sales"] = np.log1p(items["SALES_QTY"].where(items["SALES_QTY"] > 0))
    items["log_price"] = np.log(items["PRICE_BASE_W_VAT"].astype(float))
    return items, blocks


def data_checks(items: pd.DataFrame) -> dict:
    """The facts the data decisions rest on (written to the report)."""
    price = items["PRICE_BASE_W_VAT"].astype(float)
    qty = items["SALES_QTY"]
    v = items.dropna(subset=["log_sales"])
    within = [g["log_sales"].rank().corr(g["days_on_sale"].rank())
              for _, g in v.groupby(["SEA_COD", "CAT_DES_EN"]) if len(g) > 30]
    return {"price_missing": int(price.isna().sum()), "price_zero_or_less": int((price <= 0).sum()),
            "price_min": float(price.min()), "price_max": float(price.max()), "price_points": int(price.nunique()),
            "cost_missing": int(items["PRICE_LCP"].isna().sum()) if "PRICE_LCP" in items else None,
            "sales_missing": int(qty.isna().sum()), "sales_zero": int((qty == 0).sum()),
            "median_units_by_season": items.groupby("SEA_COD")["SALES_QTY"].median().round(0).to_dict(),
            "days_vs_sales_within_season": float(np.median(within))}


def split(items: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Train: the older seasons. Test: the newest season, only models never sold in the older ones (new products)."""
    old = items["SEA_COD"].astype(str).ne(TEST_SEASON).to_numpy()
    seen = set(items.loc[old, "PROD_REF"].astype(str))
    new_model = ~items["PROD_REF"].astype(str).isin(seen).to_numpy()
    return old, (~old) & new_model


# --------------------------------------------------------------------------- features
def pca(train: np.ndarray, everything: np.ndarray, k: int = PCS) -> np.ndarray:
    """Principal components fitted on the training rows only (rows of zeros = no embedding stay zero)."""
    has = np.abs(everything).sum(axis=1) > 0
    t = train[np.abs(train).sum(axis=1) > 0]
    mu = t.mean(axis=0)
    _, _, vt = np.linalg.svd(t - mu, full_matrices=False)
    out = (everything - mu) @ vt[:k].T
    out[~has] = np.nan
    return out


def analogues(items: pd.DataFrame, blocks: dict, weights: dict, target: str, pool: np.ndarray,
              k: int = K_ANALOGUES) -> tuple[np.ndarray, list[list[str]]]:
    """Similarity-weighted mean of `target` over the k most similar products of `pool` (same category, other
    models): the engine's own answer to "how did products like this one do?". Also returns the analogue keys."""
    has = engine.has_block(blocks)
    y = items[target].to_numpy(float)
    est = np.full(len(items), np.nan)
    keys: list[list[str]] = [[] for _ in range(len(items))]
    ref = items["PROD_REF"].astype(str).to_numpy()
    for _, ix in items.groupby("CAT_DES_EN", sort=True).indices.items():
        ix = np.asarray(ix)
        cand = ix[pool[ix] & np.isfinite(y[ix])]
        for start in range(0, len(ix), engine.CHUNK):
            q = ix[start:start + engine.CHUNK]
            s = engine.score_matrix(blocks, has, weights, q, cand).astype(np.float64)
            s[ref[q][:, None] == ref[cand][None, :]] = -np.inf     # never the product's own model
            top = np.argsort(-s, axis=1, kind="stable")[:, :k]
            sc = np.take_along_axis(s, top, axis=1)
            w = np.clip(sc, 0, None)
            yy = y[cand][top]
            est[q] = np.where(w.sum(1) > 0, (w * yy).sum(1) / np.maximum(w.sum(1), 1e-12), np.nan)
            for qi, row in zip(q, top):
                keys[qi] = items[KEY].to_numpy()[cand[row]].tolist()
    return est, keys


def feature_table(items: pd.DataFrame, blocks: dict, train: np.ndarray, target: str,
                  analogue: np.ndarray | None, extra: list[str] = ()) -> pd.DataFrame:
    """One row per product: attributes (categorical), numbers, embedding components and the analogue estimate."""
    X = pd.DataFrame(index=items.index)
    for c in CATEGORICAL + list(extra):
        if c in items:
            X[c] = pd.Categorical(items[c].astype("string"))
    X["n_sizes"] = items["n_skus"].astype(float)
    X["net_weight"] = pd.to_numeric(items["NET_WEIGHT"], errors="coerce")
    X["launch_month"] = items["launch_month"]
    X["days_on_sale"] = items["days_on_sale"]
    if target == "log_sales":
        X["price"] = items["PRICE_BASE_W_VAT"].astype(float)
        X["price_pct_in_cat"] = items.groupby("CAT_DES_EN")["PRICE_BASE_W_VAT"].rank(pct=True)
    comp = blocks["composition"]
    for name, m in [("img", blocks["image"]), ("txt", blocks["text"])]:
        pcs = pca(m[train], m)
        for j in range(pcs.shape[1]):
            X[f"{name}_pc{j + 1}"] = pcs[:, j]
    top_mat = np.argsort(-(comp[train] > 0).sum(axis=0))[:20]
    for j in top_mat:
        X[f"mat_{j}"] = (comp[:, j] > 0).astype(float)
    if analogue is not None:
        X["analogue"] = analogue
    return X


FAMILY_OF_FEATURE = [("img_pc", "photo (image components)"), ("txt_pc", "description (text components)"),
                     ("mat_", "materials"), ("analogue", "similar past products"), ("price", "price"),
                     ("days_on_sale", "time on sale"), ("launch_month", "launch month"),
                     ("n_sizes", "number of sizes"), ("net_weight", "weight")]


def feature_family(col: str) -> str:
    for prefix, fam in FAMILY_OF_FEATURE:
        if col.startswith(prefix):
            return fam
    return "attribute: " + col


# --------------------------------------------------------------------------- models and metrics
def fit_predict(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, fit_rows: np.ndarray):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X[fit_rows], y[fit_rows])
    return model, model.predict(X)


def shap_values(model, X: pd.DataFrame) -> pd.DataFrame:
    """Exact TreeSHAP contributions (one column per feature + the expected value); rows add up to the prediction."""
    contrib = model.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib, columns=list(X.columns) + ["expected_value"], index=X.index)


def group_baseline(items: pd.DataFrame, y: np.ndarray, fit_rows: np.ndarray, cols: list[str]) -> np.ndarray:
    """Median of the target in the same group (falls back to the category, then to everything)."""
    df = items[cols].astype("string").assign(y=np.where(fit_rows, y, np.nan))
    pred = df.groupby(cols, dropna=False)["y"].transform("median")
    cat = df.groupby("CAT_DES_EN")["y"].transform("median")
    return pred.fillna(cat).fillna(np.nanmedian(y[fit_rows])).to_numpy(float)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation; 0 when one side is constant (a constant estimate ranks nothing)."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:   # exact test: a float std of equal values can be 1e-16, not 0
        return 0.0
    r = pd.Series(a).rank().corr(pd.Series(b).rank())
    return 0.0 if pd.isna(r) else float(r)


def sales_metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    """Ranking within the category (what a buyer compares), the same ranking only among products launched in the
    same month (time on sale removed), error in log units, and finding the top 20% of the category.

    `groups` holds "category|launch month"; the category is the part before the bar."""
    cats = np.array([g.split("|")[0] for g in groups])
    within = [spearman(y[cats == c], p[cats == c]) for c in np.unique(cats) if (cats == c).sum() > 10]
    month = [(spearman(y[groups == g], p[groups == g]), (groups == g).sum()) for g in np.unique(groups)
             if (groups == g).sum() > 10]
    top_true = pd.Series(y).groupby(cats).rank(pct=True).to_numpy() > 0.8
    # ties broken at random, so a constant estimate picks a random 20% (its fair score: the base rate)
    jitter = np.random.default_rng(SEED).random(len(p)) * 1e-9
    top_pred = pd.Series(p + jitter).groupby(cats).rank(pct=True).to_numpy() > 0.8
    return {"spearman_within_cat": float(np.mean(within)),
            "spearman_same_month": float(np.average([m for m, _ in month], weights=[n for _, n in month])),
            "mae_log": float(np.mean(np.abs(y - p))), "top20_precision": float(top_true[top_pred].mean())}


def price_metrics(y: np.ndarray, p: np.ndarray, points: np.ndarray) -> dict:
    """Error in euros and in price points (the 40 list prices), on the original scale."""
    ye, pe = np.exp(y), np.exp(p)
    # both sides as an index into the list prices (exp(log(p)) is not bit-exact, so the truth is snapped too)
    def nearest(v):
        return np.abs(v[:, None] - points[None, :]).argmin(axis=1)
    steps = np.abs(nearest(pe) - nearest(ye))
    return {"mae_eur": float(np.mean(np.abs(ye - pe))), "mape": float(np.mean(np.abs(ye - pe) / ye)),
            "exact_price_point": float((steps == 0).mean()), "within_1_point": float((steps <= 1).mean()),
            "spearman": spearman(y, p)}


def bootstrap_diff(metric, y, a, b, groups, extra, n: int = BOOT) -> tuple[float, float]:
    """95% interval of metric(b) - metric(a), resampling models (colourways of a model move together)."""
    rng = np.random.default_rng(SEED)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    diffs = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        diffs.append(metric(y[idx], b[idx], extra[idx]) - metric(y[idx], a[idx], extra[idx]))
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --------------------------------------------------------------------------- experiments
def run_sales(items, blocks, weights, train, test) -> dict:
    y = items["log_sales"].to_numpy(float)
    known = np.isfinite(y)
    fit_rows, eval_rows = train & known, test & known
    cats = (items["CAT_DES_EN"].astype(str) + "|" + items["launch_month"].astype(int).astype(str)
            + "|" + items["SEA_COD"].astype(str)).to_numpy()
    preds = {
        "category median": group_baseline(items, y, fit_rows, ["CAT_DES_EN"]),
        "family + price band median": group_baseline(
            items.assign(band=pd.qcut(items["PRICE_BASE_W_VAT"].rank(method="first"), 5, labels=False).astype(str)),
            y, fit_rows, ["CAT_DES_EN", "GFA_DES_EN", "band"]),
    }
    Xt = pd.DataFrame({"CAT_DES_EN": pd.Categorical(items["CAT_DES_EN"]), "days_on_sale": items["days_on_sale"],
                       "launch_month": items["launch_month"]})
    _, preds["time on sale + category only"] = fit_predict(Xt, y, train, fit_rows)
    ana, ana_keys = analogues(items, blocks, weights, "log_sales", fit_rows)
    preds["similar past products (engine)"] = np.where(np.isfinite(ana), ana, preds["category median"])
    X0 = feature_table(items, blocks, train, "log_sales", None)
    model, preds["gradient boosting, no analogues"] = fit_predict(X0, y, train, fit_rows)   # the chosen model
    X = feature_table(items, blocks, train, "log_sales", ana)
    _, preds["gradient boosting + analogues"] = fit_predict(X, y, train, fit_rows)
    Xb = feature_table(items, blocks, train, "log_sales", ana, extra=BUYER_DECISIONS)
    _, ref = fit_predict(Xb, y, train, fit_rows)
    table = {n: sales_metrics(y[eval_rows], p[eval_rows], cats[eval_rows]) for n, p in preds.items()}
    table["reference: + buyer decisions (not used)"] = sales_metrics(y[eval_rows], ref[eval_rows], cats[eval_rows])
    groups = items["PROD_REF"].astype(str).to_numpy()[eval_rows]
    m = lambda yy, pp, cc: sales_metrics(yy, pp, cc)["spearman_within_cat"]  # noqa: E731
    ci = {n: bootstrap_diff(m, y[eval_rows], preds["category median"][eval_rows], p[eval_rows], groups, cats[eval_rows])
          for n, p in preds.items() if n != "category median"}
    ci_ana = bootstrap_diff(m, y[eval_rows], preds["gradient boosting, no analogues"][eval_rows],
                            preds["gradient boosting + analogues"][eval_rows], groups, cats[eval_rows])
    m2 = lambda yy, pp, cc: sales_metrics(yy, pp, cc)["spearman_same_month"]  # noqa: E731
    ci_month = {n: bootstrap_diff(m2, y[eval_rows], preds["time on sale + category only"][eval_rows], p[eval_rows],
                                  groups, cats[eval_rows])
                for n, p in preds.items() if n not in ("category median", "time on sale + category only")}
    return {"y": y, "pred": preds["gradient boosting, no analogues"], "analogue": ana, "analogue_keys": ana_keys,
            "model": model, "X": X0, "table": table, "ci_vs_category": ci, "ci_analogue_gain": ci_ana, "ci_same_month": ci_month,
            "n_fit": int(fit_rows.sum()), "n_eval": int(eval_rows.sum()), "eval_rows": eval_rows}


def run_price(items, blocks, weights, train, test) -> dict:
    y = items["log_price"].to_numpy(float)
    fit_rows, eval_rows = train, test
    points = np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique())
    no_price = {**weights, "numeric": 0.0}          # the numeric block contains the price: never used here
    preds = {"category median": group_baseline(items, y, fit_rows, ["CAT_DES_EN"]),
             "sub-family median": group_baseline(items, y, fit_rows, ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"])}
    ana, ana_keys = analogues(items, blocks, no_price, "log_price", fit_rows)
    preds["similar past products (engine)"] = np.where(np.isfinite(ana), ana, preds["sub-family median"])
    X0 = feature_table(items, blocks, train, "log_price", None)
    model, preds["gradient boosting, no analogues"] = fit_predict(X0, y, train, fit_rows)   # the chosen model
    X = feature_table(items, blocks, train, "log_price", ana)
    _, preds["gradient boosting + analogues"] = fit_predict(X, y, train, fit_rows)
    Xc = X.assign(cost=pd.to_numeric(items["PRICE_LCP"], errors="coerce"))
    _, ref = fit_predict(Xc, y, train, fit_rows)
    table = {n: price_metrics(y[eval_rows], p[eval_rows], points) for n, p in preds.items()}
    table["reference: + landed cost (not used)"] = price_metrics(y[eval_rows], ref[eval_rows], points)
    groups = items["PROD_REF"].astype(str).to_numpy()[eval_rows]
    m = lambda yy, pp, _: -price_metrics(yy, pp, points)["mae_eur"]  # noqa: E731  (higher is better)
    ci = {n: bootstrap_diff(m, y[eval_rows], preds["sub-family median"][eval_rows], p[eval_rows], groups, groups)
          for n, p in preds.items() if n not in ("category median", "sub-family median")}
    return {"y": y, "pred": preds["gradient boosting, no analogues"], "analogue": ana, "analogue_keys": ana_keys,
            "model": model, "X": X0, "table": table, "ci_vs_subfamily": ci, "n_fit": int(fit_rows.sum()),
            "n_eval": int(eval_rows.sum()), "eval_rows": eval_rows}


# --------------------------------------------------------------------------- explanations
def global_importance(model, X: pd.DataFrame, rows: np.ndarray) -> pd.Series:
    """Mean |SHAP| per feature family on the evaluation rows (in units of the target).

    The family's effect is summed per product first, then made absolute: summing the absolute values of its columns
    would inflate families with many columns (16 photo components that partly cancel out)."""
    sv = shap_values(model, X[rows]).drop(columns="expected_value")
    fam = sv.T.groupby(lambda c: feature_family(c)).sum().T
    return fam.abs().mean().sort_values(ascending=False)


def explain_one(model, X: pd.DataFrame, i: int, n: int = 5) -> list[tuple[str, float]]:
    """The n features that moved this product's estimate most (SHAP, in units of the target)."""
    sv = shap_values(model, X.iloc[[i]]).iloc[0].drop("expected_value")
    fam = sv.groupby(sv.index.map(feature_family)).sum()
    return [(f, float(v)) for f, v in fam.reindex(fam.abs().sort_values(ascending=False).index)[:n].items()]


def final_models(items, blocks, weights) -> pd.DataFrame:
    """Estimates for every product, models refitted on all products with a known target (all seasons), plus the
    analogue estimate (similarity-weighted mean of the most similar products of other models) and its products."""
    allrows = np.ones(len(items), bool)
    out = pd.DataFrame({KEY: items[KEY]})
    for target, w in [("log_sales", weights), ("log_price", {**weights, "numeric": 0.0})]:
        y = items[target].to_numpy(float)
        fit_rows = np.isfinite(y)
        ana, keys = analogues(items, blocks, w, target, fit_rows)
        # final model without the analogue feature: adding it did not help significantly on the test season
        # (report), so the simpler model is kept; the analogue estimate is still given as the buyer's reference
        X = feature_table(items, blocks, allrows, target, None)
        model, pred = fit_predict(X, y, allrows, fit_rows)
        sv = shap_values(model, X)
        fam = sv.drop(columns="expected_value").T.groupby(lambda c: feature_family(c)).sum().T
        name = "sales" if target == "log_sales" else "price"
        out[f"{name}_estimate"] = np.expm1(pred).round(0) if name == "sales" else np.exp(pred).round(2)
        out[f"{name}_analogue"] = np.expm1(ana).round(0) if name == "sales" else np.exp(ana).round(2)
        out[f"{name}_analogue_keys"] = ["|".join(k[:4]) for k in keys]
        top = fam.apply(lambda r: r.reindex(r.abs().sort_values(ascending=False).index)[:3], axis=1)
        out[f"{name}_reasons"] = [
            "; ".join(f"{f} {'+' if v > 0 else '−'}{abs(v):.2f}" for f, v in r.dropna().items()) for _, r in top.iterrows()]
    return out


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        cells = [f"{v:.1%}" if c in pct and pd.notna(v) else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
                 for c, v in r.items()]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def write_report(checks: dict, sales: dict, price: dict, imp_s: pd.Series, imp_p: pd.Series,
                 examples: list[str], items: pd.DataFrame) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    st = pd.DataFrame(sales["table"]).T
    pt = pd.DataFrame(price["table"]).T
    ci_s = "\n".join(f"- {n}: {lo:+.3f} to {hi:+.3f}" for n, (lo, hi) in sales["ci_vs_category"].items())
    ci_p = "\n".join(f"- {n}: {lo:+.2f} to {hi:+.2f} € (less error is positive)"
                     for n, (lo, hi) in price["ci_vs_subfamily"].items())
    ex_lines = []
    it = items.set_index(KEY)
    for k in examples:
        i = int(np.flatnonzero(items[KEY].to_numpy() == k)[0])
        r = it.loc[k]
        es = explain_one(sales["model"], sales["X"], i)
        ep = explain_one(price["model"], price["X"], i)
        sold = "unknown" if pd.isna(r["SALES_QTY"]) else f"{int(r['SALES_QTY']):,}"
        ex_lines += [f"**{k}** ({r['GFS_DES_EN']}, {r['CLR_DES']}, season {r['SEA_COD']}): sold {sold}, "
                     f"estimate {np.expm1(sales['pred'][i]):,.0f}; price {r['PRICE_BASE_W_VAT']:.2f} €, "
                     f"estimate {np.exp(price['pred'][i]):.2f} €",
                     "- sales, biggest effects (log units): " + ", ".join(f"{f} {v:+.2f}" for f, v in es),
                     "- price, biggest effects (log units): " + ", ".join(f"{f} {v:+.2f}" for f, v in ep),
                     "- similar past products used by the analogue: " + ", ".join(sales["analogue_keys"][i][:4]), ""]
    lines = [
        "# Phase 5b: explainable sales and price estimates",
        "",
        f"Generated by `src/phase5b_forecast.py run` ({datetime.now(timezone.utc):%Y-%m-%d}). Trained on seasons "
        f"232 and 241; every number below is measured on season {TEST_SEASON}, on models never sold before "
        "(new products). Intervals: 95% bootstrap over models.",
        "",
        "## The data the decisions rest on",
        "",
        f"- **Price:** {checks['price_missing']} missing, {checks['price_zero_or_less']} at 0 € or less; "
        f"{checks['price_points']} list prices from {checks['price_min']:.2f} € to {checks['price_max']:.2f} €. "
        "Every product has a usable price.",
        f"- **Landed cost** (`PRICE_LCP`): missing for {checks['cost_missing']} colourways. Not used (see the reference row).",
        f"- **Sales:** {checks['sales_missing']} colourways have no sales row and {checks['sales_zero']} have 0 units: "
        "unknown, so left out of training and evaluation (they still get an estimate).",
        f"- **Time on sale:** median units by season {checks['median_units_by_season']}; inside a season and category, "
        f"later launches sold less (median Spearman of units vs days on sale "
        f"{checks['days_vs_sales_within_season']:+.2f}). Hence the `days_on_sale` feature and the temporal test.",
        "",
        f"## Sales (log units), {sales['n_fit']:,} training / {sales['n_eval']:,} test products",
        "",
        "`spearman_within_cat`: how well the estimate ranks the products of a category (what a buyer compares). "
        "`spearman_same_month`: the same, only among products of a category launched in the same month, so the time "
        "on sale cannot help. `top20_precision`: of the products the estimate puts in the top 20% of their category, "
        "the share that really were (a random pick scores 20%).",
        "",
        _md(st, pct={"top20_precision"}),
        "",
        "Gain in ranking over the category median:",
        "",
        ci_s,
        "",
        "Gain in ranking among products launched in the same month (time on sale removed), over the model that only "
        "knows the category and the time on sale:",
        "",
        "\n".join(f"- {n}: {lo:+.3f} to {hi:+.3f}" for n, (lo, hi) in sales["ci_same_month"].items()),
        "",
        f"Gain of adding the similar past products to the gradient boosting: {sales['ci_analogue_gain'][0]:+.3f} to "
        f"{sales['ci_analogue_gain'][1]:+.3f}. The interval includes 0, so the **chosen model is the gradient "
        "boosting without them** (simpler, as good); the similar past products are still reported per product as the "
        "buyer's reference.",
        "",
        "What drives the estimate (mean |SHAP|, log units, test products):",
        "",
        _md(imp_s.round(3).to_frame("mean |SHAP|")),
        "",
        f"## Price, {price['n_fit']:,} training / {price['n_eval']:,} test products",
        "",
        "The model never sees the price, the numeric similarity block or the cost. `within_1_point`: the estimate, "
        "rounded to the nearest of the 40 list prices, is the right one or its neighbour.",
        "",
        _md(pt, pct={"mape", "exact_price_point", "within_1_point"}),
        "",
        "Error reduction over the sub-family median:",
        "",
        ci_p,
        "",
        "What drives the estimate (mean |SHAP|, log units, test products):",
        "",
        _md(imp_p.round(3).to_frame("mean |SHAP|")),
        "",
        "## Examples (explanations per product)",
        "",
        *ex_lines,
        "## How the explanations are computed",
        "",
        "- **Gradient boosting** (LightGBM): SHAP values by TreeSHAP (`pred_contrib`). They are exact for the "
        "model (no sampling), add up to the estimate, and cost milliseconds per product, so every product gets them.",
        "- **Similar past products**: the estimate is their similarity-weighted mean, so the products themselves are "
        "the explanation (keys listed per product in `data/processed/forecast.parquet`).",
        "- Features are grouped into families (photo components, description components, materials, attributes…) "
        "before reporting, because single principal components have no business meaning.",
        "",
    ]
    (REPORT_DIR / "forecast_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    items, blocks = load()
    weights = engine.load_weights()
    train, test = split(items)
    checks = data_checks(items)
    sales = run_sales(items, blocks, weights, train, test)
    price = run_price(items, blocks, weights, train, test)
    imp_s = global_importance(sales["model"], sales["X"], sales["eval_rows"])
    imp_p = global_importance(price["model"], price["X"], price["eval_rows"])
    known_test = items.loc[sales["eval_rows"], KEY].tolist()
    examples = [known_test[i] for i in np.random.default_rng(SEED).choice(len(known_test), 3, replace=False)]
    write_report(checks, sales, price, imp_s, imp_p, examples, items)
    final = final_models(items, blocks, weights)
    final.to_parquet(PROCESSED_DIR / OUT_FILE, index=False)
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "test_season": TEST_SEASON,
            "k_analogues": K_ANALOGUES, "pcs": PCS, "lgb_params": LGB_PARAMS, "weights": weights,
            "sales_test": sales["table"], "price_test": price["table"]}
    (PROCESSED_DIR / "forecast.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    print(pd.DataFrame(sales["table"]).T.round(3).to_string())
    print(pd.DataFrame(price["table"]).T.round(3).to_string())
    print(f"-> {PROCESSED_DIR / OUT_FILE}, report {REPORT_DIR / 'forecast_report.md'}")


if __name__ == "__main__":
    main()
