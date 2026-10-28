"""Phase 5b: explainable sales and price estimates for a product, from its attributes, photo and similar products.

    python src/phase5b_forecast.py run         # -> data/processed/forecast.parquet + outputs/phase5b/ (report, leakage checks)

Business use (brief, slide 4): buyers must size the order of a NEW product. The engine already finds similar
products; here their sales (and prices) become an estimate, next to a model that learns from all attributes.

Data decisions (each measured in the report):
- Unit: the colourway. Sales targets: log(1 + units). Products without a sales row (370) or with 0 units (9) are
  unknown, not zero: they are left out of training and evaluation, and still get an estimate.
- Price: every colourway has a list price (no 0 and no missing value: checked in the report), so all are used.
- Time: SALES_QTY is accumulated until the extraction (about January 2025). Seasons 232 and 241 had finished selling
  (98% of 241 already went to the outlet); season 242 had not (products launched in October 2024 show a median of 143
  units against ~3,300 for a whole life). The sales target is therefore the **whole-life units of seasons 232 and
  241 only**: trained on 232, tested on the new models of 241 (half of them for the choices, the other half reported
  once). 242 sales never train or score the units; they only check the ranking among products launched in the same
  month (equally cut). The days to the extraction are not a feature: they describe the extraction, not the product.
- Price: list prices rise between seasons (median 15.99 € in 232/241, 19.99 € for the new models of 242), which trees
  cannot extrapolate. The landed cost (`PRICE_LCP`, known when the price is set, 97-99% filled) carries that drift, and
  the markup over it is stable within a category, so the price model estimates the markup: price = cost x markup.
  Products without a cost use the model without cost. Validation 232 -> 241, test 232 + 241 -> 242 (prices are not
  cut by the extraction).
- Delivered estimates are never in-sample: cross-validated (grouped by model) where the target is known.
- Buyer decisions (PUR_BET purchase depth, TYPOLOGY store tiers) are left out: they encode the buyer's own
  expectation of the sales we want to estimate. A reference model with them is reported, not used.
- The price model never sees the price (nor the numeric similarity block, which contains it).

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
N_PERMUTATIONS = 20             # shuffled-target runs of the leakage check
CATEGORICAL = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN", "CATEGORY_MATRIX", "CLR_DES", "CLR_TYPE", "FINISHING",
               "MATERIAL", "PRINT_TYPE", "DIMENSION", "OUTFIT", "FASHIONTYPE", "PROD_SEG", "NUMBER_OF_UNITS",
               "COLLECTION_TYPE", "MADE_IN", "SUP_COD"]
BUYER_DECISIONS = ["PUR_BET", "TYPOLOGY"]
COMPLETE_SEASONS = ("232", "241")   # finished selling at the extraction; 242 was still selling
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
    # the planned store date is what a buyer knows when ordering (the final date is revised for 56% of the products)
    planned = pd.to_datetime(items["STORE_DATE_PLANNED"])
    items["launch_month"] = planned.dt.month.astype(float)
    items["log_sales"] = np.log1p(items["SALES_QTY"].where(items["SALES_QTY"] > 0))
    items["log_price"] = np.log(items["PRICE_BASE_W_VAT"].astype(float))
    cost = pd.to_numeric(items["PRICE_LCP"], errors="coerce")
    items["log_cost"] = np.log(cost.where(cost > 0))
    return items, blocks


def whole_life(items: pd.DataFrame) -> np.ndarray:
    """Products whose sales are a whole life: the seasons that had finished selling at the extraction."""
    return items["SEA_COD"].astype(str).isin(COMPLETE_SEASONS).to_numpy()


def data_checks(items: pd.DataFrame) -> dict:
    """The facts the data decisions rest on (written to the report)."""
    price = items["PRICE_BASE_W_VAT"].astype(float)
    qty = items["SALES_QTY"]
    month = pd.to_datetime(items["STORE_DATE_FINAL"]).dt.strftime("%Y-%m")
    cut = (items.assign(month=month).groupby(["SEA_COD", "month"])["SALES_QTY"].agg(["size", "median"])
           .query("size >= 5").round(0))
    markup = np.exp(items["log_price"] - items["log_cost"])
    new = ~items["PROD_REF"].astype(str).isin(set(items.loc[items["SEA_COD"].astype(str) != TEST_SEASON, "PROD_REF"]
                                                  .astype(str)))
    return {"price_missing": int(price.isna().sum()), "price_zero_or_less": int((price <= 0).sum()),
            "price_min": float(price.min()), "price_max": float(price.max()), "price_points": int(price.nunique()),
            "cost_missing": int(items["log_cost"].isna().sum()),
            "sales_missing": int(qty.isna().sum()), "sales_zero": int((qty == 0).sum()),
            "median_units_by_season": items.groupby("SEA_COD")["SALES_QTY"].median().round(0).to_dict(),
            "units_by_launch_month": cut,
            "median_price_by_season": items.groupby("SEA_COD")["PRICE_BASE_W_VAT"].median().to_dict(),
            "median_price_new_test": float(price[new & items["SEA_COD"].astype(str).eq(TEST_SEASON)].median()),
            "markup_by_category": markup.groupby([items["CAT_DES_EN"], items["SEA_COD"]]).median().unstack().round(2)}


def split(items: pd.DataFrame, train_seasons=("232", "241"), test_season: str = TEST_SEASON,
          half: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Train: the given seasons. Test: one later season, only models never sold in the training seasons (new products).
    `half` (0 or 1) keeps one random half of the test models (all colourways of a model in the same half): the
    sales choices are made on one half of 241 and the other half is reported once."""
    sea = items["SEA_COD"].astype(str)
    train = sea.isin(train_seasons).to_numpy()
    seen = set(items.loc[train, "PROD_REF"].astype(str))
    new_model = ~items["PROD_REF"].astype(str).isin(seen).to_numpy()
    test = sea.eq(test_season).to_numpy() & new_model
    if half is not None:
        groups = model_groups(items)
        uniq = np.unique(groups)
        side = dict(zip(uniq, np.random.default_rng(SEED).permutation(len(uniq)) % 2))
        test &= np.array([side[g] == half for g in groups])
    return train, test


def model_groups(items: pd.DataFrame) -> np.ndarray:
    """One id per model, joining PROD_REF and PROD_REF_EQUIV (re-coded items are the same model, as in the engine)."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in zip(items["PROD_REF"].astype(str), items["PROD_REF_EQUIV"].fillna(items["PROD_REF"]).astype(str)):
        parent[find(a)] = find(b)
    return np.array([find(a) for a in items["PROD_REF"].astype(str)])


VALIDATION = (("232",), "241")              # price: choices
TEST = (("232", "241"), TEST_SEASON)        # price: reported once; sales: same-month ranking check only (cut sales)
SALES_VALIDATION = (("232",), "241", 0)     # sales (whole lives only): choices on one half of the new 241 models
SALES_TEST = (("232",), "241", 1)           # sales: the other half, reported once


def class_shift(items: pd.DataFrame, cols=("FASHIONTYPE", "COLLECTION_TYPE"), top: int = 4) -> pd.DataFrame:
    """Share of the most variable values of the planning classes by season (they drive the sales estimate)."""
    rows = []
    for c in cols:
        t = pd.crosstab(items[c], items["SEA_COD"].astype(str), normalize="columns")
        spread = (t.max(axis=1) - t.min(axis=1)).sort_values(ascending=False)
        for v in spread.index[:top]:
            rows.append({"attribute": c, "value": v, **t.loc[v].round(3).to_dict()})
    return pd.DataFrame(rows)


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
            w = np.where(np.isfinite(sc), np.clip(sc, 0, None), 0.0)   # excluded candidates (own model) never count
            yy = y[cand][top]
            est[q] = np.where(w.sum(1) > 0, (w * yy).sum(1) / np.maximum(w.sum(1), 1e-12), np.nan)
            for qi, row, srow in zip(q, top, sc):
                keys[qi] = items[KEY].to_numpy()[cand[row[np.isfinite(srow)]]].tolist()
    return est, keys


def feature_table(items: pd.DataFrame, blocks: dict, train: np.ndarray, target: str,
                  analogue: np.ndarray | None, extra: list[str] = (), full: bool = False) -> pd.DataFrame:
    """One row per product: attributes (categorical), numbers, embedding components and the analogue estimate.
    Only the columns of the target's chosen feature set (`FEATURES`), unless `full` (the selection itself)."""
    X = pd.DataFrame(index=items.index)
    for c in CATEGORICAL + list(extra):
        if c in items:
            X[c] = pd.Categorical(items[c].astype("string"))
    X["n_sizes"] = items["n_skus"].astype(float)
    X["net_weight"] = pd.to_numeric(items["NET_WEIGHT"], errors="coerce")
    X["launch_month"] = items["launch_month"]
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
    return X if full else feature_set(X, FEATURES[target])


FAMILY_OF_FEATURE = [("img_pc", "photo (image components)"), ("txt_pc", "description (text components)"),
                     ("mat_", "materials"), ("analogue", "similar past products"), ("price", "price"),
                     ("log_cost", "landed cost"), ("launch_month", "launch month"),
                     ("n_sizes", "number of sizes"), ("net_weight", "weight")]


def feature_family(col: str) -> str:
    for prefix, fam in FAMILY_OF_FEATURE:
        if col.startswith(prefix):
            return fam
    return "attribute: " + col


# --------------------------------------------------------------------------- models and metrics
def fit_predict(X: pd.DataFrame, y: np.ndarray, train: np.ndarray, fit_rows: np.ndarray,
                weight: np.ndarray | None = None):
    import lightgbm as lgb
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X[fit_rows], y[fit_rows], sample_weight=None if weight is None else weight[fit_rows])
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

    def weighted(keys_):  # mean of the per-group rank correlations, weighted by group size (both metrics alike)
        res = [(spearman(y[keys_ == g], p[keys_ == g]), (keys_ == g).sum()) for g in np.unique(keys_)
               if (keys_ == g).sum() > 10]
        return float(np.average([r for r, _ in res], weights=[n for _, n in res])) if res else 0.0
    top_true = pd.Series(y).groupby(cats).rank(pct=True).to_numpy() > 0.8
    # ties broken at random, so a constant estimate picks a random 20% (its fair score: the base rate)
    jitter = np.random.default_rng(SEED).random(len(p)) * 1e-9
    top_pred = pd.Series(p + jitter).groupby(cats).rank(pct=True).to_numpy() > 0.8
    return {"spearman_within_cat": weighted(cats), "spearman_same_month": weighted(groups),
            "mae_log": float(np.mean(np.abs(y - p))), "bias_log": float(np.mean(p - y)),
            "within_2x": float(np.mean(np.abs(p - y) < np.log(2))), "top20_precision": float(top_true[top_pred].mean())}


def price_metrics(y: np.ndarray, p: np.ndarray, points: np.ndarray) -> dict:
    """Error in euros and in price points (the 40 list prices), on the original scale."""
    ye, pe = np.exp(y), np.exp(p)
    # both sides as an index into the list prices (exp(log(p)) is not bit-exact, so the truth is snapped too)
    def nearest(v):
        return np.abs(v[:, None] - points[None, :]).argmin(axis=1)
    steps = np.abs(nearest(pe) - nearest(ye))
    return {"mae_eur": float(np.mean(np.abs(ye - pe))), "mape": float(np.mean(np.abs(ye - pe) / ye)),
            "bias": float(np.exp(np.mean(p - y)) - 1),
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
    Xt = pd.DataFrame({"CAT_DES_EN": pd.Categorical(items["CAT_DES_EN"]), "launch_month": items["launch_month"]})
    _, preds["launch month + category only"] = fit_predict(Xt, y, train, fit_rows)
    ana, ana_keys = analogues(items, blocks, weights, "log_sales", fit_rows)
    preds["similar past products (engine)"] = np.where(np.isfinite(ana), ana, preds["category median"])
    X0 = feature_table(items, blocks, train, "log_sales", None)
    model, preds["gradient boosting, no analogues"] = fit_predict(X0, y, train, fit_rows)   # the chosen model
    X = feature_table(items, blocks, train, "log_sales", ana)
    model_ana, preds["gradient boosting + analogues"] = fit_predict(X, y, train, fit_rows)
    Xb = feature_table(items, blocks, train, "log_sales", None, extra=BUYER_DECISIONS)   # chosen model + them
    _, ref = fit_predict(Xb, y, train, fit_rows)
    table = {n: sales_metrics(y[eval_rows], p[eval_rows], cats[eval_rows]) for n, p in preds.items()}
    table["reference: + buyer decisions (not used)"] = sales_metrics(y[eval_rows], ref[eval_rows], cats[eval_rows])
    groups = model_groups(items)[eval_rows]
    m = lambda yy, pp, cc: sales_metrics(yy, pp, cc)["spearman_within_cat"]  # noqa: E731
    ci = {n: bootstrap_diff(m, y[eval_rows], preds["category median"][eval_rows], p[eval_rows], groups, cats[eval_rows])
          for n, p in preds.items() if n != "category median"}
    m_month = lambda yy, pp, cc: sales_metrics(yy, pp, cc)["spearman_same_month"]  # noqa: E731
    ci_ana = bootstrap_diff(m_month, y[eval_rows], preds["gradient boosting, no analogues"][eval_rows],
                            preds["gradient boosting + analogues"][eval_rows], groups, cats[eval_rows])
    m2 = lambda yy, pp, cc: sales_metrics(yy, pp, cc)["spearman_same_month"]  # noqa: E731
    ci_month = {n: bootstrap_diff(m2, y[eval_rows], preds["launch month + category only"][eval_rows], p[eval_rows],
                                  groups, cats[eval_rows])
                for n, p in preds.items() if n not in ("category median", "launch month + category only")}
    m_err = lambda yy, pp, cc: -sales_metrics(yy, pp, cc)["mae_log"]  # noqa: E731  (higher is better)
    sim = preds["similar past products (engine)"][eval_rows]
    ci_vs_ana = {v: {"same-month ranking": bootstrap_diff(m_month, y[eval_rows], sim, preds[v][eval_rows], groups,
                                                          cats[eval_rows]),
                     "units error saved (log)": bootstrap_diff(m_err, y[eval_rows], sim, preds[v][eval_rows], groups,
                                                               cats[eval_rows])}
                 for v in ("gradient boosting, no analogues", "gradient boosting + analogues")}
    return {"y": y, "preds": preds, "analogue": ana, "analogue_keys": ana_keys,
            "variants": {False: (model, X0, "gradient boosting, no analogues"),
                         True: (model_ana, X, "gradient boosting + analogues")},
            "table": table, "ci_vs_category": ci, "ci_analogue_gain": ci_ana, "ci_same_month": ci_month,
            "ci_vs_analogue": ci_vs_ana,
            "n_fit": int(fit_rows.sum()), "n_eval": int(eval_rows.sum()), "eval_rows": eval_rows}


def run_price(items, blocks, weights, train, test) -> dict:
    y = items["log_price"].to_numpy(float)
    fit_rows, eval_rows = train, test
    points = np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique())
    no_price = {**weights, "numeric": 0.0}          # the numeric block contains the price: never used here
    sub = ["CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN"]
    cost = items["log_cost"].to_numpy(float)
    has_cost = np.isfinite(cost)
    markup = y - cost                                  # log(price / cost): stable within a category, no season drift
    preds = {"category median": group_baseline(items, y, fit_rows, ["CAT_DES_EN"]),
             "sub-family median": group_baseline(items, y, fit_rows, sub)}
    preds["cost × sub-family median markup"] = np.where(
        has_cost, cost + group_baseline(items, markup, fit_rows & has_cost, sub), preds["sub-family median"])
    ana, ana_keys = analogues(items, blocks, no_price, "log_price", fit_rows)
    preds["similar past products (engine)"] = np.where(np.isfinite(ana), ana, preds["sub-family median"])
    X0 = feature_table(items, blocks, train, "log_price", None)
    _, no_cost = fit_predict(X0, y, train, fit_rows)
    preds["gradient boosting without cost (previous model)"] = no_cost
    # the chosen model: the markup over the landed cost; products without a cost fall back to the model without it
    X0c = X0.assign(log_cost=cost)
    model, pm = fit_predict(X0c, markup, train, fit_rows & has_cost)
    preds["gradient boosting, no analogues"] = np.where(has_cost, cost + pm, no_cost)
    X = feature_table(items, blocks, train, "log_price", ana).assign(log_cost=cost)
    model_ana, pm_ana = fit_predict(X, markup, train, fit_rows & has_cost)
    preds["gradient boosting + analogues"] = np.where(has_cost, cost + pm_ana, no_cost)
    table = {n: price_metrics(y[eval_rows], p[eval_rows], points) for n, p in preds.items()}
    groups = model_groups(items)[eval_rows]
    m = lambda yy, pp, _: -price_metrics(yy, pp, points)["mae_eur"]  # noqa: E731  (higher is better)
    ci = {n: bootstrap_diff(m, y[eval_rows], preds["sub-family median"][eval_rows], p[eval_rows], groups, groups)
          for n, p in preds.items() if n not in ("category median", "sub-family median")}
    ci_ana = bootstrap_diff(m, y[eval_rows], preds["gradient boosting, no analogues"][eval_rows],
                            preds["gradient boosting + analogues"][eval_rows], groups, groups)
    ci_vs_ana = {v: {"€ of error saved": bootstrap_diff(m, y[eval_rows], preds["similar past products (engine)"][eval_rows],
                                                        preds[v][eval_rows], groups, groups)}
                 for v in ("gradient boosting, no analogues", "gradient boosting + analogues")}
    return {"y": y, "preds": preds, "analogue": ana, "analogue_keys": ana_keys,
            "variants": {False: (model, X0c, "gradient boosting, no analogues"),
                         True: (model_ana, X, "gradient boosting + analogues")},
            "table": table, "ci_vs_subfamily": ci, "ci_analogue_gain": ci_ana, "ci_vs_analogue": ci_vs_ana, "n_fit": int(fit_rows.sum()),
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


def out_of_fold(items, blocks, target: str, usable: np.ndarray, use_analogue: bool = False,
                analogue_est: np.ndarray | None = None, offset: np.ndarray | None = None,
                folds: int = 5) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimates that never saw the product's own target: for products with a known, usable target, 5-fold
    cross-fitting grouped by model (colourways of a model stay in the same fold); for the others (new, never sold, or
    a target that cannot be used), the model fitted on every known product. With `offset` (the log cost) the model
    learns target - offset (the markup) and adds it back. Returns (estimate, kind, SHAP families per product)."""
    y = items[target].to_numpy(float)
    if offset is not None:
        y = y - offset
    known = np.isfinite(y) & usable
    allrows = np.ones(len(items), bool)
    X = feature_table(items, blocks, allrows, target, analogue_est if use_analogue else None)
    if offset is not None:
        X = X.assign(log_cost=offset)
    est = np.full(len(items), np.nan)
    fam = pd.DataFrame(index=items.index, dtype=float)
    refs = model_groups(items)
    uniq = np.unique(refs[known])
    fold_of = dict(zip(uniq, np.random.default_rng(SEED).integers(0, folds, len(uniq))))
    fold = np.array([fold_of.get(r, -1) for r in refs])
    parts = [(known & (fold != f), known & (fold == f)) for f in range(folds)] + [(known, ~known)]
    for fit_rows, pred_rows in parts:
        if not pred_rows.any():
            continue
        model, pred = fit_predict(X, y, allrows, fit_rows)
        est[pred_rows] = pred[pred_rows]
        sv = shap_values(model, X[pred_rows]).drop(columns="expected_value")
        f_ = sv.T.groupby(lambda c: feature_family(c)).sum().T
        fam = f_ if fam.empty else pd.concat([fam, f_])
    if offset is not None:
        est = est + offset
    kind = np.where(known, "cross-validated (target not seen)", "forecast (no sales / new)")
    return est, kind, fam.reindex(items.index)


def final_models(items, blocks, weights, typical_error: dict, use_analogue: dict) -> pd.DataFrame:
    """Per product: the estimate (never in-sample), its kind, the honest typical error measured on the test split,
    the three biggest SHAP effects, and the analogue estimate with its products (the buyer's reference).

    Sales: whole-life units, learned from the finished seasons only; 242 products get a forecast even when they have
    (cut) sales. Price: landed cost x estimated markup, snapped to the nearest list price; without a cost, the model
    without it."""
    out = pd.DataFrame({KEY: items[KEY]})
    life = whole_life(items)
    y = items["log_sales"].to_numpy(float)
    ana, keys = analogues(items, blocks, weights, "log_sales", np.isfinite(y) & life)
    est, kind, fam = out_of_fold(items, blocks, "log_sales", life, use_analogue["sales"], ana)
    kind = np.where(np.isfinite(y) & ~life, f"forecast (season {TEST_SEASON} was still selling)", kind)
    parts = {"sales": (np.expm1(est).round(0), kind, fam, np.expm1(ana).round(0), keys)}
    est_s = est

    no_price = {**weights, "numeric": 0.0}
    ana, keys = analogues(items, blocks, no_price, "log_price", np.isfinite(items["log_price"].to_numpy(float)))
    allrows = np.ones(len(items), bool)
    cost = items["log_cost"].to_numpy(float)
    est_c, kind_c, fam_c = out_of_fold(items, blocks, "log_price", allrows, use_analogue["price"], ana, offset=cost)
    est_n, kind_n, fam_n = out_of_fold(items, blocks, "log_price", allrows, use_analogue["price"], ana)
    has = np.isfinite(est_c)
    est = np.where(has, est_c, est_n)
    kind = np.where(has, kind_c, np.char.add(kind_n.astype(str), " (no landed cost: model without it)"))
    fam = fam_c.where(pd.Series(has, index=items.index), fam_n)
    parts["price"] = (np.exp(est).round(2), kind, fam, np.exp(ana).round(2), keys)
    points = np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique())

    for name, (value, kind, fam, ana_v, keys) in parts.items():
        out[f"{name}_estimate"] = value
        if name == "price":
            out["price_point_estimate"] = points[np.abs(value[:, None] - points[None, :]).argmin(axis=1)]
        lo, hi = typical_error[f"{name}_range_log"]
        if name == "sales":
            out["sales_low"], out["sales_high"] = np.expm1(np.maximum(est_s + lo, 0)).round(0), np.expm1(est_s + hi).round(0)
        else:
            out["price_low"], out["price_high"] = np.exp(est + lo).round(2), np.exp(est + hi).round(2)
        out[f"{name}_estimate_kind"] = kind
        out[f"{name}_typical_error"] = typical_error[name]
        out[f"{name}_analogue"] = ana_v
        out[f"{name}_analogue_keys"] = ["|".join(k[:4]) for k in keys]
        # the three biggest effects, biggest first (a Series returned by apply would be re-sorted by name)
        out[f"{name}_reasons"] = ["; ".join(f"{f} {'+' if v > 0 else '−'}{abs(v):.2f}"
                                            for f, v in r.dropna().reindex(r.dropna().abs()
                                                                           .sort_values(ascending=False).index)[:3].items())
                                  for _, r in fam.iterrows()]
    out["landed_cost"] = np.exp(cost).round(2)
    return out


def estimates_csv(items: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """One readable row per product: what it is, the estimates with an 80% range, the similar past products behind
    the analogue reference, and the three biggest reasons. Same values as `forecast.parquet`."""
    info = items[[KEY, "PROD_DES_BASE", "SEA_COD", "CAT_DES_EN", "GFA_DES_EN", "GFS_DES_EN", "CLR_DES",
                  "PRICE_BASE_W_VAT", "SALES_QTY"]].rename(columns={
        "PROD_DES_BASE": "description", "SEA_COD": "season", "CAT_DES_EN": "category", "GFA_DES_EN": "family",
        "GFS_DES_EN": "sub_family", "CLR_DES": "colour", "PRICE_BASE_W_VAT": "list_price", "SALES_QTY": "units_sold"})
    cols = [KEY, "sales_estimate", "sales_low", "sales_high", "sales_estimate_kind", "sales_analogue",
            "sales_analogue_keys", "sales_reasons", "price_estimate", "price_point_estimate", "price_low", "price_high",
            "price_estimate_kind", "landed_cost", "price_analogue", "price_analogue_keys", "price_reasons"]
    return info.merge(final[cols], on=KEY, how="left", validate="one_to_one")


FEATURE_SETS = {  # simplest first: the column prefixes each set leaves out
    "attributes only": ("img_pc", "txt_pc", "mat_", "launch_month"),
    "attributes + photo": ("txt_pc", "mat_", "launch_month"),
    "without photo and description": ("img_pc", "txt_pc"),
    "without photo": ("img_pc",),
    "without launch month": ("launch_month",),
    "all features": (),
}
# Declared rule: a block leaves only if it helps in neither check (validation/test on whole lives, and the
# same-month ranking on 242). Description, materials and launch month help in neither; the photo is neutral on 241
# but adds +0.005 to +0.052 same-month ranking on 242 (trained on 232 + 241), so it stays for sales. Price: the
# photo helps in neither (outputs/phase5b/feature_selection.md).
FEATURES = {"log_sales": "attributes + photo", "log_price": "attributes only"}


def feature_set(X: pd.DataFrame, name: str) -> pd.DataFrame:
    drop = FEATURE_SETS[name]
    return X[[c for c in X if not (drop and c.startswith(drop))]]


def select_features(items, blocks, weights, n_boot: int = 200) -> dict:
    """Fast feature selection, separately for each target, decided on validation only: the simplest set whose
    validation error is not clearly worse than the best set's (95% bootstrap interval of the difference reaches 0).
    Error = mean absolute log error of the units (sales) / mean absolute error in € (price). Test numbers are shown
    next to it, never used to choose."""
    out = {}
    # sales: one fit on 232 per set, scored on both halves of the new 241 models
    y = items["log_sales"].to_numpy(float)
    tr, va = split(items, *SALES_VALIDATION)
    _, te = split(items, *SALES_TEST)
    fit = tr & np.isfinite(y)
    va, te = va & np.isfinite(y), te & np.isfinite(y)
    groups = (items["CAT_DES_EN"].astype(str) + "|" + items["launch_month"].astype(int).astype(str) + "|"
              + items["SEA_COD"].astype(str)).to_numpy()
    X = feature_table(items, blocks, tr, "log_sales", None, full=True)
    preds = {n: fit_predict(feature_set(X, n), y, tr, fit)[1] for n in FEATURE_SETS}
    err = lambda yy, pp, _: -np.mean(np.abs(yy - pp))  # noqa: E731  (higher is better)
    rows = {n: {**{f"val_{k}": v for k, v in sales_metrics(y[va], p[va], groups[va]).items()
                   if k in ("mae_log", "within_2x", "spearman_same_month")},
                **{f"test_{k}": v for k, v in sales_metrics(y[te], p[te], groups[te]).items()
                   if k in ("mae_log", "within_2x", "spearman_same_month")}} for n, p in preds.items()}
    out["sales"] = _choose(rows, preds, y, va, model_groups(items), "val_mae_log", err, n_boot)

    # price: cost x markup with the similar products (the chosen variant); validation 232 -> 241, test -> 242
    y = items["log_price"].to_numpy(float)
    cost = items["log_cost"].to_numpy(float)
    has = np.isfinite(cost)
    no_price = {**weights, "numeric": 0.0}
    points = np.sort(items["PRICE_BASE_W_VAT"].astype(float).unique())
    preds, test_preds = {}, {}
    for seasons, store in [(VALIDATION, preds), (TEST, test_preds)]:
        tr, ev = split(items, *seasons)
        ana, _ = analogues(items, blocks, no_price, "log_price", tr)
        X = feature_table(items, blocks, tr, "log_price", ana, full=True)
        for n in FEATURE_SETS:
            Xs = feature_set(X, n)
            _, no_cost = fit_predict(Xs, y, tr, tr)
            _, pm = fit_predict(Xs.assign(log_cost=cost), y - cost, tr, tr & has)
            store[n] = np.where(has, cost + pm, no_cost)
    _, va = split(items, *VALIDATION)
    _, te = split(items, *TEST)
    err_p = lambda yy, pp, _: -np.mean(np.abs(np.exp(yy) - np.exp(pp)))  # noqa: E731
    rows = {n: {**{f"val_{k}": v for k, v in price_metrics(y[va], preds[n][va], points).items()
                   if k in ("mae_eur", "bias", "within_1_point")},
                **{f"test_{k}": v for k, v in price_metrics(y[te], test_preds[n][te], points).items()
                   if k in ("mae_eur", "bias", "within_1_point")}} for n in FEATURE_SETS}
    out["price"] = _choose(rows, preds, y, va, model_groups(items), "val_mae_eur", err_p, n_boot)
    out["price_drift"] = price_drift(items, blocks, weights, n_boot)
    return out


def price_drift(items, blocks, weights, n_boot: int = 200) -> pd.DataFrame:
    """Corrections for the rise of markups between seasons. Only the split 232 + 241 -> 242 has two training seasons,
    so the new 242 models are split in halves: a correction is adopted only if its gain on half A excludes 0; half
    B is shown, never used. Candidates: a season index (a new season priced like the latest one) and weights on 241."""
    y = items["log_price"].to_numpy(float)
    cost = items["log_cost"].to_numpy(float)
    has = np.isfinite(cost)
    sea = items["SEA_COD"].astype(str).to_numpy()
    tr, _ = split(items, *TEST)
    _, A = split(items, *TEST, half=0)
    _, B = split(items, *TEST, half=1)
    ana, _ = analogues(items, blocks, {**weights, "numeric": 0.0}, "log_price", tr)
    X = feature_table(items, blocks, tr, "log_price", ana).assign(log_cost=cost)
    latest = np.where(sea == "241", 1.0, 0.0)
    latest[~tr] = 1.0
    cands = {"none (chosen model)": (X, None), "season index (new season = latest)": (X.assign(season_idx=latest), None),
             **{f"season 241 weighted ×{k}": (X, np.where(sea == "241", float(k), 1.0)) for k in (2, 3)}}
    preds = {n: cost + fit_predict(Xc, y - cost, tr, tr & has, wt)[1] for n, (Xc, wt) in cands.items()}
    groups = model_groups(items)
    err = lambda yy, pp, _: -np.mean(np.abs(np.exp(yy) - np.exp(pp)))  # noqa: E731
    base = preds["none (chosen model)"]
    rows = {}
    for n, p in preds.items():
        a, b = A & has, B & has
        lo, hi = bootstrap_diff(err, y[a], base[a], p[a], groups[a], groups[a], n=n_boot)
        rows[n] = {"A_mae_eur": float(np.mean(np.abs(np.exp(p[a]) - np.exp(y[a])))),
                   "A_bias": float(np.exp(np.mean(p[a] - y[a])) - 1), "A_gain_lo": lo, "A_gain_hi": hi,
                   "B_mae_eur": float(np.mean(np.abs(np.exp(p[b]) - np.exp(y[b])))),
                   "B_bias": float(np.exp(np.mean(p[b] - y[b])) - 1),
                   "adopted": bool(n != "none (chosen model)" and lo > 0)}
    return pd.DataFrame(rows).T


def _choose(rows, preds, y, ev, groups, key, err, n_boot) -> dict:
    best = min(rows, key=lambda n: rows[n][key])
    for n in FEATURE_SETS:
        lo, hi = bootstrap_diff(err, y[ev], preds[best][ev], preds[n][ev], groups[ev], groups[ev], n=n_boot)
        rows[n]["val_gain_vs_best_lo"], rows[n]["val_gain_vs_best_hi"] = lo, hi
    chosen = next(n for n in FEATURE_SETS if rows[n]["val_gain_vs_best_hi"] >= 0)
    return {"table": pd.DataFrame(rows).T, "best": best, "chosen": chosen}


def leakage_checks(items, blocks, train, test) -> pd.DataFrame:
    """Diagnostics of the sales model on the test split (they choose nothing): shuffled target and feature ablations."""
    y = items["log_sales"].to_numpy(float)
    fit_rows, ev_rows = train & np.isfinite(y), test & np.isfinite(y)
    groups = (items["CAT_DES_EN"].astype(str) + "|" + items["launch_month"].astype(int).astype(str) + "|"
              + items["SEA_COD"].astype(str)).to_numpy()
    full = feature_table(items, blocks, train, "log_sales", None, full=True)
    X = feature_set(full, FEATURES["log_sales"])

    def score(Xv, yv=y):
        _, p = fit_predict(Xv, yv, train, fit_rows)
        return sales_metrics(y[ev_rows], p[ev_rows], groups[ev_rows])
    idx = np.flatnonzero(fit_rows)
    perms = []
    for k in range(N_PERMUTATIONS):
        shuffled = y.copy()
        shuffled[idx] = shuffled[np.random.default_rng(SEED + k).permutation(idx)]
        perms.append(score(X, shuffled))
    perms = pd.DataFrame(perms)
    rows = {"chosen model": score(X),
            f"target shuffled in training, mean of {N_PERMUTATIONS} runs (must be ~0)": perms.mean().to_dict(),
            f"target shuffled in training, SD of {N_PERMUTATIONS} runs": perms.std().to_dict(),
            "without fashion type and collection type": score(X.drop(columns=["FASHIONTYPE", "COLLECTION_TYPE"])),
            "without the three planning classes (+ CATEGORY_MATRIX)":
                score(X.drop(columns=["FASHIONTYPE", "COLLECTION_TYPE", "CATEGORY_MATRIX"])),
            "without the photo (attributes only)": score(feature_set(full, "attributes only")),
            "with description, materials and launch month (all features)": score(full)}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([""] + [str(c) for c in cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        cells = [f"{v:.1%}" if c in pct and pd.notna(v) else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
                 for c, v in r.items()]
        label = " · ".join(map(str, idx)) if isinstance(idx, tuple) else str(idx)
        lines.append("| " + " | ".join([label] + cells) + " |")
    return "\n".join(lines)


def bias_tables(items, test) -> dict:
    """Mean error by category of the chosen models on the test splits (as a ratio, > 0 = too high), and the
    level of the seasons the sales model learns from and is tested on."""
    out = {}
    for t in ("sales", "price"):
        r = test[t]
        ev = r["eval_rows"] & np.isfinite(r["y"])
        res = pd.Series((r["preds"][r["chosen"]] - r["y"])[ev], index=items.index[ev])
        g = res.groupby(items.loc[ev, "CAT_DES_EN"])
        out[t] = pd.DataFrame({f"{t} products": g.size(), f"{t} bias": np.exp(g.mean()) - 1})
    y = items["log_sales"]
    sea = items["SEA_COD"].astype(str)
    out["level"] = pd.DataFrame({
        "232 (training)": np.expm1(y[sea.eq("232")].groupby(items["CAT_DES_EN"]).median()).round(0),
        "241 test models": np.expm1(y[test["sales"]["eval_rows"]].groupby(items["CAT_DES_EN"]).median()).round(0)})
    return out


def write_report(checks: dict, val: dict, test: dict, rank242: dict, decisions: dict, leak: pd.DataFrame,
                 shift: pd.DataFrame, imp_s: pd.Series, imp_p: pd.Series, typical: dict, biases: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    s_v, p_v = pd.DataFrame(val["sales"]["table"]).T, pd.DataFrame(val["price"]["table"]).T
    s_t, p_t = pd.DataFrame(test["sales"]["table"]).T, pd.DataFrame(test["price"]["table"]).T
    s_r = pd.DataFrame(rank242["table"]).T[["spearman_within_cat", "spearman_same_month", "top20_precision"]]
    sp = {"top20_precision", "within_2x"}
    pp = {"mape", "exact_price_point", "within_1_point", "bias"}
    ci_s = "\n".join(f"- {n}: {lo:+.3f} to {hi:+.3f}" for n, (lo, hi) in test["sales"]["ci_same_month"].items())
    ci_p = "\n".join(f"- {n}: {lo:+.2f} to {hi:+.2f} €" for n, (lo, hi) in test["price"]["ci_vs_subfamily"].items())
    ci_a = "\n".join(f"- {t} ({test[t]['chosen']}): " + "; ".join(
        f"{m} {lo:+.3f} to {hi:+.3f}" for m, (lo, hi) in test[t]["ci_vs_analogue"][test[t]["chosen"]].items())
        for t in ("sales", "price"))
    cut = checks["units_by_launch_month"].reset_index()
    cut = cut[cut["SEA_COD"].astype(str).isin(["241", TEST_SEASON])].set_index(["SEA_COD", "month"])
    lines = [
        "# Phase 5b: explainable sales and price estimates",
        "",
        f"Generated by `src/phase5b_forecast.py run` ({datetime.now(timezone.utc):%Y-%m-%d}). Products are colourways; "
        "tests use only models never sold in the training seasons (new products); intervals are 95% bootstrap over "
        "models.",
        "",
        "## Protocol",
        "",
        "- **Sales = whole-life units.** Sales are accumulated until the extraction (about January 2025): seasons "
        f"{' and '.join(COMPLETE_SEASONS)} had finished selling, {TEST_SEASON} had not (table below). So the units are "
        "learned and scored only on whole lives: **train 232 → the new models of 241**, split at random by model into "
        "two halves: one for the choices (validation), one reported once (test).",
        f"- **{TEST_SEASON} for sales: ranking only.** Its sales are cut by the extraction, so they never score the "
        "units; they only check the ranking among products of a category launched in the same month (equally cut), "
        f"with a model trained on 232 + 241.",
        "- **No time on sale.** The days to the extraction describe the extraction, not the product (and fall outside "
        "the training range for a new season). The planned launch month (known when ordering) stays.",
        "- **Price = landed cost × markup.** The price model estimates the markup over the landed cost (`PRICE_LCP`, "
        "known when the price is set); the cost carries the rise of prices between seasons, which trees cannot "
        "extrapolate. Products without a cost use the model without it. **Validation 232 → 241, test 232 + 241 → "
        f"{TEST_SEASON}** (list prices are not cut by the extraction).",
        "- **Left out:** buyer decisions (`PUR_BET`, `TYPOLOGY`; see `buyer_report.md`); sales are never a feature; "
        "the price model never sees the price or the numeric similarity block.",
        "- **Features**, by a rule declared before the choice: a block leaves only if it helps in neither check "
        f"(whole lives 232 → 241, and the same-month ranking on {TEST_SEASON}). Description components, materials and "
        "launch month help in neither and leave both estimates. The photo components stay for **sales**: neutral on "
        f"241 (−0.04 to +0.02 same-month ranking) but +0.005 to +0.052 on {TEST_SEASON}, where the model trains on "
        "twice the products; this uses the 242 check, so it is stated. **Price**: attributes only (the photo helps in "
        "neither; `feature_selection.md`).",
        "- **A choice** (the similar products as a feature) is kept only if the 95% interval of its validation gain "
        "excludes 0.",
        "",
        "## The data the decisions rest on",
        "",
        f"- **Sales cut by the extraction:** median units by launch month (seasons 241 and {TEST_SEASON}):",
        "",
        _md(cut, digits=0),
        "",
        f"- **Price:** {checks['price_missing']} missing, {checks['price_zero_or_less']} at 0 € or less; "
        f"{checks['price_points']} list prices from {checks['price_min']:.2f} € to {checks['price_max']:.2f} €. Median "
        f"by season {checks['median_price_by_season']}; new models of {TEST_SEASON}: "
        f"{checks['median_price_new_test']:.2f} €.",
        f"- **Landed cost** missing or ≤ 0 for {checks['cost_missing']} colourways. Median markup (price / cost) by "
        "category and season:",
        "",
        _md(checks["markup_by_category"], digits=2),
        "",
        f"- **Sales:** {checks['sales_missing']} colourways without a sales row, {checks['sales_zero']} with 0 units: "
        "unknown, left out of training and evaluation (they still get an estimate).",
        "- **Planning classes change between seasons** (share of the colourways):",
        "",
        _md(shift.set_index(["attribute", "value"]), digits=3),
        "",
        "## Sales, validation (232 → half of the new 241 models): the choices",
        "",
        _md(s_v, pct=sp),
        "",
        "## Price, validation (232 → 241): the choices",
        "",
        _md(p_v, pct=pp),
        "",
        *[f"- {d}" for d in decisions.values()],
        "",
        f"## Sales, test (232 → the other half of the new 241 models): {test['sales']['n_fit']:,} training / "
        f"{test['sales']['n_eval']:,} test products",
        "",
        "`mae_log`: mean absolute error in log units (exp of it = the typical factor); `bias_log`: mean of estimate − "
        "real (> 0 = too high); `within_2x`: share of products whose estimate is within a factor 2 of the real units. "
        "`spearman_within_cat` / `spearman_same_month`: rank correlation within a category / among products of a "
        "category launched in the same month, weighted by group size. `top20_precision`: share of the estimated top "
        "20% of a category that really were (random 20%).",
        "",
        _md(s_t, pct=sp),
        "",
        "Same-month gain over the model that only knows the category and the launch month (95% CI):",
        "",
        ci_s,
        "",
        f"## Sales, ranking check on {TEST_SEASON} (232 + 241 → {TEST_SEASON}, {rank242['n_eval']:,} products, cut sales)",
        "",
        _md(s_r, pct=sp),
        "",
        f"## Price, test (232 + 241 → {TEST_SEASON}): {test['price']['n_fit']:,} training / "
        f"{test['price']['n_eval']:,} test products",
        "",
        "`bias`: estimate / real − 1 on average (geometric), > 0 = too high.",
        "",
        _md(p_t, pct=pp),
        "",
        "Error reduction over the sub-family median (95% CI):",
        "",
        ci_p,
        "",
        "## The chosen model against the similar past products (test splits, paired, 95% CI)",
        "",
        "The similar past products (the engine's analogues) are the method closest to *find similar items, then their "
        "sales and price*. Gain of the chosen gradient boosting over them on the same products (> 0 = the model is "
        "better):",
        "",
        ci_a,
        "",
        "## Biases: where they come from (test splits, by category)",
        "",
        "Mean error as a ratio (geometric; > 0 = estimate too high):",
        "",
        "Sales on the 241 test half, price on the 242 test.",
        "",
        _md(pd.concat([biases["sales"], biases["price"]], axis=1), pct={"sales bias", "price bias"}, digits=0),
        "",
        "Median units of the season the sales model learns from and of the season it is tested on:",
        "",
        _md(biases["level"], digits=0),
        "",
        "- **Sales:** the bias is mostly jewellery (under-estimated), and per category it does not follow the change "
        "of level between the seasons (apparel sold more and is over-estimated, jewellery sold a little more and is "
        "under-estimated). So it is not one season growth that a factor could remove, and with a single transition "
        "between whole seasons (232 → 241) no correction can be tested: **not corrected**. The estimates are at the "
        "level of the training seasons; a known growth plan (more stores, more stock) would scale them.",
        "- **Price:** markups rise between seasons (jewellery 232 → 241, wallets and hand bags 241 → 242). Three corrections were "
        "tried on half of the new 242 models (season index, more weight on 241): none gains clearly "
        "(`feature_selection.md`), so **none is applied**; the remaining bias is a pricing decision of 242 that the "
        "past seasons do not announce.",
        "",
        "## Leakage checks and what each part adds (sales, test split; diagnostics, no choice made here)",
        "",
        _md(leak, pct=sp),
        "",
        f"- A training target shuffled at random ({N_PERMUTATIONS} runs) must give ≈ 0 ranking: it does, so no "
        "information leaks through the pipeline (joins, row order, components fitted on training rows).",
        "",
        "## What drives the estimates (mean |SHAP| of each family's total effect per product, test products, log units)",
        "",
        "For the price, SHAP explains the markup over the landed cost (the cost itself enters 1:1, outside the trees, "
        "and also as a feature of the markup).",
        "",
        _md(pd.DataFrame({"sales": imp_s, "price (markup)": imp_p}).fillna(0).sort_values("sales", ascending=False),
            digits=3),
        "",
        "## The delivered estimates (`data/processed/forecast.parquet`)",
        "",
        "Never in-sample: products with a known whole-life sales or price get a **cross-validated** estimate (5 folds "
        "grouped by model, so neither the product nor its other colours were seen); the others (new products, "
        f"{TEST_SEASON} products whose sales were cut, products never sold) a forecast from all usable products. Sales "
        "are **whole-life units** (median-type estimate: exp of the mean log); price is landed cost × markup, also "
        "snapped to the nearest list price (`price_point_estimate`). **Typical error measured on the test split:** "
        f"sales × {typical['sales']:.2f} either way (exp of the mean absolute log error; "
        f"{typical['sales_within_2x']:.0%} of the products within a factor 2), price ± {typical['price']:.2f} €. "
        "Units per product only: the data has no store or week breakdown.",
        "",
        f"**80% range** per product (`sales_low`/`sales_high`, `price_low`/`price_high`): the estimate times the 10th "
        f"and 90th percentiles of real / estimate on the test split (sales ×{np.exp(typical['sales_range_log'][0]):.2f}"
        f" to ×{np.exp(typical['sales_range_log'][1]):.2f}, price ×{np.exp(typical['price_range_log'][0]):.2f} to "
        f"×{np.exp(typical['price_range_log'][1]):.2f}). A readable copy for every product, with the description, "
        "is `outputs/phase5b/estimates.csv`.",
        "",
        "## How the explanations are computed",
        "",
        "- **Gradient boosting** (LightGBM): exact TreeSHAP values (`pred_contrib`), adding up to the estimate, "
        "milliseconds per product; grouped into business families because single principal components mean nothing.",
        "- **Similar past products**: their similarity-weighted mean, so the products themselves are the explanation. "
        "Caveat: the engine's weights were tuned in Phase 4 with an objective that includes the sales correlation, so "
        "this reference is slightly optimistic. The delivered sales reference averages whole-life products only.",
        "",
    ]
    (REPORT_DIR / "forecast_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run", "select"])
    args = ap.parse_args()
    items, blocks = load()
    weights = engine.load_weights()
    if args.cmd == "select":
        res = select_features(items, blocks, weights)
        lines = ["# Phase 5b: feature selection (fast mode, 200 bootstrap resamples)", "",
                 "Decided on validation only: the simplest set whose validation error is not clearly worse than the "
                 "best set's (95% interval of the gain vs the best reaches 0). Test columns are shown, never used.", "",
                 f"**Applied** (`FEATURES`): sales `{FEATURES['log_sales']}`, price `{FEATURES['log_price']}`, by the "
                 f"declared rule: a block leaves only if it helps in neither check, the second being the same-month "
                 f"ranking on {TEST_SEASON} (`forecast_report.md`), where the photo helps the sales ranking.", ""]
        drift = res.pop("price_drift")
        for t, r in res.items():
            lines += [f"## {t}: best on validation `{r['best']}`, chosen `{r['chosen']}`", "",
                      _md(r["table"], pct={"val_within_2x", "test_within_2x", "val_bias", "test_bias",
                                           "val_within_1_point", "test_within_1_point"}), ""]
        lines += ["## price: corrections for the rise of markups (232 + 241 → 242, halves of the new models)", "",
                  "Adopted only if the gain on half A excludes 0; half B shown, never used.", "",
                  _md(drift, pct={"A_bias", "B_bias"}), ""]
        (REPORT_DIR / "feature_selection.md").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        return
    checks = data_checks(items)

    val = {"sales": run_sales(items, blocks, weights, *split(items, *SALES_VALIDATION)),
           "price": run_price(items, blocks, weights, *split(items, *VALIDATION))}
    # the choice is computed: the similar products enter as a feature only if their validation gain is clearly positive
    use_analogue = {t: bool(val[t]["ci_analogue_gain"][0] > 0) for t in ("sales", "price")}
    st, pt = val["sales"]["table"], val["price"]["table"]
    chosen_name = {t: val[t]["variants"][use_analogue[t]][2] for t in use_analogue}
    decisions = {
        "model": "Gradient boosting against the best baseline on validation: units error ×"
                 f"{np.exp(st[chosen_name['sales']]['mae_log']):.2f} vs ×"
                 f"{np.exp(min(v['mae_log'] for k, v in st.items() if 'boosting' not in k and 'reference' not in k)):.2f}; "
                 f"price error {pt[chosen_name['price']]['mae_eur']:.2f} € vs "
                 f"{min(v['mae_eur'] for k, v in pt.items() if k != chosen_name['price'] and 'analogues' not in k):.2f} €.",
        **{f"analogue_{t}": f"Similar past products as a feature ({t}): validation gain "
                            f"{val[t]['ci_analogue_gain'][0]:+.3f} to {val[t]['ci_analogue_gain'][1]:+.3f} "
                            + ("(same-month ranking) -> kept." if t == "sales" and use_analogue[t] else
                               "(same-month ranking) -> not kept (interval includes 0): the simpler model." if t == "sales" else
                               "(€ of error saved) -> kept." if use_analogue[t] else
                               "(€ of error saved) -> not kept (interval includes 0): the simpler model.")
           for t in ("sales", "price")},
    }

    s_tr, s_te = split(items, *SALES_TEST)
    p_tr, p_te = split(items, *TEST)
    test = {"sales": run_sales(items, blocks, weights, s_tr, s_te), "price": run_price(items, blocks, weights, p_tr, p_te)}
    for t in test:                                     # the chosen variant, decided on validation
        test[t]["model"], test[t]["X"], test[t]["chosen"] = test[t]["variants"][use_analogue[t]]
    rank242 = run_sales(items, blocks, weights, p_tr, p_te)
    leak = leakage_checks(items, blocks, s_tr, s_te)
    leak.to_csv(REPORT_DIR / "leakage_checks.csv")
    shift = class_shift(items)
    imp_s = global_importance(test["sales"]["model"], test["sales"]["X"], test["sales"]["eval_rows"])
    imp_p = global_importance(test["price"]["model"], test["price"]["X"],
                              test["price"]["eval_rows"] & np.isfinite(items["log_cost"].to_numpy(float)))
    ts, tp = test["sales"]["table"][test["sales"]["chosen"]], test["price"]["table"][test["price"]["chosen"]]
    typical = {"sales": float(np.exp(ts["mae_log"])), "sales_within_2x": ts["within_2x"], "price": tp["mae_eur"]}
    for t in ("sales", "price"):                       # 80% range: 10th-90th percentile of real / estimate on the test
        r = test[t]["eval_rows"]
        res = (test[t]["y"] - test[t]["preds"][test[t]["chosen"]])[r]
        typical[f"{t}_range_log"] = [float(np.quantile(res, 0.1)), float(np.quantile(res, 0.9))]
    write_report(checks, val, test, rank242, decisions, leak, shift, imp_s, imp_p, typical, bias_tables(items, test))
    final = final_models(items, blocks, weights, typical, use_analogue)
    final.to_parquet(PROCESSED_DIR / OUT_FILE, index=False)
    estimates_csv(items, final).to_csv(REPORT_DIR / "estimates.csv", index=False, encoding="utf-8")
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sales_validation": SALES_VALIDATION, "sales_test": SALES_TEST, "price_validation": VALIDATION,
            "price_test": TEST, "complete_seasons": COMPLETE_SEASONS,
            "k_analogues": K_ANALOGUES, "pcs": PCS, "lgb_params": LGB_PARAMS, "weights": weights,
            "image_embedding": engine.image_embedding(), "decisions": decisions, "use_analogue": use_analogue,
            "typical_error": typical,
            "sales_test_table": test["sales"]["table"], "price_test_table": test["price"]["table"],
            "sales_rank_242": rank242["table"]}
    (PROCESSED_DIR / "forecast.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    print(pd.DataFrame(test["sales"]["table"]).T.round(3).to_string())
    print(pd.DataFrame(test["price"]["table"]).T.round(3).to_string())
    print(pd.DataFrame(rank242["table"]).T.round(3).to_string())
    print(leak.round(3).to_string())
    print(decisions)
    print(f"-> {PROCESSED_DIR / OUT_FILE}, report {REPORT_DIR / 'forecast_report.md'}")


if __name__ == "__main__":
    main()
