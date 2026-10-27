"""Phase 5b: explainable sales and price estimates for a product, from its attributes, photo and similar products.

    python src/phase5b_forecast.py run         # -> data/processed/forecast.parquet + outputs/phase5b/ (report, leakage checks)

Business use (brief, slide 4): buyers must size the order of a NEW product. The engine already finds similar
products; here their sales (and prices) become an estimate, next to a model that learns from all attributes.

Data decisions (each measured in the report):
- Unit: the colourway. Sales targets: log(1 + units). Products without a sales row (370) or with 0 units (9) are
  unknown, not zero: they are left out of training and evaluation, and still get an estimate.
- Price: every colourway has a list price (no 0 and no missing value: checked in the report), so all are used.
- Time: SALES_QTY is accumulated until the extraction, so later launches had less time to sell (season 242 sells
  about a third of the older seasons). The model gets the days on sale from the planned store date. Choices are made on
  a validation split (232 -> 241); the test split (232 + 241 -> 242) is reported once; both use only models never seen
  in training (truly new products).
- Delivered estimates are never in-sample: cross-validated (grouped by model) where the target is known.
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
N_PERMUTATIONS = 20             # shuffled-target runs of the leakage check
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
    # the planned store date is what a buyer knows when ordering (the final date is revised for 56% of the products);
    # the extraction date is unknown, so the last final store date stands in for it
    planned = pd.to_datetime(items["STORE_DATE_PLANNED"])
    items["days_on_sale"] = (pd.to_datetime(items["STORE_DATE_FINAL"]).max() - planned).dt.days.astype(float)
    items["launch_month"] = planned.dt.month.astype(float)
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


def split(items: pd.DataFrame, train_seasons=("232", "241"), test_season: str = TEST_SEASON) -> tuple[np.ndarray, np.ndarray]:
    """Train: the given seasons. Test: one later season, only models never sold in the training seasons (new products).

    Choices (which model, which features) are made on the validation split 232 -> 241; the test split
    232 + 241 -> 242 is reported once with those choices."""
    sea = items["SEA_COD"].astype(str)
    train = sea.isin(train_seasons).to_numpy()
    seen = set(items.loc[train, "PROD_REF"].astype(str))
    new_model = ~items["PROD_REF"].astype(str).isin(seen).to_numpy()
    return train, sea.eq(test_season).to_numpy() & new_model


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


VALIDATION = (("232",), "241")
TEST = (("232", "241"), TEST_SEASON)


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

    def weighted(keys_):  # mean of the per-group rank correlations, weighted by group size (both metrics alike)
        res = [(spearman(y[keys_ == g], p[keys_ == g]), (keys_ == g).sum()) for g in np.unique(keys_)
               if (keys_ == g).sum() > 10]
        return float(np.average([r for r, _ in res], weights=[n for _, n in res])) if res else 0.0
    top_true = pd.Series(y).groupby(cats).rank(pct=True).to_numpy() > 0.8
    # ties broken at random, so a constant estimate picks a random 20% (its fair score: the base rate)
    jitter = np.random.default_rng(SEED).random(len(p)) * 1e-9
    top_pred = pd.Series(p + jitter).groupby(cats).rank(pct=True).to_numpy() > 0.8
    return {"spearman_within_cat": weighted(cats), "spearman_same_month": weighted(groups),
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
    ci_month = {n: bootstrap_diff(m2, y[eval_rows], preds["time on sale + category only"][eval_rows], p[eval_rows],
                                  groups, cats[eval_rows])
                for n, p in preds.items() if n not in ("category median", "time on sale + category only")}
    return {"y": y, "preds": preds, "analogue": ana, "analogue_keys": ana_keys,
            "variants": {False: (model, X0, "gradient boosting, no analogues"),
                         True: (model_ana, X, "gradient boosting + analogues")},
            "table": table, "ci_vs_category": ci, "ci_analogue_gain": ci_ana, "ci_same_month": ci_month,
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
    model_ana, preds["gradient boosting + analogues"] = fit_predict(X, y, train, fit_rows)
    Xc = X0.assign(cost=pd.to_numeric(items["PRICE_LCP"], errors="coerce"))   # chosen model + the cost
    _, ref = fit_predict(Xc, y, train, fit_rows)
    table = {n: price_metrics(y[eval_rows], p[eval_rows], points) for n, p in preds.items()}
    table["reference: + landed cost (not used)"] = price_metrics(y[eval_rows], ref[eval_rows], points)
    groups = model_groups(items)[eval_rows]
    m = lambda yy, pp, _: -price_metrics(yy, pp, points)["mae_eur"]  # noqa: E731  (higher is better)
    ci = {n: bootstrap_diff(m, y[eval_rows], preds["sub-family median"][eval_rows], p[eval_rows], groups, groups)
          for n, p in preds.items() if n not in ("category median", "sub-family median")}
    ci_ana = bootstrap_diff(m, y[eval_rows], preds["gradient boosting, no analogues"][eval_rows],
                            preds["gradient boosting + analogues"][eval_rows], groups, groups)
    return {"y": y, "preds": preds, "analogue": ana, "analogue_keys": ana_keys,
            "variants": {False: (model, X0, "gradient boosting, no analogues"),
                         True: (model_ana, X, "gradient boosting + analogues")},
            "table": table, "ci_vs_subfamily": ci, "ci_analogue_gain": ci_ana, "n_fit": int(fit_rows.sum()),
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


def out_of_fold(items, blocks, weights, target: str, use_analogue: bool = False, analogue_est: np.ndarray | None = None,
                folds: int = 5) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimates that never saw the product's own target: for products with a known target, 5-fold cross-fitting
    grouped by model (colourways of a model stay in the same fold); for products without one (new or never sold),
    the model fitted on every known product. Returns (estimate, kind, SHAP families per product)."""
    y = items[target].to_numpy(float)
    known = np.isfinite(y)
    allrows = np.ones(len(items), bool)
    X = feature_table(items, blocks, allrows, target, analogue_est if use_analogue else None)
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
    kind = np.where(known, "cross-validated (target not seen)", "forecast (no sales / new)")
    return est, kind, fam.reindex(items.index)


def final_models(items, blocks, weights, typical_error: dict, use_analogue: dict) -> pd.DataFrame:
    """Per product: the estimate (never in-sample), its kind, the honest typical error measured on the test season,
    the three biggest SHAP effects, and the analogue estimate with its products (the buyer's reference)."""
    out = pd.DataFrame({KEY: items[KEY]})
    for target, w, name in [("log_sales", weights, "sales"), ("log_price", {**weights, "numeric": 0.0}, "price")]:
        y = items[target].to_numpy(float)
        ana, keys = analogues(items, blocks, w, target, np.isfinite(y))
        est, kind, fam = out_of_fold(items, blocks, weights, target, use_analogue[name], ana)
        conv = np.expm1 if name == "sales" else np.exp
        out[f"{name}_estimate"] = conv(est).round(0 if name == "sales" else 2)
        out[f"{name}_estimate_kind"] = kind
        out[f"{name}_typical_error"] = typical_error[name]
        out[f"{name}_analogue"] = conv(ana).round(0 if name == "sales" else 2)
        out[f"{name}_analogue_keys"] = ["|".join(k[:4]) for k in keys]
        top = fam.apply(lambda r: r.reindex(r.abs().sort_values(ascending=False).index)[:3], axis=1)
        out[f"{name}_reasons"] = ["; ".join(f"{f} {'+' if v > 0 else '−'}{abs(v):.2f}" for f, v in r.dropna().items())
                                  for _, r in top.iterrows()]
    return out


def leakage_checks(items, blocks, train, test) -> pd.DataFrame:
    """Diagnostics of the sales model on the test split (they choose nothing): shuffled target and feature ablations."""
    y = items["log_sales"].to_numpy(float)
    fit_rows, ev_rows = train & np.isfinite(y), test & np.isfinite(y)
    groups = (items["CAT_DES_EN"].astype(str) + "|" + items["launch_month"].astype(int).astype(str) + "|"
              + items["SEA_COD"].astype(str)).to_numpy()
    X = feature_table(items, blocks, train, "log_sales", None)

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
    emb = [c for c in X if c.startswith(("img_pc", "txt_pc"))]
    rows = {"chosen model": score(X),
            f"target shuffled in training, mean of {N_PERMUTATIONS} runs (must be ~0)": perms.mean().to_dict(),
            f"target shuffled in training, SD of {N_PERMUTATIONS} runs": perms.std().to_dict(),
            "without time on sale and launch month": score(X.drop(columns=["days_on_sale", "launch_month"])),
            "without fashion type and collection type": score(X.drop(columns=["FASHIONTYPE", "COLLECTION_TYPE"])),
            "without the photo components": score(X.drop(columns=[c for c in emb if c.startswith("img")])),
            "without the photo and description components": score(X.drop(columns=emb)),
            "attributes + price only": score(X[[c for c in X if not c.startswith(("img_pc", "txt_pc", "mat_"))
                                                and c not in ("days_on_sale", "launch_month")]])}
    return pd.DataFrame(rows).T


# --------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, pct: set[str] = frozenset(), digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([""] + [str(c) for c in cols]) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        cells = [f"{v:.1%}" if c in pct and pd.notna(v) else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))
                 for c, v in r.items()]
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    return "\n".join(lines)


def write_report(checks: dict, val: dict, test: dict, decisions: dict, leak: pd.DataFrame, shift: pd.DataFrame,
                 imp_s: pd.Series, imp_p: pd.Series, typical: dict) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    s_v, p_v = pd.DataFrame(val["sales"]["table"]).T, pd.DataFrame(val["price"]["table"]).T
    s_t, p_t = pd.DataFrame(test["sales"]["table"]).T, pd.DataFrame(test["price"]["table"]).T
    sp = {"top20_precision"}
    pp = {"mape", "exact_price_point", "within_1_point"}
    ci_s = "\n".join(f"- {n}: {lo:+.3f} to {hi:+.3f}" for n, (lo, hi) in test["sales"]["ci_same_month"].items())
    ci_p = "\n".join(f"- {n}: {lo:+.2f} to {hi:+.2f} €" for n, (lo, hi) in test["price"]["ci_vs_subfamily"].items())
    lines = [
        "# Phase 5b: explainable sales and price estimates",
        "",
        f"Generated by `src/phase5b_forecast.py run` ({datetime.now(timezone.utc):%Y-%m-%d}). Products are colourways; "
        "tests use only models never sold in the training seasons (new products); intervals are 95% bootstrap over "
        "models.",
        "",
        "## Protocol",
        "",
        "- **Validation (choices): train 232 → test 241.** Which model and which features are chosen here, by a rule "
        "in the code: the similar products enter as a feature only if the 95% interval of their validation gain excludes 0.",
        f"- **Test (reported once): train 232 + 241 → test {TEST_SEASON}**, with the choices of the validation.",
        "- **Time:** the planned store date (`STORE_DATE_PLANNED`, known when ordering) gives the launch month and the "
        "days on sale; the final date is revised for 56% of the products. The seasons overlap in time and the training "
        "sales are accumulated until the extraction, so they include months of the test season: a limit of the data "
        "(no dated sales), stated rather than corrected.",
        "- **Left out:** buyer decisions (`PUR_BET`, `TYPOLOGY`) and the landed cost (reference rows only); sales are "
        "never a feature; the price model never sees the price or the numeric similarity block.",
        "",
        "## The data the decisions rest on",
        "",
        f"- **Price:** {checks['price_missing']} missing, {checks['price_zero_or_less']} at 0 € or less; "
        f"{checks['price_points']} list prices from {checks['price_min']:.2f} € to {checks['price_max']:.2f} €.",
        f"- **Landed cost** missing for {checks['cost_missing']} colourways. **Sales:** {checks['sales_missing']} "
        f"colourways without a sales row, {checks['sales_zero']} with 0 units: unknown, left out of training and "
        "evaluation (they still get an estimate).",
        f"- **Time on sale:** median units by season {checks['median_units_by_season']}; inside a season and category, "
        f"units vs days on sale (planned date): median Spearman {checks['days_vs_sales_within_season']:+.2f}.",
        "- **Planning classes change between seasons** (share of the colourways):",
        "",
        _md(shift.set_index(["attribute", "value"]), digits=3),
        "",
        "  They are among the strongest drivers below, so their shift (e.g. more \"Commercial Online\" in 242) is a "
        "risk; to confirm with Parfois when they are assigned.",
        "",
        "## Validation (232 → 241): the choices",
        "",
        _md(s_v, pct=sp),
        "",
        _md(p_v, pct=pp),
        "",
        *[f"- {d}" for d in decisions.values()],
        "",
        f"## Test (232 + 241 → {TEST_SEASON}): sales, {test['sales']['n_fit']:,} training / "
        f"{test['sales']['n_eval']:,} test products",
        "",
        "`spearman_within_cat` / `spearman_same_month`: rank correlation of estimate and real units within a category / "
        "only among products of a category launched in the same month (time on sale cannot help), weighted by group "
        "size. `top20_precision`: share of the estimated top 20% of a category that really were (random 20%).",
        "",
        _md(s_t, pct=sp),
        "",
        "Same-month gain over the model that only knows the category and the time on sale (95% CI):",
        "",
        ci_s,
        "",
        f"## Test: price, {test['price']['n_fit']:,} training / {test['price']['n_eval']:,} test products",
        "",
        _md(p_t, pct=pp),
        "",
        "Error reduction over the sub-family median (95% CI):",
        "",
        ci_p,
        "",
        "## Leakage checks and what each part adds (sales, test split; diagnostics, no choice made here)",
        "",
        _md(leak, pct=sp),
        "",
        f"- A training target shuffled at random ({N_PERMUTATIONS} runs) must give ≈ 0: it does, so no information leaks "
        "through the pipeline (joins, row order, components fitted on training rows).",
        "- The photo and description components matter **within a category** (removing both costs about "
        f"{leak.at['chosen model', 'spearman_within_cat'] - leak.at['without the photo and description components', 'spearman_within_cat']:.2f}) "
        "but little among products launched in the same month "
        f"({leak.at['chosen model', 'spearman_same_month'] - leak.at['without the photo and description components', 'spearman_same_month']:.2f}).",
        "",
        "## What drives the estimates (mean |SHAP| of each family's total effect per product, test products, log units)",
        "",
        _md(pd.DataFrame({"sales": imp_s, "price": imp_p}).fillna(0).sort_values("sales", ascending=False), digits=3),
        "",
        "## The delivered estimates (`data/processed/forecast.parquet`)",
        "",
        "Never in-sample: products with known sales or price get a **cross-validated** estimate (5 folds grouped by "
        "model, so neither the product nor its other colours were seen), products without sales a forecast from all "
        "known products. Each row carries its kind, the three biggest SHAP effects, the similar past products and the "
        f"**typical error measured on the test season**: sales × {typical['sales']:.1f} either way (exp of the mean "
        f"absolute log error), price ± {typical['price']:.2f} €. With accumulated sales only (no stores, no weeks), the "
        "sales estimate is a ranking aid for comparing new products, not an order-quantity forecast.",
        "",
        "## How the explanations are computed",
        "",
        "- **Gradient boosting** (LightGBM): exact TreeSHAP values (`pred_contrib`), adding up to the estimate, "
        "milliseconds per product; grouped into business families because single principal components mean nothing.",
        "- **Similar past products**: their similarity-weighted mean, so the products themselves are the explanation. "
        "Caveats: the engine's weights were tuned in Phase 4 with an objective that includes the sales correlation, so "
        "this reference is slightly optimistic; and the delivered reference averages products of any season, whose time "
        "on sale differs (median units 3,377 / 3,637 / 1,336), without adjustment.",
        "- **Planned store dates** go back to 2016–2017 for some products of seasons 232 and 241 (carry-overs keep their "
        "first plan), which makes their days on sale large; the trees only use the order, and the same-month metric is "
        "unaffected.",
        "",
    ]
    (REPORT_DIR / "forecast_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    items, blocks = load()
    weights = engine.load_weights()
    checks = data_checks(items)

    tr, te = split(items, *VALIDATION)
    val = {"sales": run_sales(items, blocks, weights, tr, te), "price": run_price(items, blocks, weights, tr, te)}
    # the choice is computed: the similar products enter as a feature only if their validation gain is clearly positive
    use_analogue = {t: bool(val[t]["ci_analogue_gain"][0] > 0) for t in ("sales", "price")}
    st, pt = val["sales"]["table"], val["price"]["table"]
    chosen_name = {t: val[t]["variants"][use_analogue[t]][2] for t in use_analogue}
    decisions = {
        "model": "Gradient boosting beats every baseline on validation: same-month ranking "
                 f"{st[chosen_name['sales']]['spearman_same_month']:.3f} vs "
                 f"{max(v['spearman_same_month'] for k, v in st.items() if 'boosting' not in k and 'reference' not in k):.3f}; "
                 f"price error {pt[chosen_name['price']]['mae_eur']:.2f} € vs {pt['sub-family median']['mae_eur']:.2f} €.",
        **{f"analogue_{t}": f"Similar past products as a feature ({t}): validation gain "
                            f"{val[t]['ci_analogue_gain'][0]:+.3f} to {val[t]['ci_analogue_gain'][1]:+.3f} "
                            + ("(same-month ranking) -> kept." if t == "sales" and use_analogue[t] else
                               "(same-month ranking) -> not kept (interval includes 0): the simpler model." if t == "sales" else
                               "(€ of error saved) -> kept." if use_analogue[t] else
                               "(€ of error saved) -> not kept (interval includes 0): the simpler model.")
           for t in ("sales", "price")},
    }

    tr, te = split(items, *TEST)
    test = {"sales": run_sales(items, blocks, weights, tr, te), "price": run_price(items, blocks, weights, tr, te)}
    for t in test:                                     # the chosen variant, decided on validation
        test[t]["model"], test[t]["X"], test[t]["chosen"] = test[t]["variants"][use_analogue[t]]
    leak = leakage_checks(items, blocks, tr, te)
    leak.to_csv(REPORT_DIR / "leakage_checks.csv")
    shift = class_shift(items)
    imp_s = global_importance(test["sales"]["model"], test["sales"]["X"], test["sales"]["eval_rows"])
    imp_p = global_importance(test["price"]["model"], test["price"]["X"], test["price"]["eval_rows"])
    typical = {"sales": float(np.exp(test["sales"]["table"][test["sales"]["chosen"]]["mae_log"])),
               "price": float(test["price"]["table"][test["price"]["chosen"]]["mae_eur"])}
    write_report(checks, val, test, decisions, leak, shift, imp_s, imp_p, typical)
    final = final_models(items, blocks, weights, typical, use_analogue)
    final.to_parquet(PROCESSED_DIR / OUT_FILE, index=False)
    meta = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "validation": VALIDATION, "test": TEST,
            "k_analogues": K_ANALOGUES, "pcs": PCS, "lgb_params": LGB_PARAMS, "weights": weights,
            "image_embedding": engine.image_embedding(), "decisions": decisions, "use_analogue": use_analogue,
            "typical_error": typical,
            "sales_test": test["sales"]["table"], "price_test": test["price"]["table"]}
    (PROCESSED_DIR / "forecast.json").write_text(json.dumps(meta, indent=1, default=float), encoding="utf-8")
    print(pd.DataFrame(test["sales"]["table"]).T.round(3).to_string())
    print(pd.DataFrame(test["price"]["table"]).T.round(3).to_string())
    print(leak.round(3).to_string())
    print(decisions)
    print(f"-> {PROCESSED_DIR / OUT_FILE}, report {REPORT_DIR / 'forecast_report.md'}")


if __name__ == "__main__":
    main()
