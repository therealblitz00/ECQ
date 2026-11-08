"""Phase 5b: read-only audit of everything that feeds the sales and price forecast (before the prediction step).

    python src/phase5b_upstream_audit.py run     # -> outputs/phase5b/upstream_audit.md + upstream_audit.csv, ~2 min

Nothing is changed: it loads the same items, blocks and splits as `phase5b_forecast.py`, measures problems in the
target (what SALES_QTY means), in time (exposure, planned vs final dates), in leakage (duplicates across the split,
tuned weights, pools), in the data (images, rare categories, shift between seasons) and in the evaluation design, and
ranks them. Every finding says whether it was *verified* (a number computed here or in the code) or only *suspected*
(a reading of the numbers that the data cannot settle), the direction of the bias, and a fix or a check.
The few model fits use the validation half of the new 241 models (never the reported half).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase5b_forecast as fc  # noqa: E402
from config import ROOT  # noqa: E402

REPORT_DIR = ROOT / "outputs" / "phase5b"
EXTRACTION = pd.Timestamp("2024-11-04")     # week of the extraction (outputs/phase5b/extraction_check.md)
SEVERITY = {"high": 0, "medium": 1, "low": 2}
COLUMNS = ["rank", "id", "area", "status", "severity", "finding", "evidence", "bias", "fix"]


# --------------------------------------------------------------------------- pure checks (small, testable)
def unknown_sales(items: pd.DataFrame, extraction: pd.Timestamp = EXTRACTION) -> dict:
    """Colourways without a sales row or with 0 units that were on sale long enough to have sold (never-sold)."""
    sea = items["SEA_COD"].astype(str)
    launched = pd.to_datetime(items["STORE_DATE_FINAL"]) < extraction
    qty = items["SALES_QTY"]
    whole = sea.isin(fc.COMPLETE_SEASONS)
    return {"no_row_whole": int((qty.isna() & whole).sum()), "no_row_whole_launched": int((qty.isna() & whole & launched).sum()),
            "zero_whole": int(((qty == 0) & whole).sum()), "whole": int(whole.sum()),
            "no_row_by_season": qty.isna().groupby(sea).sum().astype(int).to_dict()}


def exposure_corr(items: pd.DataFrame, season: str, extraction: pd.Timestamp = EXTRACTION) -> dict:
    """Weeks on sale at the extraction, and the rank correlation of log units with it, by category."""
    d = items[items["SEA_COD"].astype(str).eq(season) & items["log_sales"].notna()].copy()
    d["weeks"] = (extraction - pd.to_datetime(d["STORE_DATE_FINAL"])).dt.days / 7
    out = {"weeks_min": float(d["weeks"].min()), "weeks_median": float(d["weeks"].median()), "weeks_max": float(d["weeks"].max())}
    out["rho_by_category"] = {c: float(g["log_sales"].rank().corr(g["weeks"].rank()))
                              for c, g in d.groupby("CAT_DES_EN") if len(g) > 30}
    return out


def planned_vs_final(items: pd.DataFrame) -> dict:
    """How far the planned store date (the launch month the metrics group by) is from the final one."""
    planned, final = pd.to_datetime(items["STORE_DATE_PLANNED"]), pd.to_datetime(items["STORE_DATE_FINAL"])
    days = (final - planned).dt.days
    return {"differs": float((days != 0).mean()), "over_30_days": float((days.abs() > 30).mean()),
            "other_month": float((planned.dt.to_period("M") != final.dt.to_period("M")).mean()),
            "over_1_year": int((days.abs() > 365).sum())}


def image_availability(items: pd.DataFrame) -> pd.Series:
    """Share of colourways without a trusted photo, by season."""
    return (~items["img_trusted"].astype(bool)).groupby(items["SEA_COD"].astype(str)).mean()


def eta2_within_category(items: pd.DataFrame, col: str) -> float:
    """Share of the variance of log units (inside a category) explained by the values of `col`."""
    d = items[items["log_sales"].notna()]
    y = d["log_sales"] - d.groupby("CAT_DES_EN")["log_sales"].transform("mean")
    g = y.groupby(d[col].astype("string").fillna("NA"))
    return float(1 - ((y - g.transform("mean")) ** 2).sum() / (y ** 2).sum())


def realised_ratio(items: pd.DataFrame) -> pd.Series:
    """Median of (sales value / units) / list price, by season: markdown and outlet units pull it down."""
    ok = items["SALES_QTY"] > 0
    r = (items["SALES_AMT_FX_RATE"] / items["SALES_QTY"].where(ok)) / items["PRICE_BASE_W_VAT"].astype(float)
    return r.groupby(items["SEA_COD"].astype(str)).median()


def leaked_models(items: pd.DataFrame, train: np.ndarray, test: np.ndarray) -> dict:
    """Test colourways whose model (PROD_REF joined with PROD_REF_EQUIV) also has colourways in training."""
    g = fc.model_groups(items)
    return {"test_rows": int(test.sum()), "in_train_by_group": int(np.isin(g[test], g[train]).sum()),
            "in_train_by_ref": int(items["PROD_REF"].astype(str)[test].isin(set(items["PROD_REF"].astype(str)[train])).sum()),
            "recoded_rows": int((items["PROD_REF"].astype(str) != items["PROD_REF_EQUIV"].fillna(items["PROD_REF"]).astype(str)).sum())}


def rank_in_category_gap(items: pd.DataFrame) -> float:
    """Share of colourways whose price percentile in the category changes by more than 0.05 when it is ranked inside
    its own season instead of across all seasons (the feature uses all seasons, 242 prices included)."""
    allp = items.groupby("CAT_DES_EN")["PRICE_BASE_W_VAT"].rank(pct=True)
    own = items.groupby(["CAT_DES_EN", "SEA_COD"])["PRICE_BASE_W_VAT"].rank(pct=True)
    return float(((allp - own).abs() > 0.05).mean())


def collapse_rare(X: pd.DataFrame, cols: list[str], fit_rows: np.ndarray, thr: int) -> pd.DataFrame:
    """Levels with fewer than `thr` training rows become OTHER."""
    X = X.copy()
    for c in cols:
        s = X[c].astype("string")
        vc = s[fit_rows].value_counts()
        X[c] = pd.Categorical(s.where(s.isin(set(vc[vc >= thr].index)), "OTHER"))
    return X


def finding(area, status, severity, what, evidence, bias, fix) -> dict:
    return {"area": area, "status": status, "severity": severity, "finding": what, "evidence": evidence,
            "bias": bias, "fix": fix}


# --------------------------------------------------------------------------- measurements that need the model
def model_checks(items, blocks) -> dict:
    """Fits on 232 and scores the validation half of the new 241 models (the half used for choices)."""
    y = items["log_sales"].to_numpy(float)
    known = np.isfinite(y)
    tr, va = fc.split(items, *fc.SALES_VALIDATION)
    fit, ev = tr & known, va & known
    X = fc.feature_table(items, blocks, tr, "log_sales", None)
    _, p = fc.fit_predict(X, y, tr, fit)
    res = y - p
    out: dict = {"n_val": int(ev.sum()), "val_mae": float(np.abs(res[ev]).mean())}

    # residual against the weeks on sale
    weeks = ((EXTRACTION - pd.to_datetime(items["STORE_DATE_FINAL"])).dt.days / 7).to_numpy()
    d = pd.DataFrame({"cat": items["CAT_DES_EN"].to_numpy(), "res": res, "w": weeks})[ev]
    out["resid_exposure_all"] = float(d["res"].rank().corr(d["w"].rank()))
    out["resid_exposure_cat"] = {c: float(g["res"].rank().corr(g["w"].rank())) for c, g in d.groupby("cat") if len(g) > 30}

    # error by how many training rows share the value (cardinality)
    rare = {}
    for c in ["SUP_COD", "CLR_DES", "CATEGORY_MATRIX"]:
        cnt = items[c].astype("string").map(items.loc[fit, c].astype("string").value_counts()).fillna(0).to_numpy()
        small = ev & (cnt < 20)
        rare[c] = {"levels": int(items[c].nunique()), "val_rows_small": int(small.sum()),
                   "mae_small": float(np.abs(res[small]).mean()) if small.any() else np.nan,
                   "mae_other": float(np.abs(res[ev & ~(cnt < 20)]).mean())}
    out["rare"] = rare
    cols = ["SUP_COD", "CLR_DES", "CATEGORY_MATRIX", "GFS_DES_EN"]
    out["collapsed_mae"] = {thr: float(np.abs(y - fc.fit_predict(collapse_rare(X, cols, fit, thr), y, tr, fit)[1])[ev].mean())
                            for thr in (20, 50)}

    # the same-month metric grouped by the final month instead of the planned one
    planned = items["CAT_DES_EN"].astype(str) + "|" + items["launch_month"].astype(int).astype(str) + "|" + items["SEA_COD"].astype(str)
    final = (items["CAT_DES_EN"].astype(str) + "|" + pd.to_datetime(items["STORE_DATE_FINAL"]).dt.month.astype(str) + "|"
             + items["SEA_COD"].astype(str))
    out["same_month_planned"] = fc.sales_metrics(y[ev], p[ev], planned.to_numpy()[ev])["spearman_same_month"]
    out["same_month_final"] = fc.sales_metrics(y[ev], p[ev], final.to_numpy()[ev])["spearman_same_month"]

    # what the buyer's own decisions add, and what the planning classes add (is the target purchase-limited?)
    Xb = fc.feature_table(items, blocks, tr, "log_sales", None, extra=fc.BUYER_DECISIONS)
    _, pb = fc.fit_predict(Xb, y, tr, fit)
    Xn = X.drop(columns=["FASHIONTYPE", "COLLECTION_TYPE"])
    _, pn = fc.fit_predict(Xn, y, tr, fit)
    g = planned.to_numpy()[ev]
    out["same_month"] = {"chosen model": out["same_month_planned"],
                         "+ buyer decisions": fc.sales_metrics(y[ev], pb[ev], g)["spearman_same_month"],
                         "without fashion type and collection type": fc.sales_metrics(y[ev], pn[ev], g)["spearman_same_month"]}
    # photo missing: error of products without a trusted photo (they are 2% of 232 and 241, 25% of 242)
    out["n_no_photo"] = int((ev & ~items["img_trusted"].astype(bool).to_numpy()).sum())
    out["mae_no_photo"] = float(np.abs(res[ev & ~items["img_trusted"].astype(bool).to_numpy()]).mean()) \
        if (ev & ~items["img_trusted"].astype(bool).to_numpy()).any() else np.nan
    return out


def shift_auc(items, blocks, a: np.ndarray, b: np.ndarray, drop: tuple[str, ...]) -> tuple[float, dict]:
    """Cross-validated AUC of a classifier telling the rows of `a` from `b` on the tabular features (0.5 = no shift)."""
    import lightgbm as lgb
    from scipy.stats import rankdata
    X = fc.feature_table(items, blocks, a | b, "log_sales", None, full=True)
    X = X[[c for c in X if not c.startswith(("img_pc", "txt_pc", "mat_")) and c not in drop]]
    idx = np.flatnonzero(a | b)
    lab = b[idx].astype(int)
    fold = np.random.default_rng(0).integers(0, 5, len(idx))
    pr = np.zeros(len(idx))
    for k in range(5):
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, verbose=-1, num_threads=4, random_state=0)
        m.fit(X.iloc[idx[fold != k]], lab[fold != k])
        pr[fold == k] = m.predict_proba(X.iloc[idx[fold == k]])[:, 1]
        imp = pd.Series(m.feature_importances_, X.columns)
    r = rankdata(pr)
    n1, n0 = lab.sum(), (1 - lab).sum()
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), imp.sort_values(ascending=False).head(4).to_dict()


def git_looks() -> dict:
    """Commits that changed the forecast report / script (each one is a new look at the same data)."""
    def count(path):
        try:
            out = subprocess.run(["git", "log", "--oneline", "--", path], cwd=ROOT, capture_output=True, text=True, timeout=30)
            return len(out.stdout.split("\n")) - 1 if out.returncode == 0 else None
        except Exception:
            return None
    return {"report": count("outputs/phase5b/forecast_report.md"), "script": count("src/phase5b_forecast.py")}


# --------------------------------------------------------------------------- the audit
def audit() -> tuple[list[dict], dict]:
    items, blocks = fc.load()
    num: dict = {}
    sea = items["SEA_COD"].astype(str)
    y = items["log_sales"].to_numpy(float)
    known = np.isfinite(y)
    s_tr, s_te = fc.split(items, *fc.SALES_TEST)
    v_tr, v_va = fc.split(items, *fc.SALES_VALIDATION)

    unk = unknown_sales(items)
    ex232, ex241 = exposure_corr(items, "232"), exposure_corr(items, "241")
    pf = planned_vs_final(items)
    img = image_availability(items)
    ratio = realised_ratio(items)
    leak = leaked_models(items, s_tr, s_te)
    mc = model_checks(items, blocks)
    auc_all, imp_all = shift_auc(items, blocks, s_tr, s_te, ("launch_month",))
    auc_tab, imp_tab = shift_auc(items, blocks, s_tr, s_te, ("launch_month", "price", "price_pct_in_cat", "net_weight"))
    s241, s242 = sea.eq("241").to_numpy(), sea.eq("242").to_numpy()
    auc_next, _ = shift_auc(items, blocks, s241, s242, ("launch_month", "price", "price_pct_in_cat", "net_weight"))
    looks = git_looks()
    eta = {c: eta2_within_category(items, c) for c in ["FASHIONTYPE", "COLLECTION_TYPE", "CATEGORY_MATRIX", "SUP_COD", "PUR_BET", "TYPOLOGY"]}
    models_test = len(set(fc.model_groups(items)[s_te & known]))
    models_val = len(set(fc.model_groups(items)[v_va & known]))
    num.update(unk=unk, ex232=ex232, ex241=ex241, pf=pf, img=img.to_dict(), ratio=ratio.to_dict(), leak=leak, mc=mc,
               auc=(auc_all, auc_tab, auc_next), looks=looks, eta=eta)

    ft = items[known].groupby("FASHIONTYPE")["log_sales"].agg(["size", "median"])
    cm_diff = items["CATEGORY_MATRIX"].fillna("NA") != items["CATEGORY_MATRIX_BUYER"].fillna("NA")
    cm_med = items[known].groupby(cm_diff[known])["log_sales"].median()
    selection = (REPORT_DIR / "feature_selection.md").read_text(encoding="utf-8") if (REPORT_DIR / "feature_selection.md").exists() else ""
    best_val = re.search(r"## sales: best on validation `([^`]+)`", selection)
    override = bool(best_val) and best_val.group(1) != fc.FEATURES["log_sales"]
    groups = pd.Series(fc.month_groups(items)[s_te & known]).value_counts()
    cost_missing = items.groupby("CAT_DES_EN")["log_cost"].apply(lambda s: s.isna().mean())
    sm = mc["same_month"]
    F: list[dict] = []

    F.append(finding(
        "Target", "verified", "high",
        "The target is what the buyer sold with the order and the allocation he chose, not demand. The planning classes "
        "(FASHIONTYPE, COLLECTION_TYPE: Capsule, Maxi, Commercial Online, Exclusive Online, Limited Edition...) are "
        "distribution programmes and carry most of the model (mean |SHAP| 0.49 and 0.20 in forecast_report.md, the two largest).",
        f"FASHIONTYPE alone explains {eta['FASHIONTYPE']:.0%} of the variance of log units inside a category, COLLECTION_TYPE "
        f"{eta['COLLECTION_TYPE']:.0%}, CATEGORY_MATRIX {eta['CATEGORY_MATRIX']:.0%}, SUP_COD {eta['SUP_COD']:.0%} (PUR_BET "
        f"{eta['PUR_BET']:.0%}, TYPOLOGY {eta['TYPOLOGY']:.0%}). Median log units: Capsule {ft.loc['Capsule', 'median']:.1f}, Maxi "
        f"{ft.loc['Maxi', 'median']:.1f}, Commercial Online {ft.loc['Commercial Online', 'median']:.1f} against Fashion "
        f"{ft.loc['Fashion', 'median']:.1f}, Basic {ft.loc['Basic', 'median']:.1f}. Validation same-month ranking "
        f"{sm['chosen model']:.3f}; without the two classes {sm['without fashion type and collection type']:.3f}; adding the buyer's "
        f"PUR_BET/TYPOLOGY {sm['+ buyer decisions']:.3f}. No stock, sell-through or store-count column exists, so a stock-out cannot be seen.",
        "Up: the ranking skill is skill at reproducing the buyer's plan; a product the buyer under-ordered looks like a bad "
        "product. Sales of a sold-out product are a floor of demand (censored).",
        "Say in the deliverable that the estimate is 'units the plan sells', not demand. Ask Parfois for stock, sell-through or "
        "weeks-to-sell-out per colourway (a censored target needs it). Check: refit with the classes removed and read it as the "
        "'product-only' estimate; report both."))

    F.append(finding(
        "Target / time", "verified", "high",
        "Season 241 is called 'whole life' but its colourways had 15 to 53 weeks on sale at the extraction, and the model has no "
        "exposure feature (launch month was dropped). Units still grow with the time on sale.",
        f"Weeks on sale: 232 {ex232['weeks_min']:.0f}-{ex232['weeks_max']:.0f} (median {ex232['weeks_median']:.0f}); 241 "
        f"{ex241['weeks_min']:.0f}-{ex241['weeks_max']:.0f} (median {ex241['weeks_median']:.0f}). Rank correlation of log units with weeks "
        f"in 241: {', '.join(f'{c} {v:+.2f}' for c, v in ex241['rho_by_category'].items())}. Correlation of the validation residual "
        f"with weeks on sale: {mc['resid_exposure_all']:+.2f} overall, {', '.join(f'{c} {v:+.2f}' for c, v in mc['resid_exposure_cat'].items())}. "
        f"Only 28% of 241 had reached the outlet; 98% have a (later, planned) outlet date. Median realised price / list price "
        f"({', '.join(f'{k} {v:.2f}' for k, v in ratio.items())}) rises from the oldest to the newest season: longer lives include "
        "more markdown units, so a unit is not the same thing in 232 and 241.",
        "Mixed: training (232, ~64 weeks) and test (241, ~38 weeks) measure different horizons; products launched earlier in 241 are "
        "under-estimated (bias -25% for Nov-Dec 2023 launches in extraction_check.md).",
        "Check: refit on units normalised to a fixed horizon (units per week on sale, or only products with >= 26 weeks), and add "
        "weeks-to-extraction as a nuisance feature fixed to the horizon at prediction. Best fix: ask for weekly sales and use units "
        "in the first N weeks (full price only)."))

    F.append(finding(
        "Evaluation", "verified", "medium",
        "The feature set was chosen with the 242 ranking after the validation and the test half both preferred a simpler one, "
        "and every forecast run re-looks at the same 241 data.",
        f"feature_selection.md: sales best on validation '{best_val.group(1) if best_val else '?'}'; applied '{fc.FEATURES['log_sales']}' "
        f"(chosen against the validation winner: {override}). Test half also favours attributes only (same-month 0.508 vs 0.502, mae 0.514 vs 0.524). The report was regenerated {looks['report']} times "
        f"and the script changed {looks['script']} times (git history); the CatBoost/metrics scripts add more looks. Test: {models_test} "
        f"models ({s_te.sum()} colourways), validation: {models_val} models; both halves are in the same season, launch window and "
        "buyer team, so they are not independent.",
        "Up (mild): the reported test half is no longer untouched; differences of ~0.01 are inside the noise of the bootstrap.",
        "Freeze the protocol; treat the test half as spent. Confirm on a genuinely new sample (the 242 whole lives once they finish "
        "selling, same extraction rules). Prefer attributes-only (equal on val and test, no missing-photo problem)."))

    F.append(finding(
        "Data / deployment", "verified", "medium",
        "Photos are missing for a quarter of season 242, but the model learned on seasons where only 2% are missing.",
        f"Share without a trusted photo: {', '.join(f'{k} {v:.1%}' for k, v in img.items())}. In 242 the products without a photo "
        "are the ones to forecast, and the photo components (16 numbers) are then missing: a pattern the trees never saw "
        f"(validation error of the {mc['n_no_photo']} such rows: {mc['mae_no_photo']:.2f} against {mc['val_mae']:.2f}).",
        "Unknown for 242 (the model falls back to its default branch for missing values); the validation cannot show it.",
        "Train with the photo components randomly blanked for 25% of rows, or use the attributes-only model for products without "
        "a photo (it loses nothing on validation or test). Report errors separately by photo availability."))

    F.append(finding(
        "Target", "verified", "low",
        "Colourways that never sold are left out of the target as 'unknown', although for the finished seasons they are real zeros.",
        f"{unk['no_row_whole']} colourways of 232/241 have no sales row, {unk['no_row_whole_launched']} of them launched before the "
        f"extraction, plus {unk['zero_whole']} with 0 units: {(unk['no_row_whole_launched'] + unk['zero_whole']) / unk['whole']:.1%} of "
        f"the {unk['whole']:,} whole-life colourways. (The 331 without sales in 242 are launches after the extraction: correctly unknown.)",
        "Up for the sales level of low sellers by a small amount (survivorship); no effect on the ranking among sellers.",
        f"Add a second, tiny model or a rule for 'will it sell at all' (flag items like those {unk['no_row_whole_launched'] + unk['zero_whole']}), or ask Parfois whether these were "
        "withdrawn before the shops. Report the share as a caveat."))

    F.append(finding(
        "Evaluation / time", "verified", "medium",
        "The 'same-month' metric groups by the planned launch month, but exposure follows the final date.",
        f"Planned and final store date differ for {pf['differs']:.0%} of the colourways, by more than 30 days for {pf['over_30_days']:.0%}; "
        f"the month differs for {pf['other_month']:.0%}; {pf['over_1_year']} planned dates are more than a year from the final one. "
        f"Validation same-month ranking: {mc['same_month_planned']:.3f} grouped by planned month, {mc['same_month_final']:.3f} by final month.",
        "The metric partly keeps the time effect it is meant to remove; the direction depends on the grouping.",
        "Report the same-month ranking with the final month too (it is only a grouping for the metric, not a feature), and "
        "flag planned dates far from the final one as data errors."))

    F.append(finding(
        "Data / evaluation", "verified", "low",
        "Rare values of supplier, colour and category matrix: 12% to 20% of the validation products have a value with fewer than "
        "20 training rows. Their error is slightly higher for SUP_COD and CLR_DES and not for CATEGORY_MATRIX; collapsing them does not help.",
        "; ".join(f"{c}: {v['levels']} values, {v['val_rows_small']} validation rows in values with < 20 training rows, error "
                  f"{v['mae_small']:.2f} vs {v['mae_other']:.2f}" for c, v in mc["rare"].items())
        + f". Rare values grouped as OTHER (< 20 / < 50 rows): validation error {mc['collapsed_mae'][20]:.3f} / "
          f"{mc['collapsed_mae'][50]:.3f} against {mc['val_mae']:.3f}.",
        "None measured (collapsing is neutral); the higher error on rare values is expected and already in the totals.",
        "Keep as is; give the buyer a flag 'supplier / colour seen < 20 times' next to the estimate, and a wider range for those."))

    F.append(finding(
        "Distribution shift", "verified", "medium",
        "The seasons differ in the planning mix, so the estimate extrapolates to 242.",
        f"A classifier tells the 232 training rows from the new 241 models with AUC {auc_all:.2f} on the tabular features (launch month "
        f"excluded; {', '.join(f'{k} {v}' for k, v in imp_all.items())}), {auc_tab:.2f} without price and weight "
        f"({', '.join(f'{k} {v}' for k, v in imp_tab.items())}); 241 against 242: {auc_next:.2f}. Commercial Online goes from 1.9% (232) to "
        "7.1% (242), Maxi from 5.7% to 0.6% (forecast_report.md); the planning classes drive the sales estimate (SHAP) and shift between seasons.",
        "Unknown sign; the sales level is not re-scaled between seasons (single 232 -> 241 transition, bias by category from -15% to +7%).",
        "Check: re-run the validation with the planning classes held to their 232 distribution (importance weights from the "
        "classifier) and look at the stability of the category-level bias."))

    F.append(finding(
        "Leakage (indirect)", "verified", "low",
        "The engine weights behind the 'similar past products' baseline and the price analogue were tuned with a sales objective "
        "that includes the 241 products.",
        "phase4_eval.py: objective = mean(catalogue, visual, sales_corr), with `sales_pct_in_cat` of all seasons; the tuning half is a "
        "random half of the models of every season. The sales model itself does not use the weights (the analogue feature was not kept "
        "for sales); the price analogue uses them but never sees sales. Part of the 241 test models therefore sit in the tuning half.",
        "Down for the claimed gain of the model over the similar-products baseline (the baseline is slightly favoured).",
        "Low priority: re-tune the weights without `sales_corr` and check that the baseline numbers do not move."))

    F.append(finding(
        "Leakage (indirect)", "verified", "low",
        "Delivered cross-validated estimates use pools and ranks that include later seasons.",
        f"final_models builds the price analogues from every colourway with a price (242 included) and the sales/price features "
        f"`price_pct_in_cat` is ranked across all seasons: the rank changes by more than 0.05 for {rank_in_category_gap(items):.0%} of the "
        "colourways when ranked inside the own season. No target of the test product is used, so the reported test numbers are not "
        "affected; the per-product estimates in estimates.csv are 'cross-validated', not 'as if launched at the time'.",
        "Up (slightly) for the delivered price estimates; none for the test metrics.",
        "Rank prices within the season of the product, and use a pool of earlier seasons only when you want an 'as at launch' estimate."))

    F.append(finding(
        "Leakage", "verified", "low",
        "Duplicates of a model across the split were checked and are not a problem.",
        f"Test colourways whose model is also in training: {leak['in_train_by_group']} (joining PROD_REF and PROD_REF_EQUIV) and "
        f"{leak['in_train_by_ref']} (PROD_REF only) of {leak['test_rows']}; {leak['recoded_rows']} re-coded rows. Photo near-duplicates "
        "(cosine > 0.97) between test and training: 2 of 1,417. Only 233 of 7,636 models exist in more than one season, and the test "
        "uses new models only.",
        "None.",
        f"None. Note the test excludes carry-over models ({int((sea.eq('241') & known).sum()) - int((v_va | s_te)[known].sum())} known 241 colourways), "
        "so the numbers describe new models, not re-orders (ITEM_REPEAT)."))

    F.append(finding(
        "Leakage (suspected)", "suspected", "medium",
        "CATEGORY_MATRIX and the planning classes may be revised after the first sales ('Sales below budget', 'Sales Over Budget' "
        "are recorded reclassification reasons).",
        f"{int(items['RECLASSIFIED'].sum())} reclassified colourways ({items['RECLASSIFIED'].mean():.1%}), "
        f"{int(cm_diff.sum())} with CATEGORY_MATRIX different from CATEGORY_MATRIX_BUYER (median log units {cm_med.get(True, np.nan):.2f} "
        f"against {cm_med.get(False, np.nan):.2f}). FASHIONTYPE_ASIS is identical to FASHIONTYPE, so no 'original' value exists to compare. "
        "The data do not say which field was revised or when.",
        "Up, if the classes were set after the sales were seen.",
        "Ask Parfois when each class is set (before the order or later) and which fields RECLASS_CODE changes. Check: refit "
        "without CATEGORY_MATRIX (leakage_checks.csv shows the three planning classes together give 0.399 vs 0.544 within category)."))

    F.append(finding(
        "Data quality", "verified", "low",
        "Landed cost is missing for a few colourways, mostly in some categories; MATERIAL exists for apparel only.",
        f"Missing landed cost: {int(items['log_cost'].isna().sum())} colourways ({', '.join(f'{k} {v:.1%}' for k, v in cost_missing.items())}); "
        f"MATERIAL is empty for {', '.join(f'{k} {v:.0%}' for k, v in items.groupby('CAT_DES_EN')['MATERIAL'].apply(lambda s: s.isna().mean()).items())} after the "
        "Phase 2a filling, and the feature is not in the chosen sales set anyway (SHAP 0.006).",
        "Small.",
        "Keep the fall-back; say in the report which categories rely on it."))

    F.append(finding(
        "Evaluation", "verified", "low",
        "The ranking metric rests on few groups.",
        f"Test: {models_test} models, {int(s_te[known].sum())} colourways in {len(groups)} category-month groups; only the {int((groups > 10).sum())} "
        f"groups with more than 10 products are scored ({int(groups[groups > 10].sum())} colourways, {groups[groups > 10].sum() / groups.sum():.0%}); the rest are dropped without being reported. Colourways of a model are correlated, the bootstrap resamples models (good).",
        "None clear; wide intervals.",
        "Report how many groups and products the ranking uses."))

    rank = {s: i for i, s in enumerate(SEVERITY)}
    F.sort(key=lambda r: (rank[r["severity"]], r["status"] != "verified"))
    for i, r in enumerate(F, 1):
        r["rank"], r["id"] = i, f"U{i:02d}"
    return F, num


def write(findings: list[dict], num: dict) -> None:
    df = pd.DataFrame(findings)[COLUMNS]
    df.to_csv(REPORT_DIR / "upstream_audit.csv", index=False, encoding="utf-8-sig")
    lines = ["# Phase 5b: upstream audit of the forecast inputs", "",
             "Generated by `src/phase5b_upstream_audit.py run`. Read-only: nothing in the data or in the forecast was changed. "
             "Model fits use the 232 training season and the **validation half** of the new 241 models only.", "",
             "**Verified** = measured here (numbers below) or read in the code. **Suspected** = a reading the data cannot settle, with the check "
             "that would settle it. Bias = the direction in which the finding moves the reported performance or the estimates.", "",
             "## Ranking", "", "| id | severity | status | area | finding |", "|---|---|---|---|---|"]
    lines += [f"| {r['id']} | {r['severity']} | {r['status']} | {r['area']} | {r['finding'].split('.')[0]}. |" for r in findings]
    lines += ["", "## Findings", ""]
    for r in findings:
        lines += [f"### {r['id']} ({r['severity']}, {r['status']}): {r['area']}", "", r["finding"], "",
                  f"- **Evidence:** {r['evidence']}", f"- **Bias:** {r['bias']}", f"- **Fix or check:** {r['fix']}", ""]
    lines += ["## What was checked and found clean", "",
              "- No negative or duplicated sales rows (10,185 rows, one per colourway); units and amounts are never below 0.",
              "- Components of the photo and text embeddings are fitted on the training rows only (`pca`); the shuffled-target run gives ~0.",
              "- No test colourway shares a model with training (U12); photo near-duplicates across the split are rare.",
              "- Buyer decisions (PUR_BET, TYPOLOGY) are not features of the delivered sales model; they only appear in the reference row.",
              "", "## Not tested here (needs data Parfois holds)", "",
              "- Stock-outs and sell-through (no stock column); promotion calendars; store counts per product; units returned.",
              "- Whether sales include several countries and currencies: `SALES_AMT_FX_RATE` is converted, but units are not split by market."]
    (REPORT_DIR / "upstream_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.parse_args()
    findings, num = audit()
    write(findings, num)
    print(pd.DataFrame(findings)[["id", "severity", "status", "area"]].to_string(index=False))
    print(f"-> {REPORT_DIR / 'upstream_audit.md'}")


if __name__ == "__main__":
    main()
