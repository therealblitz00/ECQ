"""Recompute the Sprint-1 EDA/cleaning numbers once and cache them as small CSVs under
data/derived/, so later sessions can read a few KB of stats instead of reloading the
500MB+ raw CSVs and rerunning correlation matrices / groupbys over 187k rows.

Mirrors the logic in notebooks/sprint1_preprocessing.ipynb (sections 2-10) and
src/preprocessing.py exactly, so the numbers here should always match the notebook's
printed output. Re-run this whenever train.csv/test.csv change.

Run from the repo root: python src/export_summaries.py
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .preprocessing import (
        correlation_clusters,
        cluster_columns_to_drop,
        compute_majority_labels,
        sparse_binary_corr_matrix,
        flag_count_outliers,
        hamming_distance_report,
        referential_integrity_checks,
        label_imbalance_report,
    )
except ImportError:  # running as a standalone script (python src/export_summaries.py)
    from preprocessing import (
        correlation_clusters,
        cluster_columns_to_drop,
        compute_majority_labels,
        sparse_binary_corr_matrix,
        flag_count_outliers,
        hamming_distance_report,
        referential_integrity_checks,
        label_imbalance_report,
    )

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = DATA_DIR / "derived"
RELABEL_THRESHOLD = 0.65
CLUSTER_THRESHOLD = 0.95


def split_taxonomy(cols, prefix):
    packs = [c for c in cols if re.match(rf"^{prefix}_\d+_PACK$", c)]
    meta = [c for c in cols if c not in packs]
    return packs, meta


def onehot_groups(meta_cols):
    groups = {}
    for c in meta_cols:
        base = c.rsplit("_", 1)[0]
        groups.setdefault(base, []).append(c)
    return groups


def top_pairs(corr, row_names, col_names, top_n, symmetric):
    if symmetric:
        i_idx, j_idx = np.triu_indices(len(row_names), k=1)
    else:
        i_idx, j_idx = np.indices(corr.shape)
        i_idx, j_idx = i_idx.ravel(), j_idx.ravel()
    vals = corr[i_idx, j_idx]
    mask = ~np.isnan(vals)
    df = pd.DataFrame(
        {
            "col_a": np.array(row_names)[i_idx[mask]],
            "col_b": np.array(col_names)[j_idx[mask]],
            "corr": vals[mask],
        }
    )
    return df.reindex(df["corr"].abs().sort_values(ascending=False).index).head(top_n)


def compute_and_export_summaries(train, test, crm_cols, bil_cols, out_dir=OUT_DIR):
    """Recompute every EDA/cleaning number from already-loaded train/test DataFrames and
    write them to `out_dir`. This is the function the notebook calls directly (it already
    has train/test in memory, no need to re-read the CSVs) — `main()` below is just a
    thin CLI wrapper around it for regenerating data/derived/ without opening Jupyter.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- referential_integrity.csv: hard asserts (raised immediately) + soft counts ---
    integrity = referential_integrity_checks(train, test, crm_cols, bil_cols)
    pd.DataFrame([integrity]).to_csv(out_dir / "referential_integrity.csv", index=False)

    crm_var = train[crm_cols].nunique()
    bil_var = train[bil_cols].nunique()
    constant_crm = set(crm_var[crm_var <= 1].index)
    constant_bil = set(bil_var[bil_var <= 1].index)

    crm_packs, crm_meta = split_taxonomy(crm_cols, "CRM")
    bil_packs, bil_meta = split_taxonomy(bil_cols, "BIL")

    crm_corr = sparse_binary_corr_matrix(train[crm_packs], train[crm_packs])
    bil_corr = sparse_binary_corr_matrix(train[bil_packs], train[bil_packs])
    cross_corr = sparse_binary_corr_matrix(train[crm_packs], train[bil_packs])

    crm_clusters = correlation_clusters(crm_packs, crm_corr, threshold=CLUSTER_THRESHOLD)
    bil_clusters = correlation_clusters(bil_packs, bil_corr, threshold=CLUSTER_THRESHOLD)
    crm_drop = set(cluster_columns_to_drop(crm_clusters))
    bil_drop = set(cluster_columns_to_drop(bil_clusters))

    col_to_cluster = {}
    for i, group in enumerate(crm_clusters + bil_clusters):
        for c in group:
            col_to_cluster[c] = i

    # --- column_stats.csv: one row per CRM/BIL column ---
    activation = train[crm_cols + bil_cols].mean()
    rows = []
    for system, cols, packs, var, constant, drop in [
        ("CRM", crm_cols, crm_packs, crm_var, constant_crm, crm_drop),
        ("BIL", bil_cols, bil_packs, bil_var, constant_bil, bil_drop),
    ]:
        for c in cols:
            family = "pack" if c in packs else "onehot"
            rows.append(
                {
                    "column": c,
                    "system": system,
                    "family": family,
                    "group": c.rsplit("_", 1)[0] if family == "onehot" else None,
                    "n_unique": int(var[c]),
                    "constant": c in constant,
                    "activation_rate": round(float(activation[c]), 6),
                    "cluster_id": col_to_cluster.get(c),
                    "dropped_as_duplicate": c in drop,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "column_stats.csv", index=False)

    # --- correlation_clusters.csv ---
    cluster_rows = []
    for system, clusters in [("CRM", crm_clusters), ("BIL", bil_clusters)]:
        for group in clusters:
            keep, *drop = sorted(group)
            cluster_rows.append(
                {
                    "cluster_id": col_to_cluster[keep],
                    "system": system,
                    "kept_column": keep,
                    "dropped_columns": ";".join(drop),
                    "size": len(group),
                }
            )
    pd.DataFrame(cluster_rows).to_csv(out_dir / "correlation_clusters.csv", index=False)

    # --- onehot_groups.csv ---
    group_rows = []
    for system, meta_cols in [("CRM", crm_meta), ("BIL", bil_meta)]:
        for base, cols in onehot_groups(meta_cols).items():
            row_sums = train[cols].sum(axis=1)
            group_rows.append(
                {
                    "system": system,
                    "group": base,
                    "n_categories": len(cols),
                    "pct_rows_sum_to_1": round(float((row_sums == 1).mean()), 6),
                    "pct_rows_sum_to_0": round(float((row_sums == 0).mean()), 6),
                }
            )
    pd.DataFrame(group_rows).to_csv(out_dir / "onehot_groups.csv", index=False)

    # --- top_correlated_pairs.csv ---
    pair_rows = [
        top_pairs(crm_corr, crm_packs, crm_packs, 30, symmetric=True).assign(pair_type="crm-crm"),
        top_pairs(bil_corr, bil_packs, bil_packs, 30, symmetric=True).assign(pair_type="bil-bil"),
        top_pairs(cross_corr, crm_packs, bil_packs, 30, symmetric=False).assign(pair_type="crm-bil"),
    ]
    pd.concat(pair_rows, ignore_index=True).to_csv(out_dir / "top_correlated_pairs.csv", index=False)

    # --- bil_predictability.csv: per BIL pack, best single-CRM-pack |corr| ---
    max_abs_corr = np.nanmax(np.abs(cross_corr), axis=0)
    pd.DataFrame({"bil_pack": bil_packs, "max_abs_corr_with_any_crm_pack": max_abs_corr}).to_csv(
        out_dir / "bil_predictability.csv", index=False
    )

    # --- CRM<->BIL config-level ambiguity + majority-vote relabeling ---
    crm_key = train[crm_cols].astype(str).agg("".join, axis=1)
    bil_key = train[bil_cols].astype(str).agg("".join, axis=1)
    pairs = pd.DataFrame({"crm": crm_key, "bil": bil_key})
    ambiguous = pairs.groupby("crm")["bil"].nunique()
    n_ambiguous = int((ambiguous > 1).sum())

    majority = compute_majority_labels(crm_key, bil_key)
    relabel_targets = majority.loc[
        (majority["n_distinct"] > 1) & (majority["share"] >= RELABEL_THRESHOLD)
    ]
    to_relabel = crm_key.isin(relabel_targets.index)
    n_rows_relabeled = int(to_relabel.sum())
    n_groups_relabeled = len(relabel_targets)

    new_bil_key = bil_key.where(~to_relabel, crm_key.map(relabel_targets["bil"]))
    remaining_ambiguous = pd.DataFrame({"crm": crm_key, "bil": new_bil_key}).groupby("crm")[
        "bil"
    ].nunique()
    n_remaining_ambiguous = int((remaining_ambiguous > 1).sum())

    # --- pack_count_outliers.csv: IQR-flagged rows (flag-only, nothing removed/capped
    # from train_clean/test_clean — see module docstring for why) ---
    crm_count = train[crm_packs].sum(axis=1)
    bil_count = train[bil_packs].sum(axis=1)
    crm_is_outlier, crm_lo, crm_hi = flag_count_outliers(crm_count)
    bil_is_outlier, bil_lo, bil_hi = flag_count_outliers(bil_count)
    outlier_mask = crm_is_outlier | bil_is_outlier
    pd.DataFrame(
        {
            "MSISDN": train.loc[outlier_mask, "MSISDN"],
            "n_crm_packs": crm_count[outlier_mask],
            "n_bil_packs": bil_count[outlier_mask],
            "crm_outlier": crm_is_outlier[outlier_mask],
            "bil_outlier": bil_is_outlier[outlier_mask],
        }
    ).to_csv(out_dir / "pack_count_outliers.csv", index=False)

    # --- typo_audit.csv: Hamming distance from each ambiguous row to its group's
    # majority BIL config, bucketed, cross-tabbed against whether the 0.65-share
    # relabel already caught it. Small bucket distance = likely single manual-entry
    # error; large bucket distance = likely a genuinely different (legitimate)
    # provisioning outcome that a share-only rule could otherwise mislabel.
    hamming = hamming_distance_report(train, bil_cols, crm_key, bil_key, majority)
    if len(hamming):
        hamming["dist_bucket"] = pd.cut(
            hamming["hamming_dist_to_majority"],
            bins=[-1, 0, 2, np.inf],
            labels=["matches_majority(0)", "likely_typo(1-2)", "likely_distinct(3+)"],
        )
        hamming["group_was_relabeled"] = hamming["crm"].isin(relabel_targets.index)
        typo_summary = (
            hamming.groupby(["dist_bucket", "group_was_relabeled"], observed=True)
            .size()
            .reset_index(name="n_rows")
        )
    else:
        typo_summary = pd.DataFrame(columns=["dist_bucket", "group_was_relabeled", "n_rows"])
    typo_summary.to_csv(out_dir / "typo_audit.csv", index=False)

    # --- label_imbalance.csv: activation-rate tiers for CRM and BIL columns ---
    imbalance_rows = []
    for system, cols in [("CRM", crm_cols), ("BIL", bil_cols)]:
        tier_counts = label_imbalance_report(train[cols].mean())
        tier_counts.insert(0, "system", system)
        imbalance_rows.append(tier_counts)
    pd.concat(imbalance_rows, ignore_index=True).to_csv(out_dir / "label_imbalance.csv", index=False)

    # --- bil_value_counts.csv: top 30 most common BIL configs ---
    bil_counts = pairs["bil"].value_counts()
    crm_counts = pairs["crm"].value_counts()
    bil_top = (bil_counts.head(30) / len(train)).reset_index()
    bil_top.columns = ["bil_config", "share"]
    bil_top.insert(0, "rank", range(1, len(bil_top) + 1))
    bil_top["n_active_packs"] = bil_top["bil_config"].str.count("1")
    bil_top["config_hash"] = bil_top["bil_config"].apply(lambda s: hash(s) & 0xFFFFFFFF)
    bil_top.drop(columns="bil_config").to_csv(out_dir / "bil_value_counts.csv", index=False)

    n_for_80 = int(
        (bil_counts.sort_values(ascending=False).cumsum() / bil_counts.sum() <= 0.8).sum() + 1
    )

    # --- dataset_summary.csv: single-row rollup of every scalar ---
    crm_cols_clean = [c for c in crm_cols if c not in constant_crm and c not in crm_drop]
    bil_cols_clean = [c for c in bil_cols if c not in bil_drop]

    summary = {
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "n_crm_cols_raw": len(crm_cols),
        "n_bil_cols_raw": len(bil_cols),
        "n_crm_packs": len(crm_packs),
        "n_crm_onehot": len(crm_meta),
        "n_bil_packs": len(bil_packs),
        "n_bil_onehot": len(bil_meta),
        "n_constant_crm": len(constant_crm),
        "n_constant_bil": len(constant_bil),
        "mean_crm_activation_rate": round(float(train[crm_cols].mean().mean()), 6),
        "mean_bil_activation_rate": round(float(train[bil_cols].mean().mean()), 6),
        "n_duplicate_msisdn_train": int(train["MSISDN"].duplicated().sum()),
        "n_duplicate_msisdn_test": int(test["MSISDN"].duplicated().sum()),
        "n_unique_crm_configs": int(crm_key.nunique()),
        "n_unique_bil_configs": int(bil_key.nunique()),
        "n_ambiguous_crm_groups": n_ambiguous,
        "pct_ambiguous_crm_groups": round(n_ambiguous / len(ambiguous), 6),
        "avg_crm_packs_per_customer": round(float(crm_count.mean()), 4),
        "avg_bil_packs_per_customer": round(float(bil_count.mean()), 4),
        "crm_bil_pack_count_corr": round(float(crm_count.corr(bil_count)), 4),
        "n_distinct_bil_configs_for_80pct_coverage": n_for_80,
        "n_distinct_bil_configs_total": int(bil_counts.shape[0]),
        "n_distinct_crm_configs_total": int(crm_counts.shape[0]),
        "cluster_threshold": CLUSTER_THRESHOLD,
        "n_crm_dedup_clusters": len(crm_clusters),
        "n_crm_cols_dropped_as_duplicate": len(crm_drop),
        "n_bil_dedup_clusters": len(bil_clusters),
        "n_bil_cols_dropped_as_duplicate": len(bil_drop),
        "n_crm_cols_clean": len(crm_cols_clean),
        "n_bil_cols_clean": len(bil_cols_clean),
        "relabel_threshold": RELABEL_THRESHOLD,
        "n_rows_relabeled": n_rows_relabeled,
        "n_groups_relabeled": n_groups_relabeled,
        "n_remaining_ambiguous_crm_groups": n_remaining_ambiguous,
        "crm_pack_count_iqr_lower": round(float(crm_lo), 2),
        "crm_pack_count_iqr_upper": round(float(crm_hi), 2),
        "bil_pack_count_iqr_lower": round(float(bil_lo), 2),
        "bil_pack_count_iqr_upper": round(float(bil_hi), 2),
        "n_pack_count_outlier_rows": int(outlier_mask.sum()),
        "pct_pack_count_outlier_rows": round(float(outlier_mask.mean()), 6),
        **{f"referential_{k}": v for k, v in integrity.items()},
    }
    pd.DataFrame([summary]).to_csv(out_dir / "dataset_summary.csv", index=False)

    n_files = len(list(out_dir.glob("*.csv")))
    print(f"Wrote {n_files} derived summary files to {out_dir}")


def main():
    """CLI entry point: load train/test from data/ and export summaries. The notebook
    doesn't call this — it already has train/test loaded and calls
    compute_and_export_summaries() directly instead of re-reading the CSVs.
    """
    header = pd.read_csv(DATA_DIR / "train.csv", nrows=0).columns.tolist()
    crm_cols = [c for c in header if c.startswith("CRM")]
    bil_cols = [c for c in header if c.startswith("BIL")]

    dtype_map = {c: "int8" for c in crm_cols + bil_cols}
    dtype_map["MSISDN"] = str
    train = pd.read_csv(DATA_DIR / "train.csv", dtype=dtype_map)

    test_dtype_map = {c: "int8" for c in crm_cols}
    test_dtype_map["MSISDN"] = str
    test = pd.read_csv(DATA_DIR / "test.csv", dtype=test_dtype_map)

    compute_and_export_summaries(train, test, crm_cols, bil_cols)


if __name__ == "__main__":
    main()
