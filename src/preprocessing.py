"""Cleaning steps for the CRM -> BIL dataset, shared between notebooks.

Column-family taxonomy (pack vs. one-hot) lives in notebooks/sprint1_preprocessing.ipynb
since it's already computed there; this module covers the steps that turn EDA findings
into an actual cleaned, exportable dataset:

- fast correlation on sparse binary columns (sparse_binary_corr_matrix)
- collapsing near-duplicate pack columns (correlation_clusters)
- majority-vote relabeling of ambiguous CRM->BIL groups (compute_majority_labels,
  relabel_ambiguous_rows), refined by per-row Hamming distance to the majority
  config (hamming_distance_report) to separate likely single-bit entry errors
  from genuinely distinct (legitimate) provisioning
- IQR-based outlier flagging on pack-count distributions (flag_count_outliers) —
  flag-only, since a customer with an unusual pack count is still a real row the
  model must score in test, not a measurement error to cap or drop
- referential integrity checks between train/test (referential_integrity_checks)
- label-imbalance bucketing (label_imbalance_report)
"""

import numpy as np
import pandas as pd


def sparse_binary_corr_matrix(a, b):
    """Pearson correlation between columns of `a` and columns of `b`, both strictly
    0/1 (same rows). For Bernoulli columns, corr(X, Y) = (E[XY] - E[X]E[Y]) /
    sqrt(Var(X)Var(Y)); E[XY] is a sparse dot product, so this avoids ever forming
    the dense centered matrices that a generic Pearson implementation needs — the
    packs are ~2% dense, so sparse E[XY] is both faster and far lighter on memory
    than the dense mean-subtracted matmul for matrices this wide (hundreds of cols).
    """
    from scipy.sparse import csr_matrix

    a_sp = csr_matrix(a.to_numpy(dtype=np.float32))
    b_sp = csr_matrix(b.to_numpy(dtype=np.float32))
    n = a_sp.shape[0]

    p_a = np.asarray(a_sp.sum(axis=0)).ravel() / n
    p_b = np.asarray(b_sp.sum(axis=0)).ravel() / n
    cross = np.asarray((a_sp.T @ b_sp).todense(), dtype=np.float64) / n

    cov = cross - np.outer(p_a, p_b)
    std_a = np.sqrt(p_a * (1 - p_a))
    std_b = np.sqrt(p_b * (1 - p_b))
    denom = np.outer(std_a, std_b)
    denom[denom == 0] = np.nan  # constant columns -> undefined correlation
    return cov / denom


def correlation_clusters(cols, corr, threshold=0.95):
    """Union-find clusters of columns whose pairwise |corr| >= threshold.

    `corr` is a symmetric matrix aligned with `cols` (e.g. from pearson_corr_matrix(df, df)).
    Returns a list of clusters (each a list of >=2 column names); singletons are omitted.
    """
    n = len(cols)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        for j in range(i + 1, n):
            c = corr[i, j]
            if not np.isnan(c) and abs(c) >= threshold:
                union(i, j)

    clusters = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(cols[i])
    return [group for group in clusters.values() if len(group) > 1]


def cluster_columns_to_drop(clusters):
    """Keep one (sorted-first) representative column per cluster, drop the rest."""
    drop_cols = []
    for group in clusters:
        _keep, *rest = sorted(group)
        drop_cols.extend(rest)
    return drop_cols


def compute_majority_labels(crm_key, bil_key):
    """Per CRM config: its majority BIL config, that config's share of the group,
    and how many distinct BIL configs the CRM config maps to overall.

    `crm_key` / `bil_key` are the row-wise concatenated bitstrings (see notebook §3).
    Returns a DataFrame indexed by crm_key with columns [bil, n, group_total, n_distinct, share].
    """
    df = pd.DataFrame({'crm': crm_key, 'bil': bil_key})
    pair_counts = df.groupby(['crm', 'bil']).size().reset_index(name='n')
    pair_counts['group_total'] = pair_counts.groupby('crm')['n'].transform('sum')
    pair_counts['n_distinct'] = pair_counts.groupby('crm')['bil'].transform('size')
    pair_counts['share'] = pair_counts['n'] / pair_counts['group_total']
    return pair_counts.loc[pair_counts.groupby('crm')['n'].idxmax()].set_index('crm')


def iqr_bounds(series, k=1.5):
    """Standard Tukey IQR fences: [Q1 - k*IQR, Q3 + k*IQR]. Chosen over z-scores
    because pack-count distributions are right-skewed and long-tailed (a z-score
    assumes rough symmetry/normality and would over-flag the tail); IQR is robust
    to that skew since it's based on rank statistics, not the mean/std.
    """
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def flag_count_outliers(counts, k=1.5):
    """Flag (not remove/cap) rows whose pack count falls outside the IQR fences.

    Returns (is_outlier: bool Series, lower_bound, upper_bound). These are real
    customers who still need a correct BIL prediction in test, not measurement
    noise — so this is a diagnostic for the report, not a filter applied to the
    exported train/test sets.
    """
    lower, upper = iqr_bounds(counts, k=k)
    is_outlier = (counts < lower) | (counts > upper)
    return is_outlier, lower, upper


def hamming_distance_report(df, bil_cols, crm_key, bil_key, majority):
    """For every row in a CRM group with >1 distinct BIL config, how many bits does
    its BIL config differ from that group's majority BIL config?

    Rationale: the existing majority-vote relabel (relabel_ambiguous_rows) uses
    *share* (how dominant the majority is) as its only signal. Hamming distance
    adds an orthogonal, independent signal — a row 1-2 bits away from its group's
    majority looks like a single fat-fingered pack toggle (a "typo"); a row dozens
    of bits away looks like a genuinely different — and possibly legitimate —
    provisioning outcome that a naive majority vote could otherwise steamroll.
    This is diagnostic (feeds the report / a future relabel-threshold refinement);
    it does not itself change relabel_ambiguous_rows's behavior.

    Returns a DataFrame indexed like the ambiguous subset of `df`, with columns
    [crm, hamming_dist_to_majority, is_majority_row].
    """
    ambiguous_crm = majority.index[majority['n_distinct'] > 1]
    bil_matrix = df[bil_cols].to_numpy()

    chunks = []
    for crm_val in ambiguous_crm:
        mask = (crm_key == crm_val).to_numpy()
        majority_bil_str = majority.loc[crm_val, 'bil']
        majority_vec = np.array(list(majority_bil_str), dtype=bil_matrix.dtype)
        dist = (bil_matrix[mask] != majority_vec).sum(axis=1)
        chunks.append(pd.DataFrame({
            'crm': crm_val,
            'hamming_dist_to_majority': dist,
            'is_majority_row': (bil_key[mask] == majority_bil_str).to_numpy(),
        }, index=df.index[mask]))

    if not chunks:
        return pd.DataFrame(columns=['crm', 'hamming_dist_to_majority', 'is_majority_row'])
    return pd.concat(chunks)


def referential_integrity_checks(train, test, crm_cols, bil_cols):
    """Hard/soft integrity checks between train and test. Since CRM and BIL are
    already columns of the same row (not separate tables to join), there's no
    "orphaned foreign key" in the classic sense — the real leakage/consistency
    risks here are column-schema drift between train/test and stray non-binary
    values. Raises AssertionError on any hard failure; returns a dict of results
    (including soft, informational-only checks) for reporting.

    `bil_cols` should be the *raw* BIL column list (train-only; test never has
    BIL columns by design, which is intentional, not a defect).
    """
    assert list(test.columns[1:]) == crm_cols, (
        'test CRM columns differ from train CRM columns (name or order) — '
        'any column drop/reorder derived from train must be applied identically to test'
    )

    train_vals = pd.unique(train[crm_cols + bil_cols].to_numpy().ravel())
    test_vals = pd.unique(test[crm_cols].to_numpy().ravel())
    assert set(train_vals) <= {0, 1}, f'non-binary values found in train: {set(train_vals) - {0, 1}}'
    assert set(test_vals) <= {0, 1}, f'non-binary values found in test: {set(test_vals) - {0, 1}}'

    assert train[crm_cols + bil_cols].isna().sum().sum() == 0, 'nulls found in train CRM/BIL columns'
    assert test[crm_cols].isna().sum().sum() == 0, 'nulls found in test CRM columns'

    n_dup_train = int(train['MSISDN'].duplicated().sum())
    n_dup_test = int(test['MSISDN'].duplicated().sum())
    assert n_dup_train == 0, f'{n_dup_train} duplicate MSISDN in train'
    assert n_dup_test == 0, f'{n_dup_test} duplicate MSISDN in test'

    # Informational only: overlap isn't necessarily an error (could be a valid
    # longitudinal resample), but silent overlap would matter for a hold-out split.
    n_overlap = int(pd.Series(test['MSISDN']).isin(set(train['MSISDN'])).sum())

    return {
        'columns_match': True,
        'train_binary_only': True,
        'test_binary_only': True,
        'train_nulls': 0,
        'test_nulls': 0,
        'n_duplicate_msisdn_train': n_dup_train,
        'n_duplicate_msisdn_test': n_dup_test,
        'n_msisdn_overlap_train_test': n_overlap,
    }


def label_imbalance_report(rate_series, bins=(0, 0.001, 0.01, 0.1, 0.5, 1.0)):
    """Bucket a Series of per-column activation rates (e.g. BIL pack means) into
    imbalance tiers, from ultra-rare (<0.1%) to dense (>=50%). Rare BIL labels are
    the ones a naive classifier will just predict all-zero for, which is exactly
    the failure mode EMR punishes hardest (one wrong bit fails the whole row).
    """
    labels = ['ultra_rare(<0.1%)', 'rare(0.1-1%)', 'uncommon(1-10%)', 'common(10-50%)', 'dense(>=50%)']
    tiers = pd.cut(rate_series, bins=bins, labels=labels, include_lowest=True, right=False)
    counts = tiers.value_counts().reindex(labels, fill_value=0)
    return counts.rename_axis('imbalance_tier').reset_index(name='n_columns')


def relabel_ambiguous_rows(df, bil_cols, crm_key, bil_key, majority, threshold=0.65):
    """Overwrite BIL columns with the group's majority BIL config, but only for CRM
    groups that are both ambiguous (n_distinct > 1) and confidently so (share >= threshold).
    Groups below the threshold are left untouched rather than force-labeled.

    Returns (relabeled_df, n_rows_relabeled, n_groups_relabeled).
    """
    relabel_targets = majority.loc[
        (majority['n_distinct'] > 1) & (majority['share'] >= threshold), 'bil'
    ]
    to_relabel = crm_key.isin(relabel_targets.index)

    df = df.copy()
    if to_relabel.any():
        new_bil_keys = crm_key[to_relabel].map(relabel_targets)
        new_bil_values = np.array([list(s) for s in new_bil_keys], dtype='int8')
        df.loc[to_relabel, bil_cols] = new_bil_values

    return df, int(to_relabel.sum()), len(relabel_targets)
