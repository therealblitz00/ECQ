import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit


def row_keys(A):
    """Exact identity of each row of a 0/1 matrix, as bytes (no hashing, no collisions)."""
    return [r.tobytes() for r in np.packbits(np.asarray(A, dtype=np.uint8), axis=1)]


def factorize_rows(A):
    codes, uniq = pd.factorize(pd.Series(row_keys(A)), sort=False)
    return codes.astype(np.int64), len(uniq)


def consensus_targets(crm_codes, Y):
    """Replace each row's BIL vector with the modal full BIL configuration of its CRM group.

    The mode is taken over whole 731-bit rows (not per column), so the consensus is always a
    configuration that was actually observed. Ties go to the configuration seen first.
    Returns (Y_consensus, is_noisy) where is_noisy marks rows whose own label differs.
    """
    Y = np.asarray(Y, dtype=np.uint8)
    bil_codes, _ = factorize_rows(Y)
    pairs = pd.DataFrame({'crm': crm_codes, 'bil': bil_codes, 'pos': np.arange(len(Y))})
    cnt = (pairs.groupby(['crm', 'bil'], sort=False)
           .agg(n=('pos', 'size'), first=('pos', 'min')).reset_index())
    cnt = cnt.sort_values(['crm', 'n', 'first'], ascending=[True, False, True])
    modal = cnt.drop_duplicates('crm').set_index('crm')['first']
    src_rows = modal.reindex(crm_codes).to_numpy()
    Y_cons = Y[src_rows]
    is_noisy = bil_codes != bil_codes[src_rows]
    return Y_cons, is_noisy


def test_like_mask(X, crm_cols):
    """Rows whose CRM categorical blocks look like test rows: exactly one value per group
    (BUSINESS_LINE, SUBSCRIBER_TYPE, SUBSCRIBER_STATUS) and never the value '0'.
    ~1.5% of train rows fail this; essentially no test row does.
    """
    groups = {}
    for i, c in enumerate(crm_cols):
        m = re.match(r'^CRM_(.+?)_ORIG_(.+)$', c)
        if m:
            groups.setdefault(m.group(1), []).append((i, m.group(2)))
    ok = np.ones(len(X), bool)
    for items in groups.values():
        ok &= X[:, [i for i, _ in items]].sum(1) == 1
        zero = [i for i, v in items if v == '0']
        if zero:
            ok &= X[:, zero].sum(1) == 0
    return ok


RARE_FREQ = 0.002
HEAVY_K = 3


def rare_pack_count(X, crm_cols, X_ref=None, rare_freq=RARE_FREQ):
    """Per row, how many CRM packs it has that occur in < rare_freq of the reference rows
    (default: X itself). Uses CRM features only, never labels."""
    pk = np.array([c.endswith('_PACK') for c in crm_cols])
    ref = X if X_ref is None else X_ref
    rare = np.zeros(len(crm_cols), bool)
    rare[pk] = ref[:, pk].mean(0) < rare_freq
    return X[:, rare].sum(1)


def clean_like_mask(X, crm_cols, X_ref=None, heavy_k=HEAVY_K, rare_freq=RARE_FREQ):
    """Test-like rows that also carry fewer than heavy_k rare CRM packs. ~9% of train rows
    fail this (suspected injected corruption); 0.55% of test rows do."""
    return test_like_mask(X, crm_cols) & (rare_pack_count(X, crm_cols, X_ref, rare_freq) < heavy_k)


def grouped_holdout(groups, val_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr, va = next(gss.split(np.zeros(len(groups)), groups=groups))
    assert not set(np.unique(groups[tr])) & set(np.unique(groups[va])), 'group leak'
    return tr, va


def grouped_kfold(groups, n_splits=5):
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
