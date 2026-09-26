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


def grouped_holdout(groups, val_size=0.2, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    tr, va = next(gss.split(np.zeros(len(groups)), groups=groups))
    assert not set(np.unique(groups[tr])) & set(np.unique(groups[va])), 'group leak'
    return tr, va


def grouped_kfold(groups, n_splits=5):
    return list(GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups))
